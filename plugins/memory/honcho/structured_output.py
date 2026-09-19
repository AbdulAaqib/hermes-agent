"""Structured dialectic output: response_format schema validation and evidence formatting.

Client-side schema guards mirror Honcho's server limits (checklist 8.1) so a malformed
schema fails with a clear error instead of a server 422: root type must be "object",
nesting depth at most 20, and at most 500 schema nodes total.
"""

from __future__ import annotations

from typing import Any

MAX_SCHEMA_DEPTH = 20
MAX_SCHEMA_NODES = 500


def validate_response_schema(schema: Any) -> dict[str, Any]:
    """Return ``schema`` when it is a usable JSON Schema for dialectic response_format;
    raise ValueError with the specific violation otherwise."""
    if not isinstance(schema, dict):
        raise ValueError(f"response_format must be a JSON Schema object, got {type(schema).__name__}")
    if schema.get("type") != "object":
        raise ValueError(f"response_format root type must be 'object', got {schema.get('type')!r}")

    nodes = 0
    deepest = 0
    stack: list[tuple[Any, int]] = [(schema, 1)]
    seen: set[int] = set()
    while stack:
        node, depth = stack.pop()
        if not isinstance(node, dict) or id(node) in seen:
            continue
        seen.add(id(node))
        nodes += 1
        deepest = max(deepest, depth)
        if nodes > MAX_SCHEMA_NODES:
            raise ValueError(f"response_format has more than {MAX_SCHEMA_NODES} schema nodes")
        if depth > MAX_SCHEMA_DEPTH:
            raise ValueError(f"response_format nests deeper than {MAX_SCHEMA_DEPTH} levels")
        for key in ("properties", "patternProperties", "defs", "$defs", "definitions"):
            for child in (node.get(key) or {}).values():
                stack.append((child, depth + 1))
        for key in ("items", "additionalProperties", "contains", "if", "then", "else", "not"):
            child = node.get(key)
            if isinstance(child, list):
                stack.extend((c, depth + 1) for c in child)
            else:
                stack.append((child, depth + 1))
        for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
            stack.extend((c, depth + 1) for c in (node.get(key) or []))
    return schema


def format_evidence(evidence: Any) -> dict[str, Any] | None:
    """Render an SDK Evidence object for tool output: conclusions with their reasoning
    level, id, content, session_id and source_ids (checklist 8.2). Returns None when no
    evidence was requested/returned — distinct from an empty conclusions list, which means
    the dialectic ran and verifiably read no conclusions."""
    if evidence is None:
        return None
    conclusions = [
        {
            "id": c.id,
            "level": getattr(c, "level", None),
            "content": c.content,
            "session_id": getattr(c, "session_id", None),
            "source_ids": list(getattr(c, "source_ids", None) or []),
        }
        for c in (getattr(evidence, "conclusions", None) or [])
    ]
    return {
        "conclusions": conclusions,
        "messages_read": len(getattr(evidence, "messages", None) or []),
        "tool_calls": [getattr(t, "tool_name", "") for t in (getattr(evidence, "tool_calls", None) or [])],
        "reasoning_trace_id": getattr(evidence, "reasoning_trace_id", None),
    }
