"""Behavior contracts for the SDK 2.5 structured dialectic surface:

response_format schema validation, include_evidence parsing, scope/sessions mutual
exclusion, and the honcho_reasoning tool wiring (checklist 8.1, 8.2, 9.1-9.3).
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager
from plugins.memory.honcho.structured_output import (
    MAX_SCHEMA_DEPTH, MAX_SCHEMA_NODES, format_evidence, validate_response_schema,
)


# ── response_format schema validation (8.1) ─────────────────────────────────


class TestValidateResponseSchema:
    def test_accepts_plain_object_schema(self):
        schema = {"type": "object", "properties": {"mood": {"type": "string"}}}
        assert validate_response_schema(schema) is schema

    def test_rejects_non_object_root(self):
        with pytest.raises(ValueError, match="root type must be 'object'"):
            validate_response_schema({"type": "array", "items": {"type": "string"}})

    def test_rejects_non_dict(self):
        with pytest.raises(ValueError, match="must be a JSON Schema object"):
            validate_response_schema("not-a-schema")

    def test_rejects_excessive_depth(self):
        schema = {"type": "object"}
        node = schema
        for _ in range(MAX_SCHEMA_DEPTH + 1):
            node["properties"] = {"next": {"type": "object"}}
            node = node["properties"]["next"]
        with pytest.raises(ValueError, match="deeper than"):
            validate_response_schema(schema)

    def test_rejects_excessive_node_count(self):
        schema = {"type": "object",
                  "properties": {f"field_{i}": {"type": "string"} for i in range(MAX_SCHEMA_NODES + 1)}}
        with pytest.raises(ValueError, match="schema nodes"):
            validate_response_schema(schema)

    def test_defs_do_not_double_count_shared_subschema(self):
        shared = {"type": "string"}
        schema = {"type": "object", "properties": {"a": shared, "b": shared}}
        assert validate_response_schema(schema) is schema


# ── evidence formatting (8.2) ───────────────────────────────────────────────


class TestFormatEvidence:
    def test_none_evidence_stays_none(self):
        assert format_evidence(None) is None

    def test_empty_conclusions_is_verified_empty_not_missing(self):
        out = format_evidence(SimpleNamespace(conclusions=[], messages=[], tool_calls=[],
                                              reasoning_trace_id="t-1"))
        assert out["conclusions"] == []  # verified empty, distinct from None
        assert out["reasoning_trace_id"] == "t-1"

    def test_conclusion_fields_are_surfaced(self):
        conclusion = SimpleNamespace(id="c1", level="deductive", content="Q likes tea",
                                     session_id="s-1", source_ids=["m1", "m2"])
        out = format_evidence(SimpleNamespace(
            conclusions=[conclusion], messages=[SimpleNamespace(id="m1")],
            tool_calls=[SimpleNamespace(tool_name="search")], reasoning_trace_id=None))
        assert out["conclusions"] == [{"id": "c1", "level": "deductive", "content": "Q likes tea",
                                       "session_id": "s-1", "source_ids": ["m1", "m2"]}]
        assert out["messages_read"] == 1
        assert out["tool_calls"] == ["search"]


# ── dialectic_query_detailed (8.1/8.2/9.x plumbing) ─────────────────────────


def _manager() -> tuple[HonchoSessionManager, HonchoSession, MagicMock]:
    mgr = HonchoSessionManager(honcho=MagicMock(), config=None)
    mgr._authed_call = lambda label, op: op()
    session = HonchoSession(key="cli:chat-1", user_peer_id="q", assistant_peer_id="asuna",
                            honcho_session_id="chat-1")
    mgr._cache[session.key] = session
    ai_peer = MagicMock()
    mgr._get_or_create_peer = MagicMock(return_value=ai_peer)
    return mgr, session, ai_peer


class TestDialecticQueryDetailed:
    def test_scope_and_sessions_are_mutually_exclusive(self):
        mgr, session, _ = _manager()
        with pytest.raises(ValueError, match="mutually exclusive"):
            mgr.dialectic_query_detailed(session.key, "q", scope="work", sessions=["s-1"])

    def test_structured_and_scope_kwargs_reach_peer_chat(self):
        mgr, session, ai_peer = _manager()
        schema = {"type": "object", "properties": {"mood": {"type": "string"}}}
        ai_peer.chat.return_value = '{"mood": "calm"}'

        out = mgr.dialectic_query_detailed(session.key, "how is q?", response_format=schema, scope="work")
        kwargs = ai_peer.chat.call_args.kwargs
        assert kwargs["response_format"] is schema
        assert kwargs["scope"] == "work"
        assert kwargs["sessions"] is None
        assert kwargs["include_evidence"] is False
        assert kwargs["target"] == session.user_peer_id  # AI peer observes the user
        assert out["content"] == '{"mood": "calm"}'
        assert out["structured"] is True
        assert out["evidence"] is None  # not requested

    def test_include_evidence_unwraps_chat_response(self):
        mgr, session, ai_peer = _manager()
        evidence = SimpleNamespace(conclusions=[SimpleNamespace(
            id="c1", level="explicit", content="fact", session_id="s", source_ids=[])],
            messages=[], tool_calls=[], reasoning_trace_id="t")
        ai_peer.chat.return_value = SimpleNamespace(content="answer", evidence=evidence)

        out = mgr.dialectic_query_detailed(session.key, "q?", include_evidence=True)
        assert ai_peer.chat.call_args.kwargs["include_evidence"] is True
        assert out["content"] == "answer"
        assert out["evidence"]["conclusions"][0]["id"] == "c1"

    def test_include_evidence_with_null_evidence_stays_none(self):
        mgr, session, ai_peer = _manager()
        ai_peer.chat.return_value = SimpleNamespace(content="answer", evidence=None)
        out = mgr.dialectic_query_detailed(session.key, "q?", include_evidence=True)
        assert out["evidence"] is None  # requested but not returned: distinct from []

    def test_sessions_allowlist_is_forwarded(self):
        mgr, session, ai_peer = _manager()
        ai_peer.chat.return_value = "ok"
        mgr.dialectic_query_detailed(session.key, "q?", sessions=["s-1", "s-2"])
        assert ai_peer.chat.call_args.kwargs["sessions"] == ["s-1", "s-2"]
        assert ai_peer.chat.call_args.kwargs["scope"] is None

    def test_unknown_session_returns_empty(self):
        mgr, _, _ = _manager()
        out = mgr.dialectic_query_detailed("nope", "q?")
        assert out["content"] == "" and out["evidence"] is None


# ── honcho_reasoning tool wiring ────────────────────────────────────────────


class TestReasoningToolStructured:
    def _provider(self) -> HonchoMemoryProvider:
        provider = HonchoMemoryProvider()
        provider._session_key = "cli:chat-1"
        provider._session_initialized = True
        provider._manager = MagicMock()
        return provider

    def test_invalid_response_format_is_rejected_client_side(self):
        provider = self._provider()
        out = provider._tool_reasoning({"query": "q", "response_format": {"type": "array"}})
        assert "Invalid response_format" in out
        provider._manager.dialectic_query_detailed.assert_not_called()

    def test_scope_and_sessions_combo_is_rejected_client_side(self):
        provider = self._provider()
        out = provider._tool_reasoning({"query": "q", "scope": "work", "sessions": ["s-1"]})
        assert "mutually exclusive" in out
        provider._manager.dialectic_query_detailed.assert_not_called()

    def test_valid_structured_call_passes_everything_through(self):
        provider = self._provider()
        provider._manager.dialectic_query_detailed.return_value = {
            "content": '{"mood": "calm"}', "evidence": None, "structured": True}
        schema = {"type": "object", "properties": {"mood": {"type": "string"}}}
        out = json.loads(provider._tool_reasoning({"query": "q", "response_format": schema}))
        assert out["result"] == '{"mood": "calm"}'
        kwargs = provider._manager.dialectic_query_detailed.call_args.kwargs
        assert kwargs["response_format"] is schema

    def test_evidence_included_only_when_requested(self):
        provider = self._provider()
        provider._manager.dialectic_query_detailed.return_value = {
            "content": "answer", "evidence": {"conclusions": []}, "structured": False}
        out = json.loads(provider._tool_reasoning({"query": "q", "include_evidence": True}))
        assert out["evidence"] == {"conclusions": []}

    def test_evidence_null_when_not_returned(self):
        provider = self._provider()
        provider._manager.dialectic_query_detailed.return_value = {
            "content": "answer", "evidence": None, "structured": False}
        out = json.loads(provider._tool_reasoning({"query": "q", "include_evidence": True}))
        assert out["evidence"] == {"conclusions": None}

    def test_plain_call_has_no_evidence_key(self):
        provider = self._provider()
        provider._manager.dialectic_query_detailed.return_value = {
            "content": "answer", "evidence": None, "structured": False}
        out = json.loads(provider._tool_reasoning({"query": "q"}))
        assert "evidence" not in out
