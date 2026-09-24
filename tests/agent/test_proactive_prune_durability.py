"""Durability invariants for the proactive tool-result prune (#120582).

The prune demotes what the model replays; it must never destroy the durable record.
Pass 2 (`_summarize_tool_result`) and pass 3 (`_truncate_tool_call_args_at`) rewrite the
ACTIVE transcript, but `archive_and_compact` soft-archives the pre-rewrite rows, so the
original tool output and the original tool-call arguments stay byte-exact on disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

from hermes_state import SessionDB

_BIG_RESULT_CHARS = 24_000
_BIG_SCRIPT = "#!/bin/sh\n" + ("PAYLOAD_VARIABLE=abcdefghij\n" * 400)


def _assistant_call(call_id: str, arguments: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "write_file", "arguments": arguments},
        }],
    }


def _history() -> list[dict]:
    big_args = json.dumps({"path": "/tmp/s.sh", "content": _BIG_SCRIPT})
    messages: list[dict] = [{"role": "user", "content": "start"}]
    for index in range(8):
        call_id = f"call_{index}"
        messages.append(_assistant_call(call_id, big_args if index < 3 else '{"a": 1}'))
        messages.append({
            "role": "tool", "tool_call_id": call_id,
            "content": chr(65 + index) * _BIG_RESULT_CHARS if index < 3 else "ok",
        })
    return messages


def _pruning_agent(db: SessionDB, session_id: str):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
            quiet_mode=True, session_db=db, session_id=session_id, platform="telegram",
            skip_context_files=True, skip_memory=True,
        )
    compressor = agent.context_compressor
    compressor.proactive_prune_tokens = 48_000
    compressor.proactive_prune_min_result_chars = 8_000
    compressor.proactive_prune_min_reclaim_tokens = 4_096
    compressor.protect_first_n = 2
    compressor.protect_last_n = 4
    return agent


def _script_payloads(rows) -> list[str]:
    """Every ``content`` argument carried by the tool calls on *rows*."""
    payloads: list[str] = []
    for row in rows:
        raw = row.get("tool_calls")
        if not raw:
            continue
        calls = json.loads(raw) if isinstance(raw, str) else raw
        for call in calls or []:
            arguments = (call.get("function") or {}).get("arguments") or ""
            try:
                payloads.append((json.loads(arguments) or {}).get("content") or "")
            except (TypeError, ValueError):
                payloads.append(arguments)
    return payloads


def test_prune_demotes_the_replay_view_but_keeps_the_durable_originals(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PRUNE_DURABILITY"
    db.create_session(session_id, source="telegram")
    db.append_messages_batch(session_id, _history())

    agent = _pruning_agent(db, session_id)
    _, pruned_count = agent.context_compressor.prune_tool_results_only(
        db.get_messages_as_conversation(session_id), current_tokens=120_000,
    )
    assert pruned_count >= 1

    rows = [dict(row) for row in db.get_messages(session_id, include_inactive=True)]
    active = [row for row in rows if row["active"]]
    archived = [row for row in rows if not row["active"]]
    assert archived, "prune committed without soft-archiving the pre-rewrite generation"

    # The replay view is demoted: that is the point of the prune.
    assert not any("A" * _BIG_RESULT_CHARS in (row["content"] or "") for row in active)
    assert not any(_BIG_SCRIPT in payload for payload in _script_payloads(active))

    # The durable record is not: originals survive byte-exact one generation behind.
    assert any("A" * _BIG_RESULT_CHARS in (row["content"] or "") for row in archived)
    assert any(_BIG_SCRIPT in payload for payload in _script_payloads(archived))
