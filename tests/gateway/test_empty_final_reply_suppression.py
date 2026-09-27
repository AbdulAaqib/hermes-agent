"""The gateway must never deliver a transform-hook "empty reply" placeholder.

A persona ``transform_llm_output`` hook (e.g. a leak scrubber) can substitute a generic
token like ``[message unavailable]`` for a draft it refused to render. That token is not
authored prose, so the delivery shape step drops it and treats the turn as intentional
silence (send nothing), logging ``empty final reply suppressed``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.response_filters import is_empty_final_reply_placeholder


@pytest.mark.parametrize("token", [
    "[message unavailable]",
    "[MESSAGE UNAVAILABLE]",
    "  [message unavailable]  ",
])
def test_placeholder_detected(token):
    assert is_empty_final_reply_placeholder(token) is True


@pytest.mark.parametrize("text", [
    "",
    "   ",
    "(empty)",
    "Here is a real answer.",
    "The message unavailable to the API was retried.",  # prose, not the exact token
])
def test_non_placeholder_not_detected(text):
    assert is_empty_final_reply_placeholder(text) is False


@pytest.mark.asyncio
async def test_shape_step_suppresses_placeholder(caplog):
    from gateway.run_turn import GatewayTurnMixin

    runner = GatewayTurnMixin()
    source = SimpleNamespace(chat_id="chat-1", platform="discord", user_id="u1", thread_id=None)
    session_entry = SimpleNamespace(session_id="sess-1")
    agent_result = {"final_response": "[message unavailable]", "api_calls": 3, "messages": []}

    with caplog.at_level("WARNING"):
        response, intentional_silence, messages = await runner._hmwa_shape_agent_response(
            agent_result, source, history=[], session_entry=session_entry, session_key=None,
            _quick_key="q", run_generation=1, _run_start_session_id="sess-1",
            _platform_name="discord", _msg_start_time=0.0,
        )

    assert response == ""
    assert intentional_silence is True
    assert messages == []
    assert any("empty final reply suppressed" in r.message for r in caplog.records)
