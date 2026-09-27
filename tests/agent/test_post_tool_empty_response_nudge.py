"""Regression for an empty final reply after tool calls (NSFW e2e incident).

Sequence observed in a real session: ``tool_calls`` -> ``tool_calls`` -> a final
assistant message with ``finish_reason="stop"``, empty ``content`` and non-empty
``reasoning`` (the model's planning). The reasoning-only clean-stop promotion in
``agent/turn_final_response.py`` used to promote that planning text to the visible
answer BEFORE the empty-response ladder ran, so the #9400 post-tool nudge / the
``empty_response_guard`` never fired. The leaked thinking then reached delivery,
where a persona ``transform_llm_output`` hook scrubbed it to a placeholder token.

Invariants pinned here:
- After tool calls, a clean-stop empty content triggers the #9400 re-prompt (the
  existing ladder), and the model's NEXT non-empty answer is what gets delivered.
- The hidden reasoning is never promoted into the final response.
"""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest

# Stub optional heavy imports so run_agent imports cleanly in isolation.
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


def _build_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    from run_agent import AIAgent

    agent = AIAgent(
        model="test-model",
        api_key="sk-dummy",
        base_url="https://example.invalid/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    agent._disable_streaming = True
    return agent


def _tool_call(name: str = "read_file", arguments: str = '{"path": "x.md"}'):
    return SimpleNamespace(
        id="call_lookup",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _tool_call_response():
    msg = SimpleNamespace(
        content="", tool_calls=[_tool_call()],
        reasoning=None, reasoning_content=None, reasoning_details=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")],
        usage=None, model="test-model",
    )


def _reasoning_only_stop(planning: str):
    """Final clean stop with empty content but planning text in reasoning."""
    msg = SimpleNamespace(
        content="", tool_calls=None,
        reasoning=planning, reasoning_content=planning, reasoning_details=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=None, model="test-model",
    )


def _text_stop(text: str):
    msg = SimpleNamespace(
        content=text, tool_calls=None,
        reasoning=None, reasoning_content=None, reasoning_details=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=None, model="test-model",
    )


@pytest.fixture()
def silence_tool_dispatch(monkeypatch):
    monkeypatch.setattr(
        "model_tools.handle_function_call",
        lambda name, args, task_id=None, **kwargs: json.dumps({"ok": True}),
    )


def test_reasoning_only_after_tools_is_not_promoted(tmp_path, monkeypatch, silence_tool_dispatch):
    """The #9400 nudge must fire and the follow-up answer must win; planning reasoning
    must never become the delivered final response."""
    agent = _build_agent(tmp_path, monkeypatch)
    planning = "Good, I have the skill. This is rung-4 territory — plan the next tool call."
    responses = [
        _tool_call_response(),
        _tool_call_response(),
        _reasoning_only_stop(planning),
        _text_stop("Here is the real answer."),
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    result = agent.run_conversation("do the thing")

    # 2 tool rounds + the empty attempt + the nudged continuation = 4 API calls.
    # Without the fix the reasoning promotion would have ended the turn after the
    # third (empty) call, with the planning text as the final response.
    assert result["api_calls"] == 4
    assert result["final_response"] == "Here is the real answer."
    assert planning not in result["final_response"]

    # The re-prompt is the existing #9400 nudge, injected as a user turn.
    assert result["messages"][-1]["role"] == "assistant"
    assert result["messages"][-1]["content"] == "Here is the real answer."


def test_reasoning_only_without_tools_still_promoted(tmp_path, monkeypatch):
    """A single clean-stop reasoning-only turn with NO tool call keeps the original
    PR #48795 behaviour: the reasoning IS the answer and is promoted after one call."""
    agent = _build_agent(tmp_path, monkeypatch)
    monkeypatch.setattr(
        agent, "_interruptible_api_call",
        lambda api_kwargs: _reasoning_only_stop("The answer is 42."),
    )

    result = agent.run_conversation("what is the answer?")

    assert result["api_calls"] == 1
    assert result["final_response"] == "The answer is 42."
