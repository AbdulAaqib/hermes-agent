"""CallDenied must propagate out of call_llm but be swallowed by the aux callers that use
title generation / goal judging — a governor denial degrades the feature, never the turn.
"""
from __future__ import annotations

import pytest

import agent.call_governor as cg
from agent.call_governor import CallDenied


@pytest.fixture(autouse=True)
def _clear_policy():
    cg.set_policy_for_tests(None)
    yield
    cg.set_policy_for_tests(None)


def test_admit_denies_and_call_llm_raises(monkeypatch):
    cg.set_policy_for_tests({"enabled": True, "daily_budget_usd": 100.0,
                             "tasks": {"title_generation": {"daily_cap": 0}}})
    from agent.auxiliary_client import call_llm

    with pytest.raises(CallDenied):
        call_llm(task="title_generation", messages=[{"role": "user", "content": "hi"}])


def test_call_denied_is_an_exception():
    assert issubclass(CallDenied, Exception)


def test_title_generation_soft_fails(monkeypatch):
    import agent.title_generator as tg

    monkeypatch.setattr(tg, "_auto_title_enabled", lambda: True)
    monkeypatch.setattr(tg, "call_llm", lambda **kw: (_ for _ in ()).throw(CallDenied("title_generation", "breaker")))
    assert tg.generate_title("a perfectly ordinary first user message") is None


def test_goal_judge_soft_fails(monkeypatch):
    import agent.auxiliary_client as ac
    from hermes_cli.goals import judge_goal

    monkeypatch.setattr(ac, "call_llm", lambda **kw: (_ for _ in ()).throw(CallDenied("goal_judge", "breaker")))
    verdict, reason, parse_failed, wait_directive, transport_failed = judge_goal(
        "finish the report", "still working on it")
    assert verdict == "continue"
    assert transport_failed is True
