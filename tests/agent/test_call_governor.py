"""Regression tests for the central call governor + cost ledger.

Covers admission (chat never denied, per-task caps, min interval, dedupe, global breaker),
ledger roll-up + provider-reported cost, and the label/recording helpers used by the aux
call chokepoint and the main-loop hook. All state lives under the per-test HERMES_HOME.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent.call_governor as cg


@pytest.fixture(autouse=True)
def _clear_policy():
    cg.set_policy_for_tests(None)
    yield
    cg.set_policy_for_tests(None)


def _policy(**overrides):
    policy = {"enabled": True, "daily_budget_usd": 100.0, "tasks": {}}
    policy.update(overrides)
    cg.set_policy_for_tests(policy)
    return policy


def test_chat_never_denied_even_when_breaker_tripped():
    _policy(daily_budget_usd=0.0, tasks={"chat": {"daily_cap": 0}})
    cg.record("chat", prompt_tokens=1, completion_tokens=1, cost_usd=1.0)
    decision = cg.admit("chat")
    assert decision.allow is True
    assert decision.reason == "chat"


def test_moa_tasks_never_denied():
    """MoA reference/aggregator calls run inside a chat turn and must survive the breaker."""
    _policy(daily_budget_usd=0.0)
    cg.record("chat", prompt_tokens=1, completion_tokens=1, cost_usd=5.0)
    assert cg.admit("moa_reference").allow is True
    assert cg.admit("moa_aggregator").allow is True


def test_daily_cap_denies_second_call():
    _policy(tasks={"title_generation": {"daily_cap": 1}})
    assert cg.admit("title_generation").allow is True
    denied = cg.admit("title_generation")
    assert denied.allow is False
    assert denied.reason == "daily_cap"


def test_min_interval_denies_until_elapsed(monkeypatch):
    _policy(tasks={"compression": {"min_interval_s": 60}})
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(cg.time, "time", lambda: clock["t"])
    assert cg.admit("compression").allow is True
    assert cg.admit("compression").reason == "min_interval"
    clock["t"] += 61
    assert cg.admit("compression").allow is True


def test_dedupe_same_hash_denied_within_ttl(monkeypatch):
    _policy(tasks={"scene_extract": {"dedupe_ttl_s": 300}})
    clock = {"t": 2_000_000.0}
    monkeypatch.setattr(cg.time, "time", lambda: clock["t"])
    assert cg.admit("scene_extract", inputs_hash="abc").allow is True
    assert cg.admit("scene_extract", inputs_hash="abc").reason == "dedupe"
    assert cg.admit("scene_extract", inputs_hash="other").allow is True
    clock["t"] += 301
    # New day file: the dedupe entry ages out of the TTL but is still present.
    assert cg.admit("scene_extract", inputs_hash="abc", session_id="").allow is True


def test_global_breaker_denies_non_chat_and_logs_once(caplog):
    _policy(daily_budget_usd=0.10, tasks={})
    cg.record("background_review", prompt_tokens=1, completion_tokens=1, cost_usd=0.11)
    assert cg.admit("title_generation").reason == "breaker"
    assert cg.admit("compression").reason == "breaker"
    assert cg.admit("chat").allow is True
    day = cg.read_day()
    assert day["breaker_logged"] is True


def test_disabled_governor_allows_everything():
    _policy(enabled=False, daily_budget_usd=0.0, tasks={"title_generation": {"daily_cap": 0}})
    assert cg.admit("title_generation").allow is True


def test_record_rolls_up_per_task_and_total():
    cg.record("title_generation", model="m", prompt_tokens=10, completion_tokens=2, cost_usd=0.001)
    cg.record("title_generation", model="m", prompt_tokens=5, completion_tokens=1, cost_usd=0.002)
    cg.record("vision", model="v", prompt_tokens=100, completion_tokens=0, cost_usd=0.01)
    day = cg.read_day()
    assert day["tasks"]["title_generation"]["calls"] == 2
    assert day["tasks"]["title_generation"]["prompt_tokens"] == 15
    assert day["tasks"]["title_generation"]["cost_usd"] == pytest.approx(0.003)
    assert day["total"]["calls"] == 3
    assert day["total"]["cost_usd"] == pytest.approx(0.013)


def test_record_writes_session_model_usage_when_handle_given():
    calls = []

    class _DB:
        def record_auxiliary_usage(self, session_id, task, **kwargs):
            calls.append((session_id, task, kwargs))

    cg.record(
        "title_generation", model="m", provider="openrouter", prompt_tokens=3,
        completion_tokens=1, cost_usd=0.0005, session_id="s1", session_db=_DB(),
    )
    assert len(calls) == 1
    assert calls[0][0] == "s1"
    assert calls[0][1] == "title_generation"
    assert calls[0][2]["input_tokens"] == 3


def test_extract_usage_and_cost_prefers_provider_reported_cost():
    response = SimpleNamespace(
        model="deepseek/deepseek-v4.1-flash",
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, cost=0.0042),
    )
    fields = cg.extract_usage_and_cost(response, provider="openrouter")
    assert fields["cost_usd"] == pytest.approx(0.0042)
    assert fields["actual_cost_usd"] == pytest.approx(0.0042)
    assert fields["prompt_tokens"] == 100
    assert fields["completion_tokens"] == 20


def test_record_response_none_usage_is_noop():
    response = SimpleNamespace(model="m", usage=None)
    cg.record_response(response, "title_generation")
    assert cg.read_day()["total"]["calls"] == 0


def test_state_file_is_valid_json_and_under_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cg.record("vision", prompt_tokens=1, completion_tokens=1, cost_usd=0.0)
    path = tmp_path / "state" / "call_governor" / f"{cg.utc_day()}.json"
    assert path.exists()
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["date"] == cg.utc_day()
