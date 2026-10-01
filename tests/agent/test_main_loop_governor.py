"""Regression tests for main-loop task labelling (chat / cron / e2e / delegate)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import agent.main_loop_governor as mlg


@pytest.fixture(autouse=True)
def _clear_cache():
    mlg._cron_names_cache.clear()
    yield
    mlg._cron_names_cache.clear()


def _agent(**kw):
    base = dict(platform="telegram", session_id="s-1", is_subagent=False, _session_db=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_live_chat_platform_labels_chat():
    assert mlg.resolve_main_loop_task(_agent(platform="telegram")) == "chat"
    assert mlg.resolve_main_loop_task(_agent(platform="cli")) == "chat"


def test_subagent_labels_delegate():
    assert mlg.resolve_main_loop_task(_agent(is_subagent=True)) == "delegate"


def test_e2e_source_labels_e2e():
    class _DB:
        def get_session(self, session_id):
            return {"source": "e2e"}

    assert mlg.resolve_main_loop_task(_agent(platform="cli", _session_db=_DB())) == "e2e"


def test_cron_session_resolves_job_name(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    (home / "cron").mkdir(parents=True)
    (home / "cron" / "jobs.json").write_text(
        json.dumps({"jobs": [{"id": "abcdef123456", "name": "Morning Wakeup"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mlg, "_hermes_home", lambda: home)
    agent = _agent(platform="cron", session_id="cron_abcdef123456_20260928_080000")
    assert mlg.resolve_main_loop_task(agent) == "cron:Morning Wakeup"


def test_cron_session_unknown_job_falls_back_to_id(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    (home / "cron").mkdir(parents=True)
    (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": []}), encoding="utf-8")
    monkeypatch.setattr(mlg, "_hermes_home", lambda: home)
    agent = _agent(platform="cron", session_id="cron_deadbeef0000_20260928_080000")
    assert mlg.resolve_main_loop_task(agent) == "cron:deadbeef0000"


def test_cron_platform_without_session_still_cron():
    assert mlg.resolve_main_loop_task(_agent(platform="cron", session_id="x")).startswith("cron:")


def test_record_main_loop_response_rolls_up():
    import agent.call_governor as cg

    cg.set_policy_for_tests(None)
    response = SimpleNamespace(
        model="deepseek/deepseek-v4.1-flash",
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10, cost=0.001),
    )
    agent = _agent(provider="openrouter", base_url="https://openrouter.ai/api/v1")
    mlg.record_main_loop_response(agent, response, "chat")
    assert cg.read_day()["tasks"]["chat"]["calls"] == 1
