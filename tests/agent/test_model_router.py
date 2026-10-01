"""Unit tests for the heat-routed model selector (``agent/model_router``).

Covers the routing policy matrix (rung + incoming-message heat + cron allowlist), the
lewd_lens signal readers (persisted rung with TTL, classifier lexicon, cron job name),
the in-place swap/fallback pinning, and the plain-text refusal safety net.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent import model_router


BASE_CONFIG = {
    "enabled": True,
    "cheap": {"provider": "opencode-go", "model": "deepseek-v4.1-flash"},
    "heat": {"provider": "custom:openrouter-tuned", "model": "deepseek/deepseek-v4-pro"},
    "heat_rung_threshold": 3,
    "refusal_fallback": True,
    "explicit_message_heat": True,
}


class FakeAgent:
    def __init__(self, provider="custom:openrouter-tuned", model="deepseek/deepseek-v4-pro",
                 pending_fallback=True):
        self.provider = provider
        self.model = model
        self._fallback_chain = []
        self._fallback_model = None
        self._fallback_index = 0
        self._fallback_activated = False
        self._fallback_pending = pending_fallback

    def _has_pending_fallback(self):
        return self._fallback_pending


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setattr(model_router, "get_config", lambda: dict(BASE_CONFIG))
    monkeypatch.setattr(model_router, "enabled", lambda: True)
    return BASE_CONFIG


# -- pure policy -----------------------------------------------------------------


@pytest.mark.parametrize(
    "rung,msg_heat,kind,allow,expected",
    [
        (1, False, "live", False, "cheap"),
        (2, False, "live", False, "cheap"),
        (3, False, "live", False, "heat"),
        (4, False, "live", False, "heat"),
        (1, True, "live", False, "heat"),
        (3, True, "live", False, "heat"),
        (1, False, "cron", False, "cheap"),
        (4, False, "cron", False, "cheap"),
        (1, False, "cron", True, "cheap"),
        (3, False, "cron", True, "heat"),
        (4, False, "cron", True, "heat"),
    ],
)
def test_decide_route_matrix(rung, msg_heat, kind, allow, expected):
    assert model_router.decide_route(
        BASE_CONFIG, rung=rung, msg_heat=msg_heat, session_kind=kind, cron_allowlisted=allow,
    ) == expected


def test_refusal_regex_shapes():
    assert model_router.is_refusal("I can't help with that.")
    assert model_router.is_refusal("I'm sorry, but I won't continue.")
    assert model_router.is_refusal("As an AI, I cannot engage with this.")
    # In-character filth must NOT be treated as a refusal.
    assert not model_router.is_refusal("breed me till i can't think straight")
    assert not model_router.is_refusal("don't stop, i can't take it")
    assert not model_router.is_refusal("")


# -- signal readers --------------------------------------------------------------


def test_current_rung_reads_state_and_ttl(monkeypatch, tmp_path):
    plug = tmp_path / "plugins" / "lewd_lens"
    plug.mkdir(parents=True)
    monkeypatch.setattr(model_router, "_home", lambda: tmp_path)

    (plug / "state.json").write_text(json.dumps({"escalation_level": 3, "escalation_ts": 1000.0}))
    # Fresh enough (now ~1000s) -> rung 3.
    assert model_router.current_rung(now=1500.0) == 3
    # Stale beyond the TTL -> cozy base.
    assert model_router.current_rung(now=1000.0 + model_router.ESCALATION_TTL_S + 1) == 1
    # Clamped / missing.
    (plug / "state.json").write_text(json.dumps({"escalation_level": 99, "escalation_ts": 0}))
    assert model_router.current_rung(now=1.0) == model_router.ESCALATION_MAX


def test_message_is_heat_uses_plugin_classifier(monkeypatch, tmp_path):
    plug = tmp_path / "plugins" / "lewd_lens"
    plug.mkdir(parents=True)
    (plug / "classifier.py").write_text(
        "DEFAULT_THRESHOLD = 0.35\n"
        "def is_lewd(text, threshold=DEFAULT_THRESHOLD):\n"
        "    return 'explicit-marker' in text\n"
    )
    monkeypatch.setattr(model_router, "_home", lambda: tmp_path)
    monkeypatch.setattr(model_router, "get_config", lambda: dict(BASE_CONFIG))
    assert model_router.message_is_heat("say explicit-marker now")
    assert not model_router.message_is_heat("good morning, love")
    assert not model_router.message_is_heat(None)


def test_cron_job_name_and_allowlist(monkeypatch, tmp_path):
    (tmp_path / "cron").mkdir(parents=True)
    (tmp_path / "cron" / "jobs.json").write_text(json.dumps({"jobs": [
        {"id": "job_tea_se", "name": "midday tease"},
        {"id": "ci_build", "name": "CI build notifier"},
    ]}))
    monkeypatch.setattr(model_router, "_home", lambda: tmp_path)
    monkeypatch.setattr(
        model_router, "_cron_allowlisted",
        lambda name: bool(name) and "tease" in name.lower(),
    )
    assert model_router.cron_job_name("cron_job_tea_se_20260928_120000") == "midday tease"
    assert model_router.cron_job_name("cron_ci_build_20260928_120000") == "CI build notifier"
    assert model_router.cron_job_name("telegram_123") == ""


# -- application -----------------------------------------------------------------


def test_maybe_route_turn_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(model_router, "get_config", lambda: {"enabled": False})
    called = []
    monkeypatch.setattr(model_router, "_apply_swap", lambda *a, **k: called.append(a))
    agent = FakeAgent()
    assert model_router.maybe_route_turn(agent, user_message="hi") is None
    assert called == []


def test_maybe_route_turn_cheap_pins_heat_fallback(monkeypatch, cfg):
    monkeypatch.setattr(model_router, "current_rung", lambda now=None: 1)
    monkeypatch.setattr(model_router, "message_is_heat", lambda text: False)
    monkeypatch.setattr(model_router, "_resolve_target", lambda p, m: {
        "api_key": "k", "base_url": "https://opencode.ai/zen/go/v1",
        "api_mode": "chat_completions"})
    swaps = []
    monkeypatch.setattr(
        model_router, "_apply_swap",
        lambda agent, provider, model, resolved: swaps.append((provider, model)) or True,
    )
    agent = FakeAgent()
    decision = model_router.maybe_route_turn(agent, user_message="morning!", platform="telegram")
    assert decision["route"] == "cheap" and decision["applied"] is True
    assert swaps == [("opencode-go", "deepseek-v4.1-flash")]
    assert agent._fallback_chain == [
        {"provider": "custom:openrouter-tuned", "model": "deepseek/deepseek-v4-pro"}
    ]


def test_maybe_route_turn_explicit_message_goes_heat(monkeypatch, cfg):
    monkeypatch.setattr(model_router, "current_rung", lambda now=None: 1)
    monkeypatch.setattr(model_router, "message_is_heat", lambda text: True)
    monkeypatch.setattr(model_router, "_resolve_target", lambda p, m: {
        "api_key": "k", "base_url": "https://openrouter.ai/api/v1", "api_mode": "chat_completions"})
    swaps = []
    monkeypatch.setattr(
        model_router, "_apply_swap",
        lambda agent, provider, model, resolved: swaps.append((provider, model)) or True,
    )
    # Start already on the heat route: no swap needed, and no cheap fallback pinning.
    agent = FakeAgent()
    decision = model_router.maybe_route_turn(agent, user_message="explicit request")
    assert decision["route"] == "heat" and decision["msg_heat"] is True
    assert swaps == []
    assert agent._fallback_chain == []


def test_pin_heat_fallback_snapshots_operator_chain(monkeypatch, cfg):
    monkeypatch.setattr(model_router, "current_rung", lambda now=None: 1)
    monkeypatch.setattr(model_router, "message_is_heat", lambda text: False)
    monkeypatch.setattr(model_router, "_resolve_target", lambda p, m: {
        "api_key": "k", "base_url": "https://opencode.ai/zen/go/v1",
        "api_mode": "chat_completions"})
    monkeypatch.setattr(model_router, "_apply_swap", lambda *a, **k: True)
    operator = [{"provider": "openrouter", "model": "deepseek/deepseek-v4.1-flash"}]
    agent = FakeAgent()
    agent._fallback_chain = list(operator)
    model_router.maybe_route_turn(agent, user_message="morning")
    assert agent._fallback_chain == [
        {"provider": "custom:openrouter-tuned", "model": "deepseek/deepseek-v4-pro"}
    ]
    assert agent._model_router_orig_fallback_chain == operator


def test_heat_turn_restores_operator_fallback_chain(monkeypatch, cfg):
    monkeypatch.setattr(model_router, "current_rung", lambda now=None: 3)
    monkeypatch.setattr(model_router, "message_is_heat", lambda text: False)
    monkeypatch.setattr(model_router, "_resolve_target", lambda p, m: {
        "api_key": "k", "base_url": "https://openrouter.ai/api/v1", "api_mode": "chat_completions"})
    swaps = []
    monkeypatch.setattr(
        model_router, "_apply_swap",
        lambda agent, provider, model, resolved: swaps.append((provider, model)) or True,
    )
    operator = [{"provider": "openrouter", "model": "deepseek/deepseek-v4.1-flash"}]
    agent = FakeAgent(provider="opencode-go", model="deepseek-v4.1-flash")
    # Simulate a previous cheap turn having pinned heat over the operator's chain.
    agent._model_router_orig_fallback_chain = list(operator)
    agent._fallback_chain = [{"provider": "custom:openrouter-tuned", "model": "deepseek/deepseek-v4-pro"}]
    decision = model_router.maybe_route_turn(agent, user_message="so... want me")
    assert decision["route"] == "heat"
    assert swaps == [("custom:openrouter-tuned", "deepseek/deepseek-v4-pro")]
    assert agent._fallback_chain == operator


def test_on_cheap_route(monkeypatch, cfg):
    assert model_router.on_cheap_route(FakeAgent(provider="opencode-go", model="deepseek-v4.1-flash"))
    assert not model_router.on_cheap_route(FakeAgent())


# -- refusal safety net ----------------------------------------------------------


def test_text_refusal_for_fallback_gating(monkeypatch, cfg):
    monkeypatch.setattr(model_router, "on_cheap_route", lambda agent: True)
    agent = FakeAgent(provider="opencode-go", model="deepseek-v4.1-flash")
    text = "I can't help with that."
    assert model_router.text_refusal_for_fallback(agent, text) == text
    # Once-per-turn guard.
    assert model_router.text_refusal_for_fallback(agent, text) is None
    # Non-refusal text passes through untouched.
    fresh = FakeAgent(provider="opencode-go", model="deepseek-v4.1-flash")
    assert model_router.text_refusal_for_fallback(fresh, "good night, love") is None


def test_text_refusal_requires_pending_fallback(monkeypatch, cfg):
    monkeypatch.setattr(model_router, "on_cheap_route", lambda agent: True)
    agent = FakeAgent(provider="opencode-go", model="deepseek-v4.1-flash", pending_fallback=False)
    assert model_router.text_refusal_for_fallback(agent, "I won't do that") is None


def test_text_refusal_skipped_when_not_cheap(monkeypatch, cfg):
    monkeypatch.setattr(model_router, "on_cheap_route", lambda agent: False)
    agent = FakeAgent()
    assert model_router.text_refusal_for_fallback(agent, "I cannot assist") is None


def test_resolve_target_opencode_go_real_imports(monkeypatch):
    """Real resolution chain (no mock) against a temp HERMES_HOME with a stub key."""
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "dummy-test-key")
    model_router._RESOLVE_CACHE.clear()
    resolved = model_router._resolve_target("opencode-go", "deepseek-v4.1-flash")
    assert resolved is not None
    assert resolved["provider"] == "opencode-go"
    assert resolved["base_url"].startswith("https://opencode.ai/")
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["api_key"] == "dummy-test-key"
    model_router._RESOLVE_CACHE.clear()


def test_fallback_counting_after_route(monkeypatch, cfg):
    model_router._FALLBACK_COUNTS.clear()
    monkeypatch.setattr(model_router, "on_cheap_route", lambda agent: False)
    agent = FakeAgent()
    agent._model_router_last_route = "cheap"
    model_router.note_route_fallback(agent, "quota")
    # Disabled routing never records (last route unset, not on cheap).
    other = FakeAgent()
    model_router.note_route_fallback(other, "error")
    assert model_router.fallback_counts() == {"quota": 1}
