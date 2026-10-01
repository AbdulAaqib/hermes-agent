"""Regression tests for the configured output-token cap on main-loop requests.

The bug: cron/oneshot requests asked for an enormous provider default (a 402
``You requested up to 131072 tokens``) even though the operator set
``providers.<x>.extra_body.max_tokens`` / ``model.max_tokens``. The cap must apply to
every route, including when the caller sets no explicit max_tokens.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from agent.agent_init import _resolve_configured_max_output_tokens
from agent.chat_completion_helpers import _effective_max_tokens
from agent.transports.chat_completions import _apply_max_tokens


def test_effective_max_tokens_uses_cap_when_unset():
    agent = SimpleNamespace(max_tokens=None, _config_max_output_tokens=8192)
    assert _effective_max_tokens(agent) == 8192


def test_effective_max_tokens_clamps_explicit_value():
    agent = SimpleNamespace(max_tokens=131072, _config_max_output_tokens=8192)
    assert _effective_max_tokens(agent) == 8192


def test_effective_max_tokens_passthrough_without_cap():
    agent = SimpleNamespace(max_tokens=4096, _config_max_output_tokens=None)
    assert _effective_max_tokens(agent) == 4096
    assert _effective_max_tokens(SimpleNamespace(max_tokens=None, _config_max_output_tokens=None)) is None


def test_apply_max_tokens_writes_capped_value():
    kwargs: dict = {}
    _apply_max_tokens(kwargs, "m", None, {
        "max_tokens_param_fn": lambda v: {"max_tokens": v},
        "max_tokens": 8192,
    })
    assert kwargs["max_tokens"] == 8192


def test_apply_max_tokens_ephemeral_recovery_still_wins():
    kwargs: dict = {}
    _apply_max_tokens(kwargs, "m", None, {
        "max_tokens_param_fn": lambda v: {"max_tokens": v},
        "max_tokens": 8192,
        "ephemeral_max_output_tokens": 512,
    })
    assert kwargs["max_tokens"] == 512


def test_resolve_cap_from_model_config():
    agent = SimpleNamespace(max_tokens=None, provider="openrouter", model="m", base_url="https://openrouter.ai/api/v1")
    assert _resolve_configured_max_output_tokens(agent, {"model": {"max_tokens": 4096}}, []) == 4096


def test_resolve_cap_from_custom_provider_extra_body():
    agent = SimpleNamespace(
        max_tokens=None, provider="custom:openrouter-tuned",
        model="deepseek/deepseek-v4-pro", base_url="https://openrouter.ai/api/v1",
    )
    providers = [{
        "provider_key": "openrouter-tuned", "base_url": "https://openrouter.ai/api/v1",
        "extra_body": {"max_tokens": 8192},
    }]
    assert _resolve_configured_max_output_tokens(agent, {}, providers) == 8192


def test_explicit_caller_max_tokens_disables_cap():
    agent = SimpleNamespace(max_tokens=4096, provider="openrouter", model="m", base_url="https://openrouter.ai/api/v1")
    assert _resolve_configured_max_output_tokens(agent, {"model": {"max_tokens": 8192}}, []) is None


def test_resolve_cap_none_when_nothing_configured():
    agent = SimpleNamespace(max_tokens=None, provider="openrouter", model="m", base_url="https://openrouter.ai/api/v1")
    assert _resolve_configured_max_output_tokens(agent, {}, []) is None


def test_resolve_cap_end_to_end_from_providers_config(tmp_path, monkeypatch):
    """A cron/oneshot agent built from the live-style config picks up the 8192 cap."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = {
        "model": {
            "default": "deepseek/deepseek-v4-pro", "provider": "custom:openrouter-tuned",
            "base_url": "https://openrouter.ai/api/v1", "api_mode": "chat_completions",
        },
        "providers": {
            "openrouter-tuned": {
                "api": "https://openrouter.ai/api/v1", "transport": "chat_completions",
                "extra_body": {"temperature": 0.7, "max_tokens": 8192},
            },
        },
    }
    (tmp_path / "config.yaml").write_text(json.dumps(cfg), encoding="utf-8")
    from hermes_cli.config import get_compatible_custom_providers, load_config_readonly

    agent = SimpleNamespace(
        max_tokens=None, provider="custom:openrouter-tuned",
        model="deepseek/deepseek-v4-pro", base_url="https://openrouter.ai/api/v1",
    )
    cap = _resolve_configured_max_output_tokens(agent, load_config_readonly(), get_compatible_custom_providers())
    assert cap == 8192
    assert _effective_max_tokens(SimpleNamespace(max_tokens=None, _config_max_output_tokens=cap)) == 8192
