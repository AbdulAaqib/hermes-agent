"""Regression for the 2026-09-27 read-only test leak.

A read-only E2E turn discovered HERMES_E2E_READONLY in its own env via the
terminal tool, ran `unset HERMES_E2E_READONLY && hermes send --to telegram
"MEDIA:..."`, and a test image reached Q. The terminal tool must refuse any
command that re-enters the messaging/cron surface, reaches Telegram directly,
or scrubs the marker while the AGENT PROCESS is read-only.
"""

import json

import pytest

from tools.terminal_tool_guards import e2e_readonly_block

READONLY_ERR = "blocked: E2E read-only mode"


@pytest.fixture
def readonly(monkeypatch):
    monkeypatch.setenv("HERMES_E2E_READONLY", "1")


@pytest.mark.parametrize("command", [
    "hermes send --to telegram \"MEDIA:/tmp/x.jpg\"",
    "HERMES SEND --to telegram hi",
    "hermes cron run abc123",
    "hermes chat -Q --oneshot -q hi",
    "unset HERMES_E2E_READONLY && hermes send --to telegram hi",
    "env -u HERMES_E2E_READONLY hermes send --to telegram hi",
    "curl https://api.telegram.org/botTOKEN/sendPhoto",
    "echo $HERMES_E2E_READONLY",
])
def test_blocked_commands(readonly, command):
    blocked = e2e_readonly_block(command)
    assert blocked is not None
    payload = json.loads(blocked)
    assert payload["error"] == READONLY_ERR
    assert payload["status"] == "blocked"


@pytest.mark.parametrize("command", [
    "ls -la",
    "git status",
    "python3 -m pytest -q",
    "hermes doctor",
    "curl https://example.com/data.json",
])
def test_harmless_commands_pass(readonly, command):
    assert e2e_readonly_block(command) is None


def test_no_gate_without_readonly_env(monkeypatch):
    monkeypatch.delenv("HERMES_E2E_READONLY", raising=False)
    assert e2e_readonly_block("hermes send --to telegram hi") is None


def test_finalize_child_env_propagates_readonly(monkeypatch):
    from tools.environments.local import _finalize_child_env

    monkeypatch.setenv("HERMES_E2E_READONLY", "1")
    assert _finalize_child_env({})["HERMES_E2E_READONLY"] == "1"

    monkeypatch.delenv("HERMES_E2E_READONLY", raising=False)
    assert "HERMES_E2E_READONLY" not in _finalize_child_env({})
