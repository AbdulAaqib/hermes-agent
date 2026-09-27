"""`hermes send` must refuse to deliver while a read-only E2E run is active.

The env marker alone is not enough: a shell can `unset HERMES_E2E_READONLY`
before invoking the CLI (the 2026-09-27 leak). e2e_common's RunLock writes a
pid-stamped sentinel for the duration of a run; the CLI honors both.
"""

import argparse

import pytest

from hermes_cli import send_cmd


def _send_args(**over):
    base = dict(to="telegram", message="hi", list_targets=False, file=None,
                subject=None, quiet=False, json=False)
    base.update(over)
    return argparse.Namespace(**base)


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_E2E_READONLY", raising=False)
    return tmp_path


def _sentinel(home):
    p = home / "tests-e2e" / "output" / ".e2e_active"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def test_pid_alive_current_and_dead():
    import os
    assert send_cmd._pid_alive(os.getpid()) is True
    assert send_cmd._pid_alive(2 ** 30) is False


def test_send_allowed_without_marker_or_sentinel(hermes_home):
    assert send_cmd._e2e_send_blocked() is None


def test_send_blocked_by_readonly_env(hermes_home, monkeypatch):
    monkeypatch.setenv("HERMES_E2E_READONLY", "1")
    assert send_cmd._e2e_send_blocked() == "HERMES_E2E_READONLY=1"


def test_send_blocked_by_live_sentinel(hermes_home):
    import os
    _sentinel(hermes_home).write_text(str(os.getpid()))
    assert "live E2E run" in (send_cmd._e2e_send_blocked() or "")


def test_stale_sentinel_ignored(hermes_home):
    _sentinel(hermes_home).write_text(str(2 ** 30))
    assert send_cmd._e2e_send_blocked() is None


def test_cmd_send_exits_nonzero_when_blocked(hermes_home, monkeypatch):
    monkeypatch.setenv("HERMES_E2E_READONLY", "1")
    with pytest.raises(SystemExit) as ei:
        send_cmd.cmd_send(_send_args())
    assert ei.value.code == send_cmd._FAILURE_EXIT
