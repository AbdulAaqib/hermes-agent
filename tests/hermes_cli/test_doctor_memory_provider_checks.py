"""Memory-provider ``doctor_checks()`` capability rendering in hermes doctor.

The plugin hook API has no doctor/health hook, so a memory provider may
expose ``doctor_checks()`` returning rows shaped
``{"status": "ok"|"warn"|"fail", "label", "detail", "fix"}``; doctor renders
them color-coded and queues fail fixes in its remediation list.
"""
from __future__ import annotations

import pytest

from hermes_cli import doctor_state


class _Provider:
    def __init__(self, rows=None, raises=False):
        self._rows = rows if rows is not None else []
        self._raises = raises

    def is_available(self):
        return True

    def doctor_checks(self):
        if self._raises:
            raise RuntimeError("sweep exploded")
        return self._rows


class _BareProvider:
    """No doctor_checks capability — generic rows only."""

    def is_available(self):
        return True


@pytest.fixture
def patch_provider(monkeypatch):
    import plugins.memory as mem

    def _patch(provider):
        monkeypatch.setattr(mem, "load_memory_provider", lambda name: provider)

    return _patch


def test_rows_render_and_fail_fix_queued(patch_provider, capsys):
    patch_provider(_Provider(rows=[
        {"status": "ok", "label": "Hindsight daemon healthy", "detail": "url=http://localhost:8888"},
        {"status": "warn", "label": "Retry queue has 2 pending entries",
         "fix": "inspect ~/.hermes/retry_queue.jsonl"},
        {"status": "fail", "label": "Honcho workspace unreachable", "detail": "timeout",
         "fix": "check the Honcho daemon and HONCHO_API_KEY"},
    ]))
    issues = []
    doctor_state._memory_provider_generic("mnemosyne", issues)
    out = capsys.readouterr().out
    assert "mnemosyne provider active" in out
    assert "Hindsight daemon healthy" in out
    assert "Retry queue has 2 pending entries" in out
    assert "Honcho workspace unreachable" in out
    assert issues == ["check the Honcho daemon and HONCHO_API_KEY"], \
        "fail rows with a fix must land in doctor's remediation list"


def test_warn_fix_printed_as_info(patch_provider, capsys):
    patch_provider(_Provider(rows=[
        {"status": "warn", "label": "Honcho circuit breaker OPEN",
         "fix": "verify the bridge, then wait for reset_timeout_s"},
    ]))
    issues = []
    doctor_state._memory_provider_generic("mnemosyne", issues)
    out = capsys.readouterr().out
    assert "Honcho circuit breaker OPEN" in out
    assert "fix: verify the bridge" in out
    assert issues == [], "warn fixes are printed inline, not queued as failures"


def test_raising_sweep_is_warn_not_crash(patch_provider, capsys):
    patch_provider(_Provider(raises=True))
    issues = []
    doctor_state._memory_provider_generic("mnemosyne", issues)  # must not raise
    assert "health sweep failed" in capsys.readouterr().out
    assert issues == []


def test_provider_without_capability_stays_generic(patch_provider, capsys):
    patch_provider(_BareProvider())
    issues = []
    doctor_state._memory_provider_generic("mnemosyne", issues)
    out = capsys.readouterr().out
    assert "mnemosyne provider active" in out
    assert issues == []
