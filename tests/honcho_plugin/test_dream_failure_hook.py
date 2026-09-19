"""Tests for the Honcho dream-failure DLQ hook (mnemosyne composite provider).

Contract: when session-end dream scheduling fails (falsy return or raised
exception), the registered ``_DREAM_FAILURE_HOOK`` fires once with a
``{"kind": "honcho_dream", "session_key": ...}`` payload; no consumer and a
throwing consumer are both fail-open — ``on_session_end`` never raises.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.memory.honcho import (
    HonchoMemoryProvider,
    set_dream_failure_hook,
)


@pytest.fixture(autouse=True)
def _reset_hook():
    set_dream_failure_hook(None)
    yield
    set_dream_failure_hook(None)


def _provider(manager, **config_overrides):
    provider = HonchoMemoryProvider()
    provider._manager = manager
    provider._session_key = "cli:chat-1"
    provider._session_initialized = True
    provider._config = SimpleNamespace(save_messages=True, **config_overrides)
    return provider


class TestDreamFailureHook:
    def test_hook_called_when_dream_schedule_returns_falsy(self):
        manager = MagicMock()
        manager.schedule_session_dream.return_value = False
        calls = []
        set_dream_failure_hook(lambda payload: calls.append(payload))
        provider = _provider(manager, dreams_enabled=True)
        provider.on_session_end([])
        assert calls == [{"kind": "honcho_dream", "session_key": "cli:chat-1"}]

    def test_hook_called_when_dream_schedule_raises(self):
        manager = MagicMock()
        manager.schedule_session_dream.side_effect = RuntimeError("queue down")
        calls = []
        set_dream_failure_hook(lambda payload: calls.append(payload))
        provider = _provider(manager, dreams_enabled=True)
        provider.on_session_end([])
        assert calls == [{"kind": "honcho_dream", "session_key": "cli:chat-1"}]

    def test_hook_not_called_when_dream_succeeds(self):
        manager = MagicMock()
        manager.schedule_session_dream.return_value = True
        calls = []
        set_dream_failure_hook(lambda payload: calls.append(payload))
        provider = _provider(manager, dreams_enabled=True)
        provider.on_session_end([])
        assert calls == []

    def test_no_hook_nothing_raises(self):
        manager = MagicMock()
        manager.schedule_session_dream.return_value = False
        set_dream_failure_hook(None)
        provider = _provider(manager, dreams_enabled=True)
        provider.on_session_end([])  # fail-open: no raise

    def test_throwing_hook_is_swallowed(self):
        manager = MagicMock()
        manager.schedule_session_dream.return_value = False

        def _throwing_hook(_payload):
            raise RuntimeError("hook boom")

        set_dream_failure_hook(_throwing_hook)
        provider = _provider(manager, dreams_enabled=True)
        provider.on_session_end([])  # fail-open: no raise