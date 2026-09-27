"""Honcho must not write anything while HERMES_E2E_READONLY=1.

Regression for the 2026-09-27 test pollution: sessions named after test prompts
were created in the workspace, each carrying a `<prior_memory_file>` migration
message. Root cause: the provider's initialize() was not gated, so it created
the session and uploaded prior memory regardless of the read-only switch.
"""

import json

import pytest

from plugins.memory.honcho import HonchoMemoryProvider


class _FakeManager:
    def __init__(self):
        self.sync = 0
        self.get_or_create_calls = 0
        self.set_card = 0
        self.get_card = 0
        self.conclusions = []
        self.flush = 0
        self.dream = 0

    def resolve_author_peer_id(self, *a, **k):
        return None

    def get_or_create(self, *a, **k):
        self.get_or_create_calls += 1
        return self

    def save(self, *a, **k):
        self.sync += 1

    def set_peer_card(self, *a, **k):
        self.set_card += 1
        return list(a[1]) if len(a) > 1 else []

    def get_peer_card(self, *a, **k):
        self.get_card += 1
        return ["alpha", "beta"]

    def create_conclusion(self, *a, **k):
        self.conclusions.append(a)
        return True

    def delete_conclusion(self, *a, **k):
        return True

    def list_conclusions(self, *a, **k):
        return []

    def flush_all(self, *a, **k):
        self.flush += 1

    def schedule_session_dream(self, *a, **k):
        self.dream += 1
        return True

    def stop_async_writer(self, *a, **k):
        pass

    def shutdown(self, *a, **k):
        pass


class _Cfg:
    save_messages = True
    dreams_enabled = True


@pytest.fixture
def readonly(monkeypatch):
    monkeypatch.setenv("HERMES_E2E_READONLY", "1")
    import plugins.memory.honcho as mod
    monkeypatch.setattr(mod, "_readonly_logged", False)


def test_initialize_readonly_creates_no_session(readonly):
    p = HonchoMemoryProvider()
    p.initialize("20260101_000000_test", platform="telegram")
    assert p._cron_skipped is True
    assert p._manager is None
    assert p._config is None


def test_writes_disabled_in_readonly(readonly):
    p = HonchoMemoryProvider()
    p._config = _Cfg()
    assert p._writes_enabled() is False


def test_session_write_paths_noop(readonly):
    p = HonchoMemoryProvider()
    p._config = _Cfg()
    fake = _FakeManager()
    p._manager = fake
    p._session_key = "k"
    p._session_initialized = True
    p.sync_turn("user hi", "assistant hello")
    p.on_memory_write("add", "user", "Q likes tea")
    p.on_session_end([{"role": "user", "content": "hi"}])
    assert fake.sync == 0
    assert fake.get_or_create_calls == 0
    assert fake.flush == 0
    assert fake.dream == 0
    assert fake.conclusions == []


def test_profile_card_write_blocked_read_allowed(readonly):
    p = HonchoMemoryProvider()
    p._config = _Cfg()
    fake = _FakeManager()
    p._manager = fake
    p._session_key = "k"
    p._session_initialized = True

    write = json.loads(p.handle_tool_call("honcho_profile", {"card": ["new"]}))
    assert "error" in write
    assert fake.set_card == 0

    read = json.loads(p.handle_tool_call("honcho_profile", {}))
    assert fake.get_card == 1
    assert read["result"] == ["alpha", "beta"]


def test_conclude_write_blocked_list_allowed(readonly):
    p = HonchoMemoryProvider()
    p._config = _Cfg()
    fake = _FakeManager()
    p._manager = fake
    p._session_key = "k"
    p._session_initialized = True

    write = json.loads(p.handle_tool_call("honcho_conclude", {"conclusion": "Q likes tea"}))
    assert "error" in write
    assert fake.conclusions == []

    listed = json.loads(p.handle_tool_call("honcho_conclude", {"list": True}))
    assert "conclusions" in listed


def test_writes_enabled_without_flag(monkeypatch):
    monkeypatch.delenv("HERMES_E2E_READONLY", raising=False)
    p = HonchoMemoryProvider()
    p._config = _Cfg()
    assert p._writes_enabled() is True
