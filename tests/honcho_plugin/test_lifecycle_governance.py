"""Behavior contracts for the Honcho consolidation lifecycle and retrieval tuning:

session set_configuration (summary/dream), dream scheduling at session end,
message metadata on writes, the peer-card fact cap, and context() tuning kwargs.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager
from plugins.memory.honcho.session_context import PEER_CARD_MAX_FACTS
from plugins.memory.honcho import HonchoMemoryProvider


def _config(**overrides):
    base = dict(
        write_frequency="async", dialectic_reasoning_level="low", dialectic_dynamic=True,
        dialectic_max_chars=600, dialectic_max_input_chars=10000,
        user_observe_me=True, user_observe_others=True, ai_observe_me=True, ai_observe_others=True,
        summary_enabled=None, messages_per_short_summary=None, messages_per_long_summary=None,
        dreams_enabled=True, search_top_k=None, search_max_distance=None, max_conclusions=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _manager(**config_overrides) -> HonchoSessionManager:
    mgr = HonchoSessionManager(honcho=MagicMock(), config=_config(**config_overrides))
    mgr._authed_call = lambda label, op: op()
    return mgr


def _cached_session(mgr: HonchoSessionManager) -> HonchoSession:
    session = HonchoSession(
        key="cli:chat-1", user_peer_id="q", assistant_peer_id="asuna", honcho_session_id="chat-1",
    )
    mgr._cache[session.key] = session
    return session


# ── session set_configuration (summary + dream) ─────────────────────────────


class TestSessionConfiguration:
    def test_defaults_apply_no_configuration(self):
        """Nothing configured -> no set_configuration call: server defaults stand."""
        mgr = _manager()
        assert mgr._session_configuration() is None

    def test_summary_settings_build_configuration(self):
        mgr = _manager(summary_enabled=True, messages_per_short_summary=10, messages_per_long_summary=40)
        config = mgr._session_configuration()
        assert config is not None
        assert config.summary.enabled is True
        assert config.summary.messages_per_short_summary == 10
        assert config.summary.messages_per_long_summary == 40
        assert config.dream is None  # dreams default on: nothing to push

    def test_disabling_dreams_pushes_dream_off(self):
        mgr = _manager(dreams_enabled=False)
        config = mgr._session_configuration()
        assert config is not None
        assert config.dream.enabled is False
        assert config.summary is None

    def test_apply_pushes_configuration_to_session(self):
        mgr = _manager(summary_enabled=False, messages_per_short_summary=5)
        sdk_session = MagicMock()
        mgr._sdk_session = lambda session_id: sdk_session
        mgr._apply_session_configuration("chat-1")
        sdk_session.set_configuration.assert_called_once()
        pushed = sdk_session.set_configuration.call_args.args[0]
        assert pushed.summary.enabled is False
        assert pushed.summary.messages_per_short_summary == 5

    def test_apply_skips_call_when_nothing_configured(self):
        mgr = _manager()
        sdk_session = MagicMock()
        mgr._sdk_session = lambda session_id: sdk_session
        mgr._apply_session_configuration("chat-1")
        sdk_session.set_configuration.assert_not_called()

    def test_apply_is_fail_open(self):
        mgr = _manager(summary_enabled=True)
        sdk_session = MagicMock()
        sdk_session.set_configuration.side_effect = RuntimeError("server rejected config")
        mgr._sdk_session = lambda session_id: sdk_session
        mgr._apply_session_configuration("chat-1")  # must not raise


# ── dream scheduling ────────────────────────────────────────────────────────


class TestScheduleDream:
    def _run_with_client(self, mgr, honcho_client, fn):
        from unittest.mock import patch

        with patch.object(HonchoSessionManager, "honcho", new_callable=lambda: property(lambda s: honcho_client)):
            return fn()

    def test_schedules_with_ai_observer_and_user_observed(self):
        mgr = _manager()
        session = _cached_session(mgr)
        honcho_client = MagicMock()

        result = self._run_with_client(mgr, honcho_client, lambda: mgr.schedule_session_dream(session.key))
        assert result is True
        honcho_client.schedule_dream.assert_called_once_with(
            observer=session.assistant_peer_id, observed=session.user_peer_id,
            session=session.honcho_session_id,
        )

    def test_disabled_config_schedules_nothing(self):
        mgr = _manager(dreams_enabled=False)
        session = _cached_session(mgr)
        honcho_client = MagicMock()
        result = self._run_with_client(mgr, honcho_client, lambda: mgr.schedule_session_dream(session.key))
        assert result is False
        honcho_client.schedule_dream.assert_not_called()

    def test_unknown_session_schedules_nothing(self):
        mgr = _manager()
        assert mgr.schedule_session_dream("nope") is False

    def test_backend_failure_is_fail_open(self):
        mgr = _manager()
        session = _cached_session(mgr)
        honcho_client = MagicMock()
        honcho_client.schedule_dream.side_effect = RuntimeError("queue down")
        result = self._run_with_client(mgr, honcho_client, lambda: mgr.schedule_session_dream(session.key))
        assert result is False


class TestProviderSessionEndDream:
    def _provider(self, manager, **config_overrides):
        provider = HonchoMemoryProvider()
        provider._manager = manager
        provider._session_key = "cli:chat-1"
        provider._session_initialized = True
        provider._config = SimpleNamespace(save_messages=True, **config_overrides)
        return provider

    def test_session_end_flushes_then_schedules_dream(self):
        manager = MagicMock()
        provider = self._provider(manager, dreams_enabled=True)
        provider.on_session_end([])
        manager.flush_all.assert_called_once()
        manager.schedule_session_dream.assert_called_once_with("cli:chat-1")

    def test_session_end_without_dreams_schedules_nothing(self):
        manager = MagicMock()
        provider = self._provider(manager, dreams_enabled=False)
        provider.on_session_end([])
        manager.flush_all.assert_called_once()
        manager.schedule_session_dream.assert_not_called()

    def test_dream_failure_does_not_escape(self):
        manager = MagicMock()
        manager.schedule_session_dream.side_effect = RuntimeError("boom")
        provider = self._provider(manager, dreams_enabled=True)
        provider.on_session_end([])  # fail-open: no raise


# ── message metadata ────────────────────────────────────────────────────────


class TestMessageMetadata:
    def _flushable_manager(self):
        mgr = _manager()
        session = _cached_session(mgr)
        user_peer, assistant_peer = MagicMock(), MagicMock()
        peers = {session.user_peer_id: user_peer, session.assistant_peer_id: assistant_peer}
        mgr._get_or_create_peer = lambda peer_id: peers[peer_id]
        honcho_session = MagicMock()
        mgr._sessions_cache[session.honcho_session_id] = honcho_session
        return mgr, session, user_peer, assistant_peer

    def test_flush_passes_message_metadata_to_sdk(self):
        mgr, session, user_peer, assistant_peer = self._flushable_manager()
        meta = {"author_peer_id": "q", "platform": "telegram", "turn": 7}
        session.add_message("user", "hello", metadata=meta)
        session.add_message("assistant", "hi", metadata={"platform": "telegram", "turn": 7})

        assert mgr._flush_session(session) is True
        user_peer.message.assert_called_once_with("hello", metadata=meta)
        assistant_peer.message.assert_called_once_with("hi", metadata={"platform": "telegram", "turn": 7})

    def test_flush_without_metadata_sends_no_metadata_kwarg(self):
        mgr, session, user_peer, _assistant = self._flushable_manager()
        session.add_message("user", "hello")
        assert mgr._flush_session(session) is True
        user_peer.message.assert_called_once_with("hello")

    def test_sync_turn_stamps_author_platform_and_turn(self):
        provider = HonchoMemoryProvider()
        provider._config = SimpleNamespace(save_messages=True, message_max_chars=25000, a2a_sessions=True)
        provider._session_key = "cli:chat-1"
        provider._session_initialized = True
        provider._platform = "telegram"
        provider._turn_count = 3
        session = HonchoSession(key="cli:chat-1", user_peer_id="q", assistant_peer_id="asuna",
                                honcho_session_id="chat-1")
        manager = MagicMock()
        manager.get_or_create.return_value = session
        manager.resolve_author_peer_id.return_value = "q"
        provider._manager = manager

        provider.sync_turn("hi", "hello", turn_author={"id": "42", "name": "Q", "is_bot": False})
        provider._sync_thread.join(timeout=5)

        user_msg, assistant_msg = session.messages
        assert user_msg["metadata"]["author_peer_id"] == "q"
        assert user_msg["metadata"]["platform"] == "telegram"
        assert user_msg["metadata"]["turn"] == 3
        assert assistant_msg["metadata"]["platform"] == "telegram"
        assert assistant_msg["metadata"]["turn"] == 3


# ── peer card cap ───────────────────────────────────────────────────────────


class TestPeerCardCap:
    def _card_manager(self):
        mgr = _manager()
        session = _cached_session(mgr)
        assistant_peer = MagicMock()
        mgr._get_or_create_peer = MagicMock(return_value=assistant_peer)
        return mgr, session, assistant_peer

    def test_card_at_limit_passes_through(self):
        mgr, session, assistant_peer = self._card_manager()
        card = [f"fact {i}" for i in range(PEER_CARD_MAX_FACTS)]
        assistant_peer.set_card.return_value = card
        result = mgr.set_peer_card(session.key, card)
        assert result == card
        assert assistant_peer.set_card.call_args.args[0] == card

    def test_card_over_limit_is_truncated_to_oldest_facts(self):
        mgr, session, assistant_peer = self._card_manager()
        card = [f"fact {i}" for i in range(PEER_CARD_MAX_FACTS + 10)]
        assistant_peer.set_card.side_effect = lambda facts, **_: facts
        result = mgr.set_peer_card(session.key, card)
        assert len(assistant_peer.set_card.call_args.args[0]) == PEER_CARD_MAX_FACTS
        assert result == card[:PEER_CARD_MAX_FACTS]  # first facts win: callers order identity first

    def test_profile_tool_reports_truncation(self):
        provider = HonchoMemoryProvider()
        provider._session_key = "cli:chat-1"
        provider._manager = MagicMock()
        provider._manager.set_peer_card.return_value = ["f"] * PEER_CARD_MAX_FACTS
        import json

        out = json.loads(provider._tool_profile({"peer": "user",
                                                 "card": [f"fact {i}" for i in range(PEER_CARD_MAX_FACTS + 5)]}))
        assert "truncated" in out["result"]


# ── context() tuning kwargs ─────────────────────────────────────────────────


class TestContextTuning:
    def test_session_context_receives_tokens_and_search_kwargs(self):
        mgr = _manager(search_top_k=12, max_conclusions=25, search_max_distance=0.4)
        mgr._context_tokens = 1500
        session = _cached_session(mgr)
        sdk_session = MagicMock()
        sdk_session.context.return_value = SimpleNamespace(
            summary=None, peer_representation="rep", peer_card=["fact"], messages=[])
        mgr._sessions_cache[session.honcho_session_id] = sdk_session
        mgr._get_or_create_peer = MagicMock(return_value=MagicMock())

        ctx = mgr.get_session_context(session.key)
        assert ctx["representation"] == "rep"
        kwargs = sdk_session.context.call_args.kwargs
        assert kwargs["tokens"] == 1500
        assert kwargs["search_top_k"] == 12
        assert kwargs["max_conclusions"] == 25
        assert kwargs["search_max_distance"] == 0.4

    def test_unset_tuning_sends_no_search_kwargs(self):
        mgr = _manager()
        session = _cached_session(mgr)
        sdk_session = MagicMock()
        sdk_session.context.return_value = SimpleNamespace(
            summary=None, peer_representation="", peer_card=None, messages=[])
        mgr._sessions_cache[session.honcho_session_id] = sdk_session
        mgr._get_or_create_peer = MagicMock(return_value=MagicMock())

        mgr.get_session_context(session.key)
        kwargs = sdk_session.context.call_args.kwargs
        assert "search_top_k" not in kwargs
        assert "max_conclusions" not in kwargs
        assert "search_max_distance" not in kwargs

    def test_peer_context_carries_search_kwargs(self):
        mgr = _manager(search_top_k=9)
        peer = MagicMock()
        peer.context.return_value = SimpleNamespace(representation="rep", peer_card=["c"])
        mgr._get_or_create_peer = MagicMock(return_value=peer)
        mgr._fetch_peer_context("q", "what does q like", target=None)
        kwargs = peer.context.call_args.kwargs
        assert kwargs["search_query"] == "what does q like"
        assert kwargs["search_top_k"] == 9


# ── config resolution defaults ──────────────────────────────────────────────


class TestConfigDefaults:
    def _write_config(self, tmp_path, payload):
        import json

        path = tmp_path / "honcho.json"
        path.write_text(json.dumps(payload))
        return path

    def test_fresh_config_gets_bounded_context_tokens(self, tmp_path, monkeypatch):
        from plugins.memory.honcho.client import HonchoClientConfig, _DEFAULT_CONTEXT_TOKENS

        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        monkeypatch.delenv("HONCHO_BASE_URL", raising=False)
        monkeypatch.delenv("HONCHO_URL", raising=False)
        path = self._write_config(tmp_path, {"apiKey": "key-123"})
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=path)
        assert cfg.context_tokens == _DEFAULT_CONTEXT_TOKENS

    def test_existing_config_keeps_uncapped_default(self, tmp_path, monkeypatch):
        from plugins.memory.honcho.client import HonchoClientConfig

        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        path = self._write_config(tmp_path, {"enabled": True, "hosts": {"hermes": {"apiKey": "key-123"}}})
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=path)
        assert cfg.context_tokens is None  # migration guard: old configs keep legacy behavior

    def test_explicit_context_tokens_always_wins(self, tmp_path, monkeypatch):
        from plugins.memory.honcho.client import HonchoClientConfig

        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        path = self._write_config(tmp_path, {"apiKey": "key-123", "contextTokens": 800})
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=path)
        assert cfg.context_tokens == 800

    def test_summary_dream_and_search_fields_resolve(self, tmp_path, monkeypatch):
        from plugins.memory.honcho.client import HonchoClientConfig

        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        path = self._write_config(tmp_path, {
            "apiKey": "key-123", "summaryEnabled": False, "messagesPerShortSummary": 12,
            "messagesPerLongSummary": 48, "dreams": False,
            "searchTopK": 15, "searchMaxDistance": 0.35, "maxConclusions": 30,
        })
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=path)
        assert cfg.summary_enabled is False
        assert cfg.messages_per_short_summary == 12
        assert cfg.messages_per_long_summary == 48
        assert cfg.dreams_enabled is False
        assert cfg.search_top_k == 15
        assert cfg.search_max_distance == 0.35
        assert cfg.max_conclusions == 30

    def test_unset_summary_and_search_fields_stay_none(self, tmp_path, monkeypatch):
        from plugins.memory.honcho.client import HonchoClientConfig

        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        path = self._write_config(tmp_path, {"apiKey": "key-123"})
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=path)
        assert cfg.summary_enabled is None
        assert cfg.messages_per_short_summary is None
        assert cfg.messages_per_long_summary is None
        assert cfg.dreams_enabled is True
        assert cfg.search_top_k is None
        assert cfg.search_max_distance is None
        assert cfg.max_conclusions is None
