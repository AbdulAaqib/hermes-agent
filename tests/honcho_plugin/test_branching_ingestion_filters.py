"""Behavior contracts for session branching (clone), document ingestion (upload_file),
workspace ID validation, and conclusion level filtering."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import plugins.memory.honcho.cli as honcho_cli
from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig, validate_workspace_id
from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager


# ── workspace ID validation (1.1) ───────────────────────────────────────────


class TestWorkspaceIdValidation:
    @pytest.mark.parametrize("wid", ["hermes", "ws-1", "Work_Space9", "a" * 506])
    def test_valid_ids_pass(self, wid):
        assert validate_workspace_id(wid) == wid

    @pytest.mark.parametrize("wid", ["", "has space", "dots.ws", "slash/ws", "a" * 507, "ümlaut"])
    def test_invalid_ids_raise_with_clear_message(self, wid):
        with pytest.raises(ValueError, match="Invalid Honcho workspace ID"):
            validate_workspace_id(wid)

    def test_config_resolution_rejects_bad_workspace(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        path = tmp_path / "honcho.json"
        path.write_text(json.dumps({"apiKey": "k", "workspace": "bad workspace!"}))
        with pytest.raises(ValueError, match="workspace"):
            HonchoClientConfig.from_global_config(host="hermes", config_path=path)

    def test_config_resolution_accepts_good_workspace(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HONCHO_API_KEY", raising=False)
        path = tmp_path / "honcho.json"
        path.write_text(json.dumps({"apiKey": "k", "workspace": "my_ws-2"}))
        cfg = HonchoClientConfig.from_global_config(host="hermes", config_path=path)
        assert cfg.workspace_id == "my_ws-2"


# ── clone-session (1.3) ─────────────────────────────────────────────────────


def _patch_connect(monkeypatch, client, peer_name="q"):
    hcfg = SimpleNamespace(host="hermes", workspace_id="ws-1", peer_name=peer_name,
                           resolve_session_name=lambda: "chat-1")
    monkeypatch.setattr(honcho_cli, "_connect", lambda host, **kw: (hcfg, client))
    monkeypatch.setattr(honcho_cli, "_host_key", lambda: "hermes")
    return hcfg


class TestCloneSessionCommand:
    def test_clones_and_reports_new_id(self, monkeypatch, capsys):
        client = MagicMock()
        client.session.return_value.clone.return_value = SimpleNamespace(id="chat-1-fork")
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_clone_session(SimpleNamespace(name="chat-1", up_to_message=None))
        out = capsys.readouterr().out
        client.session.assert_called_once_with("chat-1")
        client.session("chat-1").clone.assert_called_once_with(message_id=None)
        assert "chat-1-fork" in out

    def test_up_to_message_is_forwarded(self, monkeypatch, capsys):
        client = MagicMock()
        client.session.return_value.clone.return_value = SimpleNamespace(id="fork")
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_clone_session(SimpleNamespace(name=None, up_to_message="msg_42"))
        client.session("chat-1").clone.assert_called_once_with(message_id="msg_42")

    def test_missing_session_reports_not_found(self, monkeypatch, capsys):
        client = MagicMock()
        err = RuntimeError("nope")
        err.status_code = 404
        client.session.return_value.clone.side_effect = err
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_clone_session(SimpleNamespace(name="ghost", up_to_message=None))
        assert "not found" in capsys.readouterr().out


# ── upload (10.1) ───────────────────────────────────────────────────────────


class TestUploadCommand:
    def test_uploads_file_attributed_to_peer_with_provenance(self, monkeypatch, capsys, tmp_path):
        doc = tmp_path / "notes.txt"
        doc.write_text("hello honcho")
        client = MagicMock()
        client.session.return_value.upload_file.return_value = [SimpleNamespace(id="m1")]
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_upload(SimpleNamespace(file=str(doc), peer=None, session=None))
        out = capsys.readouterr().out
        client.session.assert_called_once_with("chat-1")  # resolved default session
        _args, kwargs = client.session("chat-1").upload_file.call_args
        assert kwargs["peer"] == "q"  # default: configured peerName
        assert kwargs["file"][0] == "notes.txt"
        assert kwargs["file"][1] == b"hello honcho"
        assert kwargs["metadata"]["original_file"] == "notes.txt"
        assert "notes.txt" in out

    def test_explicit_peer_and_session_win(self, monkeypatch, capsys, tmp_path):
        doc = tmp_path / "doc.pdf"
        doc.write_bytes(b"%PDF-1.4 fake")
        client = MagicMock()
        client.session.return_value.upload_file.return_value = []
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_upload(SimpleNamespace(file=str(doc), peer="alice", session="s-9"))
        client.session.assert_called_once_with("s-9")
        _args, kwargs = client.session("s-9").upload_file.call_args
        assert kwargs["peer"] == "alice"
        assert kwargs["file"][2] == "application/pdf"

    def test_missing_peer_errors_without_uploading(self, monkeypatch, capsys, tmp_path):
        doc = tmp_path / "x.txt"
        doc.write_text("x")
        client = MagicMock()
        _patch_connect(monkeypatch, client, peer_name=None)

        honcho_cli.cmd_upload(SimpleNamespace(file=str(doc), peer=None, session=None))
        assert "--peer" in capsys.readouterr().out
        client.session.assert_not_called()

    def test_missing_file_errors(self, monkeypatch, capsys, tmp_path):
        client = MagicMock()
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_upload(SimpleNamespace(file=str(tmp_path / "nope.txt"), peer="q", session=None))
        assert "not found" in capsys.readouterr().out.lower()
        client.session.assert_not_called()


# ── conclusion level filtering (11.2) ───────────────────────────────────────


def _manager() -> HonchoSessionManager:
    mgr = HonchoSessionManager(honcho=MagicMock(), config=None)
    mgr._authed_call = lambda label, op: op()
    session = HonchoSession(key="cli:chat-1", user_peer_id="q", assistant_peer_id="asuna",
                            honcho_session_id="chat-1")
    mgr._cache[session.key] = session
    return mgr, session


class TestConclusionLevelFilter:
    def test_list_passes_level_filter_to_sdk(self):
        mgr, session = _manager()
        scope = MagicMock()
        scope.list.return_value = SimpleNamespace(items=[SimpleNamespace(id="c1", content="fact", level="explicit")])
        mgr._conclusions_scope = lambda s, target: scope

        result = mgr.list_conclusions(session.key, level="explicit")
        scope.list.assert_called_once_with(size=20, filters={"level": "explicit"})
        assert result == [{"id": "c1", "content": "fact", "level": "explicit"}]

    def test_query_passes_level_filter_to_sdk(self):
        mgr, session = _manager()
        scope = MagicMock()
        scope.query.return_value = [SimpleNamespace(id="c2", content="derived", level="deductive")]
        mgr._conclusions_scope = lambda s, target: scope

        result = mgr.list_conclusions(session.key, query="projects", level="deductive")
        scope.query.assert_called_once_with("projects", top_k=20, filters={"level": "deductive"})
        assert result[0]["level"] == "deductive"

    def test_no_level_sends_no_filter(self):
        mgr, session = _manager()
        scope = MagicMock()
        scope.list.return_value = SimpleNamespace(items=[])
        mgr._conclusions_scope = lambda s, target: scope

        mgr.list_conclusions(session.key)
        scope.list.assert_called_once_with(size=20, filters=None)

    def test_tool_accepts_level_only_with_list(self):
        provider = HonchoMemoryProvider()
        provider._session_key = "cli:chat-1"
        provider._manager = MagicMock()
        provider._manager.list_conclusions.return_value = []

        out = json.loads(provider._tool_conclude({"list": True, "level": "explicit"}))
        assert out["conclusions"] == []
        provider._manager.list_conclusions.assert_called_once_with(
            "cli:chat-1", query=None, peer="user", level="explicit")

        assert "level is only valid when list is true" in provider._tool_conclude(
            {"conclusion": "some fact", "level": "explicit"})
        assert "Invalid level" in provider._tool_conclude({"list": True, "level": "bogus"})
