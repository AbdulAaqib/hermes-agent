"""CLI contracts for scope management (9.1-9.3) and workspace-level chat (7.1)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import plugins.memory.honcho.cli as honcho_cli


def _patch_connect(monkeypatch, client):
    hcfg = SimpleNamespace(host="hermes", workspace_id="ws-1", peer_name="q",
                           resolve_session_name=lambda: "chat-1")
    monkeypatch.setattr(honcho_cli, "_connect", lambda host, **kw: (hcfg, client))
    monkeypatch.setattr(honcho_cli, "_host_key", lambda: "hermes")
    return hcfg


class TestScopeCommand:
    def test_list_empty_guides_to_create(self, monkeypatch, capsys):
        client = MagicMock()
        client.scopes.return_value = SimpleNamespace(items=[])
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_scope(SimpleNamespace(scope_action="list", name=None, session=None, wait=False))
        assert "scope create" in capsys.readouterr().out

    def test_list_renders_scope_ids(self, monkeypatch, capsys):
        client = MagicMock()
        client.scopes.return_value = SimpleNamespace(items=[SimpleNamespace(id="work"), SimpleNamespace(id="play")])
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_scope(SimpleNamespace(scope_action="list", name=None, session=None, wait=False))
        out = capsys.readouterr().out
        assert "work" in out and "play" in out

    def test_create_uses_get_or_create(self, monkeypatch, capsys):
        client = MagicMock()
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_scope(SimpleNamespace(scope_action="create", name="work", session=None, wait=False))
        client.scope.assert_called_once_with("work")
        assert "work" in capsys.readouterr().out

    def test_add_session_defaults_to_resolved_session(self, monkeypatch, capsys):
        client = MagicMock()
        scope = client.get_scope.return_value
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_scope(SimpleNamespace(scope_action="add-session", name="work", session=None, wait=False))
        client.get_scope.assert_called_once_with("work")
        scope.add_sessions.assert_called_once_with(["chat-1"])

    def test_add_session_reports_missing_scope(self, monkeypatch, capsys):
        client = MagicMock()
        err = RuntimeError("nope")
        err.status_code = 404
        client.get_scope.side_effect = err
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_scope(SimpleNamespace(scope_action="add-session", name="ghost", session="s-1", wait=False))
        assert "not found" in capsys.readouterr().out

    def test_status_reports_completion_and_docs_copied(self, monkeypatch, capsys):
        client = MagicMock()
        scope = client.get_scope.return_value
        scope.status.return_value = {
            "chat-1": SimpleNamespace(state="completed", docs_copied=12),
        }
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_scope(SimpleNamespace(scope_action="status", name="work", session=None, wait=False))
        out = capsys.readouterr().out
        assert "completed" in out and "12" in out and "current" in out

    def test_status_pending_stays_pending_without_wait(self, monkeypatch, capsys):
        client = MagicMock()
        scope = client.get_scope.return_value
        scope.status.return_value = {"chat-1": SimpleNamespace(state="pending", docs_copied=None)}
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_scope(SimpleNamespace(scope_action="status", name="work", session=None, wait=False))
        assert "Still backfilling" in capsys.readouterr().out

    def test_unknown_action_is_rejected(self, monkeypatch, capsys):
        client = MagicMock()
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_scope(SimpleNamespace(scope_action="explode", name="work", session=None, wait=False))
        assert "Unknown scope action" in capsys.readouterr().out


class TestAskCommand:
    def test_query_goes_to_workspace_chat(self, monkeypatch, capsys):
        client = MagicMock()
        client.chat.return_value = "workspace answer"
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_ask(SimpleNamespace(query="themes?", level="medium", scope="work", session=None))
        client.chat.assert_called_once_with("themes?", reasoning_level="medium", scope="work", session=None)
        assert "workspace answer" in capsys.readouterr().out

    def test_scope_and_session_are_mutually_exclusive(self, monkeypatch, capsys):
        client = MagicMock()
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_ask(SimpleNamespace(query="q", level=None, scope="work", session="s-1"))
        assert "mutually exclusive" in capsys.readouterr().out
        client.chat.assert_not_called()

    def test_empty_query_is_rejected(self, monkeypatch, capsys):
        client = MagicMock()
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_ask(SimpleNamespace(query="  ", level=None, scope=None, session=None))
        client.chat.assert_not_called()
