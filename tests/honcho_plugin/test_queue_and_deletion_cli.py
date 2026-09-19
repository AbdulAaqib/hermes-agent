"""CLI contracts for queue observability and deletion governance:

``hermes honcho queue`` surfaces queue_status work units; ``delete-session`` removes one
session and explains the 202 cascade; ``delete-workspace`` deletes sessions first and
maps a 409 to "not empty yet".
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import plugins.memory.honcho.cli as honcho_cli


def _status(**overrides):
    base = dict(total_work_units=10, completed_work_units=7, in_progress_work_units=1,
                pending_work_units=2, sessions=None)
    base.update(overrides)
    return SimpleNamespace(**base)


def _patch_connect(monkeypatch, client):
    hcfg = SimpleNamespace(host="hermes", workspace_id="ws-1",
                           resolve_session_name=lambda: "chat-1")
    monkeypatch.setattr(honcho_cli, "_connect", lambda host, **kw: (hcfg, client))
    monkeypatch.setattr(honcho_cli, "_host_key", lambda: "hermes")
    return hcfg


class TestQueueCommand:
    def test_scopes_to_resolved_session_and_decomposes_work_units(self, monkeypatch, capsys):
        client = MagicMock()
        client.queue_status.return_value = _status()
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_queue(SimpleNamespace(session=None, all=False))
        out = capsys.readouterr().out
        client.queue_status.assert_called_once_with(session="chat-1")
        assert "10 total" in out
        assert "Completed:" in out and "7" in out
        assert "Pending:" in out and "2" in out

    def test_all_flag_queries_workspace_wide(self, monkeypatch, capsys):
        client = MagicMock()
        client.queue_status.return_value = _status(pending_work_units=0, in_progress_work_units=0)
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_queue(SimpleNamespace(session=None, all=True))
        out = capsys.readouterr().out
        client.queue_status.assert_called_once_with()  # unscoped = whole workspace
        assert "drained" in out

    def test_per_session_rows_render(self, monkeypatch, capsys):
        client = MagicMock()
        client.queue_status.return_value = _status(sessions={
            "chat-1": SimpleNamespace(pending_work_units=2, in_progress_work_units=0, completed_work_units=5),
        })
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_queue(SimpleNamespace(session=None, all=False))
        out = capsys.readouterr().out
        assert "chat-1" in out

    def test_backend_failure_is_reported_not_raised(self, monkeypatch, capsys):
        client = MagicMock()
        client.queue_status.side_effect = RuntimeError("down")
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_queue(SimpleNamespace(session=None, all=False))
        assert "unavailable" in capsys.readouterr().out


class TestDeleteSessionCommand:
    def test_yes_flag_deletes_without_prompt(self, monkeypatch, capsys):
        client = MagicMock()
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_delete_session(SimpleNamespace(name="chat-9", yes=True))
        out = capsys.readouterr().out
        client.session.assert_called_once_with("chat-9")
        client.session("chat-9").delete.assert_called_once()
        assert "202" in out  # async cascade semantics are explained
        assert "delete_id" in out  # derived conclusions pointed at the existing path

    def test_defaults_to_resolved_session(self, monkeypatch, capsys):
        client = MagicMock()
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_delete_session(SimpleNamespace(name=None, yes=True))
        client.session.assert_called_once_with("chat-1")

    def test_declined_confirmation_deletes_nothing(self, monkeypatch, capsys):
        client = MagicMock()
        _patch_connect(monkeypatch, client)
        monkeypatch.setattr(honcho_cli, "_prompt", lambda *a, **kw: "n")

        honcho_cli.cmd_delete_session(SimpleNamespace(name="chat-9", yes=False))
        client.session.assert_not_called()

    def test_missing_session_reports_not_found(self, monkeypatch, capsys):
        client = MagicMock()
        err = RuntimeError("not found")
        err.status_code = 404
        client.session.return_value.delete.side_effect = err
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_delete_session(SimpleNamespace(name="ghost", yes=True))
        assert "not found" in capsys.readouterr().out


class TestDeleteWorkspaceCommand:
    def test_deletes_sessions_then_workspace(self, monkeypatch, capsys):
        client = MagicMock()
        client.sessions.return_value = SimpleNamespace(
            items=[SimpleNamespace(id="s1"), SimpleNamespace(id="s2")])
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_delete_workspace(SimpleNamespace(yes=True))
        out = capsys.readouterr().out
        assert client.session.call_count == 2
        client.delete_workspace.assert_called_once_with("ws-1")
        assert "Deleted 2 session(s)" in out

    def test_conflict_maps_to_not_empty_yet(self, monkeypatch, capsys):
        client = MagicMock()
        client.sessions.return_value = SimpleNamespace(items=[])
        err = RuntimeError("conflict")
        err.status_code = 409
        client.delete_workspace.side_effect = err
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_delete_workspace(SimpleNamespace(yes=True))
        assert "409" in capsys.readouterr().out

    def test_failed_session_delete_aborts_workspace_delete(self, monkeypatch, capsys):
        client = MagicMock()
        client.sessions.return_value = SimpleNamespace(items=[SimpleNamespace(id="s1")])
        client.session.return_value.delete.side_effect = RuntimeError("nope")
        _patch_connect(monkeypatch, client)

        honcho_cli.cmd_delete_workspace(SimpleNamespace(yes=True))
        client.delete_workspace.assert_not_called()
