"""Regression: HERMES_E2E_READONLY=1 must block every side-effecting tool write.

Audit 4 found an e2e continuity test created an enabled Telegram-delivering cron
job. The gate previously covered only memory writes; these tests pin the same
contract across cronjob, send_message and kanban mutation tools.
"""

import json

import pytest

from tools.cronjob_tools import cronjob
from tools.kanban_tools import (
    _handle_attach,
    _handle_attachments,
    _handle_block,
    _handle_comment,
    _handle_complete,
    _handle_create,
    _handle_heartbeat,
    _handle_link,
    _handle_list,
    _handle_show,
    _handle_unblock,
)
from tools.send_message_tool import send_message_tool

READONLY_ERR = "blocked: E2E read-only mode"


@pytest.fixture
def readonly(monkeypatch):
    monkeypatch.setenv("HERMES_E2E_READONLY", "1")


@pytest.fixture
def cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class TestCronjobReadonlyGate:
    def test_create_is_blocked(self, readonly, cron_dir):
        result = json.loads(cronjob(action="create", prompt="Ping", schedule="every 1h"))
        assert result["success"] is False
        assert result["error"] == READONLY_ERR

    def test_create_writes_nothing(self, readonly, cron_dir):
        cronjob(action="create", prompt="Ping", schedule="every 1h")
        jobs_file = cron_dir / "cron" / "jobs.json"
        assert not jobs_file.exists()

    @pytest.mark.parametrize("action", ["update", "remove", "pause", "resume", "run", "resnap"])
    def test_job_actions_blocked(self, readonly, cron_dir, action):
        result = json.loads(cronjob(action=action, job_id="deadbeefcafe"))
        assert result["success"] is False
        assert result["error"] == READONLY_ERR

    def test_list_still_allowed(self, readonly, cron_dir):
        result = json.loads(cronjob(action="list"))
        assert result["success"] is True


class TestSendMessageReadonlyGate:
    def test_send_is_blocked(self, readonly):
        result = send_message_tool({"action": "send", "target": "telegram:1", "message": "hi"})
        assert result["error"] == READONLY_ERR

    def test_react_is_blocked(self, readonly):
        result = send_message_tool({"action": "react", "target": "telegram:1", "emoji": "x"})
        assert result["error"] == READONLY_ERR

    def test_list_still_allowed(self, readonly, monkeypatch):
        monkeypatch.setattr("tools.send_message_tool._handle_list", lambda: {"ok": True})
        assert send_message_tool({"action": "list"}) == {"ok": True}


class TestKanbanReadonlyGate:
    @pytest.mark.parametrize(
        "handler,args",
        [
            (_handle_create, {}),
            (_handle_comment, {}),
            (_handle_complete, {}),
            (_handle_block, {}),
            (_handle_heartbeat, {}),
            (_handle_link, {}),
            (_handle_unblock, {}),
            (_handle_attach, {}),
        ],
    )
    def test_mutations_blocked(self, readonly, handler, args):
        assert json.loads(handler(dict(args)))["error"] == READONLY_ERR

    def test_read_only_handlers_not_blocked(self, readonly):
        # Reads must fall through the gate and reach their own validation
        # (or success) rather than returning the read-only error.
        assert json.loads(_handle_show({})).get("error") != READONLY_ERR
        assert json.loads(_handle_attachments({})).get("error") != READONLY_ERR
        assert json.loads(_handle_list({})).get("error") != READONLY_ERR
