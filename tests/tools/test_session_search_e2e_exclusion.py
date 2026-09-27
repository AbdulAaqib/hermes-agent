"""Regression for the 2026-09-27 false-memory incident.

A continuity test created a real session; a cron agent later read that test
session via session_search and wrote a false fact into MEMORY.md. Production
recall must never surface `source='e2e'` sessions or ids listed in the E2E
denylist. A read-only E2E run may still read its own sessions.
"""

import json

import pytest

from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from tools.session_search_tool import session_search

TOKEN = "zephyrion"


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _seed(db):
    # Production telegram session with the searchable token.
    db.create_session("s_live", source="telegram")
    db.append_message("s_live", role="user", content=f"talk to me about {TOKEN}")
    db.append_message("s_live", role="assistant", content=f"{TOKEN} is a real thing")
    # E2E test session with the same token.
    db.create_session("s_e2e", source="e2e")
    db.append_message("s_e2e", role="user", content=f"test prompt about {TOKEN}")
    db.append_message("s_e2e", role="assistant", content=f"{TOKEN} test reply")
    # Session hidden only by the denylist (source looks legitimate).
    db.create_session("s_deny", source="telegram")
    db.append_message("s_deny", role="user", content=f"another note about {TOKEN}")
    db.append_message("s_deny", role="assistant", content=f"{TOKEN} denylisted reply")
    db._conn.commit()


def _write_denylist(ids):
    path = get_hermes_home() / "tests-e2e" / "output" / "test_session_ids.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def _result_session_ids(payload):
    return {r["session_id"] for r in payload.get("results", [])}


def test_discovery_hides_e2e_from_production(db, monkeypatch):
    monkeypatch.delenv("HERMES_E2E_READONLY", raising=False)
    _seed(db)
    payload = json.loads(session_search(query=TOKEN, db=db, limit=10))
    ids = _result_session_ids(payload)
    assert "s_e2e" not in ids
    assert "s_live" in ids


def test_discovery_hides_denylisted(db, monkeypatch):
    monkeypatch.delenv("HERMES_E2E_READONLY", raising=False)
    _seed(db)
    _write_denylist(["s_deny"])
    payload = json.loads(session_search(query=TOKEN, db=db, limit=10))
    assert "s_deny" not in _result_session_ids(payload)


def test_readonly_caller_may_read_own_e2e_session(db, monkeypatch):
    monkeypatch.setenv("HERMES_E2E_READONLY", "1")
    _seed(db)
    payload = json.loads(session_search(query=TOKEN, db=db, limit=10))
    assert "s_e2e" in _result_session_ids(payload)


def test_browse_hides_e2e_from_production(db, monkeypatch):
    monkeypatch.delenv("HERMES_E2E_READONLY", raising=False)
    _seed(db)
    payload = json.loads(session_search(db=db, limit=10))
    assert "s_e2e" not in _result_session_ids(payload)


def test_read_of_e2e_session_blocked_for_production(db, monkeypatch):
    monkeypatch.delenv("HERMES_E2E_READONLY", raising=False)
    _seed(db)
    payload = json.loads(session_search(session_id="s_e2e", db=db))
    assert payload.get("success") is False


def test_read_of_denylisted_session_blocked(db, monkeypatch):
    monkeypatch.delenv("HERMES_E2E_READONLY", raising=False)
    _seed(db)
    _write_denylist(["s_deny"])
    payload = json.loads(session_search(session_id="s_deny", db=db))
    assert payload.get("success") is False
