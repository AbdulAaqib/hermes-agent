"""Issue #121519 premise check: does in-place compaction re-INSERT already-persisted
messages as duplicate ACTIVE rows?

Drives a real in-place compaction against a real SessionDB, then runs the reporter's
exact duplicate-detection SQL twice: unfiltered (as filed) and with ``active = 1``.
"""

import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

DUPLICATE_SQL = """
select session_id, role, content, round(timestamp,3) ts, count(*) n
from messages
where content is not null and length(content) >= 15
{extra}
group by session_id, role, content, round(timestamp,3)
having count(*) > 1
"""


def _make_agent(session_db, session_id):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent
        return AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1",
            model="test/model", quiet_mode=True, session_db=session_db,
            session_id=session_id, skip_context_files=True, skip_memory=True,
        )


def _dupes(db_path, extra=""):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(DUPLICATE_SQL.format(extra=extra)).fetchall()
    finally:
        conn.close()


def test_issue_121519_duplicates_are_archived_originals_not_live_rows():
    from agent.conversation_compression import conversation_history_after_compression
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "state.db"
        db = SessionDB(db_path=db_path)
        sid = "issue-121519-session"
        agent = _make_agent(db, sid)
        agent._ensure_db_session()

        # A persisted transcript long enough that the reporter's length>=15 filter applies.
        history = []
        for i in range(6):
            history.append({"role": "user", "content": f"user question number {i} with padding"})
            history.append({"role": "assistant", "content": f"assistant answer number {i} with padding"})
        agent._flush_messages_to_session_db(history, [])
        assert len(db.get_messages(sid)) == 12

        # In-place compaction: summary + verbatim carried tail (the retained tail the
        # issue says comes back unstamped and gets re-INSERTed).
        tail = history[-4:]
        compacted = [{"role": "assistant", "content": "[CONTEXT COMPACTION] summary of earlier turns"}] + tail
        db.archive_and_compact(sid, compacted, tail_count=len(tail))
        setattr(agent, "_last_compaction_in_place", True)
        agent._last_flushed_db_idx = 0

        # The next persist walk — the step the issue claims re-INSERTs the tail.
        post = conversation_history_after_compression(agent, compacted)
        messages = compacted + [{"role": "assistant", "content": "brand new answer after compaction"}]
        agent._flush_messages_to_session_db(messages, post)

        unfiltered = _dupes(db_path)
        live = _dupes(db_path, "and active = 1")

        # The reporter's unfiltered query DOES report duplicates: the soft-archived
        # originals (active=0) sit beside their carried-forward active copies. That is
        # archive_and_compact's documented non-destructive design, not a re-INSERT.
        assert unfiltered, "expected the unfiltered query to surface archived/active pairs"

        # The actual claim under test: no duplicate LIVE rows.
        assert live == [], f"duplicate active rows after compaction: {[dict(r) for r in live]}"

        contents = [r["content"] for r in db.get_messages(sid)]
        assert len(contents) == len(set(contents)), contents
