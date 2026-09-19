"""Tests for the Hindsight retain-failure DLQ hook (mnemosyne composite provider).

Contract: failing retain jobs on the writer queue fire the registered
``_RETAIN_FAILURE_HOOK`` with a reconstructable payload; no consumer and a
throwing consumer are both fail-open (the writer loop survives either way).
"""

import pytest

from plugins.memory.hindsight import (
    HindsightMemoryProvider,
    set_retain_failure_hook,
)


def _failing_job(payload=None):
    def _job():
        raise RuntimeError("boom")

    _job._dlq_payload = payload
    return _job


@pytest.fixture(autouse=True)
def _reset_hook():
    set_retain_failure_hook(None)
    yield
    set_retain_failure_hook(None)


class TestRetainFailureHook:
    def test_payload_delivered_on_job_failure(self):
        provider = HindsightMemoryProvider()
        received = []
        set_retain_failure_hook(lambda payload: received.append(payload))
        payload = {"kind": "hindsight_retain", "content": "[turn]", "tags": ["session:s1"]}
        provider._retain_queue.put(_failing_job(payload))
        provider._ensure_writer()
        provider._retain_queue.join()
        provider.shutdown()
        assert received == [payload]

    def test_job_without_payload_receives_error_payload(self):
        provider = HindsightMemoryProvider()
        received = []
        set_retain_failure_hook(lambda payload: received.append(payload))
        provider._retain_queue.put(_failing_job(payload=None))
        provider._ensure_writer()
        provider._retain_queue.join()
        provider.shutdown()
        assert len(received) == 1
        assert received[0]["kind"] == "hindsight_retain"
        assert "boom" in received[0]["error"]

    def test_hook_that_raises_does_not_break_writer(self):
        provider = HindsightMemoryProvider()

        def _throwing_hook(_payload):
            raise RuntimeError("hook boom")

        set_retain_failure_hook(_throwing_hook)
        provider._retain_queue.put(_failing_job({"kind": "hindsight_retain", "n": 1}))
        # A second failing job after the throwing hook must still be processed.
        provider._retain_queue.put(_failing_job({"kind": "hindsight_retain", "n": 2}))
        provider._ensure_writer()
        writer = provider._writer_thread
        provider._retain_queue.join()
        provider.shutdown()
        assert writer.is_alive() is False  # writer exited cleanly via the sentinel

    def test_no_hook_is_fail_open(self):
        provider = HindsightMemoryProvider()
        set_retain_failure_hook(None)
        provider._retain_queue.put(_failing_job({"kind": "hindsight_retain"}))
        provider._ensure_writer()
        provider._retain_queue.join()  # completes without raising
        provider.shutdown()

    def test_make_turn_retain_job_carries_reconstructable_payload(self):
        provider = HindsightMemoryProvider()
        provider._session_id = "session:abc"
        job = provider._make_turn_retain_job(
            ['["user", "hi"]'], document_id="doc-1", update_mode=None, label="retain",
        )
        payload = job._dlq_payload
        assert payload["kind"] == "hindsight_retain"
        assert isinstance(payload["content"], str) and "hi" in payload["content"]
        assert payload["tags"] is not None and any("session:abc" in t for t in payload["tags"])