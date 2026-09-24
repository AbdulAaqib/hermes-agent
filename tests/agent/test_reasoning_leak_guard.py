"""Regression guard for the reasoning-leak defect (2026-09-24).

Some providers (DeepSeek via OpenRouter) returned chain-of-thought in the
``content`` channel instead of a separate reasoning field, so internal
narration was delivered verbatim to the user — including one turn whose whole
message was reasoning with no reply. ``_subtract_reasoning_from_content`` is a
deterministic guard: it strips an exact captured-reasoning prefix from content
and never heuristic-trims a genuine reply.
"""
from __future__ import annotations

from agent.chat_completion_helpers import _subtract_reasoning_from_content


def test_strips_exact_leading_reasoning_block():
    content = "The user wants a short reply.\n\nhey babe, i'm here"
    assert _subtract_reasoning_from_content(content, "The user wants a short reply.") == "hey babe, i'm here"


def test_pure_reasoning_content_becomes_empty():
    assert _subtract_reasoning_from_content("Let me think about this.", "Let me think about this.") == ""


def test_leaves_genuine_reply_untouched():
    # A reply that does not start with the captured reasoning is never trimmed.
    assert _subtract_reasoning_from_content("hey babe", "The user wants X.") == "hey babe"


def test_no_reasoning_or_no_content_is_identity():
    assert _subtract_reasoning_from_content("reply only", None) == "reply only"
    assert _subtract_reasoning_from_content("", "reasoning") == ""


def test_trailing_newline_separator_variant():
    assert _subtract_reasoning_from_content("plan\nreply", "plan\n") == "reply"
