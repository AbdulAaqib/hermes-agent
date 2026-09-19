from gateway.response_filters import (
    is_autonomous_silence_response,
    is_intentional_silence_agent_result,
    is_intentional_silence_response,
    strip_chain_of_thought_preamble,
)


def test_exact_silence_tokens_are_intentional_silence():
    for token in ("[SILENT]", " SILENT ", "NO_REPLY", "no reply"):
        assert is_intentional_silence_response(token)


def test_autonomous_silence_accepts_marker_with_own_line_note():
    """The loose rule for cron/webhook lanes: marker + explanation suppresses."""
    assert is_autonomous_silence_response("[SILENT]")
    assert is_autonomous_silence_response("[SILENT]\n\nNothing new this tick.")
    assert is_autonomous_silence_response("2 deals filtered\n\n[SILENT]")
    assert is_autonomous_silence_response("no_reply\nduplicate inbound, already handled")
    assert is_autonomous_silence_response("[SILENT] No changes detected")


def test_autonomous_silence_tolerates_near_miss_marker_typos():
    """Cron lane: a typo'd bracketed sentinel is still silence.

    Regression for the 2026-09-18 incident where a cron job's final output was
    literally ``[SLIENT]`` and would have been delivered verbatim to Telegram.
    """
    assert is_autonomous_silence_response("[SLIENT]")
    assert is_autonomous_silence_response("[SILNET]")
    assert is_autonomous_silence_response("[SILEN]")
    assert is_autonomous_silence_response("[SILENTT]")
    assert is_autonomous_silence_response("[silent]")
    assert is_autonomous_silence_response("some note\n\n[SLIENT]")


def test_autonomous_silence_near_miss_does_not_swallow_legit_text():
    """Near-miss tolerance must not eat real messages or unrelated brackets."""
    assert not is_autonomous_silence_response("[MUTED]")
    assert not is_autonomous_silence_response("Silent night, thinking of you")
    assert not is_autonomous_silence_response("[SILENT-ish] almost")
    assert not is_autonomous_silence_response("[SOLENT GREEN]")
    assert not is_autonomous_silence_response("he went quiet\n\nit was silent in the room")


def test_interactive_lane_stays_strict_on_typos():
    """The exact interactive rule does not gain typo tolerance."""
    assert not is_intentional_silence_response("[SLIENT]")
    assert not is_intentional_silence_response("[SILENTT]")


def test_chain_of_thought_preamble_stripped():
    """Reasoning-off DeepSeek narrates its CoT into content; only the reply after
    the last stage-direction line may reach the user."""
    leaked = (
        "I have solid live data now. The Elizabeth line is on **Good Service** as of 18:07 "
        "today. The disruption relates to an earlier incident, not an ongoing one.\n\n"
        "So the answer to Q: **No, the line is fine.** His trains run on time.\n\n"
        "Let me give him the weather-tinged, warm reply per my voice rules.\n\n"
        "morning/evening check done. Answer: no delays, trains not cancelled. Reply in voice.\n\n"
        "good news first, then the warmth. Keep it short, raw word in the surface.\n\n"
        "the elizabeth line's running good service right now, my love. not delayed, "
        "nothing cancelled.\n\nso get yourself on it and come home. i've got the kettle on."
    )
    stripped = strip_chain_of_thought_preamble(leaked)
    assert "Let me give him" not in stripped
    assert "Reply in voice" not in stripped
    assert "good news first, then the warmth" not in stripped
    assert stripped.startswith("the elizabeth line's running good service right now")


def test_chain_of_thought_scrub_keeps_normal_replies():
    """The heuristic is narrow: persona prose mentioning a marker must survive."""
    normal = (
        "The elizabeth line is fine, my love. not delayed, nothing cancelled. "
        "so get yourself on it and come home. i've got the kettle on. "
        "keep it short - i missed you all day."
    )
    assert strip_chain_of_thought_preamble(normal) == normal


def test_chain_of_thought_scrub_ignores_short_and_absent_markers():
    assert strip_chain_of_thought_preamble("Reply in voice") == "Reply in voice"
    assert strip_chain_of_thought_preamble("") == ""
    assert strip_chain_of_thought_preamble(None) is None


