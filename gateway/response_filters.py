"""Gateway response filtering helpers.

These decide whether a completed agent turn should be delivered to the chat,
not what should be persisted in conversation history.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Exact whole-response markers meaning "the agent intentionally chose not to
# reply". Keep small and explicit; arbitrary empty output remains an
# error/empty-response path, not silence.
LIVE_GATEWAY_SILENT_MARKERS = frozenset({"[SILENT]", "SILENT", "NO_REPLY", "NO REPLY"})

# Longer than any marker could plausibly be, even with stray punctuation.
_MARKER_LENGTH_CAP = 64


def _canonical_silence_candidate(text: str) -> str:
    return " ".join(text.strip().upper().split())


def _is_edge_punctuation(ch: str) -> bool:
    # Square brackets stay structural so malformed ``[SILENT`` cannot become ``SILENT``.
    return ch not in "[]" and unicodedata.category(ch).startswith("P")


def _strip_edge_silence_punctuation(text: str) -> str:
    """Strip stray edge punctuation (``.NO_REPLY``, ``*NO_REPLY*``) without erasing marker structure."""
    start, end = 0, len(text)
    while start < end and _is_edge_punctuation(text[start]):
        start += 1
    while end > start and _is_edge_punctuation(text[end - 1]):
        end -= 1
    return text[start:end].strip()


_BRACKETED_SENTINEL_RE = re.compile(r"^\[([A-Z_ ]{3,14})\]$")


def _levenshtein_within(a: str, b: str, limit: int) -> bool:
    """True when edit distance(a, b) <= limit, with early exits."""
    if abs(len(a) - len(b)) > limit:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return False
        prev = cur
    return prev[-1] <= limit


def _is_near_silent_bracket(canonical_candidate: str) -> bool:
    """True for a whole-response bracketed token within edit distance 2 of [SILENT].

    Only canonical (uppercased, whitespace-collapsed) whole-candidate forms are
    accepted, so prose mentioning brackets and unrelated markers like [MUTED]
    never match.
    """
    m = _BRACKETED_SENTINEL_RE.match(canonical_candidate)
    return bool(m) and _levenshtein_within(m.group(1).replace(" ", ""), "SILENT", 2)


def _canonical_silence_candidates(text: Any) -> tuple[str, ...]:
    """Canonical forms of a short marker-sized response; ``()`` when not a candidate at all."""
    stripped = text.strip() if isinstance(text, str) else ""
    if not 0 < len(stripped) <= _MARKER_LENGTH_CAP:
        return ()
    depunctuated = _strip_edge_silence_punctuation(stripped)
    forms = (stripped,) if depunctuated == stripped else (stripped, depunctuated)
    return tuple(_canonical_silence_candidate(f) for f in forms)


# Deterministic safety net for model-side self-narration that leaks into the delivered
# content. With reasoning disabled on the wire, DeepSeek-family models occasionally emit
# their chain-of-thought as plain content (non-deterministic), so the visible text becomes
# "CoT preamble" + "actual reply". The primary fix is routing thinking into the reasoning
# channel (agent.reasoning_overrides); this scrubber backstops the residual leak at delivery
# by cutting at the last self-directed stage-direction line. DeepSeek narrates instructions
# to itself ("Reply in voice", "per my voice rules") right before producing the persona
# answer, so the final reply is everything after the LAST such paragraph.
_COT_STAGE_DIRECTION_MARKERS = (
    "reply in voice", "per my voice rules", "my voice rules", "morning/evening check done",
    "good news first", "then the warmth", "keep it short", "raw word in the surface",
    "so the answer to", "let me give him", "let me give her", "answer in voice",
)
_COT_SCRUB_MIN_CHARS = 400


def strip_chain_of_thought_preamble(text: Any) -> str:
    """Remove a leaked chain-of-thought preamble from a final response.

    Returns the text unchanged when there is no evidence of model self-narration (markers,
    length, trailing content). The heuristic is deliberately narrow so legitimate persona
    replies are never truncated: it only fires when the response is long, at least one
    stage-direction marker paragraph is present, and a non-empty reply remains after the
    last marker paragraph.
    """
    if not isinstance(text, str) or not text.strip() or len(text) < _COT_SCRUB_MIN_CHARS:
        return text
    paragraphs = [p.strip() for p in text.split("\n\n")]
    last_marker_idx = -1
    for i, para in enumerate(paragraphs):
        lower = para.lower()
        if any(m in lower for m in _COT_STAGE_DIRECTION_MARKERS):
            last_marker_idx = i
    if last_marker_idx == -1:
        return text
    tail = "".join(p + "\n\n" for p in paragraphs[last_marker_idx + 1:]).strip()
    if not tail:
        return text
    return tail


def is_intentional_silence_response(response: Any) -> bool:
    """True only when ``response`` is exactly a silence marker.

    Prose that merely mentions ``NO_REPLY`` must be delivered normally. A blank
    response is not silence either — that is the empty-response failure path.
    """
    return any(c in LIVE_GATEWAY_SILENT_MARKERS for c in _canonical_silence_candidates(response))


def is_autonomous_silence_response(response: Any) -> bool:
    """Loose silence matcher for autonomous lanes (cron, webhook).

    Models reliably bracket ``[SILENT]`` with a short note, so unlike the
    interactive EXACT rule this also suppresses when a marker sits on its own
    first/last line or the bracketed sentinel opens the response (``[SILENT] No
    changes detected``).  A token buried mid-sentence is still delivered.  A
    bracketed near-miss typo of the sentinel (``[SLIENT]``) still counts as
    silence; anything else bracketed does not.
    Shares :data:`LIVE_GATEWAY_SILENT_MARKERS` so the two sets cannot drift.
    """
    stripped = response.strip() if isinstance(response, str) else ""
    if not stripped:
        return False
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    # Bracketed form only for the prefix rule, so a bare "Silent retry succeeded" is NOT swallowed.
    candidates = (_canonical_silence_candidate(c) for c in (stripped, lines[0], lines[-1]))
    candidates = tuple(candidates)
    return (
        stripped.upper().startswith("[SILENT]")
        or any(c in LIVE_GATEWAY_SILENT_MARKERS for c in candidates)
        or any(_is_near_silent_bracket(c) for c in candidates)
    )


def is_intentional_silence_agent_result(agent_result: dict | None, response: Any) -> bool:
    """Silence markers suppress delivery only for successful agent turns."""
    return isinstance(agent_result, dict) and not agent_result.get("failed") and is_intentional_silence_response(response)


def is_partial_silence_marker(text: Any) -> bool:
    """True while streamed ``text`` could still resolve to a silence marker.

    A buffer whose canonical form is a non-empty *prefix* of a marker (``"NO"`` on
    the way to ``"NO_REPLY"``, or an exact marker not yet terminated by stream-end)
    is held back so a raw marker is never shown and then retracted.  Divergence
    from every marker, or exceeding the cap, resumes normal streaming.
    """
    return any(
        c and any(marker.startswith(c) for marker in LIVE_GATEWAY_SILENT_MARKERS)
        for c in _canonical_silence_candidates(text)
    )


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

SILENT_REPLY_TOKEN = "NO_REPLY"
# ---- END PLUGIN-COMPAT ----
