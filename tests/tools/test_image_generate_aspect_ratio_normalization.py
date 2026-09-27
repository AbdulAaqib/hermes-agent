"""image_generate accepts common aspect_ratio spellings (ratio / word aliases).

Models routinely emit ``aspect_ratio: "3:4"`` (or "vertical"/"wide") although the
schema enum is only ``landscape`` / ``square`` / ``portrait``. Before this fix the
value failed deferred-argument validation and cost a wasted, billed tool round-trip.
A per-tool normaliser now maps the common spellings before validation; unknown values
still fail exactly as before, and no other tool's validation is loosened.
"""

from __future__ import annotations

import json

import pytest

# Importing the tool module registers both the image_generate tool and its normaliser.
import tools.image_generation_tool as ig


@pytest.mark.parametrize("raw,expected", [
    ("3:4", "portrait"),
    ("2:3", "portrait"),
    ("4:5", "portrait"),
    ("9:16", "portrait"),
    ("9:21", "portrait"),
    ("vertical", "portrait"),
    ("  VERTICAL  ", "portrait"),
    ("tall", "portrait"),
    ("4:3", "landscape"),
    ("3:2", "landscape"),
    ("5:4", "landscape"),
    ("16:9", "landscape"),
    ("21:9", "landscape"),
    ("horizontal", "landscape"),
    ("Wide", "landscape"),
    ("1:1", "square"),
])
def test_aliases_map_to_canonical(raw, expected):
    out = ig.normalize_image_generate_args({"prompt": "x", "aspect_ratio": raw})
    assert out["aspect_ratio"] == expected


@pytest.mark.parametrize("raw", ["landscape", "square", "portrait"])
def test_canonical_values_unchanged_identity(raw):
    args = {"prompt": "x", "aspect_ratio": raw}
    assert ig.normalize_image_generate_args(args) is args


def test_unknown_value_left_for_validation():
    args = {"prompt": "x", "aspect_ratio": "banana"}
    assert ig.normalize_image_generate_args(args) is args


def test_missing_or_non_string_aspect_ratio_untouched():
    no_key = {"prompt": "x"}
    assert ig.normalize_image_generate_args(no_key) is no_key
    non_str = {"prompt": "x", "aspect_ratio": 34}
    assert ig.normalize_image_generate_args(non_str) is non_str


def test_validation_accepts_ratio_and_still_rejects_unknown():
    from tools.tool_search_validation import validate_deferred_call_args

    ok = validate_deferred_call_args("image_generate", {"prompt": "x", "aspect_ratio": "3:4"})
    assert ok is None

    bad = validate_deferred_call_args("image_generate", {"prompt": "x", "aspect_ratio": "banana"})
    assert bad is not None
    payload = json.loads(bad)
    assert "aspect_ratio" in json.dumps(payload)


def test_normalizer_registry_is_per_tool():
    """The normaliser is registered only for image_generate — other tools keep their
    own (unrelaxed) validation."""
    from tools.arg_coercion import _ARG_NORMALIZERS

    assert _ARG_NORMALIZERS.get("image_generate") is ig.normalize_image_generate_args
    assert "read_file" not in _ARG_NORMALIZERS
