"""Missing MEDIA paths resolve through the durable image ledger (2026-09-27 incident).

A live reply carried ``MEDIA:/home/ubuntu/.hermes/cache/images/gen_ba73f1b6c14b.jpg``
which did not exist. The gateway dropped it silently. These tests pin the repair:
a stale/wrong generated-image path resolves to the ledger's copy; a fully invented
name (no ledger stem match) still drops with the "not found on this host" log; and
bare local-file paths never use ledger resolution.

Regression for the 2026-09-27 hallucinated-MEDIA-path incident.
"""
from __future__ import annotations

import json

import pytest

import gateway.platforms.base as base
from gateway.platforms.base import BasePlatformAdapter


def _write_ledger(home, rows):
    idx = home / "images" / "index.jsonl"
    idx.parent.mkdir(parents=True, exist_ok=True)
    idx.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _make_library_image(home, name):
    p = home / "images" / "library" / "2026" / "09" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\xff\xd8\xff\xe0fake")
    return p


@pytest.fixture
def ledger_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(base, "_HERMES_HOME", home)
    monkeypatch.setattr(base, "_HERMES_ROOT", home)
    monkeypatch.setattr(base, "get_hermes_home", lambda *a, **k: home)
    # Keep the media-delivery allowlist permissive so tmp paths validate.
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "0")
    monkeypatch.delenv("HERMES_MEDIA_ALLOW_DIRS", raising=False)
    return home


def test_stale_cache_path_resolves_to_durable_library(ledger_home):
    lib = _make_library_image(ledger_home, "runpod_20260927_100134_abcd0001.jpeg")
    _write_ledger(ledger_home, [
        {"id": "img_1", "file": "images/library/2026/09/runpod_20260927_100134_abcd0001.jpeg",
         "original_cache_path": "/gone/cache/images/runpod_20260927_100134_abcd0001.jpeg"},
    ])
    missing = str(ledger_home / "cache" / "images" / "runpod_20260927_100134_abcd0001.jpeg")
    assert not base._existing_regular_file(missing)
    resolved = base.resolve_missing_media_via_ledger(missing)
    assert resolved == str(lib.resolve())


def test_wrong_directory_right_basename_resolves(ledger_home):
    lib = _make_library_image(ledger_home, "runpod_20260927_100200_beef0002.jpeg")
    _write_ledger(ledger_home, [
        {"id": "img_2", "file": "images/library/2026/09/runpod_20260927_100200_beef0002.jpeg",
         "original_cache_path": "/somewhere/else/runpod_20260927_100200_beef0002.jpeg"},
    ])
    resolved = base.resolve_missing_media_via_ledger("/tmp/whatever/runpod_20260927_100200_beef0002.jpeg")
    assert resolved == str(lib.resolve())


def test_same_stem_different_extension_resolves(ledger_home):
    """Model names ``<stem>.png`` but the ledger holds ``<stem>.jpeg``: resolve by stem."""
    lib = _make_library_image(ledger_home, "runpod_20260927_100100_bbbb0002.jpeg")
    _write_ledger(ledger_home, [
        {"id": "a", "file": "images/library/2026/09/runpod_20260927_100000_aaaa0001.jpeg",
         "original_cache_path": "/gone/a.jpeg"},
        {"id": "b", "file": "images/library/2026/09/runpod_20260927_100100_bbbb0002.jpeg",
         "original_cache_path": "/gone/b.jpeg"},
    ])
    resolved = base.resolve_missing_media_via_ledger(
        "/home/ubuntu/.hermes/cache/images/runpod_20260927_100100_bbbb0002.png")
    assert resolved == str(lib.resolve())


def test_newest_row_wins_when_stem_has_multiple_variants(ledger_home):
    older = _make_library_image(ledger_home, "runpod_20260927_100000_aaaa0001.png")
    newer = _make_library_image(ledger_home, "runpod_20260927_100000_aaaa0001.jpeg")
    _write_ledger(ledger_home, [
        {"id": "a", "file": "images/library/2026/09/runpod_20260927_100000_aaaa0001.png",
         "original_cache_path": "/gone/a.png"},
        {"id": "b", "file": "images/library/2026/09/runpod_20260927_100000_aaaa0001.jpeg",
         "original_cache_path": "/gone/b.jpeg"},
    ])
    # Both share the stem; the newer ledger row (last written) is the fallback only
    # when no exact basename matches. Here the exact .png basename exists, so it wins.
    resolved = base.resolve_missing_media_via_ledger(
        "/home/ubuntu/.hermes/cache/images/runpod_20260927_100000_aaaa0001.png")
    assert resolved == str(older.resolve())
    # A stem-only request (unknown extension) falls back to the newest row's copy.
    resolved_stem = base.resolve_missing_media_via_ledger(
        "/home/ubuntu/.hermes/cache/images/runpod_20260927_100000_aaaa0001.webp")
    assert resolved_stem == str(newer.resolve())


def test_invented_name_with_no_ledger_match_drops(ledger_home):
    _make_library_image(ledger_home, "runpod_20260927_100134_abcd0001.jpeg")
    _write_ledger(ledger_home, [
        {"id": "img_1", "file": "images/library/2026/09/runpod_20260927_100134_abcd0001.jpeg",
         "original_cache_path": "/gone/cache/images/runpod_20260927_100134_abcd0001.jpeg"},
    ])
    assert base.resolve_missing_media_via_ledger(
        str(ledger_home / "cache" / "images" / "gen_ba73f1b6c14b.jpg")) is None


def test_no_ledger_file_is_fail_open(ledger_home, caplog):
    missing = str(ledger_home / "cache" / "images" / "gen_deadbeef0000.jpg")
    assert base.resolve_missing_media_via_ledger(missing) is None
    with caplog.at_level("WARNING"):
        out = BasePlatformAdapter.filter_media_delivery_paths([(missing, False)])
    assert out == []
    assert "not found on this host" in caplog.text


def test_filter_media_paths_uses_ledger_for_media(ledger_home):
    lib = _make_library_image(ledger_home, "runpod_20260927_100300_cafe0003.jpeg")
    _write_ledger(ledger_home, [
        {"id": "img_3", "file": "images/library/2026/09/runpod_20260927_100300_cafe0003.jpeg",
         "original_cache_path": "/gone/c.jpeg"},
    ])
    missing = str(ledger_home / "cache" / "images" / "runpod_20260927_100300_cafe0003.jpeg")
    out = BasePlatformAdapter.filter_media_delivery_paths([(missing, False)])
    assert out == [(str(lib.resolve()), False)]


def test_bare_local_paths_do_not_use_ledger(ledger_home):
    """Prose citing a missing path must not be silently swapped for a recent image."""
    lib = _make_library_image(ledger_home, "runpod_20260927_100400_d00d0004.jpeg")
    _write_ledger(ledger_home, [
        {"id": "img_4", "file": "images/library/2026/09/runpod_20260927_100400_d00d0004.jpeg",
         "original_cache_path": "/gone/d.jpeg"},
    ])
    missing = str(ledger_home / "cache" / "images" / "runpod_20260927_100400_d00d0004.jpeg")
    # filter_local_delivery_paths (bare paths) keeps the drop behaviour.
    assert BasePlatformAdapter.filter_local_delivery_paths([missing]) == []
    assert lib.exists()
