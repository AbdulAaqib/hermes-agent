"""A packaged Desktop UI must not ride along behind a current source stamp.

On a git install, ``hermes update`` stamps desktop freshness from the source
tree hash. ``apps/desktop/release/`` is git-ignored, so that hash cannot see
``app.asar``. A skipped or failed rebuild then looks current, and the window
keeps the old renderer.

The stamp has to name the packaged artifact it produced. A live ``app.asar``
that is not that artifact is stale, and a skip that cannot rebuild an installed
app has to say so instead of returning through the "nothing to do" door.
"""

import json
import os
import sys

from hermes_cli import main_desktop, update_cmd
from hermes_cli.main_desktop import (
    _desktop_build_needed,
    _write_desktop_build_stamp,
)
from hermes_cli.update_cmd import _rebuild_desktop_after_update


class _Result:
    def __init__(self, returncode: int, stdout: str = ""):
        self.returncode = returncode
        self.stdout = stdout


def _packaged_app(root):
    """Minimal git-install Desktop: source tree plus packed exe and app.asar."""
    desktop = root / "apps" / "desktop"
    (desktop / "src").mkdir(parents=True)
    (desktop / "package.json").write_text('{"name": "hermes-desktop"}', encoding="utf-8")
    (desktop / "src" / "index.ts").write_text("export const x = 1\n", encoding="utf-8")
    (root / ".gitignore").write_text("apps/desktop/release/\n", encoding="utf-8")

    if sys.platform == "darwin":
        exe = desktop / "release" / "mac" / "Hermes.app" / "Contents" / "MacOS" / "Hermes"
        resources = exe.parent.parent / "Resources"
    elif sys.platform == "win32":
        exe = desktop / "release" / "win-unpacked" / "Hermes.exe"
        resources = exe.parent / "resources"
    else:
        exe = desktop / "release" / "linux-unpacked" / "hermes"
        resources = exe.parent / "resources"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"MZ")
    resources.mkdir(parents=True)
    asar = resources / "app.asar"
    asar.write_bytes(b"asar-generation-A")
    return desktop, asar


def _age_by_days(path, days):
    old = path.stat().st_mtime - days * 86400
    os.utime(path, (old, old))


def _packaged(tmp_path, monkeypatch):
    desktop, asar = _packaged_app(tmp_path)
    stamp = tmp_path / "hermes-home" / "desktop-build-stamp.json"
    monkeypatch.setattr(main_desktop, "_desktop_stamp_path", lambda: stamp)
    return tmp_path, desktop, asar, stamp


def _fake_update_main(root, *, spawned, npm: str | None = "/fake/npm", stamp=None):
    class _FakeMain:
        PROJECT_ROOT = root
        _desktop_build_needed = staticmethod(_desktop_build_needed)
        _desktop_packaged_executable = staticmethod(main_desktop._desktop_packaged_executable)
        _desktop_dist_exists = staticmethod(main_desktop._desktop_dist_exists)
        _desktop_stamp_path = staticmethod(lambda: stamp or (root / "missing-stamp.json"))
        _resolve_node_runtime_npm = staticmethod(lambda: npm)
        _run_logged_subprocess = staticmethod(
            lambda cmd, cwd=None, env=None: spawned.append(cmd) or _Result(0)
        )
        _install_rebuilt_desktop_app = staticmethod(lambda desktop_dir: ([], []))

    return _FakeMain


def test_a_build_binds_its_stamp_to_the_packaged_asar(tmp_path, monkeypatch):
    root, _desktop, asar, stamp = _packaged(tmp_path, monkeypatch)
    _write_desktop_build_stamp(root, source_mode=False)
    recorded = json.loads(stamp.read_text(encoding="utf-8"))
    stat_result = asar.stat()
    assert recorded["sourceMode"] is False
    assert recorded["artifact"] == f"{stat_result.st_mtime_ns}:{stat_result.st_size}"


def test_three_week_old_asar_behind_a_current_stamp_is_stale(tmp_path, monkeypatch, capsys):
    root, desktop, asar, _stamp = _packaged(tmp_path, monkeypatch)
    _write_desktop_build_stamp(root, source_mode=False)
    _age_by_days(asar, 21)
    assert _desktop_build_needed(desktop, root, source_mode=False) is True
    assert "stale" in capsys.readouterr().out.lower()


def test_a_replaced_asar_behind_a_current_source_stamp_is_stale(tmp_path, monkeypatch):
    root, desktop, asar, _stamp = _packaged(tmp_path, monkeypatch)
    _write_desktop_build_stamp(root, source_mode=False)
    asar.write_bytes(b"asar-generation-B-from-another-build")
    assert _desktop_build_needed(desktop, root, source_mode=False) is True


def test_missing_packaged_asar_behind_a_current_stamp_is_stale(tmp_path, monkeypatch):
    root, desktop, asar, _stamp = _packaged(tmp_path, monkeypatch)
    _write_desktop_build_stamp(root, source_mode=False)
    asar.unlink()
    assert _desktop_build_needed(desktop, root, source_mode=False) is True


def test_legacy_stamp_without_an_artifact_binding_is_rebuilt(tmp_path, monkeypatch):
    root, desktop, _asar, stamp = _packaged(tmp_path, monkeypatch)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(
        json.dumps(
            {
                "contentHash": main_desktop._compute_desktop_content_hash(root),
                "sourceMode": False,
            }
        ),
        encoding="utf-8",
    )
    assert _desktop_build_needed(desktop, root, source_mode=False) is True


def test_source_hash_alone_cannot_see_a_stale_asar(tmp_path, monkeypatch):
    root, _desktop, asar, stamp = _packaged(tmp_path, monkeypatch)
    _write_desktop_build_stamp(root, source_mode=False)
    before = json.loads(stamp.read_text(encoding="utf-8"))["contentHash"]
    _age_by_days(asar, 21)
    assert main_desktop._compute_desktop_content_hash(root) == before


def test_update_schedules_a_rebuild_for_a_stale_packaged_ui(tmp_path, monkeypatch, capsys):
    root, desktop, asar, stamp = _packaged(tmp_path, monkeypatch)
    _write_desktop_build_stamp(root, source_mode=False)
    _age_by_days(asar, 21)
    spawned = []
    monkeypatch.setattr(
        update_cmd, "_m", lambda: _fake_update_main(root, spawned=spawned, stamp=stamp)
    )
    monkeypatch.setattr("hermes_constants.with_hermes_node_path", lambda: {}, raising=False)

    assert _rebuild_desktop_after_update(desktop, had_desktop_app_before_update=True) is True
    assert spawned, "a stale packaged UI must schedule desktop --build-only"
    assert "desktop" in spawned[0] and "--build-only" in spawned[0]
    assert "stale" in capsys.readouterr().out.lower()


def test_update_does_not_rebuild_a_matching_asar(tmp_path, monkeypatch, capsys):
    root, desktop, _asar, stamp = _packaged(tmp_path, monkeypatch)
    _write_desktop_build_stamp(root, source_mode=False)
    spawned = []
    monkeypatch.setattr(
        update_cmd, "_m", lambda: _fake_update_main(root, spawned=spawned, stamp=stamp)
    )
    monkeypatch.setattr("hermes_constants.with_hermes_node_path", lambda: {}, raising=False)

    assert _rebuild_desktop_after_update(desktop, had_desktop_app_before_update=True) is True
    assert spawned == []
    assert "up to date" in capsys.readouterr().out


def test_installed_desktop_without_npm_warns_instead_of_passing_silently(
    tmp_path, monkeypatch, capsys
):
    root, desktop, _asar, stamp = _packaged(tmp_path, monkeypatch)
    spawned = []
    monkeypatch.setattr(
        update_cmd,
        "_m",
        lambda: _fake_update_main(root, spawned=spawned, npm=None, stamp=stamp),
    )

    assert _rebuild_desktop_after_update(desktop, had_desktop_app_before_update=True) is True
    out = capsys.readouterr().out
    assert spawned == []
    assert "npm" in out.lower()
    assert "stale" in out.lower()


def test_a_desktop_that_was_never_installed_stays_silent(tmp_path, monkeypatch, capsys):
    desktop = tmp_path / "apps" / "desktop"
    desktop.mkdir(parents=True)
    (desktop / "package.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        update_cmd, "_m", lambda: _fake_update_main(tmp_path, spawned=[], npm=None)
    )
    assert _rebuild_desktop_after_update(desktop, had_desktop_app_before_update=False) is True
    assert "npm" not in capsys.readouterr().out.lower()
