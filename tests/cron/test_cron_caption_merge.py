"""Cron live-adapter caption merge: a short brief + exactly one image arrives as ONE photo.

Life-update style cron jobs deliver ``text + MEDIA:<image>``; on Telegram that used to land
as a text message followed by a captionless photo. On the live-adapter lane the text now
rides as the photo's caption when it fits the 1024-char caption cap. Contracts:

1. single image + short text → no DeliveryRouter text send; one image send with caption;
2. text over the caption cap → unchanged (router text send + captionless image);
3. multiple images → unchanged;
4. caption-incapable platform (non-Telegram) → unchanged;
5. the text is NEVER lost: if the captioned photo send fails, the text still goes out.
"""

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cron.scheduler import _deliver_result
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult

CHAT_ID = "-1001234567890"


class _RecordingAdapter:
    """Stand-in for the live gateway adapter; records native image sends."""

    def __init__(self, platform=Platform.TELEGRAM, image_ok=True):
        self.platform = platform
        self._image_ok = image_ok
        self.image_calls = []

    async def send_image_file(self, chat_id, image_path, caption=None, metadata=None, **kwargs):
        self.image_calls.append({"image_path": str(image_path), "caption": caption,
                                 "metadata": metadata})
        if self._image_ok:
            return SendResult(success=True, message_id="photo-1")
        return SendResult(success=False, error="upload failed")


def _job(platform="telegram"):
    return {
        "id": "92e639af907f",
        "name": "Life Update",
        "deliver": "origin",
        "origin": {"platform": platform, "chat_id": CHAT_ID},
    }


def _gateway_config(platform):
    config = MagicMock()
    config.platforms = {platform: PlatformConfig(enabled=True)}
    config.get_home_channel = lambda p: None
    return config


def _run(job, content, adapter, media_root, monkeypatch):
    """Drive ``_deliver_result`` over the live lane with a stubbed router/loop."""
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (media_root,))
    loop = MagicMock()
    loop.is_running.return_value = True

    def fake_run_coro(coro, _loop):
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as e:  # noqa: BLE001
            future.set_exception(e)
        return future

    router_calls = []
    router = MagicMock()

    async def _deliver_to_platform(target, text, metadata):
        router_calls.append({"target": target, "text": text, "metadata": metadata})
        return SimpleNamespace(success=True, message_id=1234, raw_response=None)

    router._deliver_to_platform = _deliver_to_platform

    with patch("gateway.config.load_gateway_config",
               return_value=_gateway_config(adapter.platform)), \
         patch("cron.scheduler.load_config",
               return_value={"cron": {"wrap_response": False}}), \
         patch("cron.scheduler_delivery._record_delivery_verification", lambda *a, **k: None), \
         patch("gateway.delivery.DeliveryRouter", return_value=router), \
         patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
        error = _deliver_result(job, content, adapters={adapter.platform: adapter}, loop=loop)
    return error, router_calls


def _media_file(tmp_path, name="life.png"):
    root = tmp_path / "media-cache"
    f = root / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 64)
    return f.resolve()


def test_single_image_brief_arrives_as_one_photo_message(tmp_path, monkeypatch):
    png = _media_file(tmp_path)
    adapter = _RecordingAdapter()

    error, router_calls = _run(
        _job(), f"Morning! Painted today.\nMEDIA:{png}", adapter, png.parent, monkeypatch)

    assert error is None
    assert router_calls == []  # no separate text message
    assert adapter.image_calls == [
        {"image_path": str(png), "caption": "Morning! Painted today.",
         "metadata": adapter.image_calls[0]["metadata"]}]
    assert adapter.image_calls[0]["metadata"]["notify"] is True


def test_long_brief_keeps_text_plus_captionless_photo(tmp_path, monkeypatch):
    png = _media_file(tmp_path)
    adapter = _RecordingAdapter()
    long_text = "x" * 1025

    error, router_calls = _run(
        _job(), f"{long_text}\nMEDIA:{png}", adapter, png.parent, monkeypatch)

    assert error is None
    assert [call["text"] for call in router_calls] == [long_text]
    assert [call["caption"] for call in adapter.image_calls] == [None]


def test_multiple_images_keep_text_plus_separate_photos(tmp_path, monkeypatch):
    png1 = _media_file(tmp_path, "one.png")
    png2 = _media_file(tmp_path, "two.png")
    adapter = _RecordingAdapter()

    error, router_calls = _run(
        _job(), f"Two pieces today.\nMEDIA:{png1}\nMEDIA:{png2}",
        adapter, png1.parent, monkeypatch)

    assert error is None
    assert [call["text"] for call in router_calls] == ["Two pieces today."]
    assert [call["caption"] for call in adapter.image_calls] == [None, None]


def test_caption_incapable_platform_keeps_split_delivery(tmp_path, monkeypatch):
    png = _media_file(tmp_path)
    adapter = _RecordingAdapter(platform=Platform.SLACK)

    error, router_calls = _run(
        _job("slack"), f"Morning! Painted today.\nMEDIA:{png}",
        adapter, png.parent, monkeypatch)

    assert error is None
    assert [call["text"] for call in router_calls] == ["Morning! Painted today."]
    assert [call["caption"] for call in adapter.image_calls] == [None]


def test_failed_caption_photo_still_delivers_the_text(tmp_path, monkeypatch):
    """Fail-open: the brief text must never be lost to a failed photo upload."""
    png = _media_file(tmp_path)
    adapter = _RecordingAdapter(image_ok=False)

    error, router_calls = _run(
        _job(), f"Morning! Painted today.\nMEDIA:{png}", adapter, png.parent, monkeypatch)

    assert [call["text"] for call in router_calls] == ["Morning! Painted today."]
    assert error is not None  # the media failure is surfaced in run status
