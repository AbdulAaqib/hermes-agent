"""Caption-merge delivery: short text + exactly one image arrives as ONE photo message.

A reply like ``"Here is your painting."`` + ``MEDIA:/path/img.png`` used to arrive as TWO
Telegram messages — the text, then a captionless photo. On caption-capable adapters
(Telegram) the text now rides as the photo's caption when the response carries exactly one
image (and nothing else) and the text fits the platform's caption limit (1024 chars).

Contracts pinned here (``_process_message_background`` end to end):

1. single image + short text → one image send with the text as caption, NO text send;
2. text over the caption limit → unchanged split (text send + captionless image);
3. multiple images → unchanged split;
4. a caption-incapable adapter (non-Telegram) → unchanged split.
"""

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
    single_image_caption_merge_target,
)
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource, build_session_key


class _CaptionMergeAdapter(BasePlatformAdapter):
    """Recording adapter; the platform is switchable to exercise the capability guard."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), platform)
        self.text_sends: list = []
        self.image_sends: list = []  # (sender_kwarg, ref, caption)
        self.outcomes: list = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.text_sends.append(content)
        return SendResult(success=True, message_id=f"text-{len(self.text_sends)}")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def send_image_file(self, chat_id, image_path, caption=None,
                              reply_to=None, metadata=None, **kwargs) -> SendResult:
        self.image_sends.append(("image_path", str(image_path), caption))
        return SendResult(success=True, message_id="img-1")

    async def send_image(self, chat_id, image_url, caption=None,
                         reply_to=None, metadata=None) -> SendResult:
        self.image_sends.append(("image_url", image_url, caption))
        return SendResult(success=True, message_id="img-1")

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        self.outcomes.append(outcome)


async def _hold_typing(_chat_id, interval=2.0, metadata=None, stop_event=None):
    if stop_event is not None:
        await stop_event.wait()
    else:
        await asyncio.Event().wait()


def _allowed_image(tmp_path, monkeypatch, name: str):
    root = tmp_path / "media-cache"
    f = root / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 64)
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (root,))
    return f.resolve()


def _make_event(platform=Platform.TELEGRAM) -> MessageEvent:
    return MessageEvent(
        text="show me",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=platform, chat_id="42", chat_type="dm"),
        message_id="m1",
    )


async def _drive(adapter, response: str, platform=Platform.TELEGRAM) -> None:
    adapter._keep_typing = _hold_typing

    async def handler(_event):
        return response

    adapter.set_message_handler(handler)
    event = _make_event(platform)
    await adapter._process_message_background(event, build_session_key(event.source))


@pytest.mark.asyncio
async def test_single_image_short_text_arrives_as_one_photo_message(tmp_path, monkeypatch):
    """The text rides as the photo's caption; no separate text message is sent."""
    png = _allowed_image(tmp_path, monkeypatch, "painting.png")
    adapter = _CaptionMergeAdapter()

    await _drive(adapter, f"Here is your painting.\nMEDIA:{png}")

    assert adapter.image_sends == [("image_path", str(png), "Here is your painting.")]
    assert adapter.text_sends == []
    assert adapter.outcomes == [ProcessingOutcome.SUCCESS]


@pytest.mark.asyncio
async def test_image_url_short_text_arrives_as_one_photo_message(tmp_path, monkeypatch):
    """Markdown image URLs merge the same way (send_image with the caption)."""
    adapter = _CaptionMergeAdapter()

    await _drive(adapter, "Look at this ![pic](https://example.com/pic.png)")

    assert adapter.image_sends == [
        ("image_url", "https://example.com/pic.png", "Look at this")]
    assert adapter.text_sends == []


@pytest.mark.asyncio
async def test_text_over_caption_limit_keeps_split_delivery(tmp_path, monkeypatch):
    """A reply that cannot fit the 1024-char caption cap keeps text + captionless photo."""
    png = _allowed_image(tmp_path, monkeypatch, "painting.png")
    long_text = "x" * 1025
    adapter = _CaptionMergeAdapter()

    await _drive(adapter, f"{long_text}\nMEDIA:{png}")

    assert adapter.text_sends == [long_text]
    assert adapter.image_sends == [("image_path", str(png), None)]


@pytest.mark.asyncio
async def test_multiple_images_keep_split_delivery(tmp_path, monkeypatch):
    """More than one image → text message plus each photo sent separately, as before."""
    png1 = _allowed_image(tmp_path, monkeypatch, "one.png")
    png2 = _allowed_image(tmp_path, monkeypatch, "two.png")
    adapter = _CaptionMergeAdapter()

    await _drive(adapter, f"Two paintings.\nMEDIA:{png1}\nMEDIA:{png2}")

    assert adapter.text_sends == ["Two paintings."]
    assert sorted(adapter.image_sends) == sorted([
        ("image_path", str(png1), None),
        ("image_path", str(png2), None),
    ])


@pytest.mark.asyncio
async def test_caption_incapable_adapter_keeps_split_delivery(tmp_path, monkeypatch):
    """Non-Telegram adapters have no native photo caption — behavior stays unchanged."""
    png = _allowed_image(tmp_path, monkeypatch, "painting.png")
    adapter = _CaptionMergeAdapter(platform=Platform.SIGNAL)

    await _drive(adapter, f"Here is your painting.\nMEDIA:{png}", platform=Platform.SIGNAL)

    assert adapter.text_sends == ["Here is your painting."]
    assert adapter.image_sends == [("image_path", str(png), None)]


class TestMergeTargetShape:
    """The pure eligibility helper: exactly one image and NOTHING else."""

    def test_voice_tagged_image_is_not_a_merge_target(self):
        assert single_image_caption_merge_target([], [("/tmp/a.png", True)]) is None

    def test_image_plus_video_is_not_a_merge_target(self):
        assert single_image_caption_merge_target(
            [], [("/tmp/a.png", False), ("/tmp/b.mp4", False)]) is None

    def test_force_document_images_are_not_merge_targets(self):
        assert single_image_caption_merge_target(
            [], [("/tmp/a.png", False)], force_document=True) is None

    def test_gif_url_routes_to_the_animation_sender(self):
        assert single_image_caption_merge_target(
            [("https://example.com/dance.gif", "")], []) == (
                "animation_url", "https://example.com/dance.gif")
