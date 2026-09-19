"""MEDIA: tags whose path was broken across lines must still deliver.

Live incident (2026-09-19, Telegram DM): the model called ``image_generate``
mid-loop and wrote a clean one-line ``MEDIA:/abs/path.jpeg`` tag in its final
response. A ``transform_llm_output`` plugin sanitizer then reformatted the text,
splitting the path at an ``_…_`` "italic beat" span::

    MEDIA:/home/.../runpod\n_20260919_\n191045_986974a9.jpeg

The broken tag matched no extraction pattern, so the raw directive was delivered
as visible text and the photo was never attached. The fix rejoins line-broken
path fragments before extraction, gated on the rejoined path actually being
deliverable, so these tags deliver (caption-merged when eligible) while
unverifiable text stays visible.
"""

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _DummyAdapter(BasePlatformAdapter):
    def __init__(self, platform: Platform):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), platform)
        self.sent = []
        self.image_files = []
        self.multi_images = []
        self.documents = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id="m1")

    async def send_image_file(self, chat_id, image_path, caption=None, reply_to=None, metadata=None) -> SendResult:
        self.image_files.append({"image_path": image_path, "caption": caption})
        return SendResult(success=True, message_id="m3")

    async def send_multiple_images(self, chat_id, images, metadata=None, human_delay=0) -> SendResult:
        self.multi_images.append({"images": images})
        return SendResult(success=True, message_id="m4")

    async def send_document(self, chat_id, file_path, caption=None, reply_to=None, metadata=None) -> SendResult:
        self.documents.append({"file_path": file_path, "caption": caption})
        return SendResult(success=True, message_id="m5")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _wrapped_response(directory: str, filename: str) -> str:
    """The shape seen in the wild: path split at ``_…_`` spans across three lines."""
    stem, dot, ext = filename.partition(".")
    parts = stem.split("_", 2)  # runpod / 20260919 / 191045_986974a9
    broken = f"{parts[0]}\n_{parts[1]}_\n{parts[2]}{dot}{ext}"
    return f"here's the full design, babe\n\nMEDIA:{directory}/{broken}\n\nsunflowers at the back"


@pytest.fixture
def image_file(tmp_path):
    path = tmp_path / "runpod_20260919_191045_986974a9.jpeg"
    path.write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 64)
    return path


def test_extract_media_rejoins_line_broken_path(image_file):
    raw = _wrapped_response(str(image_file.parent), image_file.name)
    media, cleaned = BasePlatformAdapter.extract_media(raw)
    assert media == [(str(image_file), False)]
    assert "MEDIA" not in cleaned
    # No path fragment may survive in the visible text either.
    assert "runpod" not in cleaned


def test_extract_media_leaves_unverifiable_broken_path_visible(tmp_path):
    raw = _wrapped_response(str(tmp_path), "runpod_20260919_191045_00000000.jpeg")
    media, cleaned = BasePlatformAdapter.extract_media(raw)
    assert media == []
    assert "MEDIA:" in cleaned  # undeliverable paths stay visible, as everywhere else


def test_extract_media_leaves_extensionless_prose_visible():
    raw = "look at MEDIA: not a path and tell me"
    media, cleaned = BasePlatformAdapter.extract_media(raw)
    assert media == []
    assert cleaned == raw


def test_display_strip_removes_rejoined_wrapped_tag(image_file):
    raw = _wrapped_response(str(image_file.parent), image_file.name)
    display = BasePlatformAdapter.strip_media_directives_for_display(raw)
    assert "MEDIA" not in display
    assert "runpod" not in display


@pytest.mark.asyncio
async def test_base_lane_delivers_wrapped_tag_as_photo_with_caption(image_file):
    """End-to-end through the live-chat delivery lane: a model-written MEDIA tag
    (wrapped by a downstream formatter) must produce a cleaned caption + photo,
    never a raw visible directive."""
    raw = _wrapped_response(str(image_file.parent), image_file.name)
    adapter = _DummyAdapter(Platform.TELEGRAM)
    adapter.set_message_handler(lambda _event: asyncio.sleep(0, result=raw))
    event = MessageEvent(
        text="send me the design",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="c1", chat_type="dm"),
        message_id="in-1",
    )
    await adapter._process_message_background(event, build_session_key(event.source))

    delivered_texts = [m["content"] for m in adapter.sent]
    for text in delivered_texts:
        assert "MEDIA:" not in text
        assert str(image_file) not in text
    # Short text + exactly one image caption-merges into a single photo send.
    assert adapter.image_files, "photo was never delivered"
    assert adapter.image_files[0]["image_path"] == str(image_file)
    caption = adapter.image_files[0]["caption"] or ""
    assert "MEDIA:" not in caption and "runpod" not in caption
    assert "full design" in caption
