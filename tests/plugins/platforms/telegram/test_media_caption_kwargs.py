"""Caption kwargs for media sends must carry MarkdownV2 parse_mode with the caption
formatted (regression: caption-merge delivered photo captions as raw text, so
_italic_ beats showed literal underscores on Telegram)."""

from types import SimpleNamespace

from gateway.config import Platform, PlatformConfig


def _adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***")
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot")
    return adapter


def test_caption_with_italic_beat_gets_parse_mode_and_italic_preserved():
    a = _adapter()
    kwargs = a._media_caption_kwargs("come here.\n\n_tracking his hands_")
    assert kwargs["parse_mode"] is not None
    assert "_tracking his hands_" in kwargs["caption"]


def test_caption_special_chars_escaped():
    a = _adapter()
    kwargs = a._media_caption_kwargs("yes. yes! (right now) — ok.")
    caption = kwargs["caption"]
    assert "\\." in caption or "\\!" in caption or "\\(" in caption


def test_none_caption_yields_no_kwargs():
    a = _adapter()
    assert a._media_caption_kwargs(None) == {}
    assert a._media_caption_kwargs("") == {}


def test_oversized_caption_falls_back_plain_truncated():
    a = _adapter()
    huge = "word " * 500  # ~2500 chars, formatted would exceed the 1024 cap
    kwargs = a._media_caption_kwargs(huge)
    assert "parse_mode" not in kwargs
    assert len(kwargs["caption"]) <= 1024
