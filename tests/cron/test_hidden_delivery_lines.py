"""Hidden delivery lines: LIFE_LOG bookkeeping and residual MEDIA tags must never
reach the chat surface (regression: raw LIFE_LOG/MEDIA lines shown on Telegram)."""
from cron.scheduler_delivery import _strip_hidden_delivery_lines


def test_life_log_line_stripped():
    text = "come home, my love.\n\nLIFE_LOG: Asuna spent the evening painting (started 17:26)"
    assert _strip_hidden_delivery_lines(text) == "come home, my love."


def test_residual_media_line_stripped():
    text = "look what i made.\n\nMEDIA:/home/ubuntu/.hermes/cache/images/runpod_x.jpeg"
    assert _strip_hidden_delivery_lines(text) == "look what i made."


def test_blank_gap_collapsed_and_edges_trimmed():
    text = "first line.\n\n\nLIFE_LOG: something\n\n\nlast line."
    assert _strip_hidden_delivery_lines(text) == "first line.\n\nlast line."


def test_normal_text_untouched():
    text = "MEDIA is my favorite topic\nand life_log is a word here\n\nunchanged body."
    assert _strip_hidden_delivery_lines(text) == text


def test_fail_open_on_empty():
    assert _strip_hidden_delivery_lines("") == ""
    assert _strip_hidden_delivery_lines(None) is None
