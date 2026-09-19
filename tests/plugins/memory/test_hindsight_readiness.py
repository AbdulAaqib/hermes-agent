"""Tests for the Hindsight bounded-readiness handshake and per-profile banks.

Contract: ``_probe_daemon_readiness`` is a fast bounded gate — cloud mode never
touches the network, and a dead local daemon is reported in milliseconds, not
the plugin-default timeout. ``bank_id_per_profile`` derives per-profile banks
for non-default profiles while leaving the legacy shared bank untouched.
"""

import time

import pytest

from plugins.memory.hindsight import HindsightMemoryProvider


@pytest.fixture(autouse=True)
def _clean_env(tmp_path, monkeypatch):
    for key in (
        "HINDSIGHT_API_KEY", "HINDSIGHT_API_URL", "HINDSIGHT_BANK_ID",
        "HINDSIGHT_MODE", "HINDSIGHT_TIMEOUT", "HINDSIGHT_IDLE_TIMEOUT",
    ):
        monkeypatch.delenv(key, raising=False)
    yield


class TestProbeDaemonReadiness:
    def test_cloud_mode_returns_true_without_network(self):
        provider = HindsightMemoryProvider()
        provider._mode = "cloud"
        provider._api_url = "http://127.0.0.1:1"
        start = time.monotonic()
        assert provider._probe_daemon_readiness(300) is True
        assert time.monotonic() - start < 1.0  # no network round trip

    def test_local_external_dead_port_returns_false_quickly(self):
        provider = HindsightMemoryProvider()
        provider._mode = "local_external"
        provider._api_url = "http://127.0.0.1:1"
        start = time.monotonic()
        assert provider._probe_daemon_readiness(300) is False
        elapsed = time.monotonic() - start
        assert elapsed < 2.0  # far under the plugin-default 30s timeout


class TestPerProfileBanks:
    def _apply(self, identity, cfg):
        provider = HindsightMemoryProvider()
        provider._agent_identity = identity
        provider._agent_workspace = ""
        provider._platform = "cli"
        provider._user_id = ""
        provider._session_id = ""
        provider._apply_connection_settings(cfg)
        return provider

    def test_non_default_profile_derives_bank_template(self):
        provider = self._apply("work", {"bank_id_per_profile": True, "bank_id": "hermes"})
        assert provider._bank_id_template == "hermes-{profile}"
        assert provider._bank_id == "hermes-work"

    def test_default_profile_keeps_shared_bank(self):
        provider = self._apply("default", {"bank_id_per_profile": True, "bank_id": "hermes"})
        assert provider._bank_id_template == ""
        assert provider._bank_id == "hermes"

    def test_hermes_identity_keeps_shared_bank(self):
        provider = self._apply("hermes", {"bank_id_per_profile": True, "bank_id": "hermes"})
        assert provider._bank_id_template == ""
        assert provider._bank_id == "hermes"

    def test_opt_out_is_byte_identical(self):
        provider = self._apply(
            "work", {"bank_id": "hermes", "bank_id_template": "", "bank_id_per_profile": False},
        )
        assert provider._bank_id_template == ""
        assert provider._bank_id == "hermes"

    def test_explicit_template_wins_over_derivation(self):
        provider = self._apply(
            "work", {"bank_id_per_profile": True, "bank_id_template": "my-{profile}"},
        )
        assert provider._bank_id_template == "my-{profile}"
        assert provider._bank_id == "my-work"