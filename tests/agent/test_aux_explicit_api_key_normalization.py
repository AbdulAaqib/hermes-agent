"""Regression: a ``key_cmd`` CommandTokenSource must survive aux resolution.

The compression summarizer threads the main runtime's credential into
``call_llm(main_runtime={..., "api_key": self.api_key})``. When the main provider
is ``custom`` and its credential is a callable bearer provider (``key_cmd``),
the custom resolve branch used to call ``.strip()`` on it:

    Failed to generate context summary: 'CommandTokenSource' object has no
    attribute 'strip'

Fix is at the source: normalize an explicit api_key by type (strip ``str``,
PRESERVE a zero-arg callable, else ``""``).
"""

import pytest

from agent.auxiliary_client import (
    _ResolveRequest,
    _normalize_explicit_api_key,
    _resolve_custom_branch,
)
from agent.command_token_source import CommandTokenSource


class TestNormalizeExplicitApiKey:
    def test_str_is_stripped(self):
        assert _normalize_explicit_api_key("  sk-abc  ") == "sk-abc"

    def test_callable_is_preserved_not_stripped(self):
        source = CommandTokenSource("printf tok", "dbx")
        assert _normalize_explicit_api_key(source) is source

    def test_none_and_other_become_empty(self):
        assert _normalize_explicit_api_key(None) == ""
        assert _normalize_explicit_api_key(123) == ""
        assert _normalize_explicit_api_key(object()) == ""


class TestCustomBranchAcceptsCallableKey:
    def test_explicit_callable_does_not_raise(self):
        source = CommandTokenSource("printf tok", "dbx")
        req = _ResolveRequest(
            provider="custom", original_provider="custom", model="gpt-4o-mini",
            async_mode=False, raw_codex=False,
            explicit_base_url="http://custom.local/v1", explicit_api_key=source,
            api_mode=None, main_runtime=None, is_vision=False, task=None,
        )
        client, model = _resolve_custom_branch(req)
        assert client is not None
        # The SDK stores a callable bearer in _api_key_provider, not .api_key.
        assert client._api_key_provider is source

    def test_main_runtime_callable_does_not_raise(self):
        source = CommandTokenSource("printf tok", "dbx")
        req = _ResolveRequest(
            provider="custom", original_provider="custom", model=None,
            async_mode=False, raw_codex=False,
            explicit_base_url=None, explicit_api_key=None,
            api_mode=None,
            main_runtime={"base_url": "http://custom.local/v1", "api_key": source, "model": "gpt-4o-mini"},
            is_vision=False, task=None,
        )
        client, model = _resolve_custom_branch(req)
        assert client is not None
        assert client._api_key_provider is source
