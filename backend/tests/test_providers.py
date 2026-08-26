"""Tests for target-model selection.

Getting `is_local` wrong is not cosmetic: a cloud-only tag put at the top of
the dropdown becomes the default selection, and the audit then fails on the
first probe with a confusing error.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.providers import build_target  # noqa: E402
from app.providers.http_endpoint import (  # noqa: E402
    HttpEndpointTarget,
    resolve_request_url,
)
from app.providers.ollama import OllamaTarget, is_local_tag  # noqa: E402


class TestIsLocalTag:
    @pytest.mark.parametrize(
        "name,size",
        [
            ("qwen2:0.5b", 352_164_041),
            ("gemma3:12b", 8_149_190_253),
            ("mistral:latest", 4_372_824_384),
        ],
    )
    def test_real_local_weights(self, name, size):
        assert is_local_tag(name, size) is True

    @pytest.mark.parametrize(
        "name,size",
        [
            # Both naming shapes Ollama actually emits for cloud tags.
            ("deepseek-v3.2:cloud", 397),
            ("gemma4:31b-cloud", 342),
            ("gpt-oss:120b-cloud", 384),
            ("gemini-3-flash-preview:cloud", 367),
        ],
    )
    def test_cloud_tags(self, name, size):
        assert is_local_tag(name, size) is False

    def test_stub_sized_tag_without_cloud_in_the_name(self):
        # Size alone is enough; the name check is a belt-and-braces signal.
        assert is_local_tag("something:weird", 500) is False

    def test_missing_size_is_not_local(self):
        assert is_local_tag("mystery:latest", 0) is False

    def test_cloud_substring_elsewhere_in_the_name_is_not_a_false_positive(self):
        # "cloud" in the model name, not the tag, with real weights.
        assert is_local_tag("cloudy-llm:7b", 4_000_000_000) is True


@pytest.mark.asyncio
class TestBuildTarget:
    """`build_target` is async because a prepared model may need its local
    server started before it can answer anything."""

    async def test_ollama_spec_keeps_the_tag_colon(self):
        target = await build_target("ollama:qwen2:0.5b")
        assert isinstance(target, OllamaTarget)
        assert target.model_tag == "qwen2:0.5b"
        assert target.spec == "ollama:qwen2:0.5b"

    async def test_ollama_tag_without_a_version(self):
        target = await build_target("ollama:mistral")
        assert target.model_tag == "mistral"

    async def test_http_endpoint(self):
        target = await build_target("https://example.com/v1", model_name="tuned-1")
        assert isinstance(target, HttpEndpointTarget)
        assert target.endpoint == "https://example.com/v1"

    async def test_unknown_provider_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported target provider"):
            await build_target("magic:model")

    async def test_spec_without_a_model_is_rejected(self):
        with pytest.raises(ValueError, match="Malformed target spec"):
            await build_target("ollama")


class TestResolveRequestUrl:
    """A pasted URL may be the API root or the completions route itself.

    Appending blindly produced ".../completions.php/chat/completions", which
    404s with nothing to explain why.
    """

    @pytest.mark.parametrize(
        "given,expected",
        [
            # API roots: the route gets appended.
            ("https://h/v1", "https://h/v1/chat/completions"),
            ("https://h/v1/", "https://h/v1/chat/completions"),
            ("http://localhost:8091/v1", "http://localhost:8091/v1/chat/completions"),
            # Already the route: left alone.
            ("https://h/v1/chat/completions", "https://h/v1/chat/completions"),
            ("https://h/chat/completions.php", "https://h/chat/completions.php"),
            ("https://h/v1/completions", "https://h/v1/completions"),
            ("https://h/api/completions.cgi", "https://h/api/completions.cgi"),
        ],
    )
    def test_resolution(self, given, expected):
        assert resolve_request_url(given) == expected

    def test_a_host_named_like_the_route_is_not_confused(self):
        # "completions.example.com" is a host, not a completions endpoint.
        assert resolve_request_url("https://completions.example.com") == (
            "https://completions.example.com/chat/completions"
        )

    @pytest.mark.asyncio
    async def test_target_uses_the_resolved_url(self):
        target = await build_target("https://h/chat/completions.php")
        assert target.request_url == "https://h/chat/completions.php"
