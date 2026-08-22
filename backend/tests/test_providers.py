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
from app.providers.http_endpoint import HttpEndpointTarget  # noqa: E402
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


class TestBuildTarget:
    def test_ollama_spec_keeps_the_tag_colon(self):
        target = build_target("ollama:qwen2:0.5b")
        assert isinstance(target, OllamaTarget)
        assert target.model_tag == "qwen2:0.5b"
        assert target.spec == "ollama:qwen2:0.5b"

    def test_ollama_tag_without_a_version(self):
        target = build_target("ollama:mistral")
        assert target.model_tag == "mistral"

    def test_http_endpoint(self):
        target = build_target("https://example.com/v1", model_name="tuned-1")
        assert isinstance(target, HttpEndpointTarget)
        assert target.endpoint == "https://example.com/v1"

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported target provider"):
            build_target("magic:model")

    def test_spec_without_a_model_is_rejected(self):
        with pytest.raises(ValueError, match="Malformed target spec"):
            build_target("ollama")
