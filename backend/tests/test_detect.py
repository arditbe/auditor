"""Tests for identifying what a user handed us.

Detection is the first thing a non-expert touches. Getting it wrong means
either a confusing error, or worse, confidently preparing the wrong thing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.providers.detect import (  # noqa: E402
    Readiness,
    SourceKind,
    inspect_source,
)


def _write(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _mlx_adapter_dir(tmp_path: Path, *, base="mlx-community/Mistral-7B-v0.3-4bit",
                     final=True, checkpoints=()) -> Path:
    folder = tmp_path / "adapters_v2"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "adapter_config.json").write_text(
        json.dumps({"model": base, "fine_tune_type": "lora"})
    )
    if final:
        _write(folder / "adapters.safetensors")
    for step in checkpoints:
        _write(folder / f"{step:07d}_adapters.safetensors")
    return folder


class TestMlxAdapters:
    def test_recognises_a_finished_training_run(self, tmp_path):
        folder = _mlx_adapter_dir(tmp_path)
        d = inspect_source(str(folder))

        assert d.kind is SourceKind.MLX_ADAPTERS
        assert d.readiness is Readiness.NEEDS_PREPARE
        assert d.base_model == "mlx-community/Mistral-7B-v0.3-4bit"
        assert d.action_label
        assert not d.warnings

    def test_falls_back_to_the_newest_checkpoint(self, tmp_path):
        # A run stopped early leaves checkpoints but no final adapter.
        folder = _mlx_adapter_dir(
            tmp_path, final=False, checkpoints=(100, 400, 200)
        )
        d = inspect_source(str(folder))

        assert d.readiness is Readiness.NEEDS_PREPARE
        assert d.resolved_path == str(folder)
        assert d.warnings and "400" in d.warnings[0]

    def test_config_with_no_weights_is_blocked(self, tmp_path):
        folder = tmp_path / "adapters"
        folder.mkdir()
        (folder / "adapter_config.json").write_text(json.dumps({"model": "m"}))

        d = inspect_source(str(folder))
        assert d.readiness is Readiness.BLOCKED
        assert "did not save any weights" in d.detail

    def test_missing_base_model_is_blocked(self, tmp_path):
        folder = tmp_path / "adapters"
        folder.mkdir()
        (folder / "adapter_config.json").write_text(json.dumps({"iters": 100}))
        _write(folder / "adapters.safetensors")

        d = inspect_source(str(folder))
        assert d.readiness is Readiness.BLOCKED
        assert "base model is unknown" in d.title

    def test_pointing_at_the_config_file_resolves_to_its_folder(self, tmp_path):
        folder = _mlx_adapter_dir(tmp_path)
        d = inspect_source(str(folder / "adapter_config.json"))
        assert d.kind is SourceKind.MLX_ADAPTERS
        assert d.readiness is Readiness.NEEDS_PREPARE


class TestPeftAdapters:
    def test_peft_is_recognised_but_not_auto_preparable(self, tmp_path):
        folder = tmp_path / "checkpoint-500"
        folder.mkdir()
        (folder / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "meta-llama/Llama-3-8B"})
        )
        _write(folder / "adapter_model.safetensors")

        d = inspect_source(str(folder))
        assert d.kind is SourceKind.PEFT_ADAPTERS
        assert d.readiness is Readiness.BLOCKED
        assert d.base_model == "meta-llama/Llama-3-8B"


class TestGguf:
    def test_a_gguf_file(self, tmp_path):
        f = _write(tmp_path / "model.gguf", "x" * 1024)
        d = inspect_source(str(f))

        assert d.kind is SourceKind.GGUF_FILE
        assert d.readiness is Readiness.NEEDS_PREPARE
        assert d.suggested_name == "model"

    def test_folder_containing_a_gguf_picks_the_largest(self, tmp_path):
        _write(tmp_path / "small-q2.gguf", "x" * 10)
        _write(tmp_path / "big-q8.gguf", "x" * 5000)

        d = inspect_source(str(tmp_path))
        assert d.kind is SourceKind.GGUF_FILE
        assert d.resolved_path.endswith("big-q8.gguf")

    def test_suggested_name_is_tag_safe(self, tmp_path):
        f = _write(tmp_path / "My Model v2 (Q4_K_M).gguf")
        d = inspect_source(str(f))
        assert " " not in d.suggested_name
        assert "(" not in d.suggested_name


class TestModelDir:
    def test_full_model_folder(self, tmp_path):
        folder = tmp_path / "fused"
        folder.mkdir()
        (folder / "config.json").write_text(
            json.dumps({"architectures": ["MistralForCausalLM"]})
        )
        _write(folder / "model.safetensors")

        d = inspect_source(str(folder))
        assert d.kind is SourceKind.MODEL_DIR
        assert d.readiness is Readiness.NEEDS_PREPARE
        assert "MistralForCausalLM" in d.title

    def test_quantized_folder_is_flagged(self, tmp_path):
        folder = tmp_path / "fused"
        folder.mkdir()
        (folder / "config.json").write_text(
            json.dumps({"quantization": {"bits": 4}})
        )
        _write(folder / "model.safetensors")

        d = inspect_source(str(folder))
        assert any("quantized" in w for w in d.warnings)


class TestServerUrl:
    def test_plain_url(self):
        d = inspect_source("https://my-model.run.app/v1")
        assert d.kind is SourceKind.SERVER_URL
        assert d.readiness is Readiness.READY
        assert d.spec == "https://my-model.run.app/v1"

    def test_pasted_chat_route_is_trimmed_to_the_api_root(self):
        # People copy the endpoint out of their own client code.
        d = inspect_source("https://my-model.run.app/v1/chat/completions")
        assert d.spec == "https://my-model.run.app/v1"

    def test_trailing_slash(self):
        assert inspect_source("http://localhost:8080/v1/").spec == (
            "http://localhost:8080/v1"
        )


class TestOllamaTags:
    def test_installed_tag_is_ready(self):
        d = inspect_source("qwen2:0.5b", known_ollama_tags={"qwen2:0.5b"})
        assert d.kind is SourceKind.OLLAMA_TAG
        assert d.spec == "ollama:qwen2:0.5b"

    def test_unknown_tag_is_explained_not_crashed(self):
        d = inspect_source("nope", known_ollama_tags={"qwen2:0.5b"})
        assert d.readiness is Readiness.BLOCKED
        assert "No model named nope" in d.title


class TestGuidance:
    def test_parent_of_an_adapter_folder_points_inward(self, tmp_path):
        _mlx_adapter_dir(tmp_path)
        d = inspect_source(str(tmp_path))
        assert "just inside" in d.title
        assert "adapters_v2" in d.detail

    def test_nonexistent_path(self, tmp_path):
        d = inspect_source(str(tmp_path / "ghost"))
        assert d.readiness is Readiness.BLOCKED
        assert "does not exist" in d.detail

    def test_empty_folder(self, tmp_path):
        d = inspect_source(str(tmp_path))
        assert "No model found" in d.title

    @pytest.mark.parametrize("text", ["", "   ", None])
    def test_empty_input(self, text):
        d = inspect_source(text)
        assert d.readiness is Readiness.BLOCKED
        assert d.kind is SourceKind.UNKNOWN

    def test_quoted_path_from_a_drag_and_drop(self, tmp_path):
        # macOS pastes quoted paths when a name contains spaces.
        folder = _mlx_adapter_dir(tmp_path)
        d = inspect_source(f"'{folder}'")
        assert d.kind is SourceKind.MLX_ADAPTERS

    def test_unrelated_file(self, tmp_path):
        f = _write(tmp_path / "notes.txt")
        d = inspect_source(str(f))
        assert d.readiness is Readiness.BLOCKED
        assert "not a model file" in d.title
