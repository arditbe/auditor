"""Tests for the prepared-model registry and base-model resolution.

Fusing itself needs mlx-lm and gigabytes of weights, so it is not exercised
here. What is tested is everything around it: the bookkeeping that decides
which model the user sees, and the path resolution that made fusing work at
all on a real Hugging Face cache.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.providers import prepared  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Point the registry at a temp directory for every test."""
    home = tmp_path / "auditor-home"
    models = home / "models"
    models.mkdir(parents=True)
    monkeypatch.setattr(prepared, "AUDITOR_HOME", home)
    monkeypatch.setattr(prepared, "MODELS_DIR", models)
    monkeypatch.setattr(prepared, "REGISTRY_PATH", home / "registry.json")
    return home


def _model(tmp_path, name="ainara", **kwargs) -> prepared.PreparedModel:
    path = prepared.MODELS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": "mistral"}))
    return prepared.PreparedModel(
        name=name,
        path=str(path),
        kind=kwargs.pop("kind", "fused"),
        created_at=kwargs.pop("created_at", 1000.0),
        **kwargs,
    )


class TestRegistry:
    def test_round_trip(self, tmp_path):
        model = _model(tmp_path, base_model="mistralai/Mistral-7B")
        prepared.register(model)

        loaded = prepared.get_prepared("ainara")
        assert loaded is not None
        assert loaded.base_model == "mistralai/Mistral-7B"
        assert loaded.spec == "prepared:ainara"

    def test_survives_a_reload(self, tmp_path):
        prepared.register(_model(tmp_path))
        # A second process reads the same file.
        assert [m.name for m in prepared.list_prepared()] == ["ainara"]

    def test_newest_first(self, tmp_path):
        prepared.register(_model(tmp_path, name="old", created_at=1.0))
        prepared.register(_model(tmp_path, name="new", created_at=2.0))
        assert [m.name for m in prepared.list_prepared()] == ["new", "old"]

    def test_entries_whose_files_vanished_are_hidden(self, tmp_path):
        model = _model(tmp_path)
        prepared.register(model)
        # The user deleted the folder behind our back.
        import shutil

        shutil.rmtree(model.path)
        assert prepared.list_prepared() == []
        assert prepared.get_prepared("ainara") is None

    def test_unregister(self, tmp_path):
        prepared.register(_model(tmp_path))
        assert prepared.unregister("ainara") is True
        assert prepared.get_prepared("ainara") is None

    def test_unregister_unknown_is_false(self):
        assert prepared.unregister("ghost") is False

    def test_delete_files_removes_our_directory(self, tmp_path):
        model = _model(tmp_path)
        prepared.register(model)
        prepared.unregister("ainara", delete_files=True)
        assert not Path(model.path).exists()

    def test_delete_files_refuses_paths_outside_our_directory(self, tmp_path):
        """A registered folder we do not own must never be deleted."""
        outside = tmp_path / "users-own-model"
        outside.mkdir()
        (outside / "config.json").write_text("{}")
        prepared.register(
            prepared.PreparedModel(
                name="theirs", path=str(outside), kind="folder", created_at=1.0
            )
        )

        prepared.unregister("theirs", delete_files=True)
        assert outside.exists(), "must not delete a folder outside MODELS_DIR"

    def test_corrupt_registry_does_not_crash(self):
        prepared.REGISTRY_PATH.write_text("{{ not json")
        assert prepared.list_prepared() == []


class TestUniqueName:
    def test_plain_name_when_free(self):
        assert prepared._unique_name("ainara") == "ainara"

    def test_suffixes_on_collision(self, tmp_path):
        prepared.register(_model(tmp_path, name="ainara"))
        assert prepared._unique_name("ainara") == "ainara-2"

    def test_skips_a_directory_that_exists_without_a_registry_entry(self):
        (prepared.MODELS_DIR / "ainara").mkdir()
        assert prepared._unique_name("ainara") == "ainara-2"


class TestResolveBaseModel:
    def test_local_directory_passes_through(self, tmp_path):
        folder = tmp_path / "base"
        folder.mkdir()
        assert prepared.resolve_base_model(str(folder)) == str(folder)

    def test_repo_id_resolves_to_its_cached_snapshot(self, tmp_path, monkeypatch):
        # This is the case that made fusing work: mlx_lm's own offline lookup
        # rejects a snapshot missing a README, but the weights are fine.
        hub = tmp_path / "hub"
        snapshot = (
            hub / "models--mlx-community--Mistral-7B-Instruct-v0.3-4bit"
            / "snapshots" / "abc123"
        )
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
        monkeypatch.setenv("HF_HOME", str(hub))

        resolved = prepared.resolve_base_model(
            "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
        )
        assert resolved == str(snapshot)

    def test_snapshot_without_a_config_is_not_used(self, tmp_path, monkeypatch):
        hub = tmp_path / "hub"
        snapshot = hub / "models--org--model" / "snapshots" / "abc"
        snapshot.mkdir(parents=True)
        monkeypatch.setenv("HF_HOME", str(hub))

        # Falls back to the repo id so mlx_lm can try downloading it.
        assert prepared.resolve_base_model("org/model") == "org/model"

    def test_uncached_repo_id_passes_through(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(tmp_path / "empty"))
        assert prepared.resolve_base_model("org/model") == "org/model"


class TestExportGuards:
    """Export is expensive; it must refuse clearly rather than half-run."""

    async def test_unknown_model(self):
        with pytest.raises(prepared.PrepareError, match="No prepared model"):
            await prepared.export_to_ollama("ghost")

    async def test_model_without_adapters_cannot_be_reexported(self, tmp_path):
        prepared.register(_model(tmp_path, kind="folder"))
        with pytest.raises(prepared.PrepareError, match="not built from adapters"):
            await prepared.export_to_ollama("ainara")

    async def test_unsupported_architecture_is_refused(self, tmp_path):
        model = _model(
            tmp_path, base_model="org/base", adapter_path=str(tmp_path / "ad")
        )
        (Path(model.path) / "config.json").write_text(
            json.dumps({"model_type": "gemma"})
        )
        prepared.register(model)

        with pytest.raises(prepared.PrepareError, match="Ollama export supports"):
            await prepared.export_to_ollama("ainara")

    async def test_refuses_when_disk_is_short(self, tmp_path, monkeypatch):
        model = _model(
            tmp_path, base_model="org/base", adapter_path=str(tmp_path / "ad")
        )
        prepared.register(model)
        monkeypatch.setattr(prepared, "_dir_size", lambda p: 4_000_000_000)
        monkeypatch.setattr(prepared, "_free_bytes", lambda p: 1_000_000_000)

        with pytest.raises(prepared.PrepareError, match="Not enough disk space"):
            await prepared.export_to_ollama("ainara")


class TestFuseFailureMessages:
    """mlx_lm tracebacks are unreadable; users get a sentence instead."""

    def test_incomplete_download(self):
        msg = prepared._explain_fuse_failure(
            "huggingface_hub.errors.IncompleteSnapshotError: ...", "org/m"
        )
        assert "not fully downloaded" in msg

    def test_shape_mismatch(self):
        msg = prepared._explain_fuse_failure("RuntimeError: size mismatch", "org/m")
        assert "do not match that base model" in msg

    def test_missing_mlx(self):
        msg = prepared._explain_fuse_failure("No module named 'mlx_lm'", "org/m")
        assert "pip install mlx-lm" in msg

    def test_unknown_error_keeps_the_last_line(self):
        msg = prepared._explain_fuse_failure("boom\nSomething specific", "org/m")
        assert "Something specific" in msg
