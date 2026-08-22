"""Tests for receiving a model folder from a browser.

Filenames here are attacker-controlled: `webkitRelativePath` is whatever the
client says it is. Every one of these tests exists because getting it wrong
means writing a file somewhere it does not belong.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.providers import uploads  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_uploads(tmp_path, monkeypatch):
    folder = tmp_path / "uploads"
    monkeypatch.setattr(uploads, "UPLOADS_DIR", folder)
    return folder


class TestSafeComponent:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("adapters.safetensors", "adapters.safetensors"),
            ("adapter_config.json", "adapter_config.json"),
            ("My Model v2", "My-Model-v2"),
        ],
    )
    def test_keeps_ordinary_names(self, raw, expected):
        assert uploads.safe_component(raw) == expected

    @pytest.mark.parametrize("raw", ["..", ".", "", "   ", "/", "../.."])
    def test_traversal_and_empties_collapse_to_something_inert(self, raw):
        result = uploads.safe_component(raw)
        assert ".." not in result
        assert "/" not in result
        assert result

    def test_strips_separators(self):
        assert "/" not in uploads.safe_component("a/b/c")
        assert "\\" not in uploads.safe_component("a\\b")

    def test_null_byte(self):
        assert "\x00" not in uploads.safe_component("evil\x00.json")


class TestSafeRelativePath:
    def test_drops_the_picked_folder_and_keeps_the_interior(self):
        # The browser sends "adapters_v2/adapters.safetensors"; the upload
        # directory itself becomes the model root.
        assert uploads.safe_relative_path(
            "adapters_v2/adapters.safetensors"
        ) == Path("adapters.safetensors")

    def test_keeps_nested_structure_below_the_root(self):
        assert uploads.safe_relative_path(
            "model/nested/weights.safetensors"
        ) == Path("nested/weights.safetensors")

    def test_bare_filename(self):
        assert uploads.safe_relative_path("model.gguf") == Path("model.gguf")

    @pytest.mark.parametrize(
        "attack",
        [
            "../../../../etc/passwd",
            "adapters/../../../../etc/passwd",
            "/etc/passwd",
            "....//....//etc/passwd",
            "..\\..\\windows\\system32\\evil.dll",
        ],
    )
    def test_traversal_never_escapes(self, attack, tmp_path):
        relative = uploads.safe_relative_path(attack)
        assert not relative.is_absolute()
        assert ".." not in relative.parts

        # The decisive check: joining it stays inside the upload directory.
        root = tmp_path / "upload"
        resolved = (root / relative).resolve()
        assert root.resolve() in resolved.parents or resolved.parent == root.resolve()

    def test_unnameable_input_is_rejected(self):
        with pytest.raises(uploads.UploadError):
            uploads.safe_relative_path("")


class TestIsUseful:
    @pytest.mark.parametrize(
        "name",
        ["adapters.safetensors", "adapter_config.json", "model.gguf",
         "tokenizer.model", "chat_template.jinja"],
    )
    def test_model_files_are_kept(self, name):
        assert uploads.is_useful(Path(name)) is True

    @pytest.mark.parametrize(
        "name",
        ["0000200_adapters.safetensors", "0000400_adapters.safetensors"],
    )
    def test_intermediate_checkpoints_are_skipped(self, name):
        # These are superseded by the final adapters.safetensors. Uploading
        # them triples the transfer for nothing.
        assert uploads.is_useful(Path(name)) is False

    @pytest.mark.parametrize(
        "name", ["notes.md", "README", "train.log", "image.png", "script.sh"]
    )
    def test_unrelated_files_are_skipped(self, name):
        assert uploads.is_useful(Path(name)) is False

    def test_hidden_files_are_skipped(self):
        assert uploads.is_useful(Path(".DS_Store")) is False


class TestUploadDirs:
    def test_new_upload_dir_is_unique(self):
        a = uploads.new_upload_dir()
        b = uploads.new_upload_dir()
        assert a != b
        assert a.is_dir() and b.is_dir()

    def test_prune_removes_old_and_keeps_recent(self):
        old = uploads.new_upload_dir()
        fresh = uploads.new_upload_dir()
        # Backdate one by two days.
        past = time.time() - 48 * 3600
        import os

        os.utime(old, (past, past))

        removed = uploads.prune_uploads(keep_hours=24)
        assert removed == 1
        assert not old.exists()
        assert fresh.exists()

    def test_prune_on_a_missing_directory_is_a_no_op(self, tmp_path, monkeypatch):
        monkeypatch.setattr(uploads, "UPLOADS_DIR", tmp_path / "nope")
        assert uploads.prune_uploads() == 0


class TestLimits:
    def test_adapter_sized_upload_fits_comfortably(self):
        # A rank-8 LoRA for a 7B model is ~21 MB. The cap must not obstruct
        # the primary use case.
        assert 21 * 1024 * 1024 < uploads.MAX_UPLOAD_BYTES

    def test_cap_excludes_full_model_weights(self):
        # A 7B model at 4-bit is ~4 GB; uploading that through a browser is
        # not a workflow worth supporting.
        assert uploads.MAX_UPLOAD_BYTES < 4 * 1024**3
