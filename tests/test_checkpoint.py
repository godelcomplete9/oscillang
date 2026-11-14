"""Exhaustive tests for the build checkpoint."""

from __future__ import annotations

from pathlib import Path

import pytest

from oscillang.skills.checkpoint import (
    Checkpoint,
    checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture()
def base_dir(tmp_path: Path) -> Path:
    return tmp_path


class TestCheckpointRoundtrip:
    def test_save_and_load(self, base_dir: Path) -> None:
        ckpt = Checkpoint(
            lattice_id="lat",
            last_batch_index=4,
            completed_batches=[0, 1, 2, 3, 4],
            total_batches=100,
            total_oscillators=100,
            total_mini_skills=97,
            merged_duplicates=3,
            flagged_for_review=2,
        )
        save_checkpoint(ckpt, base_dir)
        loaded = load_checkpoint(base_dir, "lat")
        assert loaded == ckpt

    def test_path_layout(self, base_dir: Path) -> None:
        path = checkpoint_path(base_dir, "my-lattice")
        assert path == base_dir / "data" / "lattices" / "my-lattice" / "checkpoint.json"

    def test_load_missing_returns_none(self, base_dir: Path) -> None:
        assert load_checkpoint(base_dir, "ghost") is None

    def test_defaults(self) -> None:
        ckpt = Checkpoint(lattice_id="x")
        assert ckpt.last_batch_index == -1
        assert ckpt.completed_batches == []
        assert ckpt.total_oscillators == 0
        assert ckpt.total_mini_skills == 0

    def test_overwrite(self, base_dir: Path) -> None:
        save_checkpoint(Checkpoint(lattice_id="lat", last_batch_index=1), base_dir)
        save_checkpoint(Checkpoint(lattice_id="lat", last_batch_index=2), base_dir)
        assert load_checkpoint(base_dir, "lat").last_batch_index == 2  # type: ignore[union-attr]
