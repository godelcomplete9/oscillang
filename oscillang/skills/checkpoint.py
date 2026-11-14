"""Build checkpoint: resumable, no restart from zero after interruption."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

__all__ = ["Checkpoint", "checkpoint_path", "load_checkpoint", "save_checkpoint"]


class Checkpoint(BaseModel):
    lattice_id: str
    last_batch_index: int = -1
    completed_batches: list[int] = Field(default_factory=list)
    total_batches: int = 0
    total_oscillators: int = 0
    total_mini_skills: int = 0
    merged_duplicates: int = 0
    flagged_for_review: int = 0


def checkpoint_path(base_dir: Path | str, lattice_id: str) -> Path:
    return Path(base_dir) / "data" / "lattices" / lattice_id / "checkpoint.json"


def save_checkpoint(ckpt: Checkpoint, base_dir: Path | str) -> Path:
    path = checkpoint_path(base_dir, ckpt.lattice_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ckpt.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_checkpoint(base_dir: Path | str, lattice_id: str) -> Checkpoint | None:
    path = checkpoint_path(base_dir, lattice_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return Checkpoint.model_validate(data)
