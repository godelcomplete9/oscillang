"""SkillStore: JSON files under data/skills/, indexed by embedding."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oscillang.skills.model import EmbeddingModel, MiniSkill

__all__ = ["SkillStore"]


class SkillStore:
    """Persist mini skills as JSON and serve top-K search by embedding."""

    def __init__(self, base_dir: Path | str, embedder: EmbeddingModel) -> None:
        self.dir = Path(base_dir) / "data" / "skills"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._skills: dict[str, MiniSkill] = {}
        self._by_osc: dict[str, MiniSkill] = {}
        self._load_all()
        self._matrix: np.ndarray | None = None
        self._ids: list[str] = []

    # --- persistence ----------------------------------------------------------
    def _path_for(self, skill_id: str) -> Path:
        safe = skill_id.replace("/", "_")
        return self.dir / f"{safe}.json"

    def _load_all(self) -> None:
        for path in sorted(self.dir.glob("*.json")):
            skill = MiniSkill.model_validate(json.loads(path.read_text(encoding="utf-8")))
            self._skills[skill.id] = skill
            self._by_osc[skill.oscillator_id] = skill

    def save(self, skill: MiniSkill) -> Path:
        self._skills[skill.id] = skill
        self._by_osc[skill.oscillator_id] = skill
        path = self._path_for(skill.id)
        path.write_text(skill.model_dump_json(indent=2), encoding="utf-8")
        self._matrix = None  # index is stale
        return path

    def get(self, skill_id: str) -> MiniSkill | None:
        return self._skills.get(skill_id)

    def get_by_oscillator(self, oscillator_id: str) -> MiniSkill | None:
        return self._by_osc.get(oscillator_id)

    def count(self) -> int:
        return len(self._skills)

    def all_skills(self) -> list[MiniSkill]:
        return list(self._skills.values())

    # --- embedding index ------------------------------------------------------
    def _build_index(self) -> tuple[np.ndarray, list[str]]:
        if self._matrix is None:
            skills = self.all_skills()
            texts = [f"{s.trigger} {s.prompt_fragment} {' '.join(s.examples)}" for s in skills]
            self._ids = [s.id for s in skills]
            if texts:
                self._matrix = self.embedder.embed_texts(texts)
            else:
                self._matrix = np.zeros((0, self.embedder.dim), np.float32)
        return self._matrix, self._ids

    def search(self, query_text: str, k: int = 5) -> list[tuple[MiniSkill, float]]:
        """Batched top-K mini skills by embedding similarity."""
        mat, ids = self._build_index()
        if mat.shape[0] == 0:
            return []
        q = self.embedder.embed_texts([query_text])[0]
        qn = q / max(float(np.linalg.norm(q)), 1e-8)
        mn = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-8)
        scores = mn @ qn
        k = min(k, scores.shape[0])
        part = np.argpartition(-scores, k - 1)[:k]
        order = np.argsort(-scores[part])
        out: list[tuple[MiniSkill, float]] = []
        for idx in part[order]:
            skill = self._skills[ids[int(idx)]]
            out.append((skill, float(scores[int(idx)])))
        return out
