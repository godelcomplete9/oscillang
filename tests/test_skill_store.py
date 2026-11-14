"""Exhaustive tests for SkillStore persistence and search."""

from __future__ import annotations

from pathlib import Path

import pytest

from oscillang.skills.model import HashingEmbedder, MiniSkill
from oscillang.skills.store import SkillStore


def make_skill(i: int, osc_id: str | None = None) -> MiniSkill:
    return MiniSkill(
        id=f"MS-{i:04d}",
        oscillator_id=osc_id or f"OSC-{i:04d}",
        domain=f"D{i % 12 + 1:02d}",
        func="Voice",
        trigger=f"trigger {i}",
        prompt_fragment=f"question to clarify topic {i}",
        examples=[f"example {i} one", f"example {i} two"],
        constraints=["maxAdjectivesPerSentence: 1", "sentenceLengthRule: 8 to 14 words"],
        banned_phrases=["In conclusion", "As an AI", "It is important to note"],
        evaluation_check=f"Check {i}.",
    )


@pytest.fixture()
def store(tmp_path: Path) -> SkillStore:
    return SkillStore(tmp_path, HashingEmbedder(dim=64))


class TestPersistence:
    def test_save_creates_json_file(self, store: SkillStore, tmp_path: Path) -> None:
        path = store.save(make_skill(1))
        assert path.exists()
        assert path.parent == tmp_path / "data" / "skills"

    def test_reload_from_disk(self, store: SkillStore, tmp_path: Path) -> None:
        store.save(make_skill(7, osc_id="OSC-7"))
        reopened = SkillStore(tmp_path, HashingEmbedder(dim=16))
        assert reopened.count() == 1
        skill = reopened.get("MS-0007")
        assert skill is not None
        assert skill.oscillator_id == "OSC-7"

    def test_save_overwrites_same_id(self, store: SkillStore) -> None:
        store.save(make_skill(3))
        updated = make_skill(3, osc_id="OSC-3b")
        store.save(updated)
        assert store.count() == 1
        assert store.get("MS-0003").oscillator_id == "OSC-3b"  # type: ignore[union-attr]

    def test_get_missing_returns_none(self, store: SkillStore) -> None:
        assert store.get("nope") is None
        assert store.get_by_oscillator("nope") is None

    def test_get_by_oscillator(self, store: SkillStore) -> None:
        store.save(make_skill(1, osc_id="OSC-A"))
        store.save(make_skill(2, osc_id="OSC-B"))
        assert store.get_by_oscillator("OSC-B").id == "MS-0002"  # type: ignore[union-attr]

    def test_all_skills_and_count(self, store: SkillStore) -> None:
        for i in range(5):
            store.save(make_skill(i))
        assert store.count() == 5
        assert len(store.all_skills()) == 5


class TestSearch:
    def test_search_ranks_similar_first(self, store: SkillStore) -> None:
        for i in range(6):
            skill = make_skill(i)
            if i == 4:
                skill = skill.model_copy(
                    update={"prompt_fragment": "dissect to clarify the missing page"}
                )
            store.save(skill)
        results = store.search("dissect to clarify the missing page", k=3)
        assert len(results) == 3
        assert results[0][0].id == "MS-0004"
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_k_larger_than_store(self, store: SkillStore) -> None:
        store.save(make_skill(1))
        assert len(store.search("anything", k=10)) == 1

    def test_search_on_empty_store(self, store: SkillStore) -> None:
        assert store.search("anything", k=5) == []

    def test_index_invalidated_after_save(self, store: SkillStore) -> None:
        store.save(make_skill(1))
        fresh = make_skill(2).model_copy(update={"prompt_fragment": "confess to reveal the wound"})
        store.save(fresh)
        results = store.search("confess to reveal the wound", k=1)
        assert results[0][0].id == "MS-0002"

    def test_scores_bounded(self, store: SkillStore) -> None:
        for i in range(4):
            store.save(make_skill(i))
        for _, score in store.search("question to clarify topic 1", k=4):
            assert -1.001 <= score <= 1.001
