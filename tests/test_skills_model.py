"""Exhaustive tests for skills/model.py: ontology, spec, hz tables, embedder."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from oscillang.skills.model import (
    CODE_FUNC_MAP,
    INSTANTIATION_FUNC_HZ,
    MODE_VOCAB,
    MOVE_VOCAB,
    PURPOSE_VOCAB,
    RADII,
    STANCE_VOCAB,
    HashingEmbedder,
    InstantiationSpec,
    MiniSkill,
    Ontology,
    Reference,
)

PREMISE = "A burned-out translator rediscovers her voice"


def make_spec(**overrides) -> InstantiationSpec:
    base = dict(
        premise=PREMISE,
        references=[Reference(name="Ref", sample_text="Some sample text.")],
        guiding_words=["frayed", "luminous", "stubborn"],
        radius=1,
    )
    base.update(overrides)
    return InstantiationSpec(**base)


class TestOntologySkeleton:
    def test_one_root(self) -> None:
        onto = Ontology.skeleton(PREMISE)
        assert onto.count_at_level(0) == 1
        assert onto.nodes[onto.root_id].label == PREMISE

    def test_exactly_12_domains(self) -> None:
        onto = Ontology.skeleton(_PREMISS := PREMISE)
        domains = onto.children_of(onto.root_id)
        assert len(domains) == 12
        assert [d.label.split(" ")[0] for d in domains] == [f"D{i:02d}" for i in range(1, 13)]

    def test_14_funcs_per_domain_168_total(self) -> None:
        onto = Ontology.skeleton(_PREMISS := PREMISE)
        assert onto.count_at_level(2) == 12 * 14 == 168
        for domain in onto.children_of(onto.root_id):
            assert len(onto.children_of(domain.id)) == 14

    def test_every_domain_decomposes_into_all_funcs(self) -> None:
        onto = Ontology.skeleton(_PREMISS := PREMISE)
        func_labels = set(INSTANTIATION_FUNC_HZ)
        for domain in onto.children_of(onto.root_id):
            labels = {n.label for n in onto.children_of(domain.id)}
            assert labels == func_labels

    def test_l2_nodes_link_to_parents(self) -> None:
        onto = Ontology.skeleton(_PREMISS := PREMISE)
        for node in onto.nodes.values():
            if node.parent_id is not None:
                assert node.id in onto.nodes[node.parent_id].children


class TestInstantiationSpec:
    def test_valid_spec(self) -> None:
        spec = make_spec()
        assert spec.radius == 1
        assert spec.max_concurrent_batches == 3

    def test_radius_bounds(self) -> None:
        with pytest.raises(ValidationError):
            make_spec(radius=0)
        with pytest.raises(ValidationError):
            make_spec(radius=11)

    def test_reference_count_bounds(self) -> None:
        with pytest.raises(ValidationError):
            make_spec(references=[])
        with pytest.raises(ValidationError):
            make_spec(references=[Reference(name=f"r{i}", sample_text="x") for i in range(6)])

    def test_guiding_word_bounds(self) -> None:
        with pytest.raises(ValidationError):
            make_spec(guiding_words=["one", "two"])
        with pytest.raises(ValidationError):
            make_spec(guiding_words=[str(i) for i in range(8)])

    def test_concurrency_cap_of_three(self) -> None:
        with pytest.raises(ValidationError):
            make_spec(max_concurrent_batches=4)


class TestRadiiAndHz:
    def test_radius_table_matches_spec(self) -> None:
        assert RADII == {
            1: 2_000,
            2: 5_000,
            3: 12_000,
            4: 20_000,
            5: 50_000,
            6: 100_000,
            7: 200_000,
            8: 300_000,
            9: 400_000,
            10: 500_000,
        }

    def test_hz_table_complete_and_in_range(self) -> None:
        assert set(INSTANTIATION_FUNC_HZ) == {
            "Voice",
            "Character Voice",
            "Emotion",
            "Flaw",
            "Want vs Need",
            "Worldview",
            "Rhythm",
            "Image",
            "Scene Objective",
            "Other Characters",
            "World Facts",
            "Plot Pressure",
            "Reader Tension",
            "Memory",
        }
        for hz in INSTANTIATION_FUNC_HZ.values():
            assert 0.01 <= hz <= 1.5

    def test_slow_funcs_hold_worldview(self) -> None:
        assert INSTANTIATION_FUNC_HZ["Worldview"] < 0.1
        assert INSTANTIATION_FUNC_HZ["Voice"] > 0.9

    def test_code_func_map_covers_all_funcs(self) -> None:
        assert set(CODE_FUNC_MAP) == set(INSTANTIATION_FUNC_HZ)
        assert CODE_FUNC_MAP["Voice"] == "Code Voice"
        assert CODE_FUNC_MAP["Flaw"] == "Failure Mode"
        assert len(set(CODE_FUNC_MAP.values())) == 14

    def test_vocab_sizes(self) -> None:
        assert len(MODE_VOCAB) == 5
        assert len(STANCE_VOCAB) == 3
        assert len(MOVE_VOCAB) == 12
        assert len(PURPOSE_VOCAB) == 7


class TestMiniSkill:
    def test_defaults(self) -> None:
        skill = MiniSkill(id="MS-X", oscillator_id="OSC-X", domain="D01", func="Voice")
        assert skill.composed is False
        assert skill.promoted is False
        assert skill.used_count == 0
        assert skill.flagged_for_review is False

    def test_roundtrip_json(self) -> None:
        skill = MiniSkill(
            id="MS-X",
            oscillator_id="OSC-X",
            domain="D01",
            func="Voice",
            trigger="t",
            prompt_fragment="edit to test",
            examples=["a", "b"],
            constraints=["c1", "c2"],
            banned_phrases=["b1", "b2", "b3"],
            evaluation_check="check",
        )
        revived = MiniSkill.model_validate_json(skill.model_dump_json())
        assert revived == skill


class TestHashingEmbedder:
    def test_deterministic_across_instances(self) -> None:
        texts = ["dissect | clarify", "joke | hide"]
        e1 = HashingEmbedder(dim=32).embed_texts(texts)
        e2 = HashingEmbedder(dim=32).embed_texts(texts)
        assert np.array_equal(e1, e2)

    def test_unit_norm_and_finite(self) -> None:
        emb = HashingEmbedder(dim=32).embed_texts(["hello world", "another text"])
        assert emb.shape == (2, 32)
        assert np.all(np.isfinite(emb))
        assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)

    def test_different_texts_differ_same_texts_equal(self) -> None:
        embedder = HashingEmbedder(dim=64)
        emb = embedder.embed_texts(["alpha beta", "alpha beta", "gamma delta"])
        assert np.allclose(emb[0], emb[1])
        assert not np.allclose(emb[0], emb[2])

    def test_similar_texts_more_similar_than_disparate(self) -> None:
        embedder = HashingEmbedder(dim=64)
        emb = embedder.embed_texts(
            ["dissect to clarify precisely", "dissect to clarify exactly", "joke to hide"]
        )
        sim_close = float(emb[0] @ emb[1])
        sim_far = float(emb[0] @ emb[2])
        assert sim_close > sim_far

    def test_empty_text_is_zero_vector(self) -> None:
        emb = HashingEmbedder(dim=16).embed_texts([""])
        assert np.allclose(emb, 0.0)

    def test_token_repetition_keeps_direction(self) -> None:
        embedder = HashingEmbedder(dim=32)
        emb = embedder.embed_texts(["word", "word word word"])
        # bigrams differ, but the shared unigram keeps the vectors aligned
        assert float(emb[0] @ emb[1]) > 0.7
        assert np.isclose(np.linalg.norm(emb[1]), 1.0, atol=1e-5)
