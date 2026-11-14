"""Exhaustive tests for the Crafter: hierarchy, leaves, crafting, resume, coupling.

Deterministic end to end: FakeChatModel everywhere, seeded rng, tmp dirs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from oscillang.llm.fake import FakeChatModel
from oscillang.skills.checkpoint import load_checkpoint
from oscillang.skills.crafter import (
    BuildValidationError,
    Crafter,
    OscRecord,
    _clamp_expand,
    _default_expansion,
    validate_skill_fields,
)
from oscillang.skills.model import (
    CODE_FUNC_MAP,
    INSTANTIATION_FUNC_HZ,
    HashingEmbedder,
    InstantiationSpec,
    Ontology,
    Reference,
)

PREMISE = "A burned-out translator rediscovers her voice in a collapsing city"
WORDS = ["frayed", "luminous", "stubborn"]


def make_spec(tmp_path: Path, **overrides) -> InstantiationSpec:
    base = dict(
        premise=PREMISE,
        references=[
            Reference(
                name="Sample Author",
                sample_text=(
                    "She asked a short question. The page held its breath. Nothing here is summary."
                ),
            )
        ],
        guiding_words=WORDS,
        radius=1,
        lattice_id="lat",
    )
    base.update(overrides)
    return InstantiationSpec(**base)


def craft_responder(messages):
    """Dispatches: craft batches -> mini skill JSON; expansion prompts -> expansion JSON."""
    text = messages[-1].content
    marker = re.search(r"OSCILLATOR_BATCH_JSON: (.*) END_OSCILLATOR_BATCH_JSON", text, re.DOTALL)
    if marker is None:
        return expansion_responder(messages)
    payload = json.loads(marker.group(1))
    out = []
    for item in payload:
        p = item["path"]
        fragment = (
            f"{p['move']} to {p['purpose']}; keep the {p['mode']} register aimed {p['stance']}."
        )
        out.append(
            {
                "oscillatorId": item["oscillatorId"],
                "trigger": f"When {p['func']} in {p['domain']} needs {p['purpose']}",
                "promptFragment": fragment,
                "examples": [
                    f"[Sample Author] {p['move']} one",
                    f"[Sample Author] {p['move']} two",
                ],
                "constraints": [
                    "sentenceLengthRule: 8 to 14 words",
                    f"mustInclude: {p['purpose']}",
                ],
                "bannedPhrases": ["In conclusion", "As an AI", "It is important to note"],
                "evaluationCheck": f"Check: output {p['move']}s toward {p['stance']}.",
            }
        )
    return json.dumps(out)


def bad_fragment_responder(messages):
    """Valid JSON but adjective-only fragments -> reject -> retry -> flag."""
    text = messages[-1].content
    marker = re.search(r"OSCILLATOR_BATCH_JSON: (.*) END_OSCILLATOR_BATCH_JSON", text, re.DOTALL)
    if marker is None:
        return expansion_responder(messages)
    payload = json.loads(marker.group(1))
    return json.dumps(
        [
            {
                "oscillatorId": item["oscillatorId"],
                "trigger": "always",
                "promptFragment": "Precise, clear, beautiful and luminous.",
                "examples": ["e1", "e2"],
                "constraints": ["c1", "c2"],
                "bannedPhrases": ["b1", "b2", "b3"],
                "evaluationCheck": "check",
            }
            for item in payload
        ]
    )


def non_json_responder(messages):
    return "I cannot produce JSON today, sorry."


def expansion_responder(messages):
    """Valid per-domain expansion JSON for every func in the prompt."""
    text = messages[-1].content
    funcs = re.search(r"Funcs: (.*)", text).group(1).split(", ")
    return json.dumps(
        {
            f: {
                "modes": ["precise", "playful", "direct", "withheld", "evasive"],
                "moves": ["question", "dissect", "confess", "joke", "evade", "summarize"],
                "purposes": ["clarify", "persuade", "debug"],
            }
            for f in funcs
        }
    )


@pytest.fixture()
def crafted_crafter(tmp_path: Path) -> Crafter:
    return Crafter(
        chat_model=FakeChatModel(responder=craft_responder),
        base_dir=tmp_path,
        lattice_id="lat",
        seed=3,
    )


# --- hierarchy -----------------------------------------------------------------
class TestHierarchy:
    def test_skeleton_via_crafter(self, tmp_path: Path) -> None:
        crafter = Crafter(chat_model=FakeChatModel(), base_dir=tmp_path, lattice_id="lat")
        onto = crafter.ensure_hierarchy(make_spec(tmp_path))
        assert onto.count_at_level(0) == 1
        assert onto.count_at_level(1) == 12
        assert onto.count_at_level(2) == 168

    def test_main_brain_expansion_respected(self, tmp_path: Path) -> None:
        crafter = Crafter(
            chat_model=FakeChatModel(responder=expansion_responder),
            base_dir=tmp_path,
            lattice_id="lat",
        )
        onto = crafter.ensure_hierarchy(make_spec(tmp_path))
        for func in onto.children_of(onto.children_of(onto.root_id)[0].id):
            assert len(onto.children_of(func.id)) == 5  # max modes kept
            for mode in onto.children_of(func.id):
                assert len(onto.children_of(mode.id)) == 3
                for stance in onto.children_of(mode.id):
                    assert len(onto.children_of(stance.id)) == 6
                    for move in onto.children_of(stance.id):
                        assert len(onto.children_of(move.id)) == 3

    def test_expansion_cache_reused(self, tmp_path: Path) -> None:
        model = FakeChatModel()
        crafter = Crafter(chat_model=model, base_dir=tmp_path, lattice_id="lat")
        spec = make_spec(tmp_path)
        crafter.ensure_hierarchy(spec)
        calls_after_first = len(model.calls)
        assert (tmp_path / "data" / "lattices" / "lat" / "expansion.json").exists()
        crafter.ensure_hierarchy(spec)
        assert len(model.calls) == calls_after_first  # cache hit, no LLM calls

    def test_supplied_skeleton_gets_filled(self, tmp_path: Path) -> None:
        spec = make_spec(tmp_path, supplied_ontology=Ontology.skeleton(PREMISE))
        crafter = Crafter(chat_model=FakeChatModel(), base_dir=tmp_path, lattice_id="lat")
        onto = crafter.ensure_hierarchy(spec)
        assert onto.count_at_level(3) > 0
        assert onto.count_at_level(6) > 0
        Crafter.validate_hierarchy_shape(onto)

    def test_supplied_full_tree_used_as_is(self, tmp_path: Path) -> None:
        base = Ontology.skeleton(PREMISE)
        filler = Crafter(chat_model=FakeChatModel(), base_dir=tmp_path, lattice_id="lat")
        filler._apply_expansion(base, {})
        spec = make_spec(tmp_path, supplied_ontology=base)
        onto = filler.ensure_hierarchy(spec)
        assert onto is base

    def test_invalid_supplied_tree_fails(self, tmp_path: Path) -> None:
        onto = Ontology.skeleton(PREMISE)
        onto.nodes = onto.nodes.copy()
        bad = Ontology(root_id=onto.root_id, nodes=dict(onto.nodes))
        first_domain = bad.children_of(bad.root_id)[0]
        funcs = bad.children_of(first_domain.id)
        bad.nodes[funcs[-1].id] = funcs[-1].model_copy(update={"children": []})
        bad.nodes[first_domain.id] = first_domain.model_copy(
            update={"children": [c for c in first_domain.children if c != funcs[-1].id]}
        )
        spec = make_spec(tmp_path, supplied_ontology=bad)
        crafter = Crafter(chat_model=FakeChatModel(), base_dir=tmp_path, lattice_id="lat")
        with pytest.raises(BuildValidationError, match="14 funcs"):
            crafter.ensure_hierarchy(spec)

    def test_fallback_expansion_when_llm_prose(self, tmp_path: Path) -> None:
        crafter = Crafter(
            chat_model=FakeChatModel(responder=non_json_responder),
            base_dir=tmp_path,
            lattice_id="lat",
        )
        onto = crafter.ensure_hierarchy(make_spec(tmp_path))
        Crafter.validate_hierarchy_shape(onto)  # defaults filled the tree


class TestExpansionHelpers:
    def test_clamp_expand_enforces_limits(self) -> None:
        exp = _clamp_expand(list("abcdefghij"), list("klmnopqrstuv"), list("wxyzab"))
        assert 3 <= len(exp["modes"]) <= 5
        assert 3 <= len(exp["moves"]) <= 6
        assert 1 <= len(exp["purposes"]) <= 3
        assert exp["modes"] == ["a", "b", "c", "d", "e"]
        assert exp["moves"] == ["k", "l", "m", "n", "o", "p"]
        assert exp["purposes"] == ["w", "x", "y"]

    def test_clamp_expand_dedupes(self) -> None:
        exp = _clamp_expand(["precise", "precise", "playful"], ["q"] * 3, ["z", "z"])
        assert exp["modes"] == ["precise", "playful", "direct"]
        assert exp["moves"] == ["q", "question", "dissect"]
        assert exp["purposes"] == ["z"]  # dedupe to 1 already satisfies the minimum

    def test_clamp_expand_fills_to_minimum(self) -> None:
        exp = _clamp_expand([], [], [])
        assert len(exp["modes"]) == 3
        assert len(exp["moves"]) == 3
        assert len(exp["purposes"]) == 1

    def test_default_expansion_deterministic(self) -> None:
        assert _default_expansion(2, 3) == _default_expansion(2, 3)
        a = _default_expansion(0, 0)
        assert len(a["modes"]) == 3 and len(a["moves"]) == 3 and len(a["purposes"]) == 2


# --- leaves --------------------------------------------------------------------
class TestLeaves:
    def test_leaf_shape_and_ranges(self, tmp_path: Path) -> None:
        crafter = Crafter(chat_model=FakeChatModel(), base_dir=tmp_path, lattice_id="lat")
        onto = crafter.ensure_hierarchy(make_spec(tmp_path))
        leaves, merged, total = crafter.build_leaves(make_spec(tmp_path), onto)
        assert merged == 0
        assert total == 168 * 3 * 3 * 3 * 2  # default expansion: 3 modes, 3 moves, 2 purposes
        assert len(leaves) == 2_000
        for rec in leaves[:200]:
            assert re.fullmatch(r"OSC-D\d{2}-F\d{2}-M\d{2}-S\d{2}-MV\d{2}-P\d{2}", rec.osc_id)
            assert 0.01 <= rec.hz <= 1.5
            assert 0.0 <= rec.phase < 2.0 * np.pi
            assert np.isclose(np.linalg.norm(rec.embedding), 1.0, atol=1e-5)
            assert rec.path["root"] == PREMISE
            assert set(rec.codes) == {"domain", "func", "mode", "stance", "move", "purpose"}

    def test_subset_deterministic_across_seeds_and_runs(self, tmp_path: Path) -> None:
        def build_with(seed: int) -> tuple[list[str], np.ndarray]:
            crafter = Crafter(
                chat_model=FakeChatModel(),
                base_dir=tmp_path / f"s{seed}",
                lattice_id="lat",
                seed=seed,
            )
            spec = make_spec(tmp_path / f"s{seed}")
            onto = crafter.ensure_hierarchy(spec)
            leaves, _, _ = crafter.build_leaves(spec, onto)
            return [r.osc_id for r in leaves], np.array([r.phase for r in leaves])

        ids_a, phases_a = build_with(1)
        ids_b, _ = build_with(1)
        assert ids_a == ids_b
        # evenly-spaced subset selection is seed independent; only phase/hz jitter varies
        ids_c, phases_c = build_with(2)
        assert ids_a == ids_c
        assert not np.allclose(phases_a, phases_c)

    def test_hz_override_applied_before_jitter(self, tmp_path: Path) -> None:
        crafter = Crafter(chat_model=FakeChatModel(), base_dir=tmp_path, lattice_id="lat")
        spec = make_spec(tmp_path, hz_overrides={"Voice": 1.5})
        onto = crafter.ensure_hierarchy(spec)
        leaves, _, _ = crafter.build_leaves(spec, onto)
        voice_hz = [r.hz for r in leaves if r.path["func"] == "Voice"]
        assert voice_hz
        for hz in voice_hz:
            assert 1.35 <= hz <= 1.5  # 1.5 * (0.9..1.1), clipped to 1.5

    def test_hz_from_default_table(self, tmp_path: Path) -> None:
        crafter = Crafter(chat_model=FakeChatModel(), base_dir=tmp_path, lattice_id="lat")
        spec = make_spec(tmp_path)
        onto = crafter.ensure_hierarchy(spec)
        leaves, _, _ = crafter.build_leaves(spec, onto)
        for rec in leaves:
            base = INSTANTIATION_FUNC_HZ[rec.path["func"]]
            assert base * 0.9 <= rec.hz <= base * 1.1 + 1e-6

    def test_code_lattice_func_mapping(self, tmp_path: Path) -> None:
        crafter = Crafter(chat_model=FakeChatModel(), base_dir=tmp_path, lattice_id="lat")
        spec = make_spec(tmp_path, code_lattice=True)
        onto = crafter.ensure_hierarchy(spec)
        leaves, _, _ = crafter.build_leaves(spec, onto)
        func_labels = {r.path["func"] for r in leaves}
        assert "Code Voice" in func_labels
        assert "Failure Mode" in func_labels
        assert set(func_labels) == set(CODE_FUNC_MAP.values())

    def test_duplicate_paths_merge_with_boost(self, tmp_path: Path) -> None:
        onto = Ontology.skeleton(PREMISE)
        crafter = Crafter(chat_model=FakeChatModel(), base_dir=tmp_path, lattice_id="lat")
        # Two func NODES sharing one label receive the same expansion by label
        # lookup -> identical path texts -> merged duplicates.
        domain = onto.children_of(onto.root_id)[0]
        funcs = onto.children_of(domain.id)
        onto.nodes[funcs[1].id] = funcs[1].model_copy(update={"label": funcs[0].label})
        spec = make_spec(tmp_path, radius=1, supplied_ontology=onto)
        filled = crafter.ensure_hierarchy(spec)
        leaves, merged, _ = crafter.build_leaves(spec, filled)
        assert merged > 0
        boosted = [r for r in leaves if r.merged_boost > 0.0]
        assert boosted
        assert all(r.merged_boost <= 1.0 for r in boosted)
        # merged duplicates are dropped (one row per unique path); subset still 2000
        assert len(leaves) == 2_000


# --- mini skill validation -------------------------------------------------------
class TestSkillValidation:
    def test_valid_fragment_passes(self) -> None:
        assert (
            validate_skill_fields(
                "when X",
                "dissect to clarify; keep sentences short",
                ["a", "b"],
                ["c1", "c2"],
                ["b1", "b2", "b3"],
                "check",
            )
            == []
        )

    def test_adjective_only_rejected(self) -> None:
        errors = validate_skill_fields(
            "t",
            "Precise, clear, beautiful and luminous.",
            ["a", "b"],
            ["c1", "c2"],
            ["b1", "b2", "b3"],
            "c",
        )
        assert any("concrete" in e for e in errors)

    def test_each_reject_rule(self) -> None:
        good = dict(
            trigger="t",
            prompt_fragment="edit to test the line",
            examples=["a", "b"],
            constraints=["c1", "c2"],
            banned_phrases=["b1", "b2", "b3"],
            evaluation_check="ok",
        )
        assert validate_skill_fields(**good) == []
        cases = [
            dict(trigger=" "),
            dict(prompt_fragment=""),
            dict(prompt_fragment="word " * 90),
            dict(examples=["only"]),
            dict(constraints=["c1"]),
            dict(banned_phrases=["b1", "b2"]),
            dict(evaluation_check="  "),
        ]
        for override in cases:
            errors = validate_skill_fields(**{**good, **override})
            assert errors, override

    def test_long_but_concrete_fragment_ok_at_80(self) -> None:
        fragment = " ".join(["edit"] + [f"w{i}" for i in range(78)])
        errors = validate_skill_fields(
            "t", fragment, ["a", "b"], ["c1", "c2"], ["b1", "b2", "b3"], "c"
        )
        assert errors == []


# --- full build -----------------------------------------------------------------
class TestFullBuild:
    def test_radius1_build_end_to_end(self, crafted_crafter: Crafter, tmp_path: Path) -> None:
        result = crafted_crafter.build(make_spec(tmp_path))
        assert result.actual == 2_000
        assert result.target == 2_000
        assert result.mini_skills == 2_000
        assert result.compose_gap == 0
        assert result.batches == 100
        assert result.flagged_for_review == 0
        assert all(result.validation.values())
        # one-to-one: every oscillator carries exactly one linked skill
        n = crafted_crafter.lattice.n
        assert all(tid is not None for tid in crafted_crafter.lattice.template_id[:n])
        linked = {
            str(tid): crafted_crafter.store.get(str(tid))
            for tid in crafted_crafter.lattice.template_id[:n]
        }
        assert all(s is not None for s in linked.values())
        for i in range(n):  # skill <-> row one-to-one by oscillator id
            tid = str(crafted_crafter.lattice.template_id[i])
            oid = str(crafted_crafter.lattice.ids[i])
            skill = crafted_crafter.store.get(tid)
            assert skill is not None and skill.oscillator_id == oid

    def test_oscillator_ids_and_phase_unique(
        self, crafted_crafter: Crafter, tmp_path: Path
    ) -> None:
        crafted_crafter.build(make_spec(tmp_path))
        ids = [str(x) for x in crafted_crafter.lattice.ids[: crafted_crafter.lattice.n]]
        assert len(set(ids)) == 2_000
        phases = crafted_crafter.lattice.phase[: crafted_crafter.lattice.n]
        assert np.all((phases >= 0.0) & (phases < 2.0 * np.pi))
        assert np.allclose(crafted_crafter.lattice.amplitude[: crafted_crafter.lattice.n], 1.0)

    def test_coupling_structure(self, crafted_crafter: Crafter, tmp_path: Path) -> None:
        crafted_crafter.build(make_spec(tmp_path))
        lat = crafted_crafter.lattice
        coupling = lat.coupling
        row_sums = coupling.row_sums()
        assert np.all(row_sums > 0.0)
        dense = coupling.dense_copy()
        nnz_per_row = np.count_nonzero(dense, axis=1)
        assert np.all(nnz_per_row == 20)  # top-20 partners
        assert np.allclose(np.diag(dense), 0.0)  # no self-coupling
        # values are within the proximity vocabulary (float32-safe comparison)
        allowed = {np.float32(v) for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
        nonzero = dense[dense > 0]
        assert set(np.round(nonzero, 4)).issubset(allowed)

    def test_build_time_and_events(self, crafted_crafter: Crafter, tmp_path: Path) -> None:
        result = crafted_crafter.build(make_spec(tmp_path))
        stages = [e["stage"] for e in result.events]
        assert stages.count("batch") == 100
        assert "branching" in stages
        assert "example_trace" in stages
        assert stages[-1] == "done"
        assert result.elapsed_seconds < 120.0
        trace = next(e for e in result.events if e["stage"] == "example_trace")
        assert trace["oscillator"]["id"].startswith("OSC-D")
        assert trace["oscillator"]["templateId"] is not None
        assert trace["mini_skill"]["oscillator_id"] == trace["oscillator"]["id"]

    def test_checkpoint_after_build(self, crafted_crafter: Crafter, tmp_path: Path) -> None:
        crafted_crafter.build(make_spec(tmp_path))
        ckpt = load_checkpoint(tmp_path, "lat")
        assert ckpt is not None
        assert ckpt.completed_batches == list(range(100))
        assert ckpt.last_batch_index == 99
        assert ckpt.total_oscillators == 2_000
        assert ckpt.total_mini_skills == 2_000
        assert ckpt.total_batches == 100

    def test_llm_call_budget(self, crafted_crafter: Crafter, tmp_path: Path) -> None:
        model = crafted_crafter.chat_model
        assert isinstance(model, FakeChatModel)
        crafted_crafter.build(make_spec(tmp_path))
        # 12 hierarchy calls (fallback prose) + 100 craft batches, exactly one
        # Main Brain call per batch and never per oscillator.
        assert len(model.calls) == 12 + 100

    def test_store_files_on_disk(self, crafted_crafter: Crafter, tmp_path: Path) -> None:
        crafted_crafter.build(make_spec(tmp_path))
        files = list((tmp_path / "data" / "skills").glob("*.json"))
        assert len(files) == 2_000
        batch_files = list(
            (tmp_path / "data" / "lattices" / "lat" / "oscillators").glob("batch_*.json")
        )
        assert len(batch_files) == 100
        sample = json.loads(batch_files[0].read_text(encoding="utf-8"))[0]
        assert OscRecord.from_json(sample).osc_id == sample["osc_id"]

    def test_store_search_over_built_skills(self, crafted_crafter: Crafter, tmp_path: Path) -> None:
        crafted_crafter.build(make_spec(tmp_path))
        results = crafted_crafter.store.search("question to clarify", k=5)
        assert len(results) == 5
        assert all(isinstance(s.prompt_fragment, str) for s, _ in results)


class TestCraftingFailurePaths:
    def test_non_json_response_falls_back_to_flagged_skills(self, tmp_path: Path) -> None:
        crafter = Crafter(
            chat_model=FakeChatModel(responder=non_json_responder),
            base_dir=tmp_path,
            lattice_id="lat",
            seed=1,
        )
        result = crafter.build(make_spec(tmp_path))
        assert result.flagged_for_review == 2_000
        assert result.mini_skills == 2_000
        assert result.actual == 2_000
        assert all(result.validation.values())
        for skill in crafter.store.all_skills():
            assert skill.flagged_for_review
            assert skill.prompt_fragment  # never empty, never skipped

    def test_adjective_fragments_rejected_recrafted_flagged(self, tmp_path: Path) -> None:
        model = FakeChatModel(responder=bad_fragment_responder)
        crafter = Crafter(chat_model=model, base_dir=tmp_path, lattice_id="lat", seed=1)
        result = crafter.build(make_spec(tmp_path))
        # every batch: 1 initial call + 1 retry call; all rejected -> flagged
        assert len(model.calls) == 12 + 100 * 2
        assert result.flagged_for_review == 2_000
        assert all(s.flagged_for_review for s in crafter.store.all_skills())

    def test_mixed_batch_partial_fallback(self, tmp_path: Path) -> None:
        def half_bad(messages):
            text = messages[-1].content
            marker = re.search(
                r"OSCILLATOR_BATCH_JSON: (.*) END_OSCILLATOR_BATCH_JSON", text, re.DOTALL
            )
            if marker is None:
                return expansion_responder(messages)
            payload = json.loads(marker.group(1))
            out = []
            for item in payload:
                mv = int(re.search(r"-MV(\d+)-", item["oscillatorId"]).group(1))
                if mv % 2 == 1:  # exactly half of all oscillators -> 1000 flagged
                    out.append(
                        {
                            "oscillatorId": item["oscillatorId"],
                            "trigger": f"t {item['oscillatorId']}",
                            "promptFragment": f"dissect to {item['path']['purpose']}",
                            "examples": ["a", "b"],
                            "constraints": ["c1", "c2"],
                            "bannedPhrases": ["b1", "b2", "b3"],
                            "evaluationCheck": "ok",
                        }
                    )
                # odd ids missing entirely -> rejected on every attempt
            return json.dumps(out)

        crafter = Crafter(
            chat_model=FakeChatModel(responder=half_bad),
            base_dir=tmp_path,
            lattice_id="lat",
            seed=1,
        )
        result = crafter.build(make_spec(tmp_path))
        assert result.actual == 2_000
        assert result.mini_skills == 2_000
        assert 950 <= result.flagged_for_review <= 1050  # ~half flagged (subset parity drift)


class TestResume:
    def test_kill_at_batch5_then_resume_no_duplicates(self, tmp_path: Path) -> None:
        def crash_at_batch5(event: dict) -> None:
            if event.get("stage") == "batch" and event.get("batch") == 5:
                raise RuntimeError("simulated build interruption")

        crafter = Crafter(
            chat_model=FakeChatModel(responder=craft_responder),
            base_dir=tmp_path,
            lattice_id="lat",
            seed=5,
        )
        with pytest.raises(RuntimeError, match="interruption"):
            crafter.build(make_spec(tmp_path, max_concurrent_batches=1), progress=crash_at_batch5)

        ckpt = load_checkpoint(tmp_path, "lat")
        # batch 5 saved pre-event; one in-flight batch may finish during cancel
        assert ckpt.completed_batches[:6] == list(range(6))
        assert len(ckpt.completed_batches) in (6, 7)
        assert ckpt.total_batches == 100
        assert (
            tmp_path / "data" / "lattices" / "lat" / "oscillators" / "batch_00005.json"
        ).exists()

        survivor = Crafter(
            chat_model=FakeChatModel(responder=craft_responder),
            base_dir=tmp_path,
            lattice_id="lat",
            seed=5,
        )
        result = survivor.build(make_spec(tmp_path, max_concurrent_batches=1))
        ids = [str(x) for x in survivor.lattice.ids[: survivor.lattice.n]]
        assert len(ids) == len(set(ids)) == 2_000  # no duplicates, nothing missing
        assert survivor.store.count() == 2_000
        assert result.mini_skills == 2_000
        assert survivor.store.count() == survivor.lattice.n
        assert load_checkpoint(tmp_path, "lat").completed_batches == list(range(100))  # type: ignore[union-attr]

    def test_parallel_batches_complete(self, tmp_path: Path) -> None:
        crafter = Crafter(
            chat_model=FakeChatModel(responder=craft_responder),
            base_dir=tmp_path,
            lattice_id="lat",
            seed=9,
        )
        result = crafter.build(make_spec(tmp_path, max_concurrent_batches=3))
        assert result.actual == 2_000
        assert result.batches == 100
        assert sorted(e["batch"] for e in result.events if e["stage"] == "batch") == list(
            range(100)
        )


# --- coupling unit level ----------------------------------------------------------
class TestCouplingUnit:
    def _tiny_crafter(self, tmp_path: Path) -> tuple[Crafter, list[OscRecord]]:
        crafter = Crafter(chat_model=FakeChatModel(), base_dir=tmp_path, lattice_id="lat", seed=11)
        embedder = HashingEmbedder(dim=32)
        paths = [
            ("alpha domain", "omega func", "vast mode", "s1", "m1", "p1"),
            ("alpha domain", "omega func", "vast mode", "s1", "m1", "p2"),
            ("alpha domain", "other func", "vast mode", "s1", "m1", "p1"),
            ("beta domain", "omega func", "vast mode", "s1", "m1", "p1"),
            ("beta domain", "third func", "vast mode", "s1", "m1", "p1"),
            ("gamma domain", "fourth func", "vast mode", "s1", "m1", "p1"),
        ]
        records = [
            OscRecord(
                osc_id=f"OSC-T{i:02d}",
                batch=0,
                path=dict(
                    zip(
                        ("domain", "func", "mode", "stance", "move", "purpose"),
                        p,
                        strict=False,
                    )
                ),
                codes={
                    "domain": {"alpha domain": 0, "beta domain": 1, "gamma domain": 2}[p[0]],
                    "func": {
                        "omega func": 0,
                        "other func": 1,
                        "third func": 2,
                        "fourth func": 3,
                    }[p[1]],
                    "mode": 0,
                    "stance": 0,
                    "move": 0,
                    "purpose": 0,
                },
                hz=0.5,
                phase=0.1 * i,
                embedding=embedder.embed_texts([" | ".join(p)])[0],
            )
            for i, p in enumerate(paths)
        ]
        crafter._append_records(records)
        return crafter, records

    def test_proximity_values(self, tmp_path: Path) -> None:
        crafter, records = self._tiny_crafter(tmp_path)
        flagged, merged = crafter.build_coupling(records)
        assert flagged == 0
        assert merged == 0
        get = crafter.lattice.coupling.get
        # same domain + same func -> 0.8
        assert get(0, 1) == pytest.approx(0.8, abs=1e-6)
        # same domain, different func -> 0.5
        assert get(0, 2) == pytest.approx(0.5, abs=1e-6)
        # different domain, same func -> 0.4
        assert get(0, 3) == pytest.approx(0.4, abs=1e-6)
        # different domain, different func -> 0.1
        assert get(0, 4) == pytest.approx(0.1, abs=1e-6)
        assert get(0, 5) == pytest.approx(0.1, abs=1e-6)

    def test_no_self_coupling_and_top_partners(self, tmp_path: Path) -> None:
        crafter, records = self._tiny_crafter(tmp_path)
        crafter.build_coupling(records)
        dense = crafter.lattice.coupling.dense_copy()
        assert np.allclose(np.diag(dense), 0.0)
        for i in range(6):
            assert np.count_nonzero(dense[i]) == min(20, 5)  # n-1 = 5 partners

    def test_merged_duplicate_boosts_coupling(self, tmp_path: Path) -> None:
        crafter, records = self._tiny_crafter(tmp_path)
        records[0].merged_boost = 0.1
        crafter.build_coupling(records)
        assert crafter.lattice.coupling.get(0, 1) == pytest.approx(0.9, abs=1e-6)
        assert crafter.lattice.coupling.get(0, 2) == pytest.approx(0.6, abs=1e-6)

    def test_near_duplicates_reduced_to_02(self, tmp_path: Path) -> None:
        crafter, records = self._tiny_crafter(tmp_path)
        # make records 0 and 1 embedding-identical -> cosine 1.0
        crafter.lattice.embeddings[1] = crafter.lattice.embeddings[0]
        flagged, merged = crafter.build_coupling(records)
        assert flagged >= 1
        assert crafter.lattice.coupling.get(0, 1) == pytest.approx(0.2, abs=1e-6)
