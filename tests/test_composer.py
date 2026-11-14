"""Exhaustive tests for the Composer: composition, lightweight skills, promotion."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from oscillang.llm.fake import FakeChatModel
from oscillang.skills.composer import PROMOTION_USES, Composer
from oscillang.skills.crafter import Crafter
from oscillang.skills.model import (
    HashingEmbedder,
    InstantiationSpec,
    MiniSkill,
    Reference,
)

PREMISE = "A burned-out translator rediscovers her voice in a collapsing city"


def craft_responder(messages):
    text = messages[-1].content
    marker = re.search(r"OSCILLATOR_BATCH_JSON: (.*) END_OSCILLATOR_BATCH_JSON", text, re.DOTALL)
    if marker is None:
        # hierarchy expansion prompts
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


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[Crafter, InstantiationSpec]:
    tmp_path = tmp_path_factory.mktemp("built")
    spec = InstantiationSpec(
        premise=PREMISE,
        references=[Reference(name="Sample Author", sample_text="She asked. It held.")],
        guiding_words=["frayed", "luminous", "stubborn"],
        radius=1,
        lattice_id="lat",
    )
    crafter = Crafter(
        chat_model=FakeChatModel(responder=craft_responder),
        base_dir=tmp_path,
        lattice_id="lat",
        seed=3,
    )
    crafter.build(spec)
    return crafter, spec


def promote_responder(messages):
    """Valid promotion JSON for the single-oscillator promotion prompt."""
    osc = re.search(r"Oscillator: (.*)", messages[-1].content).group(1)
    return json.dumps(
        {
            "trigger": f"When promoted path {osc} fires",
            "promptFragment": "Dissect the scene to clarify what the page refuses to say.",
            "examples": ["[Sample Author] dissect one", "[Sample Author] dissect two"],
            "constraints": ["sentenceLengthRule: 8 to 14 words", "mustInclude: concrete action"],
            "bannedPhrases": ["In conclusion", "As an AI", "It is important to note"],
            "evaluationCheck": "Check: output dissects toward the stance.",
        }
    )


def make_composer(built, model: FakeChatModel | None = None, seed: int = 0) -> Composer:
    crafter, spec = built
    composer = Composer(
        lattice=crafter.lattice,
        store=crafter.store,
        chat_model=model or FakeChatModel(responder=promote_responder),
        embedder=HashingEmbedder(dim=crafter.config.embedding_dim),
        config=crafter.config,
        base_dir=crafter.base_dir,
        lattice_id=crafter.lattice_id,
        seed=seed,
    )
    records = [_record_from(crafter, i) for i in range(crafter.lattice.n)]
    composer.build_tables(records)
    return composer


def _record_from(crafter: Crafter, row: int):
    from oscillang.skills.crafter import OscRecord

    osc_id = str(crafter.lattice.ids[row])
    skill = crafter.store.get_by_oscillator(osc_id)
    return OscRecord(
        osc_id=osc_id,
        batch=row // 20,
        path={
            "root": PREMISE,
            "domain": skill.domain if skill else "D01",
            "func": skill.func if skill else "Voice",
            "mode": "precise",
            "stance": "toward other",
            "move": "question",
            "purpose": "clarify",
            "guiding_words": "a, b, c",
        },
        codes={
            "domain": int(crafter.lattice.domain[row]),
            "func": int(crafter.lattice.func[row]),
            "mode": int(crafter.lattice.mode[row]),
            "stance": int(crafter.lattice.stance[row]),
            "move": int(crafter.lattice.move[row]),
            "purpose": row % 2,
        },
        hz=float(crafter.lattice.hz[row]),
        phase=float(crafter.lattice.phase[row]),
        embedding=crafter.lattice.embeddings[row].copy(),
    )


class TestComposition:
    def test_compose_before_tables_raises(self, built) -> None:
        crafter, _ = built
        composer = Composer(
            lattice=crafter.lattice,
            store=crafter.store,
            chat_model=FakeChatModel(),
            embedder=HashingEmbedder(dim=32),
        )
        with pytest.raises(RuntimeError, match="build_tables"):
            composer.compose(3)

    def test_compose_extends_lattice_with_one_to_one(self, built) -> None:
        crafter, _ = built
        before = crafter.lattice.n
        skills_before = crafter.store.count()
        composer = make_composer(built)
        composed = composer.compose(30)
        assert len(composed) == 30
        assert crafter.lattice.n == before + 30
        assert crafter.store.count() == skills_before + 30  # 1:1 invariant holds
        for osc_id, skill in composed:
            assert osc_id.startswith("OSC-COMP-")
            assert skill.composed is True
            assert skill.oscillator_id == osc_id
            row = next(i for i in range(crafter.lattice.n) if str(crafter.lattice.ids[i]) == osc_id)
            assert str(crafter.lattice.template_id[row]) == skill.id

    def test_composed_embeddings_and_fields(self, built) -> None:
        crafter, _ = built
        composer = make_composer(built)
        before = crafter.lattice.n
        composer.compose(10)
        rows = list(range(before, before + 10))
        emb = crafter.lattice.embeddings[rows]
        assert np.all(np.linalg.norm(emb, axis=1) > 0.0)
        assert np.all(np.isfinite(emb))
        assert np.all(
            (crafter.lattice.phase[rows] >= 0.0) & (crafter.lattice.phase[rows] < 2 * np.pi)
        )
        assert np.all((crafter.lattice.hz[rows] >= 0.01) & (crafter.lattice.hz[rows] <= 1.5))
        assert np.allclose(crafter.lattice.amplitude[rows], crafter.config.baseline_amplitude)

    def test_lightweight_fragment_trimmed_to_80_words(self, built) -> None:
        crafter, _ = built
        composer = make_composer(built)
        for _, skill in composer.compose(10):
            assert len(skill.prompt_fragment.split()) <= 80
            assert skill.trigger
            assert len(skill.banned_phrases) >= 3
            assert skill.evaluation_check

    def test_tables_shapes(self, built) -> None:
        composer = make_composer(built)
        assert composer._tables is not None
        assert composer._tables["domain"].shape[0] == 12
        assert composer._tables["func"].shape[0] == 14
        for table in composer._tables.values():
            assert np.all(np.isfinite(table))

    def test_composed_ids_persisted(self, built, tmp_path_factory) -> None:
        crafter, _ = built
        composer = make_composer(built)
        composed = composer.compose(5)
        path = crafter.base_dir / "data" / "lattices" / "lat" / "composed_ids.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert [osc for osc, _ in composed] == data[-5:]


class TestPromotion:
    def test_no_promotion_before_threshold(self, built) -> None:
        composer = make_composer(built)
        osc_id, _ = composer.compose(1)[0]
        for _ in range(PROMOTION_USES):
            updated = composer.record_use(osc_id)
            assert updated.promoted is False
        assert updated.used_count == PROMOTION_USES

    def test_promotion_after_more_than_five_uses(self, built) -> None:
        composer = make_composer(built)
        osc_id, _ = composer.compose(1)[0]
        for _ in range(PROMOTION_USES):
            composer.record_use(osc_id)
        final = composer.record_use(osc_id)  # 6th use -> > 5 -> promote
        assert final.used_count == PROMOTION_USES + 1
        assert final.promoted is True
        assert final.flagged_for_review is False
        assert composer.events[-1]["stage"] == "promoted"
        assert composer.events[-1]["oscillator"] == osc_id

    def test_promotion_preserves_identity_and_persists(self, built) -> None:
        crafter, _ = built
        composer = make_composer(built)
        osc_id, _ = composer.compose(1)[0]
        for _ in range(PROMOTION_USES + 1):
            promoted = composer.record_use(osc_id)
        stored = crafter.store.get_by_oscillator(osc_id)
        assert stored is not None
        assert stored.id == promoted.id
        assert stored.promoted is True
        # skill file on disk reflects the promotion
        skill_path = crafter.base_dir / "data" / "skills" / f"{promoted.id}.json"
        revived = MiniSkill.model_validate_json(skill_path.read_text(encoding="utf-8"))
        assert revived.promoted is True

    def test_promotion_with_bad_llm_output_flags_not_promotes(self, built) -> None:
        def empty_trigger(messages):
            return json.dumps(
                {
                    "trigger": "",
                    "promptFragment": "Precise and beautiful.",
                    "examples": [],
                    "constraints": [],
                    "bannedPhrases": [],
                    "evaluationCheck": "",
                }
            )

        composer = make_composer(built, model=FakeChatModel(responder=empty_trigger))
        osc_id, _ = composer.compose(1)[0]
        for _ in range(PROMOTION_USES):
            composer.record_use(osc_id)
        final = composer.record_use(osc_id)
        assert final.promoted is False
        assert final.flagged_for_review is True

    def test_record_use_unknown_oscillator(self, built) -> None:
        composer = make_composer(built)
        with pytest.raises(KeyError):
            composer.record_use("OSC-NOPE")
