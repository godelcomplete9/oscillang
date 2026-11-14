"""Exhaustive tests for the runtime loop: memory, guards, engine, end to end."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_crafter import PREMISE, WORDS, craft_responder, make_spec  # noqa: E402

from oscillang.core.config import Config
from oscillang.core.lattice import OscillatorLattice
from oscillang.llm.fake import FakeChatModel
from oscillang.runtime import GuardState, MemoryTrace, Runtime
from oscillang.skills.composer import PROMOTION_USES
from oscillang.skills.crafter import Crafter
from oscillang.skills.model import HashingEmbedder, MiniSkill
from oscillang.skills.store import SkillStore

DIM = 32


def make_mini(i: int, trigger: str, fragment: str) -> MiniSkill:
    return MiniSkill(
        id=f"MS-T{i:02d}",
        oscillator_id=f"OSC-T{i:02d}",
        domain=f"D{i % 12 + 1:02d}",
        func="Voice",
        trigger=trigger,
        prompt_fragment=fragment,
        examples=["[Sample Author] first example", "[Sample Author] second example"],
        constraints=["sentenceLengthRule: 8 to 14 words", "mustInclude: concrete action"],
        banned_phrases=["In conclusion", "As an AI", "It is important to note"],
        evaluation_check="Check: output stays concrete.",
    )


def build_tiny(base: Path) -> tuple[OscillatorLattice, SkillStore, HashingEmbedder]:
    """4 oscillators; OSC-T00's text is the one the test queries hit."""
    embedder = HashingEmbedder(dim=DIM)
    lattice = OscillatorLattice(Config(embedding_dim=DIM), embedding_dim=DIM)
    texts = [
        "dissect to clarify the missing page",
        "question to persuade the silent city",
        "joke to debug the frayed voice",
        "summarize to anchor the luminous gate",
    ]
    embeddings = embedder.embed_texts(texts)
    lattice.append(
        domain=np.array([i % 3 for i in range(4)], dtype=np.int16),
        func=np.arange(4, dtype=np.int16),
        mode=np.array([i % 2 for i in range(4)], dtype=np.int16),
        stance=np.array([i % 3 for i in range(4)], dtype=np.int16),
        move=np.arange(4, dtype=np.int16),
        hz=np.array([0.3 + 0.1 * i for i in range(4)], dtype=np.float32),
        phase=np.array([0.2 * i for i in range(4)], dtype=np.float32),
        amplitude=np.ones(4, dtype=np.float32),
        embeddings=embeddings,
        ids=np.array([f"OSC-T{i:02d}" for i in range(4)], dtype=object),
    )
    lattice.coupling.set_entries(
        np.array([0, 0, 1, 2], dtype=np.int64),
        np.array([1, 2, 3, 3], dtype=np.int64),
        np.array([0.5, 0.4, 0.1, 0.1], dtype=np.float32),
    )
    store = SkillStore(base, embedder)
    for i, text in enumerate(texts):
        store.save(make_mini(i, trigger=text, fragment=f"dissect to clarify beat {i}"))
    return lattice, store, embedder


@pytest.fixture()
def tiny(tmp_path: Path) -> tuple[OscillatorLattice, SkillStore, HashingEmbedder]:
    return build_tiny(tmp_path)


def make_runtime(
    tiny, config: Config | None = None, responder=None, seed: int = 0, premise: str = PREMISE
) -> Runtime:
    lattice, store, embedder = tiny
    model = FakeChatModel(responder=responder)
    return Runtime(
        lattice=lattice,
        store=store,
        chat_model=model,
        embedder=embedder,
        config=config,
        seed=seed,
        premise=premise,
        guiding_words=WORDS,
    )


HOT_TEXT = "dissect to clarify the missing page"


# --- MemoryTrace -----------------------------------------------------------------
class TestMemoryTrace:
    def test_starts_inactive_and_zero_similarity(self) -> None:
        mem = MemoryTrace(DIM, alpha=0.9)
        assert not mem.active
        emb = np.ones((3, DIM), dtype=np.float32)
        assert np.allclose(mem.similarity(emb), 0.0)

    def test_update_is_unit_norm_and_aligns(self) -> None:
        mem = MemoryTrace(DIM)
        emb = np.zeros((2, DIM), dtype=np.float32)
        emb[0, 0] = 1.0
        emb[1, 1] = 1.0
        mem.update(emb)
        assert np.isclose(np.linalg.norm(mem.vec), 1.0, atol=1e-5)
        assert mem.active
        # trace points between the two basis vectors
        assert mem.vec[0] > 0.0 and mem.vec[1] > 0.0

    def test_ema_fades_old_trace(self) -> None:
        mem = MemoryTrace(DIM, alpha=0.9)
        first = np.zeros((1, DIM), dtype=np.float32)
        first[0, 0] = 1.0
        mem.update(first)
        second = np.zeros((1, DIM), dtype=np.float32)
        second[0, 1] = 1.0
        for _ in range(20):  # repeated updates decay the old direction
            mem.update(second)
        assert mem.vec[0] < mem.vec[1]

    def test_empty_update_is_noop(self) -> None:
        mem = MemoryTrace(DIM)
        mem.update(np.zeros((0, DIM), dtype=np.float32))
        assert not mem.active

    def test_similarity_matches_projection(self) -> None:
        mem = MemoryTrace(DIM)
        vec = np.zeros(DIM, dtype=np.float32)
        vec[2] = 1.0
        mem.vec = vec
        emb = np.zeros((2, DIM), dtype=np.float32)
        emb[0, 2] = 0.5
        emb[1, 3] = 1.0
        emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
        sim = mem.similarity(emb)
        assert np.isclose(sim[0], 1.0, atol=1e-5)
        assert np.isclose(sim[1], 0.0, atol=1e-5)


# --- GuardState -------------------------------------------------------------------
class TestGuardState:
    def cfg(self, **kw) -> Config:
        base = dict(
            no_fire_ticks=2,
            fire_every_tick_window=2,
            dead_coherence_ticks=2,
            silence_watchdog_ticks=2,
            guard_vth_step=0.05,
        )
        base.update(kw)
        return Config(**base)

    def test_no_fire_drops_threshold_after_window(self) -> None:
        g = GuardState(self.cfg(v_threshold=0.65))
        assert g.update(fired=False, max_coherence=0.5, responded=True) == []
        events = g.update(fired=False, max_coherence=0.5, responded=True)
        assert events == ["no_fire_threshold_drop"]
        assert np.isclose(g.v_threshold, 0.60)

    def test_threshold_floors_at_vth_floor(self) -> None:
        g = GuardState(self.cfg(v_threshold=0.06))
        for _ in range(10):
            g.update(fired=False, max_coherence=0.5, responded=True)
        assert g.v_threshold >= 0.05

    def test_overfire_raises_threshold(self) -> None:
        g = GuardState(self.cfg(v_threshold=0.10))
        g.update(fired=True, max_coherence=0.9, responded=True)
        events = g.update(fired=True, max_coherence=0.9, responded=True)
        assert events == ["overfire_threshold_raise"]
        assert np.isclose(g.v_threshold, 0.15)

    def test_threshold_caps_at_vth_ceil(self) -> None:
        g = GuardState(self.cfg(v_threshold=0.98))
        for _ in range(10):
            g.update(fired=True, max_coherence=0.9, responded=True)
        assert g.v_threshold <= 0.99

    def test_fire_resets_no_fire_and_vice_versa(self) -> None:
        g = GuardState(self.cfg())
        g.update(fired=False, max_coherence=0.5, responded=True)
        g.update(fired=True, max_coherence=0.5, responded=True)  # resets _no_fire
        events = g.update(fired=False, max_coherence=0.5, responded=True)
        assert events == []  # counter restarts, needs 2 more silent ticks

    def test_dead_coherence_inject(self) -> None:
        g = GuardState(self.cfg(no_fire_ticks=100, fire_every_tick_window=100))
        g.update(fired=True, max_coherence=0.01, responded=True)
        events = g.update(fired=True, max_coherence=0.01, responded=True)
        assert events == ["dead_coherence_noise_inject"]
        # healthy coherence resets the counter
        g.update(fired=True, max_coherence=0.5, responded=True)
        assert g.update(fired=True, max_coherence=0.01, responded=True) == []

    def test_silence_watchdog(self) -> None:
        g = GuardState(self.cfg(no_fire_ticks=100, fire_every_tick_window=100))
        g.update(fired=True, max_coherence=0.9, responded=False)
        events = g.update(fired=True, max_coherence=0.9, responded=False)
        assert events == ["silence_watchdog"]
        g.update(fired=True, max_coherence=0.9, responded=True)
        assert g.update(fired=True, max_coherence=0.9, responded=False) == []

    def test_custom_start_threshold(self) -> None:
        g = GuardState(self.cfg(), v_threshold=1.5)
        assert g.v_threshold == 1.5


# --- Runtime engine ---------------------------------------------------------------
class TestRuntimeValidation:
    def test_empty_lattice_rejected(self) -> None:
        embedder = HashingEmbedder(dim=DIM)
        lattice = OscillatorLattice(Config(embedding_dim=DIM), embedding_dim=DIM)
        store = SkillStore(Path("unused"), embedder)
        with pytest.raises(ValueError, match="built lattice"):
            make_runtime((lattice, store, embedder))

    def test_embedder_dim_mismatch_rejected(self, tiny) -> None:
        lattice, store, _ = tiny
        with pytest.raises(ValueError, match="dim"):
            Runtime(
                lattice=lattice,
                store=store,
                chat_model=FakeChatModel(),
                embedder=HashingEmbedder(dim=64),
            )


class TestRespond:
    def test_renders_response_with_fired_fragments(self, tiny) -> None:
        rt = make_runtime(tiny)
        result = rt.respond(HOT_TEXT)
        assert result.fired_ids, "the hot oscillator must fire"
        assert result.response  # non-empty render
        assert HOT_TEXT in result.response  # prompt tail echoes input
        assert all(sid.startswith("MS-T") for sid in result.skill_ids)
        assert result.tick == 1

    def test_fired_scores_align_with_ids(self, tiny) -> None:
        rt = make_runtime(tiny)
        result = rt.respond(HOT_TEXT)
        assert result.fired_scores.shape[0] == len(result.fired_ids)
        assert np.all(np.isfinite(result.fired_scores))
        assert np.all(np.diff(result.fired_scores) <= 0)  # descending

    def test_record_use_increments_store(self, tiny) -> None:
        rt = make_runtime(tiny)
        result = rt.respond(HOT_TEXT)
        assert result.skill_ids
        skill = rt.store.get(result.skill_ids[0])
        assert skill is not None
        assert skill.used_count == 1

    def test_evolve_ticks_advance_phase(self, tiny) -> None:
        rt = make_runtime(tiny)
        before = rt.lattice.phase[: rt.lattice.n].copy()
        rt.respond(HOT_TEXT, evolve_ticks=3)
        after = rt.lattice.phase[: rt.lattice.n]
        assert not np.allclose(before, after)
        assert rt.tick == 4  # 3 evolve + 1 fire tick

    def test_fatigue_and_refractory_after_fire(self, tiny) -> None:
        cfg = Config(embedding_dim=DIM)
        rt = make_runtime(tiny, config=cfg)
        result = rt.respond(HOT_TEXT)
        n = rt.lattice.n
        fired_rows = [i for i in range(n) if str(rt.lattice.ids[i]) in result.fired_ids]
        assert fired_rows
        assert np.all(rt.lattice.last_fired_tick[fired_rows] == 0)
        assert np.all(rt.lattice.refractory_until_tick[fired_rows] > 0)  # refractory set
        assert np.all(rt.lattice.amplitude[fired_rows] < 1.0)  # fatigue applied

    def test_refractory_blocks_immediate_refire(self, tiny) -> None:
        rt = make_runtime(tiny)
        first = rt.respond(HOT_TEXT)
        second = rt.respond(HOT_TEXT)
        assert not (set(first.fired_ids) & set(second.fired_ids))

    def test_deterministic_across_seeds_with_noise_off(self, tmp_path: Path) -> None:
        cfg = Config(embedding_dim=DIM, noise_enabled=False)
        r1 = make_runtime(build_tiny(tmp_path / "a"), config=cfg, seed=1)
        r2 = make_runtime(build_tiny(tmp_path / "b"), config=cfg, seed=99)
        a = r1.respond(HOT_TEXT)
        b = r2.respond(HOT_TEXT)
        assert a.fired_ids == b.fired_ids
        assert np.allclose(a.fired_scores, b.fired_scores)

    def test_input_embedding_consistency(self, tiny) -> None:
        rt = make_runtime(tiny)
        v1 = rt._state_vector(HOT_TEXT)
        v2 = rt._state_vector(HOT_TEXT)
        assert np.allclose(v1, v2)
        assert np.isclose(np.linalg.norm(v1), 1.0, atol=1e-5)


class TestRuntimeGuardsInLoop:
    def test_no_fire_guard_drops_threshold_in_loop(self, tiny) -> None:
        cfg = Config(embedding_dim=DIM, v_threshold=10.0, no_fire_ticks=2)
        rt = make_runtime(tiny, config=cfg)
        r1 = rt.respond(HOT_TEXT)
        assert r1.fired_ids == []  # unreachable threshold
        r2 = rt.respond(HOT_TEXT)
        assert "no_fire_threshold_drop" in r2.guard_events
        assert r2.v_threshold < 10.0

    def test_overfire_guard_raises_threshold_in_loop(self, tiny) -> None:
        cfg = Config(
            embedding_dim=DIM,
            v_threshold=-10.0,
            lock_threshold=0.0,
            phase_gating=False,
            refractory=False,
            fire_every_tick_window=2,
        )
        rt = make_runtime(tiny, config=cfg)
        r1 = rt.respond(HOT_TEXT)
        r2 = rt.respond(HOT_TEXT)
        assert r1.fired_ids and r2.fired_ids
        assert "overfire_threshold_raise" in r2.guard_events
        r3 = rt.respond(HOT_TEXT)
        assert r3.v_threshold > -10.0

    def test_silence_nudge_next_turn(self, tiny) -> None:
        def empty_render(messages):
            if any("Write concrete prose now" in m.content for m in messages):
                return "Concrete prose after the nudge."
            return "   "

        cfg = Config(
            embedding_dim=DIM,
            silence_watchdog_ticks=1,
            phase_gating=False,
            refractory=False,
        )
        rt = make_runtime(tiny, config=cfg, responder=empty_render)
        r1 = rt.respond(HOT_TEXT)
        assert r1.response.strip() == ""
        assert "silence_watchdog" in r1.guard_events
        r2 = rt.respond(HOT_TEXT)
        assert r2.response.strip() != ""  # forced re-render with nudge

    def test_events_log_grows(self, tiny) -> None:
        rt = make_runtime(tiny)
        rt.respond(HOT_TEXT)
        rt.respond(HOT_TEXT)
        assert len(rt.events) == 2
        assert rt.events[0]["stage"] == "respond"


class TestPromotion:
    def test_promotion_after_six_uses(self, tiny) -> None:
        def dispatch(messages):
            content = messages[-1].content
            if "Craft one full MiniSkill" in content:
                osc = re.search(r"Oscillator: (.*)", content).group(1)
                return json.dumps(
                    {
                        "trigger": f"When promoted {osc} fires",
                        "promptFragment": "Dissect the scene to clarify the missing page.",
                        "examples": ["[Sample Author] one", "[Sample Author] two"],
                        "constraints": [
                            "sentenceLengthRule: 8 to 14 words",
                            "mustInclude: concrete action",
                        ],
                        "bannedPhrases": ["In conclusion", "As an AI", "It is important to note"],
                        "evaluationCheck": "Check: concrete.",
                    }
                )
            return "[main] rendered"

        cfg = Config(
            embedding_dim=DIM,
            refractory=False,
            phase_gating=False,
            world_model=False,
            leaky_memory=False,
        )
        rt = make_runtime(tiny, config=cfg, responder=dispatch)
        for _ in range(PROMOTION_USES + 1):
            result = rt.respond(HOT_TEXT)
        assert result.skill_ids
        promoted = [rt.store.get(sid) for sid in result.skill_ids]
        assert any(s is not None and s.promoted for s in promoted)

    def test_promotion_with_bad_llm_flags_not_promotes(self, tiny) -> None:
        cfg = Config(
            embedding_dim=DIM,
            refractory=False,
            phase_gating=False,
            world_model=False,
            leaky_memory=False,
        )
        rt = make_runtime(tiny, config=cfg)  # default prose responder
        for _ in range(PROMOTION_USES + 1):
            result = rt.respond(HOT_TEXT)
        assert result.skill_ids
        flagged = [rt.store.get(sid) for sid in result.skill_ids]
        assert all(s is not None and (not s.promoted) for s in flagged)


# --- end to end with the full crafter build ----------------------------------------
@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Runtime:
    tmp_path = tmp_path_factory.mktemp("rt_built")
    spec = make_spec(tmp_path)
    crafter = Crafter(
        chat_model=FakeChatModel(responder=craft_responder),
        base_dir=tmp_path,
        lattice_id="lat",
        seed=3,
    )
    crafter.build(spec)
    embedder = HashingEmbedder(dim=crafter.config.embedding_dim)
    return Runtime(
        lattice=crafter.lattice,
        store=crafter.store,
        chat_model=FakeChatModel(),
        embedder=embedder,
        config=Config(embedding_dim=crafter.config.embedding_dim),
        seed=5,
        premise=PREMISE,
        guiding_words=WORDS,
    )


class TestEndToEndBuilt:
    def test_respond_on_built_lattice(self, built: Runtime) -> None:
        result = built.respond("She asked about the missing page again")
        assert result.response
        assert result.fired_ids
        assert all(oid.startswith("OSC-D") for oid in result.fired_ids)
        assert len(result.fired_ids) <= built.config.top_k
        assert result.max_coherence > 0.0

    def test_one_to_one_preserved_after_turns(self, built: Runtime) -> None:
        for _ in range(3):
            built.respond("The translator counts the words she lost")
        n = built.lattice.n
        assert built.store.count() == n  # one-to-one invariant holds
        tids = [str(t) for t in built.lattice.template_id[:n]]
        assert all(t is not None and built.store.get(t) is not None for t in tids)

    def test_used_counts_persisted_and_promotions_track(self, built: Runtime) -> None:
        result = built.respond("She reads the burned letter aloud")
        assert result.skill_ids
        for sid in result.skill_ids:
            skill = built.store.get(sid)
            assert skill is not None
            assert skill.used_count >= 1
