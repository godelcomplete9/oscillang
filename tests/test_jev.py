"""Exhaustive tests for the Jev decision layer: base, rule, llm, mlp, schema."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_runtime import DIM, build_tiny  # noqa: E402

from oscillang.core.config import Config
from oscillang.jev import (
    JevContext,
    JevDecision,
    JevScoreEntry,
    LlmJev,
    MlpJev,
    RuleJev,
    parse_jev_decision,
)
from oscillang.llm.fake import FakeChatModel
from oscillang.runtime import Runtime


def make_ctx(n: int = 4, seed: int = 0) -> JevContext:
    rng = np.random.default_rng(seed)
    return JevContext(
        ids=[f"OSC-T{i:02d}" for i in range(n)],
        coherence=rng.random(n).astype(np.float32),
        phase_prox=rng.random(n).astype(np.float32),
        memory_sim=rng.random(n).astype(np.float32),
        amplitude=rng.uniform(0.3, 1.0, n).astype(np.float32),
        used_cost=rng.random(n).astype(np.float32) * 0.3,
        coupling=rng.random(n).astype(np.float32) * 0.5,
        refractory=np.zeros(n, dtype=bool),
        noise=np.zeros(n, dtype=np.float32),
    )


# --- schema -----------------------------------------------------------------------
class TestSchema:
    def test_valid_decision_parses(self) -> None:
        payload = {
            "scores": [
                {"oscillatorId": "OSC-T00", "score": 0.9, "reason": "hot"},
                {"oscillatorId": "OSC-T01", "score": 0.1},
            ]
        }
        decision = parse_jev_decision(payload)
        assert decision is not None
        assert isinstance(decision, JevDecision)
        assert decision.scores[0].score == 0.9
        assert decision.scores[1].reason is None

    def test_out_of_range_score_rejected(self) -> None:
        payload = {"scores": [{"oscillatorId": "OSC-T00", "score": 1.5}]}
        assert parse_jev_decision(payload) is None

    def test_negative_score_rejected(self) -> None:
        payload = {"scores": [{"oscillatorId": "OSC-T00", "score": -0.1}]}
        assert parse_jev_decision(payload) is None

    def test_missing_scores_rejected(self) -> None:
        assert parse_jev_decision({"wrong": 1}) is None
        assert parse_jev_decision({}) is None

    def test_missing_oscillator_id_rejected(self) -> None:
        payload = {"scores": [{"score": 0.5}]}
        assert parse_jev_decision(payload) is None

    def test_score_entry_bounds(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            JevScoreEntry(oscillatorId="x", score=2.0)
        assert JevScoreEntry(oscillatorId="x", score=1.0).score == 1.0


# --- JevContext --------------------------------------------------------------------
class TestJevContext:
    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="coherence"):
            JevContext(
                ids=["a", "b"],
                coherence=np.zeros(3, dtype=np.float32),
                phase_prox=np.zeros(2, dtype=np.float32),
                memory_sim=np.zeros(2, dtype=np.float32),
                amplitude=np.zeros(2, dtype=np.float32),
                used_cost=np.zeros(2, dtype=np.float32),
                coupling=np.zeros(2, dtype=np.float32),
                refractory=np.zeros(2, dtype=bool),
                noise=np.zeros(2, dtype=np.float32),
            )

    def test_refractory_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="refractory"):
            JevContext(
                ids=["a"],
                coherence=np.zeros(1, dtype=np.float32),
                phase_prox=np.zeros(1, dtype=np.float32),
                memory_sim=np.zeros(1, dtype=np.float32),
                amplitude=np.zeros(1, dtype=np.float32),
                used_cost=np.zeros(1, dtype=np.float32),
                coupling=np.zeros(1, dtype=np.float32),
                refractory=np.zeros(2, dtype=bool),
                noise=np.zeros(1, dtype=np.float32),
            )


# --- RuleJev -----------------------------------------------------------------------
class TestRuleJev:
    def test_exact_weighted_sum(self) -> None:
        cfg = Config(
            noise_enabled=False,
            w_coh=1.0,
            w_phase=0.8,
            w_mem=0.6,
            w_fat=0.7,
            w_cost=0.5,
            w_coup=0.4,
        )
        ctx = make_ctx()
        v = RuleJev(cfg).score(ctx)
        expected = (
            1.0 * ctx.coherence
            + 0.8 * ctx.phase_prox
            + 0.6 * ctx.memory_sim
            + 0.7 * ctx.amplitude
            - 0.5 * ctx.used_cost
            - 0.4 * ctx.coupling
        )
        assert np.allclose(v, expected, atol=1e-6)

    def test_weights_follow_config(self) -> None:
        ctx = make_ctx()
        a = RuleJev(Config(noise_enabled=False, w_coh=1.0)).score(ctx)
        b = RuleJev(Config(noise_enabled=False, w_coh=2.0)).score(ctx)
        assert np.allclose(b - a, ctx.coherence, atol=1e-6)

    def test_leaky_memory_disabled_drops_term(self) -> None:
        cfg = Config(noise_enabled=False)
        ctx = make_ctx()
        v_on = RuleJev(cfg).score(ctx)
        v_off = RuleJev(Config(noise_enabled=False, leaky_memory=False)).score(ctx)
        assert np.allclose(v_on - v_off, cfg.w_mem * ctx.memory_sim, atol=1e-6)

    def test_noise_added_when_enabled(self) -> None:
        cfg = Config(noise=0.1, noise_enabled=True)
        ctx = dataclasses.replace(make_ctx(), noise=np.full(4, 0.05, dtype=np.float32))
        v = RuleJev(cfg).score(ctx)
        clean = RuleJev(Config(noise_enabled=False)).score(ctx)
        assert np.allclose(v - clean, 0.05, atol=1e-6)

    def test_deterministic(self) -> None:
        cfg = Config(noise_enabled=False)
        assert np.array_equal(RuleJev(cfg).score(make_ctx()), RuleJev(cfg).score(make_ctx()))


# --- LlmJev ------------------------------------------------------------------------
def jev_responder_factory(scores: dict[str, float]):
    def responder(messages):
        ids = [
            line.split(":")[0]
            for line in messages[-1].content.splitlines()
            if line.startswith("OSC-") and ": " in line
        ]
        return json.dumps(
            {
                "scores": [
                    {"oscillatorId": oid, "score": scores.get(oid, 0.0), "reason": "r"}
                    for oid in ids
                ]
            }
        )

    return responder


class TestLlmJev:
    def test_scores_mapped_to_rows(self) -> None:
        ctx = make_ctx()
        jev = LlmJev(
            FakeChatModel(responder=jev_responder_factory({"OSC-T00": 0.9, "OSC-T01": 0.2})),
            Config(),
        )
        v = jev.score(ctx)
        assert v[0] == pytest.approx(0.9, abs=1e-6)
        assert v[1] == pytest.approx(0.2, abs=1e-6)
        assert v[2] == pytest.approx(0.0, abs=1e-6)
        assert v[3] == pytest.approx(0.0, abs=1e-6)
        assert len(jev.calls) == 1  # ONE call for the whole shortlist

    def test_scores_clamped_to_unit_range(self) -> None:
        ctx = make_ctx()
        jev = LlmJev(
            FakeChatModel(responder=jev_responder_factory({"OSC-T00": 0.7})),
            Config(),
        )
        v = jev.score(ctx)
        assert 0.0 <= v[0] <= 1.0 + 0.21  # clamp plus at most noise
        assert v[0] <= 1.0 + float(ctx.noise[0]) + 1e-6

    def test_invalid_response_scores_zero(self) -> None:
        ctx = make_ctx()
        jev = LlmJev(FakeChatModel(responder=lambda m: "I cannot do JSON."), Config())
        assert np.allclose(jev.score(ctx), 0.0)

    def test_schema_violation_scores_zero(self) -> None:
        ctx = make_ctx()
        jev = LlmJev(
            FakeChatModel(
                responder=lambda m: json.dumps(
                    {"scores": [{"oscillatorId": "OSC-T00", "score": 9.9}]}
                )
            ),
            Config(),
        )
        assert np.allclose(jev.score(ctx), 0.0)

    def test_unknown_ids_ignored(self) -> None:
        ctx = make_ctx()
        jev = LlmJev(
            FakeChatModel(
                responder=lambda m: json.dumps(
                    {
                        "scores": [
                            {"oscillatorId": "OSC-NOPE", "score": 1.0},
                            {"oscillatorId": "OSC-T02", "score": 0.5},
                        ]
                    }
                )
            ),
            Config(),
        )
        v = jev.score(ctx)
        assert v[2] == pytest.approx(0.5, abs=1e-6)
        assert np.allclose(np.delete(v, 2), 0.0)

    def test_shortlist_respects_prefilter(self) -> None:
        # coherence makes OSC-T00 the rule-scorer leader; shortlist keeps it
        cfg = Config(noise_enabled=False)
        ctx = make_ctx()
        jev = LlmJev(
            FakeChatModel(responder=jev_responder_factory({oid: 0.5 for oid in ctx.ids})),
            cfg,
            shortlist_k=2,
        )
        jev.score(ctx)
        prompt = jev.calls[0][-1].content
        shortlisted = [
            line.split(":")[0] for line in prompt.splitlines() if line.startswith("OSC-")
        ]
        assert len(shortlisted) == 2
        base = RuleJev(cfg).score(ctx)
        top2 = [ctx.ids[i] for i in np.argsort(-base)[:2]]
        assert shortlisted == top2

    def test_missing_entries_score_zero(self) -> None:
        ctx = make_ctx()
        jev = LlmJev(
            FakeChatModel(responder=jev_responder_factory({"OSC-T00": 1.0})),
            Config(),
        )
        v = jev.score(ctx)
        assert v[0] == pytest.approx(1.0, abs=1e-6)
        assert np.allclose(v[1:], 0.0)


# --- MlpJev ------------------------------------------------------------------------
class TestMlpJev:
    def test_score_shape_and_bounds(self) -> None:
        ctx = make_ctx(n=16, seed=3)
        jev = MlpJev(seed=0)
        v = jev.score(ctx)
        assert v.shape == (16,)
        assert np.all(v >= 0.0)
        assert np.all(v <= 2.0)

    def test_deterministic_per_seed(self) -> None:
        ctx = make_ctx(seed=1)
        assert np.array_equal(MlpJev(seed=7).score(ctx), MlpJev(seed=7).score(ctx))

    def test_seeds_differ(self) -> None:
        ctx = make_ctx(seed=1)
        assert not np.allclose(MlpJev(seed=1).score(ctx), MlpJev(seed=2).score(ctx))

    def test_learn_reduces_loss_on_repeat(self) -> None:
        ctx = make_ctx(n=16, seed=5)
        jev = MlpJev(seed=0)
        targets = (ctx.coherence * 1.5).astype(np.float32)
        loss1 = jev.learn(ctx, targets, lr=0.1)
        loss2 = jev.learn(ctx, targets, lr=0.1)
        assert loss2 < loss1

    def test_learn_changes_scores(self) -> None:
        ctx = make_ctx(seed=2)
        jev = MlpJev(seed=0)
        before = jev.score(ctx)
        jev.learn(ctx, np.ones(ctx.coherence.shape[0], dtype=np.float32), lr=0.2)
        after = jev.score(ctx)
        assert not np.allclose(before, after)

    def test_noise_adds_after_network(self) -> None:
        ctx = make_ctx(seed=2)
        noisy = dataclasses.replace(
            ctx, noise=np.full(ctx.coherence.shape[0], 0.1, dtype=np.float32)
        )
        jev = MlpJev(seed=0)
        v = jev.score(noisy)
        clean = jev.score(ctx)
        assert np.allclose(v - clean, 0.1, atol=1e-6)


# --- Runtime integration -----------------------------------------------------------
@pytest.fixture()
def tiny(tmp_path: Path):
    return build_tiny(tmp_path)


class TestRuntimeWithJev:
    def test_default_jev_is_rule(self, tiny) -> None:
        rt = make_rt(tiny)
        assert rt.jev.name == "rule-jev"

    def test_llm_jev_drives_fire_decision(self, tiny, tmp_path: Path) -> None:
        lattice, store, embedder = build_tiny(tmp_path)
        jev = LlmJev(
            FakeChatModel(
                responder=jev_responder_factory(
                    {"OSC-T00": 0.9, "OSC-T01": 0.0, "OSC-T02": 0.0, "OSC-T03": 0.0}
                )
            ),
            Config(embedding_dim=DIM, v_threshold=0.5),
        )
        rt = Runtime(
            lattice=lattice,
            store=store,
            chat_model=FakeChatModel(),
            embedder=embedder,
            config=Config(embedding_dim=DIM, v_threshold=0.5, phase_gating=False),
            jev=jev,
        )
        result = rt.respond("dissect to clarify the missing page")
        assert result.fired_ids == ["OSC-T00"]
        assert len(jev.calls) == 1

    def test_llm_jev_silence_never_fires(self, tiny, tmp_path: Path) -> None:
        lattice, store, embedder = build_tiny(tmp_path)
        jev = LlmJev(
            FakeChatModel(responder=lambda m: "no json today"),
            Config(embedding_dim=DIM),
        )
        rt = Runtime(
            lattice=lattice,
            store=store,
            chat_model=FakeChatModel(),
            embedder=embedder,
            config=Config(embedding_dim=DIM, v_threshold=0.1),
            jev=jev,
        )
        result = rt.respond("anything at all")
        assert result.fired_ids == []
        assert result.response == ""

    def test_mlp_jev_runs_in_runtime(self, tiny, tmp_path: Path) -> None:
        lattice, store, embedder = build_tiny(tmp_path)
        jev = MlpJev(seed=11)
        rt = Runtime(
            lattice=lattice,
            store=store,
            chat_model=FakeChatModel(),
            embedder=embedder,
            config=Config(embedding_dim=DIM, v_threshold=-10.0, phase_gating=False),
            jev=jev,
        )
        result = rt.respond("dissect to clarify the missing page")
        assert result.fired_ids  # scores in [0, 2] pass a threshold of -10
        assert result.fired_scores.min() >= 0.0


def make_rt(tiny, config: Config | None = None) -> Runtime:
    lattice, store, embedder = tiny
    return Runtime(
        lattice=lattice,
        store=store,
        chat_model=FakeChatModel(),
        embedder=embedder,
        config=config,
        seed=0,
        premise="p",
        guiding_words=["w"],
    )
