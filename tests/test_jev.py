"""Exhaustive tests for the Jev decision layer: base, rule, llm, mlp, schema.

Jev verifies semantic coherence and alignment: the reranker sees the input
and the candidates' actual skill texts; the numeric models are dynamics-only
-prefilter and fallback- and carry no semantic signal.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_runtime import DIM, OSC_LINE, build_tiny, dispatch_responder  # noqa: E402

from oscillang.core.config import Config
from oscillang.jev import (
    JevCandidate,
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
        phase_prox=rng.random(n).astype(np.float32),
        recency=rng.random(n).astype(np.float32),
        amplitude=rng.uniform(0.3, 1.0, n).astype(np.float32),
        used_cost=rng.random(n).astype(np.float32) * 0.3,
        coupling=rng.random(n).astype(np.float32) * 0.5,
        refractory=np.zeros(n, dtype=bool),
        noise=np.zeros(n, dtype=np.float32),
    )


def make_candidates(ids: list[str]) -> list[JevCandidate]:
    return [
        JevCandidate(
            oscillator_id=oid,
            skill_id=f"MS-{oid}",
            trigger=f"trigger text for {oid}",
            prompt_fragment=f"fragment text for {oid}",
            examples=(f"example one for {oid}", f"example two for {oid}"),
        )
        for oid in ids
    ]


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
        with pytest.raises(ValueError, match="phase_prox"):
            JevContext(
                ids=["a", "b"],
                phase_prox=np.zeros(3, dtype=np.float32),
                recency=np.zeros(2, dtype=np.float32),
                amplitude=np.zeros(2, dtype=np.float32),
                used_cost=np.zeros(2, dtype=np.float32),
                coupling=np.zeros(2, dtype=np.float32),
                refractory=np.zeros(2, dtype=bool),
                noise=np.zeros(2, dtype=np.float32),
            )

    def test_recency_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="recency"):
            JevContext(
                ids=["a", "b"],
                phase_prox=np.zeros(2, dtype=np.float32),
                recency=np.zeros(3, dtype=np.float32),
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
                phase_prox=np.zeros(1, dtype=np.float32),
                recency=np.zeros(1, dtype=np.float32),
                amplitude=np.zeros(1, dtype=np.float32),
                used_cost=np.zeros(1, dtype=np.float32),
                coupling=np.zeros(1, dtype=np.float32),
                refractory=np.zeros(2, dtype=bool),
                noise=np.zeros(1, dtype=np.float32),
            )


# --- RuleJev -----------------------------------------------------------------------
class TestRuleJev:
    def test_exact_dynamics_weighted_sum(self) -> None:
        cfg = Config(
            noise_enabled=False,
            w_phase=0.8,
            w_rec=0.4,
            w_fat=0.7,
            w_cost=0.5,
            w_coup=0.4,
        )
        ctx = make_ctx()
        v = RuleJev(cfg).score(ctx)
        expected = (
            0.8 * ctx.phase_prox
            + 0.4 * ctx.recency
            + 0.7 * ctx.amplitude
            - 0.5 * ctx.used_cost
            - 0.4 * ctx.coupling
        )
        assert np.allclose(v, expected, atol=1e-6)

    def test_weights_follow_config(self) -> None:
        ctx = make_ctx()
        a = RuleJev(Config(noise_enabled=False, w_phase=0.8)).score(ctx)
        b = RuleJev(Config(noise_enabled=False, w_phase=1.6)).score(ctx)
        assert np.allclose(b - a, 0.8 * ctx.phase_prox, atol=1e-6)

    def test_leaky_memory_disabled_drops_recency(self) -> None:
        cfg = Config(noise_enabled=False)
        ctx = make_ctx()
        v_on = RuleJev(cfg).score(ctx)
        v_off = RuleJev(Config(noise_enabled=False, leaky_memory=False)).score(ctx)
        assert np.allclose(v_on - v_off, cfg.w_rec * ctx.recency, atol=1e-6)

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
        ids = OSC_LINE.findall(messages[-1].content)
        return json.dumps(
            {
                "scores": [
                    {"oscillatorId": oid, "score": scores[oid], "reason": "r"}
                    for oid in ids
                    if oid in scores
                ]
            }
        )

    return responder


class TestLlmJev:
    def test_verdicts_returned_for_candidates(self) -> None:
        candidates = make_candidates(["OSC-T00", "OSC-T01", "OSC-T02"])
        jev = LlmJev(FakeChatModel(responder=jev_responder_factory({"OSC-T00": 0.9})))
        verdicts = jev.rerank("the missing page", candidates)
        assert len(verdicts) == 1
        assert verdicts[0].oscillator_id == "OSC-T00"
        assert verdicts[0].score == pytest.approx(0.9, abs=1e-6)
        assert verdicts[0].reason == "r"
        assert len(jev.calls) == 1  # ONE call for the whole shortlist

    def test_scores_clamped_to_unit_range(self) -> None:
        candidates = make_candidates(["OSC-T00", "OSC-T01"])
        jev = LlmJev(FakeChatModel(responder=jev_responder_factory({"OSC-T00": 0.7})))
        verdicts = jev.rerank("input", candidates)
        assert 0.0 <= verdicts[0].score <= 1.0

    def test_empty_candidates_never_calls(self) -> None:
        jev = LlmJev(FakeChatModel(responder=lambda m: "{}"))
        assert jev.rerank("input", []) == []
        assert jev.calls == []

    def test_invalid_response_yields_no_verdicts(self) -> None:
        jev = LlmJev(FakeChatModel(responder=lambda m: "I cannot do JSON."))
        assert jev.rerank("input", make_candidates(["OSC-T00"])) == []

    def test_schema_violation_yields_no_verdicts(self) -> None:
        jev = LlmJev(
            FakeChatModel(
                responder=lambda m: json.dumps(
                    {"scores": [{"oscillatorId": "OSC-T00", "score": 9.9}]}
                )
            )
        )
        assert jev.rerank("input", make_candidates(["OSC-T00"])) == []

    def test_unknown_ids_ignored(self) -> None:
        candidates = make_candidates(["OSC-T02"])
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
            )
        )
        verdicts = jev.rerank("input", candidates)
        assert [v.oscillator_id for v in verdicts] == ["OSC-T02"]

    def test_duplicate_entries_deduped(self) -> None:
        candidates = make_candidates(["OSC-T00"])
        jev = LlmJev(
            FakeChatModel(
                responder=lambda m: json.dumps(
                    {
                        "scores": [
                            {"oscillatorId": "OSC-T00", "score": 0.9},
                            {"oscillatorId": "OSC-T00", "score": 0.1},
                        ]
                    }
                )
            )
        )
        verdicts = jev.rerank("input", candidates)
        assert len(verdicts) == 1
        assert verdicts[0].score == pytest.approx(0.9, abs=1e-6)

    def test_prompt_carries_input_and_skill_texts(self) -> None:
        candidates = make_candidates(["OSC-T00", "OSC-T01"])
        jev = LlmJev(FakeChatModel(responder=jev_responder_factory({})))
        jev.rerank("the missing page", candidates, context="the previous beat")
        prompt = jev.calls[0][-1].content
        assert "You are Jev" in prompt
        assert "Input: the missing page" in prompt
        assert "Previous beat: the previous beat" in prompt
        assert "OSC-T00: trigger=trigger text for OSC-T00" in prompt
        assert "fragment: fragment text for OSC-T00" in prompt
        assert "examples: example one for OSC-T00 | example two for OSC-T00" in prompt

    def test_no_context_omits_previous_beat(self) -> None:
        candidates = make_candidates(["OSC-T00"])
        jev = LlmJev(FakeChatModel(responder=jev_responder_factory({})))
        jev.rerank("input", candidates)
        assert "Previous beat" not in jev.calls[0][-1].content


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
        targets = (ctx.phase_prox * 1.5).astype(np.float32)
        loss1 = jev.learn(ctx, targets, lr=0.1)
        loss2 = jev.learn(ctx, targets, lr=0.1)
        assert loss2 < loss1

    def test_learn_changes_scores(self) -> None:
        ctx = make_ctx(seed=2)
        jev = MlpJev(seed=0)
        before = jev.score(ctx)
        jev.learn(ctx, np.ones(ctx.phase_prox.shape[0], dtype=np.float32), lr=0.2)
        after = jev.score(ctx)
        assert not np.allclose(before, after)

    def test_noise_adds_after_network(self) -> None:
        ctx = make_ctx(seed=2)
        noisy = dataclasses.replace(
            ctx, noise=np.full(ctx.phase_prox.shape[0], 0.1, dtype=np.float32)
        )
        jev = MlpJev(seed=0)
        v = jev.score(noisy)
        clean = jev.score(ctx)
        positive = clean > 0.0
        assert np.allclose(v[positive] - clean[positive], 0.1, atol=1e-6)
        assert np.all(v[~positive] == 0.0)  # still clamped at zero


# --- Runtime integration -----------------------------------------------------------
@pytest.fixture()
def tiny(tmp_path: Path):
    return build_tiny(tmp_path)


class TestRuntimeWithJev:
    def test_default_jev_is_llm_reranker(self, tiny) -> None:
        rt = make_rt(tiny)
        assert rt.jev.name == "llm-jev"
        assert rt.prefilter.name == "rule-jev"

    def test_llm_jev_drives_fire_decision(self, tiny) -> None:
        verdicts = {"OSC-T00": 0.9, "OSC-T01": 0.0, "OSC-T02": 0.0, "OSC-T03": 0.0}
        rt = make_rt(tiny, config=Config(embedding_dim=DIM), responder=None)
        rt.jev = LlmJev(FakeChatModel(responder=jev_responder_factory(verdicts)))
        result = rt.respond("dissect to clarify the missing page")
        assert result.fired_ids == ["OSC-T00"]

    def test_llm_jev_silence_never_fires(self, tiny) -> None:
        rt = make_rt(tiny, config=Config(embedding_dim=DIM, v_threshold=0.1), responder=None)
        rt.jev = LlmJev(FakeChatModel(responder=lambda m: "no json today"))
        result = rt.respond("anything at all")
        assert result.fired_ids == []
        assert result.response == ""

    def test_shortlist_respects_prefilter(self, tmp_path: Path) -> None:
        cfg = Config(embedding_dim=DIM, jev_shortlist=2, noise_enabled=False)
        rt = make_rt(build_tiny(tmp_path), config=cfg)
        rt.respond("dissect to clarify the missing page")
        prompt = rt.jev.calls[0][-1].content  # type: ignore[attr-defined]
        shortlisted = OSC_LINE.findall(prompt)
        assert len(shortlisted) == 2
        # the best dynamics rows: phase-prox symmetric around the reference,
        # and rows 1/2 carry the lightest coupling cost
        assert sorted(shortlisted) == ["OSC-T01", "OSC-T02"]

    def test_mlp_jev_gates_alone(self, tiny) -> None:
        lattice, store, embedder = tiny
        chat = FakeChatModel()
        rt = Runtime(
            lattice=lattice,
            store=store,
            chat_model=chat,
            embedder=embedder,
            config=Config(embedding_dim=DIM, v_threshold=-10.0, phase_gating=False),
            jev=MlpJev(seed=11),
        )
        result = rt.respond("dissect to clarify the missing page")
        assert result.fired_ids  # scores in [0, 2] pass a threshold of -10
        assert result.fired_scores.min() >= 0.0
        # dynamics-only mode: no rerank call was made (one render call only)
        assert len(chat.calls) == 1


def make_rt(tiny, config: Config | None = None, responder=None) -> Runtime:
    lattice, store, embedder = tiny
    model = FakeChatModel(responder=responder if responder is not None else dispatch_responder())
    return Runtime(
        lattice=lattice,
        store=store,
        chat_model=model,
        embedder=embedder,
        config=config,
        seed=0,
        premise="p",
        guiding_words=["w"],
    )
