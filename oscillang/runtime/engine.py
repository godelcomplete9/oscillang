"""The rotatory prompting runtime: perturb, prefilter, verify, gate, fire, render.

One ``respond`` turn:
1. Perturb phases: a uniform external drive (``g_ext``) plus any guard noise
   injection. No embedding-weighted drive - the oscillator layer is dynamics
   only.
2. Dynamics prefilter: RuleJev scores every oscillator from dynamics only
   (phase proximity to the amplitude-weighted reference, recency, amplitude,
   minus use cost and coupling pressure, plus noise). The top
   ``jev_shortlist`` non-refractory candidates form the reranker's shortlist.
3. Jev verifies semantic coherence and alignment: the reranker (default
   ``LlmJev``) receives the user input and each shortlisted mini skill's
   actual texts and returns an alignment verdict per candidate. If a numeric
   ``JevModel`` is injected instead, its score is thresholded directly
   (dynamics-only mode).
4. Fire decision: verdict above the adaptive threshold AND phase-locked to
   the leader (highest verdict) AND not refractory, capped at ``top_k``.
   Main Brain renders once for the whole fired set - never per oscillator.
5. Dynamics step (phase, fatigue, refractory), ``record_use`` per fired skill
   (promotion after more than 5 uses), recency update, guard update.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from oscillang.core.config import Config
from oscillang.core.dynamics import step_lattice
from oscillang.core.gating import (
    TWO_PI,
    lock_gate_reference,
    retrieval_top_k,
    wrap_phase_diff,
)
from oscillang.core.lattice import OscillatorLattice
from oscillang.jev.base import JevCandidate, JevContext, JevModel, JevReranker, JevVerdict
from oscillang.jev.llm_jev import LlmJev
from oscillang.jev.rule_jev import RuleJev
from oscillang.llm.base import ChatModel, InvokeOptions, Message
from oscillang.runtime.guards import GuardState
from oscillang.runtime.memory import RecencyTrace
from oscillang.skills.composer import Composer
from oscillang.skills.model import EmbeddingModel, MiniSkill
from oscillang.skills.store import SkillStore

__all__ = ["Runtime", "TickResult"]

PI = float(np.pi)


@dataclass(frozen=True)
class TickResult:
    """Outcome of one respond turn."""

    tick: int
    fired_ids: list[str]
    fired_scores: np.ndarray
    response: str
    skill_ids: list[str]
    guard_events: list[str]
    v_threshold: float
    max_verdict: float
    events: list[dict[str, str]] = field(default_factory=list)


class Runtime:
    """Gate mini skills with Jev's alignment verdicts; render with Main Brain."""

    def __init__(
        self,
        lattice: OscillatorLattice,
        store: SkillStore,
        chat_model: ChatModel,
        embedder: EmbeddingModel,
        config: Config | None = None,
        seed: int = 0,
        premise: str = "",
        guiding_words: list[str] | None = None,
        jev: JevReranker | JevModel | None = None,
        prefilter: JevModel | None = None,
    ) -> None:
        self.config = (config or Config()).validated()
        self.jev: JevReranker | JevModel = (
            jev if jev is not None else LlmJev(chat_model, temperature=0.0)
        )
        self.prefilter: JevModel = prefilter if prefilter is not None else RuleJev(self.config)
        self.lattice = lattice
        self.store = store
        self.chat_model = chat_model
        self.embedder = embedder
        if int(embedder.dim) != int(lattice.embedding_dim):
            raise ValueError(
                f"embedder dim {embedder.dim} != lattice embedding_dim {lattice.embedding_dim}"
            )
        if lattice.n == 0:
            raise ValueError("runtime needs a built lattice (n > 0)")
        self.premise = premise
        self.guiding_words = list(guiding_words or [])
        self.rng = np.random.default_rng(seed)
        self.memory = RecencyTrace(
            alpha=self.config.memory_alpha, fade_ticks=self.config.memory_fade_ticks
        )
        self.guards = GuardState(self.config)
        self.tick = 0
        self.events: list[dict[str, str]] = []
        self._composer = Composer(
            lattice=lattice,
            store=store,
            chat_model=chat_model,
            embedder=embedder,
            config=self.config,
            seed=seed,
        )
        self._last_response: str = ""
        self._inject_next = 0.0
        self._nudge_next = False

    # --- dynamics context ------------------------------------------------------
    def _reference_phase(self, phase: np.ndarray, amplitude: np.ndarray) -> float:
        weights = np.clip(amplitude, 0.0, None) + 1e-6
        z = np.sum(weights.astype(np.complex128) * np.exp(1j * phase.astype(np.float64)))
        return float(np.angle(z)) % TWO_PI

    def _jev_context(self, ids: list[str], phase: np.ndarray, ref_phase: float) -> JevContext:
        cfg = self.config
        n = self.lattice.n
        phase_prox = (1.0 - wrap_phase_diff(phase - ref_phase) / PI).astype(np.float32)
        amp = self.lattice.amplitude[:n]
        tids = [str(t) for t in self.lattice.template_id[:n]]
        used: dict[str, int] = {}
        flagged: dict[str, bool] = {}
        for skill in self.store.all_skills():
            used[skill.id] = skill.used_count
            flagged[skill.id] = skill.flagged_for_review
        cost = np.array(
            [
                min(1.0, used.get(tid, 0) / 10.0) + (0.3 if flagged.get(tid, False) else 0.0)
                for tid in tids
            ],
            dtype=np.float32,
        )
        coup = np.minimum(1.0, self.lattice.coupling.row_sums() / 4.0).astype(np.float32)
        noise = (
            self.rng.normal(0.0, cfg.noise, n).astype(np.float32)
            if cfg.noise_enabled
            else np.zeros(n, dtype=np.float32)
        )
        refractory = self.lattice.refractory_until_tick[:n] > self.tick
        recency = (
            self.memory.recency(ids, self.tick)
            if cfg.leaky_memory
            else np.zeros(n, dtype=np.float32)
        )
        return JevContext(
            ids=ids,
            phase_prox=phase_prox,
            recency=recency,
            amplitude=amp,
            used_cost=cost,
            coupling=coup,
            refractory=refractory,
            noise=noise,
        )

    def _shortlist_rows(self, pre: np.ndarray, refractory: np.ndarray) -> list[int]:
        """Top ``jev_shortlist`` non-refractory rows by prefilter score."""
        eligible = np.where(~refractory)[0]
        if eligible.size == 0:
            return []
        order = eligible[np.argsort(-pre[eligible])[: self.config.jev_shortlist]]
        return [int(r) for r in order]

    def _verdict_vector(self, ids: list[str], verdicts: list[JevVerdict]) -> np.ndarray:
        vmap = {v.oscillator_id: v.score for v in verdicts}
        return np.array([vmap.get(oid, 0.0) for oid in ids], dtype=np.float32)

    # --- rendering ------------------------------------------------------------
    def _fragment_block(self, skills: list[MiniSkill]) -> str:
        lines: list[str] = []
        for s in skills:
            lines.append(f"### {s.oscillator_id}")
            lines.append(f"trigger: {s.trigger}")
            lines.append(f"fragment: {s.prompt_fragment}")
            lines.append("examples: " + " | ".join(s.examples))
            lines.append("constraints: " + " | ".join(s.constraints))
            lines.append("banned phrases: " + " | ".join(s.banned_phrases))
            lines.append(f"check: {s.evaluation_check}")
        return "\n".join(lines)

    def _system_prompt(self) -> str:
        parts = ["You are the Main Brain of an oscillator-gated writing agent."]
        if self.premise:
            parts.append(f"Premise: {self.premise}")
        if self.guiding_words:
            parts.append(f"Guiding words: {', '.join(self.guiding_words)}")
        parts.append(
            "The fired behavioral fragments below are binding: honor their "
            "constraints and banned phrases, follow their examples' shape."
        )
        return "\n".join(parts)

    def _render(self, user_input: str, skills: list[MiniSkill], nudged: bool) -> str:
        content = (
            f"Fired behavioral fragments:\n{self._fragment_block(skills)}\n\n"
            f"User input: {user_input}\n\n"
            "Write the next beat."
        )
        messages = [Message(role="system", content=self._system_prompt())]
        if nudged:
            messages.append(
                Message(
                    role="user",
                    content="Your previous reply was empty. Write concrete prose now.",
                )
            )
        messages.append(Message(role="user", content=content))
        response = self.chat_model.invoke(messages, InvokeOptions(temperature=0.7))
        return response.text

    # --- main loop ------------------------------------------------------------
    def respond(self, user_input: str, evolve_ticks: int = 0) -> TickResult:
        """One rotatory turn: perturb, prefilter, verify, fire, render, step."""
        n = self.lattice.n
        cfg = self.config
        if evolve_ticks > 0:
            for _ in range(int(evolve_ticks)):
                step_lattice(self.lattice, self.tick, cfg)
                self.tick += 1

        kick = np.float32(cfg.g_ext + self._inject_next)
        self.lattice.phase[:n] = np.mod(self.lattice.phase[:n] + kick, TWO_PI).astype(np.float32)
        self._inject_next = 0.0

        ref_phase = self._reference_phase(self.lattice.phase[:n], self.lattice.amplitude[:n])
        ids = [str(i) for i in self.lattice.ids[:n]]
        ctx = self._jev_context(ids, self.lattice.phase[:n], ref_phase)
        pre = self.prefilter.score(ctx)

        if hasattr(self.jev, "rerank"):
            rows = self._shortlist_rows(pre, ctx.refractory)
            candidates: list[JevCandidate] = []
            for row in rows:
                skill = self.store.get_by_oscillator(ids[row])
                if skill is not None:
                    candidates.append(
                        JevCandidate(
                            oscillator_id=ids[row],
                            skill_id=skill.id,
                            trigger=skill.trigger,
                            prompt_fragment=skill.prompt_fragment,
                            examples=tuple(skill.examples),
                        )
                    )
            context = self._last_response if cfg.world_model else ""
            verdicts = self.jev.rerank(user_input, candidates, context)
            v = self._verdict_vector(ids, verdicts)
        else:
            v = np.asarray(pre, dtype=np.float32)
        max_v = float(v.max()) if n > 0 else 0.0

        pool = (v > self.guards.v_threshold) & ~ctx.refractory
        fired_idx = np.empty(0, dtype=np.int64)
        fired_scores = np.empty(0, dtype=np.float32)
        if np.any(pool):
            pool_rows = np.nonzero(pool)[0]
            leader_row = int(pool_rows[np.argmax(v[pool_rows])])
            if cfg.phase_gating:
                locked = lock_gate_reference(
                    self.lattice.phase[:n][pool_rows],
                    float(self.lattice.phase[leader_row]),
                    lock_tolerance=cfg.lock_tolerance,
                )
                locked_rows = pool_rows[locked]
            else:
                locked_rows = pool_rows
            fired_idx, fired_scores = retrieval_top_k(
                v, cfg.top_k, exclude_mask=~np.isin(np.arange(n), locked_rows)
            )
            finite = np.isfinite(fired_scores)
            fired_idx = fired_idx[finite]
            fired_scores = fired_scores[finite]
        fire_mask = np.zeros(n, dtype=bool)
        fire_mask[fired_idx] = True

        response = ""
        skills: list[MiniSkill] = []
        skill_ids: list[str] = []
        if fired_idx.size > 0:
            fired_ids = [str(i) for i in self.lattice.ids[fired_idx]]
            for osc_id in fired_ids:
                skill = self.store.get_by_oscillator(osc_id)
                if skill is not None:
                    skills.append(skill)
                    skill_ids.append(skill.id)
            if skills:
                response = self._render(user_input, skills, nudged=self._nudge_next)
            self._nudge_next = False
            for osc_id in fired_ids:
                if self.store.get_by_oscillator(osc_id) is not None:
                    self._composer.record_use(osc_id)
            self.memory.update(fired_ids, self.tick)
        else:
            self._nudge_next = False

        step_lattice(self.lattice, self.tick, cfg, fire_mask=fire_mask)
        self.lattice.last_fired_tick[:n][fire_mask] = self.tick

        if cfg.world_model and response.strip():
            self._last_response = response

        guard_events = self.guards.update(
            fired=bool(fired_idx.size), max_verdict=max_v, responded=bool(response.strip())
        )
        if "dead_coherence_noise_inject" in guard_events:
            self._inject_next = cfg.guard_noise_inject
        if "silence_watchdog" in guard_events:
            self._nudge_next = True

        self.tick += 1
        event = {
            "stage": "respond",
            "tick": str(self.tick),
            "fired": str(len(fired_idx)),
            "guards": ",".join(guard_events),
        }
        self.events.append(event)
        return TickResult(
            tick=self.tick,
            fired_ids=[str(i) for i in self.lattice.ids[fired_idx]],
            fired_scores=np.asarray(fired_scores, dtype=np.float32),
            response=response,
            skill_ids=skill_ids,
            guard_events=guard_events,
            v_threshold=self.guards.v_threshold,
            max_verdict=max_v,
            events=list(self.events),
        )
