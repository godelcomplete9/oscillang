"""The rotatory prompting runtime: perturb, evolve, gate, fire, render.

One ``respond`` turn:
1. Encode the user input into the lattice's embedding space; the state vector
   is ``g_ext`` * input + ``g_sim`` * memory trace + ``g_self`` * last output.
2. Perturb phases (external drive proportional to coherence, plus guard noise).
3. Score every oscillator with the numeric Jev: coherence, phase proximity,
   memory, amplitude, minus use cost and coupling pressure, plus noise.
4. Fire decision: lock gate (phase AND coherence AND not refractory) AND
   Jev > adaptive threshold, capped at ``top_k``. Main Brain renders once for
   the whole fired set - never per oscillator.
5. Dynamics step (phase, fatigue, refractory), ``record_use`` per fired skill
   (promotion after more than 5 uses), memory trace update, guard update.

``learned_jev`` is reserved: this milestone always uses the numeric scorer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from oscillang.core.coherence import compose_state_vector
from oscillang.core.config import Config
from oscillang.core.dynamics import step_lattice
from oscillang.core.gating import (
    TWO_PI,
    lock_gate_reference,
    retrieval_top_k,
    wrap_phase_diff,
)
from oscillang.core.lattice import OscillatorLattice, l2_normalize_rows
from oscillang.llm.base import ChatModel, InvokeOptions, Message
from oscillang.runtime.guards import GuardState
from oscillang.runtime.memory import MemoryTrace
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
    max_coherence: float
    events: list[dict[str, str]] = field(default_factory=list)


class Runtime:
    """Gate behavioral fragments with the numeric Jev; render with Main Brain."""

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
    ) -> None:
        self.config = (config or Config()).validated()
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
        self.memory = MemoryTrace(lattice.embedding_dim, self.config.memory_alpha)
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
        self._last_output_emb: np.ndarray | None = None
        self._inject_next = 0.0
        self._nudge_next = False
        self._skill_count = -1
        self._skill_normed = np.zeros((0, lattice.embedding_dim), dtype=np.float32)

    # --- skill-embedding channel ----------------------------------------------
    def _refresh_skill_embeddings(self) -> None:
        """Coherence is measured against skill texts, not path texts: the
        oscillators' retrieval meaning lives in their mini skills."""
        count = self.store.count()
        if count == self._skill_count:
            return
        n = self.lattice.n
        texts: list[str] = []
        for row in range(n):
            skill = self.store.get_by_oscillator(str(self.lattice.ids[row]))
            if skill is None:
                texts.append("")
            else:
                texts.append(f"{skill.trigger} {skill.prompt_fragment} {' '.join(skill.examples)}")
        emb = (
            self.embedder.embed_texts(texts)
            if texts
            else np.zeros((0, self.lattice.embedding_dim), dtype=np.float32)
        )
        self._skill_normed = l2_normalize_rows(emb)
        self._skill_count = count

    # --- encoding -------------------------------------------------------------
    def _state_vector(self, user_input: str) -> np.ndarray:
        q = self.embedder.embed_texts([user_input])[0]
        norm = max(float(np.linalg.norm(q)), 1e-8)
        comps: list[tuple[np.ndarray, float]] = [(q / norm, self.config.g_ext)]
        if self.config.leaky_memory and self.memory.active:
            comps.append((self.memory.vec, self.config.g_sim))
        if self.config.world_model and self._last_output_emb is not None:
            comps.append((self._last_output_emb, self.config.g_self))
        return compose_state_vector(comps, self.lattice.embedding_dim)

    # --- scoring --------------------------------------------------------------
    def _reference_phase(self, phase: np.ndarray, coherence: np.ndarray) -> float:
        weights = np.clip(coherence, 0.0, None) + 1e-6
        z = np.sum(weights.astype(np.complex128) * np.exp(1j * phase.astype(np.float64)))
        return float(np.angle(z)) % TWO_PI

    def _jev_scores(
        self,
        coherence: np.ndarray,
        phase: np.ndarray,
        ref_phase: float,
        memory_sim: np.ndarray,
    ) -> np.ndarray:
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
        jev = (
            cfg.w_coh * coherence
            + cfg.w_phase * phase_prox
            + (cfg.w_mem * memory_sim if cfg.leaky_memory else 0.0)
            + cfg.w_fat * amp
            - cfg.w_cost * cost
            - cfg.w_coup * coup
            + noise
        )
        return jev.astype(np.float32)

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
        """One rotatory turn: perturb, (optionally evolve), fire, render, step."""
        n = self.lattice.n
        cfg = self.config
        if evolve_ticks > 0:
            for _ in range(int(evolve_ticks)):
                step_lattice(self.lattice, self.tick, cfg)
                self.tick += 1

        state = self._state_vector(user_input)
        self._refresh_skill_embeddings()
        coherence = (self._skill_normed @ state).astype(np.float32)
        phase = self.lattice.phase[:n]

        kick = (cfg.g_ext * np.clip(coherence, 0.0, None) + self._inject_next).astype(np.float32)
        self.lattice.phase[:n] = np.mod(phase + kick, TWO_PI).astype(np.float32)
        self._inject_next = 0.0

        ref_phase = self._reference_phase(self.lattice.phase[:n], coherence)
        memory_sim = self.memory.similarity(self._skill_normed)
        jev = self._jev_scores(coherence, self.lattice.phase[:n], ref_phase, memory_sim)

        # Candidate pool: Jev above the (adaptive) threshold and not refractory.
        refractory = self.lattice.refractory_until_tick[:n] > self.tick
        pool = (jev > self.guards.v_threshold) & ~refractory
        fired_idx = np.empty(0, dtype=np.int64)
        fired_scores = np.empty(0, dtype=np.float32)
        if np.any(pool):
            pool_rows = np.nonzero(pool)[0]
            leader_row = int(pool_rows[np.argmax(jev[pool_rows])])
            # Lock gate against the leader: phase lock AND coherence agreement.
            coh_to_leader = self._skill_normed[pool_rows] @ self._skill_normed[leader_row]
            locked = lock_gate_reference(
                self.lattice.phase[:n][pool_rows],
                float(self.lattice.phase[leader_row]),
                coh_to_leader,
                lock_tolerance=cfg.lock_tolerance,
                lock_threshold=cfg.lock_threshold,
                phase_gating=cfg.phase_gating,
            )
            locked_rows = pool_rows[locked]
            fired_idx, fired_scores = retrieval_top_k(
                jev, cfg.top_k, exclude_mask=~np.isin(np.arange(n), locked_rows)
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
        else:
            self._nudge_next = False

        step_lattice(self.lattice, self.tick, cfg, fire_mask=fire_mask)
        self.lattice.last_fired_tick[:n][fire_mask] = self.tick

        if fired_idx.size > 0:
            self.memory.update(self.lattice.embeddings[fired_idx])
        if cfg.world_model and response.strip():
            self._last_output_emb = self.embedder.embed_texts([response])[0]

        max_coh = float(coherence.max()) if n > 0 else 0.0
        guard_events = self.guards.update(
            fired=bool(fired_idx.size), max_coherence=max_coh, responded=bool(response.strip())
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
            max_coherence=max_coh,
            events=list(self.events),
        )
