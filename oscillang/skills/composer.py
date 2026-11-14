"""Composer: on-demand extended oscillators beyond the fully crafted core.

V = V_domain + V_func + V_mode + V_stance + V_move + V_purpose, where each V
is the mean embedding of core oscillators with that component. Composed
oscillators get a lightweight mini skill immediately (concatenated fragments,
trimmed to 80 words) so the one-to-one invariant never breaks, and are
promoted to fully crafted skills after more than 5 uses.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from oscillang.core.config import Config
from oscillang.core.lattice import OscillatorLattice
from oscillang.llm.base import ChatModel, InvokeOptions, Message
from oscillang.skills.crafter import (
    OscRecord,
    _extract_json_object,
    validate_skill_fields,
)
from oscillang.skills.model import (
    CODE_FUNC_MAP,
    INSTANTIATION_FUNC_HZ,
    EmbeddingModel,
    MiniSkill,
)
from oscillang.skills.store import SkillStore

__all__ = ["Composer", "PROMOTION_USES"]

PROMOTION_USES = 5


class Composer:
    """Compose extended oscillators from core component embeddings."""

    def __init__(
        self,
        lattice: OscillatorLattice,
        store: SkillStore,
        chat_model: ChatModel,
        embedder: EmbeddingModel,
        config: Config | None = None,
        base_dir: Path | str = ".",
        lattice_id: str = "lattice",
        seed: int = 0,
    ) -> None:
        self.config = (config or Config()).validated()
        self.lattice = lattice
        self.store = store
        self.chat_model = chat_model
        self.embedder = embedder
        self.base_dir = Path(base_dir)
        self.lattice_id = lattice_id
        self.composed_dir = self.base_dir / "data" / "lattices" / lattice_id
        self.rng = np.random.default_rng(seed)
        self.events: list[dict[str, str]] = []
        self._tables: dict[str, np.ndarray] | None = None
        self._representative: dict[str, MiniSkill] = {}

    # --- component tables -----------------------------------------------------
    def build_tables(self, records: list[OscRecord]) -> None:
        """Mean embedding per component code, from the crafted core records."""
        n = len(records)
        emb = np.stack([r.embedding for r in records])
        d_codes = np.array([r.codes["domain"] for r in records])
        f_codes = np.array([r.codes["func"] for r in records])
        m_codes = np.array([r.codes["mode"] for r in records])
        s_codes = np.array([r.codes["stance"] for r in records])
        mv_codes = np.array([r.codes["move"] for r in records])
        p_codes = np.array([r.codes["purpose"] for r in records])
        self._tables = {
            "domain": _mean_by_code(emb, d_codes, 12),
            "func": _mean_by_code(emb, f_codes, 14),
            "mode": _mean_by_code(emb, m_codes, int(m_codes.max()) + 1 if n else 1),
            "stance": _mean_by_code(emb, s_codes, 3),
            "move": _mean_by_code(emb, mv_codes, int(mv_codes.max()) + 1 if n else 1),
            "purpose": _mean_by_code(emb, p_codes, int(p_codes.max()) + 1 if n else 1),
        }
        for rec in records:
            skill = self.store.get_by_oscillator(rec.osc_id)
            if skill is None:
                continue
            self._representative.setdefault(f"domain:{rec.codes['domain']}", skill)
            self._representative.setdefault(f"func:{rec.codes['func']}", skill)
            self._representative.setdefault(f"mode:{rec.codes['mode']}", skill)
            self._representative.setdefault(f"move:{rec.codes['move']}", skill)

    # --- composition ------------------------------------------------------------
    def compose(self, count: int) -> list[tuple[str, MiniSkill]]:
        """Create count composed oscillators + lightweight mini skills (1:1)."""
        if self._tables is None:
            raise RuntimeError("call build_tables(records) before compose()")
        if count <= 0:
            return []
        n_dom = self._tables["domain"].shape[0]
        n_fun = self._tables["func"].shape[0]
        n_mod = self._tables["mode"].shape[0]
        n_sta = self._tables["stance"].shape[0]
        n_mov = self._tables["move"].shape[0]
        n_pur = self._tables["purpose"].shape[0]
        dc = self.rng.integers(0, n_dom, size=count).astype(np.int16)
        fc = self.rng.integers(0, n_fun, size=count).astype(np.int16)
        mc = self.rng.integers(0, n_mod, size=count).astype(np.int16)
        sc = self.rng.integers(0, n_sta, size=count).astype(np.int16)
        mvc = self.rng.integers(0, n_mov, size=count).astype(np.int16)
        pc = self.rng.integers(0, n_pur, size=count).astype(np.int16)
        emb = (
            self._tables["domain"][dc]
            + self._tables["func"][fc]
            + self._tables["mode"][mc]
            + self._tables["stance"][sc]
            + self._tables["move"][mvc]
            + self._tables["purpose"][pc]
        ).astype(np.float32)
        base_hz = np.array(
            [INSTANTIATION_FUNC_HZ.get(self._func_label(int(f)), 0.5) for f in fc],
            dtype=np.float32,
        )
        hz = np.clip(base_hz * (0.9 + 0.2 * self.rng.random(count)), 0.01, 1.5).astype(np.float32)
        phase = self.rng.uniform(0.0, 2.0 * np.pi, size=count).astype(np.float32)
        start = int(self.lattice.n)
        ids = np.array([f"OSC-COMP-{start + i:06d}" for i in range(count)], dtype=object)
        self.lattice.append(
            domain=dc,
            func=fc,
            hz=hz,
            phase=phase,
            amplitude=np.full(count, self.config.baseline_amplitude, dtype=np.float32),
            embeddings=emb,
            mode=mc,
            stance=sc,
            move=mvc,
            ids=ids,
        )
        out: list[tuple[str, MiniSkill]] = []
        for i in range(count):
            osc_id = str(ids[i])
            skill = self._lightweight_skill(osc_id, int(dc[i]), int(fc[i]), int(mc[i]), int(mvc[i]))
            self.lattice.template_id[start + i] = skill.id
            self.store.save(skill)
            out.append((osc_id, skill))
        self._persist_composed_ids([osc for osc, _ in out])
        self.events.append({"stage": "composed", "count": str(count)})
        return out

    def _func_label(self, f_code: int) -> str:
        skill = self._representative.get(f"func:{f_code}")
        if skill is not None:
            label = skill.func
            return next((k for k, v in CODE_FUNC_MAP.items() if v == label), label)
        return "Voice"

    def _lightweight_skill(self, osc_id: str, d: int, f: int, m: int, mv: int) -> MiniSkill:
        parts = []
        for key, code in (("domain", d), ("func", f), ("mode", m), ("move", mv)):
            rep = self._representative.get(f"{key}:{code}")
            if rep is not None:
                parts.append(rep.prompt_fragment)
        fragment = " ".join(parts) if parts else "question to clarify; stay concrete."
        fragment = _trim_words(fragment, 80)
        return MiniSkill(
            id=f"MS-COMP-{osc_id.removeprefix('OSC-COMP-')}",
            oscillator_id=osc_id,
            domain=f"composed-D{d:02d}",
            func=self._func_label(f),
            trigger=f"When the composed path (domain {d}, func {f}, mode {m}, move {mv}) resonates",
            prompt_fragment=fragment,
            examples=["Composed from parent fragments; promote to craft real examples."],
            constraints=["maxAdjectivesPerSentence: 1", "mustInclude: concrete action"],
            banned_phrases=["In conclusion", "As an AI", "It is important to note"],
            evaluation_check="Check: fragment stays concrete and within 80 words.",
            composed=True,
        )

    # --- promotion ------------------------------------------------------------
    def record_use(self, osc_id: str) -> MiniSkill:
        """Track a use; promote to a fully crafted skill after >5 uses."""
        skill = self.store.get_by_oscillator(osc_id)
        if skill is None:
            raise KeyError(f"no mini skill for oscillator {osc_id}")
        skill.used_count += 1
        if skill.used_count > PROMOTION_USES and not skill.promoted:
            promoted = self._promote(skill)
            self.events.append(
                {"stage": "promoted", "oscillator": osc_id, "uses": str(skill.used_count)}
            )
            return promoted
        self.store.save(skill)
        return skill

    def _promote(self, skill: MiniSkill) -> MiniSkill:
        rec_lookup = self._find_record(skill.oscillator_id)
        path_text = json.dumps(rec_lookup or {"oscillatorId": skill.oscillator_id})
        prompt = (
            "Craft one full MiniSkill for this oscillator. Return JSON only with keys "
            "trigger, promptFragment, examples (2-5), constraints (2-4), bannedPhrases "
            "(3-8), evaluationCheck.\nOscillator: " + path_text
        )
        response = self.chat_model.invoke(
            [Message(role="user", content=prompt)], InvokeOptions(temperature=0.3)
        )
        parsed = _extract_json_object(response.text) or {}
        candidate = MiniSkill(
            id=skill.id,
            oscillator_id=skill.oscillator_id,
            domain=skill.domain,
            func=skill.func,
            trigger=str(parsed.get("trigger", skill.trigger)),
            prompt_fragment=str(parsed.get("promptFragment", skill.prompt_fragment)),
            examples=[str(e) for e in parsed.get("examples", skill.examples)][:5],
            constraints=[str(c) for c in parsed.get("constraints", skill.constraints)][:4],
            banned_phrases=[str(b) for b in parsed.get("bannedPhrases", skill.banned_phrases)][:8],
            evaluation_check=str(parsed.get("evaluationCheck", skill.evaluation_check)),
            composed=skill.composed,
            promoted=True,
            used_count=skill.used_count,
        )
        errors = validate_skill_fields(
            candidate.trigger,
            candidate.prompt_fragment,
            candidate.examples,
            candidate.constraints,
            candidate.banned_phrases,
            candidate.evaluation_check,
        )
        if errors:
            candidate = candidate.model_copy(update={"promoted": False, "flagged_for_review": True})
        self.store.save(candidate)
        return candidate

    def _find_record(self, osc_id: str) -> dict[str, str] | None:
        """Path lookup for composed oscillators from the persisted id file."""
        path = self.composed_dir / "composed_ids.json"
        if not path.exists():
            return None
        ids = json.loads(path.read_text(encoding="utf-8"))
        if osc_id not in ids:
            return None
        return {"oscillatorId": osc_id, "composed": "true"}

    def _persist_composed_ids(self, ids: list[str]) -> None:
        self.composed_dir.mkdir(parents=True, exist_ok=True)
        path = self.composed_dir / "composed_ids.json"
        existing: list[str] = []
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
        existing.extend(ids)
        path.write_text(json.dumps(existing), encoding="utf-8")


def _mean_by_code(emb: np.ndarray, codes: np.ndarray, k: int) -> np.ndarray:
    """Mean embedding per code, vectorized via a scatter matrix."""
    out = np.zeros((k, emb.shape[1]), dtype=np.float32)
    if codes.size:
        one_hot = np.zeros((k, codes.size), dtype=np.float32)
        one_hot[codes, np.arange(codes.size)] = 1.0
        counts = one_hot.sum(axis=1, keepdims=True)
        sums = one_hot @ emb
        out = np.divide(sums, np.maximum(counts, 1e-8)).astype(np.float32)
    return out


def _trim_words(text: str, limit: int) -> str:
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    return " ".join(words[:limit])
