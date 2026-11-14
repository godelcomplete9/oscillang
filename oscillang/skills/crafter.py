"""Crafter: hierarchy of ontology -> lattice of oscillators + one MiniSkill each.

Pipeline: build/validate hierarchy -> control leaf count for the Radius ->
create one oscillator per leaf (path id, Hz from func defaults + overrides +
jitter, random phase, amplitude 1.0, path embedding, proximity coupling) ->
craft exactly one MiniSkill per oscillator via Main Brain in batches of 20
(reject + re-craft once + flag for review, never skip) -> persist, checkpoint,
resume -> dedup and validation gates. Works for Book and Code lattices alike.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from oscillang.core.config import Config
from oscillang.core.lattice import OscillatorLattice
from oscillang.llm.base import ChatModel, InvokeOptions, Message
from oscillang.skills.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from oscillang.skills.model import (
    CODE_FUNC_MAP,
    INSTANTIATION_FUNC_HZ,
    MODE_VOCAB,
    MOVE_VOCAB,
    PURPOSE_VOCAB,
    RADII,
    STANCE_VOCAB,
    EmbeddingModel,
    HashingEmbedder,
    InstantiationSpec,
    MiniSkill,
    Ontology,
    OntologyNode,
)
from oscillang.skills.store import SkillStore

__all__ = [
    "Crafter",
    "OscRecord",
    "BuildResult",
    "BuildValidationError",
    "validate_skill_fields",
    "BATCH_SIZE",
    "TOP_PARTNERS",
]

BATCH_SIZE = 20
TOP_PARTNERS = 20

VERB_LEXICON: frozenset[str] = frozenset(
    MOVE_VOCAB + ("use", "ask", "keep", "aim", "write", "say", "open", "cut", "name", "hold")
)

ProgressCallback = Callable[[dict[str, Any]], None]


class BuildValidationError(RuntimeError):
    """Raised when an instantiation acceptance gate fails."""


@dataclass
class OscRecord:
    """One oscillator leaf: full ontology path + lattice row data."""

    osc_id: str
    batch: int
    path: dict[str, str]
    codes: dict[str, int]
    hz: float
    phase: float
    embedding: np.ndarray
    merged_boost: float = 0.0
    flagged: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "osc_id": self.osc_id,
            "batch": self.batch,
            "path": self.path,
            "codes": self.codes,
            "hz": self.hz,
            "phase": self.phase,
            "embedding": [float(x) for x in self.embedding],
            "merged_boost": self.merged_boost,
            "flagged": self.flagged,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> OscRecord:
        return cls(
            osc_id=data["osc_id"],
            batch=int(data["batch"]),
            path=dict(data["path"]),
            codes={k: int(v) for k, v in data["codes"].items()},
            hz=float(data["hz"]),
            phase=float(data["phase"]),
            embedding=np.asarray(data["embedding"], dtype=np.float32),
            merged_boost=float(data.get("merged_boost", 0.0)),
            flagged=bool(data.get("flagged", False)),
        )


@dataclass
class BuildResult:
    lattice_id: str
    target: int
    actual: int
    total_leaves: int
    compose_gap: int
    batches: int
    mini_skills: int
    flagged_for_review: int
    merged_duplicates: int
    near_duplicates_flagged: int
    elapsed_seconds: float
    validation: dict[str, bool]
    events: list[dict[str, Any]] = field(default_factory=list)


# --- mini skill validation -----------------------------------------------------
def validate_skill_fields(
    trigger: str,
    prompt_fragment: str,
    examples: list[str],
    constraints: list[str],
    banned_phrases: list[str],
    evaluation_check: str,
) -> list[str]:
    """Reject rules from the spec: nothing generic, nothing untestable."""
    errors: list[str] = []
    words = prompt_fragment.split()
    if not trigger.strip():
        errors.append("empty trigger")
    if len(words) == 0:
        errors.append("empty promptFragment")
    else:
        if len(words) > 80:
            errors.append("promptFragment longer than 80 words")
        if not any(tok in VERB_LEXICON for tok in _tokens(prompt_fragment)):
            errors.append("promptFragment is not concrete (no imperative verb)")
    if len(examples) < 2:
        errors.append("fewer than 2 examples")
    if len(constraints) < 2:
        errors.append("fewer than 2 constraints")
    if len(banned_phrases) < 3:
        errors.append("fewer than 3 bannedPhrases")
    if not evaluation_check.strip():
        errors.append("empty evaluationCheck")
    return errors


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z]+", text.lower()) if t]


def _extract_json_array(text: str) -> list[dict[str, Any]] | None:
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _clamp_expand(
    modes: list[str],
    moves: list[str],
    purposes: list[str],
) -> dict[str, list[str]]:
    """Enforce branching limits: L3 3-5, L5 3-6, L6 1-3; dedupe labels."""

    def clean(items: list[str], lo: int, hi: int, filler: list[str]) -> list[str]:
        seen: list[str] = []
        lowers = {s.lower() for s in seen}
        for item in items:
            label = str(item).strip()
            if label and label.lower() not in lowers:
                seen.append(label)
                lowers.add(label.lower())
        pool = [f for f in filler if f.lower() not in lowers]
        pos = 0
        while len(seen) < lo and pool:
            seen.append(pool[pos % len(pool)])
            pos += 1
        return seen[:hi]

    return {
        "modes": clean(modes, 3, 5, list(MODE_VOCAB)),
        "moves": clean(moves, 3, 6, list(MOVE_VOCAB)),
        "purposes": clean(purposes, 1, 3, list(PURPOSE_VOCAB)),
    }


def _default_expansion(d_idx: int, f_idx: int) -> dict[str, list[str]]:
    """Deterministic fallback expansion (no LLM), within branching limits."""
    m = (d_idx * 2 + f_idx) % len(MODE_VOCAB)
    v = (d_idx + f_idx) % len(MOVE_VOCAB)
    p = (d_idx + 2 * f_idx) % len(PURPOSE_VOCAB)
    return {
        "modes": [MODE_VOCAB[(m + i) % len(MODE_VOCAB)] for i in range(3)],
        "moves": [MOVE_VOCAB[(v + i) % len(MOVE_VOCAB)] for i in range(3)],
        "purposes": [PURPOSE_VOCAB[(p + i) % len(PURPOSE_VOCAB)] for i in range(2)],
    }


@dataclass
class _BatchContext:
    """Mutable state shared by concurrent batch tasks."""

    spec: InstantiationSpec
    ckpt: Checkpoint
    completed: list[int]
    batches_total: int
    merged: int
    progress: ProgressCallback | None
    emit: Callable[[dict[str, Any]], None]


class Crafter:
    """Batched, resumable, progress-reporting lattice builder."""

    def __init__(
        self,
        chat_model: ChatModel,
        config: Config | None = None,
        embedder: EmbeddingModel | None = None,
        base_dir: Path | str = ".",
        lattice_id: str = "lattice",
        seed: int = 0,
    ) -> None:
        self.config = (config or Config()).validated()
        self.chat_model = chat_model
        self.embedder = embedder or HashingEmbedder(dim=self.config.embedding_dim)
        self.base_dir = Path(base_dir)
        self.lattice_id = lattice_id
        self.lattice_dir = self.base_dir / "data" / "lattices" / lattice_id
        self.osc_dir = self.lattice_dir / "oscillators"
        self.lattice = OscillatorLattice(config=self.config)
        self.store = SkillStore(self.base_dir, self.embedder)
        self.rng = np.random.default_rng(seed)
        self.events: list[dict[str, Any]] = []
        self._row_by_id: dict[str, int] = {}

    # --- hierarchy ------------------------------------------------------------
    def ensure_hierarchy(self, spec: InstantiationSpec) -> Ontology:
        """Validate a supplied hierarchy or build L3-L6 via Main Brain."""
        onto = Ontology.skeleton(spec.premise)
        if spec.supplied_ontology is not None:
            supplied = spec.supplied_ontology
            self._fill_missing_levels(supplied)
            self.validate_hierarchy_shape(supplied)
            return supplied
        cache_path = self.lattice_dir / "expansion.json"
        expansions: dict[str, Any] = {}
        if cache_path.exists():
            expansions = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            domains = onto.children_of(onto.root_id)
            reference_text = " ".join(r.sample_text for r in spec.references)[:1500]
            for d_idx, domain in enumerate(domains):
                funcs = [n.label for n in onto.children_of(domain.id)]
                prompt = (
                    f"Premise: {spec.premise}\n"
                    f"Guiding words: {', '.join(spec.guiding_words)}\n"
                    f"Domain: {domain.label}\n"
                    f"Funcs: {', '.join(funcs)}\n"
                    f"Reference excerpts: {reference_text}\n"
                    "For each func list 3 to 5 modes, 3 to 6 moves and 1 to 3 purposes "
                    "fitting the premise. Return JSON only, no prose: "
                    '{"<func>": {"modes": [...], "moves": [...], "purposes": [...]}}'
                )
                response = asyncio.run(
                    self.chat_model.ainvoke([Message(role="user", content=prompt)])
                )
                parsed = _extract_json_object(response.text) or {}
                for f_idx, func in enumerate(funcs):
                    raw = parsed.get(func, {})
                    if not raw:
                        exp = _default_expansion(d_idx, f_idx)
                    else:
                        exp = _clamp_expand(
                            list(raw.get("modes", [])),
                            list(raw.get("moves", [])),
                            list(raw.get("purposes", [])),
                        )
                    expansions.setdefault(domain.label, {})[func] = exp
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(expansions), encoding="utf-8")
        self._apply_expansion(onto, expansions)
        self.validate_hierarchy_shape(onto)
        return onto

    def _fill_missing_levels(self, onto: Ontology) -> None:
        """A supplied tree may stop at L2; fill L3-L6 deterministically (spec:
        missing nodes are filled, never left empty)."""
        domains = onto.children_of(onto.root_id)
        expansions: dict[str, Any] = {}
        for d_idx, domain in enumerate(domains):
            for f_idx, func in enumerate(onto.children_of(domain.id)):
                if onto.children_of(func.id):
                    continue
                exp = expansions.get(domain.label, {}).get(func.label)
                if exp is None:
                    exp = _default_expansion(d_idx, f_idx)
                    expansions.setdefault(domain.label, {})[func.label] = exp
                self._expand_func(onto, func, exp)

    def _apply_expansion(self, onto: Ontology, expansions: dict[str, Any]) -> None:
        domains = onto.children_of(onto.root_id)
        for domain in domains:
            funcs = onto.children_of(domain.id)
            for f_idx, func in enumerate(funcs):
                exp = expansions.get(domain.label, {}).get(func.label)
                if exp is None:
                    exp = _default_expansion(0, f_idx)
                self._expand_func(onto, func, exp)

    def _expand_func(self, onto: Ontology, func: OntologyNode, exp: dict[str, list[str]]) -> None:
        for m_idx, mode in enumerate(exp["modes"]):
            mode_id = f"{func.id}-M{m_idx + 1:02d}"
            onto.add(
                OntologyNode(id=mode_id, label=mode, level=3, parent_id=func.id),
                parent_id=func.id,
            )
            for s_idx, stance in enumerate(STANCE_VOCAB):
                stance_id = f"{mode_id}-S{s_idx + 1:02d}"
                onto.add(
                    OntologyNode(id=stance_id, label=stance, level=4, parent_id=mode_id),
                    parent_id=mode_id,
                )
                for mv_idx, move in enumerate(exp["moves"]):
                    move_id = f"{stance_id}-MV{mv_idx + 1:02d}"
                    onto.add(
                        OntologyNode(id=move_id, label=move, level=5, parent_id=stance_id),
                        parent_id=stance_id,
                    )
                    for p_idx, purpose in enumerate(exp["purposes"]):
                        onto.add(
                            OntologyNode(
                                id=f"{move_id}-P{p_idx + 1:02d}",
                                label=purpose,
                                level=6,
                                parent_id=move_id,
                            ),
                            parent_id=move_id,
                        )

    @staticmethod
    def validate_hierarchy_shape(onto: Ontology) -> None:
        """Acceptance gate: fixed levels, 12 domains x 14 funcs, branching limits."""
        if onto.count_at_level(0) != 1:
            raise BuildValidationError("hierarchy must have exactly one L0 root")
        domains = onto.children_of(onto.root_id)
        if len(domains) != 12:
            raise BuildValidationError(f"L1 must have 12 domains, found {len(domains)}")
        for domain in domains:
            funcs = onto.children_of(domain.id)
            if len(funcs) != 14:
                raise BuildValidationError(f"{domain.id} must have 14 funcs, found {len(funcs)}")
            for func in funcs:
                modes = onto.children_of(func.id)
                if not 3 <= len(modes) <= 5:
                    raise BuildValidationError(f"{func.id} L3 branching must be 3-5")
                for mode in modes:
                    stances = onto.children_of(mode.id)
                    if len(stances) != 3:
                        raise BuildValidationError(f"{mode.id} L4 branching must be 3")
                    for stance in stances:
                        moves = onto.children_of(stance.id)
                        if not 3 <= len(moves) <= 6:
                            raise BuildValidationError(f"{stance.id} L5 branching must be 3-6")
                        for move in moves:
                            purposes = onto.children_of(move.id)
                            if not 1 <= len(purposes) <= 3:
                                raise BuildValidationError(f"{move.id} L6 branching must be 1-3")

    # --- leaves ---------------------------------------------------------------
    def build_leaves(
        self,
        spec: InstantiationSpec,
        onto: Ontology,
    ) -> tuple[list[OscRecord], int, int]:
        """One record per L6 node (L7 leaf 1:1), deduped, subset to target.

        Returns (selected_records, merged_duplicates, total_leaves).
        """
        hz_base = dict(INSTANTIATION_FUNC_HZ)
        for func_name, value in spec.hz_overrides.items():
            hz_base[func_name] = float(value)

        domain_nodes = onto.children_of(onto.root_id)
        records: list[OscRecord] = []
        seen_paths: dict[str, OscRecord] = {}
        seen_labels: dict[tuple[str, str, str, str, str, str], OscRecord] = {}
        merged = 0
        for d_idx, domain in enumerate(domain_nodes):
            domain_code = domain.label.split(" ")[0]
            for f_idx, func in enumerate(onto.children_of(domain.id)):
                for m_idx, mode in enumerate(onto.children_of(func.id)):
                    for s_idx, stance in enumerate(onto.children_of(mode.id)):
                        for mv_idx, move in enumerate(onto.children_of(stance.id)):
                            for p_idx, purpose in enumerate(onto.children_of(move.id)):
                                osc_id = (
                                    f"OSC-{domain_code}"
                                    f"-F{f_idx + 1:02d}-M{m_idx + 1:02d}"
                                    f"-S{s_idx + 1:02d}-MV{mv_idx + 1:02d}"
                                    f"-P{p_idx + 1:02d}"
                                )
                                if osc_id in seen_paths:
                                    seen_paths[osc_id].merged_boost = min(
                                        1.0, seen_paths[osc_id].merged_boost + 0.1
                                    )
                                    merged += 1
                                    continue
                                path_key = (
                                    domain.label,
                                    func.label,
                                    mode.label,
                                    stance.label,
                                    move.label,
                                    purpose.label,
                                )
                                if path_key in seen_labels:
                                    seen_labels[path_key].merged_boost = min(
                                        1.0, seen_labels[path_key].merged_boost + 0.1
                                    )
                                    merged += 1
                                    continue
                                func_label = (
                                    CODE_FUNC_MAP.get(func.label, func.label)
                                    if spec.code_lattice
                                    else func.label
                                )
                                path = {
                                    "root": spec.premise,
                                    "domain": domain.label,
                                    "func": func_label,
                                    "mode": mode.label,
                                    "stance": stance.label,
                                    "move": move.label,
                                    "purpose": purpose.label,
                                    "guiding_words": ", ".join(spec.guiding_words),
                                }
                                codes = {
                                    "domain": d_idx,
                                    "func": f_idx,
                                    "mode": m_idx,
                                    "stance": s_idx,
                                    "move": mv_idx,
                                    "purpose": p_idx,
                                }
                                base = hz_base.get(func_label, 0.5)
                                hz = base * (0.9 + 0.2 * float(self.rng.random()))
                                emb = self.embedder.embed_texts(
                                    [" | ".join(f"{k}: {v}" for k, v in path.items())]
                                )[0]
                                record = OscRecord(
                                    osc_id=osc_id,
                                    batch=0,
                                    path=path,
                                    codes=codes,
                                    hz=float(np.clip(hz, 0.01, 1.5)),
                                    phase=float(self.rng.uniform(0.0, 2.0 * np.pi)),
                                    embedding=emb,
                                )
                                records.append(record)
                                seen_paths[osc_id] = record
                                seen_labels[path_key] = record

        target = RADII[spec.radius]
        selected = self._select_subset(records, target)
        self.events.append(
            {
                "stage": "branching",
                "leaves_total": len(records),
                "target": target,
                "selected": len(selected),
                "merged_duplicates": merged,
            }
        )
        return selected, merged, len(records)

    @staticmethod
    def _select_subset(records: list[OscRecord], target: int) -> list[OscRecord]:
        """Evenly-spaced deterministic subset (keeps branching limits intact)."""
        if target >= len(records):
            return records
        positions = np.unique(np.linspace(0, len(records) - 1, target).astype(np.int64))
        return [records[int(pos)] for pos in positions]

    # --- oscillator creation --------------------------------------------------
    def _append_records(self, records: list[OscRecord]) -> list[int]:
        k = len(records)
        rows = list(range(self.lattice.n, self.lattice.n + k))
        self.lattice.append(
            domain=np.array([r.codes["domain"] for r in records], dtype=np.int16),
            func=np.array([r.codes["func"] for r in records], dtype=np.int16),
            hz=np.array([r.hz for r in records], dtype=np.float32),
            phase=np.array([r.phase for r in records], dtype=np.float32),
            amplitude=np.ones(k, dtype=np.float32),
            embeddings=np.stack([r.embedding for r in records]),
            mode=np.array([r.codes["mode"] for r in records], dtype=np.int16),
            stance=np.array([r.codes["stance"] for r in records], dtype=np.int16),
            move=np.array([r.codes["move"] for r in records], dtype=np.int16),
            ids=np.array([r.osc_id for r in records], dtype=object),
        )
        self._row_by_id.update({r.osc_id: row for r, row in zip(records, rows, strict=False)})
        return rows

    # --- mini skill crafting ---------------------------------------------------
    def _craft_prompt(self, records: list[OscRecord], spec: InstantiationSpec, retry: bool) -> str:
        payload = [
            {
                "oscillatorId": r.osc_id,
                "path": {
                    k: r.path[k] for k in ("domain", "func", "mode", "stance", "move", "purpose")
                },
                "guidingWords": spec.guiding_words,
            }
            for r in records
        ]
        excerpts = [
            f"{ref.name}: {ref.sample_text[:400]}"
            for ref in (spec.references[i % len(spec.references)] for i in range(len(records)))
        ]
        retry_note = (
            "\nThe previous response was not valid JSON or violated the schema. "
            "Return ONLY the JSON array."
            if retry
            else ""
        )
        return (
            "Craft one MiniSkill per oscillator. PromptFragment must be concrete imperative "
            "behavior (not adjectives). Examples must come from the reference excerpts. "
            "Constraints must be testable. bannedPhrases must be specific to the skill.\n"
            "Reference excerpts:\n"
            + "\n".join(excerpts)
            + "\nOSCILLATOR_BATCH_JSON: "
            + json.dumps(payload)
            + " END_OSCILLATOR_BATCH_JSON\n"
            "Return a JSON array, one object per oscillator, with keys: oscillatorId, "
            "trigger, promptFragment, examples (2-5), constraints (2-4), bannedPhrases "
            "(3-8), evaluationCheck." + retry_note
        )

    def _fallback_skill(self, rec: OscRecord, spec: InstantiationSpec) -> MiniSkill:
        """Deterministic, non-generic skill used when Main Brain fails; flagged."""
        path = rec.path
        ref = spec.references[(rec.codes["domain"] + rec.codes["func"]) % len(spec.references)]
        sentences = [s.strip() for s in re.split(r"(?<=[.!?]) ", ref.sample_text) if s.strip()]
        if len(sentences) < 2:
            sentences = [spec.premise, spec.premise]
        verb = path["move"]
        purpose = path["purpose"]
        banned = [
            "In conclusion",
            "As an AI",
            "It is important to note",
            f"as a {path['mode']} {path['func']}",
            f"generic {purpose}",
        ]
        return MiniSkill(
            id=f"MS-{rec.osc_id.removeprefix('OSC-')}",
            oscillator_id=rec.osc_id,
            domain=path["domain"],
            func=path["func"],
            trigger=(
                f"When speaking as {path['func']} in {path['domain']} with a "
                f"{path['mode']} register aimed {path['stance']}"
            ),
            prompt_fragment=(
                f"{verb} to {purpose}. Keep sentences short and concrete, aimed "
                f"{path['stance']}; no meta commentary."
            ),
            examples=[f"[{ref.name}] {sentences[0]}", f"[{ref.name}] {sentences[1]}"],
            constraints=[
                "maxAdjectivesPerSentence: 1",
                "sentenceLengthRule: 8 to 14 words",
                f"mustInclude: {purpose}",
                "mustAvoid: summary",
            ],
            banned_phrases=banned[:5],
            evaluation_check=(
                f"Check: output {verb}s toward {path['stance']} and avoids {banned[0]}."
            ),
            flagged_for_review=True,
        )

    def _skill_from_json(self, item: dict[str, Any], rec: OscRecord) -> MiniSkill | None:
        if str(item.get("oscillatorId", "")) != rec.osc_id:
            return None
        skill = MiniSkill(
            id=f"MS-{rec.osc_id.removeprefix('OSC-')}",
            oscillator_id=rec.osc_id,
            domain=rec.path["domain"],
            func=rec.path["func"],
            trigger=str(item.get("trigger", "")),
            prompt_fragment=str(item.get("promptFragment", "")),
            examples=[str(e) for e in item.get("examples", [])][:5],
            constraints=[str(c) for c in item.get("constraints", [])][:4],
            banned_phrases=[str(b) for b in item.get("bannedPhrases", [])][:8],
            evaluation_check=str(item.get("evaluationCheck", "")),
        )
        errors = validate_skill_fields(
            skill.trigger,
            skill.prompt_fragment,
            skill.examples,
            skill.constraints,
            skill.banned_phrases,
            skill.evaluation_check,
        )
        return None if errors else skill

    async def _craft_batch(
        self,
        ctx: _BatchContext,
        batch_index: int,
        records: list[OscRecord],
        semaphore: asyncio.Semaphore,
    ) -> tuple[list[MiniSkill], int]:
        async with semaphore:
            self._append_records(records)
            prompt = self._craft_prompt(records, ctx.spec, retry=False)
            response = await self.chat_model.ainvoke(
                [Message(role="user", content=prompt)],
                InvokeOptions(temperature=0.2),
            )
            parsed = _extract_json_array(response.text) or []
            by_id = {item.get("oscillatorId"): item for item in parsed if isinstance(item, dict)}
            results: list[MiniSkill | None] = [None] * len(records)
            rejected: list[OscRecord] = []
            for pos, rec in enumerate(records):
                skill = self._skill_from_json(by_id.get(rec.osc_id, {}), rec)
                if skill is None:
                    rejected.append(rec)
                else:
                    results[pos] = skill
            if rejected:
                retry_prompt = self._craft_prompt(rejected, ctx.spec, retry=True)
                retry_response = await self.chat_model.ainvoke(
                    [Message(role="user", content=retry_prompt)],
                    InvokeOptions(temperature=0.4),
                )
                retry_parsed = _extract_json_array(retry_response.text) or []
                retry_by_id = {
                    item.get("oscillatorId"): item
                    for item in retry_parsed
                    if isinstance(item, dict)
                }
                for rec in rejected:
                    pos = records.index(rec)
                    results[pos] = self._skill_from_json(retry_by_id.get(rec.osc_id, {}), rec)
            skills: list[MiniSkill] = []
            for pos, rec in enumerate(records):
                skill = results[pos] or self._fallback_skill(rec, ctx.spec)
                skills.append(skill)
                self.lattice.template_id[self._row_by_id[rec.osc_id]] = skill.id
                self.store.save(skill)
            self.osc_dir.mkdir(parents=True, exist_ok=True)
            (self.osc_dir / f"batch_{batch_index:05d}.json").write_text(
                json.dumps([r.to_json() for r in records]), encoding="utf-8"
            )
            flagged = sum(1 for s in skills if s.flagged_for_review)
            self._complete_batch(ctx, batch_index, len(skills), flagged)
            return skills, flagged

    def _complete_batch(
        self, ctx: _BatchContext, batch_index: int, success: int, flagged: int
    ) -> None:
        ctx.completed.append(batch_index)
        ckpt = ctx.ckpt
        ckpt.completed_batches = sorted(ctx.completed)
        ckpt.last_batch_index = batch_index
        ckpt.total_batches = ctx.batches_total
        ckpt.total_oscillators = self.lattice.n
        ckpt.total_mini_skills = self.store.count()
        ckpt.merged_duplicates = ctx.merged
        ckpt.flagged_for_review = flagged
        save_checkpoint(ckpt, self.base_dir)
        ctx.emit(
            {
                "stage": "batch",
                "batch": batch_index,
                "total_batches": ctx.batches_total,
                "success": success,
                "rejected": flagged,
                "oscillators": self.lattice.n,
                "mini_skills": self.store.count(),
            }
        )

    # --- coupling and dedup ---------------------------------------------------
    def build_coupling(self, records: list[OscRecord]) -> tuple[int, int]:
        """Ontology proximity coupling, top-20 partners per oscillator, no self.

        Same domain+func 0.8, same domain 0.5, same func 0.4, else 0.1;
        merged duplicates add +0.1 to the survivor's partners (clamped).
        Returns (near_duplicate_pairs_reduced, merged_duplicates).
        """
        n = self.lattice.n
        d_codes = self.lattice.domain[:n].astype(np.int32)
        f_codes = self.lattice.func[:n].astype(np.int32)
        boosts = np.zeros(n, dtype=np.float32)
        for rec in records:
            row = self._row_by_id[rec.osc_id]
            boosts[row] = rec.merged_boost
        rows_l: list[np.ndarray] = []
        cols_l: list[np.ndarray] = []
        vals_l: list[np.ndarray] = []
        chunk = 512
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            same_d = d_codes[start:end, None] == d_codes[None, :]
            same_f = f_codes[start:end, None] == f_codes[None, :]
            vals = np.where(
                same_d & same_f,
                0.8,
                np.where(same_d, 0.5, np.where(same_f, 0.4, 0.1)),
            ).astype(np.float32)
            vals[np.arange(end - start), np.arange(start, end)] = 0.0
            vals += boosts[start:end, None]
            np.clip(vals, 0.0, 1.0, out=vals)
            k = min(TOP_PARTNERS, n)
            part = np.argpartition(-vals, k - 1, axis=1)[:, :k]
            rows_l.append(np.repeat(np.arange(start, end), k))
            cols_l.append(part.ravel())
            vals_l.append(vals[np.arange(end - start)[:, None], part].ravel())
        all_rows = np.concatenate(rows_l)
        all_cols = np.concatenate(cols_l)
        all_vals = np.concatenate(vals_l)
        self.lattice.coupling.set_entries(all_rows, all_cols, all_vals)

        flagged_pairs = self._reduce_near_duplicates(n)
        merged_count = int(sum(1 for r in records if r.merged_boost > 0.0))
        return flagged_pairs, merged_count

    def _reduce_near_duplicates(self, n: int) -> int:
        """cosine > 0.98 pairs: keep both oscillators, cut their coupling to 0.2."""
        normed = self.lattice.normalized_embeddings
        pairs_rows: list[np.ndarray] = []
        pairs_cols: list[np.ndarray] = []
        chunk = 512
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            sim = normed[start:end] @ normed.T
            triu = np.triu(np.ones_like(sim, dtype=bool), k=1)
            hot = (sim > 0.98) & triu
            local_r, local_c = np.nonzero(hot)
            pairs_rows.append(local_r + start)
            pairs_cols.append(local_c)
        if not pairs_rows:
            return 0
        rows = np.concatenate(pairs_rows).astype(np.int64)
        cols = np.concatenate(pairs_cols).astype(np.int64)
        if rows.size == 0:
            return 0
        current = np.array(
            [self.lattice.coupling.get(int(r), int(c)) for r, c in zip(rows, cols, strict=False)]
        )
        hot_mask = current > 0.2
        self.lattice.coupling.set_entries(
            rows[hot_mask], cols[hot_mask], np.full(int(hot_mask.sum()), 0.2, dtype=np.float32)
        )
        return int(hot_mask.sum())

    # --- persistence / resume -------------------------------------------------
    def _reload_completed(self, completed: list[int]) -> list[OscRecord]:
        records: list[OscRecord] = []
        for batch_index in sorted(completed):
            path = self.osc_dir / f"batch_{batch_index:05d}.json"
            batch_records = [
                OscRecord.from_json(d) for d in json.loads(path.read_text(encoding="utf-8"))
            ]
            self._append_records(batch_records)
            for rec in batch_records:
                skill = self.store.get_by_oscillator(rec.osc_id)
                if skill is not None:
                    self.lattice.template_id[self._row_by_id[rec.osc_id]] = skill.id
            records.extend(batch_records)
        return records

    # --- main entry -----------------------------------------------------------
    def build(
        self,
        spec: InstantiationSpec,
        progress: ProgressCallback | None = None,
    ) -> BuildResult:
        started = time.perf_counter()

        def emit(event: dict[str, Any]) -> None:
            self.events.append(event)
            if progress is not None:
                progress(event)

        onto = self.ensure_hierarchy(spec)
        leaves, merged, total_leaves = self.build_leaves(spec, onto)
        target = RADII[spec.radius]
        batches_total = int(np.ceil(len(leaves) / BATCH_SIZE))

        ckpt = load_checkpoint(self.base_dir, self.lattice_id) or Checkpoint(
            lattice_id=self.lattice_id, total_batches=batches_total
        )
        completed = list(ckpt.completed_batches)
        self._reload_completed(completed)
        pending = [
            (i, leaves[i * BATCH_SIZE : (i + 1) * BATCH_SIZE])
            for i in range(batches_total)
            if i not in completed
        ]
        for i, batch_records in pending:
            for rec in batch_records:
                rec.batch = i

        if pending:
            ctx = _BatchContext(
                spec=spec,
                ckpt=ckpt,
                completed=completed,
                batches_total=batches_total,
                merged=merged,
                progress=progress,
                emit=emit,
            )
            asyncio.run(self._run_pending(ctx, pending))

        flagged_pairs, merged_count = self.build_coupling(leaves)
        validation = self.validate_build(target, total_leaves)
        failed = [gate for gate, ok in validation.items() if not ok]
        if failed:
            raise BuildValidationError(f"validation gate failed: {failed}")
        trace = self._example_trace(leaves)
        if trace is not None:
            emit({"stage": "example_trace", **trace})
        elapsed = time.perf_counter() - started
        emit(
            {
                "stage": "done",
                "oscillators": self.lattice.n,
                "mini_skills": self.store.count(),
                "elapsed_seconds": elapsed,
            }
        )
        return BuildResult(
            lattice_id=self.lattice_id,
            target=target,
            actual=self.lattice.n,
            total_leaves=total_leaves,
            compose_gap=max(0, target - self.lattice.n),
            batches=batches_total,
            mini_skills=self.store.count(),
            flagged_for_review=sum(1 for s in self.store.all_skills() if s.flagged_for_review),
            merged_duplicates=merged_count,
            near_duplicates_flagged=flagged_pairs,
            elapsed_seconds=elapsed,
            validation=validation,
            events=list(self.events),
        )

    async def _run_pending(
        self,
        ctx: _BatchContext,
        pending: list[tuple[int, list[OscRecord]]],
    ) -> None:
        semaphore = asyncio.Semaphore(ctx.spec.max_concurrent_batches)
        tasks = [
            asyncio.create_task(self._craft_batch(ctx, i, batch, semaphore)) for i, batch in pending
        ]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            # gather does not cancel siblings: stop them so the checkpoint
            # stays a true snapshot of what completed before the failure.
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    def _example_trace(self, leaves: list[OscRecord]) -> dict[str, Any] | None:
        rec = next(
            (r for r in leaves if r.path["func"] == "Voice" and r.path["domain"].startswith("D03")),
            None,
        )
        if rec is None and leaves:
            rec = next((r for r in leaves if r.path["func"] == "Voice"), leaves[0])
        if rec is None:
            return None
        skill = self.store.get_by_oscillator(rec.osc_id)
        return {
            "oscillator": {
                "id": rec.osc_id,
                "path": rec.path,
                "hz": rec.hz,
                "phase": rec.phase,
                "amplitude": 1.0,
                "templateId": skill.id if skill else None,
            },
            "mini_skill": skill.model_dump() if skill else None,
        }

    # --- acceptance gates -----------------------------------------------------
    def validate_build(self, target: int, total_leaves: int) -> dict[str, bool]:
        n = self.lattice.n
        emb_norms = np.linalg.norm(self.lattice.embeddings[:n], axis=1)
        coupling = self.lattice.coupling
        row_sums = coupling.row_sums()
        probe = min(n, 64)
        no_self = all(coupling.get(i, i) == 0.0 for i in range(probe)) if n > 0 else True
        has_partners = bool(np.all(row_sums[:probe] > 0.0)) if n > 0 else True
        count_ok = n == min(target, total_leaves) and n > 0
        return {
            "one_to_one": self.store.count() == n,
            "count_within_target_or_composed": count_ok,
            "embeddings_valid": bool(
                np.all(emb_norms > 0.0) and np.all(np.isfinite(self.lattice.embeddings[:n]))
            )
            if n > 0
            else False,
            "coupling_top_partners": has_partners,
            "no_self_coupling": no_self,
        }
