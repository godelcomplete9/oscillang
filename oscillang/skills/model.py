"""Data models for instantiation: ontology tree, spec, mini skills, embedder.

The same mechanism instantiates a Book Lattice and a Code Lattice; only the
supplied hierarchy and the func label mapping change.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field

__all__ = [
    "Reference",
    "InstantiationSpec",
    "OntologyNode",
    "Ontology",
    "MiniSkill",
    "RADII",
    "INSTANTIATION_FUNC_HZ",
    "CODE_FUNC_MAP",
    "MODE_VOCAB",
    "MOVE_VOCAB",
    "PURPOSE_VOCAB",
    "STANCE_VOCAB",
    "EmbeddingModel",
    "HashingEmbedder",
]

RADII: dict[int, int] = {
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

INSTANTIATION_FUNC_HZ: dict[str, float] = {
    "Voice": 1.0,
    "Character Voice": 0.8,
    "Emotion": 0.4,
    "Flaw": 0.08,
    "Want vs Need": 0.08,
    "Worldview": 0.05,
    "Rhythm": 1.2,
    "Image": 0.5,
    "Scene Objective": 1.0,
    "Other Characters": 0.6,
    "World Facts": 0.07,
    "Plot Pressure": 0.4,
    "Reader Tension": 0.3,
    "Memory": 0.05,
}

CODE_FUNC_MAP: dict[str, str] = {
    "Voice": "Code Voice",
    "Character Voice": "Reviewer Voice",
    "Emotion": "Urgency",
    "Flaw": "Failure Mode",
    "Want vs Need": "Task vs Constraint",
    "Worldview": "Architecture View",
    "Rhythm": "Edit Cadence",
    "Image": "Code Pattern",
    "Scene Objective": "Task Objective",
    "Other Characters": "Other Agents",
    "World Facts": "Repo Facts",
    "Plot Pressure": "Deadline Pressure",
    "Reader Tension": "Review Tension",
    "Memory": "Repo Memory",
}

# Ontology vocabularies (superset of the lattice's default tuples).
MODE_VOCAB: tuple[str, ...] = ("precise", "playful", "direct", "withheld", "evasive")
STANCE_VOCAB: tuple[str, ...] = ("toward self", "toward other", "toward world")
MOVE_VOCAB: tuple[str, ...] = (
    "question",
    "dissect",
    "confess",
    "joke",
    "evade",
    "summarize",
    "plan",
    "test",
    "edit",
    "read",
    "search",
    "review",
)
PURPOSE_VOCAB: tuple[str, ...] = (
    "clarify",
    "persuade",
    "debug",
    "reveal",
    "hide",
    "test",
    "refactor",
)

DOMAIN_LABELS: dict[str, str] = {
    "D01": "Identity",
    "D02": "Resources",
    "D03": "Communication",
    "D04": "Home",
    "D05": "Creation",
    "D06": "Work",
    "D07": "Partnership",
    "D08": "Transformation",
    "D09": "Meaning",
    "D10": "Public Role",
    "D11": "Collective",
    "D12": "Subconscious",
}


class Reference(BaseModel):
    """One reference source: author, repo, doc, etc."""

    name: str
    sample_text: str


class InstantiationSpec(BaseModel):
    """Inputs for a lattice build."""

    premise: str
    references: list[Reference] = Field(min_length=1, max_length=5)
    guiding_words: list[str] = Field(min_length=3, max_length=7)
    radius: int = Field(ge=1, le=10)
    hz_overrides: dict[str, float] = Field(default_factory=dict)
    code_lattice: bool = False
    supplied_ontology: Ontology | None = None  # set programmatically, not from JSON
    lattice_id: str = "lattice"
    max_concurrent_batches: int = Field(default=3, ge=1, le=3)
    core_cap: int = Field(default=5000, ge=1)  # max fully crafted oscillators


class OntologyNode(BaseModel):
    id: str
    label: str
    level: int = Field(ge=0, le=7)
    parent_id: str | None = None
    children: list[str] = Field(default_factory=list)
    description: str = ""
    examples: list[str] = Field(default_factory=list)


class Ontology(BaseModel):
    """The hierarchy of ontology: L0 root .. L7 oscillator leaves."""

    root_id: str
    nodes: dict[str, OntologyNode]

    def get(self, node_id: str) -> OntologyNode:
        return self.nodes[node_id]

    def children_of(self, node_id: str) -> list[OntologyNode]:
        return [self.nodes[cid] for cid in self.nodes[node_id].children]

    def add(self, node: OntologyNode, parent_id: str | None = None) -> None:
        self.nodes[node.id] = node
        if parent_id is not None:
            self.nodes[parent_id].children.append(node.id)

    def count_at_level(self, level: int) -> int:
        return sum(1 for n in self.nodes.values() if n.level == level)

    def leaf_paths(self) -> list[list[OntologyNode]]:
        """Paths root->leaf for every L6 node (L7 leaf = 1:1 with L6)."""
        paths: list[list[OntologyNode]] = []

        def walk(node: OntologyNode, path: list[OntologyNode]) -> None:
            extended = path + [node]
            if not node.children:
                paths.append(extended)
                return
            for child_id in node.children:
                walk(self.nodes[child_id], extended)

        walk(self.nodes[self.root_id], [])
        return paths

    @classmethod
    def skeleton(cls, premise: str) -> Ontology:
        """L0 premise + 12 domains + 14 funcs each = 168 L2 nodes. No exceptions."""
        root = OntologyNode(id="L0-ROOT", label=premise, level=0, description="The premise.")
        onto = cls(root_id=root.id, nodes={})
        onto.add(root)
        func_labels = list(INSTANTIATION_FUNC_HZ)
        for _d_idx, (d_code, d_label) in enumerate(DOMAIN_LABELS.items()):
            d_id = f"L1-{d_code}"
            onto.add(
                OntologyNode(
                    id=d_id,
                    label=f"{d_code} {d_label}",
                    level=1,
                    parent_id=root.id,
                    description=f"Domain {d_label}.",
                ),
                parent_id=root.id,
            )
            for f_idx, func in enumerate(func_labels):
                onto.add(
                    OntologyNode(
                        id=f"L2-{d_code}-F{f_idx:02d}",
                        label=func,
                        level=2,
                        parent_id=d_id,
                        description=f"{func} within {d_label}.",
                    ),
                    parent_id=d_id,
                )
        return onto


class MiniSkill(BaseModel):
    """One behavioral fragment, exactly one per oscillator."""

    id: str
    oscillator_id: str
    domain: str
    func: str
    trigger: str = ""
    prompt_fragment: str = ""
    examples: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    banned_phrases: list[str] = Field(default_factory=list)
    evaluation_check: str = ""
    composed: bool = False
    promoted: bool = False
    used_count: int = 0
    flagged_for_review: bool = False


class EmbeddingModel(Protocol):
    """Embedding seam; swap for a real model, keep HashingEmbedder for tests."""

    dim: int

    def embed_texts(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic feature-hashing embedder (no deps, stable across runs).

    Tokens (plus adjacent bigrams) are crc32-hashed into a fixed-size vector;
    occurrences are summed, then the vector is L2-normalized.
    """

    def __init__(self, dim: int = 32) -> None:
        self.dim = int(dim)

    def _index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") % self.dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = [t for t in text.lower().replace("|", " ").split() if t]
            grams: list[str] = list(tokens)
            grams.extend(f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False))
            for gram in grams:
                out[row, self._index(gram)] += 1.0
        norms = np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)
        return (out / norms).astype(np.float32)


InstantiationSpec.model_rebuild()
