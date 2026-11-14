"""Static check: no Python for/while loops over oscillators may appear in core/.

Every core operation must be vectorized numpy (or scipy.sparse); loops belong
at the runtime level over ticks, never over oscillator arrays.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

CORE_DIR = Path(__file__).resolve().parents[1] / "oscillang" / "core"

FORBIDDEN_NODES = (ast.For, ast.AsyncFor, ast.While)


def _collect_core_modules() -> list[Path]:
    return sorted(CORE_DIR.glob("*.py"))


@pytest.mark.parametrize("path", _collect_core_modules(), ids=lambda p: p.name)
def test_no_loops_in_core(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"loop statements found in core: {offenders}"


def test_core_modules_exist() -> None:
    names = {p.name for p in _collect_core_modules()}
    expected = {"lattice.py", "coherence.py", "gating.py", "dynamics.py", "coupling.py"}
    missing = expected - names
    assert not missing, f"missing core modules: {missing}"
