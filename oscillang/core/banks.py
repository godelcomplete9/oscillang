"""Domain and functional banks.

D01-D12 are the twelve creative domains. The fourteen behavioral functions are
split into the Internal bank (psychology: voice, emotion, flaw, want, ...)
and the External bank (world: facts, other characters, plot pressure, ...).
Default Hz per function come from the spec's bank tables and are configurable
per oscillator at craft time.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "Domain",
    "Func",
    "Bank",
    "DOMAINS",
    "FUNCS",
    "FUNC_HZ",
    "FUNC_BANK",
    "INTERNAL_FUNCS",
    "EXTERNAL_FUNCS",
]


class Domain(StrEnum):
    """The twelve creative domains."""

    D01 = "D01"
    D02 = "D02"
    D03 = "D03"
    D04 = "D04"
    D05 = "D05"
    D06 = "D06"
    D07 = "D07"
    D08 = "D08"
    D09 = "D09"
    D10 = "D10"
    D11 = "D11"
    D12 = "D12"


class Func(StrEnum):
    """The fourteen behavioral functions."""

    VOICE = "Voice"
    CHARACTER_VOICE = "Character Voice"
    EMOTION = "Emotion"
    FLAW = "Flaw"
    WANT_VS_NEED = "Want vs Need"
    WORLDVIEW = "Worldview"
    RHYTHM = "Rhythm"
    IMAGE = "Image"
    SCENE_OBJECTIVE = "Scene Objective"
    OTHER_CHARACTERS = "Other Characters"
    WORLD_FACTS = "World Facts"
    PLOT_PRESSURE = "Plot Pressure"
    READER_TENSION = "Reader Tension"
    MEMORY = "Memory"


class Bank(StrEnum):
    INTERNAL = "internal"
    EXTERNAL = "external"


DOMAINS: tuple[Domain, ...] = tuple(Domain)
FUNCS: tuple[Func, ...] = tuple(Func)

FUNC_HZ: dict[Func, float] = {
    Func.VOICE: 0.90,
    Func.CHARACTER_VOICE: 0.80,
    Func.EMOTION: 1.10,
    Func.FLAW: 0.35,
    Func.WANT_VS_NEED: 0.25,
    Func.WORLDVIEW: 0.15,
    Func.RHYTHM: 1.30,
    Func.IMAGE: 1.00,
    Func.SCENE_OBJECTIVE: 0.60,
    Func.OTHER_CHARACTERS: 0.70,
    Func.WORLD_FACTS: 0.05,
    Func.PLOT_PRESSURE: 0.45,
    Func.READER_TENSION: 0.85,
    Func.MEMORY: 0.30,
}

FUNC_BANK: dict[Func, Bank] = {
    Func.VOICE: Bank.INTERNAL,
    Func.CHARACTER_VOICE: Bank.INTERNAL,
    Func.EMOTION: Bank.INTERNAL,
    Func.FLAW: Bank.INTERNAL,
    Func.WANT_VS_NEED: Bank.INTERNAL,
    Func.WORLDVIEW: Bank.INTERNAL,
    Func.RHYTHM: Bank.INTERNAL,
    Func.IMAGE: Bank.INTERNAL,
    Func.MEMORY: Bank.INTERNAL,
    Func.SCENE_OBJECTIVE: Bank.EXTERNAL,
    Func.OTHER_CHARACTERS: Bank.EXTERNAL,
    Func.WORLD_FACTS: Bank.EXTERNAL,
    Func.PLOT_PRESSURE: Bank.EXTERNAL,
    Func.READER_TENSION: Bank.EXTERNAL,
}

INTERNAL_FUNCS: tuple[Func, ...] = tuple(f for f in Func if FUNC_BANK[f] is Bank.INTERNAL)
EXTERNAL_FUNCS: tuple[Func, ...] = tuple(f for f in Func if FUNC_BANK[f] is Bank.EXTERNAL)
