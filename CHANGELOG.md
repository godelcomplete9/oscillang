# Changelog

## 0.1.0 - Milestone 1: core dynamics

### Added
- `oscillang/core/config.py`: `Config` dataclass with every spec default
  (lock tolerance 0.4 / threshold 0.6, decay-per-use 0.08, recovery 0.02,
  refractory 2 fast / 5 slow, coupling delta limit 0.1 and decay 0.999,
  noise 0.05, gains 1.0/0.3/0.5, Vth 0.65) plus per-mechanism ablation flags.
- `oscillang/core/banks.py`: `Domain` (D01-D12) and `Func` (14 functions)
  enums with Internal/External bank assignment and default Hz per function.
- `oscillang/core/lattice.py`: struct-of-arrays `OscillatorLattice`
  (phase, hz, amplitude, refractory, embeddings as flat numpy arrays; pydantic
  `Oscillator` snapshot model), vectorized `generate` and `extend_composed`
  (V = V_domain + V_func + V_mode + V_stance + V_move + V_purpose), lazy
  L2-normalized embedding cache, capacity growth, coupling auto-resize.
- `oscillang/core/coupling.py`: `CouplingMatrix` with dense backend below
  20,000 oscillators and scipy.sparse CSR at/above; lazy scalar decay (O(1)
  per tick), vectorized `apply_delta` (delta clamp ±0.1, result clamp [0,1],
  duplicates accumulate), `row_sums`, `effective`, `scale_global`.
- `oscillang/core/dynamics.py`: vectorized phase advance (`(phase + 2π·hz·step)
  % 2π`), optional Kuramoto term via the sin/cos matvec identity (default off),
  fatigue (`amp*decayPerUse + recoveryPerTick` on use, recovery + drift toward
  baseline 0.5 otherwise, clamped [0,1]), refractory windows (2 ticks fast
  hz>=0.6, 5 ticks slow), and the `step_lattice` single-call tick update.
- `oscillang/core/coherence.py`: `V @ V.T` on L2-normalized embeddings
  (full/pairs/query variants) and `compose_state_vector` for query-aware
  coherence state vectors.
- `oscillang/core/gating.py`: wrapped phase diff ([0, π]), batched lock gating
  requiring BOTH `phaseDiff < 0.4` AND `coherence > 0.6` (high similarity with
  low phase lock does not gate), reference gating, and batched top-K retrieval.
- Tests: lock-resonance invariants, dense/sparse coupling equivalence, dynamics
  unit tests, lattice/composed construction, Config invariants, a static check
  that fails if any for/while loop appears in `core/`, and performance tests.
- Packaging: `pyproject.toml` (numpy/pydantic/scipy deps, torch optional),
  ruff + mypy + pytest-cov configured, `py.typed`.

### Performance (measured, Windows, Python 3.11)
- 2,000 oscillators, 100 ticks: p50 ≈ 4.8 ms/tick (budget 50 ms) — dense
  coupling row sums dominate.
- 20,000 oscillators, 100 ticks: p50 ≈ 0.8 ms/tick (budget 150 ms) — sparse
  coupling backend kicks in at 20k, so the O(N²) read disappears.
- 200,000 composed oscillators: build + 3 ticks ≈ 170 ms total (budget 400 ms).
- `core/` coverage: 94% (target >85%). ruff clean, mypy clean.

## 0.2.0 - Milestone 5: Jev is the semantic gatekeeper (redesign)

### Changed
- **Jev's role corrected**: Jev verifies semantic coherence and alignment -
  it is a reranker and the main gatekeeper of a mini skill. No embedding
  cosine anywhere in the fire path; the oscillator layer is dynamics only.
- `oscillang/core/gating.py`: lock gating is phase-only
  (`phaseDiff < lockTolerance`, refractory veto); the coherence condition and
  `lock_threshold` are gone. `phase_gating` remains as the ablation switch.
- `oscillang/core/config.py`: dropped `w_coh`, `w_mem`, `g_sim`, `g_self`,
  `query_aware`, `lock_threshold`; added `w_rec` (recency boost) and
  `jev_shortlist` (reranker shortlist size, default 24).
- `oscillang/jev/`: new contracts - `JevContext` is dynamics-only (phase
  proximity, recency, amplitude, use cost, coupling, refractory, noise);
  `JevModel.score` is the numeric role (RuleJev prefilter, MlpJev learned
  hook); `JevReranker.rerank` is the semantic role: candidates arrive as
  `JevCandidate` (actual trigger/fragment/examples texts) and come back as
  `JevVerdict` alignment scores. `LlmJev` is the default reranker (one
  schema-validated call per fire decision; silence never fires).
- `oscillang/runtime/`: new fire flow - uniform `g_ext` phase drive, dynamics
  prefilter shortlist, LlmJev alignment verification, then the hard gate
  (verdict above adaptive `v_threshold`, phase-locked to the highest-verdict
  leader, not refractory, capped at `top_k`). `MemoryTrace` (embedding EMA)
  replaced by `RecencyTrace` (per-skill last-fired decay, `alpha**age`, no
  embeddings). `world_model` now feeds the previous beat to the reranker as
  context. Guards consume the best alignment verdict.

### Tests
- Full rewrite of `test_gating`, `test_jev`, `test_runtime` for the phase-only
  gate and the reranker flow; `test_performance` ticks the phase-only path.
- Suite: 234 tests, core/skills/llm coverage 95% (target >85%). ruff clean,
  mypy clean.
