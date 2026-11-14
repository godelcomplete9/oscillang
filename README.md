# oscillang

Oscillator-gated agent nervous system for LLMs. A cheap decision model (**Jev**)
decides in parallel which behavioral fragments gate, fire, wait, avoid, or
forget. The generative LLM (**Main Brain**) only renders inside the decision.
Jev never generates text; Main Brain never decides.

Status: Milestone 1 (`core/` dynamics) in progress. See CHANGELOG.md.
