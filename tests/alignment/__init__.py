"""F7 — Alignment Audit Suite (out-of-pipeline, dedicated tests).

Replicates the paper's 3 causal experiments (blackmail, reward hacking,
sycophancy) as a per-mode harness, and asserts the non-negotiable ethical
invariant: RESIDUUM steering never exceeds the mode cap (0.15 absolute ceiling),
so the granular/expanded steering cannot be used to push behaviour past the
documented envelope. The behavioural rollouts (actually running the LLM) are
GPU-gated behind RESIDUUM_RUN_ALIGNMENT=1; the CI layer is fully deterministic.
"""
