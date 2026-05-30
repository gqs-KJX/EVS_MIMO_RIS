# Project Instructions For Codex

This repository implements EVS-RIS-OFDM tensor channel estimation with Stage-I CPD initialization, Stage-II structured projections, and optional raw-domain VP-WNLS refinement.

Always follow these project rules:

1. For PDF-only papers, invoke `$paper-pdf-ingest` before implementing algorithms from the paper.
2. For unexpected simulation results, invoke `$debug-experiment` before tuning hyperparameters.
3. For publication figures, invoke `$paper-stage-experiment` and use staged experiments.
4. Never modify physics formulas just to improve metrics.
5. Never claim Stage-II is guaranteed to improve estimation unless ablation and raw-domain metrics support it.
6. For RIS projection, remember that near-field structure is in `g_k`, while the CPD factor is `c_k = Omega_k g_k`.
7. For delay projection, `B` and `Q` share a mother delay factor and must not be projected independently.
8. Always run tests after code changes.
