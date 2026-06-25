# AGENTS.md

This is a wireless-communication simulation repository.

Work as a careful simulation-code engineer. Make minimal localized changes. Do not broadly refactor. Preserve mathematical definitions, metric normalization, random seeds, CLI compatibility, and CSV/result schemas unless explicitly asked.

Project-specific constraints:
- Never modify physics formulas just to improve metrics.
- Never claim Stage-II is guaranteed to improve estimation unless ablation and raw-domain metrics support it.
- For RIS projection, the near-field structure is in `g_k`; the CPD factor is `c_k = Omega_k g_k`.
- For delay projection, `B` and `Q` share a mother delay factor and must not be projected independently.

Do not run full Monte Carlo experiments unless explicitly requested. Do not create new `test_*.py` files by default. Validate code changes with the smallest deterministic smoke run, preferably `--n-trials 1`, `--trials 1`, or the script's equivalent option after checking argparse.

Report files inspected, files modified, what changed, the validation command, validation result, and remaining risks.