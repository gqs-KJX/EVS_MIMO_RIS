---
name: debug-experiment
description: Use this skill when an algorithm or simulation result is worse than expected, unstable across seeds, or contradicts paper claims. Focus on minimal diagnostic experiments before large Monte Carlo runs.
---

# Debug Experiment Skill

## Purpose

Diagnose why a signal-processing / wireless / tensor-estimation algorithm fails or underperforms. This skill is for debugging, not for producing final paper figures.

## Mandatory Principle

Do not run full Monte Carlo first. First isolate the failure using minimal deterministic tests.

## Required Debug Ladder

1. Reproduce one deterministic failing seed.
2. Save full config, seed, dimensions, SNR, and runtime.
3. Run self-tests:
   - tensor unfolding / folding consistency
   - Khatri-Rao ordering
   - complex conjugate correctness
   - LS update residual decrease
   - projection idempotence on true noiseless factors
   - normalization and beta absorption
4. Run oracle tests:
   - true factors + noisy observation
   - true assignment + estimated factors
   - true delay + estimated RIS
   - estimated delay + true RIS
5. Separate modules:
   - Stage-I only
   - EVS only
   - delay only
   - RIS only
   - full Stage-II
   - guarded Stage-II
6. For every accepted update, log:
   - local residual before/after
   - global Z-domain SSE before/after
   - raw-domain Y NMSE before/after when available
   - parameter error before/after if ground truth exists
   - accepted/rejected flag
   - damping rho if used
7. If local residual improves but global objective worsens, classify as:
   - objective mismatch
   - over-projection
   - wrong assignment
   - scale/permutation ambiguity
   - insufficient identifiability
   - optimizer fallback issue

## Stage-II Specific Rules

For CPD / RIS / ISAC algorithms:
- Do not claim hard projection is ML.
- Do not accept local projection updates without a global objective check.
- Do not update compressed RIS factor as if it were an uncompressed Vandermonde factor.
- Do not update B and Q independently if they share a common delay pole.
- Always check column-to-panel assignment consistency.

## Required Output

At the end of a debug task, produce:

1. Root-cause candidates ranked by likelihood.
2. Minimal evidence for each cause.
3. One next code change or experiment.
4. One rollback criterion.
5. A warning if the result does not support the paper claim.

## Forbidden Behavior

- Do not tune hyperparameters blindly.
- Do not run 1000 trials before one failing seed is understood.
- Do not hide negative results.
- Do not change physics formulas to improve a metric without a derivation.
