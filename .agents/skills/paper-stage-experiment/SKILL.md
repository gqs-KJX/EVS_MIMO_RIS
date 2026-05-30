---
name: paper-stage-experiment
description: Use this skill when designing publication-grade experiments, ablations, baselines, tables, and figures for a wireless communications / signal processing paper.
---

# Paper Stage Experiment Skill

## Purpose

Design staged, publication-grade experiments after the code passes minimal debug validation.

## Experiment Stages

### Stage 0: Smoke Tests

- 3 to 5 seeds.
- One SNR.
- Small dimensions if possible.
- Verify no NaN/Inf, no assignment crash, no optimizer fallback unless expected.

### Stage 1: Minimal Validity Tests

- 20 to 50 seeds.
- Compare:
  - Stage-I only
  - each projection module alone
  - current full Stage-II
  - guarded full Stage-II
  - final VP-WNLS if available
- Metrics:
  - median
  - p90
  - p95
  - failure rate
  - runtime
  - assignment accuracy
  - raw-domain NMSE
  - position/range/delay RMSE

### Stage 2: Claim Validation

- 100 to 300 seeds.
- Multiple SNR points.
- Multiple training overhead settings.
- Include all baselines required by the paper.
- Produce CSV and summary tables.

### Stage 3: Final Paper Figures

- Fixed configs.
- Frozen seeds.
- Full baselines.
- Confidence intervals or percentile bands.
- Runtime scalability.
- Ablation table.
- Failure case discussion.

## Required Publication Checks

For each claimed contribution:
1. Identify the exact experiment supporting it.
2. Identify the baseline it beats.
3. Identify whether it improves median, p90/p95, success rate, runtime, or robustness.
4. If improvement is only local residual but not final estimation, do not claim estimation improvement.
5. If Stage-II does not consistently improve median but reduces catastrophic failures, state that precisely.

## Stage-II Structured Projection Ablations

Required ablations:
- Stage-I only.
- Stage-I + delay projection only.
- Stage-I + RIS projection only.
- Stage-I + EVS projection only.
- Stage-I + current full Stage-II.
- Stage-I + guarded Stage-II.
- Stage-I + guarded Stage-II + raw-domain VP.
- Oracle assignment.
- Oracle delay.
- Oracle RIS geometry.
- Noiseless sanity check.
- SNR sweep.
- Training overhead sweep.

## Required Outputs

Every experiment script must:
- save config JSON
- save CSV
- save summary markdown
- save plots only from CSV
- print exact command used
- record git commit hash
- record Python/package versions
- record whether scipy optimizer was available

## Forbidden Behavior

- Do not use a single seed to support a paper claim.
- Do not report only median when failure tails are important.
- Do not compare guarded and unguarded algorithms using different noisy data.
- Do not omit a baseline because it performs well.
- Do not claim CRB-achieving behavior unless the estimator and likelihood match the CRB assumptions.
