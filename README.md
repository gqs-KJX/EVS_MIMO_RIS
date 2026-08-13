# RIS-EVS-OFDM simulation

The frozen paper estimator is **MKSC-GI-balanced -> CCOP-JVP**.  CP-NGC,
conditional assignment rescue, and the older Stage-II/JNPP routes are not part
of the final selector.  Historical implementations have been removed from the
active tree and remain recoverable from Git history.

## Experiments

The authoritative v3 paper campaign is version controlled in
[`scripts/run_paper_v3_queue.sh`](scripts/run_paper_v3_queue.sh). Each suite
has a fixed seed, grid, trial count, bootstrap count, and output directory
under `results/paper_v3/`. The smaller pipeline verification queue is
[`scripts/run_paper_v3_verify.sh`](scripts/run_paper_v3_verify.sh).

Regenerate the publication figures from saved v3 CSVs without running an
experiment:

```bash
python scripts/make_paper_figures_twc.py \
  --campaign v3 --snr-min -10 --out-dir tex/figs
```

Run one deterministic realization of the frozen route from the project root:

```bash
python -m src.experiments.run_final_mksc_ccop_ablation \
  --suites components \
  --component-variants proposed \
  --n-trials 1 \
  --snr-grid=-10 \
  --diagnostic-mode fast \
  --jobs 1 \
  --blas-threads 1 \
  --out-dir results/readme_smoke1
```

This route generates the noisy observation once, runs four-start MKSC-GI with
one Jones-anchor refresh, and then runs independent three-dimensional
CCOP-JVP. It does not call CP-NGC or rescue.

## Paper ablation figures

Generate the final paired component-ablation CSVs, hashes, summaries, and
confidence intervals with:

```bash
python -m src.experiments.run_final_mksc_ccop_ablation \
  --suites components \
  --component-variants scaled_4d,old_stage1_ccop,mksc_delay_ccop,mksc_gi_1_no_refresh_ccop,mksc_gi_4_no_refresh_ccop,proposed \
  --n-trials 100 \
  --snr-grid=-20,-15,-10,-5,0,5,10,15,20 \
  --diagnostic-mode performance \
  --jobs 24 \
  --blas-threads 1 \
  --out-dir results/final_mksc_ccop/ablation_paper100
```

All variants inside a paired task share the same generated observation and
trial seed. The older large figure runner remains in `src` because the
benchmark and robustness entry points use its common PEB and model-order
helpers; it is not the final estimator entry point.

The compact main-text table intentionally starts from scale-normalized 4-D VP.
The unscaled 4-D route, seven-start route, and oracle initialization remain
available as appendix diagnostics through explicit `--component-variants`.

The same runner also provides:

```bash
# Position/clock/channel/tail versus SNR.
python -m src.experiments.run_final_mksc_ccop_ablation --suites snr --help

# Nested scalar/dual-pol/full-EVS estimator and matched-PEB comparison.
python -m src.experiments.run_final_mksc_ccop_ablation --suites receiver --help

# Raw-delay versus MKSC single-factor comparison.
python -m src.experiments.run_final_mksc_ccop_ablation --suites compression --help
```

## Benchmark comparison figures

Run the standalone benchmark comparison entry point with:

```bash
python -m src.experiments.run_benchmark_comparison \
  --n-trials 50 \
  --paper-k 3 \
  --snr-grid "-30,-25,-20,-15,-10,-5,0,5,10" \
  --baselines "als_cpd,scaled_4d,nf_ris_groupomp_localgrid_wls,ris_momp,mksc_ccop,peb,constrained_jones_peb" \
  --grid-profile medium \
  --baseline-backend cupy \
  --gpu-device 0 \
  --gpu-batch-size 4096 \
  --include-constrained-jones-peb \
  --out-dir results/benchmark_comparison_gpu_k3_medium \
  --jobs 1 \
  --process-workers 1 \
  --blas-threads 4 \
  --force-rerun
```

The benchmark figure contains:

- `ALS-CPD`: a standalone complex CP tensor baseline. It does not initialize,
  call, or feed the proposed VP.
- `Scale-normalized 4-D Jones-VP`: the frozen scaled-4D comparator.
- `RIS-MOMP adaptation`: independent `u_x`, `u_y`, and delay dictionaries;
  coordinate-wise MOMP source competition; and an orthogonal Jones-group LS
  residual update. It does not build a Cartesian range-angle-delay dictionary.
- `NF-RIS CPD-OMP-SAGE-WLS adaptation`: sequential rank-one CPD, coarse
  delay/direction and exact near-field range recovery, cyclic raw-domain SAGE,
  and local-EFIM-weighted position/clock fusion.
- `mksc_ccop`: the frozen MKSC-GI-balanced -> CCOP-JVP paper estimator.
- `PEB`: the data-only Free-Jones EFIM/CRB reference curve. Plots that show
  this curve also include a Constrained-Jones PEB reference unless disabled.

All methods use the same generated noisy data for each seed/SNR/K. The two
paper-derived methods are explicitly labelled adaptations because the
repository observation model differs from their SISO/hybrid-MIMO source
models; their adaptation contracts are documented in `docs_for_codex/verified`.

## Robustness and system-scaling figures

Generate Figures 8--10 with:

```bash
python -m src.experiments.run_robustness_and_scaling_figures \
  --figures fig8,fig9,fig10a,fig10b,fig10c \
  --n-trials 50 \
  --snr-db 0 \
  --true-k 3 \
  --calibration-std-grid "0,1,2,5,10,20" \
  --assumed-k-grid "2,3,4,5" \
  --T-grid "64,128,256,512" \
  --ris-side-grid "16,24,32,48,64" \
  --baselines "mksc_ccop,als_cpd,scaled_4d,nf_ris_groupomp_localgrid_wls,ris_momp,peb" \
  --baseline-backend cupy \
  --gpu-device 0 \
  --include-constrained-jones-peb \
  --out-dir results/robustness_and_scaling_final_n50 \
  --jobs 1 \
  --process-workers 1 \
  --blas-threads 4 \
  --force-rerun
```

Every algorithm in a trial uses the same physical noisy observation. Figure 8
perturbs only the generated RIS--BS response while estimators retain nominal
calibration. Ordinary matched-model PEB is not a valid bound for this
mismatched experiment and is omitted. The optional
`--include-calibration-oracle-peb` curve is labeled
`Oracle-calibrated PEB (reference)` and is a reference only.

Figure 9 always generates data with the true path count and changes only the
estimator model order. Ordinary PEB is not treated as a bound under path-count
mismatch. The optional `--include-trueK-peb-reference` curve is labeled
`True-K PEB (reference only)`.

Figures 10(a)--10(c) are matched-model scaling studies and include the ordinary
matched-model, data-only Free-Jones PEB by default. All PEB calculations
explicitly Schur-eliminate clock before computing the position PEB. The optional
Anchored-Jones PEB path is disabled by default.

The final-route robustness runner covers physical generation-side
Maxwell/calibration mismatch, paired multi-position generalization with a
free-Jones geometry PEB, and system scaling:

```bash
python -m src.experiments.run_final_mksc_ccop_robustness --help
```

The delay--polarization resolution experiment uses nested receiver masks of one
full-EVS noisy realization and a predeclared success rule (path count, panel
pairing, delay tolerance, and no pole collapse):

```bash
python -m src.experiments.run_final_evs_resolvability --help
```
