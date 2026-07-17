# RIS-EVS-OFDM simulation

The frozen paper estimator is **MKSC-GI-balanced -> CCOP-JVP**.  CP-NGC,
conditional assignment rescue, and the older Stage-II/JNPP routes are not part
of the final selector; their reproducibility code is archived under
[`oldcode/`](oldcode/README.md).

## Experiments

The authoritative paper commands are version controlled in
[`scripts/paper_commands.sh`](scripts/paper_commands.sh). Each formal target
has a fixed seed, grid, trial count, bootstrap count, and output directory.
The script refuses a dirty worktree, a commit other than the annotated paper
freeze tag, or an existing result directory. Before any server run, use:

```bash
bash scripts/paper_commands.sh preflight
bash scripts/paper_commands.sh list
PAPER_DRY_RUN=1 bash scripts/paper_commands.sh components_paper400
```

Run `smoke_ablation` and `smoke_robustness` before their corresponding formal
targets. The script deliberately has no `all` target, so a typo cannot launch
every Monte Carlo suite.

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
  --out-dir results/final_mksc_ccop/readme_smoke1
```

This route generates the noisy observation once, runs four-start MKSC-GI with
one Jones-anchor refresh, and then runs independent three-dimensional
CCOP-JVP. It does not call CP-NGC or rescue.

The pre-MKSC ablation runner remains available only for historical
reproducibility:

```bash
python -m oldcode.legacy_stage2.run_proposed_ablation --help
```

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
  --baselines "als_cpd,ff_omp,ris_momp,nf_mmpsr,nf_ris_groupomp_localgrid_wls,scaled_4d,mksc_ccop,peb,constrained_jones_peb" \
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
- `FF-OMP`: a far-field angular-delay sparse baseline adapted to the current
  raw EVS-RIS-OFDM observation.
- `RIS-MOMP`: a RIS-aided multidimensional OMP-style sparse baseline with
  independent direction and delay group grids.
- `NF-MMPSR`: a near-field spherical-domain grid sparse baseline with
  top-candidate local CC refinement.
- `NF-RIS-GroupOMP-LocalGrid-WLS`: an adapted near-field RIS
  localization/synchronization baseline using group-OMP coarse estimation,
  deterministic local-grid refinement, and geometry WLS:
  `--baselines "als_cpd,ff_omp,ris_momp,nf_mmpsr,nf_ris_groupomp_localgrid_wls,scaled_4d,mksc_ccop,peb"`.
- `mksc_ccop`: the frozen MKSC-GI-balanced -> CCOP-JVP paper estimator.
- `PEB`: the data-only Free-Jones EFIM/CRB reference curve. Plots that show
  this curve also include a Constrained-Jones PEB reference unless disabled.

All non-MKSC baselines use the same generated noisy data for each
seed/SNR/K and are restricted to discrete dictionaries, CP factorization,
linear LS over selected atoms, and neutral geometry LS post-processing.

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
  --baselines "mksc_ccop,ff_omp,ris_momp,nf_mmpsr,peb" \
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
