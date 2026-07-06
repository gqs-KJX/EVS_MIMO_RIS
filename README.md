# RIS-EVS-OFDM simulation

## Experiments

Run one small fixed-SNR proposed-method demo from the project root:

```bash
python -m src.main_single_proposed
```

The single diagnostic run generates one synthetic RIS-EVS-OFDM channel sample,
builds the Hankelized tensor for Stage-I initialization, applies the current
NGC-certified RIS-only rescue policy when triggered, and runs the current
proposed final refinement: adaptive Stage-I-regularized Jones-VP in the raw
OFDM domain.

Run the current proposed ablation entry point with one ablation group at a time:

```bash
python -m src.experiments.run_proposed_ablation --ablation vp_family --n-trials 20 --snr-db -20 --out results/vp_family_ablation.csv
python -m src.experiments.run_proposed_ablation --ablation stage2_gate --n-trials 20 --snr-db -20 --out results/stage2_gate_ablation.csv
python -m src.experiments.run_proposed_ablation --ablation jones_lambda --n-trials 20 --snr-db -20 --out results/jones_lambda_ablation.csv
```

The VP-family ablation modes are `fixed_pol`, `jones_free`,
`jones_regularized`, and `adaptive_jones`. The `run_stage2_ablation.py` script
is legacy and studies the older EVS/delay/RIS Stage-II projection modules; it
is not the main ablation entry point for the revised adaptive Jones-VP paper
algorithm.

## Paper ablation figures

Generate the revised paper ablation CSVs, logs, metadata, summaries, and PDF
figures with:

```bash
python -m src.experiments.run_paper_ablation_figures \
  --figures all \
  --n-trials 100 \
  --paper-k 3 \
  --snr-grid "-10,-5,0,5,10,15,20" \
  --jobs 1 \
  --process-workers 1 \
  --blas-threads 4 \
  --global-vp-backend cupy \
  --include-constrained-jones-peb \
  --out-dir results/ablation_paper_final_n100_gpu \
  --force-rerun
```

The paper figure runner targets the revised pipeline: Stage-I initialization,
NGC-certified conditional RIS-only rescue, and adaptive Stage-I-regularized
Jones-VP. Figures 1--5 use `K=3` by default through the actual simulation
configuration. Figure 5 is the final NGC rescue ablation and plots both
outlier probability and rescue trigger rate. Figure 6 ignores `--paper-k` and
sweeps `K=1,2,3,4` at 0 dB. Figure 6 should be interpreted as the complete NGC
proposed system versus reduced polarization-model variants. It also includes
an adaptive-Jones VP without rescue arm to separate polarization modeling from
the NGC rescue mechanism.
PEB curves include the existing data-only Free-Jones EFIM/CRB reference and,
by default, a Constrained-Jones oracle polarimetric-anchor reference. The older
`run_stage2_ablation.py` entry point remains available only for legacy
structured-refinement module ablations.

Figures 3 and 4 use a nested-receiver fixed-noise convention for the
scalar/dual/full EVS comparison. The displayed SNR is referenced to the
full-6D EVS observation, and all three receiver modes share the same
per-component noise variance and underlying full-6D noise realization.
Scalar and dual observations are masks of that common realization. This is
the convention used for the EFIM information-ordering comparison.

The paper runner defaults to `--jobs 10`, `--task-grouping grouped`, and
`--blas-threads auto`. Grouped execution reuses data generation and Stage-I
initialization within each Monte Carlo trial before evaluating the requested
VP/NGC variants; it does not change the estimator or physical channel model.

`--jobs` is the total CPU-slot budget. `--process-workers` controls the number
of memory-heavy worker processes, and `--blas-threads` controls native compute
threads per worker. For a higher-CPU run when memory permits, use
`--jobs 30 --process-workers 6 --blas-threads 5`. Avoid many processes with
`--blas-threads 1` when memory is already near capacity.

## Benchmark comparison figures

Run the standalone benchmark comparison entry point with:

```bash
python -m src.experiments.run_benchmark_comparison \
  --n-trials 50 \
  --paper-k 3 \
  --snr-grid "-30,-25,-20,-15,-10,-5,0,5,10" \
  --baselines "als_cpd,ff_omp,ris_momp,nf_mmpsr,nf_ris_groupomp_localgrid_wls,proposed,peb" \
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
  `--baselines "als_cpd,ff_omp,ris_momp,nf_mmpsr,nf_ris_groupomp_localgrid_wls,proposed,peb"`.
- `NGC-Jones-VP`: the only curve using the current NGC-certified adaptive
  Jones-VP proposed pipeline.
- `PEB`: the data-only Free-Jones EFIM/CRB reference curve. Plots that show
  this curve also include a Constrained-Jones PEB reference unless disabled.

All non-proposed baselines use the same generated noisy data for each
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
  --baselines "proposed,ff_omp,ris_momp,nf_mmpsr,peb" \
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
