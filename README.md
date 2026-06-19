# RIS-EVS-OFDM simulation

## Experiments

Run one small fixed-SNR proposed-method demo from the project root:

```bash
python -m src.main_single_proposed
```

The single diagnostic run generates one synthetic RIS-EVS-OFDM channel sample,
builds the Hankelized tensor for Stage-I initialization, applies the current
reliability-gated RIS/JNPP basin-recovery policy when triggered, and runs the
current proposed final refinement: adaptive Stage-I-regularized Jones-VP in the
raw OFDM domain.

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
  --n-trials 50 \
  --paper-k 3 \
  --jobs 10 \
  --process-workers 4 \
  --blas-threads auto \
  --out-dir results/ablation_paper_k3 \
  --force-rerun
```

The paper figure runner targets the revised pipeline: Stage-I initialization,
reliability-gated RIS/JNPP basin recovery, and adaptive Stage-I-regularized
Jones-VP. Figures 1--5 use `K=3` by default through the actual simulation
configuration. Figure 6 ignores `--paper-k` and sweeps `K=1,2,3,4` at 0 dB.
PEB curves are plotted from the data-only EFIM/CRB calculation. The older
`run_stage2_ablation.py` entry point remains available only for legacy
structured-refinement module ablations.

The paper runner defaults to `--jobs 10`, `--task-grouping grouped`, and
`--blas-threads auto`. Grouped execution reuses data generation and Stage-I
initialization within each Monte Carlo trial before evaluating the requested
VP/JNPP variants; it does not change the estimator or physical channel model.

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
  --out-dir results/benchmark_comparison \
  --jobs 10 \
  --process-workers 4 \
  --blas-threads auto \
  --force-rerun
```

The benchmark figure contains:

- `ALS-CPD`: a standalone complex CP tensor baseline. It does not initialize,
  call, or feed the proposed VP.
- `FF-OMP`: a far-field angular-delay sparse baseline adapted to the current
  raw EVS-RIS-OFDM observation.
- `RIS-MOMP`: a RIS-aided multidimensional OMP-style sparse baseline with
  independent direction and delay grids.
- `NF-MMPSR`: a near-field spherical-domain grid sparse baseline.
- `Proposed`: the only curve using the current RG-JNPP-Adaptive-Jones-VP
  pipeline.
- `PEB`: the data-only EFIM/CRB reference curve.

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
  --out-dir results/robustness_and_scaling \
  --jobs 10 \
  --process-workers 4 \
  --blas-threads auto \
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
matched-model, data-only PEB by default. All PEB calculations eliminate the
linear Jones nuisance and explicitly Schur-eliminate clock before computing
the position PEB; estimator regularization is not included.
