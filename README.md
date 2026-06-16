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
python -m src.experiments.run_paper_ablation_figures --figures all --n-trials 50 --paper-k 3 --snr-grid "-30,-25,-20,-15,-10,-5,0,5,10" --out-dir results/ablation_paper --force-rerun --jobs 10 --task-grouping grouped --blas-threads 1
python -m src.experiments.run_paper_ablation_figures --figures fig1,fig2 --n-trials 50 --paper-k 3 --jobs 30 --task-grouping grouped --blas-threads 1
```

The paper figure runner targets the revised pipeline: Stage-I initialization,
reliability-gated RIS/JNPP basin recovery, and adaptive Stage-I-regularized
Jones-VP. Figures 1--5 use `K=3` by default through the actual simulation
configuration. Figure 6 ignores `--paper-k` and sweeps `K=1,2,3,4` at 0 dB.
PEB curves are plotted from the data-only EFIM/CRB calculation. The older
`run_stage2_ablation.py` entry point remains available only for legacy
structured-refinement module ablations.

The paper runner defaults to `--jobs 10`, `--task-grouping grouped`, and
`--blas-threads 1`. Grouped execution reuses data generation and Stage-I
initialization within each Monte Carlo trial before evaluating the requested
VP/JNPP variants; it does not change the estimator or physical channel model.
Use `--jobs 30` on machines with enough memory and cores.
