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
python -m src.experiments.run_paper_ablation_figures --figures all --n-trials 50 --out-dir results/ablation_paper
python -m src.experiments.run_paper_ablation_figures --figures fig1,fig2 --n-trials 50
```

The paper figure runner targets the revised pipeline: Stage-I initialization,
reliability-gated RIS/JNPP basin recovery, and adaptive Stage-I-regularized
Jones-VP. The older `run_stage2_ablation.py` entry point remains available only
for legacy structured-refinement module ablations.
