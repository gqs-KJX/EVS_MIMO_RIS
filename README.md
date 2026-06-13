# RIS-EVS-OFDM simulation

## Single proposed demo

Run one small fixed-SNR proposed-method demo from the project root:

```bash
python -m src.main_single_proposed
```

The script generates one synthetic RIS-EVS-OFDM channel sample at 0 dB SNR,
builds the Hankelized tensor for Stage-I initialization and reliability-gated
Stage-II JNPP basin recovery, then performs Stage-I-regularized Jones-VP in the
raw OFDM domain.
