# RIS-EVS-OFDM simulation

## Single proposed demo

Run one small fixed-SNR proposed-method demo from the project root:

```bash
python -m src.main_single_proposed
```

The script generates one synthetic RIS-EVS-OFDM channel sample at 0 dB SNR,
builds the Hankelized tensor for initialization and structured refinement, then
performs the final variable-projection refinement in the raw OFDM domain.
