#!/usr/bin/env bash
# =============================================================================
# 480-trial full paper batch (all suites except the already-running benchmark).
#
# - Per-suite frozen root seeds are kept (20260721...). Increasing n-trials to
#   480 deterministically EXTENDS each seed sequence: the first 400/200/150/...
#   trials reproduce the earlier pilot realizations exactly.
# - SNR-sweep suites use -20..30 dB per the 2026-07-19 decision; fixed-SNR
#   robustness suites stay at the frozen -10 dB operating point.
# - All out-dirs are NEW (no --force-rerun, no overwrite of pilot results).
# - Block 0 is the benchmark high-SNR completion run. It MUST use
#   --seed 20260526 (the cli_common default that the main 480 benchmark run
#   silently used) so trial scenes pair with benchmark_full_k3_medium-480-final.
#   Do NOT re-run overlapping SNR points.
# - Block 12 (cost30 runtime/memory) is single-process and must run while the
#   server is otherwise IDLE; keep it last.
# =============================================================================
set -euo pipefail

echo "===== 0/13: benchmark high-SNR completion (15-30 dB, seed 20260526) ====="
python -m src.experiments.run_benchmark_comparison \
  --n-trials 480 --paper-k 3 \
  --seed 20260526 \
  --snr-grid=15,20,25,30 \
  --baselines als_cpd,scaled_4d,nf_ris_groupomp_localgrid_wls,ris_vbi_sbl,mksc_ccop,peb,constrained_jones_peb \
  --grid-profile medium --baseline-backend cpu \
  --include-constrained-jones-peb \
  --jobs 24 --process-workers 24 --blas-threads 1 \
  --out-dir results/benchmark_full_k3_medium-480-final-snr15to30

echo "===== 1/13: components_paper480 (tail table, -10 dB) ====="
python -m src.experiments.run_final_mksc_ccop_ablation \
  --suites components --focus-snr-db -10 \
  --n-trials 480 --seed 20260721 \
  --bootstrap-replicates 10000 --diagnostic-mode performance \
  --jobs 24 --blas-threads 1 \
  --out-dir results/final_mksc_ccop/components_paper480

echo "===== 2/13: snr_internal_paper480 (with free-Jones PEB rows) ====="
python -m src.experiments.run_final_mksc_ccop_ablation \
  --suites snr \
  --snr-grid=-20,-15,-10,-5,0,5,10,15,20,25,30 \
  --snr-variants scaled_4d,old_stage1_ccop,mksc_gi_4_no_refresh_ccop,proposed \
  --snr-peb \
  --n-trials 480 --seed 20260722 \
  --bootstrap-replicates 10000 --diagnostic-mode performance \
  --jobs 24 --blas-threads 1 \
  --out-dir results/final_mksc_ccop/snr_internal_paper480

echo "===== 3/13: receiver_information_paper480 ====="
python -m src.experiments.run_final_mksc_ccop_ablation \
  --suites receiver \
  --snr-grid=-20,-15,-10,-5,0,5,10,15,20,25,30 \
  --receiver-modes scalar,dual_pol,full_6d \
  --receiver-variants proposed \
  --n-trials 480 --seed 20260723 \
  --bootstrap-replicates 10000 --diagnostic-mode performance \
  --jobs 24 --blas-threads 1 \
  --out-dir results/final_mksc_ccop/receiver_information_paper480

echo "===== 4/13: compression_matched_paper480 ====="
python -m src.experiments.run_final_mksc_ccop_ablation \
  --suites compression \
  --snr-grid=-20,-15,-10,-5,0,5,10,15,20,25,30 \
  --n-trials 480 --seed 20260724 \
  --bootstrap-replicates 10000 --diagnostic-mode performance \
  --jobs 24 --blas-threads 1 \
  --out-dir results/final_mksc_ccop/compression_matched_paper480

echo "===== 5/13: maxwell_mismatch_paper480 (-10 dB) ====="
python -m src.experiments.run_final_mksc_ccop_robustness \
  --suites subspace_mismatch \
  --snr-db -10 \
  --mismatch-variants raw_delay_gi_ccop,proposed \
  --phase-grid 0,1,2,5,10 \
  --gain-grid 0,0.01,0.02,0.05,0.1 \
  --ris-bs-angle-grid 0,0.1,0.25,0.5,1 \
  --bs-sensor-position-mm-grid 0,0.05,0.1,0.2,0.5 \
  --n-trials 480 --seed 20260725 \
  --diagnostic-mode performance \
  --jobs 24 --blas-threads 1 \
  --out-dir results/final_mksc_ccop/maxwell_mismatch_paper480

echo "===== 6/13: colored_noise_boundary480 (-10 dB) ====="
python -m src.experiments.run_final_mksc_ccop_robustness \
  --suites colored_noise \
  --snr-db -10 \
  --colored-noise-variants raw_delay_gi_ccop,proposed \
  --colored-noise-grid 0,0.2,0.5,0.8 \
  --n-trials 480 --seed 20260726 \
  --diagnostic-mode performance \
  --jobs 24 --blas-threads 1 \
  --out-dir results/final_mksc_ccop/colored_noise_boundary480

echo "===== 7/13: positions50x480 (-10 dB, with per-trial EFIM/PEB) ====="
python -m src.experiments.run_final_mksc_ccop_robustness \
  --suites positions \
  --snr-db -10 \
  --position-grid-shape 5,5,2 \
  --position-grid-margin-m 0.1 \
  --position-variants scaled_4d,proposed \
  --position-peb \
  --n-trials 480 --seed 20260727 \
  --diagnostic-mode performance \
  --jobs 24 --blas-threads 1 \
  --out-dir results/final_mksc_ccop/positions50x480

echo "===== 8/13: robustness_scaling480 (-10 dB) ====="
python -m src.experiments.run_final_mksc_ccop_robustness \
  --suites scaling \
  --snr-db -10 \
  --scaling-variants scaled_4d,proposed \
  --n-grid 31,47,63,95 \
  --training-grid 32,64,128,256 \
  --array-grid 4,8,16,24 \
  --ris-side-grid 16,32,48,64 \
  --k-grid 2,3,4 \
  --n-trials 480 --seed 20260728 \
  --diagnostic-mode performance \
  --jobs 24 --blas-threads 1 \
  --out-dir results/final_mksc_ccop/robustness_scaling480

echo "===== 9/13: evs_resolvability_paper480 (-10 dB) ====="
python -m src.experiments.run_final_evs_resolvability \
  --snr-db -10 \
  --delay-separation-grid-ns 0.1,0.2,0.5,1,2,5 \
  --polarization-overlap-grid 0.1,0.5,0.9,1.0 \
  --receiver-modes scalar,dual_pol,full_6d \
  --delay-error-tolerance-ns 0.5 \
  --pole-collapse-tolerance-ns 0.05 \
  --n-trials 480 --seed 20260729 \
  --diagnostic-mode performance \
  --jobs 24 --blas-threads 1 \
  --out-dir results/final_mksc_ccop/evs_resolvability_paper480

echo "===== 10/13: ris_bs_calibration_boundary480 (fig8, -10 dB) ====="
python -m src.experiments.run_robustness_and_scaling_figures \
  --figures fig8 \
  --snr-db -10 --true-k 3 \
  --calibration-std-grid 0,1,2,5,10,20 \
  --baselines mksc_ccop,stage1_only \
  --n-trials 480 --seed 20260802 \
  --grid-profile medium \
  --jobs 24 --blas-threads 1 \
  --out-dir results/final_mksc_ccop/ris_bs_calibration_boundary480

echo "===== 11/13: model_order_mismatch480 (fig9, -10 dB) ====="
python -m src.experiments.run_robustness_and_scaling_figures \
  --figures fig9 \
  --snr-db -10 --true-k 3 \
  --assumed-k-grid 2,3,4,5 \
  --baselines mksc_ccop \
  --include-trueK-peb-reference \
  --n-trials 480 --seed 20260803 \
  --grid-profile medium \
  --jobs 24 --blas-threads 1 \
  --out-dir results/final_mksc_ccop/model_order_mismatch480

echo "===== 12/13: components_cost30_cpu (single-process runtime/memory; server must be idle) ====="
python -m src.experiments.run_final_mksc_ccop_ablation \
  --suites components --focus-snr-db -10 \
  --n-trials 30 --seed 20260731 \
  --bootstrap-replicates 10000 \
  --profile-memory \
  --diagnostic-mode performance \
  --jobs 1 --blas-threads 1 \
  --out-dir results/final_mksc_ccop/components_cost30_cpu

echo "===== 13/13: benchmark_runtime30_cpu (single-process external runtime table; server must be idle) ====="
python -m src.experiments.run_benchmark_comparison \
  --n-trials 30 --paper-k 3 \
  --seed 20260731 \
  --snr-grid=-10,0 \
  --baselines als_cpd,ris_vbi_sbl,nf_ris_groupomp_localgrid_wls,scaled_4d,mksc_ccop \
  --grid-profile medium --baseline-backend cpu \
  --runtime-profile --profile-memory \
  --jobs 1 --process-workers 1 --blas-threads 1 \
  --out-dir results/final_mksc_ccop/benchmark_runtime30_cpu

echo "===== ALL DONE ====="
