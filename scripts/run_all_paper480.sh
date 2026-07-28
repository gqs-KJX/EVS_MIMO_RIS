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


echo "===== 预检查1：VBI clock评价回归测试 ====="
python -m pytest -q \
  tests/test_benchmark_baselines.py::test_external_clock_uses_raw_panel_delays_and_frozen_median \
  tests/test_benchmark_baselines.py::test_native_joint_clock_semantics_uses_result_delta_t


echo "===== 预检查2：geometry/model v2最终audit ====="
python -m src.experiments.audit_geometry_model_v2 \
  --optimizer-maxiter 200 \
  --output results/paper_geometry_v2/geometry_model_v2_audit_maxiter200.json


echo "===== 0/15：benchmark_external480（外部位置、时钟、信道性能）====="
python -m src.experiments.run_benchmark_comparison \
  --n-trials 480 --paper-k 3 \
  --seed 20260526 \
  --snr-grid=-20,-15,-10,-5,0,5,10,15,20,25,30 \
  --baselines als_cpd,scaled_4d,nf_ris_groupomp_localgrid_wls,ris_vbi_sbl,mksc_ccop,peb,constrained_jones_peb \
  --grid-profile medium --baseline-backend cpu \
  --include-constrained-jones-peb \
  --strict-ris-geometry \
  --jobs 24 --process-workers 24 --blas-threads 1 \
  --out-dir results/paper_geometry_v2/benchmark_external480


echo "===== 1/15：components_paper480（组件消融和tail，-10 dB）====="
python -m src.experiments.run_final_mksc_ccop_ablation \
  --suites components \
  --focus-snr-db -10 \
  --component-variants scaled_4d,old_stage1_ccop,mksc_delay_ccop,mksc_gi_1_no_refresh_ccop,mksc_gi_4_no_refresh_ccop,proposed,mksc_gi_7_refresh_ccop,oracle_position_start_ccop \
  --paired-reference scaled_4d \
  --paired-candidate proposed \
  --n-trials 480 --seed 20260721 \
  --bootstrap-replicates 10000 \
  --diagnostic-mode performance \
  --jobs 24 --blas-threads 1 \
  --out-dir results/paper_geometry_v2/components_paper480


echo "===== 2/15：snr_internal_paper480（内部SNR曲线和free-Jones PEB）====="
python -m src.experiments.run_final_mksc_ccop_ablation \
  --suites snr \
  --snr-grid=-20,-15,-10,-5,0,5,10,15,20,25,30 \
  --snr-variants scaled_4d,old_stage1_ccop,mksc_gi_4_no_refresh_ccop,proposed \
  --snr-peb \
  --n-trials 480 --seed 20260722 \
  --bootstrap-replicates 10000 \
  --diagnostic-mode performance \
  --jobs 24 --blas-threads 1 \
  --out-dir results/paper_geometry_v2/snr_internal_paper480


echo "===== 3/15：c3_clock_units_paper100（C3和clock单位控制，-10 dB）====="
python -m src.experiments.run_final_mksc_ccop_ablation \
  --suites c3_matrix \
  --snr-grid=-10 \
  --focus-snr-db -10 \
  --c3-variants old_4d,scaled_4d,old_stage1_ccop,mksc_gi_refresh_4d_seconds,mksc_gi_refresh_4d_nanoseconds,mksc_gi_refresh_4d_distance_m,proposed,mksc_gi_refresh_ccop_seconds,mksc_gi_refresh_ccop_nanoseconds,mksc_gi_refresh_ccop_distance_m \
  --paired-reference mksc_gi_refresh_4d_distance_m \
  --paired-candidate proposed \
  --n-trials 100 --seed 20260732 \
  --bootstrap-replicates 10000 \
  --diagnostic-mode performance \
  --jobs 24 --blas-threads 1 \
  --out-dir results/paper_geometry_v2/c3_clock_units_paper100


echo "===== 4/15：receiver_information_paper480（scalar/dual/full EVS）====="
python -m src.experiments.run_final_mksc_ccop_ablation \
  --suites receiver \
  --snr-grid=-20,-15,-10,-5,0,5,10,15,20,25,30 \
  --receiver-modes scalar,dual_pol,full_6d \
  --receiver-variants proposed \
  --n-trials 480 --seed 20260723 \
  --bootstrap-replicates 10000 \
  --diagnostic-mode performance \
  --jobs 24 --blas-threads 1 \
  --out-dir results/paper_geometry_v2/receiver_information_paper480


echo "===== 5/15：compression_matched_paper480（raw delay对比MKSC）====="
python -m src.experiments.run_final_mksc_ccop_ablation \
  --suites compression \
  --snr-grid=-20,-15,-10,-5,0,5,10,15,20,25,30 \
  --n-trials 480 --seed 20260724 \
  --bootstrap-replicates 10000 \
  --diagnostic-mode performance \
  --jobs 24 --blas-threads 1 \
  --out-dir results/paper_geometry_v2/compression_matched_paper480


echo "===== 6/15：maxwell_mismatch_paper480（Maxwell和基础设施失配，0 dB）====="
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
  --out-dir results/paper_geometry_v2/maxwell_mismatch_paper480


echo "===== 7/15：colored_noise_boundary480（相关噪声边界，-10 dB）====="
python -m src.experiments.run_final_mksc_ccop_robustness \
  --suites colored_noise \
  --snr-db -10 \
  --colored-noise-variants raw_delay_gi_ccop,proposed \
  --colored-noise-grid 0,0.2,0.5,0.8 \
  --n-trials 480 --seed 20260726 \
  --diagnostic-mode performance \
  --jobs 24 --blas-threads 1 \
  --out-dir results/paper_geometry_v2/colored_noise_boundary480


echo "===== 8/15：positions50x480（50位置泛化和逐trial PEB，0 dB）====="
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
  --out-dir results/paper_geometry_v2/positions50x480


echo "===== 9/15：robustness_scaling480（N/T/M_A/M_R/K缩放，0 dB）====="
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
  --out-dir results/paper_geometry_v2/robustness_scaling480


echo "===== 10/15：evs_resolvability_paper480（时延-极化可分辨性，-10 dB）====="
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
  --out-dir results/paper_geometry_v2/evs_resolvability_paper480


echo "===== 11/15：ris_bs_calibration_boundary480（RIS-BS校准失配，0 dB）====="
python -m src.experiments.run_robustness_and_scaling_figures \
  --figures fig8 \
  --snr-db 0 --true-k 3 \
  --calibration-std-grid 0,1,2,5,10,20 \
  --baselines mksc_ccop,stage1_only \
  --include-calibration-oracle-peb \
  --grid-profile medium \
  --strict-ris-geometry \
  --n-trials 480 --seed 20260802 \
  --jobs 24 --process-workers 24 --blas-threads 1 \
  --out-dir results/paper_geometry_v2/ris_bs_calibration_boundary480


echo "===== 12/15：model_order_mismatch480（模型阶数失配，0 dB）====="
python -m src.experiments.run_robustness_and_scaling_figures \
  --figures fig9 \
  --snr-db 0 --true-k 3 \
  --assumed-k-grid 2,3,4,5 \
  --baselines mksc_ccop \
  --include-trueK-peb-reference \
  --grid-profile medium \
  --strict-ris-geometry \
  --n-trials 480 --seed 20260803 \
  --jobs 24 --process-workers 24 --blas-threads 1 \
  --out-dir results/paper_geometry_v2/model_order_mismatch480


echo "===== 13/15：ue_range_generalization480（UE距离泛化，0 dB）====="
python -m src.experiments.run_robustness_and_scaling_figures \
  --figures fig10c \
  --snr-db 0 --true-k 3 \
  --baselines mksc_ccop,scaled_4d,peb,constrained_jones_peb \
  --grid-profile medium \
  --strict-ris-geometry \
  --n-trials 480 --seed 20260804 \
  --jobs 24 --process-workers 24 --blas-threads 1 \
  --out-dir results/paper_geometry_v2/ue_range_generalization480


echo "===== 14/15：components_cost30_cpu（组件runtime和内存，服务器空闲时运行）====="
python -m src.experiments.run_final_mksc_ccop_ablation \
  --suites components \
  --focus-snr-db -10 \
  --component-variants scaled_4d,old_stage1_ccop,mksc_delay_ccop,mksc_gi_1_no_refresh_ccop,mksc_gi_4_no_refresh_ccop,proposed \
  --n-trials 30 --seed 20260731 \
  --bootstrap-replicates 10000 \
  --profile-memory \
  --diagnostic-mode performance \
  --jobs 1 --blas-threads 1 \
  --out-dir results/paper_geometry_v2/components_cost30_cpu


echo "===== 15/15：benchmark_runtime30_cpu（外部算法runtime和内存，服务器空闲时运行）====="
python -m src.experiments.run_benchmark_comparison \
  --n-trials 30 --paper-k 3 \
  --seed 20260731 \
  --snr-grid=-10,0 \
  --baselines als_cpd,ris_vbi_sbl,nf_ris_groupomp_localgrid_wls,scaled_4d,mksc_ccop \
  --grid-profile medium --baseline-backend cpu \
  --runtime-profile --profile-memory \
  --strict-ris-geometry \
  --jobs 1 --process-workers 1 --blas-threads 1 \
  --out-dir results/paper_geometry_v2/benchmark_runtime30_cpu


echo "===== ALL DONE ====="