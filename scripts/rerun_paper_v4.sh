#!/usr/bin/env bash
# Re-run of the frozen paper campaign after the RIS-BS two-hop phase-convention
# fix (a_RB and v_B were the conjugates of the e^{-j k l} extra-path convention
# that eq:delta uses).  Only the model code changed: every command line below is
# byte-identical to the paper_v3 run except --out-dir, which points at paper_v4
# so the frozen v3 results survive for comparison.
#
# Resource notes
#   * --jobs / --process-workers below are the 64-core values.  Memory is the
#     binding constraint, not cores: budget about 3 GB per worker.
#   * The two COST suites must run on ONE host at --jobs 1, or the paper's
#     "all timings share one host" statement is false.
#
# Usage:  bash scripts/rerun_paper_v4.sh            # everything
#         bash scripts/rerun_paper_v4.sh tier1      # main-text figures/tables
set -euo pipefail

PY="${PY:-python}"
J="${J:-64}"
OUT=results/paper_v4
mkdir -p "$OUT"
TIER="${1:-all}"

run () { echo; echo "### $* "; }

# =====================================================================
# TIER 1 - feeds the main-text figures and tables   (~20.6 h @ 64 cores)
# =====================================================================
tier1 () {

# Fig. 5(c) + the delay-coincidence paragraph                     8.40 h
$PY -m src.experiments.run_final_mksc_ccop_robustness \
    --suites positions \
    --snr-db -10 \
    --position-grid-shape 5,5,2 \
    --position-grid-margin-m 0.1 \
    --position-variants scaled_4d,proposed \
    --position-peb \
    --n-trials 480 \
    --seed 20260727 \
    --diagnostic-mode performance \
    --jobs "$J" --blas-threads 1 \
    --out-dir $OUT/positions50x480

# Fig. 2 (external benchmark, native read-outs)                   4.32 h
$PY -m src.experiments.run_benchmark_comparison \
    --n-trials 960 \
    --paper-k 3 \
    --seed 20260815 \
    --snr-grid=-20,-15,-10,-5,0,5,10,15,20,25,30 \
    --baselines als_cpd,scaled_4d,nf_ris_groupomp_localgrid_wls,ris_vbi_sbl,mksc_ccop,peb,constrained_jones_peb \
    --grid-profile medium \
    --baseline-backend cpu \
    --baseline-refinement-tier as_published \
    --outlier-threshold-m 0.1 \
    --clock-catastrophic-threshold-ns 1.0 \
    --strict-ris-geometry \
    --jobs "$J" --process-workers "$J" --blas-threads 1 \
    --no-plots --progress-heartbeat-s 300 \
    --out-dir $OUT/benchmark_as_published_960_5

# Fig. 4 (polarimetric panel separability)                        3.03 h
$PY -m src.experiments.run_final_evs_resolvability \
    --snr-db -10 \
    --delay-separation-grid-ns 0.1,0.2,0.5,1,2,5 \
    --polarization-overlap-grid 0.1,0.5,0.9,1.0 \
    --receiver-modes scalar,dual_pol,full_6d \
    --delay-error-tolerance-ns 0.5 \
    --pole-collapse-tolerance-ns 0.05 \
    --n-trials 480 \
    --seed 20260729 \
    --diagnostic-mode performance \
    --jobs "$J" --blas-threads 1 \
    --out-dir $OUT/evs_resolvability_480

# Fig. 3 (C1 information, three receiver modes)                   2.77 h
$PY -m src.experiments.run_final_mksc_ccop_ablation \
    --suites receiver \
    --snr-grid=-20,-15,-10,-5,0,5,10,15,20,25,30 \
    --receiver-modes scalar,dual_pol,full_6d \
    --receiver-variants proposed \
    --coarse-codebook-mode beamspace_only \
    --n-trials 480 \
    --seed 20260723 \
    --bootstrap-replicates 10000 \
    --diagnostic-mode performance \
    --jobs "$J" --blas-threads 1 \
    --out-dir $OUT/receiver_information_480

# Fig. 5(a)(b) + 0.447/31.2 mm, 99.17%, 59.58%                    1.68 h
$PY -m src.experiments.run_final_mksc_ccop_robustness \
    --suites subspace_mismatch \
    --snr-db -10 \
    --mismatch-variants raw_delay_gi_ccop,proposed \
    --phase-grid 0,1,2,5,10 \
    --gain-grid 0,0.01,0.02,0.05,0.1 \
    --ris-bs-angle-grid 0,0.1,0.25,0.5,1 \
    --bs-sensor-position-mm-grid 0,0.05,0.1,0.2,0.5 \
    --n-trials 480 \
    --seed 20260725 \
    --diagnostic-mode performance \
    --jobs "$J" --blas-threads 1 \
    --out-dir $OUT/maxwell_mismatch_480

# Table I (Phase-I mechanism ladder at -20 dB)                    0.34 h
$PY -m src.experiments.run_final_mksc_ccop_ablation \
    --suites components \
    --focus-snr-db -20 \
    --component-variants scaled_4d,old_stage1_ccop,mksc_delay_ccop,mksc_gi_1_no_refresh_ccop,mksc_gi_4_no_refresh_ccop,proposed,mksc_gi_7_refresh_ccop,oracle_position_start_ccop \
    --paired-reference scaled_4d \
    --paired-candidate proposed \
    --coarse-codebook-mode beamspace_only \
    --n-trials 480 \
    --seed 20260721 \
    --bootstrap-replicates 10000 \
    --diagnostic-mode performance \
    --jobs "$J" --blas-threads 1 \
    --out-dir $OUT/components_480_m20

# Table II (Phase-II refinement control)                          0.09 h
$PY -m src.experiments.run_final_mksc_ccop_ablation \
    --suites c3_matrix \
    --snr-grid=-10 \
    --focus-snr-db -10 \
    --c3-variants old_4d,scaled_4d,old_stage1_ccop,mksc_gi_refresh_4d_seconds,mksc_gi_refresh_4d_nanoseconds,mksc_gi_refresh_4d_distance_m,proposed,mksc_gi_refresh_ccop_seconds,mksc_gi_refresh_ccop_nanoseconds,mksc_gi_refresh_ccop_distance_m \
    --paired-reference mksc_gi_refresh_4d_distance_m \
    --paired-candidate proposed \
    --coarse-codebook-mode beamspace_only \
    --n-trials 100 \
    --seed 20260732 \
    --bootstrap-replicates 10000 \
    --diagnostic-mode performance \
    --jobs "$J" --blas-threads 1 \
    --out-dir $OUT/c3_clock_units_100
}

# =====================================================================
# TIER 2 - feeds main-text prose numbers            (~13 h @ 64 cores)
# =====================================================================
tier2 () {

# VBI/SBL 26.2 -> 0.315 mm, CPD 41.9 % after shared refinement    7.38 h
$PY -m src.experiments.run_benchmark_comparison \
    --n-trials 960 \
    --paper-k 3 \
    --seed 20260526 \
    --snr-grid=-20,-15,-10,-5,0,5,10,15,20,25,30 \
    --baselines als_cpd,scaled_4d,nf_ris_groupomp_localgrid_wls,ris_vbi_sbl,mksc_ccop,peb,constrained_jones_peb \
    --grid-profile medium \
    --baseline-backend cpu \
    --baseline-refinement-tier refinement_matched \
    --outlier-threshold-m 0.1 \
    --clock-catastrophic-threshold-ns 1.0 \
    --strict-ris-geometry \
    --jobs "$J" --process-workers "$J" --blas-threads 1 \
    --no-plots --progress-heartbeat-s 300 \
    --out-dir $OUT/benchmark_refinement_matched_960_4

# 4320 trials, 35833 certificates, 99.93 % assoc, PEB/CEB          2.31 h
$PY -m src.experiments.run_final_mksc_ccop_ablation \
    --suites snr \
    --snr-grid=-20,-15,-10,-5,0,5,10,15,20,25,30 \
    --snr-variants scaled_4d,old_stage1_ccop,mksc_gi_4_no_refresh_ccop,proposed \
    --snr-peb \
    --coarse-codebook-mode beamspace_only \
    --n-trials 480 \
    --seed 20260722 \
    --bootstrap-replicates 10000 \
    --diagnostic-mode performance \
    --jobs "$J" --blas-threads 1 \
    --out-dir $OUT/snr_internal_480

# aperture 16^2 -> 64^2, M_A / T / bandwidth, K = 4                1.60 h
$PY -m src.experiments.run_final_mksc_ccop_robustness \
    --suites scaling \
    --snr-db -10 \
    --scaling-variants scaled_4d,proposed \
    --n-grid 31,47,63,95 \
    --training-grid 32,64,128,256 \
    --array-grid 4,8,16,24 \
    --ris-side-grid 16,32,48,64 \
    --k-grid 2,3,4 \
    --n-trials 480 \
    --seed 20260728 \
    --diagnostic-mode performance \
    --jobs "$J" --blas-threads 1 \
    --out-dir $OUT/robustness_scaling_480

# leakage 0.822 -> 0.353, basin 97.50 -> 99.38 %, 2.29 -> 0.62 %   0.92 h
$PY -m src.experiments.run_final_mksc_ccop_ablation \
    --suites compression \
    --snr-grid=-20,-15,-10,-5,0,5,10,15,20,25,30 \
    --coarse-codebook-mode beamspace_only \
    --n-trials 480 \
    --seed 20260724 \
    --bootstrap-replicates 10000 \
    --diagnostic-mode performance \
    --jobs "$J" --blas-threads 1 \
    --out-dir $OUT/compression_matched_480

# lambda sweep quoted by Section IV and Section V                  ~0.9 h
$PY -m src.experiments.run_phase2_jones_lambda_ablation \
    --lambda-grid 0,0.1,1,10 \
    --snr-grid=-10,0,10,20,30 \
    --receiver-modes full_6d,dual_pol,scalar \
    --n-trials 480 \
    --seed 20260813 \
    --bootstrap-replicates 2000 \
    --jobs "$J" --blas-threads 1 \
    --out-dir $OUT/sensitivity/phase2_jones_lambda_final

# model order Khat = 2,4,5                                         0.47 h
$PY -m src.experiments.run_robustness_and_scaling_figures \
    --figures fig9 \
    --snr-db 0 \
    --true-k 3 \
    --assumed-k-grid 2,3,4,5 \
    --baselines mksc_ccop \
    --include-trueK-peb-reference \
    --grid-profile medium \
    --strict-ris-geometry \
    --n-trials 480 \
    --seed 20260803 \
    --jobs "$J" --process-workers "$J" --blas-threads 1 \
    --out-dir $OUT/model_order_mismatch_480

# "colored noise all milder over their tested ranges"              0.34 h
$PY -m src.experiments.run_final_mksc_ccop_robustness \
    --suites colored_noise \
    --snr-db -10 \
    --colored-noise-variants raw_delay_gi_ccop,proposed \
    --colored-noise-grid 0,0.2,0.5,0.8 \
    --n-trials 480 \
    --seed 20260726 \
    --diagnostic-mode performance \
    --jobs "$J" --blas-threads 1 \
    --out-dir $OUT/colored_noise_boundary_480
}

# =====================================================================
# TIER 3 - COST.  ONE host, --jobs 1, run these two back to back.
#          The paper states all timings share one host.      (~0.5 h)
# =====================================================================
tier3 () {

# 3.20 s / 2.52 GB proposed, 2.42 s 4D-JVP, 4.51 s CPD
$PY -m src.experiments.run_benchmark_comparison \
    --n-trials 30 \
    --paper-k 3 \
    --seed 20260731 \
    --snr-grid=-10,0 \
    --baselines als_cpd,ris_vbi_sbl,nf_ris_groupomp_localgrid_wls,scaled_4d,mksc_ccop \
    --grid-profile medium \
    --baseline-backend cpu \
    --baseline-refinement-tier as_published \
    --outlier-threshold-m 0.1 \
    --clock-catastrophic-threshold-ns 1.0 \
    --strict-ris-geometry \
    --runtime-profile --profile-memory \
    --jobs 1 --process-workers 1 --blas-threads 1 \
    --no-plots --progress-heartbeat-s 300 \
    --out-dir $OUT/benchmark_runtime30_cpu_as_published_v2

# Phase I 2.86 s vs certified clock profiling 0.33 s
$PY -m src.experiments.run_final_mksc_ccop_ablation \
    --suites components \
    --focus-snr-db -10 \
    --component-variants scaled_4d,old_stage1_ccop,mksc_delay_ccop,mksc_gi_1_no_refresh_ccop,mksc_gi_4_no_refresh_ccop,proposed \
    --coarse-codebook-mode beamspace_only \
    --n-trials 30 \
    --seed 20260731 \
    --bootstrap-replicates 10000 \
    --profile-memory \
    --diagnostic-mode performance \
    --jobs 1 --blas-threads 1 \
    --out-dir $OUT/components_cost30_cpu
}

# =====================================================================
# TIER 4 - supplement only.  Skipping these saves 1.3 h of 36 h.
# =====================================================================
tier4 () {

$PY -m src.experiments.run_robustness_and_scaling_figures \
    --figures fig8 \
    --snr-db 0 \
    --true-k 3 \
    --calibration-std-grid 0,1,2,5,10,20 \
    --baselines mksc_ccop,stage1_only \
    --include-calibration-oracle-peb \
    --grid-profile medium \
    --strict-ris-geometry \
    --n-trials 480 \
    --seed 20260802 \
    --jobs "$J" --process-workers "$J" --blas-threads 1 \
    --out-dir $OUT/ris_bs_calibration_boundary_480

$PY -m src.experiments.run_final_mksc_ccop_ablation \
    --suites components \
    --focus-snr-db -10 \
    --component-variants scaled_4d,old_stage1_ccop,mksc_delay_ccop,mksc_gi_1_no_refresh_ccop,mksc_gi_4_no_refresh_ccop,proposed,mksc_gi_7_refresh_ccop,oracle_position_start_ccop \
    --paired-reference scaled_4d \
    --paired-candidate proposed \
    --coarse-codebook-mode beamspace_only \
    --n-trials 480 \
    --seed 20260721 \
    --bootstrap-replicates 10000 \
    --diagnostic-mode performance \
    --jobs "$J" --blas-threads 1 \
    --out-dir $OUT/components_480
}

case "$TIER" in
  tier1) tier1 ;;
  tier2) tier2 ;;
  tier3) tier3 ;;
  tier4) tier4 ;;
  all)   tier1; tier2; tier3; tier4 ;;
  *) echo "usage: $0 [all|tier1|tier2|tier3|tier4]"; exit 1 ;;
esac
echo "done: $TIER -> $OUT"
