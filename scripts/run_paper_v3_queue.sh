#!/usr/bin/env bash
# =============================================================================
# Paper re-run queue for the 2026-07-28 optimized source tree.
#
# Every suite in results/final_mksc_ccop/ and results/benchmark_full_* predates
# three changes that move the proposed route:
#   * Nyquist beam-space acquisition (2026-07-27),
#   * stage1_factor_domain default raw_evs -> compressed_evs (2026-07-28),
#   * benchmark path dropping the legacy coarse RIS codebook (2026-07-28),
# and three that move only cost (Stage-II VP / EFIM union-subspace isometry,
# adjoint exact-projection objective, blocked squared error).  This script
# regenerates the campaign on the current tree.
#
# Sized for a 20-core / 30 GB workstation, not the 24-core compute node the
# frozen campaign used:
#   * benchmark lanes hold ~3.6 GB per worker  -> 6 workers,
#   * internal/robustness lanes hold ~1.9 GB   -> 8 workers,
#   * cost lanes are strictly serial and must run on an idle box -> last.
# BLAS threading is pinned in the environment because threadpoolctl is absent
# here, so the in-process limiter is a no-op and only the env vars bite.
#
# Seeds and SNR grids are fixed inline below so trial scenes pair across every
# v3 suite.  Out-dirs are all new; no --force-rerun anywhere.
# =============================================================================
set -uo pipefail

PY="${PAPER_PYTHON:-/home/gqs/miniconda3/bin/python}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/results/paper_v3"
LOGS="${ROOT}/tmp/paper_v3_logs"
cd "${ROOT}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0

mkdir -p "${LOGS}"

PAR_JOBS="${PAR_JOBS:-8}"
# The scaling suite sweeps N and T *upward*: at N=95 the Hankel tensor is
# ~906 MB against ~403 MB at the reference N=63, and a worker peaks near
# 4.4 GB.  Eight of those OOM-kill this 30 GB box, so this suite -- and only
# this suite -- gets its own worker count.
SCALING_JOBS="${SCALING_JOBS:-4}"

run_block() {
  local name="$1"; shift
  if [[ -e "${OUT}/${name}" ]]; then
    echo "[$(date -Is)] SKIP ${name} (output exists)"
    return 0
  fi
  echo "[$(date -Is)] START ${name}"
  if "$@" > "${LOGS}/${name}.log" 2>&1; then
    echo "[$(date -Is)] OK   ${name}"
  else
    echo "[$(date -Is)] FAIL ${name} (see ${LOGS}/${name}.log)"
  fi
}

# --- 1. as-published tier -----------------------------------------------------
# Only ris_vbi_sbl is tier-sensitive; als_cpd and nf_ris_groupomp_localgrid_wls
# assert refinement_tier_sensitive: false, so running the tier arm on the VBI
# route alone is sufficient and 5x cheaper.
run_block benchmark_as_published_480 \
  "${PY}" -m src.experiments.run_benchmark_comparison \
  --n-trials 480 --paper-k 3 --seed 20260526 \
  --snr-grid=-15,-10,0,10 \
  --baselines ris_vbi_sbl \
  --grid-profile medium --baseline-backend cpu \
  --no-constrained-jones-peb --strict-ris-geometry \
  --baseline-refinement-tier as_published \
  --respect-existing-blas-env \
  --jobs 6 --process-workers 6 --blas-threads 1 \
  --memory-budget-gb 24 --memory-per-worker-gb 3.7 \
  --no-plots --progress-heartbeat-s 300 \
  --out-dir "${OUT}/benchmark_as_published_480"

# --- 2. internal accuracy suites ---------------------------------------------
run_block snr_internal_480 \
  "${PY}" -m src.experiments.run_final_mksc_ccop_ablation \
  --suites snr \
  --snr-grid=-20,-15,-10,-5,0,5,10,15,20,25,30 \
  --snr-variants scaled_4d,old_stage1_ccop,mksc_gi_4_no_refresh_ccop,proposed \
  --snr-peb \
  --n-trials 480 --seed 20260722 \
  --bootstrap-replicates 10000 \
  --diagnostic-mode performance \
  --jobs "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/snr_internal_480"

run_block components_480 \
  "${PY}" -m src.experiments.run_final_mksc_ccop_ablation \
  --suites components \
  --focus-snr-db -10 \
  --component-variants scaled_4d,old_stage1_ccop,mksc_delay_ccop,mksc_gi_1_no_refresh_ccop,mksc_gi_4_no_refresh_ccop,proposed,mksc_gi_7_refresh_ccop,oracle_position_start_ccop \
  --paired-reference scaled_4d \
  --paired-candidate proposed \
  --n-trials 480 --seed 20260721 \
  --bootstrap-replicates 10000 \
  --diagnostic-mode performance \
  --jobs "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/components_480"

# The 30-trial cost suite already shows every MKSC rung at 0/30 catastrophic at
# -10 dB after the Nyquist beam-space acquisition fix, i.e. the focus SNR the
# frozen campaign used no longer discriminates between the rungs.  Repeat the
# ladder at -20 dB, inside the threshold region, where a tail still exists and
# the components can still be told apart.  Reported in the supplementary.
run_block components_480_m20 \
  "${PY}" -m src.experiments.run_final_mksc_ccop_ablation \
  --suites components \
  --focus-snr-db -20 \
  --component-variants scaled_4d,old_stage1_ccop,mksc_delay_ccop,mksc_gi_1_no_refresh_ccop,mksc_gi_4_no_refresh_ccop,proposed,mksc_gi_7_refresh_ccop,oracle_position_start_ccop \
  --paired-reference scaled_4d \
  --paired-candidate proposed \
  --n-trials 480 --seed 20260721 \
  --bootstrap-replicates 10000 \
  --diagnostic-mode performance \
  --jobs "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/components_480_m20"

run_block compression_matched_480 \
  "${PY}" -m src.experiments.run_final_mksc_ccop_ablation \
  --suites compression \
  --snr-grid=-20,-15,-10,-5,0,5,10,15,20,25,30 \
  --n-trials 480 --seed 20260724 \
  --bootstrap-replicates 10000 \
  --diagnostic-mode performance \
  --jobs "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/compression_matched_480"

run_block receiver_information_480 \
  "${PY}" -m src.experiments.run_final_mksc_ccop_ablation \
  --suites receiver \
  --snr-grid=-20,-15,-10,-5,0,5,10,15,20,25,30 \
  --receiver-modes scalar,dual_pol,full_6d \
  --receiver-variants proposed \
  --n-trials 480 --seed 20260723 \
  --bootstrap-replicates 10000 \
  --diagnostic-mode performance \
  --jobs "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/receiver_information_480"

run_block c3_clock_units_100 \
  "${PY}" -m src.experiments.run_final_mksc_ccop_ablation \
  --suites c3_matrix \
  --snr-grid=-10 --focus-snr-db -10 \
  --c3-variants old_4d,scaled_4d,old_stage1_ccop,mksc_gi_refresh_4d_seconds,mksc_gi_refresh_4d_nanoseconds,mksc_gi_refresh_4d_distance_m,proposed,mksc_gi_refresh_ccop_seconds,mksc_gi_refresh_ccop_nanoseconds,mksc_gi_refresh_ccop_distance_m \
  --paired-reference mksc_gi_refresh_4d_distance_m \
  --paired-candidate proposed \
  --n-trials 100 --seed 20260732 \
  --bootstrap-replicates 10000 \
  --diagnostic-mode performance \
  --jobs "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/c3_clock_units_100"

# --- 3. robustness / generalization ------------------------------------------
run_block maxwell_mismatch_480 \
  "${PY}" -m src.experiments.run_final_mksc_ccop_robustness \
  --suites subspace_mismatch \
  --snr-db -10 \
  --mismatch-variants raw_delay_gi_ccop,proposed \
  --phase-grid 0,1,2,5,10 \
  --gain-grid 0,0.01,0.02,0.05,0.1 \
  --ris-bs-angle-grid 0,0.1,0.25,0.5,1 \
  --bs-sensor-position-mm-grid 0,0.05,0.1,0.2,0.5 \
  --n-trials 480 --seed 20260725 \
  --diagnostic-mode performance \
  --jobs "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/maxwell_mismatch_480"

run_block colored_noise_boundary_480 \
  "${PY}" -m src.experiments.run_final_mksc_ccop_robustness \
  --suites colored_noise \
  --snr-db -10 \
  --colored-noise-variants raw_delay_gi_ccop,proposed \
  --colored-noise-grid 0,0.2,0.5,0.8 \
  --n-trials 480 --seed 20260726 \
  --diagnostic-mode performance \
  --jobs "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/colored_noise_boundary_480"

run_block model_order_mismatch_480 \
  "${PY}" -m src.experiments.run_robustness_and_scaling_figures \
  --figures fig9 \
  --snr-db 0 --true-k 3 \
  --assumed-k-grid 2,3,4,5 \
  --baselines mksc_ccop \
  --include-trueK-peb-reference \
  --grid-profile medium \
  --strict-ris-geometry \
  --n-trials 480 --seed 20260803 \
  --jobs "${PAR_JOBS}" --process-workers "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/model_order_mismatch_480"

run_block ris_bs_calibration_boundary_480 \
  "${PY}" -m src.experiments.run_robustness_and_scaling_figures \
  --figures fig8 \
  --snr-db 0 --true-k 3 \
  --calibration-std-grid 0,1,2,5,10,20 \
  --baselines mksc_ccop,stage1_only \
  --include-calibration-oracle-peb \
  --grid-profile medium \
  --strict-ris-geometry \
  --n-trials 480 --seed 20260802 \
  --jobs "${PAR_JOBS}" --process-workers "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/ris_bs_calibration_boundary_480"

run_block robustness_scaling_480 \
  "${PY}" -m src.experiments.run_final_mksc_ccop_robustness \
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
  --jobs "${SCALING_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/robustness_scaling_480"

run_block evs_resolvability_480 \
  "${PY}" -m src.experiments.run_final_evs_resolvability \
  --snr-db -10 \
  --delay-separation-grid-ns 0.1,0.2,0.5,1,2,5 \
  --polarization-overlap-grid 0.1,0.5,0.9,1.0 \
  --receiver-modes scalar,dual_pol,full_6d \
  --delay-error-tolerance-ns 0.5 \
  --pole-collapse-tolerance-ns 0.05 \
  --n-trials 480 --seed 20260729 \
  --diagnostic-mode performance \
  --jobs "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/evs_resolvability_480"

run_block positions50x480 \
  "${PY}" -m src.experiments.run_final_mksc_ccop_robustness \
  --suites positions \
  --snr-db -10 \
  --position-grid-shape 5,5,2 \
  --position-grid-margin-m 0.1 \
  --position-variants scaled_4d,proposed \
  --position-peb \
  --n-trials 480 --seed 20260727 \
  --diagnostic-mode performance \
  --jobs "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/positions50x480"

# --- 4. cost suites: strictly serial, machine must be otherwise idle ----------
#
# NOTE 2026-07-28: both cost blocks below were already taken out of order, with
# the benchmark workers SIGSTOPed so the box was genuinely idle, because
# Section~\ref{subsec:complexity} and Section~\ref{subsec:res_cost} were blocked
# on them.  ``run_block`` skips a block whose out-dir exists, so they are no-ops
# unless those directories are removed.  ``components_cost30_cpu_union`` holds
# the same measurement with the legacy acquisition dictionary enabled, which is
# what the +3.3 s difference in that dictionary's cost is read from.
run_block components_cost30_cpu \
  "${PY}" -m src.experiments.run_final_mksc_ccop_ablation \
  --suites components \
  --focus-snr-db -10 \
  --component-variants scaled_4d,old_stage1_ccop,mksc_delay_ccop,mksc_gi_1_no_refresh_ccop,mksc_gi_4_no_refresh_ccop,proposed \
  --n-trials 30 --seed 20260731 \
  --bootstrap-replicates 10000 \
  --profile-memory \
  --diagnostic-mode performance \
  --coarse-codebook-mode beamspace_only \
  --jobs 1 --blas-threads 1 \
  --out-dir "${OUT}/components_cost30_cpu"

run_block benchmark_runtime30_cpu \
  "${PY}" -m src.experiments.run_benchmark_comparison \
  --n-trials 30 --paper-k 3 --seed 20260731 \
  --snr-grid=-10,0 \
  --baselines als_cpd,ris_vbi_sbl,nf_ris_groupomp_localgrid_wls,scaled_4d,mksc_ccop \
  --grid-profile medium --baseline-backend cpu \
  --runtime-profile --profile-memory \
  --strict-ris-geometry --no-constrained-jones-peb \
  --respect-existing-blas-env \
  --jobs 1 --process-workers 1 --blas-threads 1 \
  --no-plots \
  --out-dir "${OUT}/benchmark_runtime30_cpu"

echo "[$(date -Is)] QUEUE COMPLETE"
