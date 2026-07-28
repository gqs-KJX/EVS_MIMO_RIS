#!/usr/bin/env bash
# =============================================================================
# VERIFICATION campaign (10 trials per cell), not the paper campaign.
#
# Purpose: confirm end to end that the optimized tree produces sane, correctly
# shaped artifacts for every suite the manuscript depends on -- correct columns,
# no crashes, no silent regressions, and figures that build -- at a cost of
# ~1 hour rather than the ~2.5 days a 480-trial campaign needs on this box.
#
# 10 trials per cell is NOT enough to quote in the paper.  Rates carry
# Clopper-Pearson intervals roughly +-30 percentage points wide at n=10, and the
# frozen protocol's paired bootstrap and McNemar tests are meaningless at that
# sample size.  Every number these runs produce is INDICATIVE ONLY; treat them
# as a smoke test of the pipeline and of the direction of the change, never as
# evidence for a claim.
#
# The two cost suites are excluded on purpose: they already ran at 30 trials on
# an idle machine (results/paper_v3/{components_cost30_cpu,benchmark_runtime30_cpu})
# and are the authoritative cost artifacts.  Re-running them at 10 would be
# strictly worse.
# =============================================================================
set -uo pipefail

PY="${PAPER_PYTHON:-/home/gqs/miniconda3/bin/python}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/results/paper_verify10"
LOGS="${ROOT}/tmp/paper_verify10_logs"
cd "${ROOT}"

# threadpoolctl is not installed here, so the in-process BLAS limiter is a
# silent no-op and only these env vars actually pin threading.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0

mkdir -p "${LOGS}"

N="${VERIFY_TRIALS:-10}"
PAR_JOBS="${PAR_JOBS:-8}"
# The scaling suite sweeps N and T *upward*: at N=95 the Hankel tensor is
# ~906 MB against ~403 MB at the reference N=63, and a worker peaks near
# 4.4 GB.  Eight of those OOM-kill this 30 GB box, so this suite -- and only
# this suite -- gets its own worker count.
SCALING_JOBS="${SCALING_JOBS:-4}"
BENCH_WORKERS="${BENCH_WORKERS:-6}"

run_block() {
  local name="$1"; shift
  if [[ -e "${OUT}/${name}" ]]; then
    echo "[$(date -Is)] SKIP ${name} (output exists)"
    return 0
  fi
  echo "[$(date -Is)] START ${name}"
  local t0 t1
  t0=$(date +%s)
  if "$@" > "${LOGS}/${name}.log" 2>&1; then
    t1=$(date +%s)
    echo "[$(date -Is)] OK   ${name}  ($((t1-t0)) s)"
  else
    t1=$(date +%s)
    echo "[$(date -Is)] FAIL ${name}  ($((t1-t0)) s, see ${LOGS}/${name}.log)"
  fi
}

# --- external benchmark, both tiers ------------------------------------------
run_block benchmark_matched \
  "${PY}" -m src.experiments.run_benchmark_comparison \
  --n-trials "${N}" --paper-k 3 --seed 20260526 \
  --snr-grid=-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30 \
  --baselines als_cpd,scaled_4d,nf_ris_groupomp_localgrid_wls,ris_vbi_sbl,mksc_ccop,peb,constrained_jones_peb \
  --grid-profile medium --baseline-backend cpu \
  --include-constrained-jones-peb --strict-ris-geometry \
  --baseline-refinement-tier refinement_matched \
  --respect-existing-blas-env \
  --jobs "${BENCH_WORKERS}" --process-workers "${BENCH_WORKERS}" --blas-threads 1 \
  --memory-budget-gb 24 --memory-per-worker-gb 3.7 \
  --no-plots --progress-heartbeat-s 120 \
  --out-dir "${OUT}/benchmark_matched"

run_block benchmark_as_published \
  "${PY}" -m src.experiments.run_benchmark_comparison \
  --n-trials "${N}" --paper-k 3 --seed 20260526 \
  --snr-grid=-15,-10,0,10 \
  --baselines ris_vbi_sbl \
  --grid-profile medium --baseline-backend cpu \
  --no-constrained-jones-peb --strict-ris-geometry \
  --baseline-refinement-tier as_published \
  --respect-existing-blas-env \
  --jobs "${BENCH_WORKERS}" --process-workers "${BENCH_WORKERS}" --blas-threads 1 \
  --memory-budget-gb 24 --memory-per-worker-gb 3.7 \
  --no-plots --progress-heartbeat-s 120 \
  --out-dir "${OUT}/benchmark_as_published"

# --- internal accuracy suites -------------------------------------------------
run_block snr_internal \
  "${PY}" -m src.experiments.run_final_mksc_ccop_ablation \
  --suites snr \
  --snr-grid=-20,-15,-10,-5,0,5,10,15,20,25,30 \
  --snr-variants scaled_4d,old_stage1_ccop,mksc_gi_4_no_refresh_ccop,proposed \
  --snr-peb \
  --n-trials "${N}" --seed 20260722 \
  --bootstrap-replicates 2000 \
  --diagnostic-mode performance \
  --jobs "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/snr_internal"

run_block components_m10 \
  "${PY}" -m src.experiments.run_final_mksc_ccop_ablation \
  --suites components \
  --focus-snr-db -10 \
  --component-variants scaled_4d,old_stage1_ccop,mksc_delay_ccop,mksc_gi_1_no_refresh_ccop,mksc_gi_4_no_refresh_ccop,proposed,mksc_gi_7_refresh_ccop,oracle_position_start_ccop \
  --paired-reference scaled_4d --paired-candidate proposed \
  --n-trials "${N}" --seed 20260721 \
  --bootstrap-replicates 2000 \
  --diagnostic-mode performance \
  --jobs "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/components_m10"

# Same ladder inside the threshold region.  The 30-trial cost suite put every
# MKSC rung at 0/30 catastrophic at -10 dB, so this arm is the one that can
# still tell the rungs apart.
run_block components_m20 \
  "${PY}" -m src.experiments.run_final_mksc_ccop_ablation \
  --suites components \
  --focus-snr-db -20 \
  --component-variants scaled_4d,old_stage1_ccop,mksc_delay_ccop,mksc_gi_1_no_refresh_ccop,mksc_gi_4_no_refresh_ccop,proposed,mksc_gi_7_refresh_ccop,oracle_position_start_ccop \
  --paired-reference scaled_4d --paired-candidate proposed \
  --n-trials "${N}" --seed 20260721 \
  --bootstrap-replicates 2000 \
  --diagnostic-mode performance \
  --jobs "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/components_m20"

run_block compression_matched \
  "${PY}" -m src.experiments.run_final_mksc_ccop_ablation \
  --suites compression \
  --snr-grid=-20,-15,-10,-5,0,5,10,15,20,25,30 \
  --n-trials "${N}" --seed 20260724 \
  --bootstrap-replicates 2000 \
  --diagnostic-mode performance \
  --jobs "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/compression_matched"

run_block receiver_information \
  "${PY}" -m src.experiments.run_final_mksc_ccop_ablation \
  --suites receiver \
  --snr-grid=-20,-15,-10,-5,0,5,10,15,20,25,30 \
  --receiver-modes scalar,dual_pol,full_6d \
  --receiver-variants proposed \
  --n-trials "${N}" --seed 20260723 \
  --bootstrap-replicates 2000 \
  --diagnostic-mode performance \
  --jobs "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/receiver_information"

run_block c3_clock_units \
  "${PY}" -m src.experiments.run_final_mksc_ccop_ablation \
  --suites c3_matrix \
  --snr-grid=-10 --focus-snr-db -10 \
  --c3-variants old_4d,scaled_4d,old_stage1_ccop,mksc_gi_refresh_4d_seconds,mksc_gi_refresh_4d_nanoseconds,mksc_gi_refresh_4d_distance_m,proposed,mksc_gi_refresh_ccop_seconds,mksc_gi_refresh_ccop_nanoseconds,mksc_gi_refresh_ccop_distance_m \
  --paired-reference mksc_gi_refresh_4d_distance_m --paired-candidate proposed \
  --n-trials "${N}" --seed 20260732 \
  --bootstrap-replicates 2000 \
  --diagnostic-mode performance \
  --jobs "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/c3_clock_units"

# --- robustness / generalization ----------------------------------------------
run_block maxwell_mismatch \
  "${PY}" -m src.experiments.run_final_mksc_ccop_robustness \
  --suites subspace_mismatch --snr-db -10 \
  --mismatch-variants raw_delay_gi_ccop,proposed \
  --phase-grid 0,1,2,5,10 --gain-grid 0,0.01,0.02,0.05,0.1 \
  --ris-bs-angle-grid 0,0.1,0.25,0.5,1 \
  --bs-sensor-position-mm-grid 0,0.05,0.1,0.2,0.5 \
  --n-trials "${N}" --seed 20260725 \
  --diagnostic-mode performance \
  --jobs "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/maxwell_mismatch"

run_block colored_noise_boundary \
  "${PY}" -m src.experiments.run_final_mksc_ccop_robustness \
  --suites colored_noise --snr-db -10 \
  --colored-noise-variants raw_delay_gi_ccop,proposed \
  --colored-noise-grid 0,0.2,0.5,0.8 \
  --n-trials "${N}" --seed 20260726 \
  --diagnostic-mode performance \
  --jobs "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/colored_noise_boundary"

run_block model_order_mismatch \
  "${PY}" -m src.experiments.run_robustness_and_scaling_figures \
  --figures fig9 --snr-db 0 --true-k 3 \
  --assumed-k-grid 2,3,4,5 --baselines mksc_ccop \
  --include-trueK-peb-reference --grid-profile medium --strict-ris-geometry \
  --n-trials "${N}" --seed 20260803 \
  --jobs "${PAR_JOBS}" --process-workers "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/model_order_mismatch"

run_block ris_bs_calibration_boundary \
  "${PY}" -m src.experiments.run_robustness_and_scaling_figures \
  --figures fig8 --snr-db 0 --true-k 3 \
  --calibration-std-grid 0,1,2,5,10,20 \
  --baselines mksc_ccop,stage1_only \
  --include-calibration-oracle-peb --grid-profile medium --strict-ris-geometry \
  --n-trials "${N}" --seed 20260802 \
  --jobs "${PAR_JOBS}" --process-workers "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/ris_bs_calibration_boundary"

run_block robustness_scaling \
  "${PY}" -m src.experiments.run_final_mksc_ccop_robustness \
  --suites scaling --snr-db -10 \
  --scaling-variants scaled_4d,proposed \
  --n-grid 31,47,63,95 --training-grid 32,64,128,256 \
  --array-grid 4,8,16,24 --ris-side-grid 16,32,48,64 --k-grid 2,3,4 \
  --n-trials "${N}" --seed 20260728 \
  --diagnostic-mode performance \
  --jobs "${SCALING_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/robustness_scaling"

run_block evs_resolvability \
  "${PY}" -m src.experiments.run_final_evs_resolvability \
  --snr-db -10 \
  --delay-separation-grid-ns 0.1,0.2,0.5,1,2,5 \
  --polarization-overlap-grid 0.1,0.5,0.9,1.0 \
  --receiver-modes scalar,dual_pol,full_6d \
  --delay-error-tolerance-ns 0.5 --pole-collapse-tolerance-ns 0.05 \
  --n-trials "${N}" --seed 20260729 \
  --diagnostic-mode performance \
  --jobs "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/evs_resolvability"

run_block positions50 \
  "${PY}" -m src.experiments.run_final_mksc_ccop_robustness \
  --suites positions --snr-db -10 \
  --position-grid-shape 5,5,2 --position-grid-margin-m 0.1 \
  --position-variants scaled_4d,proposed --position-peb \
  --n-trials "${N}" --seed 20260727 \
  --diagnostic-mode performance \
  --jobs "${PAR_JOBS}" --blas-threads 1 \
  --out-dir "${OUT}/positions50"

echo "[$(date -Is)] VERIFY QUEUE COMPLETE"
