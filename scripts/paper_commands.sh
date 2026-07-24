#!/usr/bin/env bash
# Authoritative, version-controlled commands for the frozen paper experiments.
# Run exactly one target per invocation. Formal outputs are intentionally never
# overwritten; use a new freeze version instead of --force-rerun.

set -euo pipefail

readonly FREEZE_TAG="paper-freeze-mksc-gi-ccop-jvp-v1"
readonly EXPECTED_BRANCH="research/ccop_full_validation"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly PYTHON_BIN="${PAPER_PYTHON:-python}"
readonly DRY_RUN="${PAPER_DRY_RUN:-0}"

export PYTHONHASHSEED=0
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage: bash scripts/paper_commands.sh TARGET

Control targets:
  list                         list all formal experiment targets
  preflight                    verify the frozen source/environment only

Required gates:
  smoke_ablation               deterministic 3-trial ablation smoke
  smoke_robustness             deterministic 1-trial mismatch smoke

Formal accuracy targets:
  components_paper400          compact component ablation, seed 20260721
  snr_internal_paper200        position/clock/channel/tail SNR curves
  c3_clock_units_paper100      strong-init 4-D/CCOP and clock-unit control
  receiver_information_paper200 scalar/dual-pol/full-EVS plus matched PEB
  compression_matched_paper200 raw-delay versus MKSC compression
  benchmark_accuracy200_gpu    external benchmark accuracy with CuPy
  maxwell_mismatch_paper150    Maxwell-subspace mismatch boundary
  colored_noise_boundary150    unwhitened colored-noise boundary
  positions50x30               50-position geometry generalization
  robustness_scaling50         N/T/M_A/M_R/K scaling
  evs_resolvability_paper200   delay-polarization resolution probability

Dedicated cost targets:
  components_cost30_cpu        serial component runtime/peak RSS
  benchmark_runtime30_cpu      serial external-baseline runtime/peak RSS

Examples:
  bash scripts/paper_commands.sh preflight
  PAPER_DRY_RUN=1 bash scripts/paper_commands.sh components_paper400
  bash scripts/paper_commands.sh components_paper400
EOF
}

list_targets() {
  usage
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

preflight() {
  command -v git >/dev/null 2>&1 || die "git is not available"
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python executable not found: ${PYTHON_BIN}"

  local status head tag_commit branch
  status="$(git status --porcelain)"
  [[ -z "${status}" ]] || die "worktree is dirty; commit/tag a reviewed source state before any paper run"

  head="$(git rev-parse HEAD)"
  tag_commit="$(git rev-parse "${FREEZE_TAG}^{commit}" 2>/dev/null)" || \
    die "required freeze tag does not exist: ${FREEZE_TAG}"
  [[ "${head}" == "${tag_commit}" ]] || \
    die "HEAD ${head} does not equal ${FREEZE_TAG} (${tag_commit})"

  branch="$(git branch --show-current)"
  if [[ -n "${branch}" && "${branch}" != "${EXPECTED_BRANCH}" ]]; then
    die "expected branch ${EXPECTED_BRANCH}, found ${branch}"
  fi

  "${PYTHON_BIN}" - <<'PY'
import platform
import sys

import numpy
import scipy

print(f"python={sys.version.split()[0]} executable={sys.executable}")
print(f"platform={platform.platform()}")
print(f"numpy={numpy.__version__} scipy={scipy.__version__}")
PY
  printf 'freeze_tag=%s\ncommit=%s\nbranch=%s\nworktree=clean\n' \
    "${FREEZE_TAG}" "${head}" "${branch:-DETACHED}"
}

require_gpu() {
  "${PYTHON_BIN}" - <<'PY'
import cupy

count = int(cupy.cuda.runtime.getDeviceCount())
if count < 1:
    raise SystemExit("CuPy is installed but no CUDA device is visible")
with cupy.cuda.Device(0):
    name = cupy.cuda.runtime.getDeviceProperties(0)["name"]
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
print(f"cupy={cupy.__version__} cuda_devices={count} gpu0={name}")
PY
}

require_new_output() {
  local out_dir="$1"
  [[ ! -e "${out_dir}" ]] || \
    die "formal output already exists: ${out_dir}; do not overwrite it"
}

run_command() {
  local out_dir="$1"
  shift
  require_new_output "${out_dir}"
  printf 'command:'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'dry_run=1; estimator was not executed\n'
    return 0
  fi
  "$@"
}

target="${1:-}"
case "${target}" in
  ""|-h|--help)
    usage
    exit 0
    ;;
  list)
    list_targets
    exit 0
    ;;
  preflight)
    preflight
    exit 0
    ;;
esac

preflight

case "${target}" in
  smoke_ablation)
    out="results/final_mksc_ccop/ablation_smoke3"
    run_command "${out}" "${PYTHON_BIN}" -m src.experiments.run_final_mksc_ccop_ablation \
      --suites components --n-trials 3 --seed 20260720 \
      --diagnostic-mode fast --jobs 1 --blas-threads 1 --out-dir "${out}"
    ;;
  smoke_robustness)
    out="results/final_mksc_ccop/robustness_code_smoke1"
    run_command "${out}" "${PYTHON_BIN}" -m src.experiments.run_final_mksc_ccop_robustness \
      --suites subspace_mismatch --snr-db -10 --phase-grid 2 --gain-grid 0.05 \
      --ris-bs-angle-grid 0.5 --bs-sensor-position-mm-grid 0.2 \
      --mismatch-variants raw_delay_gi_ccop,proposed --n-trials 1 --seed 20260720 \
      --diagnostic-mode fast --jobs 1 --blas-threads 1 --out-dir "${out}"
    ;;
  components_paper400)
    out="results/final_mksc_ccop/components_paper400"
    run_command "${out}" "${PYTHON_BIN}" -m src.experiments.run_final_mksc_ccop_ablation \
      --suites components --focus-snr-db -10 --n-trials 400 --seed 20260721 \
      --bootstrap-replicates 10000 --diagnostic-mode performance \
      --jobs 8 --blas-threads 1 --out-dir "${out}"
    ;;
  components_cost30_cpu)
    out="results/final_mksc_ccop/components_cost30_cpu"
    run_command "${out}" "${PYTHON_BIN}" -m src.experiments.run_final_mksc_ccop_ablation \
      --suites components --focus-snr-db -10 --n-trials 30 --seed 20260731 \
      --bootstrap-replicates 10000 --profile-memory --diagnostic-mode performance \
      --jobs 1 --blas-threads 1 --out-dir "${out}"
    ;;
  snr_internal_paper200)
    out="results/final_mksc_ccop/snr_internal_paper200"
    run_command "${out}" "${PYTHON_BIN}" -m src.experiments.run_final_mksc_ccop_ablation \
      --suites snr --snr-grid=-30,-25,-20,-15,-10,-5,0,5,10,15,20 \
      --snr-variants scaled_4d,old_stage1_ccop,mksc_gi_4_no_refresh_ccop,proposed \
      --n-trials 200 --seed 20260722 --bootstrap-replicates 10000 \
      --diagnostic-mode performance --jobs 8 --blas-threads 1 --out-dir "${out}"
    ;;
  c3_clock_units_paper100)
    out="results/final_mksc_ccop/c3_clock_units_paper100"
    run_command "${out}" "${PYTHON_BIN}" -m src.experiments.run_final_mksc_ccop_ablation \
      --suites c3_matrix --snr-grid=-10 --focus-snr-db -10 \
      --n-trials 100 --seed 20260732 --bootstrap-replicates 5000 \
      --diagnostic-mode performance --jobs 4 --blas-threads 1 \
      --c3-variants old_4d,scaled_4d,old_stage1_ccop,mksc_gi_refresh_4d_seconds,mksc_gi_refresh_4d_nanoseconds,mksc_gi_refresh_4d_distance_m,proposed,mksc_gi_refresh_ccop_seconds,mksc_gi_refresh_ccop_nanoseconds,mksc_gi_refresh_ccop_distance_m \
      --paired-reference mksc_gi_refresh_4d_distance_m \
      --paired-candidate proposed --out-dir "${out}"
    ;;
  receiver_information_paper200)
    out="results/final_mksc_ccop/receiver_information_paper200"
    run_command "${out}" "${PYTHON_BIN}" -m src.experiments.run_final_mksc_ccop_ablation \
      --suites receiver --snr-grid=-30,-25,-20,-15,-10,-5,0,5,10,15,20 \
      --receiver-modes scalar,dual_pol,full_6d --receiver-variants proposed \
      --n-trials 200 --seed 20260723 --bootstrap-replicates 10000 \
      --diagnostic-mode performance --jobs 8 --blas-threads 1 --out-dir "${out}"
    ;;
  compression_matched_paper200)
    out="results/final_mksc_ccop/compression_matched_paper200"
    run_command "${out}" "${PYTHON_BIN}" -m src.experiments.run_final_mksc_ccop_ablation \
      --suites compression --snr-grid=-20,-15,-10,-5,0,5,10 \
      --n-trials 200 --seed 20260724 --bootstrap-replicates 10000 \
      --diagnostic-mode performance --jobs 8 --blas-threads 1 --out-dir "${out}"
    ;;
  benchmark_accuracy200_gpu)
    require_gpu
    out="results/final_mksc_ccop/benchmark_accuracy200_gpu"
    run_command "${out}" "${PYTHON_BIN}" -m src.experiments.run_benchmark_comparison \
      --n-trials 200 --seed 20260730 \
      --snr-grid=-30,-25,-20,-15,-10,-5,0,5,10,15,20 \
      --baselines als_cpd,scaled_4d,nf_ris_groupomp_localgrid_wls,ris_momp,mksc_ccop,peb,constrained_jones_peb \
      --grid-profile fine --baseline-backend cupy --gpu-device 0 \
      --jobs 1 --process-workers 1 --blas-threads 4 --out-dir "${out}"
    ;;
  benchmark_runtime30_cpu)
    out="results/final_mksc_ccop/benchmark_runtime30_cpu"
    run_command "${out}" "${PYTHON_BIN}" -m src.experiments.run_benchmark_comparison \
      --n-trials 30 --seed 20260731 --snr-grid=-10,0 \
      --baselines als_cpd,scaled_4d,nf_ris_groupomp_localgrid_wls,ris_momp,mksc_ccop \
      --grid-profile fine --baseline-backend cpu --runtime-profile --profile-memory \
      --jobs 1 --process-workers 1 --blas-threads 1 --out-dir "${out}"
    ;;
  maxwell_mismatch_paper150)
    out="results/final_mksc_ccop/maxwell_mismatch_paper150"
    run_command "${out}" "${PYTHON_BIN}" -m src.experiments.run_final_mksc_ccop_robustness \
      --suites subspace_mismatch --snr-db -10 \
      --mismatch-variants raw_delay_gi_ccop,proposed \
      --phase-grid 0,1,2,5,10 --gain-grid 0,0.01,0.02,0.05,0.1 \
      --ris-bs-angle-grid 0,0.1,0.25,0.5,1 \
      --bs-sensor-position-mm-grid 0,0.05,0.1,0.2,0.5 \
      --n-trials 150 --seed 20260725 --diagnostic-mode performance \
      --jobs 8 --blas-threads 1 --out-dir "${out}"
    ;;
  colored_noise_boundary150)
    out="results/final_mksc_ccop/colored_noise_boundary150"
    run_command "${out}" "${PYTHON_BIN}" -m src.experiments.run_final_mksc_ccop_robustness \
      --suites colored_noise --snr-db -10 \
      --colored-noise-variants raw_delay_gi_ccop,proposed \
      --colored-noise-grid 0,0.2,0.5,0.8 --n-trials 150 --seed 20260726 \
      --diagnostic-mode performance --jobs 8 --blas-threads 1 --out-dir "${out}"
    ;;
  positions50x30)
    out="results/final_mksc_ccop/positions50x30"
    run_command "${out}" "${PYTHON_BIN}" -m src.experiments.run_final_mksc_ccop_robustness \
      --suites positions --snr-db -10 --position-grid-shape 5,5,2 \
      --position-grid-margin-m 0.1 --position-variants scaled_4d,proposed --position-peb \
      --n-trials 30 --seed 20260727 --diagnostic-mode performance \
      --jobs 8 --blas-threads 1 --out-dir "${out}"
    ;;
  robustness_scaling50)
    out="results/final_mksc_ccop/robustness_scaling50"
    run_command "${out}" "${PYTHON_BIN}" -m src.experiments.run_final_mksc_ccop_robustness \
      --suites scaling --snr-db -10 --scaling-variants scaled_4d,proposed \
      --n-grid 31,47,63,95 --training-grid 32,64,128,256 \
      --array-grid 4,8,16,24 --ris-side-grid 16,32,48,64 --k-grid 2,3,4 \
      --n-trials 50 --seed 20260728 --diagnostic-mode performance \
      --jobs 8 --blas-threads 1 --out-dir "${out}"
    ;;
  evs_resolvability_paper200)
    out="results/final_mksc_ccop/evs_resolvability_paper200"
    run_command "${out}" "${PYTHON_BIN}" -m src.experiments.run_final_evs_resolvability \
      --snr-db -10 --delay-separation-grid-ns 0.1,0.2,0.5,1,2,5 \
      --polarization-overlap-grid 0.1,0.5,0.9,1.0 \
      --receiver-modes scalar,dual_pol,full_6d \
      --delay-error-tolerance-ns 0.5 --pole-collapse-tolerance-ns 0.05 \
      --n-trials 200 --seed 20260729 --diagnostic-mode performance \
      --jobs 8 --blas-threads 1 --out-dir "${out}"
    ;;
  *)
    usage >&2
    die "unknown target: ${target}"
    ;;
esac
