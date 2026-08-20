#!/usr/bin/env bash
# Re-run the campaigns invalidated by the baseline-fidelity changes:
#   * als_cpd                        -> Lin et al. algebraic GEVD CPD + ALS polish
#   * nf_ris_groupomp_localgrid_wls  -> Yan et al. Remark 2 path-consistency gate
#
# Every other campaign under results/paper_v3 is unaffected: a scan of the
# released trial CSVs shows only these three carry either baseline, so Fig. 1,
# 3, 4, 5 and Tables I-II do not move.
#
# Output goes to NEW directories so the frozen v3 campaign stays intact and the
# old and new summaries can be diffed.  Run from the repository root.
set -euo pipefail

PY="${PY:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

JOBS="${JOBS:-64}"          # the released runs used 64 process workers, 1 BLAS thread each
SNR="-20,-15,-10,-5,0,5,10,15,20,25,30"
ALL="als_cpd,scaled_4d,nf_ris_groupomp_localgrid_wls,ris_vbi_sbl,mksc_ccop,peb,constrained_jones_peb"

echo "### 0/3  fidelity + regression tests (must be green before spending GPU-days)"
$PY -m pytest tests/ -q

echo "### 1/3  headline benchmark, as-published tier   (~4.3 h at 64 workers)"
$PY -m src.experiments.run_benchmark_comparison \
  --n-trials 960 --paper-k 3 --seed 20260815 \
  --snr-grid="$SNR" \
  --baselines "$ALL" \
  --grid-profile medium --baseline-backend cpu \
  --baseline-refinement-tier as_published \
  --strict-ris-geometry --respect-existing-blas-env \
  --jobs "$JOBS" --process-workers "$JOBS" --blas-threads 1 \
  --no-plots --progress-heartbeat-s 300 \
  --out-dir results/paper_v3/benchmark_as_published_960_5

echo "### 2/3  refinement-matched diagnostic            (~7.1 h at 64 workers)"
$PY -m src.experiments.run_benchmark_comparison \
  --n-trials 960 --paper-k 3 --seed 20260526 \
  --snr-grid="$SNR" \
  --baselines "$ALL" \
  --grid-profile medium --baseline-backend cpu \
  --baseline-refinement-tier refinement_matched \
  --strict-ris-geometry --respect-existing-blas-env \
  --jobs "$JOBS" --process-workers "$JOBS" --blas-threads 1 \
  --no-plots --progress-heartbeat-s 300 \
  --out-dir results/paper_v3/benchmark_refinement_matched_960_4

echo "### 3/3  serialized runtime/memory suite          (~0.35 h, MUST stay single-threaded)"
$PY -m src.experiments.run_benchmark_comparison \
  --n-trials 30 --paper-k 3 --seed 20260731 \
  --snr-grid=-10,0 \
  --baselines als_cpd,ris_vbi_sbl,nf_ris_groupomp_localgrid_wls,scaled_4d,mksc_ccop \
  --grid-profile medium --baseline-backend cpu \
  --baseline-refinement-tier as_published \
  --runtime-profile --profile-memory \
  --strict-ris-geometry --no-constrained-jones-peb --respect-existing-blas-env \
  --jobs 1 --process-workers 1 --blas-threads 1 \
  --no-plots \
  --out-dir results/paper_v3/benchmark_runtime30_cpu_as_published_v2

echo
echo "Done.  Next:"
echo "  1) point scripts/make_paper_figures_twc.py CAMPAIGNS['v3'] at the new dirs:"
echo "       benchmark        -> benchmark_as_published_960_5"
echo "       benchmark_clock  -> benchmark_as_published_960_5"
echo "       benchmark_runtime-> benchmark_runtime30_cpu_as_published_v2"
echo "  2) python scripts/make_paper_figures_twc.py --fig benchmark --fig benchmark_clock --fig cost --out-dir tex/fig3"
echo "     python scripts/make_paper_figures_twc.py --fig benchmark --fig benchmark_clock --fig cost --snr-min -30 --out-dir tex/figs_supp"
echo "  3) refresh the numbers listed in docs_for_codex/RERUN_NUMBERS.md"
