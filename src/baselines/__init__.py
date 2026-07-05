"""Benchmark baselines for EVS-MIMO-RIS experiments."""

from .common import BaselineResult
from .nf_ris_groupomp_localgrid_wls import run_nf_ris_groupomp_localgrid_wls_baseline

__all__ = ["BaselineResult", "run_nf_ris_groupomp_localgrid_wls_baseline"]
