import csv

import numpy as np

from src import global_vp
from src import main_single_proposed as main_single
from src.config import default_config
from src.global_vp import (
    distance_to_box_boundary,
    select_z_rescue_candidate,
    z_rescue_starts,
)


def test_boundary_detection_identifies_upper_z_bound():
    bounds = np.array([[0.0, 2.0], [-1.0, 1.0], [0.35, 1.45]])
    diag = distance_to_box_boundary(np.array([1.0, 0.0, 1.44]), bounds)
    assert diag["boundary_hit"] is True
    assert diag["boundary_hit_axis"] == "z"
    assert np.isclose(diag["distance_to_position_box_boundary_m"], 0.01)


def test_boundary_detection_does_not_flag_interior_point():
    bounds = np.array([[0.0, 2.0], [-1.0, 1.0], [0.35, 1.45]])
    diag = distance_to_box_boundary(np.array([1.0, 0.0, 0.75]), bounds)
    assert diag["boundary_hit"] is False
    assert diag["boundary_hit_axis"] == []


def test_z_rescue_starts_span_bounds_and_include_interior_values():
    bounds = np.array([[0.0, 2.0], [-1.0, 1.0], [0.35, 1.45]])
    starts = z_rescue_starts(np.array([1.2, 0.3, 1.45]), bounds, 7, 0.02)
    z_values = np.array([start[2] for start in starts])
    assert np.isclose(z_values[0], 0.37)
    assert np.isclose(z_values[-1], 1.43)
    assert np.any((z_values > 0.5) & (z_values < 1.3))
    assert all(np.allclose(start[:2], [1.2, 0.3]) for start in starts)


def test_probe_then_refine_runs_only_best_z_start_and_stops_at_noise_floor(
    monkeypatch,
):
    config = default_config()
    config["noise_variance"] = 1.0
    config["global_vp"] = dict(config["global_vp"])
    config["global_vp"].update(
        {
            "z_rescue_strategy": "probe_then_refine",
            "z_rescue_num_starts": 7,
            "z_rescue_num_full_refines": 2,
            "z_rescue_early_stop_noise_floor_factor": 1.02,
        }
    )
    scene = {"K": 1, "I": 1, "N": 1, "T": 1}
    full_calls = []
    probe_calls = []

    def fake_once(y_raw, estimate, scene_arg, config_arg):
        start = np.asarray(
            estimate.get("_global_vp_initial_p_u", [1.2, 0.3, 1.45]), dtype=float
        )
        full_calls.append(float(start[2]))
        score = 1.0 + (float(start[2]) - 0.75) ** 2
        return {
            "p_u": start.copy(),
            "delta_t": 0.0,
            "raw_objective_final": score,
            "raw_objective": score,
            "global_vp_success": True,
        }

    def fake_probe(y_raw, estimate, scene_arg, config_arg):
        start = np.asarray(estimate["_global_vp_initial_p_u"], dtype=float)
        probe_calls.append(float(start[2]))
        score = 1.0 + (float(start[2]) - 0.75) ** 2
        return {"raw_objective_final": score}

    monkeypatch.setattr(global_vp, "_global_exact_spherical_vp_refinement_once", fake_once)
    monkeypatch.setattr(global_vp, "_legacy_vp_initial_result", fake_probe)

    result = global_vp.global_exact_spherical_vp_refinement(
        np.ones((1, 1, 1), dtype=complex), {}, scene, config
    )

    assert len(probe_calls) == 7
    assert len(full_calls) == 2  # normal plus one shortlisted full refinement
    assert result["z_rescue_num_full_refines"] == 1
    assert result["z_rescue_num_probes"] == 7
    assert result["z_rescue_strategy"] == "probe_then_refine"
    assert abs(result["p_u"][2] - 0.75) < 0.05


def test_z_rescue_selection_chooses_lower_raw_objective():
    bounds = default_config()["ue_bounds"]
    candidates = [
        {"p_u": np.array([1.0, 0.0, 1.45]), "raw_objective_final": 1.0},
        {"p_u": np.array([1.0, 0.0, 0.75]), "raw_objective_final": 0.9},
    ]
    selected, reason = select_z_rescue_candidate(candidates, bounds)
    assert selected is candidates[1]
    assert reason == "lowest_raw_objective"


def test_boundary_tie_break_does_not_choose_much_worse_interior():
    config = default_config()
    reliability = {"decision": "jnpp_then_vp"}
    direct = {
        "final": {"p_u": np.array([1.0, 0.0, 1.45]), "raw_objective_final": 1.0},
        "Y_true": np.ones(1),
    }
    close = {
        "branch_name": "rescue",
        "structured_diag": {},
        "final": {"p_u": np.array([1.0, 0.0, 0.75]), "raw_objective_final": 1.0005},
        "Y_true": np.ones(1),
    }
    selected, _ = main_single.select_proposed_branch(
        direct, close, reliability, config
    )
    assert selected["selected_branch"] == "rescue"
    assert selected["boundary_selection_rule_used"] is True

    much_worse = {
        **close,
        "final": {"p_u": np.array([1.0, 0.0, 0.75]), "raw_objective_final": 1.1},
    }
    selected, _ = main_single.select_proposed_branch(
        direct, much_worse, reliability, config
    )
    assert selected["selected_branch"] == "direct_vp_rollback"
    assert selected["warning"] == "boundary_solution_retained_due_to_lower_residual"


def test_direct_vp_rollback_alone_is_not_a_boundary_failure():
    config = default_config()
    reliability = {"decision": "jnpp_then_vp"}
    direct = {
        "final": {"p_u": np.array([1.0, 0.0, 0.75]), "raw_objective_final": 1.0},
        "Y_true": np.ones(1),
    }
    rescue = {
        "branch_name": "rescue",
        "structured_diag": {},
        "final": {"p_u": np.array([1.1, 0.0, 0.75]), "raw_objective_final": 1.0},
        "Y_true": np.ones(1),
    }
    selected, _ = main_single.select_proposed_branch(
        direct, rescue, reliability, config
    )
    assert selected["selected_branch"] == "direct_vp_rollback"
    assert selected["direct_boundary_hit"] is False
    assert selected["warning"] == ""


def test_repeat_output_contains_boundary_and_z_rescue_diagnostics(
    tmp_path, monkeypatch
):
    class SyncPool:
        def __init__(self, processes, initializer=None, initargs=()):
            if initializer:
                initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def imap_unordered(self, function, tasks, chunksize=1):
            return map(function, tasks)

    def fake_run(config):
        p = np.array([1.0, 0.0, 0.75])
        y = np.ones((1, 1, 1), dtype=complex)
        return {
            "scene": {"p_u_true": p, "delta_t_true": 0.0, "c0": 3.0e8},
            "Y_true": y,
            "Y_noisy": y,
            "estimate_initial": {"p_u": p},
            "estimate_used": {"p_u": p},
            "stage1_config": config,
            "selected_branch": "direct_vp_rollback",
            "final": {
                "p_u": p,
                "delta_t": 0.0,
                "Y_hat": y,
                "raw_objective_final": 1.0,
                "z_rescue_triggered": True,
                "z_rescue_num_starts": 7,
                "z_rescue_best_z": 0.75,
                "z_rescue_selected_reason": "lowest_raw_objective",
            },
        }

    monkeypatch.setattr(main_single.mp, "Pool", SyncPool)
    monkeypatch.setattr(main_single, "run_single_proposed_diagnostic", fake_run)
    main_single.run_repeated_main_single(1, 1, 1, tmp_path)
    with (tmp_path / "main_single_repeat_trials.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    for field in (
        "boundary_hit",
        "distance_to_position_box_boundary_m",
        "z_rescue_triggered",
        "z_rescue_num_starts",
        "z_rescue_best_z",
        "z_rescue_selected_reason",
        "direct_boundary_hit",
        "rescue_boundary_hit",
        "branch_score_margin",
        "warning",
    ):
        assert field in row
    assert (tmp_path / "main_single_repeat_outliers.csv").exists()


def test_rerun_seeds_are_parsed_and_used_exactly(tmp_path, monkeypatch):
    captured = {}

    def fake_repeat(**kwargs):
        captured.update(kwargs)
        return {
            "summary": {
                "n_runs": 3,
                "n_success": 3,
                "position_rmse_m": 0.0,
            },
            "out_dir": tmp_path,
        }

    monkeypatch.setattr(main_single, "run_repeated_main_single", fake_repeat)
    monkeypatch.setattr(main_single, "_print_repeated_summary", lambda result: None)
    seeds = [3933360494, 3685333214, 1092560480]
    main_single.main(
        [
            "--rerun-seeds",
            ",".join(str(seed) for seed in seeds),
            "--repeat-out-dir",
            str(tmp_path),
            "--force-rerun",
        ]
    )
    assert captured["rerun_seeds"] == seeds
    assert captured["n_runs"] == 3
