#!/usr/bin/env python3
"""
Quick test script to verify the lambda sensitivity script works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from oldcode.legacy_stage2.main_lambda_sensitivity import (
    extract_metrics_from_result,
    parse_lambda_list,
)

def test_parse_lambda_list():
    """Test lambda list parsing."""
    test_cases = [
        ("1e-8,1e-7,1e-6", [1e-8, 1e-7, 1e-6]),
        ("1e-4,1e-5", [1e-4, 1e-5]),
        ("1e-6", [1e-6]),
    ]

    for input_str, expected in test_cases:
        result = parse_lambda_list(input_str)
        assert result == expected, f"Expected {expected}, got {result}"
        print(f"✓ '{input_str}' -> {result}")

def test_extract_metrics():
    """Test metrics extraction from dummy result."""
    dummy_result = {
        'selected_branch': 'direct_vp',
        'final': {
            'position_rmse': 0.05,
            'Y_hat': None,  # Placeholder
            'raw_objective': 1e-6
        },
        'Y_true': None,  # Placeholder
        'timing': {'total': 10.5},
        'reliability': {
            'assignment_margin': 0.8,
            'sigma_delta_t_ns': 0.2,
            'rank1_ratio_max': 0.95,
            'initial_z_residual': 0.1
        },
        'direct_vp_quality': {'good': True},  # This is at the top level
        'branches': {
            'direct_vp': {
                'final': {'raw_objective': 1e-6, 'vp_nfev': 5, 'global_vp_num_iter': 5}
            }
        }
    }

    # Debug: check branches structure
    print(f"branches in result: {'branches' in dummy_result}")
    if 'branches' in dummy_result:
        print(f"direct_vp in branches: {'direct_vp' in dummy_result['branches']}")
        if 'direct_vp' in dummy_result['branches']:
            dvp = dummy_result['branches']['direct_vp']
            print(f"final in direct_vp: {'final' in dvp}")
            if 'final' in dvp:
                print(f"vp_nfev in final: {'vp_nfev' in dvp['final']}")
                print(f"vp_nfev value: {dvp['final'].get('vp_nfev', 'MISSING')}")

    metrics = extract_metrics_from_result(dummy_result, 1e-6, 1e-12, -10, 2025)

    # Debug: print all metrics
    print("Extracted metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Debug: print all metrics
    print("Extracted metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Check some key metrics
    assert metrics['lambda_rel'] == 1e-6
    assert metrics['lambda_floor'] == 1e-12
    assert metrics['selected_branch'] == 'direct_vp'
    assert metrics['ue_position_rmse_m'] == 0.05
    print(f"direct_vp_good: {metrics['direct_vp_good']} (type: {type(metrics['direct_vp_good'])})")
    assert metrics['direct_vp_good'] == True
    assert metrics['direct_vp_nfev'] == 5

    print("✓ Metrics extraction test passed")

if __name__ == '__main__':
    test_parse_lambda_list()
    test_extract_metrics()
    print("All tests passed!")
