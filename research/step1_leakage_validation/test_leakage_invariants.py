"""Automated tests for overlapping-label leakage invariants.

These tests validate the core property that PurgedKFold maintains:
no training observation's label interval overlaps the test fold's intervals.

Run with: pytest test_leakage_invariants.py -v
"""

import numpy as np
import pandas as pd
from synthetic_leakage_proof import (
    build_synthetic_data,
    assert_no_label_overlap,
)

try:
    from purgedcv import PurgedKFold
    from sklearn.model_selection import KFold
except ImportError:
    raise ImportError("Required: pip install purgedcv scikit-learn")


def test_synthetic_data_independence():
    """TEST A: Feature and target are generated independently."""
    data = build_synthetic_data(seed=0, n=1000, h=20)
    
    # x and e are independent draws → x and y are independent by construction
    # The empirical correlation should be near zero
    corr = abs(data["spurious_corr"])
    assert corr < 0.5, (
        f"Spurious correlation should be weak, got {corr:.3f}. "
        "This is a finite-sample diagnostic, not proof of relationship."
    )
    print(f"✓ TEST A PASS: corr(x, y) = {data['spurious_corr']:+.3f} (independent by construction)")


def test_naive_kfold_allows_overlap():
    """TEST B: Naive shuffled KFold allows training-test label-interval overlap."""
    data = build_synthetic_data(seed=0, n=1000, h=20)
    
    naive = KFold(n_splits=5, shuffle=True, random_state=0)
    
    total_overlap_count = 0
    total_train_size = 0
    
    for train_idx, test_idx in naive.split(data["features"], data["y"]):
        overlap = assert_no_label_overlap(train_idx, test_idx, data["pred"], data["evalu"])
        total_overlap_count += overlap
        total_train_size += len(train_idx)
    
    contamination = total_overlap_count / total_train_size
    
    # Naive should have substantial contamination (typically 80-99%)
    assert contamination > 0.5, (
        f"Naive KFold should allow high overlap, got {contamination:.1%}"
    )
    print(f"✓ TEST B PASS: Naive KFold contamination = {contamination:.1%}")


def test_purged_kfold_zero_overlap():
    """TEST C: PurgedKFold achieves zero label-interval overlap (strict invariant)."""
    data = build_synthetic_data(seed=0, n=1000, h=20)
    
    purged = PurgedKFold(
        n_splits=5,
        prediction_times=data["pred"],
        evaluation_times=data["evalu"],
    )
    
    total_overlap_count = 0
    total_train_size = 0
    
    for train_idx, test_idx in purged.split(data["features"], data["y"]):
        overlap = assert_no_label_overlap(train_idx, test_idx, data["pred"], data["evalu"])
        total_overlap_count += overlap
        total_train_size += len(train_idx)
    
    contamination = total_overlap_count / total_train_size if total_train_size > 0 else 0.0
    
    # Purged should have exactly zero overlap (strict requirement)
    assert contamination == 0.0, (
        f"PurgedKFold should achieve zero contamination, got {contamination:.1%}. "
        "This is a hard invariant."
    )
    print(f"✓ TEST C PASS: PurgedKFold contamination = {contamination:.1%} (strict zero)")


def test_consecutive_label_windows_overlap():
    """TEST D: Consecutive label intervals overlap by exactly H-1 days."""
    data = build_synthetic_data(seed=0, n=1000, h=20)
    h = data["h"]
    pred = data["pred"]
    evalu = data["evalu"]
    
    for i in range(len(pred) - 1):
        # Label windows: [pred[i], evalu[i]) and [pred[i+1], evalu[i+1])
        overlap_start = max(pred.iloc[i], pred.iloc[i + 1])
        overlap_end = min(evalu.iloc[i], evalu.iloc[i + 1])
        
        if overlap_start < overlap_end:
            duration = (overlap_end - overlap_start).days
            assert duration == h - 1, (
                f"Consecutive labels should overlap by {h-1} days, "
                f"got {duration} at index {i}"
            )
    
    print(f"✓ TEST D PASS: Consecutive labels overlap by exactly {h-1} days")


def test_label_intervals_are_half_open():
    """TEST E: Label intervals are half-open [pred, evalu) as expected."""
    data = build_synthetic_data(seed=0, n=100, h=20)
    
    # Verify interval properties
    for i in range(len(data["pred"])):
        pred_t = data["pred"].iloc[i]
        evalu_t = data["evalu"].iloc[i]
        
        # evalu should be exactly h days after pred
        expected_evalu = pred_t + pd.Timedelta(days=data["h"])
        assert evalu_t == expected_evalu, (
            f"Evaluation time should be pred + {data['h']} days, "
            f"got {evalu_t} (expected {expected_evalu})"
        )
    
    print(f"✓ TEST E PASS: Label intervals are [pred, pred+{data['h']} days)")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("RUNNING AUTOMATED TESTS FOR OVERLAPPING-LABEL LEAKAGE INVARIANTS")
    print("="*80 + "\n")
    
    test_synthetic_data_independence()
    test_naive_kfold_allows_overlap()
    test_purged_kfold_zero_overlap()
    test_consecutive_label_windows_overlap()
    test_label_intervals_are_half_open()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED")
    print("="*80 + "\n")
    print("""
SUMMARY:
  ✓ Feature and target are independent by construction
  ✓ Naive KFold allows substantial label-interval overlap
  ✓ PurgedKFold achieves zero label-interval overlap (strict invariant)
  ✓ Consecutive label windows overlap by H-1 days (by design)
  ✓ Label intervals are half-open [pred, evalu) (as expected)

These invariants form the foundation for the validation architecture.
""")
