# Step 1: Overlapping-Label Leakage Validation

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main experiment
python synthetic_leakage_proof.py

# Run automated tests
pytest test_leakage_invariants.py -v
```

## What This Does

This directory contains a controlled experiment proving that **shuffled cross-validation is invalid for overlapping-label time series**.

The experiment builds synthetic data where:
- Feature `x` and target `y` are **completely independent** (zero true predictive power)
- But `y` is computed from overlapping forward-looking windows
- Naive shuffled KFold reports R² ≈ 0.8–0.9 (fabricated)
- PurgedKFold reports R² ≈ 0 (correct, no skill)

## Files

| File | Purpose |
|------|----------|
| `synthetic_leakage_proof.py` | Main experiment: data generation, CV comparison, control |
| `test_leakage_invariants.py` | Automated tests validating core invariants |
| `STEP_1_REPORT.md` | Detailed report: results, interpretation, acceptance criteria |
| `requirements.txt` | Minimal dependencies |
| `README.md` | This file |

## Core Insight

**Label-interval overlap is the root cause of leakage.**

When a 20-day forward label is computed every day:
- Label at t=0 covers [2020-01-01, 2020-01-21)
- Label at t=1 covers [2020-01-02, 2020-01-22)
- They share 19 of 20 days

If shuffled CV puts t=0 in training and t=1 in test, training labels contain information about the same outcome window as test. A flexible model exploits this to appear predictive.

**Solution: Remove training observations whose label intervals overlap test.**

This is what PurgedKFold does.

## Expected Results

```
                              Naive R²   Purged R²   Naive Overlap   Purged Overlap
─────────────────────────────────────────────────────────────────────────────────────
RandomForest                   0.910     -1.868          98%              0%
k-NN                           0.827     -1.477          98%              0%
Baseline (predict-the-mean)   -0.012     -0.130          98%              0%
```

Key observation: **Purged overlap is exactly 0% (strict mechanical invariant).**

## Running the Experiment

### Main Experiment (≈30 seconds)

```bash
python synthetic_leakage_proof.py
```

Output:
- Data construction and verification
- Label window overlap check
- Naive vs. Purged comparison
- Control experiment (shuffled features)
- TEST A–D results
- Mechanical invariant checks

### Automated Tests (≈10 seconds)

```bash
pytest test_leakage_invariants.py -v
```

Tests:
- TEST A: Independence by construction
- TEST B: Naive KFold allows overlap
- TEST C: PurgedKFold achieves zero overlap (STRICT)
- TEST D: Consecutive labels overlap by H-1 days
- TEST E: Interval structure is correct

## Interpreting Results

### Critical Invariant: Purged Contamination = 0%

This **must** be exactly zero. If it is not:
```python
assert purged_contamination == 0.0  # STRICT
```

If this fails, stop and investigate:
- Is PurgedKFold working correctly?
- Is the overlap detection correct?
- Is there a bug in interval logic?

### Expected Deviations

These may vary across runs and platforms (do not worry):
- Naive R² (typically 0.8–0.9, but could be different)
- Control R² (typically ≈0, might be -0.1 to +0.1)
- Exact overlap fractions (but naive should be >> purged)

What **must not vary**:
- Purged contamination = 0%
- Consecutive label overlap = H-1 days

## For the Curious

### Why Does This Matter?

If you build a stock prediction model on financial data using shuffled CV:
- You think your model is 85% accurate
- You deploy it with confidence
- It loses money on real data
- Reason: your CV was validating leakage, not skill

This experiment proves that can happen even with completely unpredictable synthetic data.

### What's the Real Solution?

Not "use time-series CV." Time-series CV helps but doesn't guarantee disjoint label intervals.

The solution: **PurgedKFold** (or similar) that explicitly removes training observations whose label windows overlap test windows.

### What's the Catch?

Purged CV requires knowing when each label resolves (the `evaluation_times` parameter). If you mis-specify that, some leak remains. See `synthetic_leakage_proof.py` sensitivity section for how this works.

## No Production Code Here

This directory contains only:
- ✅ Experimental validation
- ✅ Synthetic data generation
- ✅ Cross-validation comparison
- ❌ No real ML models
- ❌ No financial data ingestion
- ❌ No application code

Step 1 is about *understanding the problem*, not building the solution.

## Next: Step 2

Once Step 1 is complete and understood, Step 2 will:
1. Design the production validation architecture
2. Implement purged CV in the pipeline
3. Add embargo (post-test buffer)
4. Build walk-forward backtesting framework
5. Integrate with model selection

But that is future work. For now: **understand the leakage.**
