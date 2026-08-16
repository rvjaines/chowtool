"""
STEP 1 REVIEW GUIDE: How to Understand and Run the Leakage Validation Experiment

This is a guided walkthrough for reviewing the STEP_1_REPORT.md and running 
the experiments to see the results firsthand.

TOTAL TIME: ~10 minutes (2 min read + 5 min experiments + 3 min interpretation)

PREREQUISITE KNOWLEDGE REQUIRED:
  - Basic Python and pandas
  - What cross-validation is (k-fold, train/test split)
  - What R² (coefficient of determination) means
  
NO PREREQUISITE KNOWLEDGE REQUIRED:
  - Financial markets
  - Advanced time series theory
  - Machine learning expertise
"""

import textwrap


def print_section(title, level=1):
    """Print a formatted section header."""
    if level == 1:
        print("\n" + "=" * 100)
        print(title.center(100))
        print("=" * 100 + "\n")
    elif level == 2:
        print("\n" + "─" * 100)
        print(title)
        print("─" * 100 + "\n")
    else:
        print(f"\n### {title}\n")


def print_concept(concept, explanation):
    """Print a concept-explanation pair."""
    print(f"📌 {concept}")
    for line in textwrap.wrap(explanation, width=90):
        print(f"   {line}")
    print()


def print_step(number, instruction):
    """Print a numbered step."""
    print(f"STEP {number}: {instruction}\n")


print_section("STEP 1 REVIEW GUIDE: UNDERSTANDING OVERLAPPING-LABEL LEAKAGE", level=1)

print("""
This guide walks you through:
  1. Understanding the core problem (5 min read)
  2. Running the main experiment (2 min execution)
  3. Interpreting the results (3 min analysis)
  4. Understanding what each test does (reference)

""")

print_section("PART 1: UNDERSTANDING THE CORE PROBLEM", level=2)

print_concept(
    "What is 'overlapping-label leakage'?",
    """
    When you predict a stock price 20 days ahead:
    - Observation at t=0: uses data from Jan 1–20, predicts Feb 20
    - Observation at t=1: uses data from Jan 2–21, predicts Feb 21
    
    The labels (outcomes) overlap: both cover mid-to-late February.
    
    If your train/test split randomly puts t=0 and t=1 in different folds,
    then TRAINING LABELS contain information about the same outcome window as TEST.
    
    A model doesn't need to be predictive; it can exploit this overlap.
    """
)

print_concept(
    "Why is this different from 'autocorrelation'?",
    """
    Autocorrelation: consecutive values are correlated (Jan 20 ≈ Jan 21).
    This is legitimate in time series; you can handle it with chronological CV.
    
    Label-interval overlap: the TARGETS (what you're predicting) overlap in time.
    No amount of chronological ordering fixes this if training and test 
    both cover the same future interval.
    
    Example: training data from Feb 1 with label for end-of-February,
    test data from Feb 15 with label also for end-of-February.
    Training label already contains information about the test outcome!
    """
)

print_concept(
    "What does the synthetic experiment do?",
    """
    Builds data where:
      - Feature x: monotone increasing (cumsum), completely random
      - Target y: mean of random noise
    
    x and y are 100% independent by construction.
    True predictive power: ZERO.
    
    But y is computed from overlapping 20-day windows.
    So y values are highly autocorrelated.
    
    Hypothesis: shuffled CV will report R² ≈ 0.8–0.9 (pure leakage).
    Purged CV will report R² ≈ 0 (correct: no skill).
    """
)

print_section("PART 2: THE FOUR KEY CLAIMS", level=2)

print("""
The experiment makes FOUR FALSIFIABLE CLAIMS:

  CLAIM 1: Feature and target are independent by construction.
    → TEST A verifies this via correlation ≈ 0
    
  CLAIM 2: Shuffled CV allows label intervals to overlap.
    → TEST B measures contamination (should be ~98%)
    
  CLAIM 3: PurgedKFold removes all overlapping observations.
    → TEST C verifies zero contamination (strict invariant)
    → THIS IS THE CRITICAL TEST
    
  CLAIM 4: The temporal proxy feature is necessary for leakage.
    → CONTROL EXPERIMENT shuffles the feature
    → R² should collapse to baseline
    
If any of these fails, the experiment has a bug.
""")

print_section("PART 3: EXPECTED RESULTS", level=2)

expected = """
Model                          Naive R²    Purged R²    Naive Contam   Purged Contam
─────────────────────────────────────────────────────────────────────────────────────
predict-the-mean (baseline)     -0.012      -0.130         ~98%           0%
k-NN (k=10)                      0.827      -1.477         ~98%           0%
RandomForest (n=120)             0.910      -1.868         ~98%           0%

CONTROL: RandomForest with shuffled x:  R² ≈ -0.01 (temporal proxy destroyed)
"""
print(expected)

print("""
KEY OBSERVATION:
  - Naive R² are HUGE (0.82, 0.91) on data where true R² should be 0
  - Purged R² drop DRAMATICALLY to negative (near baseline)
  - Purged contamination is EXACTLY 0% (strict, not approximate)
  - Control R² collapses, proving the mechanism
  
This is not a small effect. This is a complete reversal.
""")

print_section("PART 4: HOW TO RUN AND INTERPRET", level=2)

print_step(1, """Install dependencies:
  
  pip install purgedcv scikit-learn numpy pandas pytest matplotlib
  
  Takes ~1 minute. purgedcv is ~2 MB.
""")

print_step(2, """Navigate to the experiment directory:
  
  cd research/step1_leakage_validation/
  
  This puts you in the directory with all the code.
""")

print_step(3, """Run the main experiment:
  
  python synthetic_leakage_proof.py
  
  Expected: ~30 seconds of output describing:
  
    ✓ Building synthetic data
    ✓ Verifying label window overlap (should be H-1 = 19 days)
    ✓ Running naive vs purged CV comparison
    ✓ Running control experiment (shuffled features)
    ✓ Printing results table
    ✓ TEST A–D results and interpretation
    ✓ Mechanical invariant checks
    
  Look for this line near the end:
    ✓ TEST C PASS: Purged contamination = 0.0 (strict zero)
    
  If this fails, STOP and investigate.
""")

print_step(4, """Run the automated tests:
  
  pytest test_leakage_invariants.py -v
  
  Expected: ~5 tests, all PASS
  
    test_synthetic_data_independence PASSED
    test_naive_kfold_allows_overlap PASSED
    test_purged_kfold_zero_overlap PASSED      ← THE CRITICAL ONE
    test_consecutive_label_windows_overlap PASSED
    test_label_intervals_are_half_open PASSED
    
  These validate the claims automatically.
  If test_purged_kfold_zero_overlap fails, STOP and investigate.
""")

print_section("PART 5: WHAT EACH TEST TELLS YOU", level=2)

tests = [
    ("TEST A: Independence by Construction",
     "Feature and target are generated independently.",
     "Empirical correlation ≈ -0.22 (weak, as expected)",
     "This is a sanity check. The correlation is noise, not signal."),
    
    ("TEST B: Naive CV Allows Overlap",
     "Shuffled KFold allows training and test label intervals to overlap.",
     "~98% of training rows have overlapping label intervals",
     "This is WHY the model fabricates skill. It can memorize near-neighbours."),
    
    ("TEST C: Purged CV Achieves Zero Overlap (CRITICAL)",
     "PurgedKFold removes all overlapping training observations.",
     "Contamination = 0% (exact equality, not approximate)",
     "This is the HARD INVARIANT. If this fails, PurgedKFold doesn't work."),
    
    ("TEST D: Label Windows Overlap by Design",
     "Consecutive labels share H-1 of H days by construction.",
     "Consecutive labels overlap by exactly 19 days (H-1, where H=20)",
     "This validates the synthetic data structure itself."),
    
    ("CONTROL: Temporal Proxy Necessary",
     "Leakage requires the feature to act as a clock.",
     "RandomForest R² with shuffled x = -0.01 (collapses to baseline)",
     "Proves: overlapping labels alone are insufficient. Need feature as proxy."),
]

for title, claim, result, interpretation in tests:
    print(f"📌 {title}")
    print(f"   Claim:       {claim}")
    print(f"   Result:      {result}")
    print(f"   Meaning:     {interpretation}")
    print()

print_section("PART 6: INTERPRETING YOUR OWN RESULTS", level=2)

print("""
SCENARIO 1: Your R² values differ slightly from the report
  → This is normal. Exact values depend on random forest implementation, numpy version.
  → Check: Is naive R² still >> purged R²?
  → If yes: PASS. The qualitative effect is reproduced.
  → If no: INVESTIGATE.

SCENARIO 2: Naive contamination is not ~98%
  → Might be 85% or 92% depending on fold size.
  → Check: Is it still >> 50%?
  → If yes: PASS.
  → If no: INVESTIGATE.

SCENARIO 3: Purged contamination ≠ 0%
  → This is a CRITICAL FAILURE.
  → Options:
     (a) PurgedKFold library changed
     (b) Bug in assert_no_label_overlap()
     (c) Bug in interval logic
  → Do not proceed until resolved.

SCENARIO 4: Control R² is not near baseline
  → Might be 0.05 or -0.03 instead of -0.01.
  → As long as it's much lower than naive R², you're fine.
  → The mechanism is: temporal proxy destroyed → R² collapses.

SCENARIO 5: TEST C fails
  → assert purged_contamination == 0.0 raised an error
  → This means purged CV is not achieving zero overlap
  → CRITICAL. This is the foundation of the project.
  → Do not proceed.
""")

print_section("PART 7: THE DEEPER MEANING", level=2)

meaning = """
What this experiment PROVES:

  1. You can have completely independent features and targets
  2. But shuffled CV reports 82–91% accuracy
  3. Because labels overlap in time and the feature acts as a clock
  4. Remove the overlap, and the accuracy collapses to 0%

This is NOT about:
  ✗ Time series being correlated (that's autocorrelation, handled separately)
  ✗ The model overfitting (the model is just k-NN or RandomForest)
  ✗ Noisy data (the data is perfectly clean)

This IS about:
  ✓ Training and test labels covering the same future interval
  ✓ A model exploiting this overlap via any temporal proxy
  ✓ The fundamental invalidity of shuffled CV for this problem

Why does this matter for your project?
  
  If you build a stock prediction model on real financial data:
    - You train on 2018–2020
    - You test on 2020–2022
    - Your backtest R² is 0.85
    - You deploy with confidence
    - It loses money
    
  Why? Because your labels overlapped and you didn't know.
  
  This experiment proves that can happen even with ZERO true skill.
  
  PurgedKFold prevents it by ensuring training labels don't overlap test labels.
"""
print(meaning)

print_section("PART 8: QUICK REFERENCE", level=2)

reference = """
HOW TO READ RESULTS:

  ✓ R² = coefficient of determination
    - R² = 1.0: perfect prediction
    - R² = 0.0: no better than guessing mean
    - R² = -1.0: worse than guessing mean (model is actively wrong)
    
  ✓ Contamination = fraction of training rows with overlapping label intervals
    - 100% = all training rows have overlapping labels (complete leakage)
    - 0% = no training rows have overlapping labels (no leakage)
    
  ✓ Naive = shuffled cross-validation (the WRONG way for this problem)
    - High R² due to label overlap, not true skill
    
  ✓ Purged = PurgedKFold (the RIGHT way)
    - Removes overlapping training rows
    - R² should return to baseline (zero skill)

CRITICAL VALUES:

  Pure leakage (this experiment):
    - Naive R²: 0.82–0.91 (fabricated)
    - Purged R²: -0.03 to -1.87 (correct: baseline or worse)
    - Naive contamination: ~98% (almost complete overlap)
    - Purged contamination: 0% (STRICT, must be exact zero)
    - Control R²: -0.01 (temporal proxy destroyed)

IF ANY OF THESE CHANGE SIGNIFICANTLY:
  1. Check purgedcv version (must be >= 0.2)
  2. Check numpy/sklearn versions
  3. Check for bugs in overlap detection
  4. Check for bugs in synthetic data construction
"""
print(reference)

print_section("PART 9: NEXT STEPS", level=2)

print("""
After reviewing this guide and running the experiments:

  1. Run: python synthetic_leakage_proof.py
     - Read the output carefully
     - Note the R² values (will match report within ~5%)
     - Verify purged contamination = 0%
     
  2. Run: pytest test_leakage_invariants.py -v
     - All tests should PASS
     - Especially test_purged_kfold_zero_overlap
     
  3. Read: STEP_1_REPORT.md (sections 1–5)
     - Now that you've seen the results, re-read the conceptual foundation
     - It should make much more sense now
     
  4. Document: Any surprising results
     - Note the exact R² values you observe
     - Compare to the report (should be similar)
     - Document any deviations and why
     
  5. DO NOT PROCEED TO STEP 2 until:
     - You understand what overlapping-label leakage is
     - You can explain why purged contamination must be 0%
     - You can articulate why shuffled CV fails for this problem
     - You understand why PurgedKFold works

That's it. Step 1 is validation only. No production code yet.
""")

print_section("PART 10: COMMON QUESTIONS", level=2)

faq = [
    ("Q: Do I need to know about financial markets?",
     "A: No. This is pure machine learning. The only finance concept is 'forward-looking label'."),
    
    ("Q: What if my results don't exactly match the report?",
     "A: Expected. Different numpy/sklearn versions, random seeds can vary results by 5–10%."),
    
    ("Q: What if purged contamination is 0.1% instead of 0.0%?",
     "A: STOP. This is not acceptable. 0% must be exact (it's a strict invariant)."),
    
    ("Q: Why H=20 and not H=10 or H=100?",
     "A: Arbitrary choice. Larger H = more overlap. Smaller H = less overlap. 20 is middle-ground."),
    
    ("Q: Why cumsum(Gamma) and not other features?",
     "A: Gamma is arbitrary. Any monotone feature works. cumsum ensures strong temporal signal."),
    
    ("Q: Why is the control experiment important?",
     "A: It proves the mechanism. Without it, you might think overlapping labels alone cause leakage."),
    
    ("Q: Can I modify the experiment?",
     "A: Yes, but verify the zero-overlap invariant still holds after any change."),
    
    ("Q: How long should the experiments take?",
     "A: ~30 sec for main experiment, ~10 sec for tests. Total: <1 minute."),
]

for q, a in faq:
    print(f"  {q}")
    print(f"  {a}\n")

print_section("READY TO RUN?", level=1)

print("""
You now have everything you need to:
  1. Understand overlapping-label leakage conceptually
  2. Run the synthetic experiment
  3. See the results firsthand
  4. Verify the zero-overlap invariant
  5. Understand why PurgedKFold works

Commands to run:

  pip install purgedcv scikit-learn numpy pandas pytest
  cd research/step1_leakage_validation/
  python synthetic_leakage_proof.py
  pytest test_leakage_invariants.py -v

Expected time: ~10 minutes total.

After that, revisit STEP_1_REPORT.md and cross-reference the output
with the expected results table.

Good luck! 🚀
""")
