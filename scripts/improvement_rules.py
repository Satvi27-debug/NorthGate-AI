"""What counts as a legitimate improvement, and what does not.

The PRD grades methodology integrity above everything (Section 14.2: "Integrity
of methodology is the top-weighted criterion"). So the first job of any attempt
to make the models perform better is to write down, in advance, which moves are
allowed and which would be cheating. This file is that list, and
`src/features.py` plus `tests/test_causality.py` enforce the mechanical half.

ALLOWED - genuine forecasting skill
-----------------------------------
* Adding features that are computable at or before time t (Section 5). The
  Section 5.1 table says "representative features", so it is a guide, not a
  whitelist.
* Tuning hyperparameters using walk-forward CV inside the training partition
  (Section 8.2).
* Regularisation strength, model family, calibration.
* Ensembling several models averaged with equal or validated weights.
* Formulating the prediction problem differently - for example predicting a
  stock's return *relative to its peers* rather than its raw return - PROVIDED
  the task change is disclosed, given its own baseline, and is not presented as
  if it were the Section 8.3 test.
* Reporting more metrics, more diagnostics, more honest failure analysis.

NOT ALLOWED - would invalidate the whole project
------------------------------------------------
* Choosing features, hyperparameters, models or seeds by looking at test
  performance. The test set is touched exactly once, at the end.
* Computing any feature from a value that is not known at time t (Section 5:
  "a feature that peeks at the future produces excellent validation scores and
  worthless live behaviour").
* Normalising, winsorising or imputing using whole-sample statistics that
  include the future.
* Refitting the scaler on validation or test (Section 5.4).
* Changing the target definition to something easier without disclosing it
  (Section 3.4 requires the target to be "documented and fixed before
  modelling").
* Reporting a favourable metric while the headline metric is unfavourable.
* Selecting the best of many runs and presenting it as typical.
* Overlapping estimation and evaluation windows in the portfolio layer - the
  mistake that previously inflated the Sharpe from 1.357 to 2.998.
* Shipping a derived artifact that was fitted on a superseded input. A stale
  metrics file is not a neutral leftover: downstream stages consume it and will
  confidently report numbers from a run that no longer exists on disk.

ADDED AFTER THE FIRST TWO LEAKS WERE FOUND
------------------------------------------
Two leaks were made during this build. Both are now mechanised as tests, and
the pattern generalises further than either instance:

1. **A derived-from-the-label column is still the label.** Relative_Target and
   Rank_Target were created inside the cross-sectional module, added as
   ordinary columns, and then swept into the feature set by the generic
   numeric-column selector. The model read its own answer and reported a 95%
   error reduction. A blocklist naming Target cannot help here, because the
   leak had a different name.
   Guard: tests/test_causality.py::test_no_feature_is_a_repackaged_label fails
   the build if any input column correlates above 0.99 with any return-valued
   label.

2. **A negative control is necessary but not sufficient.** The shuffled-label
   check was run, it passed, and it still missed leak (1). It proved the model
   relied on the feature set; it could not prove that one member of the feature
   set was the answer. Structural assertions are needed next to behavioural
   ones.

3. **An untested function is a blind spot, not a clean bill of health.** The
   causality suite exercised build_ticker_features only. A look-ahead lived in
   build_cross_sectional_context the entire time: Beta was computed as a single
   scalar from the trailing window and assigned to the whole column, so a number
   fitted at the end of the sample was broadcast backwards over every earlier
   date. The suite could not have caught it, because it never called the
   function. Guard: TestCrossSectionalCausality now runs both the truncation
   and the future-corruption probes against the cross-sectional context.

4. **A result that is too good is evidence of a bug, not of skill.** A 19x error
   reduction should have stopped the run and started an investigation. It
   eventually did, but only after being admired. The suspicion is written down
   in advance now so it is not skipped next time.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPORTS_DIR, write_json  # noqa: E402

ALLOWED = [
    "Add causal features from the Section 5.1 families (the table lists "
    "representative features, not an exhaustive whitelist).",
    "Tune hyperparameters by walk-forward CV inside the training partition only.",
    "Tune regularisation, model family, and calibration.",
    "Ensemble validated models with disclosed weighting.",
    "Reframe the prediction target, provided the change is disclosed, given its "
    "own like-for-like baseline, and NOT counted as the Section 8.3 bar.",
    "Report more diagnostics and more honest failure analysis.",
]

FORBIDDEN = [
    "Any feature, hyperparameter, model or seed chosen by looking at test results.",
    "Any feature using a value not known at or before time t.",
    "Whole-sample normalisation, winsorising or imputation that includes the future.",
    "Fitting the scaler on validation or test data.",
    "Silently changing the target definition to something easier.",
    "Headlining a favourable metric while the PRD metric is unfavourable.",
    "Presenting the best of many runs as typical.",
    "Overlapping estimation and evaluation windows.",
    "Any input column that is a re-expression of the target under a different "
    "name (a blocklist naming Target alone does not catch these).",
    "Any feature computed in a function the causality suite does not exercise.",
    "Reporting a derived artifact fitted on a superseded feature table.",
]

PROCEDURE = [
    "1. Write the baseline down before changing anything.",
    "2. Make changes using the training and validation partitions only.",
    "3. Re-measure on validation to confirm the change helps there too.",
    "4. Touch the test set once, at the end, whatever the result.",
    "5. Report the outcome honestly, including 'still does not beat the "
    "baseline' if that is what happens.",
    "6. If a result is unexpectedly strong, treat it as a suspected bug until "
    "proven otherwise, and add a test that would have caught it.",
    "7. Confirm every derived artifact is newer than the table it was fitted on.",
]

# Both of these were found by hand during this build. Each is now a test, and
# the pairing matters: the behavioural control passed while the structural
# defect was present, so neither class of check is treated as sufficient alone.
LESSONS = [
    {
        "what_happened": (
            "The cross-sectional module created Relative_Target and "
            "Rank_Target as helper columns derived entirely from the label. "
            "The generic numeric-column feature selector admitted them as model "
            "inputs, so the model read its own answer and reported a 95% error "
            "reduction against the no-skill baseline."
        ),
        "how_it_was_caught": (
            "The improvement was implausibly large, which prompted a "
            "correlation sweep rather than acceptance. The shuffled-label "
            "negative control had already passed and had not detected it."
        ),
        "guard_now": (
            "tests/test_causality.py::test_no_feature_is_a_repackaged_label - "
            "any input above |r|=0.99 with a return-valued label fails the build."
        ),
        "general_lesson": (
            "A negative control proves the model uses its inputs. It cannot "
            "prove the inputs are legitimate. Structural invariants are needed "
            "alongside behavioural ones."
        ),
    },
    {
        "what_happened": (
            "build_cross_sectional_context computed Beta as a single scalar "
            "from a trailing window and assigned that scalar to the entire "
            "column, broadcasting a value fitted at the end of the sample "
            "backwards across every earlier date."
        ),
        "how_it_was_caught": (
            "A pre-existing invariant test (no constant feature columns) failed. "
            "The causality suite could not catch it because it only exercised "
            "build_ticker_features and never called this function."
        ),
        "guard_now": (
            "TestCrossSectionalCausality runs truncation and future-corruption "
            "probes over the cross-sectional context, plus a regression test "
            "asserting Beta varies over time."
        ),
        "general_lesson": (
            "Coverage gaps are not neutral. A function with no causality test is "
            "an untested function, and an untested function gets whatever "
            "behaviour its author happened to write."
        ),
    },
    {
        "what_happened": (
            "After the feature table changed, saved .keras models were removed "
            "as stale but dl_metrics.csv retained the previous run's numbers, "
            "and the portfolio and recommendation stages consumed them."
        ),
        "how_it_was_caught": (
            "Noticed by checking artifact timestamps against the feature table "
            "after the fact."
        ),
        "guard_now": (
            "tests/test_staleness.py asserts every derived artifact is newer "
            "than features.parquet, that the saved scaler and model agree with "
            "the current feature count, and that recorded n_features values "
            "match the table on disk."
        ),
        "general_lesson": (
            "A stale artifact is not an inert leftover. Downstream stages read "
            "it and report its numbers with total confidence."
        ),
    },
]


def main() -> int:
    doc = {
        "generated_by": "scripts/improvement_rules.py",
        "purpose": ("Written before attempting any model improvement, so the "
                    "boundary between forecasting skill and cheating is fixed "
                    "in advance rather than negotiated afterwards."),
        "prd_anchor": ("Section 14.2: 'A modest, correct, leakage-free result "
                       "with honest analysis scores higher than an "
                       "impressive-looking result built on a subtle look-ahead "
                       "bug. Integrity of methodology is the top-weighted "
                       "criterion.'"),
        "allowed": ALLOWED,
        "forbidden": FORBIDDEN,
        "procedure": PROCEDURE,
        "lessons_from_actual_leaks": LESSONS,
        "note_on_lessons": (
            "These are not hypotheticals. Each entry describes a real defect "
            "made while trying to improve the models, and each names the test "
            "that now prevents its recurrence. They are recorded because the "
            "general patterns - that a renamed label is still a label, that a "
            "negative control is not sufficient, that an untested function is a "
            "blind spot, and that a suspiciously good result is evidence of a "
            "bug - are the transferable part."
        ),
    }
    write_json(doc, REPORTS_DIR / "improvement_rules.json")
    print("Improvement rules written to reports/improvement_rules.json\n")
    print("ALLOWED:")
    for a in ALLOWED:
        print(f"  + {a}")
    print("\nNOT ALLOWED:")
    for f in FORBIDDEN:
        print(f"  x {f}")
    print("\nPROCEDURE:")
    for p in PROCEDURE:
        print(f"  {p}")
    print("\nLEARNED THE HARD WAY (each one is now a test):")
    for lesson in LESSONS:
        print(f"\n  * {lesson['how_it_was_caught']}")
        print(f"      guard: {lesson['guard_now']}")
        print(f"      lesson: {lesson['general_lesson']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
