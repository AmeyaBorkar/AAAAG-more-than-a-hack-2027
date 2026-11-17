"""Public-window smoke test: is the pipeline CORRECT? Never: which version is better?

RULES.md section 2: the published five minutes are for checking that frames are aligned, the file
is well-formed and the labels mean what we think -- not training data and not a tuning or
selection set. EVALUATION.md shows why: the smoothing window that wins on these 17 possessions
loses on the full match. So this module returns PASS/FAIL against loose correctness floors that
any sane pipeline clears, and the numbers it prints must never be used to compare two versions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import A, D, H
from ..official import PUBLIC_GT, pm

# Correctness floors, not targets. A constant guess scores macroF1 ~0.16-0.33; misaligned frames
# or swapped teams fall well below these; every official baseline clears them comfortably.
MACRO_F1_FLOOR = 0.35
TEAM_ACCURACY_FLOOR = 0.60  # H/A agreement where both say "alive": < 0.5 means teams are swapped

BANNER = (
    "SMOKE TEST ONLY -- PASS/FAIL checks correctness. Do not compare versions on these numbers "
    "(RULES.md section 2, docs/VALIDATION.md)."
)


@dataclass
class SmokeReport:
    passed: bool
    checks: dict[str, bool]
    numbers: dict[str, float]

    def lines(self) -> list[str]:
        out = [BANNER, f"result: {'PASS' if self.passed else 'FAIL'}"]
        out += [f"  [{'ok' if ok else 'FAIL'}] {name}" for name, ok in self.checks.items()]
        out.append("  " + "  ".join(f"{k}={v:.3f}" for k, v in self.numbers.items()))
        return out


def public_gt() -> np.ndarray:
    return pm.label_codes(pm.read_label_csv(PUBLIC_GT))


def public_smoke(labels: np.ndarray) -> SmokeReport:
    gt = public_gt()
    result = pm.evaluate(gt, labels.astype(np.int8))
    window = gt != pm.X
    predicted = set(np.unique(labels[window]).tolist())
    checks = {
        "row count matches the video": len(labels) == len(gt),
        f"macroF1 >= {MACRO_F1_FLOOR} (frames aligned, not garbage)": result["macro_f1"]
        >= MACRO_F1_FLOOR,
        f"team agreement >= {TEAM_ACCURACY_FLOOR} (H/A not swapped)": result["team_acc_both_alive"]
        >= TEAM_ACCURACY_FLOOR,
        "predicts H, A and D in the window": {H, A, D} <= predicted,
    }
    numbers = {
        "macroF1": result["macro_f1"],
        "spellF1": result["spell_f1_mean"],
        "team_acc": result["team_acc_both_alive"],
    }
    return SmokeReport(all(checks.values()), checks, numbers)
