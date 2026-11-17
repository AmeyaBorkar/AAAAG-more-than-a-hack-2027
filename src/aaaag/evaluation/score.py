"""Thin wrapper over the organisers' possession_metrics.evaluate. Never re-implement a metric."""

from __future__ import annotations

import numpy as np

from ..official import pm

HEADLINE = (
    "spell_f1_mean",
    "macro_f1",
    "spell_f1_at_headline",
    "n_pred_spells",
    "n_gt_spells",
    "mean_spell_duration_s",
    "gt_mean_spell_duration_s",
    "alive_dead_f1",
    "team_acc_both_alive",
    "fragmentation",
)


def evaluate(gt: np.ndarray, pred: np.ndarray, *, bootstrap_n: int = 0) -> dict:
    """Every official metric for one prediction against one ground truth (both int8 codes)."""
    return pm.evaluate(gt.astype(np.int8), pred.astype(np.int8), bootstrap_n=bootstrap_n)


def summarize(result: dict) -> dict[str, float]:
    """The numbers worth keeping per match: headline metrics, per-IoU spellF1, transition timing."""
    out = {key: float(result[key]) for key in HEADLINE}
    loose = result["transitions"][max(result["transitions"])]
    out["trans_f1_loose"] = float(loose["f1"])
    if loose["mean_abs_latency"] is not None:
        out["trans_mean_abs_latency"] = float(loose["mean_abs_latency"])
    for iou, scores in result["spell_f1"].items():
        out[f"spell_f1@{iou:.1f}"] = float(scores["f1"])
    if result.get("spell_bootstrap_low") is not None:
        out["spell_bootstrap_low"] = float(result["spell_bootstrap_low"])
        out["spell_bootstrap_high"] = float(result["spell_bootstrap_high"])
    return out
