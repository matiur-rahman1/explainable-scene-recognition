"""Statistical significance testing (McNemar's test) and multi-seed aggregation."""

import math
from typing import Any, Dict, List, Sequence

import numpy as np


def mcnemar_test(
    y_true: Sequence[int],
    y_pred_a: Sequence[int],
    y_pred_b: Sequence[int],
) -> Dict[str, Any]:
    """Compute paired McNemar's test between two classifiers on the same test set.

    Returns contingency counts (n00, n01, n10, n11), continuity-corrected chi2
    statistic, and exact two-sided binomial p-value on discordant pairs.
    """
    yt = np.asarray(y_true)
    pa = np.asarray(y_pred_a) == yt
    pb = np.asarray(y_pred_b) == yt

    n11 = int(np.sum(pa & pb))
    n10 = int(np.sum(pa & (~pb)))  # A correct, B wrong
    n01 = int(np.sum((~pa) & pb))  # A wrong, B correct
    n00 = int(np.sum((~pa) & (~pb)))

    discordant = n10 + n01
    if discordant == 0:
        chi2_cc = 0.0
        p_chi2 = 1.0
        p_binom = 1.0
    else:
        diff = abs(n10 - n01)
        chi2_cc = ((max(0.0, diff - 1.0)) ** 2) / discordant
        # Survival function of chi2(df=1) is erfc(sqrt(x / 2))
        p_chi2 = float(math.erfc(math.sqrt(chi2_cc / 2.0)))

        # Exact two-sided binomial p-value under H0: p = 0.5
        k = min(n10, n01)
        try:
            from scipy.stats import binomtest

            p_binom = float(binomtest(k, n=discordant, p=0.5).pvalue)
        except Exception:
            p_binom = p_chi2

    return {
        "n11_both_correct": n11,
        "n10_a_only_correct": n10,
        "n01_b_only_correct": n01,
        "n00_both_wrong": n00,
        "discordant_pairs": discordant,
        "chi2_cc": round(float(chi2_cc), 4),
        "p_value_chi2": round(float(p_chi2), 6),
        "p_value_exact": round(float(p_binom), 6),
        "significant_005": bool(p_binom < 0.05),
    }


def compute_pairwise_mcnemar(
    y_true: Sequence[int],
    predictions_by_model: Dict[str, Sequence[int]],
) -> Dict[str, Dict[str, Any]]:
    """Run McNemar's test across all unique pairs of models."""
    models = list(predictions_by_model.keys())
    results: Dict[str, Dict[str, Any]] = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m_a, m_b = models[i], models[j]
            key = f"{m_a}_vs_{m_b}"
            results[key] = mcnemar_test(
                y_true, predictions_by_model[m_a], predictions_by_model[m_b]
            )
    return results


def aggregate_seed_metrics(
    seed_runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate scalar, XAI, and adversarial metrics across multiple random seeds."""
    if not seed_runs:
        return {}

    def _mean_std(vals: Sequence[float]) -> Dict[str, float]:
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr, ddof=0)), 4),
        }

    agg: Dict[str, Any] = {
        "num_seeds": len(seed_runs),
        "test_accuracy": _mean_std([r["test_accuracy"] for r in seed_runs]),
        "macro_f1": _mean_std([r["macro_f1"] for r in seed_runs]),
        "weighted_f1": _mean_std([r["weighted_f1"] for r in seed_runs]),
        "stats": seed_runs[0].get("stats", {}),
    }

    # Aggregate XAI faithfulness if present
    if "xai_faithfulness" in seed_runs[0] and seed_runs[0]["xai_faithfulness"]:
        xai_agg: Dict[str, Any] = {}
        for method in seed_runs[0]["xai_faithfulness"].keys():
            ins_vals = [
                r["xai_faithfulness"][method]["insertion_auc"]
                for r in seed_runs
                if method in r.get("xai_faithfulness", {})
            ]
            del_vals = [
                r["xai_faithfulness"][method]["deletion_auc"]
                for r in seed_runs
                if method in r.get("xai_faithfulness", {})
            ]
            xai_agg[method] = {
                "insertion_auc": _mean_std(ins_vals),
                "deletion_auc": _mean_std(del_vals),
            }
        agg["xai_faithfulness"] = xai_agg

    # Aggregate adversarial curves
    for atk_key in ("fgsm_accuracy", "fgsm_pred_accuracy", "pgd_accuracy"):
        if atk_key in seed_runs[0] and seed_runs[0][atk_key]:
            atk_agg: Dict[str, Any] = {}
            for eps_str in seed_runs[0][atk_key].keys():
                vals = [r[atk_key][eps_str] for r in seed_runs if eps_str in r.get(atk_key, {})]
                atk_agg[str(eps_str)] = _mean_std(vals)
            agg[atk_key] = atk_agg

    # Aggregate per-class accuracy if present
    if "per_class_accuracy" in seed_runs[0]:
        pc_agg: Dict[str, Any] = {}
        for cls_name in seed_runs[0]["per_class_accuracy"].keys():
            vals = [
                r["per_class_accuracy"][cls_name]
                for r in seed_runs
                if cls_name in r.get("per_class_accuracy", {})
            ]
            pc_agg[cls_name] = _mean_std(vals)
        agg["per_class_accuracy"] = pc_agg

    return agg
