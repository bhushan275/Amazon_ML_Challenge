"""
Stage 4: Evaluation & Metric Tuning Module
Computes Macro F_0.5 Score per Amazon ML Challenge specification:
F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Singletons (no true matches) score 1.0 for correctly predicted empty list,
and 0.0 for any predicted match.
Tunes acceptance threshold to maximize overall Macro F_0.5 score.
"""

import numpy as np


def compute_entity_f05(pred_matches: list, true_matches: list) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity.
    """
    set_pred = set(pred_matches)
    set_true = set(true_matches)

    # Singleton case: No true matches
    if not set_true:
        return 1.0 if not set_pred else 0.0

    # Non-singleton case with no predictions
    if not set_pred:
        return 0.0

    tp = len(set_pred & set_true)
    precision = tp / len(set_pred)
    recall = tp / len(set_true)

    if precision + recall == 0:
        return 0.0

    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return f05


def compute_macro_f05(predictions: dict, ground_truth: dict) -> float:
    """
    Computes Macro-averaged F_0.5 score across all S1 entities in ground truth.
    """
    scores = []
    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, [])
        score = compute_entity_f05(pred_matches, true_matches)
        scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


def tune_threshold(pair_scores: dict, ground_truth: dict, thresholds: list = None) -> tuple:
    """
    Finds optimal decision threshold t* that maximizes Macro F_0.5 on ground truth.
    pair_scores: dict mapping s1_id -> list of (cand_id, probability_score)
    """
    if thresholds is None:
        thresholds = list(np.linspace(0.1, 0.9, 81))

    best_thresh = 0.5
    best_score = -1.0

    for t in thresholds:
        predictions = {}
        for s1_id, cand_pairs in pair_scores.items():
            preds = [cand_id for cand_id, score in cand_pairs if score >= t]
            predictions[s1_id] = preds

        macro_f05 = compute_macro_f05(predictions, ground_truth)
        if macro_f05 > best_score:
            best_score = macro_f05
            best_thresh = t

    return best_thresh, best_score
