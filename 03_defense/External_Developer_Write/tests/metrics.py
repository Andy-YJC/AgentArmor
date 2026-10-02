"""Shared classification and latency metrics for Type 4 evaluations."""

from typing import Dict, List, Tuple

from External_Developer_Write.utils import mean


def compute_latency_stats(timings_ms: List[float]) -> Dict[str, float]:
    """计算延迟百分位统计。

    Args:
        timings_ms: 毫秒单位的延迟列表
    Returns:
        {"mean": ..., "p50": ..., "p95": ..., "p99": ...}
    """
    if not timings_ms:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    sorted_t = sorted(timings_ms)
    n = len(sorted_t)
    def percentile(p):
        idx = max(0, min(n - 1, int(n * p)))
        return sorted_t[idx]
    return {
        "mean": round(mean(timings_ms), 4),
        "p50": round(percentile(0.50), 4),
        "p95": round(percentile(0.95), 4),
        "p99": round(percentile(0.99), 4),
    }


def compute_metrics(
    results: List[Tuple[str, bool, float]],
    attack_labels: Dict[str, bool],
) -> Dict[str, float]:
    """计算 Precision、Recall、F1、FPR。

    Args:
        results: [(doc_id, is_anomaly_pred, anomaly_score), ...]
        attack_labels: {doc_id: is_attack_ground_truth}
    Returns:
        {"precision": ..., "recall": ..., "f1": ..., "fpr": ...}
    """
    tp = fp = tn = fn = 0
    for doc_id, pred, _ in results:
        gt = attack_labels.get(doc_id, False)
        if pred and gt:
            tp += 1
        elif pred and not gt:
            fp += 1
        elif not pred and not gt:
            tn += 1
        elif not pred and gt:
            fn += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
