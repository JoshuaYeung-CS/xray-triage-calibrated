"""Calibration, selective prediction and evaluation metrics, in plain NumPy.

Everything here works on raw model logits for the positive class (pneumonia) and binary labels,
so the same code runs in the training job, the evaluation script, the tests and the app.
"""
from __future__ import annotations

import numpy as np


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def nll(logits, labels, temperature: float = 1.0) -> float:
    """Mean binary cross-entropy of sigmoid(logits / temperature)."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    y = np.asarray(labels, dtype=np.float64)
    # log(1 + exp(-|z|)) + max(z, 0) - y z  is the stable form of BCE with logits.
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def fit_temperature(logits, labels, lo: float = 0.05, hi: float = 20.0) -> float:
    """Temperature that minimises validation NLL (golden-section search on log T).

    NLL is convex in 1/T for a fixed set of logits, so a 1-D search finds the optimum.
    """
    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(logits, labels, np.exp(c)), nll(logits, labels, np.exp(d))
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(logits, labels, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(logits, labels, np.exp(d))
    return float(np.exp((a + b) / 2.0))


def brier(probs, labels) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def reliability_bins(probs, labels, n_bins: int = 10):
    """Equal-width bins on p(pneumonia). Returns per-bin count, mean confidence and positive rate."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins).astype(np.int64)
    sum_p = np.bincount(idx, weights=p, minlength=n_bins)
    sum_y = np.bincount(idx, weights=y, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_p = np.where(count > 0, sum_p / np.maximum(count, 1), np.nan)
        rate = np.where(count > 0, sum_y / np.maximum(count, 1), np.nan)
    return {"edges": edges, "count": count, "mean_prob": mean_p, "pos_rate": rate}


def ece(probs, labels, n_bins: int = 10) -> float:
    """Expected calibration error: count-weighted |mean p - observed rate| over equal-width bins."""
    b = reliability_bins(probs, labels, n_bins)
    m = b["count"] > 0
    w = b["count"][m] / b["count"].sum()
    return float(np.sum(w * np.abs(b["mean_prob"][m] - b["pos_rate"][m])))


def auroc(scores, labels) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties get half credit)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def confusion(probs, labels, threshold: float = 0.5):
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    pred = p >= threshold
    tp = int(np.sum(pred & y))
    tn = int(np.sum(~pred & ~y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def classification_report(probs, labels, threshold: float = 0.5):
    c = confusion(probs, labels, threshold)
    n = c["tp"] + c["tn"] + c["fp"] + c["fn"]
    sens = c["tp"] / max(c["tp"] + c["fn"], 1)
    spec = c["tn"] / max(c["tn"] + c["fp"], 1)
    prec = c["tp"] / max(c["tp"] + c["fp"], 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-12)
    return {
        "accuracy": (c["tp"] + c["tn"]) / max(n, 1),
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        **c,
    }


def threshold_for_specificity(probs, labels, target_spec: float) -> float:
    """Smallest threshold whose specificity on these labels is at least target_spec."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    neg = np.sort(p[~y])
    if len(neg) == 0:
        return 0.5
    k = int(np.ceil(target_spec * len(neg)))  # this many negatives must fall below the threshold
    k = min(max(k, 1), len(neg))
    t = neg[k - 1]
    # Predict positive when p >= threshold, so nudge just above the k-th negative.
    return float(np.nextafter(t, np.inf))


def triage_band(prob: float, lo: float, hi: float) -> str:
    """Three-way decision on a calibrated probability."""
    if prob >= hi:
        return "pneumonia"
    if prob <= lo:
        return "normal"
    return "unsure"


def selective_report(probs, labels, lo: float, hi: float):
    """Coverage and accuracy when the model only answers outside the (lo, hi) band."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    answered = (p >= hi) | (p <= lo)
    pred = p >= hi
    n_ans = int(answered.sum())
    correct = int(np.sum(answered & (pred == y)))
    missed_pneumonia = int(np.sum((p <= lo) & y))  # told "likely normal" but was pneumonia
    false_alarm = int(np.sum((p >= hi) & ~y))
    return {
        "coverage": n_ans / max(len(p), 1),
        "answered": n_ans,
        "referred": int(len(p) - n_ans),
        "selective_accuracy": correct / max(n_ans, 1),
        "answered_normal": int(np.sum(p <= lo)),
        "answered_pneumonia": int(np.sum(p >= hi)),
        "missed_pneumonia": missed_pneumonia,
        "false_alarm": false_alarm,
        "accuracy_on_referred": float(np.mean((p[~answered] >= 0.5) == y[~answered])) if n_ans < len(p) else float("nan"),
    }


def coverage_curve(probs, labels, n_points: int = 101):
    """Accuracy of the threshold-0.5 decision when keeping only the most confident fraction.

    Confidence is the distance of p from 0.5. Returns coverages and accuracies (descending coverage).
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    conf = np.abs(p - 0.5)
    order = np.argsort(-conf, kind="mergesort")
    correct = ((p >= 0.5) == y)[order].astype(np.float64)
    cum = np.cumsum(correct) / np.arange(1, len(p) + 1)
    cov = np.linspace(1.0, 0.05, n_points)
    k = np.maximum(np.round(cov * len(p)).astype(int), 1)
    return cov, cum[k - 1]


def bootstrap_ci(fn, probs, labels, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI of fn(probs, labels), resampling images with replacement."""
    rng = np.random.default_rng(seed)
    p = np.asarray(probs)
    y = np.asarray(labels)
    n = len(p)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        stats[b] = fn(p[i], y[i])
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
