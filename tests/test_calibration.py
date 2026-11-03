import itertools

import numpy as np
import pytest

from triage import calibration as cal


def test_sigmoid_is_stable_at_extremes():
    p = cal.sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.allclose(p, [0.0, 0.5, 1.0])
    assert np.all(np.isfinite(p))


def test_auroc_matches_pairwise_count():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 5, 40).astype(float)  # many ties
    y = rng.integers(0, 2, 40)
    pos, neg = s[y == 1], s[y == 0]
    pairs = [(a > b) + 0.5 * (a == b) for a, b in itertools.product(pos, neg)]
    assert cal.auroc(s, y) == pytest.approx(np.mean(pairs))


def test_auroc_edge_cases():
    assert cal.auroc([0.1, 0.9], [0, 1]) == 1.0
    assert cal.auroc([0.9, 0.1], [0, 1]) == 0.0
    assert np.isnan(cal.auroc([0.1, 0.2], [1, 1]))


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    true_logit = rng.normal(0, 2, 20000)
    y = rng.random(20000) < cal.sigmoid(true_logit)
    t = cal.fit_temperature(3.0 * true_logit, y)  # a model 3x too confident
    assert t == pytest.approx(3.0, rel=0.08)
    assert cal.nll(3.0 * true_logit, y, t) < cal.nll(3.0 * true_logit, y, 1.0)


def test_ece_small_for_calibrated_and_large_for_overconfident():
    rng = np.random.default_rng(2)
    p = rng.random(50000)
    y = rng.random(50000) < p
    assert cal.ece(p, y) < 0.01
    sharp = np.where(p > 0.5, 0.99, 0.01)
    assert cal.ece(sharp, y) > 0.2


def test_reliability_bins_count_everything_including_p_equal_one():
    p = np.array([0.0, 0.05, 0.5, 0.95, 1.0])
    b = cal.reliability_bins(p, [0, 0, 1, 1, 1], n_bins=10)
    assert b["count"].sum() == 5
    assert b["count"][-1] == 2 and b["count"][0] == 2


def test_brier():
    assert cal.brier([1.0, 0.0], [1, 0]) == 0.0
    assert cal.brier([0.5, 0.5], [1, 0]) == 0.25


def test_classification_report_counts():
    r = cal.classification_report([0.9, 0.8, 0.2, 0.6], [1, 0, 0, 1])
    assert (r["tp"], r["fp"], r["tn"], r["fn"]) == (2, 1, 1, 0)
    assert r["sensitivity"] == 1.0 and r["specificity"] == 0.5
    assert r["accuracy"] == 0.75


def test_threshold_for_specificity_hits_target():
    rng = np.random.default_rng(3)
    p = rng.random(1000)
    y = rng.random(1000) < p
    t = cal.threshold_for_specificity(p, y, 0.9)
    spec = cal.classification_report(p, y, t)["specificity"]
    assert spec >= 0.9
    assert spec < 0.9 + 2.0 / np.sum(~y)


def test_triage_band_edges():
    assert cal.triage_band(0.8, 0.2, 0.8) == "pneumonia"
    assert cal.triage_band(0.2, 0.2, 0.8) == "normal"
    assert cal.triage_band(0.5, 0.2, 0.8) == "unsure"


def test_selective_report():
    p = np.array([0.95, 0.9, 0.5, 0.1, 0.05, 0.6])
    y = np.array([1, 0, 1, 1, 0, 0])
    r = cal.selective_report(p, y, 0.2, 0.8)
    assert r["answered"] == 4 and r["referred"] == 2
    assert r["coverage"] == pytest.approx(4 / 6)
    assert r["selective_accuracy"] == pytest.approx(2 / 4)
    assert r["missed_pneumonia"] == 1 and r["false_alarm"] == 1


def test_coverage_curve_full_coverage_equals_accuracy_and_is_better_when_selective():
    rng = np.random.default_rng(4)
    z = rng.normal(0, 3, 4000)
    y = rng.random(4000) < cal.sigmoid(z)
    p = cal.sigmoid(z)
    cov, acc = cal.coverage_curve(p, y)
    assert cov[0] == 1.0
    assert acc[0] == pytest.approx(np.mean((p >= 0.5) == y))
    assert acc[-1] > acc[0]


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(5)
    p = rng.random(500)
    y = rng.random(500) < p
    lo, hi = cal.bootstrap_ci(cal.auroc, p, y, n_boot=300)
    assert lo < cal.auroc(p, y) < hi
