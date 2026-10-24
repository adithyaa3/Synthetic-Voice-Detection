"""Metric code checked against small examples computed by hand."""
import math

import numpy as np
import pytest

from voicedetect.calibration import inconclusive_stats, reliability_table
from voicedetect.metrics import binary_metrics, choose_threshold, cluster_bootstrap_ci, ece, expected_cost
from voicedetect.predict import decide

Y = np.array([0, 0, 0, 1, 1, 1])
P = np.array([0.1, 0.6, 0.3, 0.8, 0.4, 0.9])
# threshold 0.5 -> predictions 0,1,0,1,0,1 : TN=2 FP=1 FN=1 TP=2


def test_confusion_and_threshold_metrics_by_hand():
    m = binary_metrics(Y, P, 0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 1, 1, 2)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["balanced_accuracy"] == pytest.approx(2 / 3)


def test_roc_auc_by_hand():
    # positive scores {0.8,0.4,0.9} vs negative {0.1,0.6,0.3}: pairs won = 3 + 2 + 3 = 8 of 9
    assert binary_metrics(Y, P)["roc_auc"] == pytest.approx(8 / 9)


def test_pr_auc_average_precision_by_hand():
    # ranking: .9(+) .8(+) .6(-) .4(+) .3(-) .1(-) -> precision at each positive: 1, 1, 3/4
    assert binary_metrics(Y, P)["pr_auc"] == pytest.approx((1 + 1 + 0.75) / 3)


def test_brier_and_log_loss_by_hand():
    brier = np.mean((P - Y) ** 2)
    ll = -np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P))
    m = binary_metrics(Y, P)
    assert m["brier"] == pytest.approx(brier)
    assert m["log_loss"] == pytest.approx(ll)
    assert brier == pytest.approx((0.01 + 0.36 + 0.09 + 0.04 + 0.36 + 0.01) / 6)


def test_ece_by_hand():
    y = np.array([0, 1, 1, 1])
    p = np.array([0.05, 0.15, 0.85, 0.95])
    # bins: [0,.1): {.05 -> y 0} err .05 ; [.1,.2): {.15 -> y 1} err .85 ; [.8,.9): err .15 ; [.9,1]: err .05
    assert ece(y, p) == pytest.approx(0.25 * (0.05 + 0.85 + 0.15 + 0.05))
    assert ece(np.array([1, 0]), np.array([1.0, 0.0])) == 0.0


def test_reliability_table_counts_and_means():
    t = reliability_table(np.array([0, 1, 1, 1]), np.array([0.05, 0.15, 0.85, 0.95]))
    assert t["count"].sum() == 4
    assert t.loc[8, "fraction_ai"] == 1.0 and t.loc[0, "fraction_ai"] == 0.0


def test_single_class_input_gives_nan_auc_not_crash():
    m = binary_metrics(np.array([1, 1]), np.array([0.7, 0.2]))
    assert math.isnan(m["roc_auc"]) and m["recall"] == 0.5


def test_expected_cost_and_threshold_choice():
    # at t=0.5: FP=1 FN=1 -> (2*1 + 1*1)/6 = 0.5 ; at t=0.65: FP=0, FN=1 -> 1/6
    assert expected_cost(Y, P, 0.5, 2, 1) == pytest.approx(0.5)
    assert expected_cost(Y, P, 0.65, 2, 1) == pytest.approx(1 / 6)
    t, cost = choose_threshold(Y, P, 2, 1)
    assert cost == pytest.approx(1 / 6)
    assert 0.6 < t <= 0.8  # any threshold in (0.6, 0.8] removes the false alarm and keeps 2 hits


def test_decision_band_logic():
    assert decide(0.80, 0.6, 0.1) == ("ai", "ai", False)
    assert decide(0.65, 0.6, 0.1) == ("inconclusive", "ai", True)
    assert decide(0.55, 0.6, 0.1) == ("inconclusive", "human", True)
    assert decide(0.40, 0.6, 0.1) == ("human", "human", False)
    assert decide(0.7001, 0.6, 0.1)[0] == "ai"  # just outside the band (0.70 itself is float-ambiguous)


def test_inconclusive_stats():
    s = inconclusive_stats(Y, P, 0.5, 0.15)  # band (0.35, 0.65): .6 and .4 inside
    assert s["fraction_inconclusive"] == pytest.approx(2 / 6)
    assert s["accuracy_outside_band"] == pytest.approx(1.0)


def test_cluster_bootstrap_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(50), np.ones(50)]
    p = np.clip(y * 0.3 + rng.uniform(0, 0.7, 100), 0, 1)
    g = np.repeat(np.arange(20), 5)
    est, lo, hi = cluster_bootstrap_ci(y, p, g, n_boot=300, seed=1)
    assert lo <= est <= hi
