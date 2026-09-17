"""S9: successive-halving ladder and promotion (#91).

The anomaly-detection tests are calibrated from the 09.09.2026 measurement:
h256 at seed 42 → 0.7515, at seed 43 → 0.5591. The collapse (0.5591) is
detected by the IQR/Tukey-fence guard, not by a σ-based z-test — the σ test
fails because the outlier inflates its own standard deviation. The IQR test
uses the interquartile range of the whole rung, so the outlier only shifts Q3
(and therefore the IQR), leaving Q1 and the lower fence tight.
"""

import pytest

from atr_training.rungs import (
    AnomalyFlag,
    RungError,
    iqr_anomaly_cutoff,
    detect_anomaly,
    plan_rungs,
    promote,
)


# ── original ladder and promotion tests (unchanged) ─────────────────────────

def test_the_ladder_from_the_plan():
    rungs = plan_rungs(45, eta=3, base_epochs=3, pages=[2500, 5000, 24744])
    assert [(r.configs, r.epochs, r.pages) for r in rungs] == [
        (45, 3, 2500),
        (15, 9, 5000),
        (5, 27, 24744),
        (1, 81, 24744),
    ]


def test_pages_are_padded_with_the_last_value():
    rungs = plan_rungs(9, eta=3, base_epochs=2, pages=[1000])
    assert [r.pages for r in rungs] == [1000, 1000, 1000]


def test_ladder_terminates_at_one_config():
    assert plan_rungs(1)[0].configs == 1
    assert len(plan_rungs(1)) == 1


def test_max_rungs_truncates():
    assert len(plan_rungs(81, eta=3, max_rungs=2)) == 2


def test_rejects_incoherent_plans():
    with pytest.raises(RungError):
        plan_rungs(0)
    with pytest.raises(RungError):
        plan_rungs(9, eta=1)
    with pytest.raises(RungError):
        plan_rungs(9, base_epochs=0)


def test_promotes_the_top_third_by_accuracy():
    scores = {f"c{i}": i / 10 for i in range(9)}
    result = promote(scores, eta=3)
    assert result.promoted == ["c8", "c7", "c6"]
    assert len(result.eliminated) == 6


def test_ties_break_by_config_id_so_reruns_agree():
    first = promote({"b": 0.8, "a": 0.8, "c": 0.1}, eta=3, keep=1)
    second = promote({"c": 0.1, "a": 0.8, "b": 0.8}, eta=3, keep=1)
    assert first.promoted == second.promoted == ["a"]


def test_a_config_without_a_score_never_promotes_and_is_reported():
    result = promote({"good": 0.9, "crashed": None, "poor": 0.1}, eta=3)
    assert result.promoted == ["good"]
    assert result.unscored == ["crashed"]
    assert "crashed" not in result.eliminated


def test_keep_is_a_fraction_of_everything_that_entered_the_rung():
    scores = {"a": 0.9, "b": 0.8, "c": 0.7, "x": None, "y": None, "z": None}
    result = promote(scores, eta=3)
    assert result.promoted == ["a", "b"]
    assert result.eliminated == ["c"]


def test_keep_override_and_empty_input():
    assert promote({"a": 1.0, "b": 0.5}, keep=2).promoted == ["a", "b"]
    with pytest.raises(RungError):
        promote({})
    with pytest.raises(RungError):
        promote({"a": 1.0}, eta=1)


# ── IQR anomaly detection (S9, addressing 09.09.2026 comment) ───────────────

def test_iqr_cutoff_h256_collapse():
    """The h256-collapse case from 09.09: 8 configs, one collapsed at 0.5591.
    Q1=0.7355, IQR≈0.0139, fence≈0.7147. h256-c43 is well below the fence."""
    scores = {
        "h96-c42":  0.7420, "h96-c43":  0.7380,
        "h128-c42": 0.7440, "h128-c43": 0.7355,
        "h192-c42": 0.7494, "h192-c43": 0.7272,
        "h256-c42": 0.7515, "h256-c43": 0.5591,
    }
    lower_fence, q1, iqr = iqr_anomaly_cutoff(scores)
    assert abs(q1 - 0.7355) < 0.001
    assert abs(iqr - 0.0139) < 0.001
    assert abs(lower_fence - 0.7147) < 0.001


def test_h256_collapse_is_eliminated_not_promoted():
    """h256 at seed 43 (0.5591) must not be promoted, even if it ranks in the
    top-K by raw score (here it ranks 8th anyway, but the key property is the
    anomaly flag is set)."""
    scores = {
        "h96-c42":  0.7420, "h96-c43":  0.7380,
        "h128-c42": 0.7440, "h128-c43": 0.7355,
        "h192-c42": 0.7494, "h192-c43": 0.7272,
        "h256-c42": 0.7515, "h256-c43": 0.5591,
    }
    result = promote(scores, eta=3)  # keep = 8 // 3 = 2
    assert "h256-c43" not in result.promoted
    assert "h256-c43" in result.eliminated
    assert len(result.anomalies) == 1
    assert result.anomalies[0].config_id == "h256-c43"
    assert result.anomalies[0].score == 0.5591
    assert "lower fence" in result.anomalies[0].reason.lower()
    assert result.promoted == ["h256-c42", "h192-c42"]


def test_tight_cluster_outlier_is_flagged():
    """Four configs in a 0.002-wide IQR cluster (0.749–0.751) and one outlier
    at 0.680: IQR is tiny, fence is 0.746. The outlier is flagged even though
    the absolute gap (0.07) looks modest — it is 35× the IQR."""
    scores = {"a": 0.751, "b": 0.680, "c": 0.750, "d": 0.749}
    lower_fence, q1, iqr = iqr_anomaly_cutoff(scores)
    assert abs(q1 - 0.749) < 0.001
    assert abs(iqr - 0.002) < 0.001
    assert abs(lower_fence - 0.746) < 0.001

    assert detect_anomaly("b", 0.680, lower_fence) is not None
    assert detect_anomaly("a", 0.751, lower_fence) is None
    assert detect_anomaly("c", 0.750, lower_fence) is None
    assert detect_anomaly("d", 0.749, lower_fence) is None


def test_no_anomaly_when_fewer_than_four_configs():
    """IQR needs at least 4 configs (to define Q1 and Q3). With fewer the
    fence is -inf and nothing is flagged."""
    for n_scores in [{"a": 0.9, "b": 0.5, "c": 0.8}, {"x": 0.9, "y": 0.85}]:
        lf, _, _ = iqr_anomaly_cutoff(n_scores)
        assert lf == float("-inf")
        assert all(detect_anomaly(cid, v, lf) is None for cid, v in n_scores.items())


def test_no_anomaly_when_all_scores_identical():
    """Zero IQR → fence = -inf (so score < -inf is always False)."""
    scores = {"a": 0.7, "b": 0.7, "c": 0.7, "d": 0.7}
    lf, q1, iqr = iqr_anomaly_cutoff(scores)
    assert lf == float("-inf")
    assert iqr == 0.0
    assert all(detect_anomaly(cid, v, lf) is None for cid, v in scores.items())


def test_iqr_factor_controls_fence_tightness():
    """Smaller factor → tighter fence.  Scores [0.751,0.745,0.752,0.750,0.760]:
    Q1=0.750, IQR=0.002.  b=0.745 is below fence(1.5)=0.747 but above
    fence(4.0)=0.742."""
    scores = {"a": 0.751, "b": 0.745, "c": 0.752, "d": 0.750, "e": 0.760}
    lf15, _, _ = iqr_anomaly_cutoff(scores, factor=1.5)
    lf40, _, _ = iqr_anomaly_cutoff(scores, factor=4.0)
    assert lf15 > lf40  # smaller factor → looser fence (more points flagged)
    assert detect_anomaly("b", 0.745, lf15) is not None
    assert detect_anomaly("b", 0.745, lf40) is None


def test_promote_respects_iqr_factor_and_keep_window():
    """Dataset: a=0.753(1st), b=0.745(2nd), c=0.752, d=0.751, e=0.760.
    Q1=0.750, IQR=0.002. fence(1.5)=0.747 → b=0.745 below → anomaly, not promoted.
    fence(4.0)=0.742 → b=0.745 above → not anomaly, would be promoted by raw rank,
    but keep=1 so a is promoted anyway. Test the factor-sensitive elimination."""
    scores = {"a": 0.753, "b": 0.745, "c": 0.752, "d": 0.751, "e": 0.760}
    result = promote(scores, eta=3, iqr_factor=1.5)  # keep=1
    assert "b" in [a.config_id for a in result.anomalies]
    assert "b" not in result.promoted

    result2 = promote(scores, eta=3, iqr_factor=4.0)  # keep=1
    assert "b" not in [a.config_id for a in result2.anomalies]
    assert "b" not in result2.promoted  # still not promoted (keep=1, a is top)


def test_collapse_is_eliminated_not_unscored():
    """An anomaly has a score — it was not unscored. Eliminated because it
    collapsed, not because it produced nothing."""
    scores = {"good": 0.9, "collapsed": 0.45, "ok1": 0.85, "ok2": 0.80}
    result = promote(scores, eta=3, iqr_factor=1.5)
    assert "collapsed" in result.eliminated
    assert "collapsed" not in result.unscored
    assert any(a.config_id == "collapsed" for a in result.anomalies)


def test_promotion_str_includes_anomaly_ids():
    scores = {"a": 0.9, "b": 0.3, "c": 0.8, "d": 0.75}
    result = promote(scores, eta=3, iqr_factor=1.5)
    s = str(result).lower()
    assert "anomaly" in s


def test_anomalies_field_is_list_of_anomalyflags():
    scores = {
        "h96-c42":  0.7420, "h96-c43":  0.7380,
        "h128-c42": 0.7440, "h128-c43": 0.7355,
        "h192-c42": 0.7494, "h192-c43": 0.7272,
        "h256-c42": 0.7515, "h256-c43": 0.5591,
    }
    result = promote(scores, eta=3)
    assert all(isinstance(a, AnomalyFlag) for a in result.anomalies)
    assert result.anomalies[0].config_id == "h256-c43"
    assert "0.5591" in result.anomalies[0].reason
    assert "lower fence" in result.anomalies[0].reason.lower()
