"""Tests for topo_anomaly.detection — anomaly scoring."""
from __future__ import annotations

import numpy as np
import pytest

from topo_anomaly.detection import (
    AnomalyReport,
    TopologicalAnomalyDetector,
    _rank_average,
    _robust_zscore,
)


class TestRobustZscore:
    def test_shape_preserved(self):
        x = np.random.randn(50, 4)
        z = _robust_zscore(x)
        assert z.shape == x.shape

    def test_median_zero(self):
        # Robust z-score is |x - med| / scaled_mad, so its *minimum* per column
        # is 0 (achieved at the median). The median of the absolute values is
        # ~0.6745 (half-normal), which we use as a sanity check.
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1.0, size=(500, 3))
        z = _robust_zscore(x)
        # Each column's minimum should be 0 (some sample equals the median)
        assert z.min(axis=0).max() < 0.1
        # Median of |N(0,1)| ≈ 0.6745
        assert np.allclose(np.median(z, axis=0), 0.6745, atol=0.1)

    def test_handles_constant_column(self):
        x = np.ones((10, 2))
        x[:, 1] = np.arange(10)
        z = _robust_zscore(x)
        assert np.all(np.isfinite(z))
        assert np.all(z[:, 0] == 0.0)


class TestRankAverage:
    def test_two_arrays(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 2.0, 1.0])
        ra = _rank_average([a, b])
        assert ra.shape == (3,)
        # both rank 2 -> average 2.0
        assert np.allclose(ra, 2.0)

    def test_three_arrays(self):
        # rank_average treats each input as a scorer over the same set of items.
        # a=[1,2,3] → ranks [1,2,3]; b=[1,3,2] → [1,3,2]; c=[3,1,2] → [3,1,2].
        # Per-position average: [(1+1+3)/3, (2+3+1)/3, (3+2+2)/3] = [5/3, 2, 7/3].
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 3.0, 2.0])
        c = np.array([3.0, 1.0, 2.0])
        ra = _rank_average([a, b, c])
        assert np.allclose(ra, [5.0 / 3.0, 2.0, 7.0 / 3.0])


class TestTopologicalAnomalyDetector:
    @pytest.fixture
    def features_with_outliers(self):
        rng = np.random.default_rng(0)
        normal = rng.normal(0, 1.0, size=(100, 5))
        # Inject 5 strong outliers
        normal[:5] += 10.0
        return normal

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            TopologicalAnomalyDetector(method="nope")

    def test_invalid_contamination(self):
        with pytest.raises(ValueError):
            TopologicalAnomalyDetector(contamination=0.0)
        with pytest.raises(ValueError):
            TopologicalAnomalyDetector(contamination=1.0)

    def test_robust_z_finds_outliers(self, features_with_outliers):
        det = TopologicalAnomalyDetector(method="robust_z", contamination=0.1)
        rep = det.fit_predict(features_with_outliers)
        assert isinstance(rep, AnomalyReport)
        assert rep.scores.shape == (100,)
        assert rep.labels.shape == (100,)
        assert int(rep.labels.sum()) == 10  # 10% contamination
        # The first 5 (true outliers) should be among the predicted ones
        assert rep.labels[:5].sum() == 5

    def test_lof_finds_outliers(self, features_with_outliers):
        det = TopologicalAnomalyDetector(method="lof", contamination=0.1, lof_n_neighbors=10)
        rep = det.fit_predict(features_with_outliers)
        assert int(rep.labels.sum()) == 10
        # LOF should catch most of the true outliers
        assert rep.labels[:5].sum() >= 3

    def test_ensemble(self, features_with_outliers):
        det = TopologicalAnomalyDetector(method="ensemble", contamination=0.1)
        rep = det.fit_predict(features_with_outliers)
        assert "robust_z" in rep.per_scorer
        assert "lof" in rep.per_scorer
        assert int(rep.labels.sum()) == 10

    def test_scores_normalised(self, features_with_outliers):
        det = TopologicalAnomalyDetector(method="robust_z", contamination=0.1)
        rep = det.fit_predict(features_with_outliers)
        assert rep.scores.min() >= 0.0
        assert rep.scores.max() <= 1.0

    def test_metrics_with_ground_truth(self, features_with_outliers):
        gt = np.zeros(100, dtype=int)
        gt[:5] = 1
        det = TopologicalAnomalyDetector(method="robust_z", contamination=0.1)
        rep = det.fit_predict(features_with_outliers)
        m = rep.metrics(gt)
        assert m["precision"] > 0
        assert m["recall"] > 0
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_wasserstein_requires_diagrams(self, features_with_outliers):
        det = TopologicalAnomalyDetector(method="wasserstein", contamination=0.1)
        with pytest.raises(ValueError):
            det.fit_predict(features_with_outliers, diagrams=None)

    def test_metadata(self, features_with_outliers):
        det = TopologicalAnomalyDetector(method="robust_z", contamination=0.1)
        rep = det.fit_predict(features_with_outliers)
        m = rep.to_metadata()
        assert m["method"] == "robust_z"
        assert m["n_windows"] == 100
        assert "threshold" in m
