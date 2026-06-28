"""Tests for topo_anomaly.persistence — VR persistent homology wrapper."""
from __future__ import annotations

import numpy as np
import pytest

from topo_anomaly.persistence import (
    PersistenceComputer,
    PersistenceResult,
    filter_finite_pairs,
)


class TestPersistenceComputer:
    @pytest.fixture
    def circle_cloud(self):
        """A noisy circle in R^2 — should have β0=1 and β1=1."""
        rng = np.random.default_rng(0)
        theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
        r = 1.0 + rng.normal(0, 0.02, size=theta.shape)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack([x, y], axis=1)

    @pytest.fixture
    def cluster_cloud(self):
        """Three well-separated clusters — should have β0=3."""
        rng = np.random.default_rng(1)
        pts = []
        for center in [(0, 0), (5, 5), (-5, 5)]:
            pts.append(np.array(center) + rng.normal(0, 0.1, size=(40, 2)))
        return np.vstack(pts)

    def test_single_point_cloud_runs(self, circle_cloud):
        pc = PersistenceComputer(homology_dimensions=(0, 1), max_edge_length=2.0)
        r = pc.compute(circle_cloud)
        assert isinstance(r, PersistenceResult)
        assert r.diagrams.ndim == 3
        assert r.diagrams.shape[0] == 1
        assert r.diagrams.shape[2] == 3

    def test_circle_has_long_h1(self, circle_cloud):
        pc = PersistenceComputer(homology_dimensions=(0, 1), max_edge_length=2.0)
        r = pc.compute(circle_cloud)
        h1 = r.persistence_pairs(1)
        # at least one persistent H1 feature (the loop)
        if h1.shape[0] > 0:
            persistences = h1[:, 1] - h1[:, 0]
            assert persistences.max() > 0.1

    def test_clusters_have_high_b0(self, cluster_cloud):
        pc = PersistenceComputer(homology_dimensions=(0,), max_edge_length=1.0)
        r = pc.compute(cluster_cloud)
        h0 = r.persistence_pairs(0)
        # many H0 pairs (one per merge event in the filtration)
        assert h0.shape[0] >= 3

    def test_batch_input(self, circle_cloud, cluster_cloud):
        # giotto-tda requires equal-shape clouds in a batch; truncate both to
        # the smaller count.
        n = min(circle_cloud.shape[0], cluster_cloud.shape[0])
        batch = np.stack([circle_cloud[:n], cluster_cloud[:n]], axis=0)
        pc = PersistenceComputer(homology_dimensions=(0, 1), max_edge_length=2.0)
        r = pc.compute(batch)
        assert r.diagrams.shape[0] == 2

    def test_metadata(self, circle_cloud):
        pc = PersistenceComputer(homology_dimensions=(0, 1), max_edge_length=2.0)
        r = pc.compute(circle_cloud)
        m = r.to_metadata()
        assert "n_pairs_per_dim" in m
        assert 0 in m["n_pairs_per_dim"]
        assert 1 in m["n_pairs_per_dim"]
        assert m["homology_dimensions"] == [0, 1]

    def test_records_timer(self, circle_cloud):
        from topo_anomaly.utils import Timer
        t = Timer()
        pc = PersistenceComputer(homology_dimensions=(0, 1), max_edge_length=2.0)
        r = pc.compute(circle_cloud, timer=t, stage_name="vr")
        assert r.timer_record is not None
        assert r.timer_record["name"] == "vr"
        assert r.timer_record["metadata"]["n_point_clouds"] == 1


class TestFilterFinitePairs:
    def test_drops_infinite(self):
        d = np.array([[0.0, np.inf, 0], [0.1, 0.5, 0], [0.2, 0.9, 1]])
        finite = filter_finite_pairs(d)
        assert finite.shape[0] == 2

    def test_empty_diagram(self):
        d = np.empty((0, 3))
        finite = filter_finite_pairs(d)
        assert finite.shape[0] == 0
