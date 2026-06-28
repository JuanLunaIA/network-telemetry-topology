"""Tests for topo_anomaly.features — topological feature extraction."""
from __future__ import annotations

import numpy as np
import pytest

from topo_anomaly.features import (
    TopologicalFeatureExtractor,
    persistent_entropy,
    persistence_landscape_proxy,
    persistence_stats,
)


class TestPersistentEntropy:
    def test_empty_returns_zero(self):
        d = np.empty((0, 3))
        assert persistent_entropy(d, 0) == 0.0

    def test_uniform_max_entropy(self):
        # All pairs have equal persistence -> entropy = log(N)
        d = np.array([[0.0, 1.0, 0], [0.0, 1.0, 0], [0.0, 1.0, 0]])
        e = persistent_entropy(d, 0)
        assert np.isclose(e, np.log(3), atol=1e-6)

    def test_single_pair_zero_entropy(self):
        d = np.array([[0.0, 1.0, 0]])
        e = persistent_entropy(d, 0)
        assert e == 0.0

    def test_filters_by_dim(self):
        d = np.array([[0.0, 1.0, 0], [0.0, 1.0, 1]])
        e0 = persistent_entropy(d, 0)
        e1 = persistent_entropy(d, 1)
        assert e0 == 0.0  # single H0 pair
        assert e1 == 0.0  # single H1 pair


class TestPersistenceStats:
    def test_empty(self):
        s = persistence_stats(np.empty((0, 3)), 0)
        assert s["count"] == 0

    def test_basic_stats(self):
        d = np.array([[0.0, 1.0, 0], [0.0, 2.0, 0], [0.0, 3.0, 0]])
        s = persistence_stats(d, 0)
        assert s["count"] == 3
        assert np.isclose(s["mean"], 2.0)
        assert np.isclose(s["max"], 3.0)
        assert np.isclose(s["sum"], 6.0)


class TestLandscapeProxy:
    def test_empty(self):
        p = persistence_landscape_proxy(np.empty((0, 3)), 0)
        assert p["birth_mean"] == 0.0

    def test_basic(self):
        d = np.array([[0.0, 1.0, 0], [1.0, 3.0, 0]])
        p = persistence_landscape_proxy(d, 0)
        assert np.isclose(p["birth_mean"], 0.5)
        assert np.isclose(p["death_mean"], 2.0)
        assert np.isclose(p["midlife_mean"], 1.25)


class TestFeatureExtractor:
    @pytest.fixture
    def circle_diagram(self):
        from topo_anomaly.persistence import PersistenceComputer
        rng = np.random.default_rng(0)
        theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
        r = 1.0 + rng.normal(0, 0.02, size=theta.shape)
        cloud = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
        pc = PersistenceComputer(homology_dimensions=(0, 1), max_edge_length=2.0)
        return pc.compute(cloud).diagrams[0]

    def test_feature_names(self):
        fe = TopologicalFeatureExtractor(homology_dimensions=(0, 1))
        names = fe.feature_names()
        assert len(names) == 20  # 10 features per dim * 2 dims
        assert names[0] == "H0_entropy"
        assert names[10] == "H1_entropy"

    def test_extract_shape(self, circle_diagram):
        fe = TopologicalFeatureExtractor(homology_dimensions=(0, 1))
        v = fe.extract(circle_diagram)
        assert v.shape == (20,)
        assert np.all(np.isfinite(v))

    def test_extract_batch_shape(self, circle_diagram):
        fe = TopologicalFeatureExtractor(homology_dimensions=(0, 1))
        batch = np.stack([circle_diagram] * 3, axis=0)
        F = fe.extract_batch(batch)
        assert F.shape == (3, 20)

    def test_features_differ_for_different_diagrams(self, circle_diagram):
        # Perturb the diagram and verify features change
        noisy = circle_diagram.copy()
        noisy[:, :2] += 0.5
        fe = TopologicalFeatureExtractor(homology_dimensions=(0, 1))
        v1 = fe.extract(circle_diagram)
        v2 = fe.extract(noisy)
        assert not np.allclose(v1, v2)
