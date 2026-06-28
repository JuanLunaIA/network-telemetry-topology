"""Tests for topo_anomaly.betti — Betti curves & preservation verifier."""
from __future__ import annotations

import numpy as np
import pytest

from topo_anomaly.betti import (
    BettiCurveComputer,
    BettiCurves,
    BettiPreservationVerifier,
    PreservationReport,
    betti_curve,
    bottleneck_distance,
    curve_distance,
)


class TestBettiCurve:
    def test_empty_diagram(self):
        d = np.empty((0, 3))
        eps, beta = betti_curve(d, 0, n_steps=10)
        assert beta.shape == (10,)
        assert np.all(beta == 0)

    def test_simple_diagram(self):
        # one H0 pair born at 0 dying at 1
        d = np.array([[0.0, 1.0, 0]])
        eps, beta = betti_curve(d, 0, n_steps=10)
        # at eps < 1, beta == 1; at eps >= 1, beta == 0
        assert beta[0] == 1
        assert beta[-1] == 0

    def test_two_h0_classes(self):
        d = np.array([[0.0, 1.0, 0], [0.0, 0.5, 0]])
        eps, beta = betti_curve(d, 0, n_steps=10)
        assert beta[0] == 2  # both alive at small eps

    def test_max_betti(self):
        d = np.array([[0.0, 1.0, 0], [0.0, 0.5, 0], [0.0, 0.2, 0]])
        eps, beta = betti_curve(d, 0, n_steps=10)
        assert int(beta.max()) == 3


class TestCurveDistance:
    def test_l1(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.0, 0.0, 0.0])
        assert curve_distance(a, b, p=1) == 6.0

    def test_l2(self):
        a = np.array([3.0, 4.0])
        b = np.array([0.0, 0.0])
        assert curve_distance(a, b, p=2) == 5.0

    def test_linf(self):
        a = np.array([3.0, 4.0])
        b = np.array([0.0, 0.0])
        assert curve_distance(a, b, p=np.inf) == 4.0

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            curve_distance(np.array([1, 2]), np.array([1, 2, 3]))


class TestBottleneckDistance:
    def test_identical_diagrams_zero(self):
        d = np.array([[0.0, 1.0, 0], [0.1, 0.5, 1]])
        # bottleneck of a diagram with itself is 0
        b = bottleneck_distance(d, d)
        assert b < 1e-6 or b == 0.0


class TestBettiCurveComputer:
    def test_compute_returns_curves(self):
        d = np.array([[0.0, 1.0, 0], [0.1, 0.5, 1]])
        c = BettiCurveComputer(homology_dimensions=(0, 1), n_steps=20)
        result = c.compute(d)
        assert isinstance(result, BettiCurves)
        assert 0 in result.curves
        assert 1 in result.curves
        assert result.curves[0].shape == (20,)

    def test_betti_number_max(self):
        d = np.array([[0.0, 1.0, 0], [0.0, 0.5, 0], [0.1, 0.4, 1]])
        c = BettiCurveComputer(homology_dimensions=(0, 1))
        result = c.compute(d)
        assert result.betti_number(0) == 2
        assert result.betti_number(1) == 1


class TestBettiPreservationVerifier:
    @pytest.fixture
    def circle_cloud(self):
        rng = np.random.default_rng(0)
        theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
        r = 1.0 + rng.normal(0, 0.02, size=theta.shape)
        return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    @pytest.fixture
    def circle_diagram(self, circle_cloud):
        from topo_anomaly.persistence import PersistenceComputer
        pc = PersistenceComputer(homology_dimensions=(0, 1), max_edge_length=2.0)
        return pc.compute(circle_cloud).diagrams[0]

    def test_identical_diagrams_preserved(self, circle_diagram):
        v = BettiPreservationVerifier(homology_dimensions=(0, 1))
        r = v.verify(circle_diagram, circle_diagram)
        assert isinstance(r, PreservationReport)
        assert r.betti0_preserved
        assert r.betti1_preserved
        assert r.preservation_score > 0.99
        assert r.bottleneck_distance < 1e-6 or r.bottleneck_distance == 0.0

    def test_perturbed_diagram_reduces_score(self, circle_diagram):
        # Add noise to the diagram -> should reduce preservation score
        noisy = circle_diagram.copy()
        rng = np.random.default_rng(1)
        noisy[:, :2] += rng.normal(0, 0.1, size=noisy[:, :2].shape)
        v = BettiPreservationVerifier(homology_dimensions=(0, 1))
        r = v.verify(circle_diagram, noisy)
        # Betti curves should differ
        assert r.per_dim[0]["l1_distance"] > 0 or r.per_dim[1]["l1_distance"] > 0

    def test_pca_preserves_b0(self, circle_cloud):
        """PCA from R^2 to R^2 should be identity-like on a circle."""
        from sklearn.decomposition import PCA
        from topo_anomaly.persistence import PersistenceComputer

        pca = PCA(n_components=2, random_state=0)
        reduced = pca.fit_transform(circle_cloud)
        pc = PersistenceComputer(homology_dimensions=(0, 1), max_edge_length=2.0)
        d_orig = pc.compute(circle_cloud).diagrams[0]
        d_red = pc.compute(reduced).diagrams[0]
        v = BettiPreservationVerifier(homology_dimensions=(0, 1))
        r = v.verify(d_orig, d_red)
        # B0 should be preserved (both have 1 component)
        assert r.betti0_preserved

    def test_metadata(self, circle_diagram):
        v = BettiPreservationVerifier(homology_dimensions=(0, 1))
        r = v.verify(circle_diagram, circle_diagram)
        m = r.to_metadata()
        assert "bottleneck_distance" in m
        assert "preservation_score" in m
        assert "per_dim" in m
        assert "0" in m["per_dim"]
        assert "1" in m["per_dim"]
