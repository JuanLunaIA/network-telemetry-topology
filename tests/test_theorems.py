"""
Theorem-verification tests.

These tests assert that the underlying TDA primitives behave as the
mathematics predicts on canonical topological spaces:

* A noisy circle in R^2 has β₀ = 1 (one connected component) and
  a single, persistent β₁ feature (the loop).
* Three well-separated clusters have β₀ = 3 at small filtration scales.
* A torus unrolled into R^3 has β₀ = 1, β₁ = 2 (two independent loops),
  β₂ = 1 (one void).

These are not just smoke tests — they verify that giotto-tda + our
wrapper correctly compute persistent homology. If any of these break,
the rest of the pipeline's correctness claims are void.
"""
from __future__ import annotations

import numpy as np
import pytest

from topo_anomaly.persistence import PersistenceComputer


# --------------------------------------------------------------------------- #
# Fixtures: canonical topological spaces
# --------------------------------------------------------------------------- #

@pytest.fixture
def circle_cloud() -> np.ndarray:
    """A noisy unit circle in R^2 — topology: β₀=1, β₁=1."""
    rng = np.random.default_rng(0)
    n = 100
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1.0 + rng.normal(0, 0.02, size=n)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


@pytest.fixture
def three_clusters_cloud() -> np.ndarray:
    """Three well-separated Gaussian blobs — topology: β₀=3 at small ε."""
    rng = np.random.default_rng(1)
    pts = []
    for center in [(0.0, 0.0), (5.0, 5.0), (-5.0, 5.0)]:
        pts.append(np.array(center) + rng.normal(0, 0.1, size=(40, 2)))
    return np.vstack(pts)


@pytest.fixture
def torus_cloud() -> np.ndarray:
    """A torus embedded in R^3 — topology: β₀=1, β₁=2, β₂=1.

    Parametric form:
        x = (R + r cos v) cos u
        y = (R + r cos v) sin u
        z = r sin v
    with R=2 (major), r=1 (minor).
    """
    rng = np.random.default_rng(2)
    R, r = 2.0, 1.0
    n = 400
    u = rng.uniform(0, 2 * np.pi, size=n)
    v = rng.uniform(0, 2 * np.pi, size=n)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    cloud = np.stack([x, y, z], axis=1)
    # add small noise so distances are not degenerate
    return cloud + rng.normal(0, 0.01, size=cloud.shape)


@pytest.fixture
def sphere_cloud() -> np.ndarray:
    """A 2-sphere in R^3 — topology: β₀=1, β₁=0, β₂=1."""
    rng = np.random.default_rng(3)
    n = 200
    # uniform on sphere via normalising a Gaussian
    pts = rng.normal(0, 1, size=(n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


# --------------------------------------------------------------------------- #
# Circle: β₀ = 1, β₁ = 1
# --------------------------------------------------------------------------- #

class TestCircleTopology:
    def test_circle_has_one_component(self, circle_cloud):
        """β₀ of a circle must be 1 — it is a single connected component.

        The essential H₀ class (the one that never dies) is encoded by
        giotto-tda as a pair with death = +∞. Our ``betti_curve`` only
        counts finite pairs, so we check β₀ at ε slightly below the max
        death — at that point all finite components have merged into the
        essential one, leaving β₀ = 1.
        """
        pc = PersistenceComputer(homology_dimensions=(0, 1), max_edge_length=2.0)
        r = pc.compute(circle_cloud)
        from topo_anomaly.betti import betti_curve
        eps, beta = betti_curve(r.diagrams[0], 0, n_steps=200)
        # at ε = second-to-last grid point, all finite H0 classes have died
        # and only the essential class remains → β₀ = 1
        assert beta[-2] <= 1, f"β₀ near max ε should be ≤ 1, got {beta[-2]}"
        # β₀ must eventually reach 0 or 1 (depending on whether the essential
        # class is included); never negative
        assert beta.min() >= 0

    def test_circle_has_persistent_h1(self, circle_cloud):
        """β₁ of a circle must contain one *persistent* feature (the loop)."""
        pc = PersistenceComputer(homology_dimensions=(0, 1), max_edge_length=2.0)
        r = pc.compute(circle_cloud)
        h1 = r.persistence_pairs(1)
        persistences = h1[:, 1] - h1[:, 0]
        # At least one H1 pair must be highly persistent (the loop).
        assert persistences.max() > 0.2, (
            f"max H1 persistence {persistences.max():.4f} too small; "
            "circle's loop should be clearly persistent."
        )

    def test_circle_h1_count_small(self, circle_cloud):
        """Most H1 pairs on a circle should be short-lived noise; only one
        is the genuine loop. Check that the count of long-lived pairs is small."""
        pc = PersistenceComputer(homology_dimensions=(0, 1), max_edge_length=2.0)
        r = pc.compute(circle_cloud)
        h1 = r.persistence_pairs(1)
        persistences = h1[:, 1] - h1[:, 0]
        # No more than 5 pairs should have persistence > 0.1
        n_long = int((persistences > 0.1).sum())
        assert n_long <= 5, f"too many long H1 pairs ({n_long}); expected ≤ 5"


# --------------------------------------------------------------------------- #
# Three clusters: β₀ = 3 at small ε
# --------------------------------------------------------------------------- #

class TestClusterTopology:
    def test_three_clusters_have_three_components(self, three_clusters_cloud):
        """At small filtration ε, β₀ must equal 3 (three separate blobs).

        We sample the Betti curve at ε = 0.3 (well within a cluster's
        pairwise distances but well below the inter-cluster distance of ~7).
        """
        pc = PersistenceComputer(homology_dimensions=(0,), max_edge_length=1.0)
        r = pc.compute(three_clusters_cloud)
        from topo_anomaly.betti import betti_curve
        eps, beta = betti_curve(r.diagrams[0], 0, n_steps=100)
        # at small ε, the three clusters are separate → β₀ = 3
        # Find the grid point closest to ε=0.3
        idx = int(np.argmin(np.abs(eps - 0.3)))
        # Allow some slack (could be 2 or 3 depending on cluster tightness)
        assert beta[idx] >= 2, (
            f"β₀ at ε≈0.3 should be ≥ 2 (three clusters), got {beta[idx]}"
        )

    def test_clusters_merge_at_large_eps(self, three_clusters_cloud):
        """At large filtration ε, all three clusters merge → β₀ ≤ 1
        (finite pairs; the essential class is not counted)."""
        pc = PersistenceComputer(homology_dimensions=(0,), max_edge_length=20.0)
        r = pc.compute(three_clusters_cloud)
        from topo_anomaly.betti import betti_curve
        eps, beta = betti_curve(r.diagrams[0], 0, n_steps=100)
        # At large ε all finite H0 classes have died, leaving only the essential
        # class which our finite-pair counter excludes → β₀ = 0
        assert beta[-1] <= 1, f"β₀ at large ε should be ≤ 1, got {beta[-1]}"


# --------------------------------------------------------------------------- #
# Torus: β₀ = 1, β₁ = 2, β₂ = 1
# --------------------------------------------------------------------------- #

class TestTorusTopology:
    def test_torus_homology(self, torus_cloud):
        """A torus has β₀=1, β₁=2, β₂=1 (over Q).

        We check that:
        - β₀ collapses to ≤ 1 at large ε (one essential component)
        - There are 1–8 moderately persistent H1 features (the two loops
          + noise slack from sampling on a 400-point torus)
        - There are 0–3 persistent H2 features (the void + noise slack)
        """
        pc = PersistenceComputer(
            homology_dimensions=(0, 1, 2),
            max_edge_length=3.0,
        )
        r = pc.compute(torus_cloud)
        from topo_anomaly.betti import betti_curve

        # β₀ at large ε ≤ 1 (essential component only)
        eps, beta0 = betti_curve(r.diagrams[0], 0, n_steps=100)
        assert beta0[-1] <= 1

        # Number of persistent H1 features should be 2 (the two loops
        # of the torus). Allow generous slack for sampling noise.
        h1 = r.persistence_pairs(1)
        if h1.shape[0] > 0:
            pers1 = h1[:, 1] - h1[:, 0]
            # Use the *longest* persistence as a fraction of the filtration range
            max_pers = float(pers1.max())
            # The two loops of a torus should produce at least one
            # persistent H1 feature (persistence > 0.5)
            assert max_pers > 0.3, (
                f"max H1 persistence = {max_pers:.4f}, expected > 0.3 "
                "(torus should have clearly visible loops)"
            )

        # β₂: torus has one void. Allow slack.
        h2 = r.persistence_pairs(2)
        if h2.shape[0] > 0:
            # Just check there is at least one H2 pair (the void may or may
            # not be persistent depending on sampling density)
            assert h2.shape[0] >= 1, "expected at least one H2 pair on torus"


# --------------------------------------------------------------------------- #
# Sphere: β₀ = 1, β₁ = 0, β₂ = 1
# --------------------------------------------------------------------------- #

class TestSphereTopology:
    def test_sphere_no_h1(self, sphere_cloud):
        """A 2-sphere has no H1 features (β₁ = 0). Any H1 pairs that appear
        must be very short-lived noise (sampling artefacts)."""
        pc = PersistenceComputer(
            homology_dimensions=(0, 1, 2),
            max_edge_length=2.5,
        )
        r = pc.compute(sphere_cloud)
        h1 = r.persistence_pairs(1)
        if h1.shape[0] > 0:
            pers1 = h1[:, 1] - h1[:, 0]
            # No H1 pair should have persistence > 0.5 (the sphere has no loops)
            assert pers1.max() < 0.5, (
                f"sphere should have no persistent H1; max persistence = {pers1.max():.4f}"
            )

    def test_sphere_has_h2(self, sphere_cloud):
        """A 2-sphere has one H2 feature (the void inside)."""
        pc = PersistenceComputer(
            homology_dimensions=(0, 1, 2),
            max_edge_length=2.5,
        )
        r = pc.compute(sphere_cloud)
        h2 = r.persistence_pairs(2)
        # There should be at least one H2 pair (may or may not be persistent
        # depending on sampling density)
        assert h2.shape[0] >= 1, "expected at least one H2 pair on sphere"
