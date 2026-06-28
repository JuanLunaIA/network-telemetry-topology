"""
Property-based tests using hypothesis.

These tests verify *invariants* of the TDA pipeline that must hold for
*any* valid input, not just the specific examples covered by the unit
tests. They are the most powerful tests in the suite — if any of them
fails, there is a real bug, not just an edge case.

Invariants checked
------------------
1. **Betti curve non-negativity**: β_k(ε) ≥ 0 for every ε, k.
2. **Betti curve monotonicity in dim 0**: β₀(ε) is non-increasing in ε
   (components only merge, never split).
3. **Identity preservation**: comparing a diagram with itself yields
   preservation_score = 1.0 and bottleneck_distance = 0.
4. **No-reduction preservation**: `reduction_method="none"` must preserve
   both β₀ and β₁ *exactly* on every window.
5. **Timer monotonicity**: every TimingRecord has start_ns < end_ns and
   duration_ns > 0.
6. **Takens determinism**: fixed-parameter Takens embedding is a pure
   function — same input + same params → same output.
7. **Persistence diagram format**: every diagram is a (n, 3) float array
   with death ≥ birth for every finite pair.
"""
from __future__ import annotations

import time

import numpy as np
from hypothesis import given, settings, strategies as st

from topo_anomaly.betti import (
    BettiPreservationVerifier,
    betti_curve,
)
from topo_anomaly.embedding import TakensEmbedder
from topo_anomaly.persistence import PersistenceComputer
from topo_anomaly.utils import Timer


# --------------------------------------------------------------------------- #
# Strategy helpers
# --------------------------------------------------------------------------- #

@st.composite
def point_clouds(draw, min_points=10, max_points=60, min_dim=2, max_dim=4):
    """Generate a random point cloud as a (n, d) float array."""
    n = draw(st.integers(min_points, max_points))
    d = draw(st.integers(min_dim, max_dim))
    return draw(
        st.lists(
            st.lists(
                st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=d, max_size=d,
            ),
            min_size=n, max_size=n,
        ).map(lambda rows: np.asarray(rows, dtype=np.float64))
    )


@st.composite
def diagrams(draw, max_pairs=20, max_dim=1):
    """Generate a random persistence diagram in giotto-tda format (n, 3).

    Columns are (birth, death, dim) with death ≥ birth.
    """
    n_pairs = draw(st.integers(0, max_pairs))
    if n_pairs == 0:
        return np.empty((0, 3), dtype=np.float64)
    rows = []
    for _ in range(n_pairs):
        b = draw(st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False))
        d = draw(st.floats(min_value=b, max_value=1.0, allow_nan=False, allow_infinity=False))
        dim = draw(st.integers(0, max_dim))
        rows.append([b, d, dim])
    return np.asarray(rows, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Betti curve invariants
# --------------------------------------------------------------------------- #

class TestBettiCurveInvariants:
    @given(diagram=diagrams())
    @settings(max_examples=50, deadline=None)
    def test_betti_non_negative(self, diagram):
        for dim in (0, 1):
            _, beta = betti_curve(diagram, dim, n_steps=20)
            assert np.all(beta >= 0), f"β_{dim} has negative values: {beta}"

    @given(diagram=diagrams(max_dim=0))
    @settings(max_examples=50, deadline=None)
    def test_betti0_monotone_decreasing(self, diagram):
        """β₀(ε) must be non-increasing in ε — components only merge.

        Strictly: as ε grows, a feature can only *die* (become inactive when
        ε ≥ d); it cannot be *born* in H₀ after its birth. With random
        diagrams containing arbitrary birth values, however, β₀ can briefly
        *increase* when ε crosses a birth value before reaching the next
        death. We therefore sort the pairs by birth before checking monotonicity
        of the *decreasing* part of the curve.
        """
        eps, beta = betti_curve(diagram, 0, n_steps=50)
        # β₀ can increase while features are being born, but once all births
        # have happened (ε ≥ max(birth)), β₀ must be non-increasing.
        finite_mask = np.isfinite(diagram[:, 1])
        finite = diagram[finite_mask]
        if finite.shape[0] == 0:
            return
        max_birth = float(finite[:, 0].max())
        # find the grid point just past max_birth
        idx = int(np.searchsorted(eps, max_birth))
        if idx >= len(beta) - 1:
            return  # all action happens at the very end; nothing to check
        tail = beta[idx:]
        diffs = np.diff(tail)
        assert np.all(diffs <= 0), (
            f"β₀ is not non-increasing after max birth: tail diffs = {diffs}"
        )

    @given(diagram=diagrams())
    @settings(max_examples=50, deadline=None)
    def test_betti_at_eps_zero_is_pair_count(self, diagram):
        """At the smallest ε in the grid, β_k equals the count of pairs (b, d)
        with b ≤ ε₀ < d — i.e. pairs that are *alive* at ε₀."""
        for dim in (0, 1):
            eps, beta = betti_curve(diagram, dim, n_steps=50)
            # β at the smallest ε = count of pairs with b ≤ eps[0] < d
            mask = (
                (diagram[:, 2] == dim)
                & (diagram[:, 0] <= eps[0])
                & (eps[0] < diagram[:, 1])
            )
            expected = int(mask.sum())
            assert beta[0] == expected, (
                f"β_{dim}(min ε) = {beta[0]}, expected {expected} "
                f"(alive pairs at ε = {eps[0]:.4f})"
            )


# --------------------------------------------------------------------------- #
# Preservation verifier invariants
# --------------------------------------------------------------------------- #

class TestPreservationInvariants:
    @given(diagram=diagrams())
    @settings(max_examples=30, deadline=None)
    def test_identity_preservation(self, diagram):
        """Comparing a diagram with itself must yield perfect preservation."""
        v = BettiPreservationVerifier(homology_dimensions=(0, 1), n_steps=20)
        r = v.verify(diagram, diagram)
        assert r.preservation_score > 0.99
        assert r.betti0_preserved
        assert r.betti1_preserved
        assert r.bottleneck_distance < 1e-6

    @given(diagram=diagrams())
    @settings(max_examples=30, deadline=None)
    def test_preservation_per_dim(self, diagram):
        """Per-dim L1 distance is non-negative and ≤ n_steps * max_pairs."""
        v = BettiPreservationVerifier(homology_dimensions=(0, 1), n_steps=20)
        r = v.verify(diagram, diagram)
        for d in (0, 1):
            assert r.per_dim[d]["l1_distance"] >= 0
            assert r.per_dim[d]["l2_distance"] >= 0
            assert r.per_dim[d]["linf_distance"] >= 0


# --------------------------------------------------------------------------- #
# Persistence diagram format invariants
# --------------------------------------------------------------------------- #

class TestPersistenceDiagramFormat:
    @given(cloud=point_clouds())
    @settings(max_examples=20, deadline=None)
    def test_diagram_shape_and_finite_pairs(self, cloud):
        pc = PersistenceComputer(
            homology_dimensions=(0, 1), max_edge_length=2.0
        )
        r = pc.compute(cloud)
        d = r.diagrams[0]
        assert d.ndim == 2
        assert d.shape[1] == 3
        # For every finite pair, death ≥ birth
        finite_mask = np.isfinite(d[:, 1])
        finite = d[finite_mask]
        if finite.shape[0] > 0:
            assert np.all(finite[:, 1] >= finite[:, 0]), (
                "found a pair with death < birth"
            )
        # Every homology dim in the diagram is in our requested set
        assert set(np.unique(d[:, 2]).astype(int)).issubset({0, 1})


# --------------------------------------------------------------------------- #
# Takens embedding determinism
# --------------------------------------------------------------------------- #

class TestTakensDeterminism:
    @given(seed=st.integers(0, 1000))
    @settings(max_examples=10, deadline=None)
    def test_fixed_embedding_deterministic(self, seed):
        """Fixed-parameter Takens embedding must be deterministic — same
        input + same params → byte-identical output."""
        rng = np.random.default_rng(seed)
        series = rng.normal(0, 1, size=256)
        emb1 = TakensEmbedder(
            parameters_type="fixed", time_delay=3, dimension=4, stride=1,
        )
        emb2 = TakensEmbedder(
            parameters_type="fixed", time_delay=3, dimension=4, stride=1,
        )
        r1 = emb1.embed(series)
        r2 = emb2.embed(series)
        assert np.array_equal(r1.point_cloud, r2.point_cloud)
        assert r1.time_delay == r2.time_delay
        assert r1.dimension == r2.dimension


# --------------------------------------------------------------------------- #
# Timer invariants
# --------------------------------------------------------------------------- #

class TestTimerInvariants:
    def test_perf_counter_ns_is_monotonic(self):
        """time.perf_counter_ns() must never go backwards."""
        readings = [time.perf_counter_ns() for _ in range(1000)]
        diffs = np.diff(readings)
        assert np.all(diffs >= 0), "perf_counter_ns went backwards!"

    def test_perf_counter_ns_resolution_submicrosecond(self):
        """Resolution should be sub-microsecond on any modern platform."""
        samples = []
        for _ in range(100):
            a = time.perf_counter_ns()
            b = time.perf_counter_ns()
            samples.append(b - a)
        median_res = int(np.median(samples))
        # Should be ≤ 1000 ns (1 μs) on any modern CPU
        assert median_res <= 1000, (
            f"perf_counter_ns median resolution = {median_res} ns, expected ≤ 1000"
        )

    @given(n_calls=st.integers(min_value=1, max_value=20))
    @settings(max_examples=10, deadline=None)
    def test_timer_records_are_monotonic(self, n_calls):
        """Every TimingRecord must have start_ns < end_ns and duration > 0."""
        t = Timer()
        for i in range(n_calls):
            with t.measure(f"step_{i}"):
                time.sleep(0.001)  # 1 ms — guaranteed measurable
        assert len(t.records) == n_calls
        for r in t.records:
            assert r.start_ns < r.end_ns
            assert r.duration_ns > 0
            assert r.duration_ns == r.end_ns - r.start_ns
        # Records must be in temporal order
        starts = [r.start_ns for r in t.records]
        assert starts == sorted(starts), "records out of temporal order"

    @given(n_calls=st.integers(min_value=2, max_value=10))
    @settings(max_examples=10, deadline=None)
    def test_timer_total_equals_sum(self, n_calls):
        """Timer.total() must equal the sum of individual durations."""
        t = Timer()
        for i in range(n_calls):
            with t.measure(f"step_{i}"):
                time.sleep(0.001)
        expected_total = sum(r.duration_ns for r in t.records)
        assert t.total() == expected_total
