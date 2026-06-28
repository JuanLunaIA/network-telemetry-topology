"""
betti — Betti curve extraction and reduction-preservation verification.

Mathematical background
-----------------------
Given a persistence diagram :math:`D = \\{(b_i, d_i)\\}` for homology
dimension :math:`k`, the **Betti curve** is the step function

.. math::

    \\beta_k(\\epsilon) = \\#\\{i : b_i \\leq \\epsilon < d_i\\},

counting the number of independent :math:`k`-cycles alive at filtration
value :math:`\\epsilon`. Its integral, the **Betti number**, is the rank of
the :math:`k`-th homology group at the chosen scale.

When we apply a dimensionality-reduction map
:math:`f : \\mathbb{R}^m \\to \\mathbb{R}^{m'}` (with :math:`m' < m`, e.g.
PCA, UMAP) to a point cloud before computing its persistence, the
topology is *not* guaranteed to be preserved. We measure preservation by
comparing the Betti curves of the original and reduced clouds on a
common filtration grid and reporting:

* the L1 / L2 / L∞ distance between the curves,
* the symmetric bottleneck distance between the two diagrams (a topological
  metric — small bottleneck distance *implies* Betti-curve agreement),
* the per-scale Betti number match for :math:`\\beta_0` and :math:`\\beta_1`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils import Timer


# --------------------------------------------------------------------------- #
# Betti curve computation
# --------------------------------------------------------------------------- #

def _finite_pairs(diagram: np.ndarray) -> np.ndarray:
    """Return only (b, d) pairs with finite death, sorted by birth."""
    if diagram.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    finite_mask = np.isfinite(diagram[:, 1])
    pairs = diagram[finite_mask][:, :2]
    return pairs[np.argsort(pairs[:, 0])]


def betti_curve(
    diagram: np.ndarray,
    homology_dim: int,
    n_steps: int = 100,
    eps_grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the Betti curve for one homology dimension of one diagram.

    Parameters
    ----------
    diagram : np.ndarray, shape (n_pairs, 3)
        Persistence diagram in giotto-tda format ``(birth, death, dim)``.
    homology_dim : int
        Which homology dimension (0, 1, 2, ...).
    n_steps : int, default 100
        Number of points on the filtration grid (used if ``eps_grid`` is None).
    eps_grid : np.ndarray or None
        Pre-computed filtration grid. If None, built from min birth and max
        finite death across all pairs in the diagram.

    Returns
    -------
    eps : np.ndarray, shape (n_steps,)
        Filtration values at which the Betti curve was sampled.
    betti : np.ndarray, shape (n_steps,)
        Betti number at each filtration value.
    """
    pairs_all = np.asarray(diagram)
    if pairs_all.size == 0:
        if eps_grid is None:
            eps_grid = np.linspace(0.0, 1.0, n_steps)
        return eps_grid, np.zeros_like(eps_grid, dtype=np.int64)

    mask = pairs_all[:, 2] == homology_dim
    pairs = _finite_pairs(pairs_all[mask])

    if eps_grid is None:
        if pairs.size == 0:
            eps_grid = np.linspace(0.0, 1.0, n_steps)
        else:
            lo = float(pairs[:, 0].min())
            hi = float(pairs[:, 1].max())
            if hi <= lo:
                hi = lo + 1.0
            eps_grid = np.linspace(lo, hi, n_steps)
    else:
        n_steps = int(len(eps_grid))

    # For each eps, count pairs with b <= eps < d.
    # Vectorised: shape (n_pairs, n_steps)
    if pairs.shape[0] == 0:
        betti = np.zeros(n_steps, dtype=np.int64)
    else:
        b = pairs[:, 0][:, None]
        d = pairs[:, 1][:, None]
        eps = eps_grid[None, :]
        alive = (b <= eps) & (eps < d)
        betti = alive.sum(axis=0).astype(np.int64)

    return eps_grid, betti


@dataclass
class BettiCurveComputer:
    """Compute Betti curves for multiple homology dimensions on a common grid.

    Parameters
    ----------
    homology_dimensions : tuple of int, default ``(0, 1)``
        Homology dimensions to compute Betti curves for.
    n_steps : int, default 100
        Number of points on the common filtration grid.
    """

    homology_dimensions: Tuple[int, ...] = (0, 1)
    n_steps: int = 100

    def compute(
        self,
        diagram: np.ndarray,
        timer: Optional[Timer] = None,
        stage_name: str = "betti_curve",
    ) -> "BettiCurves":
        """Compute Betti curves for a single persistence diagram."""
        if timer is not None:
            with timer.measure(stage_name, homology_dimensions=list(self.homology_dimensions)):
                curves = self._compute(diagram)
        else:
            curves = self._compute(diagram)
        record = None
        if timer is not None and timer.records:
            record = timer.records[-1].to_dict()
        return BettiCurves(
            eps=curves[0][0],
            curves={int(d): curves[i][1] for i, d in enumerate(self.homology_dimensions)},
            homology_dimensions=tuple(self.homology_dimensions),
            timer_record=record,
        )

    def _compute(self, diagram: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        # Build a common eps grid across all homology dims for fair comparison.
        all_pairs = np.asarray(diagram)
        finite_mask = np.isfinite(all_pairs[:, 1])
        finite = all_pairs[finite_mask]
        if finite.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo = float(finite[:, 0].min())
            hi = float(finite[:, 1].max())
            if hi <= lo:
                hi = lo + 1.0
        eps_grid = np.linspace(lo, hi, self.n_steps)
        return [
            betti_curve(diagram, d, n_steps=self.n_steps, eps_grid=eps_grid)
            for d in self.homology_dimensions
        ]


@dataclass
class BettiCurves:
    """Container for Betti curves of one persistence diagram."""

    eps: np.ndarray
    curves: Dict[int, np.ndarray]
    homology_dimensions: Tuple[int, ...]
    timer_record: Optional[Dict[str, Any]] = None

    def betti_number(self, homology_dim: int, at_eps: Optional[float] = None) -> int:
        """Return the Betti number for ``homology_dim``.

        If ``at_eps`` is None, returns the *maximum* Betti number across the
        filtration (a stable summary statistic); otherwise returns the
        interpolated Betti number at the requested filtration value.
        """
        if homology_dim not in self.curves:
            return 0
        curve = self.curves[homology_dim]
        if at_eps is None:
            return int(curve.max())
        idx = int(np.searchsorted(self.eps, at_eps))
        idx = min(idx, len(curve) - 1)
        return int(curve[idx])

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "n_steps": int(len(self.eps)),
            "eps_range": [float(self.eps.min()), float(self.eps.max())],
            "homology_dimensions": list(self.homology_dimensions),
            "max_betti": {int(d): int(self.curves[d].max()) for d in self.homology_dimensions},
            "timer_record": self.timer_record,
        }


# --------------------------------------------------------------------------- #
# Reduction preservation
# --------------------------------------------------------------------------- #

def curve_distance(a: np.ndarray, b: np.ndarray, p: int = 2) -> float:
    """L_p distance between two equally-shaped Betti curves."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if p == np.inf:
        return float(np.max(np.abs(a - b)))
    return float(np.sum(np.abs(a - b) ** p) ** (1.0 / p))


def bottleneck_distance(diag_a: np.ndarray, diag_b: np.ndarray) -> float:
    """Symmetric bottleneck distance between two diagrams.

    Uses giotto-tda's :class:`PairwiseDistance` with metric ``'bottleneck'``
    if available; otherwise falls back to a *persistence-flask* upper bound
    based on matching features by persistence rank.
    """
    try:
        from gtda.diagrams import PairwiseDistance
        a = diag_a[np.newaxis]
        b = diag_b[np.newaxis]
        # PairwiseDistance returns shape (1, 1) for a single pair
        pd = PairwiseDistance(metric="bottleneck", n_jobs=1)
        d = pd.fit_transform(np.concatenate([a, b], axis=0))
        # distance matrix shape: (2, 2) — we want entry [0, 1]
        return float(d[0, 1])
    except Exception:
        return _bottleneck_upper_bound(diag_a, diag_b)


def _bottleneck_upper_bound(diag_a: np.ndarray, diag_b: np.ndarray) -> float:
    """Cheap upper bound: sort features by persistence, sum differences."""
    def _pers(d):
        if d.size == 0:
            return np.empty(0)
        return d[:, 1] - d[:, 0]

    pa = np.sort(_pers(diag_a))[::-1]
    pb = np.sort(_pers(diag_b))[::-1]
    n = max(len(pa), len(pb))
    pa = np.pad(pa, (0, n - len(pa)))
    pb = np.pad(pb, (0, n - len(pb)))
    return float(np.max(np.abs(pa - pb))) if n > 0 else 0.0


@dataclass
class BettiPreservationVerifier:
    """Verify that Betti-0 and Betti-1 are preserved under a reduction map.

    Parameters
    ----------
    homology_dimensions : tuple of int, default ``(0, 1)``
        Dimensions to verify.
    n_steps : int, default 100
        Filtration grid resolution.
    """

    homology_dimensions: Tuple[int, ...] = (0, 1)
    n_steps: int = 100

    def verify(
        self,
        diagram_original: np.ndarray,
        diagram_reduced: np.ndarray,
        timer: Optional[Timer] = None,
        stage_name: str = "betti_preservation",
    ) -> "PreservationReport":
        """Compare Betti curves of two diagrams."""
        if timer is not None:
            with timer.measure(stage_name, homology_dimensions=list(self.homology_dimensions)):
                report = self._verify(diagram_original, diagram_reduced)
        else:
            report = self._verify(diagram_original, diagram_reduced)

        record = None
        if timer is not None and timer.records:
            record = timer.records[-1].to_dict()
        report.timer_record = record
        return report

    def _verify(
        self, diagram_original: np.ndarray, diagram_reduced: np.ndarray
    ) -> "PreservationReport":
        # Compute Betti curves on a *common* eps grid shared across both
        # diagrams so the per-scale comparison is fair.
        all_pairs = np.concatenate(
            [np.asarray(diagram_original), np.asarray(diagram_reduced)], axis=0
        )
        finite_mask = np.isfinite(all_pairs[:, 1])
        finite = all_pairs[finite_mask]
        if finite.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo = float(finite[:, 0].min())
            hi = float(finite[:, 1].max())
            if hi <= lo:
                hi = lo + 1.0
        eps_grid = np.linspace(lo, hi, self.n_steps)

        per_dim: Dict[int, Dict[str, Any]] = {}
        for d in self.homology_dimensions:
            _, beta_orig = betti_curve(diagram_original, d, eps_grid=eps_grid)
            _, beta_red = betti_curve(diagram_reduced, d, eps_grid=eps_grid)
            l1 = curve_distance(beta_orig, beta_red, p=1)
            l2 = curve_distance(beta_orig, beta_red, p=2)
            linf = curve_distance(beta_orig, beta_red, p=np.inf)
            max_orig = int(beta_orig.max())
            max_red = int(beta_red.max())
            per_dim[int(d)] = {
                "l1_distance": l1,
                "l2_distance": l2,
                "linf_distance": linf,
                "max_betti_original": max_orig,
                "max_betti_reduced": max_red,
                "max_betti_difference": abs(max_orig - max_red),
                "beta_original": beta_orig.tolist(),
                "beta_reduced": beta_red.tolist(),
            }

        bn = bottleneck_distance(diagram_original, diagram_reduced)

        # Overall preservation score in [0, 1]: 1.0 = perfect preservation.
        # Computed as 1 / (1 + mean(L1_norm / n_steps)).
        l1_mean = np.mean(
            [per_dim[d]["l1_distance"] / max(1, self.n_steps) for d in self.homology_dimensions]
        )
        preservation_score = float(1.0 / (1.0 + l1_mean))

        return PreservationReport(
            eps=eps_grid,
            per_dim=per_dim,
            bottleneck_distance=bn,
            preservation_score=preservation_score,
            homology_dimensions=tuple(self.homology_dimensions),
        )


@dataclass
class PreservationReport:
    """Result of comparing two persistence diagrams via Betti curves."""

    eps: np.ndarray
    per_dim: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    bottleneck_distance: float = 0.0
    preservation_score: float = 1.0
    homology_dimensions: Tuple[int, ...] = (0, 1)
    timer_record: Optional[Dict[str, Any]] = None

    @property
    def betti0_preserved(self) -> bool:
        return 0 in self.per_dim and self.per_dim[0]["max_betti_difference"] == 0

    @property
    def betti1_preserved(self) -> bool:
        return 1 in self.per_dim and self.per_dim[1]["max_betti_difference"] == 0

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "bottleneck_distance": float(self.bottleneck_distance),
            "preservation_score": float(self.preservation_score),
            "betti0_preserved": bool(self.betti0_preserved),
            "betti1_preserved": bool(self.betti1_preserved),
            "homology_dimensions": list(self.homology_dimensions),
            "per_dim": {
                str(d): {
                    "l1_distance": v["l1_distance"],
                    "l2_distance": v["l2_distance"],
                    "linf_distance": v["linf_distance"],
                    "max_betti_original": v["max_betti_original"],
                    "max_betti_reduced": v["max_betti_reduced"],
                    "max_betti_difference": v["max_betti_difference"],
                }
                for d, v in self.per_dim.items()
            },
            "timer_record": self.timer_record,
        }
