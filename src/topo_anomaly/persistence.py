"""
persistence — Vietoris–Rips persistent homology via giotto-tda.

Mathematical background
-----------------------
Given a finite point cloud :math:`P \\subset \\mathbb{R}^m` and a radius
:math:`\\epsilon > 0`, the **Vietoris–Rips complex** :math:`\\mathrm{VR}_\\epsilon(P)`
is the simplicial complex whose *k*-simplices are exactly the subsets of
:math:`P` of size ``k+1`` with pairwise distance :math:`\\leq 2\\epsilon`.

A **filtration** is a nested family of simplicial complexes
:math:`K_{\\epsilon_0} \\subseteq K_{\\epsilon_1} \\subseteq \\dots` indexed
by an increasing scale parameter :math:`\\epsilon`. As :math:`\\epsilon` grows,
new simplices appear, possibly creating or merging connected components,
closing loops, filling voids, etc.

**Persistent homology** tracks the *birth* and *death* of every homology
class throughout the filtration. The result is a multiset of points
:math:`(b, d) \\in \\mathbb{R}^2 \\cup \\{\\infty\\}` called a **persistence
diagram**. The difference :math:`d-b` is the **persistence** of the feature;
high-persistence features are interpreted as genuine topological signal,
low-persistence ones as noise.

The k-th **Betti number** :math:`\\beta_k(\\epsilon)` counts the number of
independent k-dimensional holes alive at filtration value :math:`\\epsilon`:

* :math:`\\beta_0` — connected components,
* :math:`\\beta_1` — independent loops (1-cycles),
* :math:`\\beta_2` — voids (2-cycles).

This module wraps :class:`gtda.homology.VietorisRipsPersistence` so the rest of
the package can call it with a uniform interface and capture timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from gtda.homology import VietorisRipsPersistence

from .utils import Timer


@dataclass
class PersistenceResult:
    """Container for a persistence diagram."""

    diagrams: np.ndarray  # shape (n_point_clouds, n_pairs, 3) — (b, d, dim)
    max_edge_length: float
    homology_dimensions: Tuple[int, ...]
    timer_record: Optional[Dict[str, Any]] = None

    def to_metadata(self) -> Dict[str, Any]:
        n_pairs_per_dim = {
            int(d): int((self.diagrams[..., 2] == d).sum())
            for d in self.homology_dimensions
        }
        return {
            "max_edge_length": float(self.max_edge_length),
            "homology_dimensions": list(self.homology_dimensions),
            "diagrams_shape": list(self.diagrams.shape),
            "n_pairs_per_dim": n_pairs_per_dim,
            "timer_record": self.timer_record,
        }

    def persistence_pairs(self, dim: int) -> np.ndarray:
        """Return the ``(birth, death)`` pairs for homology dimension ``dim``."""
        mask = self.diagrams[0, :, 2] == dim
        return self.diagrams[0, mask, :2]


@dataclass
class PersistenceComputer:
    """Compute Vietoris–Rips persistent homology.

    Parameters
    ----------
    homology_dimensions : tuple of int, default ``(0, 1)``
        Which homology dimensions to compute. ``(0, 1, 2)`` adds voids.
    max_edge_length : float, default ``np.inf``
        Cutoff for the VR filtration. Smaller values shorten runtime.
    collapse_edges : bool, default False
        Use giotto-tda's edge-collapse acceleration (faster, slightly different
        diagrams).
    n_jobs : int or None, default 1
        Parallelism level passed to giotto-tda.
    """

    homology_dimensions: Tuple[int, ...] = (0, 1)
    max_edge_length: float = float(np.inf)
    collapse_edges: bool = False
    n_jobs: Optional[int] = 1

    def compute(
        self,
        point_clouds: np.ndarray,
        timer: Optional[Timer] = None,
        stage_name: str = "persistence",
    ) -> PersistenceResult:
        """Compute persistence diagrams for one or more point clouds.

        ``point_clouds`` may be either 2-D ``(n_points, m)`` (single cloud) or
        3-D ``(n_clouds, n_points, m)``.
        """
        point_clouds = np.asarray(point_clouds, dtype=np.float64)
        if point_clouds.ndim == 2:
            point_clouds = point_clouds[np.newaxis, :, :]

        vr = VietorisRipsPersistence(
            metric="euclidean",
            homology_dimensions=list(self.homology_dimensions),
            max_edge_length=self.max_edge_length,
            collapse_edges=self.collapse_edges,
            n_jobs=self.n_jobs,
        )

        if timer is not None:
            with timer.measure(
                stage_name,
                n_point_clouds=int(point_clouds.shape[0]),
                n_points=int(point_clouds.shape[1]),
                homology_dimensions=list(self.homology_dimensions),
            ):
                diagrams = vr.fit_transform(point_clouds)
        else:
            diagrams = vr.fit_transform(point_clouds)

        record = None
        if timer is not None and timer.records:
            record = timer.records[-1].to_dict()

        return PersistenceResult(
            diagrams=diagrams,
            max_edge_length=float(self.max_edge_length),
            homology_dimensions=tuple(self.homology_dimensions),
            timer_record=record,
        )


def filter_finite_pairs(diagram: np.ndarray) -> np.ndarray:
    """Drop pairs whose death is ``+inf`` (essential classes).

    Returns the finite (birth, death) sub-array.
    """
    finite_mask = np.isfinite(diagram[:, 1])
    return diagram[finite_mask]
