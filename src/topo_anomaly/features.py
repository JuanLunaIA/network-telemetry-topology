"""
features — topological feature engineering from persistence diagrams.

Mathematical background
-----------------------
A persistence diagram :math:`D_k = \\{(b_i, d_i)\\}_{i=1}^{N_k}` for homology
dimension :math:`k` is converted to a fixed-length vector so that downstream
anomaly detectors can consume it. We provide four canonical summaries:

* **Persistence statistics** — count, mean, std, max of :math:`d_i - b_i`.
* **Persistent entropy** —

  .. math::

     H(D_k) = -\\sum_i p_i \\log p_i, \\quad
     p_i = \\frac{d_i - b_i}{\\sum_j (d_j - b_j)}.

  A uniform distribution of persistences maximises :math:`H`; sharp peaks
  collapse it. Sudden changes in :math:`H` over sliding windows flag changes
  in the topology of the underlying attractor.

* **Betti numbers** — :math:`\\beta_0` (connected components) and
  :math:`\\beta_1` (loops) summarised by their maximum across the filtration.

* **Landscape proxies** — min / max / mean of the persistence pair
  coordinates, lightweight surrogates for the full persistence landscape.

These features are extracted from every *windowed subseries* of the input
telemetry, producing a feature matrix :math:`F \\in \\mathbb{R}^{T \\times p}`
that can be fed to any classical outlier detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .betti import BettiCurveComputer
from .utils import Timer


def persistent_entropy(diagram: np.ndarray, homology_dim: int) -> float:
    """Compute the persistent entropy of a single homology dim of a diagram."""
    pairs_all = np.asarray(diagram)
    if pairs_all.size == 0:
        return 0.0
    mask = pairs_all[:, 2] == homology_dim
    finite_mask = np.isfinite(pairs_all[:, 1])
    pairs = pairs_all[mask & finite_mask]
    if pairs.shape[0] == 0:
        return 0.0
    persistences = pairs[:, 1] - pairs[:, 0]
    total = persistences.sum()
    if total <= 0:
        return 0.0
    p = persistences / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def persistence_stats(diagram: np.ndarray, homology_dim: int) -> Dict[str, float]:
    """Basic statistics of the persistences for one homology dim."""
    pairs_all = np.asarray(diagram)
    if pairs_all.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "max": 0.0, "sum": 0.0}
    mask = pairs_all[:, 2] == homology_dim
    finite_mask = np.isfinite(pairs_all[:, 1])
    pairs = pairs_all[mask & finite_mask]
    if pairs.shape[0] == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "max": 0.0, "sum": 0.0}
    persistences = pairs[:, 1] - pairs[:, 0]
    return {
        "count": int(persistences.shape[0]),
        "mean": float(persistences.mean()),
        "std": float(persistences.std()),
        "max": float(persistences.max()),
        "sum": float(persistences.sum()),
    }


def persistence_landscape_proxy(
    diagram: np.ndarray, homology_dim: int
) -> Dict[str, float]:
    """Lightweight landscape proxies: min/max/mean of birth, death, midlife."""
    pairs_all = np.asarray(diagram)
    if pairs_all.size == 0:
        return {
            "birth_min": 0.0, "birth_max": 0.0, "birth_mean": 0.0,
            "death_min": 0.0, "death_max": 0.0, "death_mean": 0.0,
            "midlife_min": 0.0, "midlife_max": 0.0, "midlife_mean": 0.0,
        }
    mask = pairs_all[:, 2] == homology_dim
    finite_mask = np.isfinite(pairs_all[:, 1])
    pairs = pairs_all[mask & finite_mask]
    if pairs.shape[0] == 0:
        return {
            "birth_min": 0.0, "birth_max": 0.0, "birth_mean": 0.0,
            "death_min": 0.0, "death_max": 0.0, "death_mean": 0.0,
            "midlife_min": 0.0, "midlife_max": 0.0, "midlife_mean": 0.0,
        }
    b = pairs[:, 0]
    d = pairs[:, 1]
    m = (b + d) / 2.0
    return {
        "birth_min": float(b.min()), "birth_max": float(b.max()), "birth_mean": float(b.mean()),
        "death_min": float(d.min()), "death_max": float(d.max()), "death_mean": float(d.mean()),
        "midlife_min": float(m.min()), "midlife_max": float(m.max()), "midlife_mean": float(m.mean()),
    }


@dataclass
class TopologicalFeatureExtractor:
    """Extract a fixed-length topological feature vector per window.

    Parameters
    ----------
    homology_dimensions : tuple of int, default ``(0, 1)``
        Dimensions to extract features for.
    n_steps : int, default 50
        Betti-curve grid resolution (used for max-Betti summary).
    """

    homology_dimensions: Tuple[int, ...] = (0, 1)
    n_steps: int = 50

    def feature_names(self) -> List[str]:
        names: List[str] = []
        for d in self.homology_dimensions:
            names.extend([
                f"H{d}_entropy",
                f"H{d}_count",
                f"H{d}_persistence_mean",
                f"H{d}_persistence_std",
                f"H{d}_persistence_max",
                f"H{d}_persistence_sum",
                f"H{d}_birth_mean",
                f"H{d}_death_mean",
                f"H{d}_midlife_mean",
                f"H{d}_max_betti",
            ])
        return names

    def extract(
        self,
        diagram: np.ndarray,
        timer: Optional[Timer] = None,
        stage_name: str = "feature_extraction",
    ) -> np.ndarray:
        """Extract a 1-D feature vector for one persistence diagram."""
        if timer is not None:
            with timer.measure(stage_name, homology_dimensions=list(self.homology_dimensions)):
                feats = self._extract(diagram)
        else:
            feats = self._extract(diagram)
        return feats

    def extract_batch(
        self, diagrams: np.ndarray, timer: Optional[Timer] = None
    ) -> np.ndarray:
        """Extract features for a batch of diagrams.

        ``diagrams`` has shape ``(n_windows, n_pairs, 3)`` (giotto-tda format).
        Returns a 2-D matrix ``(n_windows, n_features)``.
        """
        if timer is not None:
            with timer.measure(
                "feature_extraction_batch",
                n_windows=int(diagrams.shape[0]),
                homology_dimensions=list(self.homology_dimensions),
            ):
                feats = np.stack([self._extract(diagrams[i]) for i in range(diagrams.shape[0])])
        else:
            feats = np.stack([self._extract(diagrams[i]) for i in range(diagrams.shape[0])])
        return feats

    def _extract(self, diagram: np.ndarray) -> np.ndarray:
        feats: List[float] = []
        bcc = BettiCurveComputer(
            homology_dimensions=self.homology_dimensions, n_steps=self.n_steps
        )
        bettis = bcc._compute(diagram)
        for i, d in enumerate(self.homology_dimensions):
            ent = persistent_entropy(diagram, d)
            stats = persistence_stats(diagram, d)
            land = persistence_landscape_proxy(diagram, d)
            _, beta = bettis[i]
            feats.extend([
                ent,
                float(stats["count"]),
                stats["mean"],
                stats["std"],
                stats["max"],
                stats["sum"],
                land["birth_mean"],
                land["death_mean"],
                land["midlife_mean"],
                float(beta.max()),
            ])
        return np.asarray(feats, dtype=np.float64)
