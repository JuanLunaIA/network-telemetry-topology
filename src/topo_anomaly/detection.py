"""
detection — anomaly scoring on topological feature matrices.

Mathematical background
-----------------------
After windowing the input telemetry, embedding each window via Takens,
computing its persistence diagram and extracting topological features, we
obtain a feature matrix :math:`F \\in \\mathbb{R}^{T \\times p}` whose rows
describe the topology of each window.

Anomaly detection then reduces to outlier detection on :math:`F`. We support
three complementary scorers:

* **Robust Z-score (MAD)** — uses median and median-absolute-deviation,
  robust to the very anomalies we are trying to detect:

  .. math::

     z_i = \\frac{|f_i - \\mathrm{median}(F)|}{1.4826 \\cdot \\mathrm{MAD}(F)}.

* **Local Outlier Factor (LOF)** — a density-based local outlier score from
  ``scikit-learn``; high LOF means the point lies in a sparser region than
  its neighbours.

* **Topological distance from baseline** — compute the Wasserstein distance
  between each window's persistence diagram and a *baseline* diagram (the
  median diagram or the first window); large distances flag topological
  shifts that classical outlier detectors might miss.

The final anomaly score is the *rank-averaged* combination of all available
scorers, which makes the detector robust to the failure mode of any single
scorer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .utils import Timer


def _robust_zscore(x: np.ndarray) -> np.ndarray:
    """Per-column robust z-score using median and MAD."""
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0)
    scaled_mad = 1.4826 * mad
    scaled_mad = np.where(scaled_mad < 1e-12, 1e-12, scaled_mad)
    return np.abs(x - med) / scaled_mad


def _rank_average(scores: List[np.ndarray]) -> np.ndarray:
    """Average the ranks of multiple 1-D score arrays."""
    from scipy.stats import rankdata
    n = len(scores[0])
    acc = np.zeros(n, dtype=np.float64)
    for s in scores:
        acc += rankdata(s)
    return acc / len(scores)


@dataclass
class TopologicalAnomalyDetector:
    """Topological anomaly detector over a feature matrix.

    Parameters
    ----------
    method : str, default ``"ensemble"``
        One of ``{"robust_z", "lof", "wasserstein", "ensemble"}``.
    contamination : float, default 0.05
        Expected fraction of anomalies. Used to threshold the final score.
    lof_n_neighbors : int, default 20
        ``n_neighbors`` for the LOF scorer.
    random_state : int, default 42
        RNG seed for reproducibility (used by LOF).
    """

    method: str = "ensemble"
    contamination: float = 0.05
    lof_n_neighbors: int = 20
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.method not in {"robust_z", "lof", "wasserstein", "ensemble"}:
            raise ValueError(
                f"method must be one of robust_z|lof|wasserstein|ensemble, got {self.method!r}"
            )
        if not (0.0 < self.contamination < 1.0):
            raise ValueError("contamination must be in (0, 1)")

    def fit_predict(
        self,
        features: np.ndarray,
        diagrams: Optional[np.ndarray] = None,
        timer: Optional[Timer] = None,
        stage_name: str = "anomaly_detection",
    ) -> "AnomalyReport":
        """Fit on ``features`` and return per-row anomaly scores + labels.

        Parameters
        ----------
        features : np.ndarray, shape (n_windows, n_features)
            Topological feature matrix from
            :class:`TopologicalFeatureExtractor`.
        diagrams : np.ndarray or None, shape (n_windows, n_pairs, 3)
            Optional persistence diagrams, required if ``method`` is
            ``"wasserstein"`` or ``"ensemble"`` (for the Wasserstein scorer).
        """
        if timer is not None:
            with timer.measure(stage_name, method=self.method, n_windows=int(features.shape[0])):
                report = self._fit_predict(features, diagrams)
        else:
            report = self._fit_predict(features, diagrams)
        record = None
        if timer is not None and timer.records:
            record = timer.records[-1].to_dict()
        report.timer_record = record
        return report

    def _fit_predict(
        self, features: np.ndarray, diagrams: Optional[np.ndarray]
    ) -> "AnomalyReport":
        scores: List[np.ndarray] = []
        score_names: List[str] = []

        if self.method in {"robust_z", "ensemble"}:
            z = _robust_zscore(features).max(axis=1)
            scores.append(z)
            score_names.append("robust_z")

        if self.method in {"lof", "ensemble"}:
            from sklearn.neighbors import LocalOutlierFactor
            n = features.shape[0]
            # LOF requires at least 3 samples (n_neighbors ≥ 2 and ≥1 neighbour
            # other than the query point). Skip if too few windows and we're
            # in ensemble mode (which has other scorers); raise if LOF-only.
            if n < 3:
                if self.method == "lof":
                    raise ValueError(
                        f"LOF requires ≥3 samples, got {n}. Use a smaller step "
                        f"or a different method."
                    )
            else:
                k = min(self.lof_n_neighbors, max(2, n - 1))
                lof = LocalOutlierFactor(n_neighbors=k, novelty=False)
                lof.fit(features)
                # negative_outlier_factor_ is more negative for outliers; negate
                lof_scores = -lof.negative_outlier_factor_
                scores.append(lof_scores)
                score_names.append("lof")

        if self.method in {"wasserstein", "ensemble"}:
            if diagrams is None:
                if self.method == "wasserstein":
                    raise ValueError("method='wasserstein' requires `diagrams`")
                # Fall back: skip Wasserstein in ensemble mode
            else:
                w = self._wasserstein_scores(diagrams)
                scores.append(w)
                score_names.append("wasserstein")

        final = _rank_average(scores) if len(scores) > 1 else scores[0]
        # Normalise to [0, 1] for interpretability
        if final.max() > final.min():
            final_norm = (final - final.min()) / (final.max() - final.min())
        else:
            final_norm = np.zeros_like(final)

        threshold = np.quantile(final_norm, 1.0 - self.contamination)
        labels = (final_norm >= threshold).astype(np.int64)

        per_scorer = {}
        for name, s in zip(score_names, scores):
            if s.max() > s.min():
                per_scorer[name] = ((s - s.min()) / (s.max() - s.min())).tolist()
            else:
                per_scorer[name] = np.zeros_like(s).tolist()

        return AnomalyReport(
            scores=final_norm,
            labels=labels,
            threshold=float(threshold),
            contamination=self.contamination,
            method=self.method,
            per_scorer=per_scorer,
            n_windows=int(features.shape[0]),
        )

    def _wasserstein_scores(self, diagrams: np.ndarray) -> np.ndarray:
        """Compute Wasserstein-2 distance from the median diagram per dim."""
        try:
            from gtda.diagrams import PairwiseDistance
        except ImportError:
            # Fallback: persistence-flask distance
            return self._persistence_distance_scores(diagrams)

        # PairwiseDistance expects (n_samples, n_features, 3) and returns
        # (n_samples, n_samples). We use dimension-2 wasserstein.
        try:
            pd = PairwiseDistance(metric="wasserstein", order=2.0, n_jobs=1)
            d_matrix = pd.fit_transform(diagrams)  # (T, T)
            # Reference: median row (sum of distances)
            ref = np.argmin(d_matrix.sum(axis=1))
            return d_matrix[ref]
        except Exception:
            return self._persistence_distance_scores(diagrams)

    def _persistence_distance_scores(self, diagrams: np.ndarray) -> np.ndarray:
        """Fallback scorer: distance between persistence-flask vectors."""
        n = diagrams.shape[0]
        # Build a feature vector per diagram: per-dim persistence stats
        from .features import persistence_stats
        mats = []
        for i in range(n):
            row = []
            for d in (0, 1):
                s = persistence_stats(diagrams[i], d)
                row.extend([s["count"], s["mean"], s["std"], s["max"], s["sum"]])
            mats.append(row)
        F = np.asarray(mats, dtype=np.float64)
        ref = np.median(F, axis=0)
        return np.linalg.norm(F - ref, axis=1)


@dataclass
class AnomalyReport:
    """Container for anomaly-detection results."""

    scores: np.ndarray
    labels: np.ndarray
    threshold: float
    contamination: float
    method: str
    per_scorer: Dict[str, List[float]] = field(default_factory=dict)
    n_windows: int = 0
    timer_record: Optional[Dict[str, Any]] = None

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "n_windows": int(self.n_windows),
            "n_anomalies": int(self.labels.sum()),
            "contamination": float(self.contamination),
            "threshold": float(self.threshold),
            "timer_record": self.timer_record,
        }

    def confusion(self, ground_truth: np.ndarray) -> Dict[str, int]:
        """Confusion matrix against a boolean / 0-1 ground-truth mask."""
        gt = np.asarray(ground_truth).astype(bool).astype(int)
        pred = self.labels.astype(int)
        tp = int(np.sum((gt == 1) & (pred == 1)))
        fp = int(np.sum((gt == 0) & (pred == 1)))
        tn = int(np.sum((gt == 0) & (pred == 0)))
        fn = int(np.sum((gt == 1) & (pred == 0)))
        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    def metrics(self, ground_truth: np.ndarray) -> Dict[str, float]:
        """Precision / recall / F1 / accuracy against ground truth."""
        c = self.confusion(ground_truth)
        tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / max(1, tp + fp + tn + fn)
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }
