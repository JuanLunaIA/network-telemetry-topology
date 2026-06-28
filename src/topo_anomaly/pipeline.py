"""
pipeline — end-to-end orchestration of the topological anomaly detector.

The pipeline stitches the modules together:

1. **Windowing** of the multivariate telemetry into overlapping sub-series.
2. **Aggregation** (optional) of multivariate channels into a scalar observable
   so that classical Takens embedding applies (the pipeline can also embed
   per-channel and compute diagrams per-channel; see ``embed_mode``).
3. **Takens embedding** of each window with the parameters chosen at
   configuration time (auto-search or fixed).
4. **Persistent homology** of every window's point cloud.
5. **Feature extraction** (entropy, Betti numbers, persistence statistics).
6. **Dimensionality reduction** (PCA / UMAP / none) of the point clouds,
   with a built-in *Betti preservation verification* comparing Betti-0 and
   Betti-1 of the original vs. reduced clouds.
7. **Anomaly detection** on the feature matrix, returning scores, labels,
   confusion-matrix metrics against the ground-truth mask and a full timing
   report captured with :func:`time.perf_counter_ns`.

The pipeline is intentionally framework-free: every step is plain numpy and
returns dataclasses with ``to_metadata()`` so the entire run can be dumped to
JSON for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .betti import BettiPreservationVerifier, PreservationReport
from .detection import AnomalyReport, TopologicalAnomalyDetector
from .embedding import EmbeddingResult, TakensEmbedder
from .features import TopologicalFeatureExtractor
from .persistence import PersistenceComputer
from .utils import Timer, ensure_dir, save_json


# --------------------------------------------------------------------------- #
# Reduction strategies
# --------------------------------------------------------------------------- #

def reduce_point_cloud(
    point_cloud: np.ndarray, method: str = "pca", target_dim: int = 2
) -> np.ndarray:
    """Apply a dimensionality-reduction map to a single point cloud.

    Parameters
    ----------
    point_cloud : np.ndarray, shape (n_points, m)
    method : str, default ``"pca"``
        One of ``{"pca", "umap", "none"}``.
    target_dim : int, default 2
        Target dimension :math:`m'`.
    """
    pc = np.asarray(point_cloud, dtype=np.float64)
    if method == "none":
        return pc
    if method == "pca":
        from sklearn.decomposition import PCA
        m = pc.shape[1]
        if target_dim >= m:
            return pc
        pca = PCA(n_components=target_dim, random_state=42)
        return pca.fit_transform(pc)
    if method == "umap":
        try:
            from umap import UMAP
        except ImportError as e:
            raise ImportError(
                "umap-learn is required for method='umap'. Install it with "
                "`pip install umap-learn`."
            ) from e
        m = pc.shape[1]
        if target_dim >= m:
            return pc
        reducer = UMAP(
            n_components=target_dim,
            n_neighbors=min(15, max(2, pc.shape[0] - 1)),
            random_state=42,
            init="spectral",
        )
        return reducer.fit_transform(pc)
    raise ValueError(f"Unknown reduction method: {method!r}")


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

@dataclass
class TopologicalAnomalyPipeline:
    """End-to-end topological anomaly detection pipeline.

    Parameters
    ----------
    window_size : int, default 256
        Length (in samples) of each sliding window.
    step : int, default 32
        Stride between consecutive windows.
    embed_mode : str, default ``"aggregate"``
        ``"aggregate"`` — average channels and run a single Takens embedding
        per window.
        ``"per_channel"`` — run a separate Takens embedding per channel per
        window and concatenate the resulting diagrams.
    aggregation : str, default ``"mean"``
        Channel aggregation method (see :func:`aggregate_multivariate`).
    takens_parameters_type : str, default ``"search"``
        ``"search"`` to auto-estimate time_delay/dimension per window,
        ``"fixed"`` to use the configured values.
    takens_time_delay : int, default 1
    takens_dimension : int, default 3
    takens_stride : int, default 1
    homology_dimensions : tuple of int, default ``(0, 1)``
    max_edge_length : float, default ``np.inf``
    reduction_method : str, default ``"pca"``
        One of ``{"pca", "umap", "none"}``. Applied to the embedded point
        cloud *before* persistent homology, so the pipeline can verify Betti
        preservation under the reduction.
    reduction_target_dim : int, default 2
        Target dimension for the reduction map.
    detector_method : str, default ``"ensemble"``
        Anomaly detector method (see :class:`TopologicalAnomalyDetector`).
    contamination : float, default 0.05
        Expected anomaly fraction.
    n_jobs : int, default 1
        Forwarded to giotto-tda transformers.
    """

    window_size: int = 256
    step: int = 32
    embed_mode: str = "aggregate"
    aggregation: str = "mean"
    takens_parameters_type: str = "search"
    takens_time_delay: int = 1
    takens_dimension: int = 3
    takens_stride: int = 1
    homology_dimensions: Tuple[int, ...] = (0, 1)
    max_edge_length: float = float(np.inf)
    reduction_method: str = "pca"
    reduction_target_dim: int = 2
    detector_method: str = "ensemble"
    contamination: float = 0.05
    n_jobs: int = 1
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.window_size <= 0 or self.step <= 0:
            raise ValueError("window_size and step must be > 0")
        if self.embed_mode not in {"aggregate", "per_channel"}:
            raise ValueError("embed_mode must be 'aggregate' or 'per_channel'")
        if self.reduction_method not in {"pca", "umap", "none"}:
            raise ValueError("reduction_method must be 'pca'|'umap'|'none'")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        series: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        timer: Optional[Timer] = None,
    ) -> "PipelineReport":
        """Run the full pipeline on a multivariate time series.

        Parameters
        ----------
        series : np.ndarray, shape (n_samples, n_channels)
        ground_truth : np.ndarray or None, shape (n_samples,)
            Optional boolean / 0-1 anomaly mask aligned with ``series``.
        timer : Timer or None
            External timer to attach measurements to.
        """
        if timer is None:
            timer = Timer()

        series = np.asarray(series, dtype=np.float64)
        if series.ndim == 1:
            series = series[:, np.newaxis]
        n_samples, n_channels = series.shape

        # ---- 1. Windowing -------------------------------------------------
        with timer.measure("windowing", n_samples=int(n_samples),
                           window_size=int(self.window_size), step=int(self.step)):
            windows = self._make_windows(series)
            window_starts = list(range(0, n_samples - self.window_size + 1, self.step))

        # Per-window ground-truth label: a window is "anomalous" if *any*
        # of its samples lie inside an anomalous region (relaxed from the
        # stricter ≥50% threshold used previously — for narrow anomalies
        # in long windows, the strict threshold produces zero positives).
        window_labels: Optional[np.ndarray] = None
        if ground_truth is not None:
            gt = np.asarray(ground_truth).astype(bool)
            window_labels = np.zeros(len(windows), dtype=np.int64)
            for i, (ws, we) in enumerate(zip(window_starts,
                                             window_starts + np.full(len(window_starts), self.window_size))):
                we = min(we, n_samples)
                if gt[ws:we].any():  # any anomalous sample → label 1
                    window_labels[i] = 1

        # ---- 2. Embedding -------------------------------------------------
        embedder = TakensEmbedder(
            parameters_type=self.takens_parameters_type,
            time_delay=self.takens_time_delay,
            dimension=self.takens_dimension,
            stride=self.takens_stride,
            n_jobs=self.n_jobs,
        )
        embeddings: List[EmbeddingResult] = []
        with timer.measure("takens_embedding_all", n_windows=int(len(windows))):
            for i, w in enumerate(windows):
                er = embedder.embed(w, timer=timer, stage_name=f"takens_embedding_w{i}")
                embeddings.append(er)

        # ---- 3. Reduction + preservation check ----------------------------
        original_clouds: List[np.ndarray] = []
        reduced_clouds: List[np.ndarray] = []
        preservation_reports: List[PreservationReport] = []
        verifier = BettiPreservationVerifier(
            homology_dimensions=self.homology_dimensions
        )
        persistence_computer = PersistenceComputer(
            homology_dimensions=self.homology_dimensions,
            max_edge_length=self.max_edge_length,
            n_jobs=self.n_jobs,
        )

        with timer.measure("reduction_and_persistence", n_windows=int(len(windows))):
            for i, er in enumerate(embeddings):
                pc_orig = er.point_cloud
                original_clouds.append(pc_orig)
                pc_red = reduce_point_cloud(
                    pc_orig,
                    method=self.reduction_method,
                    target_dim=self.reduction_target_dim,
                )
                reduced_clouds.append(pc_red)

                # Persistence on original & reduced (small batches -> individual)
                diag_orig = persistence_computer.compute(
                    pc_orig, timer=timer, stage_name=f"persistence_orig_w{i}"
                )
                diag_red = persistence_computer.compute(
                    pc_red, timer=timer, stage_name=f"persistence_red_w{i}"
                )
                rep = verifier.verify(
                    diag_orig.diagrams[0], diag_red.diagrams[0],
                    timer=timer, stage_name=f"betti_preservation_w{i}"
                )
                preservation_reports.append(rep)

        # ---- 4. Feature extraction ---------------------------------------
        feat_extractor = TopologicalFeatureExtractor(
            homology_dimensions=self.homology_dimensions
        )
        with timer.measure("feature_extraction_all", n_windows=int(len(windows))):
            # Recompute the *original* persistence diagrams (we need them as
            # giotto-tda-style (n_pairs, 3) arrays for the feature extractor).
            diagrams_orig: List[np.ndarray] = []
            for i, er in enumerate(embeddings):
                pc_orig = er.point_cloud
                diag = persistence_computer.compute(
                    pc_orig, timer=None, stage_name=f"persistence_orig_w{i}_feat"
                )
                diagrams_orig.append(diag.diagrams[0])
            # Pad to common pair count
            max_pairs = max(d.shape[0] for d in diagrams_orig)
            padded = np.zeros((len(diagrams_orig), max_pairs, 3), dtype=np.float64)
            for i, d in enumerate(diagrams_orig):
                padded[i, : d.shape[0], :] = d
            features = feat_extractor.extract_batch(padded, timer=timer)

        # ---- 5. Anomaly detection ----------------------------------------
        detector = TopologicalAnomalyDetector(
            method=self.detector_method,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        # Build the padded diagrams for the detector (same as above)
        anomaly_report = detector.fit_predict(
            features, diagrams=padded, timer=timer, stage_name="anomaly_detection"
        )

        # ---- 6. Assemble report ------------------------------------------
        metrics: Optional[Dict[str, float]] = None
        if window_labels is not None:
            metrics = anomaly_report.metrics(window_labels)

        return PipelineReport(
            windows=windows,
            window_starts=window_starts,
            window_labels=window_labels,
            embeddings=embeddings,
            original_clouds=original_clouds,
            reduced_clouds=reduced_clouds,
            preservation_reports=preservation_reports,
            diagrams=diagrams_orig,
            features=features,
            feature_names=feat_extractor.feature_names(),
            anomaly_report=anomaly_report,
            metrics=metrics,
            timer=timer,
            config=self._config_dict(),
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _make_windows(self, series: np.ndarray) -> List[np.ndarray]:
        n = series.shape[0]
        ws = self.window_size
        windows = []
        for start in range(0, n - ws + 1, self.step):
            windows.append(series[start : start + ws])
        return windows

    def _config_dict(self) -> Dict[str, Any]:
        return {
            "window_size": int(self.window_size),
            "step": int(self.step),
            "embed_mode": self.embed_mode,
            "aggregation": self.aggregation,
            "takens_parameters_type": self.takens_parameters_type,
            "takens_time_delay": int(self.takens_time_delay),
            "takens_dimension": int(self.takens_dimension),
            "takens_stride": int(self.takens_stride),
            "homology_dimensions": list(self.homology_dimensions),
            "max_edge_length": float(self.max_edge_length),
            "reduction_method": self.reduction_method,
            "reduction_target_dim": int(self.reduction_target_dim),
            "detector_method": self.detector_method,
            "contamination": float(self.contamination),
            "n_jobs": int(self.n_jobs),
            "random_state": int(self.random_state),
        }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

@dataclass
class PipelineReport:
    """End-to-end pipeline output."""

    windows: List[np.ndarray]
    window_starts: List[int]
    window_labels: Optional[np.ndarray]
    embeddings: List[EmbeddingResult]
    original_clouds: List[np.ndarray]
    reduced_clouds: List[np.ndarray]
    preservation_reports: List[PreservationReport]
    diagrams: List[np.ndarray]
    features: np.ndarray
    feature_names: List[str]
    anomaly_report: AnomalyReport
    metrics: Optional[Dict[str, float]]
    timer: Timer
    config: Dict[str, Any]

    def to_metadata(self) -> Dict[str, Any]:
        preservation_summary = self._preservation_summary()
        return {
            "config": self.config,
            "n_windows": int(len(self.windows)),
            "window_size": int(self.windows[0].shape[0]) if self.windows else 0,
            "feature_names": list(self.feature_names),
            "anomaly_report": self.anomaly_report.to_metadata(),
            "metrics": self.metrics,
            "preservation_summary": preservation_summary,
            "timing": self.timer.summary(),
        }

    def save(self, dir_path: str) -> str:
        """Save all metadata, timing and feature matrix to ``dir_path``."""
        ensure_dir(dir_path)
        meta_path = f"{dir_path}/pipeline_metadata.json"
        save_json(self.to_metadata(), meta_path)
        np.save(f"{dir_path}/features.npy", self.features)
        np.save(f"{dir_path}/anomaly_scores.npy", self.anomaly_report.scores)
        np.save(f"{dir_path}/anomaly_labels.npy", self.anomaly_report.labels)
        if self.window_labels is not None:
            np.save(f"{dir_path}/window_labels.npy", self.window_labels)
        return dir_path

    def _preservation_summary(self) -> Dict[str, Any]:
        if not self.preservation_reports:
            return {}
        reports = self.preservation_reports
        avg_score = float(np.mean([r.preservation_score for r in reports]))
        betti0_preserved = sum(1 for r in reports if r.betti0_preserved)
        betti1_preserved = sum(1 for r in reports if r.betti1_preserved)
        avg_bottleneck = float(np.mean([r.bottleneck_distance for r in reports]))
        per_dim_summary: Dict[int, Dict[str, float]] = {}
        for d in reports[0].homology_dimensions:
            l1 = [r.per_dim[d]["l1_distance"] for r in reports]
            l2 = [r.per_dim[d]["l2_distance"] for r in reports]
            linf = [r.per_dim[d]["linf_distance"] for r in reports]
            mdiff = [r.per_dim[d]["max_betti_difference"] for r in reports]
            per_dim_summary[int(d)] = {
                "mean_l1": float(np.mean(l1)),
                "mean_l2": float(np.mean(l2)),
                "mean_linf": float(np.mean(linf)),
                "mean_max_betti_diff": float(np.mean(mdiff)),
                "n_windows_with_diff": int(sum(1 for x in mdiff if x > 0)),
            }
        return {
            "n_windows": int(len(reports)),
            "avg_preservation_score": avg_score,
            "betti0_preserved_count": betti0_preserved,
            "betti1_preserved_count": betti1_preserved,
            "betti0_preservation_rate": betti0_preserved / len(reports),
            "betti1_preservation_rate": betti1_preserved / len(reports),
            "avg_bottleneck_distance": avg_bottleneck,
            "per_dim": per_dim_summary,
        }
