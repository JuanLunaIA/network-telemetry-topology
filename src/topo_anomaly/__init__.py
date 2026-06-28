"""
topo_anomaly — Topological Anomaly Detection for Network Telemetry
==================================================================

A Python library that builds an end-to-end anomaly detection pipeline for
multivariate network-telemetry time series using tools from Topological Data
Analysis (TDA) provided by `giotto-tda <https://giotto-tda.readthedocs.io/>`_.

Theoretical pillars
-------------------
* **Takens' Embedding** — delay-coordinate reconstruction of an attractor from a
  scalar (or aggregated) time series so that its topology is preserved.
* **Simplicial Complexes & Filtration** — Vietoris–Rips complexes grown over a
  radius parameter produce a nested sequence of simplicial complexes whose
  topology changes with the scale.
* **Persistent Homology** — tracks the birth and death of topological features
  (connected components, loops, voids, ...) across the filtration, yielding a
  multi-scale topological descriptor called a *persistence diagram*.
* **Betti Numbers** — the rank of the homology groups ``H_0`` (connected
  components) and ``H_1`` (independent loops) at every filtration step.

Subpackages
-----------
``topo_anomaly.data``       — synthetic and real telemetry generators
``topo_anomaly.embedding``  — Takens embedding utilities
``topo_anomaly.persistence``— Vietoris–Rips persistent homology
``topo_anomaly.betti``      — Betti curve extraction and reduction-preservation
``topo_anomaly.features``   — topological feature engineering
``topo_anomaly.detection``  — anomaly scoring and detection
``topo_anomaly.pipeline``   — orchestration of the full pipeline
``topo_anomaly.utils``      — timing, IO and plotting helpers
"""

from .data import (
    SyntheticTelemetryGenerator,
    generate_bursty_traffic,
    inject_anomalies,
)
from .embedding import TakensEmbedder
from .persistence import PersistenceComputer
from .betti import BettiCurveComputer, BettiPreservationVerifier
from .features import TopologicalFeatureExtractor
from .detection import TopologicalAnomalyDetector
from .pipeline import TopologicalAnomalyPipeline
from .utils import Timer, save_json, load_json, ensure_dir

__version__ = "0.1.0"
__all__ = [
    "SyntheticTelemetryGenerator",
    "generate_bursty_traffic",
    "inject_anomalies",
    "TakensEmbedder",
    "PersistenceComputer",
    "BettiCurveComputer",
    "BettiPreservationVerifier",
    "TopologicalFeatureExtractor",
    "TopologicalAnomalyDetector",
    "TopologicalAnomalyPipeline",
    "Timer",
    "save_json",
    "load_json",
    "ensure_dir",
    "__version__",
]
