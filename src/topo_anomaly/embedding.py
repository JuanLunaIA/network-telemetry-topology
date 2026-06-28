"""
embedding — Takens' delay-coordinate embedding for time-series.

Mathematical background
-----------------------
Given a scalar observable :math:`x(t)` sampled at uniform intervals and an
underlying dynamical system with state :math:`y(t) \\in \\mathbb{R}^d`, the
**Whitney–Takens embedding theorem** says that, *generically*, the delay
coordinate map

.. math::

    \\Phi(x)(t) = (x(t), x(t-\\tau), x(t-2\\tau), \\dots, x(t-(m-1)\\tau))
    \\in \\mathbb{R}^m

is an embedding of the attractor into :math:`\\mathbb{R}^m` whenever
:math:`m \\geq 2d+1`. In practice, the embedding dimension :math:`m` and the
time delay :math:`\\tau` are *not* known a-priori and must be estimated from
the data:

* :math:`\\tau` — typically chosen as the first zero crossing of the
  autocorrelation function, or the first minimum of the mutual information.
* :math:`m` — estimated via *false-nearest-neighbours* (FNN).

``giotto-tda`` exposes :class:`~gtda.time_series.SingleTakensEmbedding` which
performs both searches automatically. This module wraps it with a uniform
interface, returns the resulting point cloud, and records the chosen
parameters so they can be reported alongside the Betti curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from gtda.time_series import SingleTakensEmbedding, TakensEmbedding

from .utils import Timer


@dataclass
class EmbeddingResult:
    """Container for a Takens embedding output."""

    point_cloud: np.ndarray
    time_delay: int
    dimension: int
    stride: int
    parameters_type: str  # "search" or "fixed"
    timer_record: Optional[Dict[str, Any]] = None

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "time_delay": self.time_delay,
            "dimension": self.dimension,
            "stride": self.stride,
            "parameters_type": self.parameters_type,
            "n_points": int(self.point_cloud.shape[0]),
            "embedding_dim": int(self.point_cloud.shape[1]),
            "timer_record": self.timer_record,
        }


@dataclass
class TakensEmbedder:
    """Takens delay-coordinate embedder.

    Parameters
    ----------
    parameters_type : str, default ``"search"``
        ``"search"``  — auto-estimate ``time_delay`` and ``dimension`` per
        series using :class:`SingleTakensEmbedding`.
        ``"fixed"``   — use ``time_delay`` and ``dimension`` as given.
    time_delay : int, default 1
        Delay :math:`\\tau` in samples (used directly if ``parameters_type``
        is ``"fixed"``, used as a starting point if ``"search"``).
    dimension : int, default 3
        Embedding dimension :math:`m`. Same comment as above.
    stride : int, default 1
        Stride between consecutive embedded vectors.
    n_jobs : int or None, default 1
        ``n_jobs`` passed to the underlying giotto-tda transformer.
    """

    parameters_type: str = "search"
    time_delay: int = 1
    dimension: int = 3
    stride: int = 1
    n_jobs: Optional[int] = 1

    def __post_init__(self) -> None:
        if self.parameters_type not in {"search", "fixed"}:
            raise ValueError("parameters_type must be 'search' or 'fixed'")

    def embed(self, series: np.ndarray, timer: Optional[Timer] = None,
              stage_name: str = "takens_embedding") -> EmbeddingResult:
        """Embed a 1-D (or aggregated 1-D) time series.

        If a 2-D ``(n_samples, n_channels)`` array is supplied, the channels
        are aggregated by default via ``mean`` (override with
        ``self.aggregate_fn``) to produce a single observable suitable for
        classical Takens embedding. The original multivariate data can still
        be recovered from the embedding metadata.
        """
        series = np.asarray(series, dtype=np.float64)
        if series.ndim == 2:
            series = series.mean(axis=1)
        elif series.ndim != 1:
            raise ValueError(f"series must be 1-D or 2-D, got {series.ndim}-D")

        if self.parameters_type == "search":
            transformer = SingleTakensEmbedding(
                parameters_type="search",
                time_delay=self.time_delay,
                dimension=self.dimension,
                stride=self.stride,
                n_jobs=self.n_jobs,
            )
            if timer is not None:
                with timer.measure(stage_name, n_samples=int(series.shape[0])):
                    point_cloud = transformer.fit_transform(series)
                    td = int(transformer.time_delay_)
                    dim = int(transformer.dimension_)
            else:
                point_cloud = transformer.fit_transform(series)
                td = int(transformer.time_delay_)
                dim = int(transformer.dimension_)
            params_type = "search"
        else:
            transformer = TakensEmbedding(
                time_delay=self.time_delay,
                dimension=self.dimension,
                stride=self.stride,
            )
            # TakensEmbedding expects (n_series, n_samples) input shape
            series_2d = series[np.newaxis, :]
            if timer is not None:
                with timer.measure(stage_name, n_samples=int(series.shape[0])):
                    pc_3d = transformer.fit_transform(series_2d)
                    point_cloud = pc_3d[0]
            else:
                pc_3d = transformer.fit_transform(series_2d)
                point_cloud = pc_3d[0]
            td = int(self.time_delay)
            dim = int(self.dimension)
            params_type = "fixed"

        record = None
        if timer is not None and timer.records:
            record = timer.records[-1].to_dict()

        return EmbeddingResult(
            point_cloud=point_cloud,
            time_delay=td,
            dimension=dim,
            stride=int(self.stride),
            parameters_type=params_type,
            timer_record=record,
        )


def aggregate_multivariate(series: np.ndarray, method: str = "mean") -> np.ndarray:
    """Reduce a ``(n_samples, n_channels)`` array to 1-D for Takens embedding."""
    series = np.asarray(series, dtype=np.float64)
    if series.ndim == 1:
        return series
    method = method.lower()
    if method == "mean":
        return series.mean(axis=1)
    if method == "sum":
        return series.sum(axis=1)
    if method == "norm":
        return np.linalg.norm(series, axis=1)
    if method == "max":
        return series.max(axis=1)
    raise ValueError(f"Unknown aggregation method: {method!r}")
