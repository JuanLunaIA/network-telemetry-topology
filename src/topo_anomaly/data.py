"""
data — synthetic network-telemetry time-series generators with controllable
anomalies, plus an injection utility that lets the user spike arbitrary
sub-windows of an existing series with several canonical anomaly types.

The generators are deliberately lightweight (pure numpy) so the whole pipeline
can be exercised in tests in a few seconds on a laptop. Real telemetry feeds
can be dropped in by implementing the same interface — i.e. returning a
``(n_samples, n_channels)`` float array.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


CHANNEL_NAMES = (
    "throughput_mbps",
    "rtt_ms",
    "packet_loss_pct",
    "jitter_ms",
    "active_flows",
)


@dataclass
class SyntheticTelemetryGenerator:
    """Generate multivariate synthetic network-telemetry time series.

    The generator produces ``n_channels`` correlated channels. Each channel is
    built from

    * a slowly varying baseline (low-frequency sinusoid + drift),
    * a diurnal component (24h period sampled at ``sample_rate_hz``),
    * coloured AR(1) noise,
    * cross-channel coupling so the data is *not* trivially separable.

    Parameters
    ----------
    n_samples : int
        Number of time samples to produce.
    n_channels : int, default 5
        Number of telemetry channels.
    sample_rate_hz : float, default 1.0
        Sampling rate in Hz. Used to build the diurnal component.
    baseline_drift_std : float, default 0.05
        Std of the per-channel drift process.
    ar1_coef : float, default 0.7
        Autoregressive coefficient for the coloured noise.
    noise_std : float, default 0.1
        Std of the white-noise innovation.
    seed : int, default 42
        RNG seed for reproducibility.
    """

    n_samples: int = 2048
    n_channels: int = 5
    sample_rate_hz: float = 1.0
    baseline_drift_std: float = 0.05
    ar1_coef: float = 0.7
    noise_std: float = 0.1
    seed: int = 42

    def generate(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        n, c = self.n_samples, self.n_channels
        t = np.arange(n) / self.sample_rate_hz

        # 1) baselines per channel — different amplitudes & phases
        amps = rng.uniform(0.5, 1.5, size=c)
        phases = rng.uniform(0, 2 * np.pi, size=c)
        baseline = np.zeros((n, c))
        for k in range(c):
            # two harmonics + slow drift
            slow = 0.5 * amps[k] * np.sin(2 * np.pi * t / (3600.0) + phases[k])
            drift = np.cumsum(rng.normal(0, self.baseline_drift_std, size=n))
            drift -= drift.mean()
            baseline[:, k] = slow + 0.3 * drift

        # 2) diurnal (24h) component with per-channel shape
        diurnal = np.zeros((n, c))
        for k in range(c):
            diurnal[:, k] = 0.4 * np.sin(2 * np.pi * t / 86400.0 + phases[k])

        # 3) AR(1) coloured noise
        noise = np.zeros((n, c))
        for k in range(c):
            x = 0.0
            for i in range(n):
                x = self.ar1_coef * x + rng.normal(0, self.noise_std)
                noise[i, k] = x

        # 4) cross-channel coupling: throughput ↔ rtt ↔ packet_loss ↔ jitter ↔ flows
        # Each coupling is guarded so we degrade gracefully when n_channels < 5.
        coupled = np.zeros((n, c))
        if c >= 2:
            coupled[:, 1] = 0.3 * baseline[:, 0]   # rtt reacts to throughput
        if c >= 3:
            coupled[:, 2] = -0.2 * baseline[:, 0]  # loss inversely correlated
        if c >= 4:
            coupled[:, 3] = 0.4 * baseline[:, 1]   # jitter follows rtt
        if c >= 5:
            coupled[:, 4] = 0.6 * baseline[:, 0]   # active flows follow throughput

        series = baseline + diurnal + noise + coupled
        # global normalisation per channel for downstream TDA
        series = (series - series.mean(axis=0)) / (series.std(axis=0) + 1e-9)
        return series.astype(np.float64)

    def channel_names(self) -> List[str]:
        return list(CHANNEL_NAMES[: self.n_channels])


# --------------------------------------------------------------------------- #
# Anomaly injection
# --------------------------------------------------------------------------- #


@dataclass
class AnomalySpec:
    """Specification of one injected anomaly.

    Parameters
    ----------
    kind : str
        One of ``{"spike", "level_shift", "variance_change", "missing",
        "burst"}``.
    start : int
        Start sample (inclusive).
    end : int
        End sample (exclusive).
    channel : int or None
        Channel index, or ``None`` for all channels.
    magnitude : float
        Anomaly strength multiplier / shift size (interpretation depends on
        ``kind``).
    """

    kind: str
    start: int
    end: int
    channel: Optional[int] = None
    magnitude: float = 3.0

    def __post_init__(self) -> None:
        if self.kind not in {
            "spike",
            "level_shift",
            "variance_change",
            "missing",
            "burst",
        }:
            raise ValueError(f"Unknown anomaly kind: {self.kind!r}")
        if self.end <= self.start:
            raise ValueError("end must be > start")


def inject_anomalies(
    series: np.ndarray,
    specs: List[AnomalySpec],
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(modified_series, anomaly_mask)``.

    ``anomaly_mask`` is a 1-D boolean array of length ``len(series)`` marking
    samples that belong to at least one anomalous window.
    """
    rng = np.random.default_rng(seed)
    modified = series.copy()
    mask = np.zeros(series.shape[0], dtype=bool)
    for spec in specs:
        sl = slice(spec.start, spec.end)
        chans = (
            [spec.channel]
            if spec.channel is not None
            else list(range(series.shape[1]))
        )
        for k in chans:
            block = modified[sl, k]
            if spec.kind == "spike":
                # narrow spike — triangular shape
                width = max(1, (spec.end - spec.start) // 4)
                center = (spec.start + spec.end) // 2
                spike = np.zeros_like(block)
                for i, idx in enumerate(range(spec.start, spec.end)):
                    dist = abs(idx - center)
                    if dist <= width:
                        spike[i] = spec.magnitude * (1 - dist / width)
                modified[sl, k] = block + spike
            elif spec.kind == "level_shift":
                modified[sl, k] = block + spec.magnitude
            elif spec.kind == "variance_change":
                modified[sl, k] = block * spec.magnitude + rng.normal(
                    0, 0.1, size=block.shape
                )
            elif spec.kind == "missing":
                modified[sl, k] = 0.0
            elif spec.kind == "burst":
                modified[sl, k] = block + rng.normal(
                    0, spec.magnitude, size=block.shape
                )
        mask[spec.start: spec.end] = True
    return modified, mask


def generate_bursty_traffic(
    n_samples: int = 2048,
    n_channels: int = 5,
    seed: int = 42,
    anomaly_fraction: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, List[AnomalySpec]]:
    """Convenience helper: build a synthetic telemetry series with anomalies.

    Returns ``(series, anomaly_mask, specs)``.
    """
    gen = SyntheticTelemetryGenerator(
        n_samples=n_samples, n_channels=n_channels, seed=seed
    )
    series = gen.generate()

    rng = np.random.default_rng(seed + 1)
    n_anomalies = max(1, int(anomaly_fraction * n_samples / 50))
    specs: List[AnomalySpec] = []
    kinds = ["spike", "level_shift", "variance_change", "burst"]
    for _ in range(n_anomalies):
        kind = rng.choice(kinds)
        length = rng.integers(10, 60)
        start = rng.integers(0, n_samples - length)
        end = start + length
        channel = int(rng.integers(0, n_channels))
        magnitude = float(rng.uniform(2.0, 6.0))
        specs.append(
            AnomalySpec(kind=str(kind), start=int(start), end=int(end),
                        channel=channel, magnitude=magnitude)
        )

    modified, mask = inject_anomalies(series, specs, seed=seed + 2)
    return modified, mask, specs


def specs_to_dict(specs: List[AnomalySpec]) -> List[Dict]:
    return [
        {
            "kind": s.kind,
            "start": s.start,
            "end": s.end,
            "channel": s.channel,
            "magnitude": s.magnitude,
        }
        for s in specs
    ]


def specs_from_dict(d: List[Dict]) -> List[AnomalySpec]:
    return [AnomalySpec(**x) for x in d]
