"""Tests for topo_anomaly.data — synthetic telemetry generation & injection."""
from __future__ import annotations

import numpy as np
import pytest

from topo_anomaly.data import (
    AnomalySpec,
    SyntheticTelemetryGenerator,
    generate_bursty_traffic,
    inject_anomalies,
    specs_from_dict,
    specs_to_dict,
)


class TestSyntheticTelemetryGenerator:
    def test_default_shape(self):
        gen = SyntheticTelemetryGenerator(n_samples=512, n_channels=5)
        s = gen.generate()
        assert s.shape == (512, 5)
        assert s.dtype == np.float64

    def test_normalised_per_channel(self):
        s = SyntheticTelemetryGenerator(n_samples=256, n_channels=3).generate()
        # per-channel mean ≈ 0, std ≈ 1
        assert np.allclose(s.mean(axis=0), 0.0, atol=1e-9)
        assert np.allclose(s.std(axis=0), 1.0, atol=1e-6)

    def test_reproducible_with_seed(self):
        a = SyntheticTelemetryGenerator(seed=7).generate()
        b = SyntheticTelemetryGenerator(seed=7).generate()
        assert np.array_equal(a, b)

    def test_different_seed_differs(self):
        a = SyntheticTelemetryGenerator(seed=7).generate()
        b = SyntheticTelemetryGenerator(seed=8).generate()
        assert not np.array_equal(a, b)

    @pytest.mark.parametrize("n_channels", [1, 2, 3, 5, 8])
    def test_various_channel_counts(self, n_channels):
        s = SyntheticTelemetryGenerator(n_samples=128, n_channels=n_channels).generate()
        assert s.shape == (128, n_channels)
        assert np.all(np.isfinite(s))

    def test_channel_names(self):
        gen = SyntheticTelemetryGenerator(n_channels=3)
        names = gen.channel_names()
        assert len(names) == 3
        assert names[0] == "throughput_mbps"


class TestAnomalySpec:
    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError):
            AnomalySpec(kind="nope", start=0, end=10)

    def test_end_before_start_raises(self):
        with pytest.raises(ValueError):
            AnomalySpec(kind="spike", start=10, end=5)

    def test_valid_spec(self):
        s = AnomalySpec(kind="spike", start=0, end=10, channel=0, magnitude=3.0)
        assert s.kind == "spike"
        assert s.magnitude == 3.0


class TestInjectAnomalies:
    @pytest.fixture
    def series(self):
        return SyntheticTelemetryGenerator(n_samples=256, n_channels=3, seed=0).generate()

    def test_level_shift_modifies_series(self, series):
        spec = AnomalySpec(kind="level_shift", start=50, end=60, channel=0, magnitude=5.0)
        modified, mask = inject_anomalies(series, [spec])
        assert np.any(modified != series)
        assert mask[50:60].all()
        assert not mask[:50].any()

    def test_spike_centers_at_middle(self, series):
        spec = AnomalySpec(kind="spike", start=50, end=70, channel=0, magnitude=5.0)
        modified, mask = inject_anomalies(series, [spec])
        center = 60
        # delta should peak at center
        delta = modified[:, 0] - series[:, 0]
        assert np.argmax(delta) == center

    def test_variance_change(self, series):
        spec = AnomalySpec(kind="variance_change", start=10, end=30, channel=1, magnitude=4.0)
        modified, mask = inject_anomalies(series, [spec])
        # Variance should be larger in the anomalous window than baseline
        before = series[:10, 1].std()
        after = modified[10:30, 1].std()
        assert after > before

    def test_missing_zeros_window(self, series):
        spec = AnomalySpec(kind="missing", start=20, end=30, channel=2)
        modified, _ = inject_anomalies(series, [spec])
        assert np.all(modified[20:30, 2] == 0.0)

    def test_multiple_specs_overlap_mask(self, series):
        specs = [
            AnomalySpec(kind="spike", start=10, end=20, channel=0),
            AnomalySpec(kind="level_shift", start=15, end=25, channel=1),
        ]
        _, mask = inject_anomalies(series, specs)
        assert mask[10:25].all()
        assert not mask[5].any() and not mask[26].any()

    def test_channel_none_affects_all(self, series):
        spec = AnomalySpec(kind="level_shift", start=10, end=20, channel=None, magnitude=2.0)
        modified, _ = inject_anomalies(series, [spec])
        # all channels shifted
        for k in range(series.shape[1]):
            assert np.allclose(modified[10:20, k] - series[10:20, k], 2.0)


class TestGenerateBurstyTraffic:
    def test_returns_three_objects(self):
        series, mask, specs = generate_bursty_traffic(
            n_samples=512, n_channels=3, seed=42, anomaly_fraction=0.05
        )
        assert series.shape[0] == 512
        assert mask.shape == (512,)
        assert mask.dtype == bool
        assert isinstance(specs, list)
        assert len(specs) > 0

    def test_anomaly_fraction_reasonable(self):
        series, mask, specs = generate_bursty_traffic(
            n_samples=1024, n_channels=3, seed=42, anomaly_fraction=0.1
        )
        # should have non-trivial anomaly presence
        assert mask.sum() > 0
        assert mask.sum() < 1024


class TestSpecsRoundtrip:
    def test_dict_roundtrip(self):
        specs = [
            AnomalySpec(kind="spike", start=1, end=2, channel=0, magnitude=1.0),
            AnomalySpec(kind="level_shift", start=3, end=4, channel=None, magnitude=2.0),
        ]
        d = specs_to_dict(specs)
        specs2 = specs_from_dict(d)
        assert all(a.kind == b.kind and a.start == b.start and a.end == b.end
                   for a, b in zip(specs, specs2))
