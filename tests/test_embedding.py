"""Tests for topo_anomaly.embedding — Takens delay-coordinate embedding."""
from __future__ import annotations

import numpy as np
import pytest

from topo_anomaly.embedding import (
    EmbeddingResult,
    TakensEmbedder,
    aggregate_multivariate,
)


class TestAggregateMultivariate:
    def test_mean(self):
        s = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = aggregate_multivariate(s, "mean")
        assert np.allclose(out, [1.5, 3.5])

    def test_sum(self):
        s = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = aggregate_multivariate(s, "sum")
        assert np.allclose(out, [3.0, 7.0])

    def test_norm(self):
        s = np.array([[3.0, 4.0]])
        out = aggregate_multivariate(s, "norm")
        assert np.allclose(out, [5.0])

    def test_max(self):
        s = np.array([[1.0, 5.0, 2.0]])
        out = aggregate_multivariate(s, "max")
        assert np.allclose(out, [5.0])

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            aggregate_multivariate(np.zeros((5, 2)), "nope")

    def test_1d_pass_through(self):
        s = np.arange(10)
        out = aggregate_multivariate(s, "mean")
        assert np.array_equal(out, s)


class TestTakensEmbedder:
    @pytest.fixture
    def series(self):
        # Simple sinusoid — guaranteed clean attractor
        t = np.linspace(0, 20 * np.pi, 1024)
        return np.sin(t)

    def test_fixed_embedding_shape(self, series):
        emb = TakensEmbedder(
            parameters_type="fixed",
            time_delay=4,
            dimension=3,
            stride=1,
        )
        r = emb.embed(series)
        assert isinstance(r, EmbeddingResult)
        assert r.point_cloud.shape[1] == 3
        # expected n_points = n_samples - (dim - 1) * time_delay
        expected = len(series) - (3 - 1) * 4
        assert r.point_cloud.shape[0] == expected

    def test_search_embedding_runs(self, series):
        emb = TakensEmbedder(
            parameters_type="search",
            time_delay=1,
            dimension=3,
            stride=1,
        )
        r = emb.embed(series)
        assert r.point_cloud.ndim == 2
        assert r.point_cloud.shape[1] >= 2
        assert r.time_delay >= 1
        assert r.dimension >= 2

    def test_2d_input_aggregates(self):
        s = np.random.randn(256, 3)
        emb = TakensEmbedder(parameters_type="fixed", time_delay=2, dimension=3)
        r = emb.embed(s)
        assert r.point_cloud.ndim == 2

    def test_invalid_parameters_type(self):
        with pytest.raises(ValueError):
            TakensEmbedder(parameters_type="bad")

    def test_invalid_ndim_raises(self):
        emb = TakensEmbedder()
        with pytest.raises(ValueError):
            emb.embed(np.zeros((2, 2, 2)))

    def test_metadata(self, series):
        emb = TakensEmbedder(parameters_type="fixed", time_delay=4, dimension=3, stride=2)
        r = emb.embed(series)
        m = r.to_metadata()
        assert m["time_delay"] == 4
        assert m["dimension"] == 3
        assert m["stride"] == 2
        assert m["parameters_type"] == "fixed"
        assert m["embedding_dim"] == 3
        assert m["n_points"] > 0

    def test_records_timer_when_provided(self, series):
        from topo_anomaly.utils import Timer
        t = Timer()
        emb = TakensEmbedder(parameters_type="fixed", time_delay=4, dimension=3)
        r = emb.embed(series, timer=t, stage_name="my_embed")
        assert r.timer_record is not None
        assert r.timer_record["name"] == "my_embed"
