"""End-to-end tests for topo_anomaly.pipeline."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from topo_anomaly.pipeline import (
    PipelineReport,
    TopologicalAnomalyPipeline,
    reduce_point_cloud,
)


class TestReducePointCloud:
    def test_pca_reduces_dim(self):
        rng = np.random.default_rng(0)
        pc = rng.normal(0, 1, size=(50, 5))
        red = reduce_point_cloud(pc, method="pca", target_dim=2)
        assert red.shape == (50, 2)

    def test_none_returns_same(self):
        pc = np.random.randn(20, 3)
        red = reduce_point_cloud(pc, method="none", target_dim=2)
        assert np.array_equal(red, pc)

    def test_target_dim_ge_input_returns_same(self):
        pc = np.random.randn(20, 2)
        red = reduce_point_cloud(pc, method="pca", target_dim=2)
        assert red.shape == (20, 2)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            reduce_point_cloud(np.zeros((10, 3)), method="bad")


@pytest.fixture
def small_data():
    from topo_anomaly.data import generate_bursty_traffic
    series, mask, specs = generate_bursty_traffic(
        n_samples=512, n_channels=3, seed=42, anomaly_fraction=0.1
    )
    return series, mask, specs


@pytest.fixture
def fast_pipeline():
    return TopologicalAnomalyPipeline(
        window_size=128,
        step=64,
        takens_parameters_type="fixed",
        takens_time_delay=2,
        takens_dimension=3,
        homology_dimensions=(0, 1),
        max_edge_length=2.0,
        reduction_method="pca",
        reduction_target_dim=2,
        detector_method="ensemble",
        contamination=0.2,
    )


class TestTopologicalAnomalyPipeline:
    def test_init_validates_params(self):
        with pytest.raises(ValueError):
            TopologicalAnomalyPipeline(window_size=0)
        with pytest.raises(ValueError):
            TopologicalAnomalyPipeline(embed_mode="bad")
        with pytest.raises(ValueError):
            TopologicalAnomalyPipeline(reduction_method="bad")

    def test_run_returns_report(self, small_data, fast_pipeline):
        series, mask, _ = small_data
        report = fast_pipeline.run(series, ground_truth=mask)
        assert isinstance(report, PipelineReport)
        assert len(report.windows) > 0
        assert report.features.shape[0] == len(report.windows)
        assert report.anomaly_report.scores.shape[0] == len(report.windows)

    def test_run_with_ground_truth_produces_metrics(self, small_data, fast_pipeline):
        series, mask, _ = small_data
        report = fast_pipeline.run(series, ground_truth=mask)
        assert report.metrics is not None
        assert "precision" in report.metrics
        assert "recall" in report.metrics
        assert "f1" in report.metrics

    def test_preservation_reports_populated(self, small_data, fast_pipeline):
        series, mask, _ = small_data
        report = fast_pipeline.run(series, ground_truth=mask)
        assert len(report.preservation_reports) == len(report.windows)
        # B0 should be mostly preserved under PCA
        ps = report._preservation_summary()
        assert ps["betti0_preservation_rate"] >= 0.5

    def test_timing_records_present(self, small_data, fast_pipeline):
        series, mask, _ = small_data
        report = fast_pipeline.run(series, ground_truth=mask)
        names = [r.name for r in report.timer.records]
        assert "windowing" in names
        assert any("takens_embedding" in n for n in names)
        assert any("persistence_orig" in n for n in names)
        assert any("persistence_red" in n for n in names)
        assert any("betti_preservation" in n for n in names)
        assert "anomaly_detection" in names

    def test_metadata_structure(self, small_data, fast_pipeline):
        series, mask, _ = small_data
        report = fast_pipeline.run(series, ground_truth=mask)
        m = report.to_metadata()
        assert "config" in m
        assert "anomaly_report" in m
        assert "preservation_summary" in m
        assert "timing" in m

    def test_save_to_dir(self, small_data, fast_pipeline, tmp_path):
        series, mask, _ = small_data
        report = fast_pipeline.run(series, ground_truth=mask)
        out = report.save(str(tmp_path / "run1"))
        assert os.path.isfile(f"{out}/pipeline_metadata.json")
        assert os.path.isfile(f"{out}/features.npy")
        assert os.path.isfile(f"{out}/anomaly_scores.npy")
        assert os.path.isfile(f"{out}/anomaly_labels.npy")
        # JSON is valid
        with open(f"{out}/pipeline_metadata.json") as f:
            d = json.load(f)
        assert d["n_windows"] == len(report.windows)

    def test_no_reduction_preserves_b1_better(self, small_data):
        """With reduction_method='none', Betti-1 should also be preserved."""
        series, mask, _ = small_data
        pipe = TopologicalAnomalyPipeline(
            window_size=128, step=64,
            takens_parameters_type="fixed",
            takens_time_delay=2, takens_dimension=3,
            homology_dimensions=(0, 1),
            max_edge_length=2.0,
            reduction_method="none",
            reduction_target_dim=2,
            detector_method="robust_z",
            contamination=0.2,
        )
        report = pipe.run(series, ground_truth=mask)
        ps = report._preservation_summary()
        # With no reduction, B0 AND B1 should be fully preserved
        assert ps["betti0_preservation_rate"] == 1.0
        assert ps["betti1_preservation_rate"] == 1.0

    def test_1d_input_handled(self, fast_pipeline):
        # Pipeline should accept 1-D series by adding a channel dim
        s = np.sin(np.linspace(0, 10 * np.pi, 512))
        report = fast_pipeline.run(s)
        assert len(report.windows) > 0
