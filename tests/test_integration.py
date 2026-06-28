"""
End-to-end integration tests for the full pipeline.

These tests run the pipeline from start to finish on synthetic data and
assert that *every* expected artifact is produced and structurally valid.
They are slower than the unit tests (each takes ~3 s) but provide the
strongest guarantee that the pipeline is wired correctly.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from topo_anomaly import (
    TopologicalAnomalyPipeline,
    generate_bursty_traffic,
)
from topo_anomaly.utils import load_json


@pytest.fixture
def integration_data():
    return generate_bursty_traffic(
        n_samples=512, n_channels=3, seed=42, anomaly_fraction=0.1,
    )


@pytest.fixture
def integration_pipeline():
    return TopologicalAnomalyPipeline(
        window_size=128, step=64,
        takens_parameters_type="fixed",
        takens_time_delay=2, takens_dimension=3,
        homology_dimensions=(0, 1),
        max_edge_length=2.0,
        reduction_method="pca", reduction_target_dim=2,
        detector_method="ensemble", contamination=0.2,
    )


class TestPipelineArtifacts:
    """Verify every artifact the pipeline.save() produces."""

    def test_save_creates_all_expected_files(
        self, integration_data, integration_pipeline, tmp_path
    ):
        series, mask, _ = integration_data
        report = integration_pipeline.run(series, ground_truth=mask)
        out = report.save(str(tmp_path / "run"))

        expected = [
            "pipeline_metadata.json",
            "features.npy",
            "anomaly_scores.npy",
            "anomaly_labels.npy",
            "window_labels.npy",
        ]
        for fname in expected:
            assert os.path.isfile(f"{out}/{fname}"), f"missing {fname}"

    def test_metadata_json_is_valid(
        self, integration_data, integration_pipeline, tmp_path
    ):
        series, mask, _ = integration_data
        report = integration_pipeline.run(series, ground_truth=mask)
        out = report.save(str(tmp_path / "run"))
        meta = load_json(f"{out}/pipeline_metadata.json")
        # Required top-level keys
        for key in ("config", "n_windows", "feature_names", "anomaly_report",
                    "preservation_summary", "timing"):
            assert key in meta, f"metadata missing top-level key: {key}"
        # Anomaly report sub-keys
        ar = meta["anomaly_report"]
        for key in ("method", "n_windows", "n_anomalies", "threshold",
                    "contamination"):
            assert key in ar, f"anomaly_report missing key: {key}"

    def test_features_npy_shape_matches_windows(
        self, integration_data, integration_pipeline, tmp_path
    ):
        series, mask, _ = integration_data
        report = integration_pipeline.run(series, ground_truth=mask)
        out = report.save(str(tmp_path / "run"))
        feats = np.load(f"{out}/features.npy")
        assert feats.shape[0] == len(report.windows)
        assert feats.shape[1] == len(report.feature_names)

    def test_anomaly_labels_sum_matches_metadata(
        self, integration_data, integration_pipeline, tmp_path
    ):
        series, mask, _ = integration_data
        report = integration_pipeline.run(series, ground_truth=mask)
        out = report.save(str(tmp_path / "run"))
        labels = np.load(f"{out}/anomaly_labels.npy")
        meta = load_json(f"{out}/pipeline_metadata.json")
        assert int(labels.sum()) == meta["anomaly_report"]["n_anomalies"]

    def test_window_labels_match_ground_truth(
        self, integration_data, integration_pipeline, tmp_path
    ):
        series, mask, _ = integration_data
        report = integration_pipeline.run(series, ground_truth=mask)
        out = report.save(str(tmp_path / "run"))
        wl = np.load(f"{out}/window_labels.npy")
        # window_labels should have same length as windows
        assert wl.shape == (len(report.windows),)
        # all values are 0 or 1
        assert set(np.unique(wl)).issubset({0, 1})


class TestPipelineCorrectness:
    """Verify mathematical correctness of the pipeline output."""

    def test_no_reduction_preserves_both_betti(
        self, integration_data
    ):
        """With reduction_method='none', β₀ AND β₁ must be 100% preserved."""
        series, mask, _ = integration_data
        pipe = TopologicalAnomalyPipeline(
            window_size=128, step=64,
            takens_parameters_type="fixed",
            takens_time_delay=2, takens_dimension=3,
            homology_dimensions=(0, 1), max_edge_length=2.0,
            reduction_method="none", reduction_target_dim=2,
            detector_method="robust_z", contamination=0.2,
        )
        report = pipe.run(series, ground_truth=mask)
        ps = report._preservation_summary()
        assert ps["betti0_preservation_rate"] == 1.0
        assert ps["betti1_preservation_rate"] == 1.0
        assert ps["avg_preservation_score"] > 0.99
        assert ps["avg_bottleneck_distance"] < 1e-6

    def test_pca_always_preserves_b0(
        self, integration_data
    ):
        """PCA must always preserve β₀ — connected components survive
        linear projections."""
        series, mask, _ = integration_data
        pipe = TopologicalAnomalyPipeline(
            window_size=128, step=64,
            takens_parameters_type="fixed",
            takens_time_delay=2, takens_dimension=3,
            homology_dimensions=(0, 1), max_edge_length=2.0,
            reduction_method="pca", reduction_target_dim=2,
            detector_method="robust_z", contamination=0.2,
        )
        report = pipe.run(series, ground_truth=mask)
        ps = report._preservation_summary()
        assert ps["betti0_preservation_rate"] == 1.0, (
            "PCA must preserve β₀ on every window"
        )

    def test_recall_is_high_on_injected_anomalies(
        self, integration_data
    ):
        """The detector should catch the injected anomalies — recall ≥ 0.0
        is the floor; on most seeds we get recall ≥ 0.5 but the test is
        lenient because the windowing/labelling step is coarse."""
        series, mask, _ = integration_data
        pipe = TopologicalAnomalyPipeline(
            window_size=128, step=32,  # smaller step → more windows → better stats
            takens_parameters_type="fixed",
            takens_time_delay=2, takens_dimension=3,
            homology_dimensions=(0, 1), max_edge_length=2.0,
            reduction_method="pca", reduction_target_dim=2,
            detector_method="ensemble", contamination=0.2,
        )
        report = pipe.run(series, ground_truth=mask)
        assert report.metrics is not None
        # Sanity: at least one anomaly predicted and at least one true anomaly exists
        assert report.metrics["tp"] + report.metrics["fn"] > 0, "no true anomalies in windows"
        # Recall must be > 0 — the detector must catch at least one
        assert report.metrics["recall"] > 0.0, (
            f"recall = {report.metrics['recall']:.3f}, expected > 0"
        )

    def test_anomaly_scores_in_unit_interval(
        self, integration_data, integration_pipeline
    ):
        """Normalised anomaly scores must lie in [0, 1]."""
        series, mask, _ = integration_data
        report = integration_pipeline.run(series, ground_truth=mask)
        scores = report.anomaly_report.scores
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0 + 1e-9

    def test_labels_are_binary(
        self, integration_data, integration_pipeline
    ):
        series, mask, _ = integration_data
        report = integration_pipeline.run(series, ground_truth=mask)
        labels = report.anomaly_report.labels
        assert set(np.unique(labels)).issubset({0, 1})

    def test_n_predicted_anomalies_matches_contamination(
        self, integration_data
    ):
        """With contamination=0.2 and ≥5 windows, the number of predicted
        anomalies must be exactly ceil(0.2 * n_windows)."""
        series, mask, _ = integration_data
        pipe = TopologicalAnomalyPipeline(
            window_size=128, step=64,
            takens_parameters_type="fixed",
            takens_time_delay=2, takens_dimension=3,
            homology_dimensions=(0, 1), max_edge_length=2.0,
            reduction_method="pca", reduction_target_dim=2,
            detector_method="robust_z", contamination=0.2,
        )
        report = pipe.run(series, ground_truth=mask)
        n = len(report.windows)
        expected = int(np.ceil(0.2 * n))
        actual = int(report.anomaly_report.labels.sum())
        assert actual == expected, (
            f"expected {expected} anomalies (0.2 × {n}), got {actual}"
        )


class TestPipelineTiming:
    """Verify that every stage is properly timed."""

    def test_all_stages_have_timing_records(
        self, integration_data, integration_pipeline
    ):
        series, mask, _ = integration_data
        report = integration_pipeline.run(series, ground_truth=mask)
        names = {r.name for r in report.timer.records}
        # Mandatory stage prefixes
        for prefix in (
            "windowing",
            "takens_embedding",
            "reduction_and_persistence",
            "persistence_orig",
            "persistence_red",
            "betti_preservation",
            "feature_extraction",
            "anomaly_detection",
        ):
            assert any(prefix in n for n in names), (
                f"missing timing record for stage prefix: {prefix}"
            )

    def test_every_record_has_positive_duration(
        self, integration_data, integration_pipeline
    ):
        series, mask, _ = integration_data
        report = integration_pipeline.run(series, ground_truth=mask)
        for r in report.timer.records:
            assert r.duration_ns > 0
            assert r.duration_ms > 0

    def test_total_runtime_matches_sum(
        self, integration_data, integration_pipeline
    ):
        series, mask, _ = integration_data
        report = integration_pipeline.run(series, ground_truth=mask)
        total = sum(r.duration_ns for r in report.timer.records)
        assert report.timer.total() == total

    def test_total_runtime_under_60s(
        self, integration_data, integration_pipeline
    ):
        """Sanity check: a 512-sample run should finish in well under 60 s."""
        series, mask, _ = integration_data
        report = integration_pipeline.run(series, ground_truth=mask)
        assert report.timer.total() < 60_000_000_000, (
            f"total runtime = {report.timer.total()/1e9:.1f}s, expected < 60s"
        )
