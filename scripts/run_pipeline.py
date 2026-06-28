"""
Run the end-to-end pipeline on a synthetic telemetry stream, save the
report (metadata + features + labels) and produce a few diagnostic plots
(persistence diagram, Betti curves, anomaly timeline).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from topo_anomaly import (  # noqa: E402
    TopologicalAnomalyPipeline,
    generate_bursty_traffic,
)
from topo_anomaly.utils import ensure_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--n-channels", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anomaly-fraction", type=float, default=0.08)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--step", type=int, default=64)
    parser.add_argument("--reduction", default="pca", choices=["pca", "umap", "none"])
    parser.add_argument("--detector", default="ensemble",
                        choices=["robust_z", "lof", "ensemble", "wasserstein"])
    parser.add_argument("--out", default="results/runs/default")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)
    print(f"=== Pipeline run → {out_dir} ===")
    print(f"data: n_samples={args.n_samples}, n_channels={args.n_channels}, "
          f"anomaly_fraction={args.anomaly_fraction}")

    series, mask, specs = generate_bursty_traffic(
        n_samples=args.n_samples, n_channels=args.n_channels,
        seed=args.seed, anomaly_fraction=args.anomaly_fraction,
    )

    pipe = TopologicalAnomalyPipeline(
        window_size=args.window_size,
        step=args.step,
        takens_parameters_type="fixed",
        takens_time_delay=2,
        takens_dimension=3,
        homology_dimensions=(0, 1),
        max_edge_length=2.0,
        reduction_method=args.reduction,
        reduction_target_dim=2,
        detector_method=args.detector,
        contamination=args.anomaly_fraction,
    )

    report = pipe.run(series, ground_truth=mask)
    report.save(out_dir)

    print("\nResults:")
    print(f"  n_windows:           {len(report.windows)}")
    print(f"  anomalies predicted: {int(report.anomaly_report.labels.sum())}")
    if report.metrics:
        m = report.metrics
        print(f"  precision:           {m['precision']:.3f}")
        print(f"  recall:              {m['recall']:.3f}")
        print(f"  f1:                  {m['f1']:.3f}")
        print(f"  accuracy:            {m['accuracy']:.3f}")
    ps = report._preservation_summary()
    print(f"  β0 preservation:     {ps['betti0_preservation_rate']:.2%}")
    print(f"  β1 preservation:     {ps['betti1_preservation_rate']:.2%}")
    print(f"  avg preservation:    {ps['avg_preservation_score']:.3f}")
    print(f"  avg bottleneck:      {ps['avg_bottleneck_distance']:.4f}")
    print(f"  total runtime:       {report.timer.summary()['total_duration_human']}")

    print(f"\nAll artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
