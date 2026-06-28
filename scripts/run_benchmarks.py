"""
Benchmark — execution-time and Betti-preservation measurements.

Runs the pipeline across multiple configurations, records execution times
with ``time.perf_counter_ns`` and verifies Betti-0 / Betti-1 preservation
under the configured dimensionality-reduction map.

Outputs JSON + a matplotlib figure under ``results/benchmarks/``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

# Make src/ importable when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from topo_anomaly import (  # noqa: E402
    BettiPreservationVerifier,
    PersistenceComputer,
    TopologicalAnomalyPipeline,
    generate_bursty_traffic,
)
from topo_anomaly.utils import ensure_dir, save_json  # noqa: E402


def benchmark_stage_timing(n_samples: int = 2048, n_channels: int = 3,
                            seed: int = 42) -> Dict[str, Any]:
    """Run the pipeline once and report per-stage timing in nanoseconds."""
    series, mask, _ = generate_bursty_traffic(
        n_samples=n_samples, n_channels=n_channels, seed=seed
    )
    pipe = TopologicalAnomalyPipeline(
        window_size=256, step=64,
        takens_parameters_type="fixed",
        takens_time_delay=2, takens_dimension=3,
        homology_dimensions=(0, 1),
        max_edge_length=2.0,
        reduction_method="pca", reduction_target_dim=2,
        detector_method="ensemble", contamination=0.1,
    )
    report = pipe.run(series, ground_truth=mask)
    timing = report.timer.summary()

    # Aggregate by stage prefix
    stage_totals: Dict[str, int] = {}
    for r in report.timer.records:
        prefix = r.name.rsplit("_w", 1)[0] if "_w" in r.name else r.name
        stage_totals[prefix] = stage_totals.get(prefix, 0) + r.duration_ns
    return {
        "n_samples": int(n_samples),
        "n_channels": int(n_channels),
        "n_windows": int(len(report.windows)),
        "total_runtime_ns": int(timing["total_duration_ns"]),
        "total_runtime_human": timing["total_duration_human"],
        "stage_totals_ns": stage_totals,
        "all_records": timing["records"],
        "preservation_summary": report._preservation_summary(),
        "anomaly_metrics": report.metrics,
    }


def benchmark_scaling() -> Dict[str, Any]:
    """Run pipeline across increasing data sizes; report wall-clock + Betti."""
    sizes = [256, 512, 1024, 2048]
    results: List[Dict[str, Any]] = []
    for n in sizes:
        print(f"  → running scaling benchmark for n_samples={n}")
        t0 = time.perf_counter_ns()
        r = benchmark_stage_timing(n_samples=n, n_channels=3)
        wall = time.perf_counter_ns() - t0
        r["wall_clock_ns"] = int(wall)
        results.append(r)
    return {"scaling": results}


def benchmark_reduction_methods() -> Dict[str, Any]:
    """Compare PCA vs UMAP vs None for Betti preservation and timing."""
    series, mask, _ = generate_bursty_traffic(
        n_samples=1024, n_channels=3, seed=42
    )
    out: Dict[str, Any] = {}
    for method in ["none", "pca", "umap"]:
        print(f"  → reduction method: {method}")
        try:
            pipe = TopologicalAnomalyPipeline(
                window_size=256, step=64,
                takens_parameters_type="fixed",
                takens_time_delay=2, takens_dimension=3,
                homology_dimensions=(0, 1),
                max_edge_length=2.0,
                reduction_method=method,
                reduction_target_dim=2,
                detector_method="ensemble", contamination=0.1,
            )
            report = pipe.run(series, ground_truth=mask)
            ps = report._preservation_summary()
            out[method] = {
                "total_runtime_ns": int(report.timer.total()),
                "total_runtime_human": report.timer.summary()["total_duration_human"],
                "betti0_preservation_rate": ps["betti0_preservation_rate"],
                "betti1_preservation_rate": ps["betti1_preservation_rate"],
                "avg_preservation_score": ps["avg_preservation_score"],
                "avg_bottleneck_distance": ps["avg_bottleneck_distance"],
                "anomaly_metrics": report.metrics,
            }
        except Exception as e:
            out[method] = {"error": str(e)}
    return out


def benchmark_betti_preservation_detail() -> Dict[str, Any]:
    """For a single representative window, dump detailed Betti curves."""
    series, _, _ = generate_bursty_traffic(n_samples=512, n_channels=3, seed=42)
    # Take one window
    window = series[:256, :]
    from topo_anomaly.embedding import TakensEmbedder
    from topo_anomaly.pipeline import reduce_point_cloud

    emb = TakensEmbedder(parameters_type="fixed", time_delay=2, dimension=3)
    er = emb.embed(window.mean(axis=1))
    pc_orig = er.point_cloud
    pc_red = reduce_point_cloud(pc_orig, method="pca", target_dim=2)

    pcomp = PersistenceComputer(homology_dimensions=(0, 1), max_edge_length=2.0)
    d_orig = pcomp.compute(pc_orig).diagrams[0]
    d_red = pcomp.compute(pc_red).diagrams[0]

    v = BettiPreservationVerifier(homology_dimensions=(0, 1), n_steps=50)
    rep = v.verify(d_orig, d_red)

    return {
        "eps_grid": rep.eps.tolist(),
        "betti0_original": rep.per_dim[0]["beta_original"],
        "betti0_reduced": rep.per_dim[0]["beta_reduced"],
        "betti1_original": rep.per_dim[1]["beta_original"],
        "betti1_reduced": rep.per_dim[1]["beta_reduced"],
        "max_betti0_original": rep.per_dim[0]["max_betti_original"],
        "max_betti0_reduced": rep.per_dim[0]["max_betti_reduced"],
        "max_betti1_original": rep.per_dim[1]["max_betti_original"],
        "max_betti1_reduced": rep.per_dim[1]["max_betti_reduced"],
        "l1_distance_H0": rep.per_dim[0]["l1_distance"],
        "l1_distance_H1": rep.per_dim[1]["l1_distance"],
        "bottleneck_distance": rep.bottleneck_distance,
        "preservation_score": rep.preservation_score,
        "betti0_preserved": bool(rep.betti0_preserved),
        "betti1_preserved": bool(rep.betti1_preserved),
    }


def benchmark_perf_counter_resolution() -> Dict[str, Any]:
    """Measure the resolution of time.perf_counter_ns on this machine."""
    samples = []
    for _ in range(1000):
        a = time.perf_counter_ns()
        b = time.perf_counter_ns()
        samples.append(b - a)
    arr = np.asarray(samples, dtype=np.int64)
    return {
        "n_samples": int(arr.shape[0]),
        "min_resolution_ns": int(arr.min()),
        "median_resolution_ns": int(np.median(arr)),
        "max_resolution_ns": int(arr.max()),
        "mean_resolution_ns": float(arr.mean()),
        "std_resolution_ns": float(arr.std()),
        "note": ("time.perf_counter_ns uses CLOCK_MONOTONIC on Linux and "
                  "provides nanosecond resolution. Two consecutive calls "
                  "typically differ by a few ns."),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/benchmarks",
                        help="Output directory for benchmark results.")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)
    print(f"=== Benchmark output dir: {out_dir} ===")

    print("[1/4] time.perf_counter_ns resolution...")
    res = benchmark_perf_counter_resolution()
    save_json(res, f"{out_dir}/perf_counter_resolution.json")
    print(f"      median resolution: {res['median_resolution_ns']} ns")

    print("[2/4] Stage timing (single 2048-sample run)...")
    stage = benchmark_stage_timing(n_samples=2048, n_channels=3)
    save_json(stage, f"{out_dir}/stage_timing.json")
    print(f"      total runtime: {stage['total_runtime_human']}")
    print("      stage totals (ns):")
    for k, v in sorted(stage["stage_totals_ns"].items(), key=lambda x: -x[1]):
        print(f"        {k:<35s} {v:>15d}")

    print("[3/4] Scaling benchmark...")
    scaling = benchmark_scaling()
    save_json(scaling, f"{out_dir}/scaling.json")
    for r in scaling["scaling"]:
        ps = r["preservation_summary"]
        print(f"      n={r['n_samples']:<5d} windows={r['n_windows']:<3d} "
              f"runtime={r['total_runtime_human']:<10s} "
              f"β0_pres={ps['betti0_preservation_rate']:.2f} "
              f"β1_pres={ps['betti1_preservation_rate']:.2f}")

    print("[4/4] Reduction-method comparison...")
    red = benchmark_reduction_methods()
    save_json(red, f"{out_dir}/reduction_methods.json")
    for method, info in red.items():
        if "error" in info:
            print(f"      {method}: ERROR {info['error']}")
        else:
            print(f"      {method:<6s} runtime={info['total_runtime_human']:<10s} "
                  f"β0_pres={info['betti0_preservation_rate']:.2f} "
                  f"β1_pres={info['betti1_preservation_rate']:.2f} "
                  f"score={info['avg_preservation_score']:.3f}")

    print("\n[bonus] Detailed Betti preservation for one window...")
    detail = benchmark_betti_preservation_detail()
    save_json(detail, f"{out_dir}/betti_preservation_detail.json")
    print(f"      β0 original (max): {detail['max_betti0_original']}, "
          f"reduced: {detail['max_betti0_reduced']}, preserved: {detail['betti0_preserved']}")
    print(f"      β1 original (max): {detail['max_betti1_original']}, "
          f"reduced: {detail['max_betti1_reduced']}, preserved: {detail['betti1_preserved']}")
    print(f"      bottleneck distance: {detail['bottleneck_distance']:.6f}")

    print(f"\nAll benchmarks saved to: {out_dir}")


if __name__ == "__main__":
    main()
