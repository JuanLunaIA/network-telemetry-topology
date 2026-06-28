"""
Generate diagnostic plots from a saved pipeline run:
- persistence diagram (original vs reduced) for one window
- Betti curves overlay
- anomaly-score timeline
- timing breakdown bar chart
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# Use Noto Sans SC + DejaVu Sans for glyph fallback
try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf")
except Exception:
    pass
try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
except Exception:
    pass
plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from topo_anomaly import (  # noqa: E402
    BettiPreservationVerifier,
    PersistenceComputer,
    TakensEmbedder,
    generate_bursty_traffic,
)
from topo_anomaly.pipeline import reduce_point_cloud  # noqa: E402
from topo_anomaly.utils import ensure_dir  # noqa: E402


def plot_persistence_diagram(diag_orig: np.ndarray, diag_red: np.ndarray,
                              out_path: str, title_suffix: str = "") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for ax, d, name in zip(axes, [diag_orig, diag_red], ["Original", "Reduced"]):
        finite_mask = np.isfinite(d[:, 1])
        d = d[finite_mask]
        if d.shape[0] == 0:
            ax.text(0.5, 0.5, "no finite pairs", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"{name} persistence diagram")
            continue
        for dim_val, color, marker in [(0, "tab:blue", "o"), (1, "tab:orange", "s")]:
            mask = d[:, 2] == dim_val
            ax.scatter(d[mask, 0], d[mask, 1], c=color, marker=marker,
                       s=30, alpha=0.7, label=f"H{dim_val}", edgecolors="k", linewidths=0.3)
        # diagonal y = x
        lo = float(d[:, :2].min())
        hi = float(d[:, :2].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("birth")
        ax.set_ylabel("death")
        ax.set_title(f"{name} persistence diagram {title_suffix}".rstrip())
        ax.legend(loc="best")
        ax.set_aspect("equal", adjustable="box")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_betti_curves(rep, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, dim_val in zip(axes, [0, 1]):
        eps = rep.eps
        b_orig = rep.per_dim[dim_val]["beta_original"]
        b_red = rep.per_dim[dim_val]["beta_reduced"]
        ax.step(eps, b_orig, where="post", label="original", color="tab:blue", lw=1.5)
        ax.step(eps, b_red, where="post", label="reduced", color="tab:orange", lw=1.5, ls="--")
        ax.set_xlabel(r"filtration value $\epsilon$")
        ax.set_ylabel(rf"$\beta_{{{dim_val}}}(\epsilon)$")
        ax.set_title(rf"Betti-{dim_val} curve")
        ax.legend(loc="best")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_anomaly_timeline(scores: np.ndarray, labels: np.ndarray,
                           window_starts, window_size: int, n_samples: int,
                           ground_truth=None, out_path: str = None) -> None:
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    centers = np.array(window_starts) + window_size / 2
    ax.plot(centers, scores, "-o", color="tab:blue", lw=1.0, ms=3, label="anomaly score")
    anomaly_centers = centers[labels.astype(bool)]
    anomaly_scores = scores[labels.astype(bool)]
    ax.scatter(anomaly_centers, anomaly_scores, color="tab:red", s=80,
               zorder=5, label="predicted anomaly", edgecolors="k", linewidths=0.5)
    if ground_truth is not None:
        gt = np.asarray(ground_truth).astype(bool)
        # Vertical spans
        in_anom = False
        start = 0
        for i, g in enumerate(gt):
            if g and not in_anom:
                start = i
                in_anom = True
            elif not g and in_anom:
                ax.axvspan(start, i, alpha=0.15, color="tab:green", label="true anomaly" if start == 0 else None)
                in_anom = False
        if in_anom:
            ax.axvspan(start, len(gt), alpha=0.15, color="tab:green",
                       label="true anomaly" if not any(True for _ in []) else None)
    ax.set_xlabel("sample index")
    ax.set_ylabel("anomaly score (normalised)")
    ax.set_title("Anomaly detection timeline")
    ax.set_xlim(0, n_samples)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_timing_breakdown(meta: dict, out_path: str) -> None:
    records = meta["timing"]["records"]
    # Aggregate by stage prefix
    stage_totals: dict = {}
    for r in records:
        prefix = r["name"].rsplit("_w", 1)[0] if "_w" in r["name"] else r["name"]
        stage_totals[prefix] = stage_totals.get(prefix, 0) + r["duration_ns"]
    sorted_stages = sorted(stage_totals.items(), key=lambda x: -x[1])
    labels = [k for k, _ in sorted_stages]
    values = np.array([v / 1e6 for _, v in sorted_stages])  # ms

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(labels))), constrained_layout=True)
    y = np.arange(len(labels))
    ax.barh(y, values, color="tab:blue")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("duration (ms, perf_counter_ns)")
    ax.set_title("Per-stage execution time")
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, f"{v:.2f} ms", va="center", fontsize=8)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/figures")
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--n-channels", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)
    print(f"=== Generating figures → {out_dir} ===")

    # Run a small pipeline to get a single detailed window
    series, mask, _ = generate_bursty_traffic(
        n_samples=args.n_samples, n_channels=args.n_channels, seed=args.seed
    )

    # Take first window
    window = series[:256, :]
    emb = TakensEmbedder(parameters_type="fixed", time_delay=2, dimension=3)
    er = emb.embed(window.mean(axis=1))
    pc_orig = er.point_cloud
    pc_red = reduce_point_cloud(pc_orig, method="pca", target_dim=2)

    pcomp = PersistenceComputer(homology_dimensions=(0, 1), max_edge_length=2.0)
    d_orig = pcomp.compute(pc_orig).diagrams[0]
    d_red = pcomp.compute(pc_red).diagrams[0]

    v = BettiPreservationVerifier(homology_dimensions=(0, 1), n_steps=50)
    rep = v.verify(d_orig, d_red)

    # 1. Persistence diagrams
    plot_persistence_diagram(d_orig, d_red,
                              f"{out_dir}/persistence_diagrams.png",
                              title_suffix="(window 0)")
    print("  ✓ persistence_diagrams.png")

    # 2. Betti curves
    plot_betti_curves(rep, f"{out_dir}/betti_curves.png")
    print("  ✓ betti_curves.png")

    # 3. Run full pipeline for timeline + timing
    from topo_anomaly import TopologicalAnomalyPipeline
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
    plot_anomaly_timeline(
        report.anomaly_report.scores, report.anomaly_report.labels,
        report.window_starts, pipe.window_size, args.n_samples,
        ground_truth=mask, out_path=f"{out_dir}/anomaly_timeline.png",
    )
    print("  ✓ anomaly_timeline.png")

    # 4. Timing breakdown
    meta = report.to_metadata()
    plot_timing_breakdown(meta, f"{out_dir}/timing_breakdown.png")
    print("  ✓ timing_breakdown.png")

    # Also save the metadata
    with open(f"{out_dir}/pipeline_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print("  ✓ pipeline_metadata.json")

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
