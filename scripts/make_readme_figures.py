"""
Generate README-friendly charts:
  - architecture.svg        : pipeline architecture diagram
  - scaling.svg             : wall-clock vs n_samples
  - reduction_compare.svg   : β-preservation + runtime for PCA / UMAP / none
  - timing_pie.svg          : per-stage timing share
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# Font fallback for CJK + symbols
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

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT = os.path.join(RESULTS, "readme_figures")
os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. Architecture diagram (SVG)
# --------------------------------------------------------------------------- #

def make_architecture_svg() -> None:
    """Hand-rolled SVG pipeline diagram (no external deps)."""
    svg = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 280" font-family="Inter, Noto Sans SC, DejaVu Sans, sans-serif" font-size="13">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#475569"/>
    </marker>
    <linearGradient id="g1" x1="0" x2="1">
      <stop offset="0" stop-color="#3b82f6"/>
      <stop offset="1" stop-color="#1d4ed8"/>
    </linearGradient>
    <linearGradient id="g2" x1="0" x2="1">
      <stop offset="0" stop-color="#10b981"/>
      <stop offset="1" stop-color="#047857"/>
    </linearGradient>
    <linearGradient id="g3" x1="0" x2="1">
      <stop offset="0" stop-color="#f59e0b"/>
      <stop offset="1" stop-color="#b45309"/>
    </linearGradient>
    <linearGradient id="g4" x1="0" x2="1">
      <stop offset="0" stop-color="#ef4444"/>
      <stop offset="1" stop-color="#991b1b"/>
    </linearGradient>
  </defs>

  <!-- Title -->
  <text x="550" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#0f172a">
    Topological Anomaly Detection Pipeline
  </text>

  <!-- Boxes -->
  <g>
    <rect x="10"  y="80" width="140" height="60" rx="8" fill="url(#g1)"/>
    <text x="80"  y="105" text-anchor="middle" fill="white" font-weight="bold">Telemetry</text>
    <text x="80"  y="125" text-anchor="middle" fill="#dbeafe" font-size="11">(n, c) array</text>

    <rect x="170" y="80" width="140" height="60" rx="8" fill="url(#g1)"/>
    <text x="240" y="105" text-anchor="middle" fill="white" font-weight="bold">Windowing</text>
    <text x="240" y="125" text-anchor="middle" fill="#dbeafe" font-size="11">W windows</text>

    <rect x="330" y="80" width="140" height="60" rx="8" fill="url(#g2)"/>
    <text x="400" y="105" text-anchor="middle" fill="white" font-weight="bold">Takens Embed</text>
    <text x="400" y="125" text-anchor="middle" fill="#d1fae5" font-size="11">τ, m auto-search</text>

    <rect x="490" y="80" width="140" height="60" rx="8" fill="url(#g3)"/>
    <text x="560" y="105" text-anchor="middle" fill="white" font-weight="bold">Reduction</text>
    <text x="560" y="125" text-anchor="middle" fill="#fef3c7" font-size="11">PCA / UMAP / none</text>

    <rect x="650" y="80" width="140" height="60" rx="8" fill="url(#g3)"/>
    <text x="720" y="105" text-anchor="middle" fill="white" font-weight="bold">VR Persistence</text>
    <text x="720" y="125" text-anchor="middle" fill="#fef3c7" font-size="11">H_0, H_1 diagrams</text>

    <rect x="810" y="80" width="140" height="60" rx="8" fill="url(#g4)"/>
    <text x="880" y="105" text-anchor="middle" fill="white" font-weight="bold">Feature Extract</text>
    <text x="880" y="125" text-anchor="middle" fill="#fee2e2" font-size="11">entropy, Betti</text>

    <rect x="970" y="80" width="120" height="60" rx="8" fill="url(#g4)"/>
    <text x="1030" y="105" text-anchor="middle" fill="white" font-weight="bold">Detector</text>
    <text x="1030" y="125" text-anchor="middle" fill="#fee2e2" font-size="11">ensemble</text>
  </g>

  <!-- Arrows between boxes -->
  <g stroke="#475569" stroke-width="2" fill="none" marker-end="url(#arrow)">
    <line x1="150" y1="110" x2="170" y2="110"/>
    <line x1="310" y1="110" x2="330" y2="110"/>
    <line x1="470" y1="110" x2="490" y2="110"/>
    <line x1="630" y1="110" x2="650" y2="110"/>
    <line x1="790" y1="110" x2="810" y2="110"/>
    <line x1="950" y1="110" x2="970" y2="110"/>
  </g>

  <!-- Betti preservation verifier box -->
  <g>
    <rect x="490" y="190" width="300" height="60" rx="8" fill="#1e293b" stroke="#334155" stroke-width="1.5"/>
    <text x="640" y="215" text-anchor="middle" fill="white" font-weight="bold">Betti Preservation Verifier</text>
    <text x="640" y="232" text-anchor="middle" fill="#cbd5e1" font-size="11">β₀ &amp; β₁ check + bottleneck distance</text>
  </g>

  <!-- Dashed arrows from VR Persistence & Reduction to verifier -->
  <g stroke="#64748b" stroke-width="1.5" fill="none" stroke-dasharray="5,3" marker-end="url(#arrow)">
    <line x1="560" y1="140" x2="580" y2="190"/>
    <line x1="720" y1="140" x2="700" y2="190"/>
  </g>

  <!-- perf_counter_ns label -->
  <text x="550" y="170" text-anchor="middle" fill="#0f172a" font-style="italic" font-size="11">
    time.perf_counter_ns() wraps every stage
  </text>
</svg>
"""
    out = os.path.join(OUT, "architecture.svg")
    with open(out, "w") as f:
        f.write(svg)
    print("  ✓ architecture.svg")


# --------------------------------------------------------------------------- #
# 2. Scaling chart
# --------------------------------------------------------------------------- #

def make_scaling_chart() -> None:
    scaling_path = os.path.join(RESULTS, "benchmarks", "scaling.json")
    with open(scaling_path) as f:
        scaling = json.load(f)
    runs = scaling["scaling"]
    sizes = [r["n_samples"] for r in runs]
    wall = [r["wall_clock_ns"] / 1e9 for r in runs]
    b0 = [r["preservation_summary"]["betti0_preservation_rate"] * 100 for r in runs]
    b1 = [r["preservation_summary"]["betti1_preservation_rate"] * 100 for r in runs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    # Left: runtime vs n_samples
    ax1.plot(sizes, wall, "o-", color="tab:blue", lw=2, ms=8, label="wall-clock (s)")
    ax1.set_xlabel("n_samples")
    ax1.set_ylabel("Wall-clock runtime (s)")
    ax1.set_title("Pipeline runtime vs data size")
    ax1.set_xticks(sizes)
    ax1.grid(True, alpha=0.3)
    for x, y in zip(sizes, wall):
        ax1.annotate(f"{y:.2f}s", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)
    ax1.legend(loc="upper left")

    # Right: Betti preservation vs n_samples
    ax2.plot(sizes, b0, "s-", color="tab:green", lw=2, ms=8, label=r"$\beta_0$ preserved")
    ax2.plot(sizes, b1, "D--", color="tab:red", lw=2, ms=8, label=r"$\beta_1$ preserved")
    ax2.set_xlabel("n_samples")
    ax2.set_ylabel("Preservation rate (%)")
    ax2.set_title("Betti preservation under PCA (R³→R²)")
    ax2.set_xticks(sizes)
    ax2.set_ylim(-5, 110)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="center right")

    out = os.path.join(OUT, "scaling.svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ scaling.svg")


# --------------------------------------------------------------------------- #
# 3. Reduction-method comparison chart
# --------------------------------------------------------------------------- #

def make_reduction_chart() -> None:
    red_path = os.path.join(RESULTS, "benchmarks", "reduction_methods.json")
    with open(red_path) as f:
        red = json.load(f)

    methods = ["none", "pca", "umap"]
    runtimes = [red[m]["total_runtime_ns"] / 1e9 for m in methods]
    b0 = [red[m]["betti0_preservation_rate"] * 100 for m in methods]
    b1 = [red[m]["betti1_preservation_rate"] * 100 for m in methods]
    bottlenecks = [red[m]["avg_bottleneck_distance"] for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    colors = ["#10b981", "#3b82f6", "#f59e0b"]

    # Runtime
    bars = axes[0].bar(methods, runtimes, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Runtime (s)")
    axes[0].set_title("Wall-clock runtime")
    for b, v in zip(bars, runtimes):
        axes[0].text(b.get_x() + b.get_width() / 2, v + max(runtimes) * 0.02,
                     f"{v:.2f}s", ha="center", fontsize=10)

    # Betti preservation
    x = np.arange(len(methods))
    w = 0.35
    axes[1].bar(x - w / 2, b0, w, label=r"$\beta_0$", color="#22c55e", edgecolor="black", linewidth=0.5)
    axes[1].bar(x + w / 2, b1, w, label=r"$\beta_1$", color="#ef4444", edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].set_ylabel("Preservation rate (%)")
    axes[1].set_title("Betti preservation")
    axes[1].set_ylim(0, 115)
    axes[1].legend(loc="upper right")
    for i, (b0v, b1v) in enumerate(zip(b0, b1)):
        axes[1].text(i - w / 2, b0v + 3, f"{b0v:.0f}%", ha="center", fontsize=9)
        axes[1].text(i + w / 2, b1v + 3, f"{b1v:.0f}%", ha="center", fontsize=9)

    # Bottleneck distance
    bars = axes[2].bar(methods, bottlenecks, color=colors, edgecolor="black", linewidth=0.5)
    axes[2].set_ylabel("Avg bottleneck distance")
    axes[2].set_title("Topological distance (lower = better)")
    for b, v in zip(bars, bottlenecks):
        axes[2].text(b.get_x() + b.get_width() / 2, v + max(bottlenecks) * 0.02,
                     f"{v:.3f}", ha="center", fontsize=10)

    out = os.path.join(OUT, "reduction_compare.svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ reduction_compare.svg")


# --------------------------------------------------------------------------- #
# 4. Timing pie chart
# --------------------------------------------------------------------------- #

def make_timing_pie() -> None:
    stage_path = os.path.join(RESULTS, "benchmarks", "stage_timing.json")
    with open(stage_path) as f:
        stage = json.load(f)
    totals = stage["stage_totals_ns"]
    # Group small stages into "other"
    sorted_items = sorted(totals.items(), key=lambda x: -x[1])
    threshold = sum(totals.values()) * 0.02  # 2% threshold
    main = [(k, v) for k, v in sorted_items if v >= threshold]
    other = sum(v for k, v in sorted_items if v < threshold)
    if other > 0:
        main.append(("other", other))

    labels = [k for k, _ in main]
    sizes = [v for _, v in main]
    colors = plt.cm.Set3(np.arange(len(main)))

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 9},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    ax.set_title("Per-stage runtime share (single 2048-sample run, perf_counter_ns)")
    out = os.path.join(OUT, "timing_pie.svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ timing_pie.svg")


def main() -> None:
    print("=== Generating README figures →", OUT, "===")
    make_architecture_svg()
    make_scaling_chart()
    make_reduction_chart()
    make_timing_pie()
    print(f"\nAll README figures saved to: {OUT}")


if __name__ == "__main__":
    main()
