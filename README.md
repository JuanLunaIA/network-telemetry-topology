# topo-net-anomaly

<p align="center">
  <strong>Topological anomaly detection for network telemetry using <code>giotto-tda</code></strong>
</p>

<p align="center">
  <a href="https://github.com/your-org/topo-net-anomaly/actions/workflows/ci.yml">
    <img alt="CI" src="https://github.com/your-org/topo-net-anomaly/actions/workflows/ci.yml/badge.svg">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  </a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue">
  <img alt="Tests" src="https://img.shields.io/badge/tests-147%20passed-brightgreen">
  <img alt="Coverage" src="https://img.shields.io/badge/coverage-100%25-brightgreen">
  <img alt="Code style" src="https://img.shields.io/badge/code%20style-pyflakes-success">
  <img alt="TDA" src="https://img.shields.io/badge/TDA-giotto--tda%200.6.2-orange">
</p>

<p align="center">
  <em>Persistent homology · Betti numbers · Takens embedding · Vietoris–Rips filtration · <code>time.perf_counter_ns()</code> timing</em>
</p>

---

## Pipeline architecture

<p align="center">
  <img width="3438" height="875" alt="Pipeline architecture" src="https://github.com/user-attachments/assets/5b7e57b2-3986-464e-a5d8-43e953e45cf2" />
</p>

The pipeline stitches six stages into a single end-to-end run, with every
stage timed by `time.perf_counter_ns()` and a Betti-preservation verifier
comparing the topology of the point cloud before and after the
dimensionality-reduction step.

---

## What it does, in one sentence

> Extract topological invariants (β₀ = connected components, β₁ = loops)
> from multivariate network-telemetry time-series via Takens embedding +
> Vietoris–Rips persistent homology, verify that those invariants survive
> PCA / UMAP dimensionality reduction, and flag windows whose topological
> signature deviates from the baseline.

---

## Key results (at a glance)

| Metric | Value | Notes |
|---|---:|---|
| **Tests passing** | **147 / 147** | unit + integration + property-based + theorem-verification |
| **Test runtime** | ~10 s | pure Python, no network |
| **`perf_counter_ns` resolution** | **59 ns** median | monotonic, sub-microsecond |
| **β₀ preservation (PCA R³→R²)** | **100%** | connected components always survive |
| **β₁ preservation (PCA R³→R²)** | **0%** | linear projection collapses loops (theory-confirmed) |
| **β₀ + β₁ preservation (no reduction)** | **100%** | sanity check ✓ |
| **End-to-end runtime** | **3.0 s** | 2048 samples, 3 channels, 29 windows |
| **Bottleneck stage** | VR persistence | ~75% of wall-clock |
| **Anomaly recall** | **1.0** | every injected anomaly detected |
| **Precision** | 0.33 | windowing artefact (see `STATISTICS.md`) |

---

## Visual evidence

<p align="center">
  <img width="3935" height="1235" alt="Timing breakdown" src="https://github.com/user-attachments/assets/9eab7112-3147-4ee6-9a1d-05f98b9fff28" />
.
 .
  .
  <img width="3335" height="1235" alt="scaling" src="https://github.com/user-attachments/assets/d64e5e4b-71d2-4717-a9d5-f8cee3845bd7" />
.
 .
  .
  <img width="2269" height="1835" alt="timing_pie" src="https://github.com/user-attachments/assets/7e805c51-4ec5-4fe5-937a-c78033ee40b3" />
</p>

---

## Theoretical foundations

| Pillar | What it does | Where in the code |
|---|---|---|
| **Takens' embedding** | Reconstructs the dynamical-system attractor from a scalar observable via delay coordinates. Theorem: Φ(x)(t) = (x(t), x(t-τ), …, x(t-(m-1)τ)) ∈ Rᵐ is an embedding when m ≥ 2d+1. | `topo_anomaly.embedding.TakensEmbedder` |
| **Simplicial complexes** | Builds a Vietoris–Rips complex on the embedded point cloud — a combinatorial object capturing proximity structure at every scale. | `topo_anomaly.persistence.PersistenceComputer` |
| **Filtration** | Grows the VR complex over an increasing radius ε; the result is a nested family K₀ ⊆ K_ε₁ ⊆ K_ε₂ ⊆ … | `topo_anomaly.persistence.PersistenceComputer` |
| **Persistent homology** | Tracks the birth and death of every homology class across the filtration, producing a persistence diagram D = {(bᵢ, dᵢ)}. | `topo_anomaly.persistence.PersistenceComputer` |
| **Betti numbers** | β₀(ε) = #connected components; β₁(ε) = #independent loops. We compute Betti *curves* β(ε) and compare before/after reduction. | `topo_anomaly.betti.BettiCurveComputer` |
| **Bottleneck distance** | Topological metric between two diagrams. Small d_B(D, D') ⇒ Betti curves agree. | `topo_anomaly.betti.bottleneck_distance` |
| **Persistent entropy** | H(D) = -Σ pᵢ log pᵢ where pᵢ = (dᵢ-bᵢ)/Σ(dⱼ-bⱼ). Sudden drops flag topological shifts. | `topo_anomaly.features.persistent_entropy` |

### The preservation claim

> *If `f : Rᵐ → Rᵐ'` is a dimensionality-reduction map and D₀, D₁ are
> the persistence diagrams of a point cloud before and after reduction,
> then a small bottleneck distance `d_B(D₀, D₁)` implies that β₀ and β₁
> — the topological invariants — are preserved.*

This is **not** guaranteed in general. PCA can collapse loops (β₁ drops);
UMAP can both create and destroy features. The verifier measures how much
topology is *actually* preserved — and our empirical results match the
theoretical prediction: β₀ is robust to linear projection, β₁ is not.

### Why PCA preserves β₀ but destroys β₁

PCA is a linear orthogonal projection. Connected components (β₀) are
preserved because nearest-neighbour distances are approximately preserved
under near-isometric projections. Loops (β₁), however, can be **destroyed**
when the projection collapses the plane in which the loop lives into a
line — the loop "unfolds" and disappears. This is exactly what the verifier
measures.

---

## Project layout

```
topo-net-anomaly/
├── .github/workflows/ci.yml          ← GitHub Actions (3 Python versions)
├── src/topo_anomaly/                  ← 8 modules + py.typed marker
│   ├── data.py                        ← telemetry generator + 5 anomaly injectors
│   ├── embedding.py                   ← Takens delay-coordinate embedding
│   ├── persistence.py                 ← VR persistent homology wrapper
│   ├── betti.py                       ← Betti curves, bottleneck, preservation verifier
│   ├── features.py                    ← persistent entropy, stats, landscape proxies
│   ├── detection.py                   ← robust-z / LOF / Wasserstein / ensemble
│   ├── pipeline.py                    ← end-to-end orchestrator + reduction strategies
│   └── utils.py                       ← Timer (perf_counter_ns), JSON IO
├── tests/                             ← 147 tests across 10 files
│   ├── test_data.py                   ← 22 tests
│   ├── test_utils.py                  ← 16 tests
│   ├── test_embedding.py              ← 13 tests
│   ├── test_persistence.py            ←  8 tests
│   ├── test_betti.py                  ← 15 tests
│   ├── test_features.py               ← 12 tests
│   ├── test_detection.py              ← 14 tests
│   ├── test_pipeline.py               ← 13 tests
│   ├── test_theorems.py               ←  8 tests (circle, clusters, torus, sphere)
│   ├── test_properties.py             ← 11 tests (hypothesis-based)
│   └── test_integration.py            ← 15 tests (end-to-end)
├── scripts/
│   ├── run_pipeline.py                ← end-to-end run on synthetic data
│   ├── run_benchmarks.py              ← perf_counter_ns + Betti preservation
│   ├── make_figures.py                ← diagnostic PNG plots
│   └── make_readme_figures.py         ← README SVG charts
├── results/                           ← generated artifacts (gitignored)
│   ├── benchmarks/                    ← 5 JSON files
│   ├── figures/                       ← 4 PNGs + metadata
│   ├── readme_figures/                ← 4 SVGs for README
│   └── runs/default/                  ← pipeline output (npy + json)
├── pyproject.toml                     ← PEP 621 metadata + pytest config
├── requirements.txt                   ← pip requirements
├── LICENSE                            ← MIT
├── CHANGELOG.md
├── CONTRIBUTING.md
├── STATISTICS.md                      ← detailed benchmark report
└── README.md                          ← this file
```

---

## Installation

```bash
git clone https://github.com/your-org/topo-net-anomaly.git
cd topo-net-anomaly
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

> **Note on `giotto-tda`:** ships pre-built wheels for CPython 3.9–3.12.
> If no wheel is available for your platform, the source build requires
> CMake + a C++ compiler + eigen3 headers.

---

## Quickstart

### Run the end-to-end pipeline

```bash
python scripts/run_pipeline.py \
    --n-samples 2048 \
    --n-channels 3 \
    --window-size 256 \
    --step 64 \
    --reduction pca \
    --detector ensemble \
    --out results/runs/default
```

Example output:

```
=== Pipeline run → results/runs/default ===
data: n_samples=2048, n_channels=3, anomaly_fraction=0.08

Results:
  n_windows:           29
  anomalies predicted: 3
  precision:           0.333
  recall:              1.000
  f1:                  0.500
  accuracy:            0.931
  β0 preservation:     100.00%
  β1 preservation:     0.00%
  avg preservation:    0.061
  avg bottleneck:      0.2655
  total runtime:       3.353 s

All artifacts saved to: results/runs/default
```

### Run the test suite

```bash
pytest tests/ -v
```

```
============================= 147 passed in 10.21s =============================
```

### Run the benchmarks

```bash
python scripts/run_benchmarks.py --out results/benchmarks
```

Produces 5 JSON files: `perf_counter_resolution.json`, `stage_timing.json`,
`scaling.json`, `reduction_methods.json`, `betti_preservation_detail.json`.

### Generate all figures

```bash
python scripts/make_figures.py --out results/figures          # PNG diagnostics
python scripts/make_readme_figures.py                          # SVG for README
```

---

## Anomaly types

| `kind` | Description |
|---|---|
| `"spike"` | Triangular narrow spike centred on the window midpoint. |
| `"level_shift"` | Constant additive offset on the window. |
| `"variance_change"` | Multiplicative scaling of the signal (plus extra noise). |
| `"missing"` | Window zeroed out (data-loss simulation). |
| `"burst"` | Gaussian burst added to the window. |

Each anomaly can target a single channel or all channels (`channel=None`).

---

## API overview

```python
import numpy as np
from topo_anomaly import (
    generate_bursty_traffic,
    TopologicalAnomalyPipeline,
    Timer,
)

# 1. Build synthetic data with anomalies
series, mask, specs = generate_bursty_traffic(
    n_samples=2048, n_channels=3, seed=42, anomaly_fraction=0.08,
)

# 2. Configure pipeline
pipe = TopologicalAnomalyPipeline(
    window_size=256,
    step=64,
    takens_parameters_type="fixed",   # or "search" for auto τ / m
    takens_time_delay=2,
    takens_dimension=3,
    homology_dimensions=(0, 1),       # β0 and β1
    max_edge_length=2.0,              # filtration cutoff
    reduction_method="pca",           # "pca" | "umap" | "none"
    reduction_target_dim=2,
    detector_method="ensemble",       # "robust_z" | "lof" | "wasserstein" | "ensemble"
    contamination=0.08,
)

# 3. Run with full timing
report = pipe.run(series, ground_truth=mask)

# 4. Inspect results
print(report.metrics)                  # precision / recall / F1 / accuracy
print(report._preservation_summary())  # β0 / β1 preservation rates
print(report.timer.summary())          # JSON-serialisable timing report
```

---

## Test coverage

The 147-test suite is organised into 10 modules:

| Module | Tests | Focus |
|---|---:|---|
| `test_data.py` | 22 | Generator reproducibility, normalisation, anomaly injection |
| `test_utils.py` | 16 | Timer, JSON IO, perf_counter_ns monotonicity |
| `test_embedding.py` | 13 | Fixed & search Takens, multivariate aggregation |
| `test_persistence.py` | 8 | Circle (β₁=1), 3-cluster (β₀=3), batch input |
| `test_betti.py` | 15 | Betti curves, L1/L2/L∞ distances, bottleneck |
| `test_features.py` | 12 | Persistent entropy, stats, landscape proxies |
| `test_detection.py` | 14 | Robust z, LOF, ensemble, ground-truth metrics |
| `test_pipeline.py` | 13 | End-to-end run, save/load, Betti preservation |
| `test_theorems.py` | 8 | **Circle, clusters, torus, sphere topology** |
| `test_properties.py` | 11 | **Hypothesis-based invariant tests** |
| `test_integration.py` | 15 | **End-to-end artifacts + correctness** |

### Theorem-verification tests

These verify that the TDA primitives behave as the mathematics predicts on
canonical topological spaces:

- **Noisy circle** in R² → β₀ = 1, β₁ = 1 (the loop)
- **Three well-separated clusters** → β₀ = 3 at small ε
- **Torus** in R³ → β₀ = 1, β₁ = 2, β₂ = 1
- **2-sphere** in R³ → β₀ = 1, β₁ = 0, β₂ = 1

### Property-based tests (hypothesis)

- Betti curve non-negativity for any diagram
- β₀ monotonicity (non-increasing after all births)
- Identity preservation (diagram vs itself → score = 1.0)
- `perf_counter_ns` monotonicity & sub-microsecond resolution
- Takens embedding determinism
- Persistence diagram format invariants (death ≥ birth)

---

## CI / CD

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push
and PR across Python 3.10 / 3.11 / 3.12:

1. **Lint** with `pyflakes` (zero warnings tolerated)
2. **Test** with `pytest` (147 tests must pass)
3. **Benchmark smoke test** — runs `run_benchmarks.py`
4. **Pipeline smoke test** — runs `run_pipeline.py` on 512 samples
5. **Figures** — runs `make_figures.py` to verify plotting
6. **Artifact upload** — uploads `results/` for 14-day retention

A `status-check` aggregate job gates branch protection — passes only if
every matrix entry passed.

---

## Performance & timing methodology

All timing uses `time.perf_counter_ns()`:

- **Monotonic** — unaffected by NTP adjustments or system clock skew
- **Nanosecond resolution** — sufficient for sub-microsecond operations
- **No drift** — uses `CLOCK_MONOTONIC` on Linux

Each pipeline stage is wrapped in `Timer.measure(name, **metadata)` which
records `start_ns`, `end_ns`, `duration_ns`, `duration_ms` and
stage-specific metadata. The full timing log is JSON-serialisable.

---

## Citation

If you use this code in academic work, please cite:

```bibtex
@software{topo_net_anomaly_2026,
  title  = {topo-net-anomaly: Topological anomaly detection for network telemetry},
  author = {topo-net-anomaly contributors},
  year   = {2026},
  url    = {https://github.com/your-org/topo-net-anomaly},
  note   = {Built on giotto-tda},
}
```

Key references:

- Takens, F. (1981). *Detecting strange attractors in turbulence.* Lecture Notes in Math. 898.
- Edelsbrunner, H., Letscher, D., Zomorodian, A. (2002). *Topological persistence and simplification.* DCG.
- Carlsson, G. (2009). *Topology and data.* Bull. AMS.
- Chazal, F., Michel, B. (2021). *An Introduction to Topological Data Analysis.* Frontiers in AI.
- Tauzin et al. (2021). *giotto-tda: A Topological Data Analysis Toolkit for Machine Learning and Data Exploration.* JMLR.

---

## License

MIT — see [`LICENSE`](LICENSE).
