# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-06-28

### Added
- **Core library** (`src/topo_anomaly/`)
  - `data.py` — synthetic multivariate network-telemetry generator with
    AR(1) noise, diurnal cycles, cross-channel coupling, and an anomaly
    injection utility (spike, level_shift, variance_change, missing, burst).
  - `embedding.py` — Takens delay-coordinate embedding wrapper around
    `gtda.time_series.SingleTakensEmbedding` / `TakensEmbedding`, with
    auto-search or fixed parameters, plus a multivariate-channel aggregator.
  - `persistence.py` — Vietoris–Rips persistent-homology wrapper around
    `gtda.homology.VietorisRipsPersistence`.
  - `betti.py` — Betti-curve computation, L1/L2/L∞ curve distances,
    bottleneck-distance wrapper, and a `BettiPreservationVerifier` that
    checks whether Betti-0 and Betti-1 survive a dimensionality-reduction map.
  - `features.py` — topological feature engineering (persistent entropy,
    persistence statistics, landscape proxies, max Betti) producing a
    fixed-length vector per window.
  - `detection.py` — anomaly scorer combining robust z-score, LOF and
    Wasserstein distance, with rank-averaged ensemble and confusion-matrix
    metrics against ground truth.
  - `pipeline.py` — end-to-end orchestrator (windowing → embedding →
    reduction → persistence → feature extraction → detection), with full
    `time.perf_counter_ns` timing at every stage.
  - `utils.py` — `Timer`, `TimingRecord`, JSON IO helpers, directory utils.
- **Tests** (`tests/`) — 113 unit & integration tests covering every module,
  including Betti-preservation tests on circles and clusters.
- **Scripts** (`scripts/`)
  - `run_pipeline.py` — runs the pipeline on synthetic telemetry and saves
    the full report.
  - `run_benchmarks.py` — measures `time.perf_counter_ns` resolution, per-stage
    timing, scaling, and reduction-method comparison.
  - `make_figures.py` — produces persistence diagrams, Betti curves, anomaly
    timeline and timing breakdown PNGs.
- **Project metadata** — `pyproject.toml`, `requirements.txt`, `LICENSE`,
  `.gitignore`, `CHANGELOG.md`, `CONTRIBUTING.md`, `README.md`.

### Theoretical foundations covered
- Whitney–Takens embedding theorem (delay-coordinate reconstruction).
- Vietoris–Rips filtration and simplicial complexes.
- Persistent homology and persistence diagrams.
- Betti numbers (β₀ = connected components, β₁ = loops).
- Persistent entropy and Betti-curve distances as topological summaries.
- Bottleneck distance for diagram comparison.
- Verification of topological-invariant preservation under PCA / UMAP
  dimensionality reduction.
