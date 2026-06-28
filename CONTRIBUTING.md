# Contributing

Thanks for considering a contribution! This project is small and research-
oriented, so the bar is correspondingly light.

## Development setup

```bash
git clone https://github.com/your-org/topo-net-anomaly.git
cd topo-net-anomaly
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
```

Tests are pure-Python and exercise every module in a few seconds on a laptop.
No network access is required.

## Running benchmarks

```bash
python scripts/run_benchmarks.py --out results/benchmarks
python scripts/run_pipeline.py --out results/runs/default
python scripts/make_figures.py --out results/figures
```

## Code style

- Use `from __future__ import annotations` for forward-compatible typing.
- Prefer `dataclass` for plain data containers.
- All high-resolution timing **must** go through `topo_anomaly.utils.Timer`
  which uses `time.perf_counter_ns()`. Do not introduce `time.time()`,
  `time.perf_counter()`, or `datetime.now()` for timing measurements.
- Keep modules focused — one mathematical concept per file.
- Every public class should have a `to_metadata()` method returning a
  JSON-serialisable dict.

## Adding new anomaly types

1. Add the `kind` to the validation set in `AnomalySpec.__post_init__`
   (in `src/topo_anomaly/data.py`).
2. Implement the injection logic in `inject_anomalies`.
3. Add a test in `tests/test_data.py`.
4. Add an entry to the README's "Anomaly types" table.

## Adding new reduction methods

1. Add the method to `reduce_point_cloud` in `src/topo_anomaly/pipeline.py`.
2. Add the method to the validation in `TopologicalAnomalyPipeline.__post_init__`.
3. Add a benchmark entry in `scripts/run_benchmarks.py`
   (`benchmark_reduction_methods`).
4. Add a test in `tests/test_pipeline.py::TestReducePointCloud`.

## Adding new topological features

1. Implement the feature function in `src/topo_anomaly/features.py`.
2. Wire it into `TopologicalFeatureExtractor._extract` and update
   `feature_names()`.
3. Add a test in `tests/test_features.py`.

## Pull-request checklist

- [ ] Tests pass: `pytest tests/ -v`
- [ ] No new flake8 / pylint errors introduced.
- [ ] If you added a public API, update `__init__.py` `__all__` and `README.md`.
- [ ] If you added a stage to the pipeline, ensure it is wrapped in a
      `timer.measure(...)` context.
- [ ] `CHANGELOG.md` updated under the `[Unreleased]` section.

## Reporting issues

Please include:
- Python version and OS.
- `pip show topo-anomaly giotto-tda numpy scipy scikit-learn` output.
- Minimal reproducer.
- Whether you can reproduce on the latest `main`.
