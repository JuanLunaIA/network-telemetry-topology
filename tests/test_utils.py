"""Tests for topo_anomaly.utils — Timer, IO helpers."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

from topo_anomaly.utils import (
    Timer,
    ensure_dir,
    format_duration,
    load_json,
    save_json,
)


class TestFormatDuration:
    @pytest.mark.parametrize("ns,expected_unit", [
        (1, "ns"),
        (500, "ns"),
        (1_500, "us"),
        (1_500_000, "ms"),
        (1_500_000_000, "s"),
    ])
    def test_units(self, ns, expected_unit):
        s = format_duration(ns)
        assert expected_unit in s


class TestTimer:
    def test_measure_records_duration(self):
        t = Timer()
        with t.measure("sleep"):
            time.sleep(0.01)
        assert len(t.records) == 1
        r = t.records[0]
        assert r.name == "sleep"
        assert r.duration_ns > 0
        # ≥ 5ms in duration
        assert r.duration_ms >= 5.0

    def test_measure_callable_returns_result(self):
        t = Timer()
        result, record = t.measure_callable("square", lambda x: x * x, 5)
        assert result == 25
        assert record.name == "square"

    def test_metadata_attached(self):
        t = Timer()
        with t.measure("step", n_samples=128, kind="fit"):
            pass
        assert t.records[0].metadata["n_samples"] == 128
        assert t.records[0].metadata["kind"] == "fit"

    def test_total_sums_durations(self):
        t = Timer()
        with t.measure("a"):
            time.sleep(0.005)
        with t.measure("b"):
            time.sleep(0.005)
        assert t.total() >= 10_000_000  # ≥ 10ms

    def test_summary_structure(self):
        t = Timer()
        with t.measure("a"):
            pass
        s = t.summary()
        assert "total_duration_ns" in s
        assert "total_duration_human" in s
        assert "n_records" in s
        assert "records" in s
        assert s["n_records"] == 1

    def test_to_json_writes_file(self, tmp_path):
        t = Timer()
        with t.measure("a"):
            pass
        p = str(tmp_path / "timings.json")
        t.to_json(p)
        with open(p) as f:
            d = json.load(f)
        assert d["n_records"] == 1

    def test_getitem_by_name(self):
        t = Timer()
        with t.measure("foo"):
            pass
        with t.measure("bar"):
            pass
        assert t["foo"].name == "foo"
        with pytest.raises(KeyError):
            _ = t["nope"]

    def test_perf_counter_ns_monotonic(self):
        """Ensure we are using the monotonic perf_counter_ns source."""
        t = Timer()
        a = time.perf_counter_ns()
        with t.measure("m"):
            pass
        b = time.perf_counter_ns()
        r = t["m"]
        # start_ns must be inside [a, b]
        assert a <= r.start_ns <= b


class TestIO:
    def test_ensure_dir_creates(self, tmp_path):
        d = ensure_dir(str(tmp_path / "a" / "b" / "c"))
        assert os.path.isdir(d)

    def test_save_and_load_json(self, tmp_path):
        p = str(tmp_path / "x.json")
        obj = {"a": 1, "b": [1, 2, 3], "c": {"d": 4.5}}
        save_json(obj, p)
        loaded = load_json(p)
        assert loaded == obj

    def test_save_json_handles_numpy(self, tmp_path):
        p = str(tmp_path / "np.json")
        obj = {"arr": np.arange(5), "scalar": np.float64(3.14), "int": np.int64(7)}
        save_json(obj, p)
        loaded = load_json(p)
        assert loaded["arr"] == [0, 1, 2, 3, 4]
        assert loaded["scalar"] == 3.14
        assert loaded["int"] == 7
