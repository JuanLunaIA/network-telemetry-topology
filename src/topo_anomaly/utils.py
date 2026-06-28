"""
utils — timing, IO and small helpers used throughout the package.

All high-resolution timing in this project uses :func:`time.perf_counter_ns`
which provides nanosecond resolution and is **monotonic** — it is not affected
by system clock adjustments, which is essential when comparing execution times
of multiple pipeline stages.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class TimingRecord:
    """A single named timing measurement."""

    name: str
    start_ns: int
    end_ns: int
    duration_ns: int
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # keep also a human readable form
        d["duration_human"] = format_duration(self.duration_ns)
        return d


def format_duration(duration_ns: int) -> str:
    """Format a nanosecond duration as a human-readable string."""
    if duration_ns < 1_000:
        return f"{duration_ns} ns"
    if duration_ns < 1_000_000:
        return f"{duration_ns / 1_000:.3f} us"
    if duration_ns < 1_000_000_000:
        return f"{duration_ns / 1_000_000:.3f} ms"
    return f"{duration_ns / 1_000_000_000:.3f} s"


class Timer:
    """High-resolution timer using :func:`time.perf_counter_ns`.

    The timer keeps an ordered log of every measurement it has taken and can
    be serialised to JSON for reproducibility.

    Example
    -------
    >>> timer = Timer()
    >>> with timer.measure("embedding"):
    ...     # expensive computation
    ...     pass
    >>> timer.records[0].name
    'embedding'
    """

    def __init__(self) -> None:
        self.records: List[TimingRecord] = []

    @contextmanager
    def measure(self, name: str, **metadata: Any) -> Iterator[Timer]:
        start = time.perf_counter_ns()
        try:
            yield self
        finally:
            end = time.perf_counter_ns()
            duration = end - start
            record = TimingRecord(
                name=name,
                start_ns=start,
                end_ns=end,
                duration_ns=duration,
                duration_ms=duration / 1_000_000.0,
                metadata=dict(metadata),
            )
            self.records.append(record)

    def measure_callable(self, name: str, fn, *args, **kwargs):
        """Call ``fn(*args, **kwargs)`` and time it; return ``(result, record)``."""
        start = time.perf_counter_ns()
        result = fn(*args, **kwargs)
        end = time.perf_counter_ns()
        duration = end - start
        record = TimingRecord(
            name=name,
            start_ns=start,
            end_ns=end,
            duration_ns=duration,
            duration_ms=duration / 1_000_000.0,
            metadata={k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool))},
        )
        self.records.append(record)
        return result, record

    def total(self) -> int:
        return sum(r.duration_ns for r in self.records)

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary of all measurements."""
        return {
            "total_duration_ns": self.total(),
            "total_duration_human": format_duration(self.total()),
            "n_records": len(self.records),
            "records": [r.to_dict() for r in self.records],
        }

    def to_json(self, path: Optional[str] = None) -> str:
        s = json.dumps(self.summary(), indent=2)
        if path is not None:
            with open(path, "w") as f:
                f.write(s)
        return s

    def __getitem__(self, name: str) -> TimingRecord:
        for r in self.records:
            if r.name == name:
                return r
        raise KeyError(name)


def ensure_dir(path: str) -> str:
    """Create ``path`` (and parents) if it does not exist; return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    """Write ``obj`` to ``path`` as JSON (parent dirs are created)."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=_json_default)


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def _json_default(o: Any):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return list(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serialisable")
