
from __future__ import annotations

from datetime import datetime
from typing import Any


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def percentile(values: list[float], p: float) -> float:
    """
    简单百分位数计算。
    p 取值如 95 表示 P95。
    """
    clean_values = sorted(float(v) for v in values if v is not None)

    if not clean_values:
        return 0.0

    if len(clean_values) == 1:
        return round(clean_values[0], 2)

    rank = (p / 100.0) * (len(clean_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(clean_values) - 1)
    weight = rank - lower

    value = clean_values[lower] * (1 - weight) + clean_values[upper] * weight
    return round(value, 2)


def summarize_latency_records(records: list[dict[str, Any]], threshold_ms: int = 1000) -> dict[str, Any]:
    latencies = [
        float(item.get("latency_ms"))
        for item in records
        if item.get("latency_ms") is not None
    ]

    if not latencies:
        return {
            "count": 0,
            "avg_latency_ms": 0,
            "min_latency_ms": 0,
            "max_latency_ms": 0,
            "p95_latency_ms": 0,
            "pass_count": 0,
            "fail_count": 0,
            "pass_rate": 0,
            "threshold_ms": threshold_ms,
            "is_realtime": False,
        }

    count = len(latencies)
    pass_count = sum(1 for value in latencies if value <= threshold_ms)
    fail_count = count - pass_count

    return {
        "count": count,
        "avg_latency_ms": round(sum(latencies) / count, 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "p95_latency_ms": percentile(latencies, 95),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": round(pass_count / count, 4),
        "threshold_ms": threshold_ms,
        "is_realtime": pass_count == count,
    }
