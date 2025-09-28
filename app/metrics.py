from __future__ import annotations

import math
import shutil
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from threading import Lock
from typing import Deque, Dict, List, Optional

import psutil


@dataclass
class BatchLogEntry:
    """记录每个批次摄取的核心信息"""

    timestamp: float
    tenant: str
    duration_seconds: float
    accepted: int
    failed: int
    batch_id: str


@dataclass
class RecordLogEntry:
    """记录最近摄取的交易，便于统计业务指标"""

    timestamp: float
    tenant: str
    account_id: Optional[str]
    merchant: Optional[str]
    amount: Optional[float]


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return ordered[int(k)]
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    return lower_value + (upper_value - lower_value) * (k - lower)


class IngestionMetrics:
    """汇总摄取请求信息，供监控面板使用"""

    def __init__(self, retention_seconds: int = 3600, record_retention: int = 2000):
        self._lock = Lock()
        self._retention_seconds = retention_seconds
        self._batches: Deque[BatchLogEntry] = deque()
        self._records: Deque[RecordLogEntry] = deque(maxlen=record_retention)
        self.total_batches: int = 0
        self.total_transactions: int = 0
        self.total_failed: int = 0
        self._last_updated: Optional[float] = None

    def record_batch(
        self,
        tenant: str,
        batch_id: str,
        accepted: int,
        failed: int,
        duration_seconds: float,
        clean_rows: List[Dict[str, object]],
    ) -> None:
        """记录一次批量摄取的结果"""

        timestamp = time.time()
        entry = BatchLogEntry(
            timestamp=timestamp,
            tenant=tenant,
            duration_seconds=duration_seconds,
            accepted=accepted,
            failed=failed,
            batch_id=batch_id,
        )

        with self._lock:
            self._batches.append(entry)
            self.total_batches += 1
            self.total_transactions += accepted
            self.total_failed += failed
            self._last_updated = timestamp

            for row in clean_rows:
                amount_value = _safe_float(row.get("amount"))
                merchant_value = _string_or_none(row.get("merchant_id") or row.get("merchant"))
                account_value = _string_or_none(row.get("account_id"))
                record_entry = RecordLogEntry(
                    timestamp=timestamp,
                    tenant=tenant,
                    account_id=account_value,
                    merchant=merchant_value,
                    amount=amount_value,
                )
                self._records.append(record_entry)

            self._prune_locked(timestamp)

    def get_dashboard(self) -> Dict[str, object]:
        """构建用于前端展示的监控数据"""

        with self._lock:
            now = time.time()
            self._prune_locked(now)

            batches = list(self._batches)
            records = list(self._records)

        durations = [b.duration_seconds for b in batches]
        total_rows_window = sum(b.accepted + b.failed for b in batches)
        rows_last_minute = sum(
            b.accepted + b.failed for b in batches if now - b.timestamp <= 60
        )
        rows_last_five_minutes = sum(
            b.accepted + b.failed for b in batches if now - b.timestamp <= 300
        )
        errors_last_hour = sum(b.failed for b in batches)
        rows_last_hour = sum(b.accepted + b.failed for b in batches)
        requests_last_minute = sum(1 for b in batches if now - b.timestamp <= 60)

        tenant_recent = Counter()
        for batch in batches:
            if now - batch.timestamp <= self._retention_seconds:
                tenant_recent[batch.tenant] += batch.accepted + batch.failed

        amounts = [r.amount for r in records if r.amount is not None]
        active_accounts = {r.account_id for r in records if r.account_id}
        merchants = Counter(r.merchant for r in records if r.merchant)

        api_metrics = {
            "records_per_second": round(rows_last_minute / 60.0, 3) if rows_last_minute else 0.0,
            "records_per_minute": rows_last_minute,
            "batches_per_minute": requests_last_minute,
            "average_response_time_ms": round(mean(durations) * 1000, 2) if durations else 0.0,
            "p95_response_time_ms": round(_percentile(durations, 95) * 1000, 2) if durations else 0.0,
            "error_rate": round(errors_last_hour / rows_last_hour, 4) if rows_last_hour else 0.0,
        }

        business_metrics = {
            "transactions_processed": self.total_transactions,
            "failed_transactions": self.total_failed,
            "failure_rate": round(
                self.total_failed / max(1, self.total_transactions + self.total_failed), 4
            ),
            "rolling_5m_volume": rows_last_five_minutes,
            "recent_average_amount": round(mean(amounts), 2) if amounts else 0.0,
            "recent_median_amount": round(median(amounts), 2) if amounts else 0.0,
            "high_value_transaction_rate": round(
                sum(1 for amt in amounts if amt >= 10000) / len(amounts), 4
            ) if amounts else 0.0,
            "active_accounts": len(active_accounts),
        }

        system_metrics = _collect_system_metrics()

        tenant_breakdown = [
            {
                "tenant_id": tenant,
                "records": count,
                "share": round(count / total_rows_window, 4) if total_rows_window else 0.0,
            }
            for tenant, count in tenant_recent.most_common(5)
        ]

        top_merchants = [
            {"merchant": merchant, "records": count}
            for merchant, count in merchants.most_common(5)
        ]

        return {
            "api_metrics": api_metrics,
            "business_metrics": business_metrics,
            "system_metrics": system_metrics,
            "tenant_breakdown": tenant_breakdown,
            "top_merchants": top_merchants,
            "retention_window_seconds": self._retention_seconds,
            "last_updated": self._last_updated,
        }

    def _prune_locked(self, now: float) -> None:
        """清理超出窗口期的数据（需在已持有锁的情况下调用）"""

        while self._batches and now - self._batches[0].timestamp > self._retention_seconds:
            self._batches.popleft()

        while self._records and now - self._records[0].timestamp > self._retention_seconds:
            self._records.popleft()


def _collect_system_metrics() -> Dict[str, float]:
    cpu_usage = psutil.cpu_percent(interval=None)
    virtual_mem = psutil.virtual_memory()
    disk_usage = shutil.disk_usage(Path.cwd())

    return {
        "cpu_usage_percent": round(cpu_usage, 2),
        "memory_usage_percent": round(virtual_mem.percent, 2),
        "memory_used_mb": round(virtual_mem.used / (1024 * 1024), 2),
        "disk_usage_percent": round(disk_usage.used / disk_usage.total * 100, 2) if disk_usage.total else 0.0,
    }


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _string_or_none(value: object) -> Optional[str]:
    if value is None:
        return None
    value_str = str(value).strip()
    return value_str or None


metrics_store = IngestionMetrics()

