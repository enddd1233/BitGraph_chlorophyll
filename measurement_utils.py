from __future__ import annotations

import csv
import json
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import torch
except Exception:
    torch = None


class StopMeasurement(RuntimeError):
    pass


def ensure_parent_dir(path: Optional[str]) -> None:
    if not path:
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_iter_times_csv(path: str, records: List[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    fieldnames = [
        "global_iter",
        "epoch",
        "iter_in_epoch",
        "stage",
        "iter_type",
        "phase",
        "elapsed_ms",
        "rate_scope",
        "timestamp",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k) for k in fieldnames})


def _normalize_device(device: Any) -> Optional[str]:
    if device is None:
        return None
    if hasattr(device, "type"):
        index = getattr(device, "index", None)
        if device.type == "cuda":
            return f"cuda:{0 if index is None else index}"
        return str(device.type)
    return str(device)


def _cuda_available(device: Optional[str]) -> bool:
    return bool(torch is not None and torch.cuda.is_available() and device and str(device).startswith("cuda"))


@dataclass
class MeasurementSummary:
    stage: str
    iter_type: str
    rate_scope: str
    warmup_iters: int
    measure_iters: int
    measured_iters: int
    total_iters_seen: int
    measurement_complete: bool
    time_ms_per_iter_mean: Optional[float]
    time_ms_per_iter_median: Optional[float]
    torch_peak_alloc_mb: Optional[float]
    torch_peak_reserved_mb: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "iter_type": self.iter_type,
            "rate_scope": self.rate_scope,
            "warmup_iters": self.warmup_iters,
            "measure_iters": self.measure_iters,
            "measured_iters": self.measured_iters,
            "total_iters_seen": self.total_iters_seen,
            "measurement_complete": self.measurement_complete,
            "time_ms_per_iter_mean": self.time_ms_per_iter_mean,
            "time_ms_per_iter_median": self.time_ms_per_iter_median,
            "torch_peak_alloc_mb": self.torch_peak_alloc_mb,
            "torch_peak_reserved_mb": self.torch_peak_reserved_mb,
        }


class IterationMeasurer:
    def __init__(
        self,
        *,
        enabled: bool,
        warmup_iters: int,
        measure_iters: int,
        stage: str,
        iter_type: str,
        rate_scope: str,
        device: Any = None,
        shared_records: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.warmup_iters = int(max(0, warmup_iters))
        self.measure_iters = int(max(0, measure_iters))
        self.stage = stage
        self.iter_type = iter_type
        self.rate_scope = rate_scope
        self.device = _normalize_device(device)
        self.shared_records = shared_records
        self.total_iters_seen = 0
        self._start_time = None
        self._records: List[Dict[str, Any]] = []
        self._measurement_complete = False
        if self.enabled and _cuda_available(self.device):
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    @property
    def records(self) -> List[Dict[str, Any]]:
        return list(self._records)

    def _sync_cuda(self) -> None:
        if not self.enabled or not _cuda_available(self.device):
            return
        try:
            if self.device and ":" in self.device:
                torch.cuda.synchronize(int(self.device.split(":", 1)[1]))
            else:
                torch.cuda.synchronize()
        except Exception:
            pass

    def start_iter(self) -> None:
        if not self.enabled:
            return
        self._sync_cuda()
        self._start_time = time.perf_counter()

    def end_iter(self, *, epoch: Optional[int], iter_in_epoch: Optional[int]) -> None:
        if not self.enabled or self._start_time is None:
            return
        self._sync_cuda()
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0
        self.total_iters_seen += 1
        global_iter = self.total_iters_seen
        phase = "warmup" if global_iter <= self.warmup_iters else "measure"
        record = {
            "global_iter": global_iter,
            "epoch": epoch,
            "iter_in_epoch": iter_in_epoch,
            "stage": self.stage,
            "iter_type": self.iter_type,
            "phase": phase,
            "elapsed_ms": elapsed_ms,
            "rate_scope": self.rate_scope,
            "timestamp": time.time(),
        }
        self._records.append(record)
        if self.shared_records is not None:
            self.shared_records.append(record)
        self._start_time = None
        if self.total_iters_seen >= self.warmup_iters + self.measure_iters:
            self._measurement_complete = True
            raise StopMeasurement(f"{self.stage} measurement complete")

    def summary(self) -> MeasurementSummary:
        measured = [r["elapsed_ms"] for r in self._records if r["phase"] == "measure"]
        alloc_mb = None
        reserved_mb = None
        if self.enabled and _cuda_available(self.device):
            try:
                alloc_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
                reserved_mb = float(torch.cuda.max_memory_reserved() / (1024 ** 2))
            except Exception:
                alloc_mb = None
                reserved_mb = None
        return MeasurementSummary(
            stage=self.stage,
            iter_type=self.iter_type,
            rate_scope=self.rate_scope,
            warmup_iters=self.warmup_iters,
            measure_iters=self.measure_iters,
            measured_iters=len(measured),
            total_iters_seen=self.total_iters_seen,
            measurement_complete=self._measurement_complete and len(measured) >= self.measure_iters,
            time_ms_per_iter_mean=float(sum(measured) / len(measured)) if measured else None,
            time_ms_per_iter_median=float(statistics.median(measured)) if measured else None,
            torch_peak_alloc_mb=alloc_mb,
            torch_peak_reserved_mb=reserved_mb,
        )

    def write_measure_json(self, path: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = self.summary().to_dict()
        if extra:
            payload.update(extra)
        write_json(path, payload)
        return payload


def build_combined_measure(stage_payloads: Dict[str, Dict[str, Any]], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "stage": "combined",
        "iter_type": "multi_stage",
        "rate_scope": next(iter(stage_payloads.values())).get("rate_scope") if stage_payloads else None,
        "stages": stage_payloads,
    }
    if extra:
        payload.update(extra)
    return payload
