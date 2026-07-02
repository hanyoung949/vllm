#!/usr/bin/env python3
"""Lightweight JSONL trace logger for split training / rollout."""

import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional


class SplitTraceLogger:
    """Per-process JSONL trace logger.

    Enabled only when SPLIT_TRACE_DIR is set. Each process writes to its own
    jsonl file so that events are never deduplicated/folded by Ray.
    """

    _instances: Dict[str, "SplitTraceLogger"] = {}
    _lock = threading.Lock()

    def __new__(cls, name: Optional[str] = None) -> "SplitTraceLogger":
        name = name or "default"
        with cls._lock:
            if name not in cls._instances:
                cls._instances[name] = super().__new__(cls)
            return cls._instances[name]

    def __init__(self, name: Optional[str] = None):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        self._write_lock = threading.Lock()
        self._file = None
        self._enabled = False
        self._rank = self._infer_rank()
        self._stage = os.environ.get("SPLIT_STAGE", "unknown")
        self._name = name or f"rank{self._rank}"

        trace_dir = os.environ.get("SPLIT_TRACE_DIR")
        if not trace_dir:
            return

        os.makedirs(trace_dir, exist_ok=True)
        pid = os.getpid()
        ts = time.strftime("%m%d_%H%M%S")
        fname = f"{self._name}_{self._stage}_{pid}_{ts}.jsonl"
        path = os.path.join(trace_dir, fname)
        try:
            self._file = open(path, "a", buffering=1)  # line buffered
            self._enabled = True
            self._write({
                "ts_ns": time.time_ns(),
                "rank": self._rank,
                "stage": self._stage,
                "phase": "logger",
                "event": "init",
                "path": path,
            })
        except Exception:
            self._enabled = False

    @staticmethod
    def _infer_rank() -> int:
        for key in ("RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK"):
            val = os.environ.get(key)
            if val is not None:
                try:
                    return int(val)
                except ValueError:
                    pass
        return 0

    def is_enabled(self) -> bool:
        return self._enabled

    def log(self,
            phase: str,
            event: str,
            step: Optional[int] = None,
            micro_step: Optional[int] = None,
            token_idx: Optional[int] = None,
            bytes_: Optional[int] = None,
            extra: Optional[Dict[str, Any]] = None,
            **kwargs):
        if not self._enabled:
            return
        rec: Dict[str, Any] = {
            "ts_ns": time.time_ns(),
            "rank": self._rank,
            "stage": self._stage,
            "phase": phase,
            "event": event,
        }
        if step is not None:
            rec["step"] = step
        if micro_step is not None:
            rec["micro_step"] = micro_step
        if token_idx is not None:
            rec["token_idx"] = token_idx
        if bytes_ is not None:
            rec["bytes"] = bytes_
        if extra:
            rec.update(extra)
        if kwargs:
            rec.update(kwargs)
        self._write(rec)

    def _write(self, rec: Dict[str, Any]):
        if self._file is None:
            return
        with self._write_lock:
            try:
                self._file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                pass

    @contextmanager
    def trace(self, phase: str, **kwargs):
        self.log(phase=phase, event="start", **kwargs)
        try:
            yield
        finally:
            self.log(phase=phase, event="end", **kwargs)

    def flush(self):
        if self._file is not None:
            with self._write_lock:
                self._file.flush()

    def close(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
            self._enabled = False


@contextmanager
def split_trace(phase: str, name: Optional[str] = None, **kwargs):
    logger = SplitTraceLogger(name)
    with logger.trace(phase, **kwargs):
        yield


def get_split_trace_logger(name: Optional[str] = None) -> SplitTraceLogger:
    return SplitTraceLogger(name)
