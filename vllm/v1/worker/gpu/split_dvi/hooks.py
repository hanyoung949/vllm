"""High-level telemetry hooks for DVI L0.

These hooks are intended to be called from the vLLM model runner / sampler.
They hide the reserve/submit protocol and provide feature flags and timing.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from vllm.v1.worker.gpu.split_dvi.telemetry import (
    DVIRecordKey,
    DVISessionKey,
    DVIStage0TelemetryProducer,
    DVIStage2TelemetryProducer,
)


class DVIStage2TelemetryHook:
    """Stage-2 hook that captures the processed verifier distribution.

    reserve is a cheap deterministic sampling decision. CUDA tensors from
    reserved rows use the producer's bounded pinned-memory/event handoff;
    already-CPU data keeps a synchronous submission method for tests and
    offline callers.
    """

    def __init__(
        self,
        producer: DVIStage2TelemetryProducer,
        enabled: bool = True,
    ) -> None:
        self.producer = producer
        self.enabled = enabled
        self._reserve_wall_ms = 0.0
        self._submit_wall_ms = 0.0
        self._cancel_wall_ms = 0.0
        self._reserved_count = 0
        self._submitted_count = 0
        self._capture_failed_count = 0
        self._skipped_count = 0

    @property
    def session_key(self) -> DVISessionKey:
        return self.producer.session_key

    @property
    def dvi_top_k(self) -> int:
        return self.producer.sampler.config.top_k

    def make_record_key(self, request_id: str, position: int) -> DVIRecordKey:
        return self.producer.make_record_key(request_id, position)

    def reserve(self, key: DVIRecordKey) -> Any | None:
        """Return a ticket if this position should capture verifier data."""
        if not self.enabled:
            return None
        t0 = time.perf_counter()
        ticket = self.producer.reserve(key)
        self._reserve_wall_ms += (time.perf_counter() - t0) * 1000.0
        if ticket is not None:
            self._reserved_count += 1
        else:
            self._skipped_count += 1
        return ticket

    def submit_processed_topk(
        self,
        ticket: Any,
        verifier_topk_ids: list[int],
        verifier_topk_logprobs: list[float],
        verifier_residual_mass: float,
        verifier_top1_id: int,
    ) -> bool:
        """Enqueue the processed verifier top-k for a reserved position."""
        if not self.enabled or ticket is None:
            return False
        t0 = time.perf_counter()
        ok = self.producer.submit(
            ticket,
            verifier_topk_ids,
            verifier_topk_logprobs,
            verifier_residual_mass,
            verifier_top1_id,
        )
        self._submit_wall_ms += (time.perf_counter() - t0) * 1000.0
        if ok:
            self._submitted_count += 1
        return ok

    def submit_device_topk(
        self,
        ticket: Any,
        *,
        topk_ids: torch.Tensor,
        topk_logprobs: torch.Tensor,
        residual_mass: torch.Tensor,
        top1_id: torch.Tensor,
        valid_count: torch.Tensor,
    ) -> bool:
        """Submit one processed verifier row without synchronizing the runner."""
        if not self.enabled or ticket is None:
            return False
        t0 = time.perf_counter()
        ok = self.producer.submit_device(
            ticket,
            topk_ids=topk_ids,
            topk_logprobs=topk_logprobs,
            residual_mass=residual_mass,
            top1_id=top1_id,
            valid_count=valid_count,
        )
        self._submit_wall_ms += (time.perf_counter() - t0) * 1000.0
        if ok:
            self._submitted_count += 1
        return ok

    def cancel(self, ticket: Any) -> bool:
        """Mark a reserved slot as capture-failed.

        The position remains in the deterministic sample set; the other stage
        will see the same position as an incomplete record if it has data.
        """
        if not self.enabled or ticket is None:
            return False
        t0 = time.perf_counter()
        ok = self.producer.cancel(ticket)
        self._cancel_wall_ms += (time.perf_counter() - t0) * 1000.0
        if ok:
            self._capture_failed_count += 1
        return ok

    def capture(
        self,
        key: DVIRecordKey,
        verifier_topk_ids: list[int],
        verifier_topk_logprobs: list[float],
        verifier_residual_mass: float,
        verifier_top1_id: int,
    ) -> bool:
        """One-shot convenience wrapper.

        NOTE: This defeats the "avoid 95% top-k compute" optimization because
        the caller must already provide top-k data. Prefer ``reserve`` +
        ``submit_processed_topk`` when integrating into the sampler.
        """
        ticket = self.reserve(key)
        if ticket is None:
            return False
        return self.submit_processed_topk(
            ticket,
            verifier_topk_ids,
            verifier_topk_logprobs,
            verifier_residual_mass,
            verifier_top1_id,
        )

    @property
    def metrics(self) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {
            "enabled": self.enabled,
            "reserve_wall_ms": self._reserve_wall_ms,
            "submit_wall_ms": self._submit_wall_ms,
            "cancel_wall_ms": self._cancel_wall_ms,
            "reserved_count": self._reserved_count,
            "submitted_count": self._submitted_count,
            "capture_failed_count": self._capture_failed_count,
            "skipped_count": self._skipped_count,
        }
        metrics.update(self.producer.handoff_metrics)
        return metrics


class DVIStage0TelemetryProbe:
    """Stage-0 probe that captures boundary hidden states asynchronously.

    Mirrors :class:`DVIStage2TelemetryHook` on the first split stage: cheap
    deterministic reserve per position, then an independent bounded handoff.
    """

    def __init__(
        self,
        producer: DVIStage0TelemetryProducer,
        enabled: bool = True,
    ) -> None:
        self.producer = producer
        self.enabled = enabled
        self._reserve_wall_ms = 0.0
        self._submit_wall_ms = 0.0
        self._reserved_count = 0
        self._submitted_count = 0
        self._skipped_count = 0

    @property
    def session_key(self) -> DVISessionKey:
        return self.producer.session_key

    def make_record_key(self, request_id: str, position: int) -> DVIRecordKey:
        return self.producer.make_record_key(request_id, position)

    def capture(
        self, request_id: str, position: int, hidden_row: torch.Tensor
    ) -> bool:
        """Reserve + submit one boundary hidden row for a decode position."""
        if not self.enabled:
            return False
        t0 = time.perf_counter()
        ticket = self.producer.reserve(
            self.make_record_key(request_id, position)
        )
        self._reserve_wall_ms += (time.perf_counter() - t0) * 1000.0
        if ticket is None:
            self._skipped_count += 1
            return False
        t0 = time.perf_counter()
        ok = self.producer.submit_device(ticket, hidden_row)
        if not ok:
            self.producer.cancel(ticket)
        self._submit_wall_ms += (time.perf_counter() - t0) * 1000.0
        if ok:
            self._reserved_count += 1
            self._submitted_count += 1
        return ok

    def finalize_request(self, request_id: str) -> None:
        self.producer.finalize_request(self.session_key, request_id)

    @property
    def metrics(self) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {
            "enabled": self.enabled,
            "reserve_wall_ms": self._reserve_wall_ms,
            "submit_wall_ms": self._submit_wall_ms,
            "reserved_count": self._reserved_count,
            "submitted_count": self._submitted_count,
            "skipped_count": self._skipped_count,
        }
        metrics.update(self.producer.handoff_metrics)
        return metrics


__all__ = [
    "DVIStage2TelemetryHook",
    "DVIStage0TelemetryProbe",
]
