"""Partial spool telemetry capture for DVI L0.

This module lives in vLLM and must not depend on verl. It produces per-worker
partial spools that are later merged offline into schema v1 artifacts.

The spool writer is asynchronous: hook threads call `enqueue_*` with small
immutable records; a background thread batches and flushes them to disk.
Telemetry never blocks rollout: when the bounded queue is full, records are
dropped and counted.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file as save_safetensors

from vllm.v1.worker.gpu.split_dvi.artifact import DVIArtifactError


@dataclass(frozen=True)
class DVISamplingConfig:
    """Deterministic online sampling configuration."""

    sample_rate: float = 0.05
    max_per_request: int = 16
    seed: int = 0
    top_k: int = 8

    def validate(self) -> None:
        if not 0.0 < self.sample_rate <= 1.0:
            raise DVIArtifactError(
                f"sample_rate must be in (0, 1], got {self.sample_rate}"
            )
        if self.max_per_request <= 0:
            raise DVIArtifactError(
                f"max_per_request must be positive, got {self.max_per_request}"
            )
        if self.top_k <= 0:
            raise DVIArtifactError(f"top_k must be positive, got {self.top_k}")


# (run_id, rollout_id, policy_version)
DVISessionKey = tuple[str, str, str]
# (run_id, rollout_id, policy_version, request_id, position)
DVIRecordKey = tuple[str, str, str, str, int]


@dataclass(frozen=True)
class DVITelemetryTicket:
    """Opaque ticket returned by reserve() and consumed by submit()."""

    key: DVIRecordKey
    priority: int


def _canonical_bytes(key: DVIRecordKey, salt: str = "") -> bytes:
    run_id, rollout_id, policy_version, request_id, position = key
    obj = {
        "run_id": run_id,
        "rollout_id": rollout_id,
        "policy_version": policy_version,
        "request_id": request_id,
        "position": position,
        "salt": salt,
    }
    return json.dumps(obj, sort_keys=True, ensure_ascii=True).encode("utf-8")


def _deterministic_int(key: DVIRecordKey, salt: str = "") -> int:
    digest = hashlib.sha256(_canonical_bytes(key, salt)).digest()
    return int.from_bytes(digest[:8], "big")


def _key_to_dict(key: DVIRecordKey) -> dict[str, Any]:
    return {
        "run_id": key[0],
        "rollout_id": key[1],
        "policy_version": key[2],
        "request_id": key[3],
        "position": key[4],
    }


def _key_from_dict(d: dict[str, Any]) -> DVIRecordKey:
    return (
        d["run_id"],
        d["rollout_id"],
        d["policy_version"],
        d["request_id"],
        int(d["position"]),
    )


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class DVIPerRequestSampler:
    """Deterministic per-(session, request) position sampler."""

    def __init__(self, config: DVISamplingConfig) -> None:
        config.validate()
        self.config = config
        self._state: dict[tuple[str, ...], list[tuple[int, int]]] = {}

    def on_position(
        self,
        session_key: DVISessionKey,
        request_id: str,
        position: int,
    ) -> int | None:
        """Return integer priority if position is a candidate, else None."""
        state_key = session_key + (request_id,)
        positions = self._state.setdefault(state_key, [])
        if any(p == position for p, _ in positions):
            return None
        key: DVIRecordKey = session_key + (request_id, position)
        value = _deterministic_int(key, salt=f"candidate:{self.config.seed}")
        threshold = int(self.config.sample_rate * (2**64 - 1))
        if value > threshold:
            return None
        priority = _deterministic_int(key, salt=f"priority:{self.config.seed}")
        positions.append((position, priority))
        return priority

    def drop_position(
        self,
        session_key: DVISessionKey,
        request_id: str,
        position: int,
    ) -> bool:
        """Remove a position from the candidate set.

        Used by producers when a non-canceled position is evicted from the
        runtime heap so that the sampler state stays synchronized with the
        buffered data. Canceled positions must NOT be dropped; they remain in
        the candidate set to become incomplete records at finalize time.
        """
        state_key = session_key + (request_id,)
        positions = self._state.get(state_key, [])
        new_positions = [(p, pr) for p, pr in positions if p != position]
        if len(new_positions) == len(positions):
            return False
        self._state[state_key] = new_positions
        return True

    def finalize_request(
        self,
        session_key: DVISessionKey,
        request_id: str,
    ) -> list[int]:
        """Return up to max_per_request positions with smallest priority."""
        state_key = session_key + (request_id,)
        positions = self._state.pop(state_key, [])
        positions.sort(key=lambda x: x[1])
        return [p for p, _ in positions[: self.config.max_per_request]]


@dataclass
class _QueueItem:
    kind: str
    key: DVIRecordKey | None
    data: Any


class DVIPartialSpoolWriter:
    """Bounded asynchronous partial spool writer for one worker side."""

    SPOOL_SCHEMA_VERSION = "spool-v1"

    def __init__(
        self,
        spool_dir: str | Path,
        spool_metadata: dict[str, Any],
        quota_bytes: int,
        shard_size: int = 128,
        queue_maxsize: int = 1024,
    ) -> None:
        if quota_bytes <= 0:
            raise DVIArtifactError("quota_bytes must be positive")
        if shard_size <= 0:
            raise DVIArtifactError("shard_size must be positive")
        if queue_maxsize <= 0:
            raise DVIArtifactError("queue_maxsize must be positive")

        self.spool_dir = Path(spool_dir)
        self.spool_metadata = dict(spool_metadata)
        self.quota_bytes = quota_bytes
        self.shard_size = shard_size
        self.queue_maxsize = queue_maxsize

        self.spool_dir.mkdir(parents=True, exist_ok=True)
        if any(self.spool_dir.iterdir()):
            raise DVIArtifactError(
                f"Spool directory is not empty: {self.spool_dir}"
            )

        self._used_bytes = 0
        self._pending_bytes = 0
        self._dropped_records = 0
        self._stage0_enqueued = 0
        self._stage2_enqueued = 0
        self._request_enqueued = 0
        self._quota_exceeded = False
        self._closed = False
        self._worker_error: BaseException | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._queue: queue.Queue[_QueueItem | None] = queue.Queue(
            maxsize=queue_maxsize
        )
        self._thread = threading.Thread(target=self._writer_thread, daemon=True)
        self._thread.start()

    @property
    def metrics(self) -> dict[str, int | float | bool]:
        return {
            "used_bytes": self._used_bytes,
            "quota_bytes": self.quota_bytes,
            "pending_bytes": self._pending_bytes,
            "stage0_enqueued": self._stage0_enqueued,
            "stage2_enqueued": self._stage2_enqueued,
            "request_enqueued": self._request_enqueued,
            "dropped_records": self._dropped_records,
            "quota_exceeded": self._quota_exceeded,
        }

    def _run_id(self) -> str:
        return str(self.spool_metadata.get("run_id", ""))

    def _rollout_id(self) -> str:
        return str(self.spool_metadata.get("rollout_id", ""))

    def _policy_version(self) -> str:
        return str(self.spool_metadata.get("policy_version", ""))

    def _validate_key(self, key: DVIRecordKey) -> None:
        if key[:3] != (
            self._run_id(),
            self._rollout_id(),
            self._policy_version(),
        ):
            raise DVIArtifactError(
                f"Record key {key} does not match spool session "
                f"({self._run_id()}, {self._rollout_id()}, "
                f"{self._policy_version()})"
            )

    def _have_quota(self, estimated_bytes: int) -> bool:
        if self._quota_exceeded:
            return False
        if (
            self._used_bytes + self._pending_bytes + estimated_bytes
            <= self.quota_bytes
        ):
            return True
        self._dropped_records += 1
        self._quota_exceeded = True
        return False

    def _enqueue(self, item: _QueueItem) -> bool:
        """Thread-safe enqueue; returns True if accepted, False if dropped.

        The accept/drop counters are updated inside the same lock, and the
        success counters are incremented before the item is placed in the
        queue. This guarantees that the manifest can never observe a record
        on disk without its counter increment being visible.
        """
        with self._lock:
            if self._closed:
                self._dropped_records += 1
                return False
            if item.kind == "stage0":
                self._stage0_enqueued += 1
            elif item.kind == "stage2":
                self._stage2_enqueued += 1
            elif item.kind == "request":
                self._request_enqueued += 1
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                # Roll back the counter increment we just performed.
                if item.kind == "stage0":
                    self._stage0_enqueued -= 1
                elif item.kind == "stage2":
                    self._stage2_enqueued -= 1
                elif item.kind == "request":
                    self._request_enqueued -= 1
                self._dropped_records += 1
                return False
            return True

    def enqueue_stage0(
        self,
        key: DVIRecordKey,
        hidden: torch.Tensor,
    ) -> bool:
        """Enqueue a stage_0 hidden tensor. Safe to call from model runner."""
        return self._enqueue(
            _QueueItem(kind="stage0", key=key, data=hidden)
        )

    def enqueue_stage2(
        self,
        key: DVIRecordKey,
        verifier_topk_ids: list[int],
        verifier_topk_logprobs: list[float],
        verifier_residual_mass: float,
        verifier_top1_id: int,
    ) -> bool:
        """Enqueue a stage_2 verifier partial record."""
        rec = {
            "key": _key_to_dict(key),
            "verifier_topk_ids": tuple(verifier_topk_ids),
            "verifier_topk_logprobs": tuple(verifier_topk_logprobs),
            "verifier_residual_mass": verifier_residual_mass,
            "verifier_top1_id": verifier_top1_id,
        }
        return self._enqueue(_QueueItem(kind="stage2", key=key, data=rec))

    def enqueue_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        response_token_ids: list[int],
    ) -> bool:
        """Enqueue request metadata."""
        rec = {
            "request_id": request_id,
            "prompt_token_ids": tuple(prompt_token_ids),
            "response_token_ids": tuple(response_token_ids),
        }
        return self._enqueue(
            _QueueItem(kind="request", key=None, data=rec)
        )

    def _estimate_stage0(self, hidden: torch.Tensor) -> int:
        return hidden.element_size() * hidden.nelement() + 256

    def _estimate_stage2(self, rec: dict[str, Any]) -> int:
        return len(json.dumps(rec, ensure_ascii=False).encode("utf-8")) + 256

    def _estimate_request(self, rec: dict[str, Any]) -> int:
        return len(json.dumps(rec, ensure_ascii=False).encode("utf-8")) + 256

    def _process_item(self, item: _QueueItem) -> None:
        if item.kind == "stage0":
            hidden = item.data
            if not isinstance(hidden, torch.Tensor):
                self._dropped_records += 1
                return
            hidden = (
                hidden.detach().to(device="cpu", dtype=torch.float32).contiguous()
            )
            if item.key is None:
                self._dropped_records += 1
                return
            self._validate_key(item.key)
            estimated = self._estimate_stage0(hidden)
            if not self._have_quota(estimated):
                return
            self._stage0_buffer.append((item.key, hidden))
            self._pending_bytes += estimated
            if len(self._stage0_buffer) >= self.shard_size:
                self._flush_stage0_shard()

        elif item.kind == "stage2":
            rec = item.data
            if item.key is None:
                self._dropped_records += 1
                return
            self._validate_key(item.key)
            estimated = self._estimate_stage2(rec)
            if not self._have_quota(estimated):
                return
            self._stage2_buffer.append(rec)
            self._pending_bytes += estimated
            if len(self._stage2_buffer) >= self.shard_size:
                self._flush_stage2_shard()

        elif item.kind == "request":
            rec = item.data
            estimated = self._estimate_request(rec)
            if not self._have_quota(estimated):
                return
            self._request_buffer.append(rec)
            self._pending_bytes += estimated

    def _writer_thread(self) -> None:
        self._stage0_buffer: list[tuple[DVIRecordKey, torch.Tensor]] = []
        self._stage2_buffer: list[dict[str, Any]] = []
        self._request_buffer: list[dict[str, Any]] = []
        self._stage0_shard_idx = 0
        self._stage2_shard_idx = 0

        try:
            while True:
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if self._stop_event.is_set():
                        break
                    continue
                if item is None:
                    break
                self._process_item(item)

            while self._stop_event.is_set():
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    break
                self._process_item(item)

            self._flush_stage0_shard()
            self._flush_stage2_shard()
            self._flush_requests()
            self._write_manifest()
        except BaseException as e:  # noqa: BLE001
            self._worker_error = e
            self._cleanup_tmp_files()

    def _flush_stage0_shard(self) -> None:
        if not self._stage0_buffer:
            return
        estimated = sum(
            self._estimate_stage0(hidden) for _, hidden in self._stage0_buffer
        )
        idx = self._stage0_shard_idx
        jsonl_name = f"stage0-{idx:05d}.jsonl"
        st_name = f"stage0-{idx:05d}.safetensors"
        jsonl_tmp = self.spool_dir / f".tmp.{jsonl_name}"
        st_tmp = self.spool_dir / f".tmp.{st_name}"

        tensor_dict: dict[str, torch.Tensor] = {}
        json_records: list[dict[str, Any]] = []
        for rec_idx, (key, hidden) in enumerate(self._stage0_buffer):
            tkey = f"s0-{idx:05d}-{rec_idx:05d}-h"
            tensor_dict[tkey] = hidden
            json_records.append({"key": _key_to_dict(key), "hidden_key": tkey})

        with open(jsonl_tmp, "w", encoding="utf-8") as f:
            for rec in json_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        save_safetensors(tensor_dict, str(st_tmp))

        jsonl_path = self.spool_dir / jsonl_name
        st_path = self.spool_dir / st_name
        os.replace(jsonl_tmp, jsonl_path)
        os.replace(st_tmp, st_path)

        self._pending_bytes -= estimated
        self._used_bytes += (
            jsonl_path.stat().st_size + st_path.stat().st_size
        )
        self._stage0_buffer = []
        self._stage0_shard_idx += 1

    def _flush_stage2_shard(self) -> None:
        if not self._stage2_buffer:
            return
        estimated = sum(
            self._estimate_stage2(rec) for rec in self._stage2_buffer
        )
        idx = self._stage2_shard_idx
        name = f"stage2-{idx:05d}.jsonl"
        tmp_path = self.spool_dir / f".tmp.{name}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for rec in self._stage2_buffer:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        final_path = self.spool_dir / name
        os.replace(tmp_path, final_path)
        self._pending_bytes -= estimated
        self._used_bytes += final_path.stat().st_size
        self._stage2_buffer = []
        self._stage2_shard_idx += 1

    def _flush_requests(self) -> None:
        if not self._request_buffer:
            return
        estimated = sum(
            self._estimate_request(rec) for rec in self._request_buffer
        )
        tmp_path = self.spool_dir / ".tmp.requests.jsonl"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for rec in self._request_buffer:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        final_path = self.spool_dir / "requests.jsonl"
        os.replace(tmp_path, final_path)
        self._pending_bytes -= estimated
        self._used_bytes += final_path.stat().st_size
        self._request_buffer = []

    def _write_manifest(self) -> None:
        file_checksums: dict[str, str] = {}
        for child in sorted(self.spool_dir.iterdir()):
            if child.is_file() and not child.name.startswith("."):
                file_checksums[child.name] = _sha256_file(child)
        manifest = {
            **self.spool_metadata,
            "schema_version": self.SPOOL_SCHEMA_VERSION,
            "finalized": True,
            "file_checksums": file_checksums,
            "metrics": self.metrics,
        }
        tmp_path = self.spool_dir / ".tmp.spool_manifest.json"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.spool_dir / "spool_manifest.json")

    def _cleanup_tmp_files(self) -> None:
        for child in self.spool_dir.glob(".tmp.*"):
            try:
                child.unlink()
            except OSError:
                pass

    def close(self) -> dict[str, int | float | bool]:
        with self._lock:
            if self._closed:
                self._maybe_raise_worker_error()
                return self.metrics
            self._closed = True
            self._stop_event.set()
        self._thread.join(timeout=30.0)
        self._cleanup_tmp_files()
        if self._thread.is_alive():
            raise DVIArtifactError("Spool writer thread did not terminate")
        self._maybe_raise_worker_error()
        return self.metrics

    def _maybe_raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise self._worker_error

    def __enter__(self) -> "DVIPartialSpoolWriter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def _sampling_config_from_metadata(
    spool_metadata: dict[str, Any],
) -> DVISamplingConfig:
    """Extract and validate the runtime sampling config from spool metadata."""
    cfg = spool_metadata.get("sampling_config", {})
    if not isinstance(cfg, dict):
        raise DVIArtifactError(
            f"spool_metadata['sampling_config'] must be a dict, got {type(cfg)}"
        )
    try:
        return DVISamplingConfig(
            sample_rate=float(cfg["sample_rate"]),
            max_per_request=int(cfg["max_per_request"]),
            seed=int(cfg["seed"]),
            top_k=int(cfg.get("top_k", 8)),
        )
    except (KeyError, ValueError, TypeError) as e:
        raise DVIArtifactError(
            f"Invalid sampling_config in spool metadata: {cfg!r}"
        ) from e


class _RequestBuffer:
    """Per-session/request buffer that keeps only selected candidate data."""

    def __init__(self) -> None:
        self.heap: list[tuple[int, int]] = []
        self.stage0: dict[int, torch.Tensor | None] = {}
        self.stage2: dict[int, dict[str, Any] | None] = {}
        self.canceled: set[int] = set()


class DVIStage0TelemetryProducer:
    """Capture stage_0 hidden states into a partial spool."""

    def __init__(
        self,
        writer: DVIPartialSpoolWriter,
        sampling_config: DVISamplingConfig,
    ) -> None:
        expected = _sampling_config_from_metadata(writer.spool_metadata)
        if sampling_config != expected:
            raise DVIArtifactError(
                f"DVIStage0TelemetryProducer sampling_config {sampling_config} "
                f"does not match spool metadata {expected}"
            )
        self.writer = writer
        self.sampler = DVIPerRequestSampler(sampling_config)
        self._buffers: dict[tuple[str, ...], _RequestBuffer] = {}
        self._capture_failed_count = 0

    @property
    def capture_failed_count(self) -> int:
        return self._capture_failed_count

    @property
    def session_key(self) -> DVISessionKey:
        return (
            str(self.writer.spool_metadata.get("run_id", "")),
            str(self.writer.spool_metadata.get("rollout_id", "")),
            str(self.writer.spool_metadata.get("policy_version", "")),
        )

    def make_record_key(self, request_id: str, position: int) -> DVIRecordKey:
        """Build a validated record key for the current spool session."""
        return self.session_key + (request_id, position)

    def reserve(self, key: DVIRecordKey) -> DVITelemetryTicket | None:
        """Reserve a capture slot for this position.

        Returns a ticket if the position is selected. The caller should then
        obtain the data and call submit(). If the caller cannot produce the
        data, it must call cancel() so the position is recorded as a capture
        failure.

        Canceled positions stay in the heap/sampler state and participate in
        the normal deterministic priority comparison. A later better candidate
        evicts them just like any other selected position, keeping stage_0 and
        stage_2 sample sets consistent.
        """
        session_key = key[:3]
        request_id = key[3]
        position = key[4]
        priority = self.sampler.on_position(session_key, request_id, position)
        if priority is None:
            return None

        buf = self._buffers.setdefault(
            session_key + (request_id,), _RequestBuffer()
        )
        max_n = self.sampler.config.max_per_request
        selected = {pos for _, pos in buf.heap}
        if position in selected:
            # Already selected (should not happen because the sampler dedups).
            self.sampler.drop_position(session_key, request_id, position)
            return None

        if len(buf.heap) < max_n:
            heapq.heappush(buf.heap, (-priority, position))
            buf.stage0[position] = None
            return DVITelemetryTicket(key=key, priority=priority)

        worst_neg, worst_pos = buf.heap[0]
        if priority < -worst_neg:
            # Evict the worst-priority position (canceled or not). Keep sampler
            # state synchronized so that both stage_0 and stage_2 finalize to
            # the same position set.
            self.sampler.drop_position(session_key, request_id, worst_pos)
            heapq.heapreplace(buf.heap, (-priority, position))
            buf.stage0.pop(worst_pos, None)
            buf.canceled.discard(worst_pos)
            buf.stage0[position] = None
            return DVITelemetryTicket(key=key, priority=priority)

        # New position is worse than the current worst selection; roll back.
        self.sampler.drop_position(session_key, request_id, position)
        return None

    def submit(
        self,
        ticket: DVITelemetryTicket,
        hidden: torch.Tensor,
    ) -> bool:
        """Submit the hidden tensor for a previously reserved position."""
        session_key = ticket.key[:3]
        request_id = ticket.key[3]
        position = ticket.key[4]
        buf = self._buffers.get(session_key + (request_id,))
        if buf is None:
            return False
        # The position may have been evicted by a later reserve or canceled.
        selected = {pos for _, pos in buf.heap}
        if position not in selected or position in buf.canceled:
            return False
        # NOTE: baseline path performs synchronous GPU->CPU copy here. The
        # production path will instead enqueue the GPU tensor with a CUDA event
        # and let the writer thread copy it later.
        buf.stage0[position] = (
            hidden.detach().to(device="cpu", dtype=torch.float32).contiguous()
        )
        return True

    def cancel(self, ticket: DVITelemetryTicket) -> bool:
        """Mark a reserved capture slot as failed.

        The position stays in the deterministic sample set and continues to
        compete by priority. Its payload is cleared so finalize_request skips
        it if it remains selected. A later better candidate evicts it normally,
        keeping stage_0 and stage_2 sample sets consistent.
        """
        session_key = ticket.key[:3]
        request_id = ticket.key[3]
        position = ticket.key[4]
        buf = self._buffers.get(session_key + (request_id,))
        if buf is None:
            return False
        selected = {pos for _, pos in buf.heap}
        if position not in selected or position in buf.canceled:
            return False
        buf.canceled.add(position)
        buf.stage0[position] = None
        self._capture_failed_count += 1
        return True

    def maybe_capture(
        self,
        key: DVIRecordKey,
        hidden: torch.Tensor,
    ) -> bool:
        """Convenience wrapper for callers that already have the tensor."""
        ticket = self.reserve(key)
        if ticket is None:
            return False
        return self.submit(ticket, hidden)

    def finalize_request(
        self,
        session_key: DVISessionKey,
        request_id: str,
    ) -> None:
        buf = self._buffers.pop(session_key + (request_id,), None)
        if buf is None:
            return
        selected_positions = set(
            self.sampler.finalize_request(session_key, request_id)
        )
        for position in selected_positions:
            hidden = buf.stage0.get(position)
            if hidden is None:
                continue
            key = session_key + (request_id, position)
            self.writer.enqueue_stage0(key, hidden)


class DVIStage2TelemetryProducer:
    """Capture stage_2 verifier outputs into a partial spool."""

    def __init__(
        self,
        writer: DVIPartialSpoolWriter,
        sampling_config: DVISamplingConfig,
    ) -> None:
        expected = _sampling_config_from_metadata(writer.spool_metadata)
        if sampling_config != expected:
            raise DVIArtifactError(
                f"DVIStage2TelemetryProducer sampling_config {sampling_config} "
                f"does not match spool metadata {expected}"
            )
        self.writer = writer
        self.sampler = DVIPerRequestSampler(sampling_config)
        self._buffers: dict[tuple[str, ...], _RequestBuffer] = {}
        self._capture_failed_count = 0

    @property
    def capture_failed_count(self) -> int:
        return self._capture_failed_count

    @property
    def session_key(self) -> DVISessionKey:
        return (
            str(self.writer.spool_metadata.get("run_id", "")),
            str(self.writer.spool_metadata.get("rollout_id", "")),
            str(self.writer.spool_metadata.get("policy_version", "")),
        )

    def make_record_key(self, request_id: str, position: int) -> DVIRecordKey:
        """Build a validated record key for the current spool session."""
        return self.session_key + (request_id, position)

    def reserve(self, key: DVIRecordKey) -> DVITelemetryTicket | None:
        """Reserve a capture slot before computing verifier top-k.

        Canceled positions stay in the heap/sampler state and compete by
        priority like any other selected position. A later better candidate
        evicts them normally, so stage_0 and stage_2 finalize to the same
        deterministic position set.
        """
        session_key = key[:3]
        request_id = key[3]
        position = key[4]
        priority = self.sampler.on_position(session_key, request_id, position)
        if priority is None:
            return None

        buf = self._buffers.setdefault(
            session_key + (request_id,), _RequestBuffer()
        )
        max_n = self.sampler.config.max_per_request
        selected = {pos for _, pos in buf.heap}
        if position in selected:
            self.sampler.drop_position(session_key, request_id, position)
            return None

        if len(buf.heap) < max_n:
            heapq.heappush(buf.heap, (-priority, position))
            buf.stage2[position] = None
            return DVITelemetryTicket(key=key, priority=priority)

        worst_neg, worst_pos = buf.heap[0]
        if priority < -worst_neg:
            self.sampler.drop_position(session_key, request_id, worst_pos)
            heapq.heapreplace(buf.heap, (-priority, position))
            buf.stage2.pop(worst_pos, None)
            buf.canceled.discard(worst_pos)
            buf.stage2[position] = None
            return DVITelemetryTicket(key=key, priority=priority)

        # New position is worse than the current worst selection; roll back.
        self.sampler.drop_position(session_key, request_id, position)
        return None

    def submit(
        self,
        ticket: DVITelemetryTicket,
        verifier_topk_ids: list[int],
        verifier_topk_logprobs: list[float],
        verifier_residual_mass: float,
        verifier_top1_id: int,
    ) -> bool:
        """Submit verifier top-k data for a previously reserved position."""
        session_key = ticket.key[:3]
        request_id = ticket.key[3]
        position = ticket.key[4]
        buf = self._buffers.get(session_key + (request_id,))
        if buf is None:
            return False
        selected = {pos for _, pos in buf.heap}
        if position not in selected or position in buf.canceled:
            return False
        buf.stage2[position] = {
            "verifier_topk_ids": tuple(verifier_topk_ids),
            "verifier_topk_logprobs": tuple(verifier_topk_logprobs),
            "verifier_residual_mass": verifier_residual_mass,
            "verifier_top1_id": verifier_top1_id,
        }
        return True

    def cancel(self, ticket: DVITelemetryTicket) -> bool:
        """Mark a reserved verifier slot as failed.

        The position stays in the deterministic sample set and continues to
        compete by priority. Its payload is cleared so finalize_request skips
        it if it remains selected. A later better candidate evicts it normally,
        keeping stage_0 and stage_2 sample sets consistent.
        """
        session_key = ticket.key[:3]
        request_id = ticket.key[3]
        position = ticket.key[4]
        buf = self._buffers.get(session_key + (request_id,))
        if buf is None:
            return False
        selected = {pos for _, pos in buf.heap}
        if position not in selected or position in buf.canceled:
            return False
        buf.canceled.add(position)
        buf.stage2[position] = None
        self._capture_failed_count += 1
        return True

    def maybe_capture(
        self,
        key: DVIRecordKey,
        verifier_topk_ids: list[int],
        verifier_topk_logprobs: list[float],
        verifier_residual_mass: float,
        verifier_top1_id: int,
    ) -> bool:
        """Convenience wrapper for callers that already have the data."""
        ticket = self.reserve(key)
        if ticket is None:
            return False
        return self.submit(
            ticket,
            verifier_topk_ids,
            verifier_topk_logprobs,
            verifier_residual_mass,
            verifier_top1_id,
        )

    def finalize_request(
        self,
        session_key: DVISessionKey,
        request_id: str,
        prompt_token_ids: list[int],
        response_token_ids: list[int],
    ) -> None:
        buf = self._buffers.pop(session_key + (request_id,), None)
        if buf is None:
            return
        self.writer.enqueue_request(request_id, prompt_token_ids, response_token_ids)
        selected_positions = set(
            self.sampler.finalize_request(session_key, request_id)
        )
        for position in selected_positions:
            rec = buf.stage2.get(position)
            if rec is None:
                continue
            key = session_key + (request_id, position)
            self.writer.enqueue_stage2(
                key,
                list(rec["verifier_topk_ids"]),
                list(rec["verifier_topk_logprobs"]),
                rec["verifier_residual_mass"],
                rec["verifier_top1_id"],
            )


def deterministic_hash(key: DVIRecordKey, salt: str = "") -> int:
    """Public helper for tests and external samplers."""
    return _deterministic_int(key, salt)


__all__ = [
    "DVIPartialSpoolWriter",
    "DVIPerRequestSampler",
    "DVISamplingConfig",
    "DVISessionKey",
    "DVIStage0TelemetryProducer",
    "DVIStage2TelemetryProducer",
    "DVITelemetryTicket",
    "deterministic_hash",
]
