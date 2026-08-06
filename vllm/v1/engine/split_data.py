# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Data structures for layer-wise split (stage_0 -> stage_1 -> stage_2).

Provides ``SplitTensorPacket`` (intermediate activations + batch metadata) and
``SplitTokenPacket`` (sampled tokens + finish reasons).  Serialization uses
``msgspec.msgpack`` for metadata and raw bytes for tensor payloads; bfloat16 is
handled via a uint16 view since numpy has no native bfloat16 dtype.

Deferred: logprobs, spec_token_ids, pooler_output, KV cache lifecycle/free
metadata (KV cache is not transferred across stages).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import msgspec
import numpy as np
import torch

from vllm.sequence import IntermediateTensors


class SplitPacketKind(str, Enum):
    """Kind of a split tensor/token packet.

    NORMAL: baseline per-step packets (single boundary forward).
    DVI_BLOCK: Stage-DVI draft/verify block packets carrying draft metadata
    alongside the per-position intermediate tensors.
    """

    NORMAL = "normal"
    DVI_BLOCK = "dvi_block"


class SplitDVIProtocolError(RuntimeError):
    """Fatal Stage-DVI protocol invariant violation.

    Raised on any split worker when cross-stage state provably diverges:
    packet schema/kind misuse, cycle or generation mismatch, awaiting-state
    violations, request identity mismatch, or illegal commit/rollback
    bookkeeping.  Unlike ordinary worker errors this must terminate the
    whole engine: continuing after a protocol desync silently corrupts
    KV/token state on every stage (see the mixed-packet incident in
    docs/PROGRESS.md).

    Non-protocol conditions must NOT raise this type: normal EOS, max
    model length, user cancellation, request timeout, scheduler
    preemption, empty draft, and sampling-admission failures are ordinary
    control flow.
    """

    def __init__(
        self,
        reason: str,
        *,
        stage_id: str | None = None,
        request_id: str | None = None,
        generation_id: int | None = None,
        cycle_id: int | None = None,
        expected: object = None,
        actual: object = None,
    ) -> None:
        self.stage_id = stage_id
        self.request_id = request_id
        self.generation_id = generation_id
        self.cycle_id = cycle_id
        self.expected = expected
        self.actual = actual
        ctx = []
        if stage_id is not None:
            ctx.append(f"stage={stage_id}")
        if request_id is not None:
            ctx.append(f"req={request_id!r}")
        if generation_id is not None:
            ctx.append(f"gen={generation_id}")
        if cycle_id is not None:
            ctx.append(f"cycle={cycle_id}")
        if expected is not None or actual is not None:
            ctx.append(f"expected={expected!r} actual={actual!r}")
        suffix = f" ({', '.join(ctx)})" if ctx else ""
        super().__init__(f"SplitDVI protocol error: {reason}{suffix}")


class _TensorDescriptor(msgspec.Struct, array_like=True):
    """Lightweight descriptor for one tensor in a SplitTensorPacket."""

    key: str
    shape: list[int]
    dtype: str
    num_bytes: int


class _SplitTensorPacketSerialized(msgspec.Struct):
    """Serialized representation of SplitTensorPacket."""

    req_ids: list[str]
    num_scheduled_tokens: list[int]
    is_prompt: bool
    descriptors: list[_TensorDescriptor]
    packet_kind: str = SplitPacketKind.NORMAL.value
    # DVI block metadata (present iff packet_kind == DVI_BLOCK).
    cycle_ids: list[int] | None = None
    draft_token_ids: list[int] | None = None
    draft_lengths: list[int] | None = None
    # Schema v2: lifecycle epoch per request (scheduler-issued), block-row
    # positions, and version contracts (unversioned in v1).
    generation_ids: list[int] | None = None
    draft_positions: list[int] | None = None
    policy_version: str | None = None
    draft_version: str | None = None
    sampling_mode: str | None = None
    draft_support_offsets: list[int] | None = None
    draft_support_token_ids: list[int] | None = None
    draft_support_logits: list[float] | None = None
    is_fallback: bool = False


@dataclass
class SplitTensorPacket:
    """Packet sent from stage_0 to stage_1 and from stage_1 to stage_2.

    Contains the intermediate activation tensors (hidden_states + residual) plus
    the minimal metadata needed by the receiving stage to reconstruct the input
    batch.

    Stage-DVI: when ``packet_kind == DVI_BLOCK`` the tensors hold the full
    draft block (one row per draft position, ``sum(num_scheduled_tokens)``
    rows) and the DVI metadata fields describe the block: ``cycle_ids`` per
    request, ``draft_token_ids`` (flat, ``sum(draft_lengths)`` entries) and
    ``draft_lengths`` per request.  ``draft_lengths[r]`` equals the number of
    logit rows for decode requests (k proposals) and 0 for prefilling ones.
    """

    req_ids: list[str]
    num_scheduled_tokens: list[int]
    is_prompt: bool
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    packet_kind: str = SplitPacketKind.NORMAL.value
    cycle_ids: list[int] | None = None
    draft_token_ids: list[int] | None = None
    draft_lengths: list[int] | None = None
    # Schema v2 fields.
    generation_ids: list[int] | None = None
    draft_positions: list[int] | None = None
    policy_version: str | None = None
    draft_version: str | None = None
    sampling_mode: str | None = None
    draft_support_offsets: list[int] | None = None
    draft_support_token_ids: list[int] | None = None
    draft_support_logits: list[float] | None = None
    is_fallback: bool = False

    @property
    def is_dvi_block(self) -> bool:
        return self.packet_kind == SplitPacketKind.DVI_BLOCK.value

    def to_intermediate_tensors(self) -> IntermediateTensors:
        return IntermediateTensors(self.tensors)

    @classmethod
    def from_intermediate_tensors(
        cls,
        req_ids: list[str],
        num_scheduled_tokens: list[int],
        is_prompt: bool,
        intermediate_tensors: IntermediateTensors,
        packet_kind: str = SplitPacketKind.NORMAL.value,
        cycle_ids: list[int] | None = None,
        draft_token_ids: list[int] | None = None,
        draft_lengths: list[int] | None = None,
        generation_ids: list[int] | None = None,
        draft_positions: list[int] | None = None,
        policy_version: str | None = None,
        draft_version: str | None = None,
        sampling_mode: str | None = None,
        draft_support_offsets: list[int] | None = None,
        draft_support_token_ids: list[int] | None = None,
        draft_support_logits: list[float] | None = None,
        is_fallback: bool = False,
    ) -> "SplitTensorPacket":
        return cls(
            req_ids=req_ids,
            num_scheduled_tokens=num_scheduled_tokens,
            is_prompt=is_prompt,
            tensors=intermediate_tensors.tensors,
            packet_kind=packet_kind,
            cycle_ids=cycle_ids,
            draft_token_ids=draft_token_ids,
            draft_lengths=draft_lengths,
            generation_ids=generation_ids,
            draft_positions=draft_positions,
            policy_version=policy_version,
            draft_version=draft_version,
            sampling_mode=sampling_mode,
            draft_support_offsets=draft_support_offsets,
            draft_support_token_ids=draft_support_token_ids,
            draft_support_logits=draft_support_logits,
            is_fallback=is_fallback,
        )

    def validate(
        self,
        expected_req_ids: list[str] | None = None,
        vocab_size: int | None = None,
        max_draft_length: int | None = None,
    ) -> None:
        """Fail-fast structural validation.  Any inconsistency indicates
        cross-stage desync, so raise immediately instead of guessing."""
        if expected_req_ids is not None and self.req_ids != expected_req_ids:
            raise SplitDVIProtocolError(
                f"SplitTensorPacket req_ids mismatch: expected "
                f"{expected_req_ids!r}, got {self.req_ids!r}"
            )
        if len(self.req_ids) != len(self.num_scheduled_tokens):
            raise SplitDVIProtocolError(
                f"SplitTensorPacket req_ids/num_scheduled_tokens length "
                f"mismatch: {len(self.req_ids)} vs "
                f"{len(self.num_scheduled_tokens)}"
            )
        if not self.is_dvi_block:
            if (
                self.cycle_ids is not None
                or self.draft_token_ids is not None
                or self.draft_lengths is not None
                or self.generation_ids is not None
                or self.draft_positions is not None
                or self.policy_version is not None
                or self.draft_version is not None
                or self.sampling_mode is not None
                or self.draft_support_offsets is not None
                or self.draft_support_token_ids is not None
                or self.draft_support_logits is not None
                or self.is_fallback
            ):
                raise SplitDVIProtocolError(
                    "SplitTensorPacket kind is NORMAL but DVI metadata is set"
                )
            return

        # DVI block validation.
        if self.cycle_ids is None or self.draft_lengths is None:
            raise SplitDVIProtocolError("DVI block packet missing cycle_ids/draft_lengths")
        if self.draft_token_ids is None:
            raise SplitDVIProtocolError("DVI block packet missing draft_token_ids")
        if self.generation_ids is None:
            raise SplitDVIProtocolError("DVI block packet missing generation_ids")
        if self.draft_positions is None:
            raise SplitDVIProtocolError("DVI block packet missing draft_positions")
        if len(self.cycle_ids) != len(self.req_ids):
            raise SplitDVIProtocolError(
                f"DVI block cycle_ids length {len(self.cycle_ids)} != "
                f"num reqs {len(self.req_ids)}"
            )
        if len(self.generation_ids) != len(self.req_ids):
            raise SplitDVIProtocolError(
                f"DVI block generation_ids length {len(self.generation_ids)} "
                f"!= num reqs {len(self.req_ids)}"
            )
        for gen_id in self.generation_ids:
            if gen_id < 0:
                raise SplitDVIProtocolError(f"Negative generation_id {gen_id}")
        for cycle_id in self.cycle_ids:
            if cycle_id < 0:
                raise SplitDVIProtocolError(f"Negative cycle_id {cycle_id}")
        if len(self.draft_lengths) != len(self.req_ids):
            raise SplitDVIProtocolError(
                f"DVI block draft_lengths length {len(self.draft_lengths)} != "
                f"num reqs {len(self.req_ids)}"
            )
        if sum(self.draft_lengths) != len(self.draft_token_ids):
            raise SplitDVIProtocolError(
                f"DVI block draft_token_ids has {len(self.draft_token_ids)} "
                f"entries but draft_lengths sum to {sum(self.draft_lengths)}"
            )
        if len(self.draft_positions) != len(self.draft_token_ids):
            raise SplitDVIProtocolError(
                f"DVI block draft_positions has {len(self.draft_positions)} "
                f"entries but draft_token_ids has {len(self.draft_token_ids)}"
            )
        for draft_len in self.draft_lengths:
            if draft_len < 0:
                raise SplitDVIProtocolError(f"Negative draft_length {draft_len}")
            if max_draft_length is not None and draft_len > max_draft_length:
                raise SplitDVIProtocolError(
                    f"draft_length {draft_len} exceeds max_draft_length "
                    f"{max_draft_length}"
                )
        # Per-request draft positions must be strictly increasing.
        offset = 0
        for r, draft_len in enumerate(self.draft_lengths):
            segment = self.draft_positions[offset : offset + draft_len]
            for i in range(1, len(segment)):
                if segment[i] <= segment[i - 1]:
                    raise SplitDVIProtocolError(
                        f"DVI block draft_positions for req {r} not strictly "
                        f"increasing: {segment}"
                    )
            offset += draft_len
        if vocab_size is not None:
            for token_id in self.draft_token_ids:
                if not 0 <= token_id < vocab_size:
                    raise SplitDVIProtocolError(
                        f"Draft token id {token_id} out of vocab range "
                        f"[0, {vocab_size})"
                    )
        sampling_mode = self.sampling_mode or "greedy"
        support_fields = (
            self.draft_support_offsets,
            self.draft_support_token_ids,
            self.draft_support_logits,
        )
        if sampling_mode == "greedy":
            if any(field is not None for field in support_fields):
                raise SplitDVIProtocolError(
                    "Greedy DVI block must not carry stochastic support fields"
                )
        elif sampling_mode == "stochastic":
            if any(field is None for field in support_fields):
                raise SplitDVIProtocolError(
                    "Stochastic DVI block missing draft support fields"
                )
            assert self.draft_support_offsets is not None
            assert self.draft_support_token_ids is not None
            assert self.draft_support_logits is not None
            offsets = self.draft_support_offsets
            if len(offsets) != len(self.draft_token_ids) + 1:
                raise SplitDVIProtocolError(
                    "draft_support_offsets length must equal "
                    "len(draft_token_ids) + 1"
                )
            if not offsets or offsets[0] != 0:
                raise SplitDVIProtocolError(
                    "draft_support_offsets must start at zero"
                )
            if any(b <= a for a, b in zip(offsets, offsets[1:])):
                raise SplitDVIProtocolError(
                    "draft_support_offsets must be strictly increasing"
                )
            if offsets[-1] != len(self.draft_support_token_ids):
                raise SplitDVIProtocolError(
                    "draft support offset terminus does not match token ids"
                )
            if len(self.draft_support_token_ids) != len(
                self.draft_support_logits
            ):
                raise SplitDVIProtocolError(
                    "draft support token ids/logits length mismatch"
                )
            if not all(math.isfinite(x) for x in self.draft_support_logits):
                raise SplitDVIProtocolError(
                    "draft support logits must all be finite"
                )
            if vocab_size is not None:
                for token_id in self.draft_support_token_ids:
                    if not 0 <= token_id < vocab_size:
                        raise SplitDVIProtocolError(
                            f"Draft support token id {token_id} out of vocab "
                            f"range [0, {vocab_size})"
                        )
        else:
            raise SplitDVIProtocolError(
                f"Unsupported DVI sampling_mode {sampling_mode!r}"
            )
        num_rows = sum(self.num_scheduled_tokens)
        for key, tensor in self.tensors.items():
            if tensor.shape[0] != num_rows:
                raise SplitDVIProtocolError(
                    f"DVI block tensor {key!r} has {tensor.shape[0]} rows, "
                    f"expected {num_rows}"
                )

    def serialize(self) -> list[bytes]:
        """Serialize to ZMQ-style multipart message.

        Returns [metadata_frame, tensor_bytes_frame].
        """
        descriptors: list[_TensorDescriptor] = []
        tensor_bytes_parts: list[bytes] = []
        total_tensor_bytes = 0

        for key in sorted(self.tensors.keys()):
            tensor = self.tensors[key]
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            cpu_tensor = tensor.detach().cpu()
            dtype_str = str(cpu_tensor.dtype).replace("torch.", "")
            if cpu_tensor.dtype == torch.bfloat16:
                # numpy does not support bfloat16; view as uint16.
                cpu_tensor = cpu_tensor.view(torch.uint16)
            buf = cpu_tensor.numpy().tobytes()
            descriptors.append(
                _TensorDescriptor(
                    key=key,
                    shape=list(tensor.shape),
                    dtype=dtype_str,
                    num_bytes=len(buf),
                )
            )
            tensor_bytes_parts.append(buf)
            total_tensor_bytes += len(buf)

        metadata = _SplitTensorPacketSerialized(
            req_ids=self.req_ids,
            num_scheduled_tokens=self.num_scheduled_tokens,
            is_prompt=self.is_prompt,
            descriptors=descriptors,
            packet_kind=self.packet_kind,
            cycle_ids=self.cycle_ids,
            draft_token_ids=self.draft_token_ids,
            draft_lengths=self.draft_lengths,
            generation_ids=self.generation_ids,
            draft_positions=self.draft_positions,
            policy_version=self.policy_version,
            draft_version=self.draft_version,
            sampling_mode=self.sampling_mode,
            draft_support_offsets=self.draft_support_offsets,
            draft_support_token_ids=self.draft_support_token_ids,
            draft_support_logits=self.draft_support_logits,
            is_fallback=self.is_fallback,
        )
        metadata_bytes = msgspec.msgpack.encode(metadata)

        # Single contiguous bytes frame for all tensors.
        tensor_bytes = b"".join(tensor_bytes_parts)
        assert len(tensor_bytes) == total_tensor_bytes
        return [metadata_bytes, tensor_bytes]

    @classmethod
    def deserialize(
        cls,
        frames: list[bytes],
        device: torch.device | str = "cpu",
    ) -> "SplitTensorPacket":
        """Deserialize from ZMQ-style multipart message."""
        metadata_bytes, tensor_bytes = frames[0], frames[1]
        metadata = msgspec.msgpack.decode(
            metadata_bytes, type=_SplitTensorPacketSerialized
        )

        tensors: dict[str, torch.Tensor] = {}
        offset = 0
        for desc in metadata.descriptors:
            buf = tensor_bytes[offset : offset + desc.num_bytes]
            offset += desc.num_bytes

            np_dtype = _dtype_str_to_numpy(desc.dtype)
            # Copy the buffer into a writable numpy array to avoid the
            # non-writable tensor warning from torch.from_numpy.
            arr = np.frombuffer(buf, dtype=np_dtype).copy().reshape(desc.shape)
            tensor = torch.from_numpy(arr)
            if desc.dtype == "bfloat16":
                tensor = tensor.view(torch.bfloat16)
            tensors[desc.key] = tensor.to(device)

        return cls(
            req_ids=metadata.req_ids,
            num_scheduled_tokens=metadata.num_scheduled_tokens,
            is_prompt=metadata.is_prompt,
            tensors=tensors,
            packet_kind=metadata.packet_kind,
            cycle_ids=metadata.cycle_ids,
            draft_token_ids=metadata.draft_token_ids,
            draft_lengths=metadata.draft_lengths,
            generation_ids=metadata.generation_ids,
            draft_positions=metadata.draft_positions,
            policy_version=metadata.policy_version,
            draft_version=metadata.draft_version,
            sampling_mode=metadata.sampling_mode,
            draft_support_offsets=metadata.draft_support_offsets,
            draft_support_token_ids=metadata.draft_support_token_ids,
            draft_support_logits=metadata.draft_support_logits,
            is_fallback=metadata.is_fallback,
        )


@dataclass
class SplitTokenPacket:
    """Packet sent from stage_2 back to stage_0/stage_1.

    Contains the sampled tokens and optional per-request metadata
    (``num_sampled``, ``num_rejected``) needed by the V2 model runner.
    Logprobs and other optional fields are left out in Phase 1.

    Stage-DVI: ``cycle_ids`` echoes the DVI block cycle each answer belongs
    to (used for fail-fast cross-stage desync detection) and
    ``accepted_counts`` records how many draft tokens were accepted per
    request (debug/metrics only; the commit semantics are driven by
    ``num_sampled``/``num_rejected``).
    """

    req_ids: list[str]
    sampled_token_ids: list[list[int]]
    finish_reasons: list[str | None] = field(default_factory=list)
    num_sampled: list[int] | None = None
    num_rejected: list[int] | None = None
    packet_kind: str = SplitPacketKind.NORMAL.value
    cycle_ids: list[int] | None = None
    accepted_counts: list[int] | None = None
    # Schema v2 echo fields: must match the DVI block they answer.
    generation_ids: list[int] | None = None
    policy_version: str | None = None
    draft_version: str | None = None

    @property
    def is_dvi_block(self) -> bool:
        return self.packet_kind == SplitPacketKind.DVI_BLOCK.value

    def serialize(self) -> bytes:
        data: dict[str, Any] = {
            "req_ids": self.req_ids,
            "sampled_token_ids": self.sampled_token_ids,
            "finish_reasons": self.finish_reasons,
            "packet_kind": self.packet_kind,
        }
        if self.num_sampled is not None:
            data["num_sampled"] = self.num_sampled
        if self.num_rejected is not None:
            data["num_rejected"] = self.num_rejected
        if self.cycle_ids is not None:
            data["cycle_ids"] = self.cycle_ids
        if self.accepted_counts is not None:
            data["accepted_counts"] = self.accepted_counts
        if self.generation_ids is not None:
            data["generation_ids"] = self.generation_ids
        if self.policy_version is not None:
            data["policy_version"] = self.policy_version
        if self.draft_version is not None:
            data["draft_version"] = self.draft_version
        return msgspec.msgpack.encode(data)

    @classmethod
    def deserialize(cls, data: bytes) -> "SplitTokenPacket":
        decoded = msgspec.msgpack.decode(data)
        return cls(
            req_ids=decoded["req_ids"],
            sampled_token_ids=decoded["sampled_token_ids"],
            finish_reasons=decoded.get("finish_reasons", []),
            num_sampled=decoded.get("num_sampled"),
            num_rejected=decoded.get("num_rejected"),
            packet_kind=decoded.get("packet_kind", SplitPacketKind.NORMAL.value),
            cycle_ids=decoded.get("cycle_ids"),
            accepted_counts=decoded.get("accepted_counts"),
            generation_ids=decoded.get("generation_ids"),
            policy_version=decoded.get("policy_version"),
            draft_version=decoded.get("draft_version"),
        )

    @classmethod
    def from_token_tensor(
        cls, req_ids: list[str], token_tensor: torch.Tensor
    ) -> "SplitTokenPacket":
        """Create a packet from the GPU tensor produced by the sampler.

        ``token_tensor`` is expected to have shape ``[num_reqs, 1]`` and dtype
        ``torch.int32``, matching the format used by
        ``_pp_broadcast_prev_sampled_token_ids``.
        """
        if token_tensor.dim() != 2 or token_tensor.shape[-1] != 1:
            raise SplitDVIProtocolError(
                f"Expected token_tensor shape [num_reqs, 1], got {token_tensor.shape}"
            )
        token_ids = token_tensor.squeeze(-1).tolist()
        return cls(
            req_ids=req_ids,
            sampled_token_ids=[[t] for t in token_ids],
            finish_reasons=[],
        )

    def to_token_tensor(
        self, device: torch.device | str, num_reqs: int
    ) -> torch.Tensor:
        """Reconstruct the GPU tensor expected by the runner.

        Returns a tensor of shape ``[num_reqs, 1]`` and dtype ``torch.int32``.
        """
        tensor = torch.zeros((num_reqs, 1), dtype=torch.int32, device=device)
        if len(self.sampled_token_ids) != num_reqs:
            raise SplitDVIProtocolError(
                f"Expected {num_reqs} sampled token entries, got "
                f"{len(self.sampled_token_ids)}"
            )
        for i, ids in enumerate(self.sampled_token_ids):
            if not ids:
                raise SplitDVIProtocolError(f"Empty sampled token list at index {i}")
            tensor[i, 0] = ids[0]
        return tensor

    @classmethod
    def from_tensors(
        cls,
        req_ids: list[str],
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        packet_kind: str = SplitPacketKind.NORMAL.value,
        cycle_ids: list[int] | None = None,
        accepted_counts: list[int] | None = None,
        generation_ids: list[int] | None = None,
        policy_version: str | None = None,
        draft_version: str | None = None,
    ) -> "SplitTokenPacket":
        """Create a packet from the V2 model runner's sampled-token tensors.

        ``sampled_token_ids`` has shape ``[num_reqs, max_sample_len]`` and dtype
        ``torch.int64``.  ``num_sampled`` and ``num_rejected`` have shape
        ``[num_reqs]`` and dtype ``torch.int32``.
        """
        if sampled_token_ids.dim() != 2:
            raise SplitDVIProtocolError(
                f"Expected sampled_token_ids shape [num_reqs, max_sample_len], "
                f"got {sampled_token_ids.shape}"
            )
        token_ids = sampled_token_ids.tolist()
        return cls(
            req_ids=req_ids,
            sampled_token_ids=token_ids,
            finish_reasons=[],
            num_sampled=num_sampled.tolist(),
            num_rejected=num_rejected.tolist(),
            packet_kind=packet_kind,
            cycle_ids=cycle_ids,
            accepted_counts=accepted_counts,
            generation_ids=generation_ids,
            policy_version=policy_version,
            draft_version=draft_version,
        )

    def validate_req_ids(self, expected_req_ids: list[str]) -> None:
        """Fail fast if the packet's request ids do not match the local batch.

        This guards against associating sampled tokens with the wrong requests
        when driver/worker state diverges across the split TCP transport.
        """
        if self.req_ids != expected_req_ids:
            raise SplitDVIProtocolError(
                f"SplitTokenPacket req_ids mismatch: expected "
                f"{expected_req_ids!r}, got {self.req_ids!r}"
            )

    def validate_dvi(
        self,
        expected_cycle_ids: list[int] | None = None,
        expected_generation_ids: list[int] | None = None,
        expected_policy_version: str | None = None,
        expected_draft_version: str | None = None,
    ) -> None:
        """Fail-fast validation of DVI token packet semantics.

        The ``expected_*`` parameters pin the packet to the block it must
        answer: local per-request cycle counters, scheduler-issued generation
        epochs, and the version contracts of the outgoing block.  Any
        mismatch indicates cross-stage desync or a stale packet from an
        earlier lifecycle, and must abort rather than silently apply results.
        """
        if not self.is_dvi_block:
            return
        num_reqs = len(self.req_ids)
        if len(self.sampled_token_ids) != num_reqs:
            raise SplitDVIProtocolError(
                f"DVI token packet has {len(self.sampled_token_ids)} sampled "
                f"entries for {num_reqs} requests"
            )
        if self.num_sampled is None or self.num_rejected is None:
            raise SplitDVIProtocolError("DVI token packet missing num_sampled/num_rejected")
        if len(self.num_sampled) != num_reqs or len(self.num_rejected) != num_reqs:
            raise SplitDVIProtocolError(
                "DVI token packet num_sampled/num_rejected length mismatch"
            )
        if len(set(self.req_ids)) != num_reqs:
            raise SplitDVIProtocolError("DVI token packet contains duplicate req_ids")
        for i, ids in enumerate(self.sampled_token_ids):
            if len(ids) < self.num_sampled[i]:
                raise SplitDVIProtocolError(
                    f"DVI token packet req {i}: num_sampled="
                    f"{self.num_sampled[i]} exceeds padded token list "
                    f"(len={len(ids)})"
                )
            if self.num_sampled[i] < 0:
                raise SplitDVIProtocolError(
                    f"DVI token packet req {i}: negative num_sampled"
                )
            if self.num_rejected[i] < 0:
                raise SplitDVIProtocolError(
                    f"DVI token packet req {i}: negative num_rejected"
                )
        if self.cycle_ids is None:
            raise SplitDVIProtocolError("DVI token packet missing cycle_ids")
        if len(self.cycle_ids) != num_reqs:
            raise SplitDVIProtocolError("DVI token packet cycle_ids length mismatch")
        if (
            expected_cycle_ids is not None
            and self.cycle_ids != expected_cycle_ids
        ):
            raise SplitDVIProtocolError(
                f"DVI token packet cycle mismatch: expected "
                f"{expected_cycle_ids}, got {self.cycle_ids}"
            )
        if self.generation_ids is None:
            raise SplitDVIProtocolError("DVI token packet missing generation_ids")
        if len(self.generation_ids) != num_reqs:
            raise SplitDVIProtocolError("DVI token packet generation_ids length mismatch")
        if (
            expected_generation_ids is not None
            and self.generation_ids != expected_generation_ids
        ):
            raise SplitDVIProtocolError(
                f"DVI token packet generation mismatch: expected "
                f"{expected_generation_ids}, got {self.generation_ids}"
            )
        if (
            expected_policy_version is not None
            and self.policy_version != expected_policy_version
        ):
            raise SplitDVIProtocolError(
                f"DVI token packet policy_version mismatch: expected "
                f"{expected_policy_version!r}, got {self.policy_version!r}"
            )
        if (
            expected_draft_version is not None
            and self.draft_version != expected_draft_version
        ):
            raise SplitDVIProtocolError(
                f"DVI token packet draft_version mismatch: expected "
                f"{expected_draft_version!r}, got {self.draft_version!r}"
            )

    def to_tensors(
        self,
        device: torch.device | str,
        num_reqs: int,
        max_sample_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct the V2 sampled-token tensors.

        Returns ``(sampled_token_ids [num_reqs, max_sample_len] int64,
        num_sampled [num_reqs] int32, num_rejected [num_reqs] int32)``.
        """
        sampled_tokens = torch.zeros(
            (num_reqs, max_sample_len), dtype=torch.int64, device=device
        )
        if len(self.sampled_token_ids) != num_reqs:
            raise SplitDVIProtocolError(
                f"Expected {num_reqs} sampled token entries, got "
                f"{len(self.sampled_token_ids)}"
            )
        for i, ids in enumerate(self.sampled_token_ids):
            if len(ids) > max_sample_len:
                raise SplitDVIProtocolError(
                    f"Sampled token list at index {i} has length {len(ids)}, "
                    f"exceeding max_sample_len={max_sample_len}"
                )
            sampled_tokens[i, : len(ids)] = torch.tensor(
                ids, dtype=torch.int64, device=device
            )

        def _to_int32_tensor(values: list[int] | None) -> torch.Tensor:
            tensor = torch.zeros((num_reqs,), dtype=torch.int32, device=device)
            if values is not None:
                if len(values) != num_reqs:
                    raise SplitDVIProtocolError(
                        f"Expected {num_reqs} metadata entries, got {len(values)}"
                    )
                tensor.copy_(
                    torch.tensor(values, dtype=torch.int32, device=device)
                )
            return tensor

        return (
            sampled_tokens,
            _to_int32_tensor(self.num_sampled),
            _to_int32_tensor(self.num_rejected),
        )


def _dtype_str_to_numpy(dtype_str: str) -> Any:
    """Map a torch dtype string to a numpy dtype suitable for frombuffer.

    bfloat16 does not have a native numpy dtype, so we view it as uint16 and
    convert back to torch.bfloat16 after deserialization.
    """
    mapping = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "bfloat16": np.uint16,
    }
    if dtype_str not in mapping:
        raise SplitDVIProtocolError(f"Unsupported dtype for split transport: {dtype_str}")
    return mapping[dtype_str]
