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

from dataclasses import dataclass, field
from typing import Any

import msgspec
import numpy as np
import torch

from vllm.sequence import IntermediateTensors


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


@dataclass
class SplitTensorPacket:
    """Packet sent from stage_0 to stage_1 and from stage_1 to stage_2.

    Contains the intermediate activation tensors (hidden_states + residual) plus
    the minimal metadata needed by the receiving stage to reconstruct the input
    batch.
    """

    req_ids: list[str]
    num_scheduled_tokens: list[int]
    is_prompt: bool
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)

    def to_intermediate_tensors(self) -> IntermediateTensors:
        return IntermediateTensors(self.tensors)

    @classmethod
    def from_intermediate_tensors(
        cls,
        req_ids: list[str],
        num_scheduled_tokens: list[int],
        is_prompt: bool,
        intermediate_tensors: IntermediateTensors,
    ) -> "SplitTensorPacket":
        return cls(
            req_ids=req_ids,
            num_scheduled_tokens=num_scheduled_tokens,
            is_prompt=is_prompt,
            tensors=intermediate_tensors.tensors,
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
        )


@dataclass
class SplitTokenPacket:
    """Packet sent from stage_2 back to stage_0.

    Contains the sampled tokens and finish reasons for each request in the
    batch.  Logprobs and other optional fields are left out in Phase 1.
    """

    req_ids: list[str]
    sampled_token_ids: list[list[int]]
    finish_reasons: list[str | None] = field(default_factory=list)

    def serialize(self) -> bytes:
        data: dict[str, Any] = {
            "req_ids": self.req_ids,
            "sampled_token_ids": self.sampled_token_ids,
            "finish_reasons": self.finish_reasons,
        }
        return msgspec.msgpack.encode(data)

    @classmethod
    def deserialize(cls, data: bytes) -> "SplitTokenPacket":
        decoded = msgspec.msgpack.decode(data)
        return cls(
            req_ids=decoded["req_ids"],
            sampled_token_ids=decoded["sampled_token_ids"],
            finish_reasons=decoded.get("finish_reasons", []),
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
            raise ValueError(
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
            raise ValueError(
                f"Expected {num_reqs} sampled token entries, got "
                f"{len(self.sampled_token_ids)}"
            )
        for i, ids in enumerate(self.sampled_token_ids):
            if not ids:
                raise ValueError(f"Empty sampled token list at index {i}")
            tensor[i, 0] = ids[0]
        return tensor


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
        raise ValueError(f"Unsupported dtype for split transport: {dtype_str}")
    return mapping[dtype_str]
