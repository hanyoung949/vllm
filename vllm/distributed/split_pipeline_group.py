# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pipeline-parallel group replacement for Edge-Cloud layer-wise split.

When ``enable_layerwise_split`` is set, vLLM's native NCCL-based pipeline
parallel group is replaced by this class.  It keeps the same rank/world-size
abstraction as :class:`GroupCoordinator` but routes ``IntermediateTensors``
between stage_0/stage_1/stage_2 over the configured ``SplitActivationTransport`` instead of
NCCL.

This lets the rest of vLLM (scheduler, model runner, executor) believe it is
running a normal ``pipeline_parallel_size=3`` job while the actual inter-stage
activation communication goes through TCP/ZMQ (or another backend).  In Phase 2,
sampled token ids are also rerouted over the custom transport so that split
mode no longer depends on NCCL P2P/collectives for stage-to-stage traffic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import os
import time

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.split_trace import get_split_trace_logger
from vllm.v1.engine.split_data import SplitTensorPacket, SplitTokenPacket
from vllm.v1.engine.split_transport import (
    LocalLoopbackTransport,
    SplitActivationTransport,
    SplitTransportEndpoints,
    ZmqSplitActivationTransport,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = init_logger(__name__)


def _tensor_dict_bytes(tensor_dict: dict[str, torch.Tensor]) -> int:
    """Return total bytes of all tensors in the dict."""
    return sum(t.numel() * t.element_size() for t in tensor_dict.values())


def _parse_recv_addrs(env_var: str | None) -> list[str] | None:
    """Read a comma-separated list of addresses from an env var."""
    if not env_var:
        return None
    return [a.strip() for a in env_var.split(",")]


def _make_split_transport(
    rank: int, world_size: int, tensor_recv_addrs: list[str]
) -> SplitActivationTransport:
    """Create a transport that sends tensors to the next split stage."""
    assert len(tensor_recv_addrs) == world_size, (
        f"VLLM_SPLIT_TENSOR_RECV_ADDRS must contain {world_size} addresses, "
        f"got {len(tensor_recv_addrs)}"
    )

    # For unit tests we allow the special value ``local`` to use the in-memory
    # loopback transport.
    if tensor_recv_addrs[0] == "local":
        assert world_size == 2, "local loopback transport is only for 2-rank tests"
        pair = LocalLoopbackTransport.make_pair()
        return pair[rank]

    endpoints = SplitTransportEndpoints(
        tensor_recv_addr=tensor_recv_addrs[rank],
        tensor_send_addr=tensor_recv_addrs[(rank + 1) % world_size],
    )
    return ZmqSplitActivationTransport(endpoints, stage_label=f"PP{rank}")


def _make_token_transport(
    rank: int,
    world_size: int,
    token_recv_addrs: list[str] | None,
) -> SplitActivationTransport | None:
    """Create a transport for sampled token broadcast from stage_2 to stage_0/stage_1.

    ``token_recv_addrs`` is required when split mode is enabled; the caller
    (``SplitPipelineGroup.__init__``) raises before calling this function if the
    environment variable is missing.
    """
    if token_recv_addrs is None:
        return None

    assert len(token_recv_addrs) == world_size, (
        f"VLLM_SPLIT_TOKEN_RECV_ADDRS must contain {world_size} addresses, "
        f"got {len(token_recv_addrs)}"
    )

    # For unit tests we allow the special value ``local``.
    if token_recv_addrs[0] == "local":
        assert world_size == 3, "local token loopback is only for 3-rank tests"
        s0, s1, s2 = LocalLoopbackTransport.make_star()
        return [s0, s1, s2][rank]

    recv_addr = token_recv_addrs[rank]
    # stage_2 sends to all earlier ranks: stage_0 (rank 0) and stage_1 (rank 1).
    send_addrs = [
        token_recv_addrs[i]
        for i in range(world_size)
        if i != rank and i != world_size - 1
    ] if rank == world_size - 1 else []

    endpoints = SplitTransportEndpoints(
        token_recv_addr=recv_addr,
        token_send_addrs=send_addrs,
    )
    return ZmqSplitActivationTransport(endpoints, stage_label=f"PP{rank}")


class SplitPipelineGroup:
    """Wrapper around a real pipeline-parallel group for layer-wise split.

    Most attributes and methods are forwarded to the underlying
    :class:`GroupCoordinator` so that small metadata collectives (graph capture,
    weight broadcast during init, etc.) continue to work over NCCL.  Only the
    stage-to-stage traffic (activation tensor dicts and sampled token ids) is
    overridden to use the custom transport.
    """

    def __init__(
        self,
        base_group: Any,
        rank: int,
        world_size: int,
        stage_label: str,
        is_representative: bool = True,
        tensor_recv_addrs: list[str] | None = None,
        token_recv_addrs: list[str] | None = None,
    ) -> None:
        self._base = base_group
        self.rank = rank
        self.world_size = world_size
        self._stage_label = stage_label
        os.environ.setdefault("SPLIT_STAGE", stage_label)
        self._is_representative = is_representative
        self._trace = get_split_trace_logger("rollout_comm")

        if is_representative:
            if tensor_recv_addrs is None:
                tensor_recv_addrs = _parse_recv_addrs(
                    envs.VLLM_SPLIT_TENSOR_RECV_ADDRS
                )
            if tensor_recv_addrs is None:
                raise RuntimeError(
                    "VLLM_SPLIT_TENSOR_RECV_ADDRS must be set when "
                    "enable_layerwise_split=True and pipeline_parallel_size > 1."
                )
            self._transport = _make_split_transport(
                rank, world_size, tensor_recv_addrs
            )

            if token_recv_addrs is None:
                token_recv_addrs = _parse_recv_addrs(
                    envs.VLLM_SPLIT_TOKEN_RECV_ADDRS
                )
            if token_recv_addrs is None:
                raise RuntimeError(
                    "VLLM_SPLIT_TOKEN_RECV_ADDRS must be set when "
                    "enable_layerwise_split=True and pipeline_parallel_size > 1."
                )
            self._token_transport = _make_token_transport(
                rank, world_size, token_recv_addrs
            )

            logger.info(
                "Initialized SplitPipelineGroup: rank=%d/%d stage_label=%s "
                "tensor_recv=%s tensor_send=%s token_recv=%s token_sends=%s",
                rank,
                world_size,
                stage_label,
                tensor_recv_addrs[rank],
                tensor_recv_addrs[(rank + 1) % world_size],
                token_recv_addrs[rank] if token_recv_addrs else None,
                [token_recv_addrs[i] for i in range(world_size)
                 if i != rank and i != world_size - 1]
                if (token_recv_addrs and rank == world_size - 1) else [],
            )
        else:
            self._transport = None
            self._token_transport = None
            logger.info(
                "Initialized SplitPipelineGroup (non-representative): "
                "rank=%d/%d stage_label=%s; inter-stage traffic via intra-stage "
                "TP collectives",
                rank,
                world_size,
                stage_label,
            )

        self._token_broadcast_logged = False
        self._tensor_metadata: dict[str, Any] | None = None
        self._token_metadata: list[str] | None = None

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the underlying NCCL group."""
        # Guard against attribute access before __init__ completes (e.g. during
        # exception handling, diagnostic inspection, or pickling).  We use
        # self.__dict__ directly because object.__getattribute__ is not traceable
        # by torch.compile and would break the compiled model forward path.
        base = self.__dict__.get("_base")
        if base is None:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}' "
                f"(and '_base' is not set — the object may not have been "
                f"fully initialized)"
            )
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(base, name)

    def _broadcast_tensor_dict_within_stage(
        self,
        tensor_dict: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        """Broadcast a tensor dict from the stage representative to all TP ranks.

        For stage_1 with TP>1, the representative rank (TP rank 0) receives the
        stage input over ZMQ and then broadcasts it to the other stage_1 TP ranks.
        For stage_0/stage_2 or TP=1 this is a no-op.
        """
        from vllm.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        if tp_group.world_size <= 1:
            assert tensor_dict is not None
            return tensor_dict

        src = 0  # TP rank 0 is the stage representative.
        my_tp_rank = tp_group.rank_in_group

        if my_tp_rank == src:
            assert tensor_dict is not None
            descriptors = [
                (key, list(tensor.shape), str(tensor.dtype))
                for key, tensor in tensor_dict.items()
            ]
        else:
            descriptors = []

        # Broadcast metadata (shape/dtype) first so non-representative ranks can
        # allocate matching tensors.
        obj_list = [descriptors]
        tp_group.broadcast_object_list(obj_list, src=src)
        descriptors = obj_list[0]

        if my_tp_rank != src:
            tensor_dict = {}

        for key, shape, dtype_str in descriptors:
            if my_tp_rank == src:
                tensor = tensor_dict[key]
            else:
                dtype = getattr(torch, dtype_str.replace("torch.", ""))
                tensor = torch.empty(shape, dtype=dtype, device=self._base.device)
            tp_group.broadcast(tensor, src=src)
            if my_tp_rank != src:
                tensor_dict[key] = tensor

        return tensor_dict

    def _broadcast_token_ids_within_stage(
        self, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Broadcast sampled token ids from the stage_1 representative to all stage_1 TP ranks."""
        from vllm.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        if tp_group.world_size <= 1:
            return token_ids
        return tp_group.broadcast(token_ids, src=0)

    def isend_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor],
        dst: int | None = None,
        all_gather_group: Any | None = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> list[Any]:
        """Send a tensor dict to the next split stage over the custom transport.

        Returns a list of handles with a ``wait()`` method so the caller can
        wait asynchronously.
        """
        if not self._is_representative:
            # Non-representative ranks do not send inter-stage; the stage
            # representative handles the network send.
            class _NoOpSendHandle:
                def wait(self) -> None:
                    return None

            return [_NoOpSendHandle()]

        expected_dst = (self.rank + 1) % self.world_size
        if dst is not None and dst != expected_dst:
            raise NotImplementedError(
                f"SplitPipelineGroup only supports sending to the next stage "
                f"(rank {expected_dst}), got dst={dst}."
            )
        metadata = self._tensor_metadata
        dvi_metadata = metadata.get("dvi") if metadata else None
        packet = SplitTensorPacket(
            req_ids=metadata.get("req_ids", []) if metadata else [],
            num_scheduled_tokens=metadata.get("num_scheduled_tokens", [])
            if metadata
            else [],
            is_prompt=metadata.get("is_prompt", False) if metadata else False,
            tensors=tensor_dict,
            packet_kind=dvi_metadata.get("packet_kind", "normal")
            if dvi_metadata
            else "normal",
            is_fallback=bool(dvi_metadata.get("is_fallback", False))
            if dvi_metadata
            else False,
            cycle_ids=dvi_metadata.get("cycle_ids") if dvi_metadata else None,
            draft_token_ids=dvi_metadata.get("draft_token_ids")
            if dvi_metadata
            else None,
            draft_lengths=dvi_metadata.get("draft_lengths")
            if dvi_metadata
            else None,
            generation_ids=dvi_metadata.get("generation_ids")
            if dvi_metadata
            else None,
            draft_positions=dvi_metadata.get("draft_positions")
            if dvi_metadata
            else None,
            policy_version=dvi_metadata.get("policy_version")
            if dvi_metadata
            else None,
            draft_version=dvi_metadata.get("draft_version")
            if dvi_metadata
            else None,
            sampling_mode=dvi_metadata.get("sampling_mode")
            if dvi_metadata
            else None,
            draft_support_offsets=dvi_metadata.get("draft_support_offsets")
            if dvi_metadata
            else None,
            draft_support_token_ids=dvi_metadata.get("draft_support_token_ids")
            if dvi_metadata
            else None,
            draft_support_logits=dvi_metadata.get("draft_support_logits")
            if dvi_metadata
            else None,
        )
        self._tensor_metadata = None
        assert self._transport is not None
        send_bytes = _tensor_dict_bytes(tensor_dict)
        self._trace.log(phase="activation_send", event="start",
                        bytes_=send_bytes, extra={"is_prompt": packet.is_prompt})
        t0 = time.perf_counter()
        self._transport.send_tensor_packet(packet)
        self._trace.log(phase="activation_send", event="end",
                        bytes_=send_bytes, extra={"is_prompt": packet.is_prompt})
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "SPLIT_TIMING isend_tensor_dict rank=%d bytes=%d tensors=%d "
            "is_prompt=%s time_ms=%.3f",
            self.rank,
            send_bytes,
            len(tensor_dict),
            packet.is_prompt,
            elapsed_ms,
        )

        class _SendHandle:
            def wait(self) -> None:
                return None

        return [_SendHandle()]

    def irecv_tensor_dict(
        self,
        src: int | None = None,
        all_gather_group: Any | None = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[Any], list[Any]]:
        """Receive a tensor dict from the previous split stage.

        Returns ``(tensor_dict, comm_handles, comm_postprocess)`` to match the
        :class:`AsyncIntermediateTensors` contract.
        """
        if not self._is_representative:
            # Non-representative ranks receive the stage input via intra-stage
            # TP broadcast from the representative.
            tensor_dict = self._broadcast_tensor_dict_within_stage(None)
            return tensor_dict, [], []

        expected_src = (self.rank - 1) % self.world_size
        if src is not None and src != expected_src:
            raise NotImplementedError(
                f"SplitPipelineGroup only supports receiving from the previous "
                f"stage (rank {expected_src}), got src={src}."
            )
        assert self._transport is not None
        self._trace.log(phase="activation_recv", event="start")
        t0 = time.perf_counter()
        packet = self._transport.recv_tensor_packet(device=self._base.device)
        self._trace.log(phase="activation_recv", event="end")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        recv_bytes = _tensor_dict_bytes(packet.tensors)
        logger.info(
            "SPLIT_TIMING irecv_tensor_dict rank=%d bytes=%d tensors=%d "
            "is_prompt=%s time_ms=%.3f",
            self.rank,
            recv_bytes,
            len(packet.tensors),
            packet.is_prompt,
            elapsed_ms,
        )
        self._tensor_metadata = {
            "req_ids": packet.req_ids,
            "num_scheduled_tokens": packet.num_scheduled_tokens,
            "is_prompt": packet.is_prompt,
            "dvi": {
                "packet_kind": packet.packet_kind,
                "is_fallback": packet.is_fallback,
                "cycle_ids": packet.cycle_ids,
                "draft_token_ids": packet.draft_token_ids,
                "draft_lengths": packet.draft_lengths,
                "generation_ids": packet.generation_ids,
                "draft_positions": packet.draft_positions,
                "policy_version": packet.policy_version,
                "draft_version": packet.draft_version,
                "sampling_mode": packet.sampling_mode,
                "draft_support_offsets": packet.draft_support_offsets,
                "draft_support_token_ids": packet.draft_support_token_ids,
                "draft_support_logits": packet.draft_support_logits,
            }
            if packet.is_dvi_block
            else None,
        }
        tensors = self._broadcast_tensor_dict_within_stage(packet.tensors)
        return tensors, [], []

    def send_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor],
        dst: int | None = None,
        all_gather_group: Any | None = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> None:
        # Synchronous version of isend_tensor_dict.
        self.isend_tensor_dict(
            tensor_dict,
            dst=dst,
            all_gather_group=all_gather_group,
            all_gather_tensors=all_gather_tensors,
        )

    def broadcast(self, input_: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast ``input_`` from ``src`` to all ranks over the base NCCL group.

        All generic broadcasts continue to use NCCL.  Sampled token ids use
        :meth:`broadcast_sampled_token_ids` so that split mode can route them
        over the custom token transport explicitly.
        """
        return self._base.broadcast(input_, src=src)

    def broadcast_sampled_token_ids(
        self, token_ids: torch.Tensor, src: int
    ) -> torch.Tensor:
        """Broadcast sampled token ids from ``src`` over the custom transport.

        In split mode, sampled token ids from the last rank are sent to stage_0/stage_1
        via the star token transport instead of NCCL.  Only broadcasts from the
        last PP rank are intercepted; other sources fall back to NCCL so that
        unrelated collectives are not disrupted.

        ``src`` is the *global* rank of the source (the caller passes
        ``pp.rank`` / ``pp.last_rank``), so we first map it to the local PP
        rank before deciding whether to intercept.
        """
        if src in self.ranks:
            src_local = self.ranks.index(src)
        else:
            src_local = src
        if src_local != self.world_size - 1:
            return self._base.broadcast_sampled_token_ids(
                token_ids, src=src_local
            )

        if not self._is_representative:
            # Non-representative stage_1 ranks receive sampled tokens from the stage_1
            # representative via the intra-stage TP group.
            return self._broadcast_token_ids_within_stage(token_ids)

        if not self._token_broadcast_logged:
            self._token_broadcast_logged = True
            logger.info(
                "AUDIT Intercepting sampled token broadcast "
                "(rank=%d, src=%d, shape=%s)",
                self.rank,
                src_local,
                token_ids.shape,
            )
        else:
            logger.debug(
                "Intercepting sampled token broadcast (rank=%d, src=%d, shape=%s)",
                self.rank,
                src_local,
                token_ids.shape,
            )
        if self.rank == src_local:
            # stage_2: send sampled tokens to stage_0 and stage_1.
            req_ids = self._token_metadata or []
            if len(req_ids) != token_ids.shape[0]:
                raise ValueError(
                    f"Token metadata length mismatch: "
                    f"len(req_ids)={len(req_ids)} != "
                    f"token_ids.shape[0]={token_ids.shape[0]}"
                )
            packet = SplitTokenPacket.from_token_tensor(
                req_ids=req_ids, token_tensor=token_ids
            )
            self._token_metadata = None
            assert self._token_transport is not None
            token_bytes = int(token_ids.numel() * token_ids.element_size())
            self._trace.log(phase="token_send", event="start", bytes_=token_bytes)
            t0 = time.perf_counter()
            self._token_transport.send_token_packet(packet)
            self._trace.log(phase="token_send", event="end", bytes_=token_bytes)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "SPLIT_TIMING send_token_packet rank=%d shape=%s time_ms=%.3f",
                self.rank,
                list(token_ids.shape),
                elapsed_ms,
            )
        else:
            # stage_0/stage_1 representative: receive and fill the input tensor in-place.
            assert self._token_transport is not None
            self._trace.log(phase="token_recv", event="start")
            t0 = time.perf_counter()
            packet = self._token_transport.recv_token_packet()
            self._trace.log(phase="token_recv", event="end")
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "SPLIT_TIMING recv_token_packet rank=%d shape=%s time_ms=%.3f",
                self.rank,
                list(token_ids.shape),
                elapsed_ms,
            )
            self._token_metadata = packet.req_ids
            num_reqs = token_ids.shape[0]
            token_tensor = packet.to_token_tensor(
                device=self._base.device, num_reqs=num_reqs
            )
            token_ids.copy_(token_tensor)

        # stage_1 representative must further distribute tokens to the rest of stage_1.
        return self._broadcast_token_ids_within_stage(token_ids)

    def set_tensor_metadata(
        self,
        req_ids: list[str],
        num_scheduled_tokens: list[int],
        is_prompt: bool,
        dvi_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store metadata to be included in the next sent tensor packet.

        ``dvi_metadata`` carries the Stage-DVI block fields (packet_kind,
        cycle_ids, draft_token_ids, draft_lengths) when the outgoing packet
        is a DVI block; it is None for normal packets.
        """
        self._tensor_metadata = {
            "req_ids": req_ids,
            "num_scheduled_tokens": num_scheduled_tokens,
            "is_prompt": is_prompt,
            "dvi": dvi_metadata,
        }

    def get_tensor_metadata(self) -> dict[str, Any] | None:
        """Return metadata from the last received tensor packet."""
        return self._tensor_metadata

    def set_token_metadata(self, req_ids: list[str]) -> None:
        """Store request IDs to be included in the next token packet."""
        self._token_metadata = req_ids

    def get_token_metadata(self) -> list[str] | None:
        """Return request IDs from the last received token packet."""
        return self._token_metadata

    def destroy(self) -> None:
        """Close the custom transports and then tear down the underlying group."""
        for transport in (self._transport, self._token_transport):
            if transport is None:
                continue
            try:
                transport.close()
            except Exception:
                logger.exception("Error closing split transport during destroy")
        self._base.destroy()

    def close(self) -> None:
        for transport in (self._transport, self._token_transport):
            if transport is not None:
                transport.close()
