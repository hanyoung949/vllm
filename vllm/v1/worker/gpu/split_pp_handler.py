# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-aware pipeline-parallel sampled-token handler for the V2 model runner."""

from collections import deque

import torch

from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.platforms import current_platform
from vllm.v1.engine.split_data import SplitTokenPacket
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.pp_utils import (
    PPHandler,
    PendingRecv,
    compute_need_sampled_mask,
)


class SplitPPHandler(PPHandler):
    """PPHandler variant that routes sampled tokens over the split TCP transport.

    The base V2 ``PPHandler`` creates a dedicated NCCL sibling group for the
    sampled-token broadcast, which bypasses the layer-wise split's cross-machine
    TCP token transport and does not work with stage_1 TP > 1.  This subclass
    keeps the same queue/event semantics but uses ``SplitPipelineGroup``'s
    token transport instead.
    """

    def __init__(
        self, max_num_reqs: int, num_speculative_steps: int, device: torch.device
    ):
        # Replicate the base init but skip creating the NCCL sibling group.
        pp = get_pp_group()
        self.is_last_rank = pp.is_last_rank
        self.last_rank = pp.last_rank
        self.rank = pp.rank
        self.max_sample_len = num_speculative_steps + 1
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)

        self.queue: deque[PendingRecv | None] = (
            deque() if self.is_last_rank else deque([None] * pp.world_size)
        )

        self.req_idx_gen = torch.zeros(
            max_num_reqs, dtype=torch.int32, device="cpu"
        )
        self.req_idx_gen_np = self.req_idx_gen.numpy()

        # Cache the split PP group and stage-1 TP group for fan-out.
        self._split_pp = pp
        self._tp_group = get_tp_group()
        self._is_stage_1 = self.rank == 1

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def receive(self, input_batch: InputBatch) -> bool:
        """Receive sampled tokens from stage_2 via the split token transport."""
        assert not self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            return False

        gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]
        num_reqs = input_batch.num_reqs

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            sampled_tokens = torch.empty(
                num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device
            )
            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)

            if self._split_pp._is_representative:
                # stage_0/stage_1 representative: receive over TCP from stage_2.
                packet = self._split_pp._token_transport.recv_token_packet()
                recv_sampled, recv_num_sampled, recv_num_rejected = (
                    packet.to_tensors(
                        device=self.device,
                        num_reqs=num_reqs,
                        max_sample_len=self.max_sample_len,
                    )
                )
                sampled_tokens.copy_(recv_sampled)
                combined[0].copy_(recv_num_sampled)
                combined[1].copy_(recv_num_rejected)

            # Fan out to all stage_1 TP ranks.  For stage_0/stage_2 the TP group
            # has world_size 1 and this is a no-op.
            if self._is_stage_1:
                self._tp_group.broadcast(sampled_tokens, src=0)
                self._tp_group.broadcast(combined, src=0)

            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)

        self.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
        )
        return bool(need_sampled_mask.all())

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        """Send sampled tokens from stage_2 to stage_0/stage_1 via TCP."""
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            return

        assert sampled_token_ids.dtype == torch.int64

        if current_platform.is_xpu():
            self.main_stream.synchronize()

        # Record streams like the base class does, and perform the CPU-side TCP
        # send inside the side-stream context so the sampled-token tensors are
        # guaranteed to be ready on the main stream before we read them.
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            if self._split_pp._is_representative:
                packet = SplitTokenPacket.from_tensors(
                    req_ids=input_batch.req_ids,
                    sampled_token_ids=sampled_token_ids,
                    num_sampled=num_sampled,
                    num_rejected=num_rejected,
                )
                self._split_pp._token_transport.send_token_packet(packet)
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)
