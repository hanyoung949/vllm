# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-DVI draft block generator (runs on split stage_0).

Given the scheduler-booked expanded batch for a decode step (k rows per
request: 1 boundary + k-1 placeholder spec tokens), the generator replaces
the single expanded forward with:

1. a boundary forward of the last sampled token (position c), and
2. k-1 sequential single-token draft forwards: draft head predicts
   ``d_{j+1}`` from the boundary hidden of position c+j, the token is
   embedded and forwarded at position c+j+1, producing the next boundary
   hidden.

Every substep writes KV at its true position through the request's regular
block table, so accepted-prefix KV is natively reusable and rejected-suffix
KV (positions >= num_computed after the verdict) is dead and overwritten by
later cycles — native V2 spec-decode semantics.

The collected per-position ``IntermediateTensors`` (hidden_states *and*
residual, both required by pre-norm models) form the DVI block sent to
stage_1, exactly aligned row-for-row with the expanded batch every stage
built from the same scheduler output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.split_dvi.draft_head import SplitDVIDraftHead

logger = init_logger(__name__)


@dataclass
class SplitDVIDraftBlock:
    """One DVI cycle's payload for the whole batch (uniform k per request)."""

    req_ids: list[str]
    cycle_ids: list[int]
    draft_token_ids: list[int]  # flat, req-major: [d_1..d_k] per request
    draft_lengths: list[int]  # k per request
    draft_positions: list[int]  # flat, req-major position of each block row
    intermediate_tensors: IntermediateTensors

    @property
    def num_rows(self) -> int:
        return sum(self.draft_lengths)


class SplitDVIDraftBlockGenerator:
    """Generates draft blocks on stage_0 via sequential draft forwards."""

    def __init__(
        self,
        runner,
        draft_head: SplitDVIDraftHead,
        max_num_reqs: int,
    ):
        self.runner = runner
        self.draft_head = draft_head
        self.device = runner.device
        self.max_num_reqs = max_num_reqs
        # Private buffers so the runner's main input buffers are untouched
        # (mirrors how the native speculators isolate their draft buffers).
        self.input_buffers = InputBuffers(
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_reqs,
            device=self.device,
        )
        self.arange_cpu = torch.arange(max_num_reqs + 1, dtype=torch.int32)
        self.arange_gpu = torch.arange(
            max_num_reqs + 1, dtype=torch.int32, device=self.device
        )

    @torch.inference_mode()
    def generate(
        self,
        input_batch: InputBatch,
        num_draft_per_req: int,
        cycle_ids: list[int],
    ) -> SplitDVIDraftBlock:
        """Run the boundary + draft loop and return the block.

        ``num_draft_per_req`` is the number of block positions per request
        (k = 1 + scheduled placeholder spec tokens); the batch is guaranteed
        uniform by the caller.
        """
        runner = self.runner
        num_reqs = input_batch.num_reqs
        k = num_draft_per_req
        idx_mapping = input_batch.idx_mapping[:num_reqs]

        req_states = runner.req_states
        # Committed-token frontier per request (position of the boundary row).
        num_computed = req_states.num_computed_tokens.gpu[idx_mapping]
        boundary_tokens = req_states.last_sampled_tokens[idx_mapping].squeeze(-1)

        # positions_all[j] = c + j for j in 0..k-1, computed without sync.
        positions_all = (
            num_computed.unsqueeze(0)
            + torch.arange(k, device=self.device, dtype=torch.int64).unsqueeze(1)
        )  # [k, num_reqs]

        query_start_loc_gpu = self.arange_gpu[: num_reqs + 1]
        query_start_loc_cpu = self.arange_cpu[: num_reqs + 1]
        block_tables = [
            bt[:num_reqs] for bt in runner.block_tables.input_block_tables
        ]
        input_ids_buf = self.input_buffers.input_ids[:num_reqs]
        positions_buf = self.input_buffers.positions[:num_reqs]
        seq_lens_buf = self.input_buffers.seq_lens[:num_reqs]

        # CPU twin for max_seq_len metadata (exact, no sync: the CPU mirror
        # equals the GPU frontier at the start of the step).
        num_computed_np = input_batch.num_computed_tokens_np.astype("int64")

        hidden_rows: list[torch.Tensor] = []
        residual_rows: list[torch.Tensor] = []
        draft_ids_rows: list[torch.Tensor] = []  # d_{j+1} per substep

        draft_start_ns = time.perf_counter_ns()
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        for j in range(k):
            positions_j = positions_all[j]
            positions_buf.copy_(positions_j)
            seq_lens_buf.copy_(positions_j + 1)
            if j == 0:
                input_ids_buf.copy_(boundary_tokens)
            else:
                input_ids_buf.copy_(draft_ids_rows[-1])

            slot_mappings = runner.block_tables.compute_slot_mappings(
                idx_mapping,
                query_start_loc_gpu,
                positions_buf,
                num_reqs,
            )
            attn_metadata = build_attn_metadata(
                attn_groups=runner.attn_groups,
                num_reqs=num_reqs,
                num_tokens=num_reqs,
                query_start_loc_gpu=query_start_loc_gpu,
                query_start_loc_cpu=query_start_loc_cpu,
                max_query_len=1,
                seq_lens=seq_lens_buf,
                max_seq_len=int((num_computed_np + j + 1).max()),
                block_tables=block_tables,
                slot_mappings=slot_mappings,
                kv_cache_config=runner.kv_cache_config,
                # Match the normal decode path's metadata construction
                # field-for-field (model_states/default.py:prepare_attn) so
                # the substep forward is numerically identical to a plain
                # decode step.
                seq_lens_cpu_upper_bound=torch.from_numpy(
                    num_computed_np + j + 1
                ).to(torch.int32),
                positions=positions_buf,
            )
            slot_mappings_by_layer = build_slot_mappings_by_layer(
                slot_mappings, runner.kv_cache_config
            )
            with set_forward_context(
                attn_metadata,
                runner.vllm_config,
                num_tokens=num_reqs,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=slot_mappings_by_layer,
            ):
                output = runner.model(
                    input_ids=input_ids_buf,
                    positions=positions_buf,
                    intermediate_tensors=None,
                    inputs_embeds=None,
                )
            assert isinstance(output, IntermediateTensors), (
                f"stage_0 model must return IntermediateTensors, got "
                f"{type(output)}"
            )
            hidden_j = output.tensors["hidden_states"]
            hidden_rows.append(hidden_j)
            if "residual" in output.tensors:
                residual_rows.append(output.tensors["residual"])
            # Greedy draft proposal for position c+j+1.
            draft_ids_rows.append(
                self.draft_head.predict_token_ids(hidden_j).to(torch.int32)
            )
        draft_ms = (time.perf_counter_ns() - draft_start_ns) / 1e6
        ev_end.record()

        # Assemble the block in the expanded batch's row order (req-major,
        # position-minor): [k, R, H] -> [R, k, H] -> [R*k, H].
        hidden_block = (
            torch.stack(hidden_rows, dim=0)
            .permute(1, 0, 2)
            .reshape(num_reqs * k, -1)
        )
        block_tensors: dict[str, torch.Tensor] = {"hidden_states": hidden_block}
        if residual_rows:
            block_tensors["residual"] = (
                torch.stack(residual_rows, dim=0)
                .permute(1, 0, 2)
                .reshape(num_reqs * k, -1)
            )

        # Draft ids: rows are d_1..d_k per substep -> req-major [R, k].
        draft_ids_block = torch.stack(draft_ids_rows, dim=0).permute(1, 0)

        # Positions of each block row (req-major), for validation/telemetry.
        positions_block = positions_all.permute(1, 0).reshape(-1)

        runtime = getattr(runner, "split_dvi_runtime", None)
        if runtime is not None and runtime.metrics is not None:
            runtime.metrics.draft_wall_ms += draft_ms
            runtime.metrics.record_draft_events(ev_start, ev_end)

        return SplitDVIDraftBlock(
            req_ids=list(input_batch.req_ids),
            cycle_ids=cycle_ids,
            draft_token_ids=draft_ids_block.reshape(-1).tolist(),
            draft_lengths=[k] * num_reqs,
            draft_positions=positions_block.tolist(),
            intermediate_tensors=IntermediateTensors(block_tensors),
        )
