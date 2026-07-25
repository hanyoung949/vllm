# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import NamedTuple

import torch

from vllm.v1.outputs import LogprobsTensors


class DVITelemetrySideOutput(NamedTuple):
    """Compact GPU side output for DVI L0 telemetry.

    Carries the processed verifier distribution (after penalties, grammar,
    temperature, top-k, top-p) for only the rows reserved by the telemetry
    hook. Kept worker-local; not serialized to the scheduler.
    """

    # [num_reserved_rows, K]
    topk_ids: torch.Tensor
    # [num_reserved_rows, K]
    topk_logprobs: torch.Tensor
    # [num_reserved_rows]
    residual_mass: torch.Tensor
    # [num_reserved_rows]
    top1_id: torch.Tensor
    # [num_reserved_rows], int32
    # Number of finite entries in topk_logprobs/topk_ids for each row.
    # Remaining entries may be -inf/masked and must not be written to spool.
    valid_count: torch.Tensor


@dataclass
class SamplerOutput:
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None
    num_nans: torch.Tensor | None
    num_sampled: torch.Tensor | None
    num_rejected: torch.Tensor | None = None
    # Worker-local DVI telemetry side output; None when telemetry is disabled
    # or no rows were reserved.
    dvi_telemetry_tensors: DVITelemetrySideOutput | None = None
