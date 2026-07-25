# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-DVI: distributed draft/verify speculative decoding for layer-wise split.

This package hosts both the L0 telemetry infrastructure (artifact/telemetry/
hooks) and the Stage-DVI v1 runtime (config, draft head, draft block
generator, greedy block verifier, request state, metrics).

Runtime modules are imported lazily by the model runner to avoid pulling
torch-heavy dependencies into unrelated processes.
"""

from vllm.config.split_dvi import SPLIT_DVI_METHOD, SplitDVIConfig

__all__ = [
    "SPLIT_DVI_METHOD",
    "SplitDVIConfig",
]
