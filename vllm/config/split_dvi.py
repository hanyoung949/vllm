# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for Stage-DVI (distributed draft/verify) on layer-wise split.

Stage-DVI turns the 3-stage split pipeline into a speculative decoding
protocol:

- ``stage_0`` drafts ``draft_length`` candidate tokens per decode cycle with a
  fixed draft head mounted on its boundary hidden states;
- the full block of boundary ``IntermediateTensors`` is sent to ``stage_1`` /
  ``stage_2`` in one packet;
- ``stage_2`` runs greedy block verification and returns up to
  ``draft_length`` committed tokens through the usual ``SplitTokenPacket``.

The v1 scope is deliberately narrow: V2 model runner only, greedy sampling
only, no bonus token, no draft-head training, no hot updates.  KV lifecycle
reuses the native V2 spec-decode semantics (``num_sampled``/``num_rejected``
and ``num_computed_tokens`` rollback) — the "recompute" mode from the design
doc is not needed because the PP decode cadence plus placeholder scheduling
already makes every DVI step shaped exactly like a native spec-decode step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)

SPLIT_DVI_METHOD = "split_dvi"


@config
@dataclass
class SplitDVIConfig:
    """Feature configuration for Stage-DVI on layer-wise split."""

    enabled: bool = False
    """Master switch.  When False the engine behaves exactly like the
    baseline layer-wise split (zero behavior change)."""

    mode: str = "greedy"
    """Verification mode.  v1 only supports ``"greedy"``."""

    draft_length: int = 4
    """Number of block positions k per DVI cycle: 1 boundary forward plus
    k-1 draft forwards on stage_0, and k draft token proposals verified by
    stage_2.  Mapped internally to ``num_speculative_tokens = k - 1`` plus
    one bonus position of scheduler accounting."""

    bonus_token: bool = False
    """v1 does not support bonus tokens; must stay False."""

    # ---- draft head -----------------------------------------------------
    draft_head_path: str | None = None
    """Optional independent draft-head checkpoint (safetensors or .pt) with
    ``base_projection.weight``, optional ``norm.*``, ``lora_A.weight`` and
    ``lora_B.weight``.  When None, the base projection is initialized from
    the target model's ``lm_head`` (or tied ``embed_tokens``) weights and any
    LoRA delta starts at zero."""

    draft_head_norm: str = "none"
    """Optional normalization before the projection: ``none`` | ``rmsnorm``."""

    draft_head_rank: int = 8
    """LoRA rank of the draft-head delta (kept for forward compatibility;
    an all-zero delta is used when no checkpoint provides weights)."""

    draft_head_alpha: int = 16
    """LoRA alpha; scaling is alpha / rank."""

    draft_head_dtype: str = "bfloat16"
    """Compute dtype of the draft head: ``bfloat16`` | ``float16`` |
    ``float32``."""

    # ---- KV lifecycle ---------------------------------------------------
    kv_mode: str = "native_rollback"
    """KV lifecycle mode.  v1 implements ``native_rollback`` only: accepted
    prefix KV is kept, rejected suffix KV is abandoned by rolling back
    ``num_computed_tokens`` (native V2 spec-decode semantics) and gets
    overwritten by later cycles.  ``recompute`` is reserved."""

    # ---- safety ----------------------------------------------------------
    max_draft_length: int = 8
    """Hard upper bound for ``draft_length``."""

    validate_stage_state: bool = False
    """Extra (expensive) cross-stage consistency assertions: cycle ids and
    draft metadata are validated on every hop.  Cycle validation is always
    on; this flag adds tensor-level checks."""

    # ---- telemetry -------------------------------------------------------
    enable_metrics: bool = True
    """Collect per-cycle DVI counters (acceptance, advancement, timings) and
    log them periodically."""

    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if self.mode != "greedy":
            raise ValueError(
                f"SplitDVI only supports mode='greedy' in v1, got {self.mode!r}"
            )
        if self.bonus_token:
            raise ValueError("SplitDVI v1 does not support bonus_token=True")
        if not (2 <= self.draft_length <= self.max_draft_length):
            raise ValueError(
                f"draft_length must be in [2, {self.max_draft_length}], got "
                f"{self.draft_length}"
            )
        if self.kv_mode != "native_rollback":
            raise ValueError(
                f"SplitDVI v1 implements kv_mode='native_rollback' only, got "
                f"{self.kv_mode!r}"
            )
        if self.draft_head_norm not in ("none", "rmsnorm"):
            raise ValueError(
                f"draft_head_norm must be 'none' or 'rmsnorm', got "
                f"{self.draft_head_norm!r}"
            )
        if self.draft_head_dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(
                f"Unsupported draft_head_dtype {self.draft_head_dtype!r}"
            )
        if self.draft_head_rank <= 0 or self.draft_head_alpha <= 0:
            raise ValueError("draft_head_rank/alpha must be positive")

    @property
    def num_scheduler_spec_tokens(self) -> int:
        """Number of placeholder spec tokens the scheduler must book per
        decode step: k block positions = 1 bonus + (k-1) spec tokens."""
        return self.draft_length - 1

    def compute_hash(self) -> str:
        # The draft head runs eagerly outside the compiled target-model
        # graph; num_speculative_tokens (which shapes cudagraphs) is already
        # covered by SpeculativeConfig.compute_hash().
        return ""

    def validate_against_vllm_config(self, vllm_config: Any) -> None:
        """Cross-checks against the rest of the engine configuration.

        Separated from ``__post_init__`` because it needs the fully built
        VllmConfig (parallel/scheduler/model configs).
        """
        if not self.enabled:
            return
        parallel = vllm_config.parallel_config
        if not parallel.enable_layerwise_split:
            raise ValueError("SplitDVI requires enable_layerwise_split=True")
        if parallel.pipeline_parallel_size != 3:
            raise ValueError(
                "SplitDVI requires pipeline_model_parallel_size == 3 "
                f"(stage_0/stage_1/stage_2), got {parallel.pipeline_parallel_size}"
            )
        spec = vllm_config.speculative_config
        if spec is not None and spec.method != SPLIT_DVI_METHOD:
            raise ValueError(
                f"SplitDVI cannot be combined with another speculator "
                f"(speculative_config.method={spec.method!r})"
            )
        model_config = vllm_config.model_config
        if model_config is not None and getattr(model_config, "is_moe", False):
            raise ValueError(
                "SplitDVI v1 does not support MoE models (the draft loop "
                "does not drive EPLB rebalancing)"
            )
        if model_config is not None and _has_recurrent_state(model_config):
            raise ValueError(
                "SplitDVI v1 does not support hybrid/linear-attention models "
                "(e.g. Qwen3.5): the draft loop and native_rollback are built "
                "on KV-block attention semantics; recurrent-state layers have "
                "no KV blocks to write or roll back"
            )
        if (
            model_config is not None
            and getattr(model_config, "is_multimodal_model", False)
        ):
            logger.warning_once(
                "SplitDVI with multimodal models is untested; DVI drafting "
                "only applies to text decode steps."
            )
        if spec is not None and spec.num_speculative_tokens is not None:
            expected = self.num_scheduler_spec_tokens
            if spec.num_speculative_tokens != expected:
                raise ValueError(
                    f"speculative_config.num_speculative_tokens="
                    f"{spec.num_speculative_tokens} is inconsistent with "
                    f"SplitDVI draft_length={self.draft_length} "
                    f"(expected {expected})"
                )
        if not vllm_config.use_v2_model_runner:
            raise ValueError(
                "SplitDVI requires the V2 model runner "
                "(VLLM_USE_V2_MODEL_RUNNER must not be 0)"
            )
        if not vllm_config.scheduler_config.async_scheduling:
            raise ValueError(
                "SplitDVI requires async scheduling (the PP decode cadence "
                "lives in AsyncScheduler). Do not pass --no-async-scheduling."
            )
        kv_transfer = vllm_config.kv_transfer_config
        if (
            kv_transfer is not None
            and kv_transfer.is_kv_transfer_instance
        ):
            raise ValueError(
                "SplitDVI v1 does not support KV transfer connectors: the "
                "stage_0 draft sub-forwards bypass kv_connector.pre_forward, "
                "so remote-KV load/sync semantics are undefined. Disable the "
                "KV connector or the DRAFT path."
            )


def _has_recurrent_state(model_config: Any) -> bool:
    """Detect models with recurrent-state (mamba-style) layers.

    Primary: vLLM's model-info capability flags (covers families without
    ``layer_types``).  Fallback: exact ``layer_types`` strings on the HF
    config (or its ``text_config``) for families the flags miss.
    KV-cache-based variants such as sliding-window attention are allowed.
    """
    if getattr(model_config, "has_inner_state", False):
        return True
    if getattr(model_config, "is_attention_free", False):
        return True
    recurrent = {
        "linear_attention",
        "mamba",
        "mamba2",
        "recurrent",
        "gated_delta_net",
    }
    hf = getattr(model_config, "hf_config", None)
    for cfg in (hf, getattr(hf, "text_config", None)):
        layer_types = getattr(cfg, "layer_types", None)
        if layer_types and any(t in recurrent for t in layer_types):
            return True
    return False


def materialize_split_dvi_speculative_config(
    split_dvi: SplitDVIConfig,
    target_model_config: Any,
    target_parallel_config: Any,
) -> Any:
    """Build the internal SpeculativeConfig that lights up the native
    spec-decode bookkeeping (placeholder scheduling, KV allocation and
    output reconciliation) for Stage-DVI.

    No draft model and no speculator are ever created for method
    ``split_dvi``; the draft lives on stage_0 and travels inside split
    packets instead of the scheduler round-trip.
    """
    from vllm.config.speculative import SpeculativeConfig

    return SpeculativeConfig(
        method=SPLIT_DVI_METHOD,
        num_speculative_tokens=split_dvi.num_scheduler_spec_tokens,
        target_model_config=target_model_config,
        target_parallel_config=target_parallel_config,
    )
