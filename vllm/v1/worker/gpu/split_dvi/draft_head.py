# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-DVI runtime draft head (lives on split stage_0).

Architecture (fixed for v1):

    boundary hidden -> [optional RMSNorm] -> frozen base H x V projection
                     -> [optional LoRA delta] -> draft logits

The base projection is initialized from the target model's ``lm_head`` (or
the tied ``embed_tokens``) and stays frozen at inference time.  The LoRA
delta is kept in the module definition even though v1 does not train it, so
a future trained delta can be hot-loaded without changing the module.

Checkpoint (path A) is a single ``.safetensors`` / ``.pt`` file holding:

    base_projection.weight  [vocab_size, hidden_size]   (embedded mode)
    norm.weight             [hidden_size]               (required iff rmsnorm)
    lora_A.weight           [rank, hidden_size]         (optional)
    lora_B.weight           [vocab_size, rank]          (optional, pair w/ A)

Compact checkpoints explicitly omit ``base_projection.weight`` and declare
``base_projection_source="target_model"`` plus its canonical SHA256 in the
sidecar.  Runtime reloads that projection from the target model, verifies it,
and clones it into an independent frozen parameter.

plus an optional sidecar ``<name>.json`` with metadata
``{"norm": ..., "rank": ..., "alpha": ..., "vocab_size": ..., "hidden_size":
...}``.  A ``.pt`` file may instead be a dict with ``state_dict`` and
``metadata`` entries.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from vllm.config.split_dvi import SplitDVIConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm

logger = init_logger(__name__)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class SplitDVIDraftHead(nn.Module):
    """Frozen base projection + optional LoRA delta producing draft logits."""

    def __init__(
        self,
        base_projection: nn.Linear,
        norm: nn.Module | None,
        lora_a: nn.Linear | None,
        lora_b: nn.Linear | None,
        scaling: float,
    ):
        super().__init__()
        for param in base_projection.parameters():
            param.requires_grad_(False)
        self.base_projection = base_projection
        self.norm = norm
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.scaling = scaling

    @property
    def vocab_size(self) -> int:
        return self.base_projection.out_features

    @property
    def hidden_size(self) -> int:
        return self.base_projection.in_features

    def forward(self, boundary_hidden: torch.Tensor) -> torch.Tensor:
        """Return draft logits of shape ``[num_tokens, vocab_size]``."""
        x = boundary_hidden
        if self.norm is not None:
            x = self.norm(x)
        logits = self.base_projection(x)
        if self.lora_a is not None and self.lora_b is not None:
            logits = logits + self.lora_b(self.lora_a(x)) * self.scaling
        return logits

    def predict_token_ids(self, boundary_hidden: torch.Tensor) -> torch.Tensor:
        """Greedy draft proposal: argmax over the draft logits."""
        return self.forward(boundary_hidden).argmax(dim=-1)


@dataclass
class DraftHeadMetadata:
    norm: str
    rank: int
    alpha: int
    vocab_size: int | None = None
    hidden_size: int | None = None
    base_projection_source: str | None = None
    base_projection_sha256: str | None = None


def _sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _load_state_and_metadata(
    path: str,
) -> tuple[dict[str, torch.Tensor], dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Draft head checkpoint not found: {path}")
    if path.endswith(".safetensors"):
        from safetensors import safe_open

        state: dict[str, torch.Tensor] = {}
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)
        sidecar = os.path.splitext(path)[0] + ".json"
        metadata: dict = {}
        if os.path.isfile(sidecar):
            with open(sidecar) as f:
                metadata = json.load(f)
        return state, metadata

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        return dict(obj["state_dict"]), dict(obj.get("metadata", {}))
    if isinstance(obj, dict):
        return dict(obj), {}
    raise ValueError(f"Unsupported draft head checkpoint format: {path}")


def _build_from_state(
    state: dict[str, torch.Tensor],
    metadata: DraftHeadMetadata,
    device: torch.device,
    dtype: torch.dtype,
) -> SplitDVIDraftHead:
    if "base_projection.weight" not in state:
        raise ValueError(
            "Draft head checkpoint is missing 'base_projection.weight'"
        )
    base_weight = state["base_projection.weight"]
    if base_weight.dim() != 2:
        raise ValueError(
            f"base_projection.weight must be 2-D, got shape {base_weight.shape}"
        )
    vocab_size, hidden_size = base_weight.shape
    if metadata.vocab_size is not None and metadata.vocab_size != vocab_size:
        raise ValueError(
            f"Draft head vocab mismatch: metadata {metadata.vocab_size} vs "
            f"weight {vocab_size}"
        )
    if metadata.hidden_size is not None and metadata.hidden_size != hidden_size:
        raise ValueError(
            f"Draft head hidden mismatch: metadata {metadata.hidden_size} vs "
            f"weight {hidden_size}"
        )

    base_projection = nn.Linear(hidden_size, vocab_size, bias=False)
    with torch.no_grad():
        base_projection.weight.copy_(base_weight)

    norm: nn.Module | None = None
    if metadata.norm == "rmsnorm":
        if "norm.weight" not in state:
            raise ValueError(
                "draft_head_norm='rmsnorm' but checkpoint has no 'norm.weight'"
            )
        norm = RMSNorm(hidden_size, eps=1e-6)
        with torch.no_grad():
            norm.weight.copy_(state["norm.weight"])
    elif metadata.norm != "none":
        raise ValueError(f"Unsupported draft head norm {metadata.norm!r}")

    lora_a: nn.Linear | None = None
    lora_b: nn.Linear | None = None
    has_a = "lora_A.weight" in state
    has_b = "lora_B.weight" in state
    if has_a != has_b:
        raise ValueError(
            "Draft head LoRA must provide both lora_A.weight and lora_B.weight"
        )
    if has_a and has_b:
        a_weight = state["lora_A.weight"]
        b_weight = state["lora_B.weight"]
        if a_weight.dim() != 2 or b_weight.dim() != 2:
            raise ValueError("LoRA weights must be 2-D")
        rank, a_hidden = a_weight.shape
        b_vocab, b_rank = b_weight.shape
        if a_hidden != hidden_size or b_vocab != vocab_size or b_rank != rank:
            raise ValueError(
                f"LoRA shape mismatch: A {a_weight.shape}, B {b_weight.shape}, "
                f"expected A [{metadata.rank}, {hidden_size}], "
                f"B [{vocab_size}, {metadata.rank}]"
            )
        if rank != metadata.rank:
            raise ValueError(
                f"LoRA rank mismatch: checkpoint {rank} vs config {metadata.rank}"
            )
        lora_a = nn.Linear(hidden_size, rank, bias=False)
        lora_b = nn.Linear(rank, vocab_size, bias=False)
        with torch.no_grad():
            lora_a.weight.copy_(a_weight)
            lora_b.weight.copy_(b_weight)

    scaling = metadata.alpha / metadata.rank
    head = SplitDVIDraftHead(base_projection, norm, lora_a, lora_b, scaling)
    return head.to(device=device, dtype=dtype)


def load_draft_head_from_checkpoint(
    path: str,
    config: SplitDVIConfig,
    device: torch.device,
    model: nn.Module | None = None,
    model_config: object | None = None,
) -> SplitDVIDraftHead:
    """Path A: load an independent draft-head checkpoint."""
    state, raw_metadata = _load_state_and_metadata(path)
    metadata = DraftHeadMetadata(
        norm=raw_metadata.get("norm", config.draft_head_norm),
        rank=int(raw_metadata.get("rank", config.draft_head_rank)),
        alpha=int(raw_metadata.get("alpha", config.draft_head_alpha)),
        vocab_size=raw_metadata.get("vocab_size"),
        hidden_size=raw_metadata.get("hidden_size"),
        base_projection_source=raw_metadata.get("base_projection_source"),
        base_projection_sha256=raw_metadata.get("base_projection_sha256"),
    )
    if config.draft_head_norm == "rmsnorm" and metadata.norm != "rmsnorm":
        raise ValueError(
            "Config requires rmsnorm but checkpoint metadata norm="
            f"{metadata.norm!r}"
        )
    if "base_projection.weight" not in state:
        if metadata.base_projection_source != "target_model":
            raise ValueError(
                "Draft head checkpoint is missing 'base_projection.weight'; "
                "compact checkpoints must declare "
                "base_projection_source='target_model'"
            )
        if not metadata.base_projection_sha256:
            raise ValueError(
                "Compact draft head metadata is missing "
                "base_projection_sha256"
            )
        if model is None or model_config is None:
            raise ValueError(
                "Compact draft head requires the runtime target model to "
                "reconstruct its base projection"
            )
        base_weight, source = _resolve_target_projection(model, model_config)
        actual_hash = _sha256_tensor(base_weight)
        if actual_hash != metadata.base_projection_sha256:
            raise ValueError(
                "Compact draft head base projection checksum mismatch: "
                f"metadata {metadata.base_projection_sha256} vs target "
                f"{actual_hash} ({source})"
            )
        state["base_projection.weight"] = base_weight

    dtype = _DTYPE_MAP[config.draft_head_dtype]
    head = _build_from_state(state, metadata, device, dtype)
    logger.info(
        "Loaded Stage-DVI draft head from %s (vocab=%d, hidden=%d, norm=%s, "
        "lora=%s)",
        path,
        head.vocab_size,
        head.hidden_size,
        metadata.norm,
        f"r{metadata.rank}" if head.lora_a is not None else "none",
    )
    return head


def _read_checkpoint_tensor(
    model_path: str, tensor_name: str
) -> torch.Tensor | None:
    """Read a single tensor by name from a HF safetensors checkpoint dir."""
    from safetensors import safe_open

    candidates: list[str] = []
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard = index.get("weight_map", {}).get(tensor_name)
        if shard is None:
            return None
        candidates.append(os.path.join(model_path, shard))
    else:
        single = os.path.join(model_path, "model.safetensors")
        if os.path.isfile(single):
            candidates.append(single)
        else:
            import glob

            candidates.extend(sorted(glob.glob(os.path.join(model_path, "*.safetensors"))))

    for shard_path in candidates:
        with safe_open(shard_path, framework="pt") as f:
            if tensor_name in f.keys():
                return f.get_tensor(tensor_name)
    return None


def _resolve_target_projection(
    model: nn.Module,
    model_config: object,
) -> tuple[torch.Tensor, str]:
    """Resolve and clone the target LM projection used by the draft head."""
    hf_config = getattr(model_config, "hf_config", None)
    tie = bool(getattr(hf_config, "tie_word_embeddings", False))
    expected_vocab = getattr(hf_config, "vocab_size", None)

    base_weight: torch.Tensor | None = None
    source = ""
    if tie:
        embed = getattr(getattr(model, "model", None), "embed_tokens", None)
        if embed is not None and hasattr(embed, "weight"):
            base_weight = embed.weight.detach().clone()
            source = "tied embed_tokens (on device)"
    if base_weight is None:
        model_path = getattr(model_config, "model", None)
        if model_path and os.path.isdir(model_path):
            for name in ("lm_head.weight", "model.embed_tokens.weight"):
                tensor = _read_checkpoint_tensor(model_path, name)
                if tensor is not None:
                    base_weight = tensor.detach().clone()
                    source = f"checkpoint:{name}"
                    break
    if base_weight is None:
        raise ValueError(
            "Cannot initialize the Stage-DVI draft head: no lm_head weight "
            "found (tied embeddings unavailable and checkpoint read failed). "
            "Provide draft_head_path with an embedded base projection instead."
        )
    if base_weight.dim() != 2:
        raise ValueError(f"lm_head weight must be 2-D, got {base_weight.shape}")
    vocab_size, _ = base_weight.shape
    if expected_vocab is not None and vocab_size < expected_vocab:
        raise ValueError(
            f"lm_head vocab {vocab_size} < tokenizer vocab {expected_vocab}"
        )
    return base_weight, source


def init_draft_head_from_target_model(
    config: SplitDVIConfig,
    model: nn.Module,
    model_config: object,
    device: torch.device,
) -> SplitDVIDraftHead:
    """Path B: initialize the frozen base projection from the target model's
    ``lm_head`` (or tied ``embed_tokens``) weights.

    ``model`` is the stage_0 partial model; with tied embeddings its
    ``embed_tokens`` weight is exactly the ``lm_head`` weight.
    """
    base_weight, source = _resolve_target_projection(model, model_config)
    _, hidden_size = base_weight.shape

    metadata = DraftHeadMetadata(
        norm=config.draft_head_norm,
        rank=config.draft_head_rank,
        alpha=config.draft_head_alpha,
        vocab_size=None,
        hidden_size=None,
    )
    state = {"base_projection.weight": base_weight}
    if metadata.norm == "rmsnorm":
        # No trained norm available; initialize to identity (equivalent to
        # no scaling) so the structure is in place for future training.
        state["norm.weight"] = torch.ones(hidden_size, dtype=base_weight.dtype)
    dtype = _DTYPE_MAP[config.draft_head_dtype]
    head = _build_from_state(state, metadata, device, dtype)
    logger.info(
        "Initialized Stage-DVI draft head from %s (vocab=%d, hidden=%d, "
        "norm=%s, lora=none)",
        source,
        head.vocab_size,
        head.hidden_size,
        metadata.norm,
    )
    return head


def load_split_dvi_draft_head(
    config: SplitDVIConfig,
    model: nn.Module,
    model_config: object,
    device: torch.device,
) -> SplitDVIDraftHead:
    """Entry point: independent checkpoint when given, else target lm_head."""
    if config.draft_head_path is not None:
        return load_draft_head_from_checkpoint(
            config.draft_head_path,
            config,
            device,
            model=model,
            model_config=model_config,
        )
    return init_draft_head_from_target_model(config, model, model_config, device)
