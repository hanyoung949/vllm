# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Stage-DVI runtime draft head (split_dvi/draft_head.py)."""

from __future__ import annotations

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from vllm.config.split_dvi import SplitDVIConfig
from vllm.v1.worker.gpu.split_dvi.draft_head import (
    SplitDVIDraftHead,
    _sha256_tensor,
    load_draft_head_from_checkpoint,
    init_draft_head_from_target_model,
    load_split_dvi_draft_head,
)

VOCAB = 32
HIDDEN = 16
RANK = 4
ALPHA = 8
DEVICE = torch.device("cpu")


def _cfg(**overrides) -> SplitDVIConfig:
    base = dict(
        enabled=True,
        draft_length=4,
        draft_head_rank=RANK,
        draft_head_alpha=ALPHA,
        draft_head_dtype="float32",
    )
    base.update(overrides)
    return SplitDVIConfig(**base)


def _random_state(norm: bool = False, lora: bool = False) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    state = {"base_projection.weight": torch.randn(VOCAB, HIDDEN)}
    if norm:
        state["norm.weight"] = torch.randn(HIDDEN).abs() + 0.5
    if lora:
        state["lora_A.weight"] = torch.randn(RANK, HIDDEN)
        state["lora_B.weight"] = torch.randn(VOCAB, RANK)
    return state


def _write_safetensors_checkpoint(
    tmp_path, state: dict[str, torch.Tensor], metadata: dict | None = None
) -> str:
    path = str(tmp_path / "split_dvi_draft_head.safetensors")
    save_file(state, path)
    if metadata is not None:
        with open(str(tmp_path / "split_dvi_draft_head.json"), "w") as f:
            json.dump(metadata, f)
    return path


def test_forward_shape_and_dtype():
    head = SplitDVIDraftHead(
        base_projection=torch.nn.Linear(HIDDEN, VOCAB, bias=False),
        norm=None,
        lora_a=None,
        lora_b=None,
        scaling=1.0,
    )
    x = torch.randn(5, HIDDEN)
    logits = head(x)
    assert logits.shape == (5, VOCAB)
    assert logits.dtype == torch.float32
    ids = head.predict_token_ids(x)
    assert ids.shape == (5,)
    assert torch.equal(ids, logits.argmax(dim=-1))


def test_frozen_base_projection():
    head = SplitDVIDraftHead(
        torch.nn.Linear(HIDDEN, VOCAB, bias=False), None, None, None, 1.0
    )
    assert not any(p.requires_grad for p in head.base_projection.parameters())


def test_checkpoint_full_load_matches_manual(tmp_path, default_vllm_config):
    state = _random_state(norm=True, lora=True)
    path = _write_safetensors_checkpoint(
        tmp_path, state, {"norm": "rmsnorm", "rank": RANK, "alpha": ALPHA}
    )
    head = load_draft_head_from_checkpoint(path, _cfg(draft_head_norm="rmsnorm"), DEVICE)

    x = torch.randn(3, HIDDEN)
    manual_norm = x / x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    manual_norm = manual_norm * state["norm.weight"]
    manual = manual_norm @ state["base_projection.weight"].T
    manual = manual + (manual_norm @ state["lora_A.weight"].T @ state["lora_B.weight"].T) * (
        ALPHA / RANK
    )
    assert torch.allclose(head(x), manual, atol=1e-5)


def test_checkpoint_requires_base_projection(tmp_path):
    path = _write_safetensors_checkpoint(tmp_path, {"norm.weight": torch.randn(HIDDEN)})
    with pytest.raises(ValueError, match="base_projection.weight"):
        load_draft_head_from_checkpoint(path, _cfg(), DEVICE)


def test_checkpoint_rmsnorm_requires_weight(tmp_path):
    path = _write_safetensors_checkpoint(
        tmp_path, _random_state(), {"norm": "rmsnorm", "rank": RANK, "alpha": ALPHA}
    )
    with pytest.raises(ValueError, match="norm.weight"):
        load_draft_head_from_checkpoint(path, _cfg(draft_head_norm="rmsnorm"), DEVICE)


def test_checkpoint_lora_pair_required(tmp_path):
    state = _random_state()
    state["lora_A.weight"] = torch.randn(RANK, HIDDEN)
    path = _write_safetensors_checkpoint(tmp_path, state)
    with pytest.raises(ValueError, match="both lora_A"):
        load_draft_head_from_checkpoint(path, _cfg(), DEVICE)


def test_checkpoint_vocab_hidden_mismatch_fails(tmp_path):
    state = _random_state()
    path = _write_safetensors_checkpoint(
        tmp_path,
        state,
        {"norm": "none", "rank": RANK, "alpha": ALPHA, "vocab_size": VOCAB + 1},
    )
    with pytest.raises(ValueError, match="vocab mismatch"):
        load_draft_head_from_checkpoint(path, _cfg(), DEVICE)

    path2 = _write_safetensors_checkpoint(
        tmp_path,
        state,
        {"norm": "none", "rank": RANK, "alpha": ALPHA, "hidden_size": HIDDEN + 1},
    )
    with pytest.raises(ValueError, match="hidden mismatch"):
        load_draft_head_from_checkpoint(path2, _cfg(), DEVICE)


def test_missing_checkpoint_file_fails():
    with pytest.raises(FileNotFoundError):
        load_draft_head_from_checkpoint("/nonexistent/head.pt", _cfg(), DEVICE)


class _FakeEmbed(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)


class _FakeInner(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.embed_tokens = _FakeEmbed(weight)


class _FakeModel(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.model = _FakeInner(weight)


class _FakeHFConfig:
    def __init__(self, tie: bool, vocab_size: int):
        self.tie_word_embeddings = tie
        self.vocab_size = vocab_size


class _FakeModelConfig:
    def __init__(self, tie: bool, vocab_size: int, model_path: str | None):
        self.hf_config = _FakeHFConfig(tie, vocab_size)
        self.model = model_path


def test_init_from_tied_embeddings():
    torch.manual_seed(1)
    weight = torch.randn(VOCAB, HIDDEN)
    model = _FakeModel(weight)
    model_config = _FakeModelConfig(tie=True, vocab_size=VOCAB, model_path=None)
    head = init_draft_head_from_target_model(_cfg(), model, model_config, DEVICE)
    assert head.base_projection.weight.data_ptr() != weight.data_ptr()
    assert torch.equal(head.base_projection.weight, weight)
    x = torch.randn(2, HIDDEN)
    assert torch.equal(head(x), x @ weight.T)


def test_init_from_checkpoint_lm_head(tmp_path):
    torch.manual_seed(2)
    lm_head = torch.randn(VOCAB, HIDDEN)
    save_file({"lm_head.weight": lm_head}, str(tmp_path / "model.safetensors"))
    model = torch.nn.Module()  # no embed_tokens
    model_config = _FakeModelConfig(tie=False, vocab_size=VOCAB, model_path=str(tmp_path))
    head = init_draft_head_from_target_model(_cfg(), model, model_config, DEVICE)
    assert torch.equal(head.base_projection.weight, lm_head)


def test_init_fails_without_any_source():
    model = torch.nn.Module()
    model_config = _FakeModelConfig(tie=False, vocab_size=VOCAB, model_path=None)
    with pytest.raises(ValueError, match="draft_head_path"):
        init_draft_head_from_target_model(_cfg(), model, model_config, DEVICE)


def test_load_entrypoint_prefers_checkpoint(tmp_path):
    state = _random_state()
    path = _write_safetensors_checkpoint(
        tmp_path, state, {"norm": "none", "rank": RANK, "alpha": ALPHA}
    )
    model = _FakeModel(torch.randn(VOCAB, HIDDEN))
    model_config = _FakeModelConfig(tie=True, vocab_size=VOCAB, model_path=None)
    head = load_split_dvi_draft_head(
        _cfg(draft_head_path=path), model, model_config, DEVICE
    )
    assert torch.equal(head.base_projection.weight, state["base_projection.weight"])


def test_dtype_conversion(tmp_path):
    state = _random_state()
    path = _write_safetensors_checkpoint(tmp_path, state)
    head = load_draft_head_from_checkpoint(
        path, _cfg(draft_head_dtype="bfloat16"), DEVICE
    )
    assert head.base_projection.weight.dtype == torch.bfloat16
    assert head(torch.randn(2, HIDDEN, dtype=torch.bfloat16)).dtype == torch.bfloat16


def _compact_metadata(weight: torch.Tensor) -> dict:
    return {
        "norm": "none",
        "rank": RANK,
        "alpha": ALPHA,
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "base_projection_source": "target_model",
        "base_projection_sha256": _sha256_tensor(weight),
    }


def test_compact_checkpoint_reconstructs_independent_tied_projection(tmp_path):
    torch.manual_seed(3)
    target_weight = torch.randn(VOCAB, HIDDEN)
    state = {
        "lora_A.weight": torch.randn(RANK, HIDDEN),
        "lora_B.weight": torch.randn(VOCAB, RANK),
    }
    path = _write_safetensors_checkpoint(
        tmp_path, state, _compact_metadata(target_weight)
    )
    model = _FakeModel(target_weight)
    model_config = _FakeModelConfig(tie=True, vocab_size=VOCAB, model_path=None)

    head = load_split_dvi_draft_head(
        _cfg(draft_head_path=path), model, model_config, DEVICE
    )

    assert torch.equal(head.base_projection.weight, target_weight)
    assert head.base_projection.weight.data_ptr() != target_weight.data_ptr()
    assert torch.equal(head.lora_a.weight, state["lora_A.weight"])
    assert torch.equal(head.lora_b.weight, state["lora_B.weight"])


def test_compact_checkpoint_rejects_target_projection_hash_mismatch(tmp_path):
    target_weight = torch.randn(VOCAB, HIDDEN)
    path = _write_safetensors_checkpoint(
        tmp_path,
        {
            "lora_A.weight": torch.randn(RANK, HIDDEN),
            "lora_B.weight": torch.randn(VOCAB, RANK),
        },
        _compact_metadata(target_weight),
    )
    model = _FakeModel(target_weight + 1)
    model_config = _FakeModelConfig(tie=True, vocab_size=VOCAB, model_path=None)

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_split_dvi_draft_head(
            _cfg(draft_head_path=path), model, model_config, DEVICE
        )


def test_compact_checkpoint_requires_runtime_target_source(tmp_path):
    target_weight = torch.randn(VOCAB, HIDDEN)
    path = _write_safetensors_checkpoint(
        tmp_path,
        {
            "lora_A.weight": torch.randn(RANK, HIDDEN),
            "lora_B.weight": torch.randn(VOCAB, RANK),
        },
        _compact_metadata(target_weight),
    )

    with pytest.raises(ValueError, match="requires the runtime target model"):
        load_draft_head_from_checkpoint(path, _cfg(), DEVICE)
