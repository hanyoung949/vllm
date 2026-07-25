# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Stage-DVI greedy block verifier."""

from __future__ import annotations

import pytest
import torch

from vllm.v1.engine.split_data import SplitDVIProtocolError
from vllm.v1.worker.gpu.split_dvi.block_verifier import (
    SplitDVIGreedyBlockVerifier,
)


def _logits_for_targets(target_ids: list[int], vocab: int = 50) -> torch.Tensor:
    """Build logits whose argmax at each row equals target_ids[row]."""
    logits = torch.full((len(target_ids), vocab), -10.0)
    for i, t in enumerate(target_ids):
        logits[i, t] = 10.0
    return logits


V = SplitDVIGreedyBlockVerifier()


def test_first_token_rejected_returns_single_correction():
    # draft [A,B,C,D] vs target [X,...] -> sampled = [X], accepted = 0.
    logits = _logits_for_targets([7, 8, 9, 10])
    result = V.verify(
        logits,
        req_ids=["r0"],
        draft_token_ids=[1, 2, 3, 4],
        draft_lengths=[4],
        cu_num_logits=[0, 4],
    )
    assert result.sampled_token_ids == [[7]]
    assert result.accepted_counts == [0]
    assert result.num_sampled == [1]


def test_middle_rejection_returns_prefix_plus_correction():
    # draft [A,B,C,D] vs target [A,B,X,.] -> sampled = [A,B,X], accepted = 2.
    logits = _logits_for_targets([1, 2, 9, 10])
    result = V.verify(
        logits,
        req_ids=["r0"],
        draft_token_ids=[1, 2, 3, 4],
        draft_lengths=[4],
        cu_num_logits=[0, 4],
    )
    assert result.sampled_token_ids == [[1, 2, 9]]
    assert result.accepted_counts == [2]
    assert result.num_sampled == [3]


def test_last_position_rejection():
    # draft [A,B,C,D] vs target [A,B,C,X] -> sampled = [A,B,C,X], accepted = 3.
    logits = _logits_for_targets([1, 2, 3, 10])
    result = V.verify(
        logits,
        req_ids=["r0"],
        draft_token_ids=[1, 2, 3, 4],
        draft_lengths=[4],
        cu_num_logits=[0, 4],
    )
    assert result.sampled_token_ids == [[1, 2, 3, 10]]
    assert result.accepted_counts == [3]
    assert result.num_sampled == [4]


def test_all_accepted_no_bonus():
    # draft == target for all k rows -> sampled = all drafts, num_sampled = k.
    logits = _logits_for_targets([1, 2, 3, 4])
    result = V.verify(
        logits,
        req_ids=["r0"],
        draft_token_ids=[1, 2, 3, 4],
        draft_lengths=[4],
        cu_num_logits=[0, 4],
    )
    assert result.sampled_token_ids == [[1, 2, 3, 4]]
    assert result.accepted_counts == [4]
    assert result.num_sampled == [4]


def test_draft_length_one_equals_plain_greedy():
    # A single-position block behaves like ordinary greedy decode.
    logits = _logits_for_targets([5])
    result = V.verify(
        logits,
        req_ids=["r0"],
        draft_token_ids=[5],
        draft_lengths=[1],
        cu_num_logits=[0, 1],
    )
    assert result.sampled_token_ids == [[5]]
    assert result.num_sampled == [1]


def test_prefill_row_plain_argmax():
    # draft_length=0 rows (prefill) commit exactly one argmax token.
    logits = _logits_for_targets([11])
    result = V.verify(
        logits,
        req_ids=["r0"],
        draft_token_ids=[],
        draft_lengths=[0],
        cu_num_logits=[0, 1],
    )
    assert result.sampled_token_ids == [[11]]
    assert result.accepted_counts == [0]
    assert result.num_sampled == [1]


def test_correction_eos_propagates():
    # When the target correction is EOS, it is committed like any token.
    eos = 49
    logits = _logits_for_targets([1, eos, 3, 4], vocab=50)
    result = V.verify(
        logits,
        req_ids=["r0"],
        draft_token_ids=[1, 2, 3, 4],
        draft_lengths=[4],
        cu_num_logits=[0, 4],
    )
    assert result.sampled_token_ids == [[1, eos]]
    assert result.num_sampled == [2]


def test_multi_request_mixed_lengths():
    logits = _logits_for_targets([1, 2, 3, 4, 20, 21, 22, 23])
    result = V.verify(
        logits,
        req_ids=["r0", "r1"],
        draft_token_ids=[1, 2, 9, 4, 20, 21, 22, 23],
        draft_lengths=[4, 4],
        cu_num_logits=[0, 4, 8],
    )
    assert result.sampled_token_ids == [[1, 2, 3], [20, 21, 22, 23]]
    assert result.accepted_counts == [2, 4]
    assert result.num_sampled == [3, 4]


def test_to_padded_tensors_shape_and_values():
    logits = _logits_for_targets([1, 2, 9, 10])
    result = V.verify(
        logits,
        req_ids=["r0"],
        draft_token_ids=[1, 2, 3, 4],
        draft_lengths=[4],
        cu_num_logits=[0, 4],
    )
    sampled, num_sampled = result.to_padded_tensors(4, torch.device("cpu"))
    assert sampled.shape == (1, 4)
    assert sampled.dtype == torch.int64
    assert sampled[0, :3].tolist() == [1, 2, 9]
    assert num_sampled.tolist() == [3]


def test_mismatched_metadata_fails_fast():
    logits = _logits_for_targets([1, 2, 3, 4])
    with pytest.raises(SplitDVIProtocolError, match="cu_num_logits"):
        V.verify(logits, ["r0"], [1, 2, 3, 4], [4], [0, 4, 8])
    with pytest.raises(SplitDVIProtocolError, match="draft_lengths"):
        V.verify(logits, ["r0"], [1, 2, 3, 4], [4, 4], [0, 4])
    with pytest.raises(SplitDVIProtocolError, match="draft_token_ids"):
        V.verify(logits, ["r0"], [1, 2, 3], [4], [0, 4])
    with pytest.raises(SplitDVIProtocolError, match="draft_length 5 > num_logits"):
        V.verify(logits, ["r0"], [1, 2, 3, 4, 5], [5], [0, 4])
    with pytest.raises(SplitDVIProtocolError, match="zero logit rows"):
        V.verify(logits, ["r0", "r1"], [1, 2, 3, 4], [4, 0], [0, 4, 4])
