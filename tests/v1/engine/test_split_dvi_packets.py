# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Stage-DVI split packet extensions (split_data.py)."""

from __future__ import annotations

import pytest
import torch

from vllm.v1.engine.split_data import (
    SplitPacketKind,
    SplitTensorPacket,
    SplitTokenPacket,
)

# ----------------------------------------------------------------------
# SplitTensorPacket (DVI block)
# ----------------------------------------------------------------------


def _dvi_tensor_packet(num_reqs: int = 2, k: int = 4) -> SplitTensorPacket:
    n = num_reqs * k
    return SplitTensorPacket(
        req_ids=[f"r{i}" for i in range(num_reqs)],
        num_scheduled_tokens=[k] * num_reqs,
        is_prompt=False,
        tensors={
            "hidden_states": torch.randn(n, 8, dtype=torch.bfloat16),
            "residual": torch.randn(n, 8, dtype=torch.bfloat16),
        },
        packet_kind=SplitPacketKind.DVI_BLOCK.value,
        cycle_ids=list(range(1, num_reqs + 1)),
        draft_token_ids=list(range(n)),
        draft_lengths=[k] * num_reqs,
        generation_ids=[0] * num_reqs,
        draft_positions=[p for r in range(num_reqs) for p in range(10, 10 + k)],
        policy_version=None,
        draft_version=None,
    )


def test_dvi_block_round_trip_multi_request():
    packet = _dvi_tensor_packet(num_reqs=3, k=4)
    packet.validate(vocab_size=100, max_draft_length=8)
    frames = packet.serialize()
    decoded = SplitTensorPacket.deserialize(frames)
    assert decoded.is_dvi_block
    assert decoded.req_ids == packet.req_ids
    assert decoded.cycle_ids == packet.cycle_ids
    assert decoded.draft_token_ids == packet.draft_token_ids
    assert decoded.draft_lengths == packet.draft_lengths
    assert decoded.generation_ids == packet.generation_ids
    assert decoded.draft_positions == packet.draft_positions
    assert decoded.policy_version is None
    assert decoded.draft_version is None
    assert decoded.tensors["hidden_states"].shape == (12, 8)
    assert decoded.tensors["hidden_states"].dtype == torch.bfloat16
    assert torch.equal(
        decoded.tensors["residual"].float(), packet.tensors["residual"].float()
    )
    decoded.validate(
        expected_req_ids=packet.req_ids, vocab_size=100, max_draft_length=8
    )


def test_dvi_block_version_round_trip():
    packet = _dvi_tensor_packet(num_reqs=1, k=2)
    packet.policy_version = "pol_v11"
    packet.draft_version = "draft_v3"
    decoded = SplitTensorPacket.deserialize(packet.serialize())
    assert decoded.policy_version == "pol_v11"
    assert decoded.draft_version == "draft_v3"


def test_normal_packet_round_trip_and_no_dvi_fields():
    packet = SplitTensorPacket(
        req_ids=["a"],
        num_scheduled_tokens=[5],
        is_prompt=True,
        tensors={"hidden_states": torch.randn(5, 8)},
    )
    decoded = SplitTensorPacket.deserialize(packet.serialize())
    assert not decoded.is_dvi_block
    assert decoded.cycle_ids is None
    assert decoded.draft_token_ids is None
    assert decoded.generation_ids is None
    assert decoded.draft_positions is None
    decoded.validate(expected_req_ids=["a"])
    with pytest.raises(ValueError, match="DVI metadata is set"):
        SplitTensorPacket(
            req_ids=["a"],
            num_scheduled_tokens=[1],
            is_prompt=False,
            generation_ids=[0],
        ).validate()


def test_dvi_block_variable_draft_lengths():
    packet = SplitTensorPacket(
        req_ids=["a", "b", "c"],
        num_scheduled_tokens=[4, 2, 1],
        is_prompt=False,
        tensors={"hidden_states": torch.randn(7, 8)},
        packet_kind=SplitPacketKind.DVI_BLOCK.value,
        cycle_ids=[1, 1, 1],
        draft_token_ids=[10, 11, 12, 13, 20, 21],
        draft_lengths=[4, 2, 0],
        generation_ids=[0, 0, 1],
        draft_positions=[5, 6, 7, 8, 9, 10],
    )
    packet.validate(vocab_size=100, max_draft_length=8)
    decoded = SplitTensorPacket.deserialize(packet.serialize())
    assert decoded.draft_lengths == [4, 2, 0]
    assert decoded.draft_positions == [5, 6, 7, 8, 9, 10]
    decoded.validate(vocab_size=100, max_draft_length=8)


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(cycle_ids=[1]), "cycle_ids length"),
        (dict(generation_ids=None), "missing generation_ids"),
        (dict(draft_positions=None), "missing draft_positions"),
        (dict(generation_ids=[0, -1]), "Negative generation_id"),
        (dict(draft_lengths=[4, 4, 4]), "draft_lengths length"),
        (dict(draft_token_ids=[1, 2, 3]), "draft_token_ids has 3"),
        (
            dict(draft_positions=[10, 11, 12, 13, 14]),
            "draft_positions has 5",
        ),
        (
            dict(draft_positions=[10, 11, 11, 12, 10, 11, 12, 13]),
            "not strictly increasing",
        ),
        (
            dict(
                draft_lengths=[4, -1],
                draft_token_ids=[1, 2, 3],
                draft_positions=[10, 11, 12],
            ),
            "Negative draft_length",
        ),
        (
            dict(
                draft_lengths=[4, 9],
                draft_token_ids=[1] * 13,
                draft_positions=list(range(13)),
            ),
            "max_draft_length",
        ),
        (
            dict(draft_token_ids=[10, 11, 12, 13, 14, 15, 16, 999]),
            "out of vocab range",
        ),
    ],
)
def test_dvi_block_validation_failures(overrides, match):
    packet = _dvi_tensor_packet(num_reqs=2, k=4)
    data = dict(
        req_ids=packet.req_ids,
        num_scheduled_tokens=packet.num_scheduled_tokens,
        is_prompt=False,
        tensors=packet.tensors,
        packet_kind=packet.packet_kind,
        cycle_ids=packet.cycle_ids,
        draft_token_ids=packet.draft_token_ids,
        draft_lengths=packet.draft_lengths,
        generation_ids=packet.generation_ids,
        draft_positions=packet.draft_positions,
    )
    data.update(overrides)
    bad = SplitTensorPacket(**data)
    with pytest.raises(ValueError, match=match):
        bad.validate(vocab_size=100, max_draft_length=8)


def test_dvi_block_row_count_mismatch():
    packet = _dvi_tensor_packet(num_reqs=2, k=4)
    packet.tensors["hidden_states"] = torch.randn(7, 8)
    with pytest.raises(ValueError, match="rows"):
        packet.validate()


def test_req_ids_mismatch_fails():
    packet = _dvi_tensor_packet()
    with pytest.raises(ValueError, match="req_ids mismatch"):
        packet.validate(expected_req_ids=["x", "y"])


# ----------------------------------------------------------------------
# SplitTokenPacket (DVI result)
# ----------------------------------------------------------------------


def _dvi_token_packet() -> SplitTokenPacket:
    return SplitTokenPacket(
        req_ids=["r0", "r1"],
        sampled_token_ids=[[10, 11, 12], [20]],
        num_sampled=[3, 1],
        num_rejected=[1, 3],
        packet_kind=SplitPacketKind.DVI_BLOCK.value,
        cycle_ids=[4, 2],
        accepted_counts=[2, 0],
        generation_ids=[0, 0],
        policy_version=None,
        draft_version=None,
    )


def test_dvi_token_packet_round_trip():
    packet = _dvi_token_packet()
    decoded = SplitTokenPacket.deserialize(packet.serialize())
    assert decoded.is_dvi_block
    assert decoded.cycle_ids == [4, 2]
    assert decoded.accepted_counts == [2, 0]
    assert decoded.num_sampled == [3, 1]
    assert decoded.num_rejected == [1, 3]
    assert decoded.generation_ids == [0, 0]
    decoded.validate_req_ids(["r0", "r1"])
    decoded.validate_dvi(expected_cycle_ids=[4, 2], expected_generation_ids=[0, 0])


def test_dvi_token_packet_version_echo():
    packet = _dvi_token_packet()
    packet.policy_version = "pol_v11"
    packet.draft_version = "draft_v3"
    decoded = SplitTokenPacket.deserialize(packet.serialize())
    assert decoded.policy_version == "pol_v11"
    assert decoded.draft_version == "draft_v3"
    decoded.validate_dvi(
        expected_policy_version="pol_v11", expected_draft_version="draft_v3"
    )
    with pytest.raises(ValueError, match="policy_version mismatch"):
        decoded.validate_dvi(expected_policy_version="pol_v12")
    with pytest.raises(ValueError, match="draft_version mismatch"):
        decoded.validate_dvi(expected_draft_version="draft_v4")


def test_dvi_token_packet_cycle_mismatch():
    packet = _dvi_token_packet()
    with pytest.raises(ValueError, match="cycle mismatch"):
        packet.validate_dvi(expected_cycle_ids=[4, 3])


def test_dvi_token_packet_generation_mismatch():
    packet = _dvi_token_packet()
    with pytest.raises(ValueError, match="generation mismatch"):
        packet.validate_dvi(expected_generation_ids=[0, 1])
    packet2 = _dvi_token_packet()
    packet2.generation_ids = None
    with pytest.raises(ValueError, match="generation_ids"):
        packet2.validate_dvi()


def test_dvi_token_packet_structure_failures():
    # Missing cycle ids.
    packet = _dvi_token_packet()
    packet.cycle_ids = None
    with pytest.raises(ValueError, match="cycle_ids"):
        packet.validate_dvi()
    # Duplicate req ids.
    packet = _dvi_token_packet()
    packet.req_ids = ["r0", "r0"]
    with pytest.raises(ValueError, match="duplicate"):
        packet.validate_dvi()
    # num_sampled exceeds padded list.
    packet = _dvi_token_packet()
    packet.num_sampled = [5, 1]
    with pytest.raises(ValueError, match="num_sampled"):
        packet.validate_dvi()
    # Negative num_rejected.
    packet = _dvi_token_packet()
    packet.num_rejected = [1, -1]
    with pytest.raises(ValueError, match="num_rejected"):
        packet.validate_dvi()


def test_token_packet_validate_dvi_noop_for_normal():
    packet = SplitTokenPacket(req_ids=["a"], sampled_token_ids=[[1]])
    packet.validate_dvi()  # must be a no-op for normal packets


def test_to_tensors_multi_token():
    packet = _dvi_token_packet()
    sampled, num_sampled, num_rejected = packet.to_tensors(
        device="cpu", num_reqs=2, max_sample_len=4
    )
    assert sampled.shape == (2, 4)
    assert sampled[0, :3].tolist() == [10, 11, 12]
    assert sampled[1, 0].item() == 20
    assert num_sampled.tolist() == [3, 1]
    assert num_rejected.tolist() == [1, 3]
