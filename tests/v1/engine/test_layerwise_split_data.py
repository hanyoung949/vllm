# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SplitTensorPacket / SplitTokenPacket serialization."""

import torch

from vllm.v1.engine.split_data import (
    SplitDVIProtocolError,
    SplitTensorPacket,
    SplitTokenPacket,
)


def test_split_tensor_packet_roundtrip_fp16():
    tensors = {
        "hidden_states": torch.randn(2, 2048, dtype=torch.float16),
        "residual": torch.randn(2, 2048, dtype=torch.float16),
    }
    packet = SplitTensorPacket(
        req_ids=["req-0", "req-1"],
        num_scheduled_tokens=[10, 5],
        is_prompt=True,
        tensors=tensors,
    )
    frames = packet.serialize()
    restored = SplitTensorPacket.deserialize(frames)

    assert restored.req_ids == packet.req_ids
    assert restored.num_scheduled_tokens == packet.num_scheduled_tokens
    assert restored.is_prompt == packet.is_prompt
    assert set(restored.tensors.keys()) == set(tensors.keys())
    for key in tensors:
        torch.testing.assert_close(restored.tensors[key], tensors[key])


def test_split_tensor_packet_roundtrip_bf16():
    tensors = {
        "hidden_states": torch.randn(2, 2048, dtype=torch.bfloat16),
    }
    packet = SplitTensorPacket(
        req_ids=["req-0"],
        num_scheduled_tokens=[3],
        is_prompt=False,
        tensors=tensors,
    )
    frames = packet.serialize()
    restored = SplitTensorPacket.deserialize(frames)

    assert restored.tensors["hidden_states"].dtype == torch.bfloat16
    torch.testing.assert_close(
        restored.tensors["hidden_states"], tensors["hidden_states"]
    )


def test_split_token_packet_roundtrip():
    packet = SplitTokenPacket(
        req_ids=["req-0", "req-1"],
        sampled_token_ids=[[42], [12345]],
        finish_reasons=[None, "stop"],
    )
    data = packet.serialize()
    restored = SplitTokenPacket.deserialize(data)

    assert restored.req_ids == packet.req_ids
    assert restored.sampled_token_ids == packet.sampled_token_ids
    assert restored.finish_reasons == packet.finish_reasons


def test_split_token_packet_from_token_tensor():
    token_tensor = torch.tensor([[42], [12345]], dtype=torch.int32)
    packet = SplitTokenPacket.from_token_tensor(
        req_ids=["req-0", "req-1"], token_tensor=token_tensor
    )
    assert packet.sampled_token_ids == [[42], [12345]]

    reconstructed = packet.to_token_tensor(device="cpu", num_reqs=2)
    torch.testing.assert_close(reconstructed, token_tensor)


def test_split_token_packet_wrong_shape_raises():
    wrong = torch.tensor([42, 12345], dtype=torch.int32)
    try:
        SplitTokenPacket.from_token_tensor(req_ids=["a", "b"], token_tensor=wrong)
        assert False, "Expected SplitDVIProtocolError"
    except SplitDVIProtocolError:
        pass


def test_split_token_packet_v2_roundtrip_with_metadata():
    req_ids = ["req_0", "req_1", "req_2"]
    sampled_token_ids = torch.tensor(
        [[1, 2, 3], [4, 5, 0], [6, 0, 0]], dtype=torch.int64
    )
    num_sampled = torch.tensor([3, 2, 1], dtype=torch.int32)
    num_rejected = torch.tensor([0, 1, 0], dtype=torch.int32)

    packet = SplitTokenPacket.from_tensors(
        req_ids=req_ids,
        sampled_token_ids=sampled_token_ids,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
    )
    restored = SplitTokenPacket.deserialize(packet.serialize())

    assert restored.req_ids == req_ids
    assert restored.num_sampled == num_sampled.tolist()
    assert restored.num_rejected == num_rejected.tolist()

    out_sampled, out_num_sampled, out_num_rejected = restored.to_tensors(
        device="cpu", num_reqs=3, max_sample_len=3
    )
    expected_sampled = torch.tensor(
        [[1, 2, 3], [4, 5, 0], [6, 0, 0]], dtype=torch.int64
    )
    torch.testing.assert_close(out_sampled, expected_sampled)
    torch.testing.assert_close(out_num_sampled, num_sampled)
    torch.testing.assert_close(out_num_rejected, num_rejected)


def test_split_token_packet_v2_req_count_mismatch_raises():
    packet = SplitTokenPacket.from_tensors(
        req_ids=["req_0", "req_1"],
        sampled_token_ids=torch.tensor([[1], [2]], dtype=torch.int64),
        num_sampled=torch.tensor([1, 1], dtype=torch.int32),
        num_rejected=torch.tensor([0, 0], dtype=torch.int32),
    )
    restored = SplitTokenPacket.deserialize(packet.serialize())

    try:
        restored.to_tensors(device="cpu", num_reqs=3, max_sample_len=1)
        assert False, "Expected SplitDVIProtocolError for req count mismatch"
    except SplitDVIProtocolError:
        pass

    try:
        restored.validate_req_ids(["req_0", "req_1", "req_2"])
        assert False, "Expected SplitDVIProtocolError for validate_req_ids count mismatch"
    except SplitDVIProtocolError:
        pass


def test_split_token_packet_v2_req_order_mismatch_raises():
    req_ids = ["req_0", "req_1", "req_2"]
    packet = SplitTokenPacket.from_tensors(
        req_ids=req_ids,
        sampled_token_ids=torch.tensor([[1], [2], [3]], dtype=torch.int64),
        num_sampled=torch.tensor([1, 1, 1], dtype=torch.int32),
        num_rejected=torch.tensor([0, 0, 0], dtype=torch.int32),
    )

    # Same set, wrong order: must fail fast.
    try:
        packet.validate_req_ids(["req_1", "req_0", "req_2"])
        assert False, "Expected SplitDVIProtocolError for req order mismatch"
    except SplitDVIProtocolError:
        pass
