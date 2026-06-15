# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SplitTensorPacket / SplitTokenPacket serialization."""

import torch

from vllm.v1.engine.split_data import SplitTensorPacket, SplitTokenPacket


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
        assert False, "Expected ValueError"
    except ValueError:
        pass
