# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for split activation transport backends."""

import time

import torch

from vllm.v1.engine.split_data import SplitTensorPacket, SplitTokenPacket
from vllm.v1.engine.split_transport import (
    LocalLoopbackTransport,
    SplitTransportEndpoints,
    ZmqSplitActivationTransport,
)


def test_local_loopback_tensor_roundtrip():
    a, b = LocalLoopbackTransport.make_pair()
    tensors = {
        "hidden_states": torch.randn(2, 2048, dtype=torch.float16),
        "residual": torch.randn(2, 2048, dtype=torch.bfloat16),
    }
    packet = SplitTensorPacket(
        req_ids=["req-0"],
        num_scheduled_tokens=[7],
        is_prompt=True,
        tensors=tensors,
    )
    a.send_tensor_packet(packet)
    received = b.recv_tensor_packet(device="cpu")

    assert received.req_ids == packet.req_ids
    assert received.num_scheduled_tokens == packet.num_scheduled_tokens
    assert received.is_prompt == packet.is_prompt
    for key in tensors:
        torch.testing.assert_close(received.tensors[key], tensors[key])


def test_local_loopback_token_roundtrip():
    a, b = LocalLoopbackTransport.make_pair()
    packet = SplitTokenPacket(
        req_ids=["req-0", "req-1"],
        sampled_token_ids=[[1], [2]],
        finish_reasons=[None, "length"],
    )
    b.send_token_packet(packet)
    received = a.recv_token_packet()

    assert received.req_ids == packet.req_ids
    assert received.sampled_token_ids == packet.sampled_token_ids
    assert received.finish_reasons == packet.finish_reasons


def test_local_loopback_star_token_broadcast():
    """stage_2 sends tokens to both stage_0 and stage_1; each receives the same packet."""
    s0, s1, s2 = LocalLoopbackTransport.make_star()
    token_tensor = torch.tensor([[42], [12345]], dtype=torch.int32)
    packet = SplitTokenPacket.from_token_tensor(
        req_ids=["req-0", "req-1"], token_tensor=token_tensor
    )
    s2.send_token_packet(packet)

    s0_received = s0.recv_token_packet()
    s1_received = s1.recv_token_packet()

    for received in (s0_received, s1_received):
        assert received.req_ids == packet.req_ids
        assert received.sampled_token_ids == packet.sampled_token_ids
        reconstructed = received.to_token_tensor(device="cpu", num_reqs=2)
        torch.testing.assert_close(reconstructed, token_tensor)


def test_zmq_transport_roundtrip():
    s0_endpoints = SplitTransportEndpoints(
        tensor_send_addr="tcp://127.0.0.1:15201",
        token_recv_addr="tcp://127.0.0.1:15203",
    )
    s1_endpoints = SplitTransportEndpoints(
        tensor_recv_addr="tcp://127.0.0.1:15201",
        tensor_send_addr="tcp://127.0.0.1:15202",
    )
    s2_endpoints = SplitTransportEndpoints(
        tensor_recv_addr="tcp://127.0.0.1:15202",
        token_send_addrs=["tcp://127.0.0.1:15203"],
    )

    s0 = ZmqSplitActivationTransport(s0_endpoints, stage_label="stage_0")
    s1 = ZmqSplitActivationTransport(s1_endpoints, stage_label="stage_1")
    s2 = ZmqSplitActivationTransport(s2_endpoints, stage_label="stage_2")

    time.sleep(0.2)  # Allow ZMQ sockets to connect.

    try:
        tensor_packet = SplitTensorPacket(
            req_ids=["r0"],
            num_scheduled_tokens=[3],
            is_prompt=True,
            tensors={
                "hidden_states": torch.randn(1, 2048, dtype=torch.float16),
            },
        )
        token_packet = SplitTokenPacket(
            req_ids=["r0"],
            sampled_token_ids=[[7]],
            finish_reasons=[None],
        )

        s0.send_tensor_packet(tensor_packet)
        recv_s1 = s1.recv_tensor_packet(device="cpu")
        assert recv_s1.req_ids == tensor_packet.req_ids
        torch.testing.assert_close(
            recv_s1.tensors["hidden_states"], tensor_packet.tensors["hidden_states"]
        )

        s1.send_tensor_packet(recv_s1)
        recv_s2 = s2.recv_tensor_packet(device="cpu")
        torch.testing.assert_close(
            recv_s2.tensors["hidden_states"], tensor_packet.tensors["hidden_states"]
        )

        s2.send_token_packet(token_packet)
        recv_s0 = s0.recv_token_packet()
        assert recv_s0.sampled_token_ids == token_packet.sampled_token_ids
    finally:
        s0.close()
        s1.close()
        s2.close()


def test_zmq_star_token_broadcast():
    """stage_2 sends sampled tokens to both stage_0 and stage_1 over separate ZMQ sockets."""
    s0_endpoints = SplitTransportEndpoints(
        tensor_recv_addr="tcp://127.0.0.1:15211",
        token_recv_addr="tcp://127.0.0.1:15213",
    )
    s1_endpoints = SplitTransportEndpoints(
        tensor_recv_addr="tcp://127.0.0.1:15212",
        token_recv_addr="tcp://127.0.0.1:15214",
    )
    s2_endpoints = SplitTransportEndpoints(
        tensor_recv_addr="tcp://127.0.0.1:15215",
        token_send_addrs=[
            "tcp://127.0.0.1:15213",
            "tcp://127.0.0.1:15214",
        ],
    )

    s0 = ZmqSplitActivationTransport(s0_endpoints, stage_label="stage_0")
    s1 = ZmqSplitActivationTransport(s1_endpoints, stage_label="stage_1")
    s2 = ZmqSplitActivationTransport(s2_endpoints, stage_label="stage_2")

    time.sleep(0.2)

    try:
        token_tensor = torch.tensor([[42], [12345]], dtype=torch.int32)
        packet = SplitTokenPacket.from_token_tensor(
            req_ids=["req-0", "req-1"], token_tensor=token_tensor
        )

        s2.send_token_packet(packet)

        s0_received = s0.recv_token_packet()
        s1_received = s1.recv_token_packet()

        for received in (s0_received, s1_received):
            assert received.req_ids == packet.req_ids
            assert received.sampled_token_ids == packet.sampled_token_ids
            reconstructed = received.to_token_tensor(device="cpu", num_reqs=2)
            torch.testing.assert_close(reconstructed, token_tensor)
    finally:
        s0.close()
        s1.close()
        s2.close()
