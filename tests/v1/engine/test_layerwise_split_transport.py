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


def test_dvi_block_send_reports_serialize_metrics():
    """The DVI metrics sink fires for DVI block packets only (not NORMAL)."""
    import threading
    from unittest import mock

    def _bare_transport(sink):
        t = ZmqSplitActivationTransport.__new__(ZmqSplitActivationTransport)
        t._tensor_send = mock.Mock()
        t._lock = threading.Lock()
        t._stage_label = "PP0"
        t.dvi_metrics_sink = sink
        return t

    packet = mock.Mock()
    packet.is_dvi_block = True
    packet.serialize.return_value = [b"abc", b"defg"]

    sink = mock.Mock()
    _bare_transport(sink).send_tensor_packet(packet)
    sink.assert_called_once()
    serialize_ms, num_bytes = sink.call_args[0]
    assert serialize_ms >= 0.0
    assert num_bytes == 7

    normal = mock.Mock()
    normal.is_dvi_block = False
    normal.serialize.return_value = [b"x"]
    sink2 = mock.Mock()
    _bare_transport(sink2).send_tensor_packet(normal)
    sink2.assert_not_called()


def test_wire_model_disabled_by_default():
    """Without the env knobs the wire model is fully bypassed."""
    t = ZmqSplitActivationTransport(
        SplitTransportEndpoints(), stage_label="stage_0"
    )
    try:
        assert t._wire_queue is None
        assert t._wire_thread is None
        assert t._wire_latency_s == 0.0
        assert t._wire_bps == 0.0
    finally:
        t.close()


def _make_wire_pair(monkeypatch, latency_ms: str, bps: str, port: int):
    """One-hop sender/receiver pair with the wire model configured."""
    monkeypatch.setenv("VLLM_SPLIT_TRANSPORT_LATENCY_MS", latency_ms)
    monkeypatch.setenv("VLLM_SPLIT_TRANSPORT_BPS", bps)
    sender = ZmqSplitActivationTransport(
        SplitTransportEndpoints(tensor_send_addr=f"tcp://127.0.0.1:{port}"),
        stage_label="stage_0",
    )
    receiver = ZmqSplitActivationTransport(
        SplitTransportEndpoints(tensor_recv_addr=f"tcp://127.0.0.1:{port}"),
        stage_label="stage_1",
    )
    time.sleep(0.2)  # Allow ZMQ sockets to connect.
    return sender, receiver


def _tensor_packet(req_id: str, value: float, n: int = 2048):
    return SplitTensorPacket(
        req_ids=[req_id],
        num_scheduled_tokens=[1],
        is_prompt=False,
        tensors={
            "hidden_states": torch.full((1, n), value, dtype=torch.float16),
        },
    )


def test_wire_latency_delays_delivery(monkeypatch):
    """A configured one-way latency delays every hop by at least that amount,
    and FIFO ordering is preserved across packets."""
    sender, receiver = _make_wire_pair(monkeypatch, "60", "0", 15301)
    try:
        assert sender._wire_thread is not None
        t0 = time.perf_counter()
        for i in range(3):
            sender.send_tensor_packet(_tensor_packet(f"r{i}", float(i)))
        for i in range(3):
            received = receiver.recv_tensor_packet(device="cpu")
            assert received.req_ids == [f"r{i}"]
        elapsed = time.perf_counter() - t0
        # First packet alone must pay the full one-way latency.
        assert elapsed >= 0.055, f"latency not applied: {elapsed:.3f}s"
    finally:
        sender.close()
        receiver.close()


def test_wire_bandwidth_serializes_delivery(monkeypatch):
    """The bps cap serializes the wire: back-to-back packets arrive spaced by
    at least bytes/bps."""
    # Payload ~4.2KB per packet; cap 100 B/ms -> >=42 ms between arrivals.
    sender, receiver = _make_wire_pair(monkeypatch, "0", "100000", 15311)
    try:
        sender.send_tensor_packet(_tensor_packet("a", 1.0))
        sender.send_tensor_packet(_tensor_packet("b", 2.0))
        t0 = time.perf_counter()
        first = receiver.recv_tensor_packet(device="cpu")
        t_first = time.perf_counter() - t0
        second = receiver.recv_tensor_packet(device="cpu")
        t_second = time.perf_counter() - t0
        assert first.req_ids == ["a"]
        assert second.req_ids == ["b"]
        nbytes = 2048 * 2  # hidden fp16 payload bytes (lower bound)
        assert t_second - t_first >= nbytes / 100000 * 0.9, (
            f"wire not serialized: gap {t_second - t_first:.3f}s"
        )
    finally:
        sender.close()
        receiver.close()


def test_wire_close_drains_in_flight(monkeypatch):
    """close() must not lose a packet still inside the synthetic wire."""
    sender, receiver = _make_wire_pair(monkeypatch, "80", "0", 15321)
    sender.send_tensor_packet(_tensor_packet("last", 9.0))
    sender.close()  # returns only after the wire has delivered
    received = receiver.recv_tensor_packet(device="cpu")
    assert received.req_ids == ["last"]
    receiver.close()
