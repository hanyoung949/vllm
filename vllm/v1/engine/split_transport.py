# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Transport layer for layer-wise split activations.

Current scope (Phase 2):
- ZMQ TCP/PUSH-PULL for tensor packets (stage_0→stage_1, stage_1→stage_2) in a ring.
- ZMQ TCP/PUSH-PULL for token packets (stage_2→stage_0 and stage_2→stage_1) in a star.
- In-memory loopback for unit tests.
- Cross-machine deployment validated via manual endpoint configuration.

Deferred:
- Production auto-discovery, NAT traversal, TLS.
- RDMA/NIXL/GPUDirect backends (Phase 4+).
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import zmq

from vllm.v1.engine.split_data import SplitTensorPacket, SplitTokenPacket

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class SplitTransportEndpoints:
    """Endpoint addresses for a single split stage.

    Each stage binds the addresses on which it receives data and connects to the
    addresses of the next stage to which it sends data.
    """

    # Address on which this stage receives IntermediateTensors from the previous
    # stage.  stage_0 leaves this empty because it does not receive tensors.
    tensor_recv_addr: str = ""

    # Address of the next stage to which this stage sends IntermediateTensors.
    # stage_2 leaves this empty because it does not send tensors.
    tensor_send_addr: str = ""

    # Address on which this stage receives sampled tokens from stage_2.  stage_0 and stage_1
    # populate this; stage_2 leaves it empty.
    token_recv_addr: str = ""

    # Addresses of the downstream stages to which stage_2 sends sampled tokens.
    # Only stage_2 populates this (typically [stage_0_addr, stage_1_addr]).
    token_send_addrs: list[str] = field(default_factory=list)


class SplitActivationTransport(ABC):
    """Backend-agnostic transport for IntermediateTensors between split stages."""

    @abstractmethod
    def send_tensor_packet(self, packet: SplitTensorPacket) -> None:
        """Send a SplitTensorPacket to the next stage."""
        raise NotImplementedError

    @abstractmethod
    def recv_tensor_packet(
        self, device: torch.device | str = "cpu"
    ) -> SplitTensorPacket:
        """Receive a SplitTensorPacket from the previous stage."""
        raise NotImplementedError

    @abstractmethod
    def send_token_packet(self, packet: SplitTokenPacket) -> None:
        """Send a SplitTokenPacket to all configured token receivers."""
        raise NotImplementedError

    @abstractmethod
    def recv_token_packet(self) -> SplitTokenPacket:
        """Receive a SplitTokenPacket from stage_2."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class ZmqSplitActivationTransport(SplitActivationTransport):
    """TCP/ZMQ-based transport for split activations.

    Uses separate PUSH/PULL sockets for tensor and token traffic.  The stage
    receiving a traffic type binds the socket; the sending stage connects to it.
    For token traffic stage_2 may connect to multiple receivers (stage_0 and stage_1).
    """

    def __init__(
        self,
        endpoints: SplitTransportEndpoints,
        stage_label: str,
        poll_timeout_ms: int = 100,
    ) -> None:
        self._endpoints = endpoints
        self._stage_label = stage_label
        self._poll_timeout_ms = poll_timeout_ms
        self._context = zmq.Context()
        self._closed = False
        self._lock = threading.Lock()

        # Initialize every socket attribute up front so that close() stays
        # safe when __init__ aborts midway (e.g. a bind failure).
        self._tensor_recv: zmq.Socket | None = None
        self._tensor_send: zmq.Socket | None = None
        self._token_recv: zmq.Socket | None = None
        self._token_sends: list[zmq.Socket] = []
        # Optional Stage-DVI metrics sink: called with (serialize_ms, bytes)
        # for each DVI block tensor packet; wired by the model runner.
        self.dvi_metrics_sink = None

        # Tensor sockets.
        if endpoints.tensor_recv_addr:
            self._tensor_recv = self._context.socket(zmq.PULL)
            self._tensor_recv.bind(endpoints.tensor_recv_addr)
        if endpoints.tensor_send_addr:
            self._tensor_send = self._context.socket(zmq.PUSH)
            self._tensor_send.connect(endpoints.tensor_send_addr)

        # Token sockets.
        if endpoints.token_recv_addr:
            self._token_recv = self._context.socket(zmq.PULL)
            self._token_recv.bind(endpoints.token_recv_addr)
        for addr in endpoints.token_send_addrs:
            sock = self._context.socket(zmq.PUSH)
            sock.connect(addr)
            self._token_sends.append(sock)

    def send_tensor_packet(self, packet: SplitTensorPacket) -> None:
        if self._tensor_send is None:
            raise RuntimeError(
                f"Stage {self._stage_label} is not configured to send tensor packets."
            )
        t0 = time.perf_counter_ns()
        frames = packet.serialize()
        if packet.is_dvi_block and self.dvi_metrics_sink is not None:
            self.dvi_metrics_sink(
                (time.perf_counter_ns() - t0) / 1e6,
                sum(len(f) for f in frames),
            )
        with self._lock:
            self._tensor_send.send_multipart(frames)

    def recv_tensor_packet(
        self, device: torch.device | str = "cpu"
    ) -> SplitTensorPacket:
        if self._tensor_recv is None:
            raise RuntimeError(
                f"Stage {self._stage_label} is not configured to receive tensor packets."
            )
        frames = self._tensor_recv.recv_multipart()
        return SplitTensorPacket.deserialize(frames, device=device)

    def send_token_packet(self, packet: SplitTokenPacket) -> None:
        if not self._token_sends:
            raise RuntimeError(
                f"Stage {self._stage_label} is not configured to send token packets."
            )
        data = packet.serialize()
        with self._lock:
            for sock in self._token_sends:
                sock.send(data)

    def recv_token_packet(self) -> SplitTokenPacket:
        if self._token_recv is None:
            raise RuntimeError(
                f"Stage {self._stage_label} is not configured to receive token packets."
            )
        data = self._token_recv.recv()
        return SplitTokenPacket.deserialize(data)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for sock in (
            self._tensor_recv,
            self._tensor_send,
            self._token_recv,
            *self._token_sends,
        ):
            if sock is not None:
                sock.close()
        self._context.term()

    def __del__(self) -> None:
        self.close()


class LocalLoopbackTransport(SplitActivationTransport):
    """In-memory transport for unit tests.

    A pair of transports where ``a`` sends tensors to ``b`` and ``b`` sends
    tokens back to ``a``.  This matches the stage_0 -> stage_1/stage_2 tensor flow plus the
    stage_2 -> stage_0 token flow.
    """

    def __init__(
        self,
        tensor_send_queue: "queue.Queue[list[bytes]]",
        tensor_recv_queue: "queue.Queue[list[bytes]]",
        token_send_queues: "list[queue.Queue[bytes]]",
        token_recv_queue: "queue.Queue[bytes]",
    ) -> None:
        self._tensor_send_queue = tensor_send_queue
        self._tensor_recv_queue = tensor_recv_queue
        self._token_send_queues = token_send_queues
        self._token_recv_queue = token_recv_queue
        self._closed = False

    def send_tensor_packet(self, packet: SplitTensorPacket) -> None:
        frames = packet.serialize()
        self._tensor_send_queue.put(frames)

    def recv_tensor_packet(
        self, device: torch.device | str = "cpu"
    ) -> SplitTensorPacket:
        frames = self._tensor_recv_queue.get(timeout=30)
        return SplitTensorPacket.deserialize(frames, device=device)

    def send_token_packet(self, packet: SplitTokenPacket) -> None:
        data = packet.serialize()
        for q in self._token_send_queues:
            q.put(data)

    def recv_token_packet(self) -> SplitTokenPacket:
        data = self._token_recv_queue.get(timeout=30)
        return SplitTokenPacket.deserialize(data)

    def close(self) -> None:
        self._closed = True

    @classmethod
    def make_pair(cls) -> tuple["LocalLoopbackTransport", "LocalLoopbackTransport"]:
        import queue

        tensor_ab: queue.Queue[list[bytes]] = queue.Queue()
        tensor_ba: queue.Queue[list[bytes]] = queue.Queue()
        token_ab: queue.Queue[bytes] = queue.Queue()
        token_ba: queue.Queue[bytes] = queue.Queue()
        # ``a`` sends tensors via tensor_ab (read by ``b``) and receives tokens
        # via token_ba (written by ``b``).
        a = cls(tensor_ab, tensor_ba, [token_ab], token_ba)
        b = cls(tensor_ba, tensor_ab, [token_ba], token_ab)
        return a, b

    @classmethod
    def make_star(
        cls,
    ) -> tuple[
        "LocalLoopbackTransport",
        "LocalLoopbackTransport",
        "LocalLoopbackTransport",
    ]:
        """Create a 3-stage star loopback (stage_0, stage_1, stage_2).

        Tensor flow: stage_0 -> stage_1 -> stage_2.
        Token flow: stage_2 -> [stage_0, stage_1].
        """
        import queue

        tensor_s0_s1: queue.Queue[list[bytes]] = queue.Queue()
        tensor_s1_s2: queue.Queue[list[bytes]] = queue.Queue()
        token_s2_s0: queue.Queue[bytes] = queue.Queue()
        token_s2_s1: queue.Queue[bytes] = queue.Queue()

        s0 = cls(
            tensor_send_queue=tensor_s0_s1,
            tensor_recv_queue=None,  # type: ignore[arg-type]
            token_send_queues=[],
            token_recv_queue=token_s2_s0,
        )
        s1 = cls(
            tensor_send_queue=tensor_s1_s2,
            tensor_recv_queue=tensor_s0_s1,
            token_send_queues=[],
            token_recv_queue=token_s2_s1,
        )
        s2 = cls(
            tensor_send_queue=None,  # type: ignore[arg-type]
            tensor_recv_queue=tensor_s1_s2,
            token_send_queues=[token_s2_s0, token_s2_s1],
            token_recv_queue=None,  # type: ignore[arg-type]
        )
        return s0, s1, s2
