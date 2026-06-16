# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ray-based endpoint registry for layer-wise split auto-discovery.

When ``enable_layerwise_split`` is used with the Ray executor, each PP rank
needs to know the ZMQ receive addresses of every other rank.  Rather than
requiring users to manually export ``VLLM_SPLIT_TENSOR_RECV_ADDRS`` and
``VLLM_SPLIT_TOKEN_RECV_ADDRS`` on every node, the driver can create a
``SplitEndpointRegistry`` actor.  Each worker registers its own IP and chosen
ports, then fetches the complete address list before building its split
transport.

Manual address configuration still takes precedence when set.
"""

from __future__ import annotations

import os
import socket
import threading
from typing import TYPE_CHECKING

from vllm import envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from typing import Optional

try:
    import ray
except ImportError:
    ray = None  # type: ignore

logger = init_logger(__name__)


def get_split_endpoint_registry_name(instance_id: str) -> str:
    """Return a unique Ray actor name for this engine instance."""
    return f"vllm_split_endpoint_registry_{instance_id}"


def get_open_port() -> int:
    """Return an available TCP port on the local machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


if ray is not None:

    @ray.remote(num_cpus=1)
    class SplitEndpointRegistry:
        """Central registry where split stages publish their ZMQ endpoints."""

        def __init__(self, world_size: int) -> None:
            self.world_size = world_size
            self._endpoints: dict[int, tuple[str, int, int]] = {}
            self._event = threading.Event()

        def register(
            self,
            rank: int,
            ip: str,
            tensor_port: int,
            token_port: int,
        ) -> None:
            """Register the receive addresses for one split stage."""
            self._endpoints[rank] = (ip, tensor_port, token_port)
            logger.info(
                "SplitEndpointRegistry: rank %d registered at %s (tensor=%d, "
                "token=%d); have %d/%d",
                rank,
                ip,
                tensor_port,
                token_port,
                len(self._endpoints),
                self.world_size,
            )
            if len(self._endpoints) == self.world_size:
                self._event.set()

        def get_endpoints(self) -> tuple[list[str], list[str]]:
            """Return the full list of tensor and token receive addresses.

            Blocks until every rank has registered.
            """
            timeout_seconds = 600
            if not self._event.wait(timeout=timeout_seconds):
                registered = sorted(self._endpoints.keys())
                raise RuntimeError(
                    f"SplitEndpointRegistry timeout: only {len(registered)}/"
                    f"{self.world_size} ranks registered (ranks={registered}). "
                    "Check that all split workers started and can reach the "
                    "registry actor."
                )

            tensor_addrs: list[str] = []
            token_addrs: list[str] = []
            for r in range(self.world_size):
                ip, tensor_port, token_port = self._endpoints[r]
                tensor_addrs.append(f"tcp://{ip}:{tensor_port}")
                token_addrs.append(f"tcp://{ip}:{token_port}")
            return tensor_addrs, token_addrs


def create_split_endpoint_registry(
    instance_id: str,
    world_size: int,
):
    """Create and return a named ``SplitEndpointRegistry`` actor.

    The actor name is deterministic per engine instance so that Ray workers
    can look it up via ``VLLM_SPLIT_ENDPOINT_REGISTRY_NAME``.
    """
    if ray is None:
        raise ImportError(
            "Layer-wise split endpoint auto-discovery requires Ray."
        )
    name = get_split_endpoint_registry_name(instance_id)
    return SplitEndpointRegistry.options(
        name=name,
        get_if_exists=True,
        max_concurrency=world_size,
    ).remote(world_size), name


def _get_interface_ip(iface: str) -> str | None:
    """Return the IPv4 address assigned to ``iface``, or ``None``."""
    try:
        import fcntl
        import struct

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        addr_bytes = fcntl.ioctl(
            sock.fileno(),
            0x8915,  # SIOCGIFADDR
            struct.pack("256s", iface.encode("utf-8")[:15]),
        )[20:24]
        sock.close()
        return socket.inet_ntoa(addr_bytes)
    except Exception:
        return None


def _get_worker_node_ip() -> str:
    """Return the IP address this split worker should bind/connect to.

    Priority:
      1. For explicit split stage-node maps, use the current worker node IP
         instead of ``VLLM_HOST_IP``. Ray runtime envs can accidentally copy
         the driver host IP to remote split workers.
      2. ``VLLM_HOST_IP`` if explicitly configured per node.
      3. The IP of the interface named in ``NCCL_SOCKET_IFNAME``.
      4. Ray node IP address.
      5. Generic outgoing-interface detection.
    """
    if os.environ.get("VLLM_SPLIT_STAGE_NODE_MAP"):
        nccl_iface = os.environ.get("NCCL_SOCKET_IFNAME", "")
        if nccl_iface:
            first_iface = nccl_iface.split(",")[0].strip()
            if first_iface:
                iface_ip = _get_interface_ip(first_iface)
                if iface_ip:
                    return iface_ip

        if ray is not None:
            try:
                return ray.util.get_node_ip_address()
            except Exception:
                pass

    host_ip = envs.VLLM_HOST_IP
    if host_ip:
        return host_ip

    nccl_iface = os.environ.get("NCCL_SOCKET_IFNAME", "")
    if nccl_iface:
        # NCCL supports comma-separated interface names; use the first one.
        first_iface = nccl_iface.split(",")[0].strip()
        if first_iface:
            iface_ip = _get_interface_ip(first_iface)
            if iface_ip:
                return iface_ip

    if ray is not None:
        try:
            return ray.util.get_node_ip_address()
        except Exception:
            pass

    # Last resort: use generic outgoing-interface detection.
    from vllm.utils.network_utils import get_ip

    return get_ip()


def discover_split_endpoints(
    pp_rank: int,
    world_size: int,
    registry_name: str,
) -> tuple[list[str], list[str]]:
    """Register this rank's endpoints and fetch the complete address list.

    The returned addresses are also written to ``os.environ`` so that legacy
    code that reads ``VLLM_SPLIT_TENSOR_RECV_ADDRS`` /
    ``VLLM_SPLIT_TOKEN_RECV_ADDRS`` directly will see the discovered values.
    """
    if ray is None:
        raise ImportError(
            "Layer-wise split endpoint auto-discovery requires Ray."
        )

    registry = ray.get_actor(registry_name)
    ip = _get_worker_node_ip()
    tensor_port = get_open_port()
    token_port = get_open_port()

    logger.info(
        "Registering split endpoints for rank %d/%d: ip=%s tensor_port=%d "
        "token_port=%d",
        pp_rank,
        world_size,
        ip,
        tensor_port,
        token_port,
    )

    ray.get(registry.register.remote(pp_rank, ip, tensor_port, token_port))
    tensor_addrs, token_addrs = ray.get(registry.get_endpoints.remote())

    os.environ["VLLM_SPLIT_TENSOR_RECV_ADDRS"] = ",".join(tensor_addrs)
    os.environ["VLLM_SPLIT_TOKEN_RECV_ADDRS"] = ",".join(token_addrs)
    logger.info(
        "Discovered split endpoints: tensor=%s token=%s",
        tensor_addrs,
        token_addrs,
    )
    return tensor_addrs, token_addrs
