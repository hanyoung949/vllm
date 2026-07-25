# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SplitPPHandler DVI notifier fan-out tests.

The notifier that returns per-request DVI protocol state to READY must fire
on every non-last rank for a draft-booked step — including non-representative
stage_1 TP ranks, which never see the TCP token packet.  An earlier version
derived the flag from the received packet, so it never fired on those ranks
and their awaiting flags stayed stuck; the flag is now derived locally from
``input_batch.num_draft_tokens`` (equal to the packet kind by construction).
"""

from __future__ import annotations

from collections import deque
from unittest import mock

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.split_pp_handler import SplitPPHandler

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="handler streams need CUDA"
)


def _bare_handler(*, is_representative: bool) -> SplitPPHandler:
    handler = SplitPPHandler.__new__(SplitPPHandler)
    handler.is_last_rank = False
    handler.max_sample_len = 4
    handler.device = torch.device("cuda", torch.cuda.current_device())
    handler.main_stream = torch.cuda.current_stream(handler.device)
    handler.broadcast_stream = torch.cuda.Stream(handler.device)
    handler.queue = deque([None] * 3)
    handler.req_idx_gen_np = np.zeros(8, dtype=np.int32)
    handler._split_pp = mock.Mock(_is_representative=is_representative)
    handler._tp_group = mock.Mock()
    handler._is_stage_1 = True
    handler.dvi_token_validator = None
    handler.dvi_result_notifier = mock.Mock()
    return handler


def _input_batch(
    num_draft_tokens: int, num_draft_per_req: list[int] | None = None
) -> mock.Mock:
    ib = mock.Mock()
    ib.req_ids = ["r0", "r1"]
    ib.num_reqs = 2
    ib.num_draft_tokens = num_draft_tokens
    ib.num_draft_tokens_per_req = (
        np.array(num_draft_per_req, dtype=np.int32)
        if num_draft_per_req is not None
        else np.array([num_draft_tokens, 0], dtype=np.int32)
    )
    ib.num_computed_tokens_np = np.array([10, 10], dtype=np.int64)
    ib.prefill_len_np = np.array([10, 10], dtype=np.int64)
    ib.max_seq_len_np = np.array([64, 64], dtype=np.int64)
    ib.num_scheduled_tokens = np.array([4, 4], dtype=np.int64)
    ib.idx_mapping_np = np.array([1, 2], dtype=np.int32)
    ib.idx_mapping = torch.tensor([1, 2], dtype=torch.int32)
    return ib


def test_notifier_fires_on_non_representative_rank_for_dvi_step():
    handler = _bare_handler(is_representative=False)
    result = handler.receive(_input_batch(4, [4, 4]))
    assert result is True
    handler.dvi_result_notifier.assert_called_once_with(["r0", "r1"])
    # The TP fan-out still ran (broadcast is a no-op only off stage_1).
    assert handler._tp_group.broadcast.call_count == 2


def test_notifier_scoped_to_spec_booked_requests():
    # Mixed batch: r0 is spec-booked, r1 is a plain 0-draft row.  The
    # notifier must only mark r0 — the plain row's protocol state is none of
    # this packet's business.
    handler = _bare_handler(is_representative=False)
    handler.receive(_input_batch(4, [4, 0]))
    handler.dvi_result_notifier.assert_called_once_with(["r0"])


def test_notifier_skipped_for_normal_step():
    handler = _bare_handler(is_representative=False)
    handler.receive(_input_batch(0, [0, 0]))
    handler.dvi_result_notifier.assert_not_called()


def test_representative_validates_and_notifies():
    handler = _bare_handler(is_representative=True)
    handler.dvi_token_validator = mock.Mock()
    packet = mock.Mock()
    packet.is_dvi_block = True
    packet.to_tensors.return_value = (
        torch.ones(2, 4, dtype=torch.int64),
        torch.ones(2, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
    )
    handler._split_pp._token_transport.recv_token_packet.return_value = packet

    ib = _input_batch(4, [4, 4])
    assert handler.receive(ib) is True
    packet.validate_req_ids.assert_called_once_with(["r0", "r1"])
    handler.dvi_token_validator.assert_called_once_with(packet, ib)
    handler.dvi_result_notifier.assert_called_once_with(["r0", "r1"])
