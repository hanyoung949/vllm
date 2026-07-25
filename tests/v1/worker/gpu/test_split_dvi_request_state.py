# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Stage-DVI request state tracker."""

from __future__ import annotations

import pytest

from vllm.v1.engine.split_data import SplitDVIProtocolError
from vllm.v1.worker.gpu.split_dvi.request_state import (
    DVIRequestPhase,
    SplitDVIStateTracker,
)


def test_add_advance_and_cycle_ids():
    tracker = SplitDVIStateTracker()
    tracker.on_request_added("a")
    tracker.on_request_added("b")
    assert tracker.advance_cycle("a") == 1
    assert tracker.advance_cycle("a") == 2
    assert tracker.advance_cycle("b") == 1
    assert tracker.cycle_ids_for(["a", "b"]) == [2, 1]
    state = tracker.states["a"]
    assert state.phase is DVIRequestPhase.BLOCK_SENT
    assert state.awaiting_result


def test_validate_cycles_pass_and_fail():
    tracker = SplitDVIStateTracker()
    tracker.on_request_added("a")
    tracker.on_request_added("b")
    tracker.advance_cycle("a")
    tracker.advance_cycle("b")
    tracker.validate_cycles(["a", "b"], [1, 1])
    with pytest.raises(SplitDVIProtocolError, match="cycle desync"):
        tracker.validate_cycles(["a", "b"], [1, 2])
    with pytest.raises(SplitDVIProtocolError, match="no state"):
        tracker.validate_cycles(["a", "c"], [1, 1])


def test_advance_untracked_fails():
    tracker = SplitDVIStateTracker()
    with pytest.raises(SplitDVIProtocolError, match="untracked request"):
        tracker.advance_cycle("ghost")


def test_remove_and_readd_resets_cycle():
    tracker = SplitDVIStateTracker()
    tracker.on_request_added("a")
    tracker.advance_cycle("a")
    tracker.advance_cycle("a")
    tracker.on_request_removed("a")
    tracker.on_request_added("a")
    assert tracker.cycle_ids_for(["a"]) == [0]
    assert tracker.advance_cycle("a") == 1


def test_generation_epoch_from_scheduler():
    tracker = SplitDVIStateTracker()
    tracker.on_request_added("a", generation_id=0)
    tracker.on_request_added("b", generation_id=0)
    assert tracker.generation_ids_for(["a", "b"]) == [0, 0]
    # Re-add after preemption carries the new scheduler-issued epoch.
    tracker.on_request_removed("a")
    tracker.on_request_added("a", generation_id=1)
    assert tracker.generation_ids_for(["a", "b"]) == [1, 0]
    tracker.validate_generations(["a", "b"], [1, 0])
    with pytest.raises(SplitDVIProtocolError, match="generation desync"):
        # Stale packet from the previous lifecycle (epoch 0) is rejected.
        tracker.validate_generations(["a", "b"], [0, 0])


def test_validate_generations_unknown_req():
    tracker = SplitDVIStateTracker()
    with pytest.raises(SplitDVIProtocolError, match="no state"):
        tracker.validate_generations(["ghost"], [0])


def test_mark_result_received():
    tracker = SplitDVIStateTracker()
    tracker.on_request_added("a")
    tracker.advance_cycle("a")
    tracker.mark_result_received("a")
    assert not tracker.states["a"].awaiting_result
    assert tracker.states["a"].phase is DVIRequestPhase.READY
    # Receiving for a non-awaiting request must fail.
    with pytest.raises(SplitDVIProtocolError, match="not awaiting"):
        tracker.validate_awaiting(["a"])


def test_validate_awaiting_unknown_req():
    tracker = SplitDVIStateTracker()
    with pytest.raises(SplitDVIProtocolError):
        tracker.validate_awaiting(["ghost"])
