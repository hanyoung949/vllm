# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-fatal protocol errors: type routing and raise-site coverage.

SplitDVIProtocolError is reserved for cross-stage protocol desync; the
executor forwards it from every worker rank so the engine core aborts all
workers instead of silently continuing on corrupted state.  Ordinary
errors (admission, config, runtime bugs) must NOT take this path.
"""

from __future__ import annotations

import pytest

from vllm.v1.engine.split_data import (
    SplitDVIProtocolError,
    SplitPacketKind,
    SplitTokenPacket,
)
from vllm.v1.executor.multiproc_executor import _is_engine_fatal
from vllm.v1.worker.gpu.split_dvi.block_verifier import (
    SplitDVIGreedyBlockVerifier,
)
from vllm.v1.worker.gpu.split_dvi.request_state import SplitDVIStateTracker
from vllm.v1.worker.gpu.split_dvi.runtime import SplitDVIRuntime


def _runtime(tracker: SplitDVIStateTracker) -> SplitDVIRuntime:
    runtime = SplitDVIRuntime.__new__(SplitDVIRuntime)
    runtime.tracker = tracker
    runtime.policy_version = None
    runtime.draft_version = None
    runtime.is_first_stage = False
    runtime.metrics = None
    return runtime


def _packet(cycles=(1,), gens=(0,)) -> SplitTokenPacket:
    n = len(cycles)
    return SplitTokenPacket(
        req_ids=[f"r{i}" for i in range(n)],
        sampled_token_ids=[[10 + i] for i in range(n)],
        num_sampled=[1] * n,
        num_rejected=[3] * n,
        packet_kind=SplitPacketKind.DVI_BLOCK.value,
        cycle_ids=list(cycles),
        generation_ids=list(gens),
    )


class TestTrackerRaisesProtocolError:
    def test_cycle_desync(self):
        t = SplitDVIStateTracker()
        t.on_request_added("r0", 0)
        t.advance_cycle("r0")
        with pytest.raises(SplitDVIProtocolError, match="cycle desync"):
            t.validate_cycles(["r0"], [7])

    def test_generation_desync(self):
        t = SplitDVIStateTracker()
        t.on_request_added("r0", 1)
        with pytest.raises(SplitDVIProtocolError, match="generation desync"):
            t.validate_generations(["r0"], [0])

    def test_not_awaiting(self):
        t = SplitDVIStateTracker()
        t.on_request_added("r0", 0)
        with pytest.raises(SplitDVIProtocolError, match="not awaiting"):
            t.validate_awaiting(["r0"])

    def test_untracked_cycle_advance(self):
        t = SplitDVIStateTracker()
        with pytest.raises(SplitDVIProtocolError, match="untracked"):
            t.advance_cycle("ghost")

    def test_unknown_request_state(self):
        t = SplitDVIStateTracker()
        with pytest.raises(SplitDVIProtocolError, match="no state"):
            t.cycle_ids_for(["ghost"])


class TestPacketValidationRaisesProtocolError:
    def test_req_ids_mismatch(self):
        with pytest.raises(SplitDVIProtocolError):
            _packet().validate_req_ids(["other"])

    def test_cycle_mismatch(self):
        with pytest.raises(SplitDVIProtocolError):
            _packet().validate_dvi(expected_cycle_ids=[9])

    def test_generation_mismatch(self):
        with pytest.raises(SplitDVIProtocolError):
            _packet().validate_dvi(expected_generation_ids=[9])

    def test_version_mismatch(self):
        with pytest.raises(SplitDVIProtocolError):
            _packet().validate_dvi(expected_policy_version="vX")


class TestRuntimeValidatorRaisesProtocolError:
    def test_wrong_cycle(self):
        t = SplitDVIStateTracker()
        t.on_request_added("r0", 0)
        t.advance_cycle("r0")
        with pytest.raises(SplitDVIProtocolError):
            _runtime(t).validate_token_packet(_packet(cycles=(7,)))

    def test_not_awaiting(self):
        t = SplitDVIStateTracker()
        t.on_request_added("r0", 0)
        with pytest.raises(SplitDVIProtocolError):
            _runtime(t).validate_token_packet(_packet(cycles=(0,)))


class TestVerifierRaisesProtocolError:
    def test_draft_lengths_mismatch(self):
        v = SplitDVIGreedyBlockVerifier()
        import torch

        logits = torch.zeros(4, 100)
        with pytest.raises(SplitDVIProtocolError):
            v.verify(
                logits,
                req_ids=["r0", "r1"],
                draft_token_ids=[1, 2, 3],
                draft_lengths=[2, 2],  # sums to 4 != len(draft_token_ids)
                cu_num_logits=[0, 2, 4],
            )


class TestScopeGuards:
    def test_admission_stays_plain_value_error(self):
        from vllm.sampling_params import SamplingParams
        from vllm.v1.worker.gpu.split_dvi.runtime import (
            check_sampling_params_supported,
        )

        # Admission violations return a reason string (request-level config
        # error); they never become engine-fatal protocol errors.
        reason = check_sampling_params_supported(
            SamplingParams(temperature=0.7)
        )
        assert reason is not None

    def test_is_engine_fatal_routing(self):
        assert _is_engine_fatal(SplitDVIProtocolError("x"))
        assert not _is_engine_fatal(ValueError("x"))
        assert not _is_engine_fatal(RuntimeError("x"))
