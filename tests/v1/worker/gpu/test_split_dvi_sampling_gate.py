# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Stage-DVI admission gate and protocol-state closure.

Covers:
- sampling-param gate (greedy-only), including structured output /
  allowed_token_ids / min_tokens rejection (the verifier cannot honor them);
- KV-transfer-connector fail-fast (draft sub-forwards bypass the connector);
- awaiting-result invariant in validate_token_packet and the non-
  representative result notifier.
"""

from __future__ import annotations

import pytest

from vllm.config.split_dvi import SplitDVIConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.split_data import (
    SplitDVIProtocolError,
    SplitPacketKind,
    SplitTokenPacket,
)
from vllm.v1.worker.gpu.split_dvi.request_state import SplitDVIStateTracker
from vllm.v1.worker.gpu.split_dvi.runtime import (
    SplitDVIRuntime,
    check_sampling_params_supported,
)

MODEL = "/workspace/models/Qwen3-4B"


# ----------------------------------------------------------------------
# sampling gate
# ----------------------------------------------------------------------


def test_plain_greedy_allowed():
    assert check_sampling_params_supported(SamplingParams(temperature=0.0)) is None
    assert (
        check_sampling_params_supported(
            SamplingParams(temperature=0.0, top_k=1, max_tokens=16)
        )
        is None
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(temperature=0.7), "temperature"),
        (dict(logprobs=1), "logprobs"),
        (dict(prompt_logprobs=1), "prompt_logprobs"),
        (dict(presence_penalty=0.1), "presence_penalty"),
        (dict(frequency_penalty=0.1), "frequency_penalty"),
        (dict(repetition_penalty=1.1), "repetition_penalty"),
        (dict(logit_bias={1: 1.0}), "logit_bias"),
        (dict(bad_words=["foo"]), "bad_words"),
    ],
)
def test_distribution_altering_rejected(kwargs, match):
    params = {"temperature": 0.0}
    params.update(kwargs)
    sp = SamplingParams(**params)
    reason = check_sampling_params_supported(sp)
    assert reason is not None and match in reason


def test_top_p_top_k_normalized_to_noop_under_greedy():
    # SamplingParams.__post_init__ normalizes top_p/top_k to neutral when
    # temperature == 0, so they can never reach the verifier as active
    # constraints; accepting them matches the baseline greedy path exactly.
    sp = SamplingParams(temperature=0.0, top_p=0.9, top_k=8)
    assert sp.top_p == 1.0 and sp.top_k == 0
    assert check_sampling_params_supported(sp) is None


def test_n_greater_than_1_rejected():
    # Greedy construction itself refuses n > 1 ...
    with pytest.raises(ValueError, match="n must be 1"):
        SamplingParams(temperature=0.0, n=2)
    # ... and the gate rejects any surviving n != 1 as well.
    sp = SamplingParams(temperature=0.7, n=2)
    assert check_sampling_params_supported(sp) is not None


def test_structured_output_rejected():
    from vllm.sampling_params import StructuredOutputsParams

    sp = SamplingParams(
        temperature=0.0, structured_outputs=StructuredOutputsParams(regex="[0-9]+")
    )
    reason = check_sampling_params_supported(sp)
    assert reason is not None and "structured output" in reason


def test_allowed_token_ids_rejected():
    sp = SamplingParams(temperature=0.0, allowed_token_ids=[1, 2, 3])
    reason = check_sampling_params_supported(sp)
    assert reason is not None and "allowed_token_ids" in reason


def test_min_tokens_rejected():
    sp = SamplingParams(temperature=0.0, min_tokens=4, max_tokens=16)
    reason = check_sampling_params_supported(sp)
    assert reason is not None and "min_tokens" in reason


def test_plain_stochastic_grpo_sampling_allowed():
    sp = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        top_k=32,
        logprobs=1,
    )
    assert check_sampling_params_supported(sp, mode="stochastic") is None


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(temperature=0.0), "temperature"),
        (dict(temperature=0.7, min_p=0.1), "min_p"),
        (dict(temperature=0.7, presence_penalty=0.1), "presence_penalty"),
    ],
)
def test_stochastic_mode_stays_fail_closed(kwargs, match):
    reason = check_sampling_params_supported(
        SamplingParams(**kwargs), mode="stochastic"
    )
    assert reason is not None and match in reason


def test_stochastic_config_requires_bonus_and_bounded_draft_support():
    with pytest.raises(ValueError, match="bonus_token"):
        SplitDVIConfig(enabled=True, mode="stochastic")
    with pytest.raises(ValueError, match="draft_top_k"):
        SplitDVIConfig(
            enabled=True,
            mode="stochastic",
            bonus_token=True,
            draft_top_k=0,
        )
    cfg = SplitDVIConfig(
        enabled=True,
        mode="stochastic",
        bonus_token=True,
        draft_top_k=16,
    )
    assert cfg.num_scheduler_spec_tokens == cfg.draft_length - 1


# ----------------------------------------------------------------------
# KV connector fail-fast
# ----------------------------------------------------------------------


def test_kv_connector_rejected():
    from vllm.engine.arg_utils import EngineArgs

    with pytest.raises(ValueError, match="KV transfer connectors"):
        EngineArgs(
            model=MODEL,
            enforce_eager=True,
            max_model_len=64,
            gpu_memory_utilization=0.3,
            tensor_parallel_size=1,
            pipeline_parallel_size=3,
            enable_layerwise_split=True,
            split_stage_0_size=2,
            split_stage_2_size=2,
            split_stage_1_tensor_parallel_size=1,
            split_dvi_config={"enabled": True, "draft_length": 4},
            kv_transfer_config={"kv_connector": "P2pNcclConnector", "kv_role": "kv_both"},
        ).create_engine_config()


def test_config_without_kv_connector_ok():
    from vllm.engine.arg_utils import EngineArgs

    cfg = EngineArgs(
        model=MODEL,
        enforce_eager=True,
        max_model_len=64,
        gpu_memory_utilization=0.3,
        tensor_parallel_size=1,
        pipeline_parallel_size=3,
        enable_layerwise_split=True,
        split_stage_0_size=2,
        split_stage_2_size=2,
        split_stage_1_tensor_parallel_size=1,
        split_dvi_config={"enabled": True, "draft_length": 4},
    ).create_engine_config()
    assert cfg.split_dvi_config.enabled


def test_hybrid_linear_attention_rejected():
    from unittest import mock

    with pytest.raises(ValueError, match="hybrid/linear-attention"):
        SplitDVIConfig(enabled=True).validate_against_vllm_config(
            mock.Mock(
                parallel_config=mock.Mock(
                    enable_layerwise_split=True,
                    pipeline_parallel_size=3,
                ),
                speculative_config=None,
                model_config=mock.Mock(
                    is_moe=False,
                    has_inner_state=False,
                    is_attention_free=False,
                    hf_config=mock.Mock(
                        text_config=None,
                        layer_types=["linear_attention", "full_attention"],
                    ),
                ),
            )
        )


class TestRecurrentStateGate:
    """Focused checks of _has_recurrent_state: capability flags first,
    layer_types as fallback, sliding-window attention untouched."""

    def test_has_inner_state_rejects(self):
        from unittest import mock

        from vllm.config.split_dvi import _has_recurrent_state

        mc = mock.Mock(has_inner_state=True, is_attention_free=False)
        assert _has_recurrent_state(mc) is True

    def test_is_attention_free_rejects(self):
        from unittest import mock

        from vllm.config.split_dvi import _has_recurrent_state

        mc = mock.Mock(has_inner_state=False, is_attention_free=True)
        assert _has_recurrent_state(mc) is True

    def test_layer_types_fallback_rejects(self):
        from unittest import mock

        from vllm.config.split_dvi import _has_recurrent_state

        hf = mock.Mock(text_config=None, layer_types=["linear_attention", "full_attention"])
        mc = mock.Mock(
            has_inner_state=False, is_attention_free=False, hf_config=hf
        )
        assert _has_recurrent_state(mc) is True

    def test_sliding_window_allowed(self):
        from unittest import mock

        from vllm.config.split_dvi import _has_recurrent_state

        hf = mock.Mock(text_config=None, layer_types=["sliding_attention", "full_attention"])
        mc = mock.Mock(
            has_inner_state=False, is_attention_free=False, hf_config=hf
        )
        assert _has_recurrent_state(mc) is False


# ----------------------------------------------------------------------
# awaiting-result invariant
# ----------------------------------------------------------------------


def _runtime_for_tracker(tracker: SplitDVIStateTracker) -> SplitDVIRuntime:
    runtime = SplitDVIRuntime.__new__(SplitDVIRuntime)
    runtime.tracker = tracker
    runtime.policy_version = None
    runtime.draft_version = None
    runtime.is_first_stage = False
    runtime.metrics = None
    return runtime


def _packet(cycle: int = 1, gen: int = 0) -> SplitTokenPacket:
    return SplitTokenPacket(
        req_ids=["r0"],
        sampled_token_ids=[[10, 11]],
        num_sampled=[2],
        num_rejected=[2],
        packet_kind=SplitPacketKind.DVI_BLOCK.value,
        cycle_ids=[cycle],
        generation_ids=[gen],
    )


def test_duplicate_result_rejected_by_awaiting():
    tracker = SplitDVIStateTracker()
    tracker.on_request_added("r0", generation_id=0)
    tracker.advance_cycle("r0")
    runtime = _runtime_for_tracker(tracker)

    packet = _packet(cycle=1, gen=0)
    runtime.validate_token_packet(packet)  # first result: OK, marks READY
    with pytest.raises(SplitDVIProtocolError, match="not awaiting"):
        # Same packet arriving twice (duplicate): cycle/generation match but
        # the request is not awaiting a result anymore.
        runtime.validate_token_packet(packet)


def test_result_for_pending_cycle_ok():
    tracker = SplitDVIStateTracker()
    tracker.on_request_added("r0", generation_id=0)
    tracker.advance_cycle("r0")
    runtime = _runtime_for_tracker(tracker)
    runtime.validate_token_packet(_packet(cycle=1, gen=0))
    tracker.advance_cycle("r0")
    runtime.validate_token_packet(_packet(cycle=2, gen=0))


def test_result_for_wrong_cycle_rejected():
    tracker = SplitDVIStateTracker()
    tracker.on_request_added("r0", generation_id=0)
    tracker.advance_cycle("r0")
    runtime = _runtime_for_tracker(tracker)
    with pytest.raises(SplitDVIProtocolError, match="cycle mismatch"):
        runtime.validate_token_packet(_packet(cycle=7, gen=0))


def test_notifier_marks_without_validation():
    tracker = SplitDVIStateTracker()
    tracker.on_request_added("r0", generation_id=0)
    tracker.advance_cycle("r0")
    runtime = _runtime_for_tracker(tracker)
    # Non-representative rank: no packet visibility, but the flag must still
    # return to READY after the fan-out.
    runtime.note_dvi_result_received(["r0"])
    assert not tracker.states["r0"].awaiting_result


def test_mixed_packet_marks_only_spec_booked_requests():
    from unittest import mock

    tracker = SplitDVIStateTracker()
    tracker.on_request_added("r0", generation_id=0)
    tracker.on_request_added("r1", generation_id=0)
    tracker.advance_cycle("r0")  # r0 is spec-booked this step, r1 is not
    runtime = _runtime_for_tracker(tracker)

    packet = SplitTokenPacket(
        req_ids=["r0", "r1"],
        sampled_token_ids=[[10], [11]],
        num_sampled=[1, 1],
        num_rejected=[3, 0],  # r0: first-token reject; r1: plain 0-draft row
        packet_kind=SplitPacketKind.DVI_BLOCK.value,
        cycle_ids=[1, 0],
        generation_ids=[0, 0],
    )
    input_batch = mock.Mock()
    input_batch.req_ids = ["r0", "r1"]
    input_batch.num_draft_tokens_per_req = [4, 0]

    # Simulate an abnormal in-flight flag on the plain-row request: the
    # validator must NOT clear it — r1's protocol state is none of this
    # packet's business.
    tracker.states["r1"].awaiting_result = True

    runtime.validate_token_packet(packet, input_batch)
    assert not tracker.states["r0"].awaiting_result  # spec row: marked
    assert tracker.states["r1"].awaiting_result  # plain row: untouched
