# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SplitDVIRuntime: per-worker orchestration of the Stage-DVI protocol.

One instance lives on every split worker (all three stages).  It owns:

- the request-level protocol tracker (cycle ids, awaiting flags),
- the step planner classifying each engine step as DRAFT / FALLBACK / NORMAL,
- on stage_0: the draft head + draft block generator,
- on stage_2: the greedy block verifier,
- the DVI packet metadata handoff between model runner and gpu_worker.

Step kinds (all stages classify identically from the same SchedulerOutput):

- DRAFT: pure-decode batch, every request booked with the full k-1 spec
  placeholders.  stage_0 replaces the expanded forward with the boundary +
  draft loop and ships real draft proposals.
- FALLBACK: any spec-booked rows exist but the batch is not pure/full
  (mixed prefill, clipped near max_model_len).  stage_0 runs the normal
  expanded forward with dummy (zero) proposals; correctness is unchanged
  (the verifier always commits target-greedy tokens), just no speedup.
- NORMAL: no spec-booked rows (e.g. pure prefill steps): baseline path.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.config.split_dvi import SplitDVIConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.engine.split_data import SplitDVIProtocolError
from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
from vllm.v1.worker.gpu.split_dvi.block_verifier import (
    SplitDVIGreedyBlockVerifier,
    SplitDVIVerificationResult,
)
from vllm.v1.worker.gpu.split_dvi.metrics import DVIMetrics
from vllm.v1.worker.gpu.split_dvi.request_state import SplitDVIStateTracker

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.sample.output import SamplerOutput
    from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)


class DVIStepKind(enum.Enum):
    DRAFT = "draft"
    FALLBACK = "fallback"
    NORMAL = "normal"


_WARMUP_REQ_PREFIX = "_warmup_"


def _is_warmup_req(req_id: str) -> bool:
    """Internal KV-sizing/cudagraph warmup requests are excluded from DVI
    drafting (they use non-greedy sampler-warmup params and must exercise
    the expanded forward instead)."""
    return req_id.startswith(_WARMUP_REQ_PREFIX)


def check_sampling_params_supported(sp: "SamplingParams") -> str | None:
    """Return a violation reason if the request is not plain greedy, else None.

    The verifier bypasses the sampler entirely, so anything that would alter
    the distribution or output shape must be rejected up front to keep DVI
    output identical to the baseline greedy path.  This includes structured
    output / allowed-token / min-token constraints, which the verifier's raw
    argmax cannot honor.
    """
    if sp is None:
        return "missing sampling params (greedy required)"
    if getattr(sp, "temperature", 1.0) != 0.0:
        return f"temperature={getattr(sp, 'temperature', None)} != 0"
    if getattr(sp, "top_p", 1.0) != 1.0:
        return "top_p != 1.0"
    top_k = getattr(sp, "top_k", 0)
    if top_k not in (0, -1, 1):
        return f"top_k={top_k} unsupported"
    if getattr(sp, "n", 1) != 1:
        return "n != 1"
    if getattr(sp, "use_beam_search", False):
        return "beam search"
    if getattr(sp, "logprobs", None) is not None:
        return "logprobs requested"
    if getattr(sp, "prompt_logprobs", None) is not None:
        return "prompt_logprobs requested"
    for penalty in ("presence_penalty", "frequency_penalty"):
        if getattr(sp, penalty, 0.0) != 0.0:
            return f"{penalty} != 0"
    if getattr(sp, "repetition_penalty", 1.0) != 1.0:
        return "repetition_penalty != 1.0"
    if getattr(sp, "logit_bias", None):
        return "logit_bias set"
    if getattr(sp, "bad_words", None):
        return "bad_words set"
    if getattr(sp, "structured_outputs", None) is not None:
        return "structured output"
    if getattr(sp, "allowed_token_ids", None) is not None:
        return "allowed_token_ids set"
    if getattr(sp, "min_tokens", 0) > 0:
        return "min_tokens > 0 (EOS-suppression masks are not applied by "
        "the DVI verifier)"
    return None


@dataclass
class DVIBlockResult:
    sampler_output: "SamplerOutput"
    num_sampled: torch.Tensor
    num_rejected: torch.Tensor
    cycle_ids: list[int]
    accepted_counts: list[int]
    generation_ids: list[int]
    policy_version: str | None
    draft_version: str | None


class SplitDVIRuntime:
    def __init__(self, runner: "GPUModelRunner", config: SplitDVIConfig):
        self.runner = runner
        self.config = config
        self.is_first_stage = runner.is_first_pp_rank
        self.is_last_stage = runner.is_last_pp_rank
        stage_name = (
            "stage_0"
            if self.is_first_stage
            else ("stage_2" if self.is_last_stage else "stage_1")
        )
        self.stage_name = stage_name
        self.tracker = SplitDVIStateTracker()
        self.metrics = (
            DVIMetrics(stage=stage_name) if config.enable_metrics else None
        )
        self._draft_head = None
        self._generator = None
        self._verifier = (
            SplitDVIGreedyBlockVerifier() if self.is_last_stage else None
        )
        # Metadata produced by this worker for its next outgoing packet.
        self._outgoing_metadata: dict[str, Any] | None = None
        # Metadata from the last incoming packet (representative ranks only).
        self._incoming_metadata: dict[str, Any] | None = None
        self._incoming_is_dvi_block = False
        # Version contracts: v1 runs unversioned (fixed policy and draft head);
        # these become real hashes once hot updates / draft training land.
        self.policy_version: str | None = None
        self.draft_version: str | None = None

    # ------------------------------------------------------------------
    # request lifecycle
    # ------------------------------------------------------------------
    def on_request_added(
        self,
        req_id: str,
        sampling_params: "SamplingParams",
        generation_id: int = 0,
    ) -> None:
        if not _is_warmup_req(req_id):
            violation = check_sampling_params_supported(sampling_params)
            if violation is not None:
                # v1 fail-fast: unsupported sampling would silently diverge
                # from the greedy baseline, so reject at admission.  This is
                # a request-level config error, NOT a protocol violation, so
                # it stays a plain ValueError (engine-fatal is reserved for
                # cross-stage desync via SplitDVIProtocolError).
                raise ValueError(
                    f"SplitDVI unsupported sampling for request {req_id!r}: "
                    f"{violation}. Greedy-only (temperature=0) in v1."
                )
        self.tracker.on_request_added(req_id, generation_id)

    def on_request_removed(self, req_id: str) -> None:
        self.tracker.on_request_removed(req_id)

    # ------------------------------------------------------------------
    # draft head (stage_0): loaded with the model so its memory is accounted
    # in the KV-cache sizing profile; idempotent.
    # ------------------------------------------------------------------
    def on_model_loaded(self) -> None:
        if not self.is_first_stage or self._generator is not None:
            return
        from vllm.v1.worker.gpu.split_dvi.draft_block_generator import (
            SplitDVIDraftBlockGenerator,
        )
        from vllm.v1.worker.gpu.split_dvi.draft_head import (
            load_split_dvi_draft_head,
        )

        self._draft_head = load_split_dvi_draft_head(
            self.config,
            model=self.runner.model,
            model_config=self.runner.model_config,
            device=self.runner.device,
        )
        self._generator = SplitDVIDraftBlockGenerator(
            self.runner, self._draft_head, self.runner.max_num_reqs
        )

    # ------------------------------------------------------------------
    # step planning (called once per execute_model on every stage)
    # ------------------------------------------------------------------
    def plan_step(self, input_batch: "InputBatch") -> DVIStepKind:
        num_draft_per_req = input_batch.num_draft_tokens_per_req
        if input_batch.num_draft_tokens == 0 or num_draft_per_req is None:
            self._incoming_is_dvi_block = False
            return DVIStepKind.NORMAL

        # Advance the cycle for every spec-booked request, identically on all
        # stages.
        for i, req_id in enumerate(input_batch.req_ids):
            if num_draft_per_req[i] > 0:
                self.tracker.advance_cycle(req_id)

        num_spec = self.runner.num_speculative_steps
        all_full_spec = bool((num_draft_per_req == num_spec).all())
        any_prefill = bool(input_batch.is_prefilling_np.any())
        any_warmup = any(_is_warmup_req(r) for r in input_batch.req_ids)
        if (
            self.is_first_stage
            and all_full_spec
            and not any_prefill
            and not any_warmup
        ):
            return DVIStepKind.DRAFT
        return DVIStepKind.FALLBACK

    # ------------------------------------------------------------------
    # stage_0: draft / fallback block production
    # ------------------------------------------------------------------
    def generate_draft_block(
        self, input_batch: "InputBatch"
    ) -> IntermediateTensors:
        """DRAFT step on stage_0: run the draft loop, stash outgoing metadata."""
        assert self.is_first_stage
        self.on_model_loaded()
        k = self.runner.num_speculative_steps + 1
        cycle_ids = self.tracker.cycle_ids_for(list(input_batch.req_ids))
        if self.metrics is not None:
            # cycle wall clock starts here (host monotonic, stage_0 only);
            # it ends when the answering token packet arrives back at this
            # rank (see validate_token_packet).  Keyed by (req_ids,
            # cycle_ids) so concurrent blocks can't overwrite each other.
            self.metrics.cycle_start(list(input_batch.req_ids), cycle_ids)
        block = self._generator.generate(input_batch, k, cycle_ids)
        self._outgoing_metadata = {
            "packet_kind": "dvi_block",
            "cycle_ids": block.cycle_ids,
            "draft_token_ids": block.draft_token_ids,
            "draft_lengths": block.draft_lengths,
            "generation_ids": self.tracker.generation_ids_for(
                list(input_batch.req_ids)
            ),
            "draft_positions": block.draft_positions,
            "policy_version": self.policy_version,
            "draft_version": self.draft_version,
        }
        return block.intermediate_tensors

    def make_fallback_metadata(self, input_batch: "InputBatch") -> None:
        """FALLBACK step on stage_0: normal forward already ran; stash dummy
        (zero) draft metadata so downstream stages stay on the DVI path."""
        assert self.is_first_stage
        num_draft_per_req = input_batch.num_draft_tokens_per_req
        assert num_draft_per_req is not None
        draft_lengths = [
            int(num_draft_per_req[i] + 1) if num_draft_per_req[i] > 0 else 0
            for i in range(input_batch.num_reqs)
        ]
        # Block-row positions for each spec-booked request: c..c+k_r-1; the
        # CPU mirror equals the GPU frontier at the start of the step.
        draft_positions: list[int] = []
        for i in range(input_batch.num_reqs):
            if draft_lengths[i] > 0:
                base = int(input_batch.num_computed_tokens_np[i])
                draft_positions.extend(base + j for j in range(draft_lengths[i]))
        self._outgoing_metadata = {
            "packet_kind": "dvi_block",
            "cycle_ids": self.tracker.cycle_ids_for(list(input_batch.req_ids)),
            "draft_token_ids": [0] * sum(draft_lengths),
            "draft_lengths": draft_lengths,
            "generation_ids": self.tracker.generation_ids_for(
                list(input_batch.req_ids)
            ),
            "draft_positions": draft_positions,
            "policy_version": self.policy_version,
            "draft_version": self.draft_version,
        }
        if self.metrics is not None and not getattr(
            self.runner, "in_warmup", False
        ):
            # Skip synthetic warmup batches: they flow through the real
            # execute path (and the DVI protocol) but are not real cycles.
            self.metrics.record_fallback()

    # ------------------------------------------------------------------
    # stage_1/stage_2: incoming packet metadata
    # ------------------------------------------------------------------
    def capture_incoming_metadata(self, input_batch: "InputBatch") -> None:
        """Read and validate the incoming packet's DVI metadata (called by the
        model runner on non-first stages during execute_model)."""
        self._incoming_metadata = None
        self._incoming_is_dvi_block = False
        pp_group = get_pp_group()
        if not getattr(pp_group, "_is_representative", True):
            # Non-representative TP rank of stage_1: it never sees inter-stage
            # packets (the representative does the ZMQ recv and fans tensors
            # out over TP collectives).  Its own set_tensor_metadata stash for
            # the outgoing send must NOT be misread as an incoming packet.
            # Cycle counters were still advanced identically in plan_step.
            return
        metadata = pp_group.get_tensor_metadata()
        if metadata is None:
            return
        local_has_spec = input_batch.num_draft_tokens > 0
        dvi = metadata.get("dvi")
        packet_req_ids = metadata.get("req_ids")
        if packet_req_ids is not None and packet_req_ids != list(input_batch.req_ids):
            raise SplitDVIProtocolError(
                f"DVI packet req_ids {packet_req_ids!r} != local batch "
                f"{list(input_batch.req_ids)!r}"
            )
        if dvi is None:
            if local_has_spec:
                raise SplitDVIProtocolError(
                    "DVI desync: local batch has spec-booked rows but the "
                    "incoming packet is NORMAL"
                )
            return
        if not local_has_spec:
            raise SplitDVIProtocolError(
                "DVI desync: incoming DVI_BLOCK packet but local batch has no "
                "spec-booked rows"
            )
        cycle_ids = dvi.get("cycle_ids")
        if cycle_ids is None:
            raise SplitDVIProtocolError("DVI block metadata missing cycle_ids")
        self.tracker.validate_cycles(list(input_batch.req_ids), cycle_ids)
        generation_ids = dvi.get("generation_ids")
        if generation_ids is None:
            raise SplitDVIProtocolError("DVI block metadata missing generation_ids")
        self.tracker.validate_generations(
            list(input_batch.req_ids), generation_ids
        )
        if "draft_positions" not in dvi:
            raise SplitDVIProtocolError("DVI block metadata missing draft_positions")
        draft_lengths = dvi.get("draft_lengths")
        draft_token_ids = dvi.get("draft_token_ids")
        num_draft_per_req = input_batch.num_draft_tokens_per_req
        if draft_lengths is None or draft_token_ids is None:
            raise SplitDVIProtocolError("DVI block metadata missing draft fields")
        if len(draft_lengths) != input_batch.num_reqs:
            raise SplitDVIProtocolError(
                f"DVI draft_lengths length {len(draft_lengths)} != num reqs "
                f"{input_batch.num_reqs}"
            )
        # draft_lengths must match the locally scheduled block shape: for a
        # spec-booked request it equals its logit rows (num_draft + 1); for a
        # prefilling request it is 0.
        expected_lengths = [
            int(num_draft_per_req[i] + 1) if num_draft_per_req[i] > 0 else 0
            for i in range(input_batch.num_reqs)
        ]
        if draft_lengths != expected_lengths:
            raise SplitDVIProtocolError(
                f"DVI draft_lengths {draft_lengths} != expected block shape "
                f"{expected_lengths}"
            )
        if sum(draft_lengths) != len(draft_token_ids):
            raise SplitDVIProtocolError(
                f"DVI draft_token_ids length {len(draft_token_ids)} != "
                f"sum(draft_lengths) {sum(draft_lengths)}"
            )
        if self.config.validate_stage_state:
            num_scheduled = metadata.get("num_scheduled_tokens")
            if num_scheduled is not None and (
                sum(num_scheduled) != input_batch.num_tokens
            ):
                raise SplitDVIProtocolError(
                    f"DVI block row count {sum(num_scheduled)} != local "
                    f"batch tokens {input_batch.num_tokens}"
                )
            vocab_size = self.runner.vocab_size
            for token_id in draft_token_ids:
                if not 0 <= token_id < vocab_size:
                    raise SplitDVIProtocolError(
                        f"DVI draft token id {token_id} out of vocab range"
                    )
        self._incoming_metadata = dvi
        self._incoming_is_dvi_block = True

    # ------------------------------------------------------------------
    # gpu_worker metadata handoff
    # ------------------------------------------------------------------
    def outgoing_metadata(self) -> dict[str, Any] | None:
        """Metadata for the worker's next outgoing tensor packet.

        stage_0: produced by generate_draft_block / make_fallback_metadata.
        stage_1: passthrough of the incoming block metadata.
        """
        if self.is_first_stage:
            meta = self._outgoing_metadata
        else:
            meta = self._incoming_metadata
        self._outgoing_metadata = None
        return meta

    @property
    def incoming_is_dvi_block(self) -> bool:
        return self._incoming_is_dvi_block

    # ------------------------------------------------------------------
    # stage_2: block verification
    # ------------------------------------------------------------------
    def verify_block(
        self,
        hidden_states: torch.Tensor,
        input_batch: "InputBatch",
    ) -> DVIBlockResult:
        """Verify the DVI block on stage_2 and package a native-looking
        sampler output plus the DVI token-packet extras."""
        assert self.is_last_stage and self._verifier is not None
        t0 = time.perf_counter_ns()
        try:
            return self._verify_block(hidden_states, input_batch)
        finally:
            if self.metrics is not None:
                self.metrics.verify_wall_ms += (
                    time.perf_counter_ns() - t0
                ) / 1e6

    def _verify_block(
        self,
        hidden_states: torch.Tensor,
        input_batch: "InputBatch",
    ) -> DVIBlockResult:
        if not self._incoming_is_dvi_block or self._incoming_metadata is None:
            raise SplitDVIProtocolError("verify_block called without an incoming DVI block")

        from vllm.v1.worker.gpu.sample.output import SamplerOutput

        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.runner.model.compute_logits(sample_hidden_states)

        metadata = self._incoming_metadata
        result: SplitDVIVerificationResult = self._verifier.verify(
            logits,
            req_ids=list(input_batch.req_ids),
            draft_token_ids=metadata["draft_token_ids"],
            draft_lengths=metadata["draft_lengths"],
            cu_num_logits=input_batch.cu_num_logits_np.tolist(),
        )

        max_sample_len = self.runner.num_speculative_steps + 1
        sampled, num_sampled = result.to_padded_tensors(
            max_sample_len, self.runner.device
        )
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.runner.req_states.prefill_len.gpu,
        )
        sampler_output = SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
        if self.metrics is not None and not getattr(
            self.runner, "in_warmup", False
        ):
            # Skip synthetic warmup batches (see record_fallback above).
            self.metrics.record_verification(
                metadata["draft_lengths"],
                result.accepted_counts,
                result.num_sampled,
            )
        return DVIBlockResult(
            sampler_output=sampler_output,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            cycle_ids=metadata["cycle_ids"],
            accepted_counts=result.accepted_counts,
            generation_ids=metadata["generation_ids"],
            policy_version=metadata.get("policy_version"),
            draft_version=metadata.get("draft_version"),
        )

    # ------------------------------------------------------------------
    # stage_0/stage_1: token packet validation (called by SplitPPHandler)
    # ------------------------------------------------------------------
    def validate_token_packet(self, packet, input_batch=None) -> None:
        """Fail-fast validation of an incoming DVI token packet.

        The awaiting check runs first: a duplicate or out-of-order packet
        that happens to carry matching cycle/generation ids is still rejected
        because the request is not awaiting a result.  The check is scoped to
        requests that actually had a draft block in flight this step (their
        ``num_draft_tokens_per_req > 0``); plain rows piggybacked for
        0-draft requests in a mixed/fallback block are ordinary samples, not
        DVI verifications, and gating them would drop legitimate packets.
        """
        if not packet.is_dvi_block:
            return
        awaiting_ids = list(packet.req_ids)
        if input_batch is not None:
            num_draft = input_batch.num_draft_tokens_per_req
            nd = (
                num_draft.tolist()
                if hasattr(num_draft, "tolist")
                else (list(num_draft) if num_draft is not None else None)
            )
            awaiting_ids = (
                [r for r, n in zip(input_batch.req_ids, nd) if n > 0]
                if nd is not None
                else []
            )
        self.tracker.validate_awaiting(awaiting_ids)
        expected_cycles = self.tracker.cycle_ids_for(list(packet.req_ids))
        expected_gens = self.tracker.generation_ids_for(list(packet.req_ids))
        packet.validate_dvi(
            expected_cycle_ids=expected_cycles,
            expected_generation_ids=expected_gens,
            expected_policy_version=self.policy_version,
            expected_draft_version=self.draft_version,
        )
        # Mark only the spec-booked requests (same scope as the awaiting
        # check and the handler notifier): 0-draft rows in a mixed packet
        # carry ordinary samples, and marking them could clear an unrelated
        # awaiting flag in abnormal states.
        for req_id in awaiting_ids:
            self.tracker.mark_result_received(req_id)
        if self.is_first_stage and self.metrics is not None:
            # The answering token packet just arrived back at stage_0:
            # close the keyed cycle wall clock started in generate_draft_block.
            self.metrics.cycle_end(list(packet.req_ids), list(packet.cycle_ids))

    def note_block_serialized(self, serialize_ms: float, num_bytes: int) -> None:
        """Metrics sink for the split tensor transport (DVI block packets).

        Skipped for synthetic warmup batches, same as record_verification /
        record_fallback.  Scope note: the identity

            block_count == draft_cycles + fallback_cycles

        holds ONLY on stage_0's final flush (stage_0 is the only stage that
        both runs draft loops and emits fallback blocks; stage_1 counts
        forwarded blocks, stage_2 counts verifications, each with their own
        denominators).  Batch runners must apply the assertion to stage_0's
        terminal metrics line only, not to every DVI_METRICS row.
        """
        if self.metrics is not None and not getattr(
            self.runner, "in_warmup", False
        ):
            self.metrics.record_block(num_bytes, serialize_ms)

    def note_dvi_result_received(self, req_ids: list[str]) -> None:
        """Mark DVI results as received without validating them.

        Used by non-representative stage_1 TP ranks: they cannot see the
        token packet (the representative validates it) but still participate
        in the fan-out, so their per-request protocol state must return to
        READY instead of accumulating stale awaiting flags.
        """
        for req_id in req_ids:
            self.tracker.mark_result_received(req_id)

    def flush_metrics(self) -> None:
        """Emit the metrics counters at shutdown.

        Short runs may never reach ``log_interval_cycles``; this guarantees
        the final counters (including the num_sampled histogram) land in the
        log.  ``DVIMetrics.flush`` deduplicates, so a worker never logs the
        same cumulative snapshot twice.
        """
        if self.metrics is not None:
            self.metrics.flush()

    def reset_metrics(self) -> None:
        """Drop all accumulated metrics (benchmark warmup boundary).

        Called by the worker RPC ``reset_dvi_metrics`` after a warmup
        workload so the measured window starts from clean counters.
        """
        if self.metrics is not None:
            self.metrics = DVIMetrics(
                stage=self.metrics.stage,
                log_interval_cycles=self.metrics.log_interval_cycles,
            )
