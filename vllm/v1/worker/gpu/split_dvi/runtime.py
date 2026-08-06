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
import hashlib
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


def check_sampling_params_supported(
    sp: "SamplingParams", mode: str = "greedy"
) -> str | None:
    """Return a violation reason for the selected DVI sampling mode.

    The verifier bypasses the sampler entirely, so anything that would alter
    the distribution or output shape must be rejected up front to keep DVI
    output identical to the baseline greedy path.  This includes structured
    output / allowed-token / min-token constraints, which the verifier's raw
    argmax cannot honor.
    """
    if sp is None:
        return "missing sampling params"
    temperature = getattr(sp, "temperature", 1.0)
    if mode == "greedy":
        if temperature != 0.0:
            return f"temperature={temperature} != 0"
        if getattr(sp, "top_p", 1.0) != 1.0:
            return "top_p != 1.0"
        top_k = getattr(sp, "top_k", 0)
        if top_k not in (0, -1, 1):
            return f"top_k={top_k} unsupported"
    elif mode == "stochastic":
        if temperature <= 0.0:
            return f"temperature={temperature} must be > 0"
        if getattr(sp, "min_p", 0.0) != 0.0:
            return "min_p != 0"
    else:
        return f"unknown DVI mode {mode!r}"
    if getattr(sp, "n", 1) != 1:
        return "n != 1"
    if getattr(sp, "use_beam_search", False):
        return "beam search"
    if mode == "greedy" and getattr(sp, "logprobs", None) is not None:
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


@dataclass(frozen=True)
class DVIStochasticSampling:
    temperature: float
    seed: int

    def proposal_seed(self, generation_id: int) -> int:
        value = (
            (self.seed & ((1 << 64) - 1))
            ^ 0xD1B54A32D192ED03
            ^ ((generation_id * 0x9E3779B97F4A7C15) & ((1 << 64) - 1))
        )
        return value - (1 << 64) if value >= (1 << 63) else value


def _default_request_seed(req_id: str) -> int:
    digest = hashlib.sha256(req_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=True)


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
        self._stochastic_sampling: dict[str, DVIStochasticSampling] = {}

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
            violation = check_sampling_params_supported(
                sampling_params, self.config.mode
            )
            if violation is not None:
                # v1 fail-fast: unsupported sampling would silently diverge
                # from the greedy baseline, so reject at admission.  This is
                # a request-level config error, NOT a protocol violation, so
                # it stays a plain ValueError (engine-fatal is reserved for
                # cross-stage desync via SplitDVIProtocolError).
                raise ValueError(
                    f"SplitDVI unsupported sampling for request {req_id!r}: "
                    f"{violation}. mode={self.config.mode!r}."
                )
            if self.config.mode == "stochastic":
                seed = getattr(sampling_params, "seed", None)
                self._stochastic_sampling[req_id] = DVIStochasticSampling(
                    temperature=float(sampling_params.temperature),
                    seed=_default_request_seed(req_id) if seed is None else int(seed),
                )
        self.tracker.on_request_added(req_id, generation_id)

    def on_request_removed(self, req_id: str) -> None:
        self._stochastic_sampling.pop(req_id, None)
        self.tracker.on_request_removed(req_id)

    def stochastic_sampling_for(
        self, req_ids: list[str]
    ) -> list[DVIStochasticSampling]:
        try:
            return [self._stochastic_sampling[req_id] for req_id in req_ids]
        except KeyError as exc:
            raise SplitDVIProtocolError(
                f"Missing stochastic sampling state for request {exc.args[0]!r}"
            ) from exc

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
        generation_ids = self.tracker.generation_ids_for(
            list(input_batch.req_ids)
        )
        if self.metrics is not None:
            # cycle wall clock starts here (host monotonic, stage_0 only);
            # it ends when the answering token packet arrives back at this
            # rank (see validate_token_packet).  Keyed by (req_ids,
            # cycle_ids) so concurrent blocks can't overwrite each other.
            self.metrics.cycle_start(list(input_batch.req_ids), cycle_ids)
        block = self._generator.generate(
            input_batch, k, cycle_ids, generation_ids
        )
        self._outgoing_metadata = {
            "packet_kind": "dvi_block",
            "is_fallback": False,
            "cycle_ids": block.cycle_ids,
            "draft_token_ids": block.draft_token_ids,
            "draft_lengths": block.draft_lengths,
            "generation_ids": generation_ids,
            "draft_positions": block.draft_positions,
            "policy_version": self.policy_version,
            "draft_version": self.draft_version,
            "sampling_mode": self.config.mode,
            "draft_support_offsets": block.draft_support_offsets,
            "draft_support_token_ids": block.draft_support_token_ids,
            "draft_support_logits": block.draft_support_logits,
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
            "is_fallback": True,
            "cycle_ids": self.tracker.cycle_ids_for(list(input_batch.req_ids)),
            "draft_token_ids": [0] * sum(draft_lengths),
            "draft_lengths": draft_lengths,
            "generation_ids": self.tracker.generation_ids_for(
                list(input_batch.req_ids)
            ),
            "draft_positions": draft_positions,
            "policy_version": self.policy_version,
            "draft_version": self.draft_version,
            "sampling_mode": self.config.mode,
            "draft_support_offsets": (
                list(range(sum(draft_lengths) + 1))
                if self.config.mode == "stochastic"
                else None
            ),
            "draft_support_token_ids": (
                [0] * sum(draft_lengths)
                if self.config.mode == "stochastic"
                else None
            ),
            "draft_support_logits": (
                [0.0] * sum(draft_lengths)
                if self.config.mode == "stochastic"
                else None
            ),
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
        packet_mode = dvi.get("sampling_mode") or "greedy"
        if packet_mode != self.config.mode:
            raise SplitDVIProtocolError(
                f"DVI sampling mode {packet_mode!r} != local mode "
                f"{self.config.mode!r}"
            )
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
        if self.config.mode == "stochastic":
            support_offsets = dvi.get("draft_support_offsets")
            support_ids = dvi.get("draft_support_token_ids")
            support_logits = dvi.get("draft_support_logits")
            if (
                support_offsets is None
                or support_ids is None
                or support_logits is None
            ):
                raise SplitDVIProtocolError(
                    "Stochastic DVI metadata missing draft support fields"
                )
            if len(support_offsets) != len(draft_token_ids) + 1:
                raise SplitDVIProtocolError(
                    "Stochastic DVI support offset count does not match "
                    "draft token count"
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
    def peek_outgoing_metadata(self) -> dict[str, Any] | None:
        """Inspect the next outgoing packet metadata without consuming it."""
        if self.is_first_stage:
            return self._outgoing_metadata
        return self._incoming_metadata

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
        if self.config.mode == "stochastic":
            return self._verify_stochastic_block(
                logits, input_batch, metadata
            )

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

    def _capture_stochastic_stage2_first_rows(
        self,
        processed_logits: torch.Tensor,
        input_batch: "InputBatch",
        metadata: dict[str, Any],
        proposal_counts: list[int],
        cu_num_logits: list[int],
    ) -> None:
        """Capture the committed-path first target row of each DVI block."""
        hook = self.runner.dvi_telemetry_hook
        if hook is None or not hook.enabled or hook.evaluation_mode or metadata.get("is_fallback", False):
            return

        draft_positions = metadata["draft_positions"]
        draft_lengths = metadata["draft_lengths"]
        prefill_len = input_batch.prefill_len_np
        draft_offset = 0
        for req_idx, (draft_len, proposal_count) in enumerate(
            zip(draft_lengths, proposal_counts)
        ):
            if proposal_count <= 0:
                draft_offset += draft_len
                continue
            response_pos = (
                int(draft_positions[draft_offset])
                + 1
                - int(prefill_len[req_idx])
            )
            draft_offset += draft_len
            if response_pos < 0:
                continue

            key = hook.make_record_key(
                input_batch.req_ids[req_idx], response_pos
            )
            ticket = hook.reserve(key)
            if ticket is None:
                continue

            row = processed_logits[cu_num_logits[req_idx]].to(torch.float32)
            top_k = min(hook.dvi_top_k, row.shape[-1])
            topk_values, topk_ids = torch.topk(row, k=top_k, dim=-1)
            log_probs = torch.log_softmax(row, dim=-1)
            topk_logprobs = log_probs.gather(0, topk_ids)
            valid_count = torch.isfinite(topk_values).sum(dtype=torch.int32)
            residual_mass = (
                1.0 - topk_logprobs.exp().sum()
            ).clamp(0.0, 1.0)
            top1_id = row.argmax(dim=-1)
            ok = hook.submit_device_topk(
                ticket,
                topk_ids=topk_ids.to(torch.int32),
                topk_logprobs=topk_logprobs,
                residual_mass=residual_mass,
                top1_id=top1_id.to(torch.int32),
                valid_count=valid_count,
            )
            if not ok:
                hook.cancel(ticket)

    def _capture_stochastic_evaluation_rows(
        self,
        processed_logits: torch.Tensor,
        input_batch: "InputBatch",
        metadata: dict[str, Any],
        proposal_counts: list[int],
        cu_num_logits: list[int],
        sampled: torch.Tensor,
        raw_num_sampled: torch.Tensor,
        expected_first_acceptances: list[float | None],
        first_coverages: list[float | None],
        request_seeds: list[int],
    ) -> None:
        """Capture every target row and realized cycle decision for v2."""
        hook = self.runner.dvi_telemetry_hook
        if (
            hook is None
            or not hook.enabled
            or not hook.evaluation_mode
            or metadata.get("is_fallback", False)
        ):
            return

        draft_positions = metadata["draft_positions"]
        draft_lengths = metadata["draft_lengths"]
        support_offsets = metadata["draft_support_offsets"]
        support_ids = metadata["draft_support_token_ids"]
        support_logits = metadata["draft_support_logits"]
        prefill_len = input_batch.prefill_len_np
        records: list[dict[str, Any]] = []
        draft_offset = 0
        cu = list(cu_num_logits)
        for req_idx, (draft_len, proposal_count) in enumerate(
            zip(draft_lengths, proposal_counts)
        ):
            row_count = cu[req_idx + 1] - cu[req_idx]
            sample_count = int(raw_num_sampled[req_idx])
            committed = sampled[req_idx, :sample_count].tolist()
            accepted_count = int(raw_num_sampled[req_idx]) - 1
            accepted_count = max(0, min(accepted_count, proposal_count))
            expected_row_count = proposal_count + 1
            if row_count != expected_row_count:
                raise SplitDVIProtocolError(
                    "evaluation capture row count does not match the runtime "
                    f"proposal/bonus contract: rows={row_count}, "
                    f"expected={expected_row_count}, proposals={proposal_count}"
                )
            correction_token_id = (
                int(committed[-1])
                if accepted_count < proposal_count and committed
                else None
            )
            for row_index in range(expected_row_count):
                row = processed_logits[cu[req_idx] + row_index].to(torch.float32)
                top_k = min(hook.dvi_top_k, row.shape[-1])
                topk_values, topk_ids = torch.topk(row, k=top_k, dim=-1)
                log_probs = torch.log_softmax(row, dim=-1)
                topk_logprobs = log_probs.gather(0, topk_ids)
                valid_count = int(torch.isfinite(topk_values).sum())
                topk_ids_list = topk_ids[:valid_count].tolist()
                topk_logprobs_list = topk_logprobs[:valid_count].tolist()
                residual_mass = max(
                    0.0, 1.0 - sum(float(value.exp()) for value in topk_logprobs[:valid_count])
                )
                support_row_ids: list[int] = []
                if row_index < proposal_count:
                    support_start = int(
                        support_offsets[draft_offset + row_index]
                    )
                    support_end = int(
                        support_offsets[draft_offset + row_index + 1]
                    )
                    support_row_ids = [
                        int(value)
                        for value in support_ids[support_start:support_end]
                    ]
                teacher_support_probs = (
                    torch.softmax(row, dim=-1)[
                        torch.as_tensor(
                            support_row_ids,
                            dtype=torch.long,
                            device=row.device,
                        )
                    ].tolist()
                    if support_row_ids
                    else []
                )
                if row_index < accepted_count:
                    accepted: bool | None = True
                elif row_index == accepted_count and row_index < proposal_count:
                    accepted = False
                else:
                    accepted = None
                record: dict[str, Any] = {
                    "request_id": input_batch.req_ids[req_idx],
                    "cycle_id": int(metadata["cycle_ids"][req_idx]),
                    "generation_id": int(metadata["generation_ids"][req_idx]),
                    "row_index": row_index,
                    "absolute_position": (
                        int(draft_positions[draft_offset + row_index])
                        + 1
                        - int(prefill_len[req_idx])
                    ),
                    "rng_position": int(
                        draft_positions[draft_offset + row_index]
                    ),
                    "request_seed": int(request_seeds[req_idx]),
                    "row_kind": (
                        "proposal" if row_index < proposal_count else "bonus"
                    ),
                    "num_proposals": proposal_count,
                    "verifier_topk_ids": topk_ids_list,
                    "verifier_topk_logprobs": [
                        float(value) for value in topk_logprobs_list
                    ],
                    "verifier_residual_mass": residual_mass,
                    "teacher_probs_on_draft_support": [
                        float(value) for value in teacher_support_probs
                    ],
                    "accepted": accepted,
                    "accepted_count": accepted_count,
                    "committed_token_ids": (
                        [int(value) for value in committed]
                        if row_index == 0
                        else []
                    ),
                    "correction_token_id": (
                        correction_token_id if row_index == 0 else None
                    ),
                    "terminal_token_id": (
                        int(committed[-1]) if committed and row_index == 0 else None
                    ),
                    "advancement": sample_count if row_index == 0 else 0,
                    "expected_first_acceptance": (
                        expected_first_acceptances[req_idx]
                        if row_index == 0 else None
                    ),
                    "coverage": (
                        first_coverages[req_idx] if row_index == 0 else None
                    ),
                }
                records.append(record)
            draft_offset += int(draft_len)
        hook.capture_evaluation_rows(records)

    def _verify_stochastic_block(
        self,
        logits: torch.Tensor,
        input_batch: "InputBatch",
        metadata: dict[str, Any],
    ) -> DVIBlockResult:
        from vllm.v1.worker.gpu.sample.output import SamplerOutput
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
            rejection_sample,
        )

        assert self.runner.sampler is not None
        assert self.runner.rejection_sampler is not None
        num_reqs = input_batch.num_reqs
        num_speculative_steps = self.runner.num_speculative_steps
        vocab_size = logits.shape[-1]
        device = logits.device

        draft_sampled = torch.zeros(
            logits.shape[0],
            dtype=input_batch.input_ids.dtype,
            device=device,
        )
        draft_logits = torch.full(
            (num_reqs, num_speculative_steps, vocab_size),
            -torch.inf,
            dtype=torch.float32,
            device=device,
        )
        support_offsets = metadata["draft_support_offsets"]
        support_ids = metadata["draft_support_token_ids"]
        support_logits = metadata["draft_support_logits"]
        draft_token_ids = metadata["draft_token_ids"]
        draft_lengths = metadata["draft_lengths"]

        proposal_offset = 0
        proposal_counts: list[int] = []
        cu_num_logits = input_batch.cu_num_logits_np.tolist()
        for req_idx, draft_len in enumerate(draft_lengths):
            row_start = cu_num_logits[req_idx]
            row_end = cu_num_logits[req_idx + 1]
            num_logits = row_end - row_start
            num_proposals = min(
                num_speculative_steps,
                draft_len,
                max(0, num_logits - 1),
            )
            proposal_counts.append(num_proposals)
            for local_pos in range(num_proposals):
                proposal_idx = proposal_offset + local_pos
                proposal = draft_token_ids[proposal_idx]
                support_start = support_offsets[proposal_idx]
                support_end = support_offsets[proposal_idx + 1]
                ids = torch.tensor(
                    support_ids[support_start:support_end],
                    dtype=torch.int64,
                    device=device,
                )
                values = torch.tensor(
                    support_logits[support_start:support_end],
                    dtype=torch.float32,
                    device=device,
                )
                if not bool((ids == proposal).any()):
                    raise SplitDVIProtocolError(
                        f"Stochastic proposal {proposal} is absent from its "
                        "draft support"
                    )
                draft_logits[req_idx, local_pos, ids] = values
                draft_sampled[row_start + local_pos + 1] = proposal
            proposal_offset += draft_len

        pos = torch.tensor(
            metadata["draft_positions"], dtype=torch.int64, device=device
        )
        processed_logits = self.runner.sampler.apply_sampling_params(
            logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
            pos,
            draft_sampled,
            input_batch.expanded_local_pos,
        )
        self._capture_stochastic_stage2_first_rows(
            processed_logits,
            input_batch,
            metadata,
            proposal_counts,
            cu_num_logits,
        )

        expected_first_acceptances: list[float | None] = [None] * num_reqs
        first_coverages: list[float | None] = [None] * num_reqs
        first_distribution_diagnostics: list[
            tuple[bool, bool, float, float] | None
        ] = [None] * num_reqs
        valid_req_indices = [
            req_idx
            for req_idx, count in enumerate(proposal_counts)
            if count > 0
        ]
        if valid_req_indices:
            request_indices = torch.tensor(
                valid_req_indices, dtype=torch.int64, device=device
            )
            first_target_rows = torch.tensor(
                [cu_num_logits[index] for index in valid_req_indices],
                dtype=torch.int64,
                device=device,
            )
            target_first = processed_logits[first_target_rows].to(torch.float32)
            draft_first = draft_logits[request_indices, 0].to(torch.float32)
            target_probs = torch.softmax(target_first, dim=-1)
            draft_probs = torch.softmax(draft_first, dim=-1)
            overlaps = torch.minimum(target_probs, draft_probs).sum(dim=-1)
            target_top1_probs, target_top1_ids = target_probs.max(dim=-1)
            draft_top1_probs, draft_top1_ids = draft_probs.max(dim=-1)
            target_top1_in_support = torch.isfinite(
                draft_first.gather(1, target_top1_ids.unsqueeze(1)).squeeze(1)
            )
            top1_matches = draft_top1_ids == target_top1_ids
            coverage_top_k = min(
                self.runner.dvi_top_k,
                target_first.shape[-1],
            )
            target_support_ids = torch.topk(
                target_first, k=coverage_top_k, dim=-1
            ).indices
            coverages = draft_probs.gather(1, target_support_ids).sum(dim=-1)
            diagnostics = zip(
                target_top1_in_support.tolist(),
                top1_matches.tolist(),
                target_top1_probs.tolist(),
                draft_top1_probs.tolist(),
            )
            for req_idx, overlap, coverage, diagnostic in zip(
                valid_req_indices,
                overlaps.tolist(),
                coverages.tolist(),
                diagnostics,
            ):
                expected_first_acceptances[req_idx] = overlap
                first_coverages[req_idx] = coverage
                first_distribution_diagnostics[req_idx] = diagnostic

        logits_per_req = torch.diff(input_batch.cu_num_logits)
        local_idx_mapping = torch.arange(
            num_reqs, dtype=torch.int32, device=device
        )
        local_expanded_idx_mapping = torch.repeat_interleave(
            local_idx_mapping, logits_per_req
        )
        stochastic_temperature = torch.ones(
            num_reqs, dtype=torch.float32, device=device
        )
        seeds = self.runner.sampler.sampling_states.seeds.gpu[
            input_batch.idx_mapping
        ].contiguous()
        sampled, raw_num_sampled = rejection_sample(
            target_logits=processed_logits,
            draft_logits=draft_logits,
            draft_sampled=draft_sampled,
            cu_num_logits=input_batch.cu_num_logits,
            pos=pos,
            idx_mapping=local_idx_mapping,
            expanded_idx_mapping=local_expanded_idx_mapping,
            expanded_local_pos=input_batch.expanded_local_pos,
            temperature=stochastic_temperature,
            seed=seeds,
            num_speculative_steps=num_speculative_steps,
            use_fp64=self.runner.sampler.use_fp64_gumbel,
        )
        logprob_logits = (
            processed_logits
            if self.runner.sampler.logprobs_mode == "processed_logprobs"
            else logits
        )
        logprobs_tensors = self.runner.rejection_sampler._get_logprobs_tensors(
            input_batch,
            sampled,
            raw_num_sampled,
            logprob_logits,
        )
        accepted_counts = (
            (raw_num_sampled - 1)
            .clamp(min=0, max=num_speculative_steps)
            .tolist()
        )
        raw_num_sampled_list = raw_num_sampled.tolist()
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            raw_num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.runner.req_states.prefill_len.gpu,
        )
        self._capture_stochastic_evaluation_rows(
            processed_logits,
            input_batch,
            metadata,
            proposal_counts,
            cu_num_logits,
            sampled,
            raw_num_sampled,
            expected_first_acceptances,
            first_coverages,
            seeds.tolist(),
        )
        sampler_output = SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
        if self.metrics is not None and not getattr(
            self.runner, "in_warmup", False
        ):
            self.metrics.record_verification(
                draft_lengths,
                accepted_counts,
                raw_num_sampled_list,
                proposal_counts,
                expected_first_acceptances,
                first_distribution_diagnostics,
            )
        return DVIBlockResult(
            sampler_output=sampler_output,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            cycle_ids=metadata["cycle_ids"],
            accepted_counts=accepted_counts,
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

    def metrics_snapshot(self) -> dict | None:
        """Return structured counters for benchmark/evaluation RPC callers."""
        return self.metrics.snapshot() if self.metrics is not None else None
