# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request Stage-DVI protocol state, tracked identically on every stage.

The split pipeline transports are strictly in-order and the V2+PP decode
cadence guarantees that a DVI block produced at scheduler step T is answered
by the token packet of the same step T and applied at step T+pp_size.  The
state tracked here is therefore deliberately small — it exists to *detect*
cross-stage desync (and abort loudly) rather than to drive the protocol:

- ``cycle_id``: incremented on every stage for each request that is
  scheduled with speculative (placeholder) tokens, i.e. contributes more
  than one logit row to the step.  All stages observe the same scheduler
  outputs, so the counters stay in lockstep; packets carry the ids and any
  divergence is a hard error.
- ``awaiting_result``: True between "block sent/forwarded at step T" and
  "result received at step T's sample_tokens".  A result arriving for a
  request not awaiting one (or vice versa) indicates desync.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from vllm.logger import init_logger

logger = init_logger(__name__)


class DVIRequestPhase(str, Enum):
    READY = "ready"
    BLOCK_SENT = "block_sent"
    WAITING_RESULT = "waiting_result"


@dataclass
class SplitDVIRequestState:
    req_id: str
    generation_id: int = 0
    cycle_id: int = 0
    phase: DVIRequestPhase = DVIRequestPhase.READY
    last_draft_length: int = 0
    awaiting_result: bool = False


@dataclass
class SplitDVIStateTracker:
    """Tracks DVI protocol state for all requests on one stage."""

    states: dict[str, SplitDVIRequestState] = field(default_factory=dict)

    def on_request_added(self, req_id: str, generation_id: int = 0) -> None:
        # (Re-)adding a request starts a fresh protocol state keyed by the
        # scheduler-issued generation epoch.  Stale packets from earlier
        # lifecycles are rejected by the generation check.
        self.states[req_id] = SplitDVIRequestState(
            req_id=req_id, generation_id=generation_id
        )

    def on_request_removed(self, req_id: str) -> None:
        self.states.pop(req_id, None)

    def advance_cycle(self, req_id: str) -> int:
        """Increment and return the new DVI cycle id for a spec-scheduled
        request.  Called identically on every stage."""
        state = self.states.get(req_id)
        if state is None:
            raise ValueError(
                f"DVI cycle advanced for untracked request {req_id!r}; the "
                f"request must be registered on addition on every stage"
            )
        state.cycle_id += 1
        state.phase = DVIRequestPhase.BLOCK_SENT
        state.awaiting_result = True
        return state.cycle_id

    def mark_result_received(self, req_id: str) -> None:
        state = self.states.get(req_id)
        if state is not None:
            state.phase = DVIRequestPhase.READY
            state.awaiting_result = False

    def cycle_ids_for(self, req_ids: list[str]) -> list[int]:
        """Current cycle ids for a batch of requests (batch order)."""
        result: list[int] = []
        for req_id in req_ids:
            state = self.states.get(req_id)
            if state is None:
                raise ValueError(
                    f"DVI state tracker has no state for request {req_id!r} "
                    f"(known: {len(self.states)})"
                )
            result.append(state.cycle_id)
        return result

    def generation_ids_for(self, req_ids: list[str]) -> list[int]:
        """Current scheduler-issued generation epochs (batch order)."""
        result: list[int] = []
        for req_id in req_ids:
            state = self.states.get(req_id)
            if state is None:
                raise ValueError(
                    f"DVI state tracker has no state for request {req_id!r}"
                )
            result.append(state.generation_id)
        return result

    def validate_cycles(self, req_ids: list[str], cycle_ids: list[int]) -> None:
        """Fail fast if packet cycle ids diverge from local counters."""
        expected = self.cycle_ids_for(req_ids)
        if cycle_ids != expected:
            raise ValueError(
                f"DVI cycle desync: local cycles {expected}, packet carries "
                f"{cycle_ids} (req_ids={req_ids!r})"
            )

    def validate_generations(
        self, req_ids: list[str], generation_ids: list[int]
    ) -> None:
        """Fail fast if packet generation epochs are stale or diverged."""
        expected = self.generation_ids_for(req_ids)
        if generation_ids != expected:
            raise ValueError(
                f"DVI generation desync: local generations {expected}, "
                f"packet carries {generation_ids} (req_ids={req_ids!r})"
            )

    def validate_awaiting(self, req_ids: list[str]) -> None:
        for req_id in req_ids:
            state = self.states.get(req_id)
            if state is None or not state.awaiting_result:
                raise ValueError(
                    f"DVI result received for request {req_id!r} which is not "
                    f"awaiting a result (phase="
                    f"{state.phase if state else 'missing'})"
                )
