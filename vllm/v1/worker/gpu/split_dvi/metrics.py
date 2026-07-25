# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-DVI per-cycle metrics.

Kept deliberately lightweight: CPU-side counters plus wall-time
accumulators, flushed to the log every ``log_interval_cycles`` cycles and
once more at engine shutdown (deduplicated — a flush never repeats the
same cumulative snapshot twice).

Denominators matter:

- ``cycles`` counts *batch* verification events (one per DVI step).
- ``verified_requests`` counts *request rows with draft_length > 0* across
  those events.  Per-request headline numbers use it as the denominator:
  ``mean_advancement`` is E[num_sampled] per DVI request-cycle (the spec-
  decode beta) and ``first_token_reject_rate`` is a true rate in [0, 1].
- ``sampled_per_cycle`` is the batch-level companion (all committed rows,
  including plain 0-draft rows in mixed/fallback blocks, per scheduler
  cycle) and is NOT a per-request advancement.

Plain rows (draft_length == 0) are excluded from the per-request counters:
they commit an ordinary greedy token, not a DVI verification.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import ClassVar

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class DVIMetrics:
    stage: str
    log_interval_cycles: int = 50

    cycles: int = 0
    fallback_cycles: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    sampled_tokens: int = 0
    # Multiset of num_sampled values (1..k) over draft>0 rows only, for
    # acceptance-path coverage analysis (the perfect-draft oracle checks
    # 1/3/4 appear; plain 0-draft rows are ordinary greedy commits and are
    # excluded so a "1" unambiguously means a first-token reject).
    sampled_hist: dict[int, int] = field(default_factory=dict)
    # Per-request (draft>0 rows only) counters for meaningful rates.
    verified_requests: int = 0
    dvi_sampled_tokens: int = 0
    dvi_first_token_rejects: int = 0

    # Host-side wall times (perf_counter); only fields measured with real
    # CUDA events may carry "_cuda_ms" (draft_cuda_ms below).
    draft_wall_ms: float = 0.0
    verify_wall_ms: float = 0.0
    block_serialize_ms: float = 0.0
    block_bytes: int = 0
    cycle_wall_ms: float = 0.0
    # Number of unsettled draft CUDA event pairs dropped by the hard cap;
    # nonzero means draft_cuda_ms is a lower bound.
    draft_events_dropped: int = 0
    # Denominators for per-event averages: wall-clock cycles completed on
    # this stage, DVI block packets serialized here, and draft loops run
    # here (``cycles`` only counts stage_2 verifications, so it cannot
    # serve for these).
    wall_cycles: int = 0
    block_count: int = 0
    draft_cycles: int = 0
    # stage_0 draft-loop GPU time from CUDA events; pairs are settled
    # eagerly via non-blocking query so the list stays bounded, and only
    # unsettled pairs are read (sync) at flush time.
    _draft_events: list = field(default_factory=list, repr=False)
    _draft_cuda_settled_ms: float = field(default=0.0, repr=False)
    # Wall-clock cycles are keyed by (req_ids, cycle_ids): with several
    # requests' DVI blocks in flight, a single start slot would silently
    # overwrite.  Leftover keys at flush indicate an unanswered block.
    _pending_wall: dict = field(default_factory=dict, repr=False)

    _last_flush_snapshot: tuple | None = field(default=None, repr=False)
    _zero: ClassVar[tuple] = (
        0, 0, 0, 0, 0, 0, 0, 0, (), 0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0, 0.0, 0, 0
    )

    def cycle_start(self, req_ids: list, cycle_ids: list) -> None:
        key = (tuple(req_ids), tuple(cycle_ids))
        self._pending_wall[key] = time.perf_counter_ns()

    def cycle_end(self, req_ids: list, cycle_ids: list) -> None:
        key = (tuple(req_ids), tuple(cycle_ids))
        start = self._pending_wall.pop(key, None)
        if start is not None:
            self.cycle_wall_ms += (time.perf_counter_ns() - start) / 1e6
            self.wall_cycles += 1

    def record_draft_events(self, start, end) -> None:
        """Stash a CUDA event pair and eagerly settle completed ones."""
        self.draft_cycles += 1
        self._draft_events.append((start, end))
        remaining = []
        for s, e in self._draft_events:
            if e.query():
                self._draft_cuda_settled_ms += s.elapsed_time(e)
            else:
                remaining.append((s, e))
        # Hard cap as a safety net (e.g. events orphaned by an abort);
        # dropped counts are reported at flush, never silently lost.
        if len(remaining) > 128:
            self.draft_events_dropped += len(remaining) - 128
            remaining = remaining[-128:]
        self._draft_events = remaining

    def draft_cuda_ms(self) -> float:
        """Total draft-loop GPU time (ms); syncs only on unsettled events."""
        return self._draft_cuda_settled_ms + sum(
            s.elapsed_time(e) for s, e in self._draft_events
        )

    def record_block(self, num_bytes: int, serialize_ms: float) -> None:
        self.block_bytes += num_bytes
        self.block_serialize_ms += serialize_ms
        self.block_count += 1

    def record_verification(
        self,
        draft_lengths: list[int],
        accepted_counts: list[int],
        num_sampled: list[int],
    ) -> None:
        self.cycles += 1
        self.drafted_tokens += sum(draft_lengths)
        self.accepted_tokens += sum(accepted_counts)
        self.sampled_tokens += sum(num_sampled)
        for dl, ns, ac in zip(draft_lengths, num_sampled, accepted_counts):
            if dl > 0:
                # DVI verification rows only: plain 0-draft rows are ordinary
                # greedy commits and must not pollute the acceptance-path
                # histogram (a "1" would otherwise be ambiguous between a
                # first-token reject and a plain row).
                self.verified_requests += 1
                self.dvi_sampled_tokens += ns
                self.sampled_hist[ns] = self.sampled_hist.get(ns, 0) + 1
                if ac == 0:
                    self.dvi_first_token_rejects += 1
        if self.cycles % self.log_interval_cycles == 0:
            self.flush()

    def record_fallback(self) -> None:
        self.fallback_cycles += 1

    @property
    def mean_advancement(self) -> float:
        """E[num_sampled] per DVI request-cycle (draft>0 rows only)."""
        return self.dvi_sampled_tokens / max(self.verified_requests, 1)

    @property
    def mean_acceptance(self) -> float:
        return self.accepted_tokens / max(self.drafted_tokens, 1)

    @property
    def first_token_reject_rate(self) -> float:
        """Fraction of DVI request-cycles rejected at the first draft token."""
        return self.dvi_first_token_rejects / max(self.verified_requests, 1)

    @property
    def sampled_per_cycle(self) -> float:
        """Committed rows (incl. plain 0-draft rows) per scheduler cycle."""
        return self.sampled_tokens / max(self.cycles, 1)

    def flush(self) -> None:
        # Cumulative snapshot; never emit the same snapshot twice.  The
        # snapshot covers EVERY mutable field of the metrics state (no
        # inferred "usually changes together" assumptions), so any mutation
        # at all produces exactly one more emission.
        snapshot = (
            self.cycles,
            self.fallback_cycles,
            self.drafted_tokens,
            self.accepted_tokens,
            self.sampled_tokens,
            self.verified_requests,
            self.dvi_sampled_tokens,
            self.dvi_first_token_rejects,
            tuple(sorted(self.sampled_hist.items())),
            self.wall_cycles,
            self.block_count,
            self.draft_cycles,
            self.draft_wall_ms,
            self.verify_wall_ms,
            self.block_serialize_ms,
            self.block_bytes,
            self.cycle_wall_ms,
            self.draft_events_dropped,
            self._draft_cuda_settled_ms,
            len(self._draft_events),
            len(self._pending_wall),
        )
        if snapshot == self._last_flush_snapshot or snapshot == self._zero:
            return
        self._last_flush_snapshot = snapshot
        logger.info(
            "DVI_METRICS stage=%s cycles=%d dvi_reqs=%d fallbacks=%d "
            "mean_advancement=%.3f first_token_reject_rate=%.3f "
            "mean_acceptance=%.3f sampled_hist=%s sampled_per_cycle=%.3f "
            "cycle_wall_ms_avg=%.2f wall_cycles=%d unclosed_wall=%d "
            "draft_cycles=%d draft_wall_ms_avg=%.2f draft_cuda_ms_avg=%.2f "
            "draft_events_dropped=%d "
            "verify_wall_ms_avg=%.2f block_serialize_ms_avg=%.2f "
            "block_count=%d block_bytes_avg=%.0f",
            self.stage,
            self.cycles,
            self.verified_requests,
            self.fallback_cycles,
            self.mean_advancement,
            self.first_token_reject_rate,
            self.mean_acceptance,
            dict(sorted(self.sampled_hist.items())),
            self.sampled_per_cycle,
            self.cycle_wall_ms / max(self.wall_cycles, 1),
            self.wall_cycles,
            len(self._pending_wall),
            self.draft_cycles,
            self.draft_wall_ms / max(self.draft_cycles, 1),
            self.draft_cuda_ms() / max(self.draft_cycles, 1),
            self.draft_events_dropped,
            self.verify_wall_ms / max(self.cycles, 1),
            self.block_serialize_ms / max(self.block_count, 1),
            self.block_count,
            self.block_bytes / max(self.block_count, 1),
        )
