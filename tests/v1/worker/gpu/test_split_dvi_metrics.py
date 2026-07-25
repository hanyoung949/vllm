# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Stage-DVI metrics denominators and flush dedup."""

from __future__ import annotations

from vllm.v1.worker.gpu.split_dvi.metrics import DVIMetrics


def test_per_request_denominators_exclude_plain_rows():
    m = DVIMetrics(stage="stage_2")
    # One batch cycle: two DVI rows (draft=4) + one plain 0-draft row.
    m.record_verification([4, 4, 0], [1, 0, 0], [2, 1, 1])
    assert m.cycles == 1
    assert m.verified_requests == 2
    assert m.dvi_sampled_tokens == 3
    # ac==0 counts only draft>0 rows; the plain row must not be a "reject".
    assert m.dvi_first_token_rejects == 1
    assert m.mean_advancement == 1.5
    assert m.first_token_reject_rate == 0.5
    # Batch-level companion counts all committed rows (incl. plain).
    assert m.sampled_per_cycle == 4.0
    # Histogram covers DVI rows only — the plain row's "1" must not leak in.
    assert m.sampled_hist == {1: 1, 2: 1}


def test_zero_acceptance_rates_are_per_request():
    # Untrained draft: every DVI row rejects at the first token -> per-request
    # advancement 1.0 and reject rate 1.0 (never > 1 regardless of batch size).
    m = DVIMetrics(stage="stage_2")
    for _ in range(3):
        m.record_verification([4, 4], [0, 0], [1, 1])
    assert m.mean_advancement == 1.0
    assert m.first_token_reject_rate == 1.0
    assert m.sampled_per_cycle == 2.0


def test_flush_emits_again_after_fallback_only_mutation(monkeypatch):
    calls = []

    class _Logger:
        def info(self, *args):
            calls.append(args)

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.split_dvi.metrics.logger", _Logger()
    )
    m = DVIMetrics(stage="stage_2", log_interval_cycles=1000)
    m.record_verification([4], [0], [1])
    m.flush()
    assert len(calls) == 1
    # A fallback-only mutation (no new cycle) must not be suppressed by the
    # shutdown-time flush.
    m.record_fallback()
    m.flush()
    assert len(calls) == 2
    # Nothing new: suppressed again.
    m.flush()
    assert len(calls) == 2


def test_flush_emits_again_after_timing_only_mutation(monkeypatch):
    calls = []

    class _Logger:
        def info(self, *args):
            calls.append(args)

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.split_dvi.metrics.logger", _Logger()
    )
    m = DVIMetrics(stage="stage_2", log_interval_cycles=1000)
    m.record_verification([4], [0], [1])
    m.flush()
    assert len(calls) == 1
    # A timing-only mutation (cycle_end after the periodic flush) changes the
    # log line and must not be suppressed at shutdown.
    m.cycle_start()
    m.cycle_end()
    m.flush()
    assert len(calls) == 2


def test_flush_deduplicates(monkeypatch):
    calls = []

    class _Logger:
        def info(self, *args):
            calls.append(args)

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.split_dvi.metrics.logger", _Logger()
    )
    m = DVIMetrics(stage="stage_2", log_interval_cycles=1000)
    m.record_verification([4], [0], [1])
    m.flush()
    m.flush()  # same cumulative snapshot: must not re-emit
    m.record_verification([4], [0], [1])
    m.flush()  # new snapshot: emits
    assert len(calls) == 2
    fmt1, args1 = calls[0][0], calls[0][1:]
    d = dict(zip(("stage", "cycles", "dvi_reqs"), args1[:3]))
    assert d["cycles"] == 1 and d["dvi_reqs"] == 1
    assert calls[1][2] == 2  # cycles on the second emission
