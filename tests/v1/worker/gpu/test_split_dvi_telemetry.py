from __future__ import annotations

from typing import Any

from vllm.v1.worker.gpu.split_dvi.telemetry import (
    DVISamplingConfig,
    DVIStage2TelemetryProducer,
)


class _EvaluationWriter:
    def __init__(self) -> None:
        self.spool_metadata = {
            "run_id": "run",
            "rollout_id": "rollout",
            "policy_version": "policy",
            "capture_mode": "evaluation",
            "sampling_config": {
                "sample_rate": 1.0,
                "max_per_request": 16,
                "seed": 7,
                "top_k": 16,
            },
        }
        self.stage2_batches: list[list[dict[str, Any]]] = []
        self.finalization_batches: list[list[dict[str, Any]]] = []
        self.requests: list[tuple[str, list[int], list[int]]] = []

    def enqueue_evaluation_stage2(self, session_key, records):
        self.stage2_batches.append(records)
        return True

    def enqueue_evaluation_finalization(self, session_key, records):
        self.finalization_batches.append(records)
        return True

    def enqueue_request(self, request_id, prompt_token_ids, response_token_ids):
        self.requests.append((request_id, prompt_token_ids, response_token_ids))
        return True


def _rows(cycle_id: int, positions: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "request_id": "request-0",
            "cycle_id": cycle_id,
            "num_proposals": 3,
            "absolute_position": position,
            "row_index": row_index,
            "accepted_count": 1,
            "advancement": 2,
            "expected_first_acceptance": 0.25,
            "coverage": 0.5,
        }
        for row_index, position in enumerate(positions)
    ]


def test_evaluation_finalize_writes_all_cycle_statuses() -> None:
    writer = _EvaluationWriter()
    producer = DVIStage2TelemetryProducer(
        writer,
        DVISamplingConfig(
            sample_rate=1.0,
            max_per_request=16,
            seed=7,
            top_k=16,
        ),
    )
    producer.capture_evaluation_rows(_rows(0, [1, 2, 3, 4]))
    producer.capture_evaluation_rows(_rows(1, [1, 2, 3]))
    producer.capture_evaluation_rows(_rows(2, [4, 5]))

    producer.finalize_request(
        producer.session_key,
        "request-0",
        [10],
        [20, 21, 22, 23],
    )

    markers = {
        marker["cycle_id"]: marker
        for marker in writer.finalization_batches[0]
    }
    assert markers[0]["cycle_status"] == "complete"
    assert markers[0]["num_rows_seen"] == 4
    assert markers[0]["accepted_count"] == 1
    assert markers[0]["advancement"] == 2
    assert markers[0]["expected_first_acceptance"] == 0.25
    assert markers[0]["coverage"] == 0.5
    assert markers[1]["cycle_status"] == "aborted"
    assert markers[1]["num_rows_seen"] == 3
    assert "advancement" not in markers[1]
    assert markers[2]["cycle_status"] == "terminal_truncated"
    assert markers[2]["num_rows_seen"] == 2
    assert "advancement" not in markers[2]
    assert all(marker["num_rows_expected"] == 4 for marker in markers.values())
    assert all(marker["finalized"] is True for marker in markers.values())
    assert producer._evaluation_cycles == {}
    assert producer._evaluation_request_ids == set()
    assert writer.requests == [("request-0", [10], [20, 21, 22, 23])]


def test_close_leaves_zero_pending_bytes(tmp_path) -> None:
    """Regression: enqueue/flush byte estimates must be strictly paired.

    Before the fix, evaluation_stage2 was estimated once per enqueue batch
    (single +512 overhead) but flushed with a per-record estimate (+512 per
    record), driving pending_bytes negative after close.
    """
    import torch

    from vllm.v1.worker.gpu.split_dvi.telemetry import DVIPartialSpoolWriter

    writer = DVIPartialSpoolWriter(
        tmp_path / "spool",
        {
            "run_id": "run",
            "rollout_id": "rollout",
            "policy_version": "policy",
            "capture_mode": "evaluation",
            "sampling_config": {
                "sample_rate": 1.0,
                "max_per_request": 16,
                "seed": 7,
                "top_k": 16,
            },
        },
        quota_bytes=1 << 30,
    )
    session_key = ("run", "rollout", "policy")
    stage2_records = 0
    # Multiple multi-record batches: the per-record flush estimate used to
    # over-subtract 512 bytes per record.
    for cycle_id in range(3):
        records = _rows(cycle_id, [1, 2, 3, 4])
        stage2_records += len(records)
        assert writer.enqueue_evaluation_stage2(session_key, records)
        assert writer.enqueue_evaluation_stage0(
            session_key,
            records,
            [torch.zeros(8, dtype=torch.float32) for _ in records],
        )
        assert writer.enqueue_evaluation_finalization(
            session_key,
            [
                {
                    "request_id": "request-0",
                    "cycle_id": cycle_id,
                    "cycle_status": "complete",
                    "num_rows_expected": 4,
                    "num_rows_seen": 4,
                    "finalized": True,
                }
            ],
        )
    assert writer.enqueue_request("request-0", [10], [20, 21, 22, 23])

    metrics = writer.close()
    assert metrics["dropped_records"] == 0
    assert metrics["pending_bytes"] == 0
    assert metrics["queue_depth"] == 0

    # The batch-boundary change in flush must still write every record.
    stage2_lines = 0
    for path in (tmp_path / "spool").glob("evaluation-stage2-*.jsonl"):
        stage2_lines += sum(1 for _ in open(path, encoding="utf-8"))
    assert stage2_lines == stage2_records
