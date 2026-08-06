# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DVI L0 artifact schema, sharding, checksums and validation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from vllm.v1.worker.gpu.split_dvi.artifact import (
    CAPTURE_SHARD_SIZE,
    DVIArtifactError,
    DVIArtifactManifest,
    DVIArtifactReader,
    DVIArtifactWriter,
    DVICaptureRecord,
    DVIEvalRecord,
    DVIEvaluationRecord,
    DVIRequestRecord,
)


def _logits_to_topk(logits: torch.Tensor, k: int):
    """Return (topk_ids, topk_logprobs, residual_mass, top1_id) from raw logits."""
    logprobs = torch.log_softmax(logits, dim=-1)
    topk_lp, topk_ids = torch.topk(logprobs, k)
    topk_mass = topk_lp.exp().sum().item()
    residual = max(0.0, 1.0 - topk_mass)
    top1_idx = int(torch.argmax(topk_lp).item())
    top1_id = int(topk_ids[top1_idx].item())
    return (
        topk_ids.tolist(),
        topk_lp.tolist(),
        residual,
        top1_id,
    )


def _make_manifest(**overrides) -> DVIArtifactManifest:
    defaults = {
        "schema_version": "v1",
        "base_checkpoint": "/share/models/Qwen2.5-3B-Instruct",
        "base_checkpoint_hash": "sha256:abc123",
        "head_adapter_hash": "sha256:head000",
        "tail_adapter_hash": "sha256:tail000",
        "tokenizer_revision": "main",
        "split_stage_0_size": 8,
        "split_stage_2_size": 8,
        "hidden_size": 8,
        "vocab_size": 1000,
        "draft_length": 4,
        "dtype": "bfloat16",
        "sampling_config": {"temperature": 0.7, "top_p": 1.0},
        "seed": 42,
        "max_response_length": 256,
    }
    defaults.update(overrides)
    return DVIArtifactManifest(**defaults)


def _make_request(request_id: str = "r0", response_len: int = 8) -> DVIRequestRecord:
    return DVIRequestRecord(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        response_token_ids=list(range(10, 10 + response_len)),
    )


def _make_capture(
    request_id: str = "r0",
    position: int = 0,
    hidden_size: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    topk: int = 5,
    vocab_size: int = 100,
) -> DVICaptureRecord:
    logits = torch.randn(vocab_size)
    topk_ids, topk_lps, residual, top1_id = _logits_to_topk(logits, topk)
    return DVICaptureRecord(
        request_id=request_id,
        position=position,
        stage_0_hidden=torch.randn(hidden_size, dtype=dtype),
        verifier_topk_ids=topk_ids,
        verifier_topk_logprobs=topk_lps,
        verifier_residual_mass=residual,
        verifier_top1_id=top1_id,
        stage_0_cuda_time_ms=1.0,
        verify_cuda_time_ms=2.0,
        serialization_time_ms=0.1,
        network_time_ms=0.5,
    )


def _make_eval(
    request_id: str = "r0",
    position: int = 0,
    k: int = 4,
    accepted_count: int | None = None,
) -> DVIEvalRecord:
    if accepted_count is None:
        accepted_count = k
    draft_token_ids = list(range(100, 100 + k))
    # verifier top-1 matches draft for the accepted prefix, then diverges.
    verifier_top1_ids = draft_token_ids[:accepted_count] + [
        900 + i for i in range(k - accepted_count)
    ]
    advancement = k if accepted_count == k else accepted_count + 1
    return DVIEvalRecord(
        request_id=request_id,
        position=position,
        draft_token_ids=draft_token_ids,
        verifier_top1_ids=verifier_top1_ids,
        accepted_count=accepted_count,
        advancement=advancement,
        alignment_check_passed=True,
    )


def test_round_trip(tmp_path: Path) -> None:
    manifest = _make_manifest()
    req = _make_request()
    cap = _make_capture()
    ev = _make_eval(accepted_count=2)

    out_dir = tmp_path / "artifact"
    with DVIArtifactWriter(out_dir, manifest) as writer:
        writer.write_requests([req])
        writer.write_capture_records([cap])
        writer.write_eval_records([ev])

    reader = DVIArtifactReader(out_dir)
    assert reader.manifest.schema_version == "v1"
    assert len(reader.requests) == 1
    assert reader.requests[0].request_id == "r0"

    caps = list(reader.iter_capture_records())
    assert len(caps) == 1
    assert caps[0].request_id == "r0"
    assert caps[0].position == 0
    assert caps[0].stage_0_hidden.shape == (8,)
    assert str(caps[0].stage_0_hidden.dtype) == "torch.bfloat16"

    evs = list(reader.iter_eval_records())
    assert len(evs) == 1
    assert evs[0].accepted_count == 2
    assert evs[0].advancement == 3


def test_sharding(tmp_path: Path) -> None:
    manifest = _make_manifest(
        max_response_length=CAPTURE_SHARD_SIZE + 10,
        vocab_size=CAPTURE_SHARD_SIZE + 50,
    )
    req = _make_request(response_len=CAPTURE_SHARD_SIZE + 10)
    records = [
        _make_capture(position=i)
        for i in range(CAPTURE_SHARD_SIZE + 10)
    ]

    out_dir = tmp_path / "artifact"
    with DVIArtifactWriter(out_dir, manifest, shard_size=CAPTURE_SHARD_SIZE) as writer:
        writer.write_requests([req])
        writer.write_capture_records(records)

    jsonl_files = sorted(out_dir.glob("capture-*.jsonl"))
    st_files = sorted(out_dir.glob("capture-*.safetensors"))
    assert len(jsonl_files) == 2
    assert len(st_files) == 2

    reader = DVIArtifactReader(out_dir)
    caps = list(reader.iter_capture_records())
    assert len(caps) == CAPTURE_SHARD_SIZE + 10


def test_multi_request(tmp_path: Path) -> None:
    manifest = _make_manifest()
    requests = [_make_request(f"r{i}", response_len=4) for i in range(3)]
    captures = [
        _make_capture(request_id=f"r{i % 3}", position=i % 4)
        for i in range(9)
    ]

    out_dir = tmp_path / "artifact"
    with DVIArtifactWriter(out_dir, manifest) as writer:
        writer.write_requests(requests)
        writer.write_capture_records(captures)

    reader = DVIArtifactReader(out_dir)
    assert {r.request_id for r in reader.requests} == {"r0", "r1", "r2"}
    assert len(list(reader.iter_capture_records())) == 9


def test_corrupt_checksum(tmp_path: Path) -> None:
    manifest = _make_manifest()
    req = _make_request()
    cap = _make_capture()

    out_dir = tmp_path / "artifact"
    with DVIArtifactWriter(out_dir, manifest) as writer:
        writer.write_requests([req])
        writer.write_capture_records([cap])

    # Corrupt a data file.
    capture_jsonl = out_dir / "capture-00000.jsonl"
    with open(capture_jsonl, "a", encoding="utf-8") as f:
        f.write("\n# corrupted")

    with pytest.raises(DVIArtifactError, match="Checksum mismatch"):
        DVIArtifactReader(out_dir)


def test_extra_file_rejected(tmp_path: Path) -> None:
    manifest = _make_manifest()
    out_dir = tmp_path / "artifact"
    with DVIArtifactWriter(out_dir, manifest) as writer:
        writer.write_requests([_make_request()])

    extra = out_dir / "extra.txt"
    extra.write_text("unexpected")

    with pytest.raises(DVIArtifactError, match="Unexpected files"):
        DVIArtifactReader(out_dir)


def test_refuse_overwrite_existing(tmp_path: Path) -> None:
    manifest = _make_manifest()
    out_dir = tmp_path / "artifact"
    out_dir.mkdir()
    (out_dir / "existing.txt").write_text("old")

    with pytest.raises(DVIArtifactError, match="Refusing to overwrite"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            writer.write_requests([_make_request()])


def test_shard_size_positive(tmp_path: Path) -> None:
    manifest = _make_manifest()
    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="shard_size must be positive"):
        DVIArtifactWriter(out_dir, manifest, shard_size=0)


def test_illegal_position(tmp_path: Path) -> None:
    manifest = _make_manifest()
    req = _make_request(response_len=3)
    cap = _make_capture(position=5)  # > response_len

    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="Invalid position"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            writer.write_requests([req])
            writer.write_capture_records([cap])


def test_illegal_residual_mass(tmp_path: Path) -> None:
    manifest = _make_manifest()
    req = _make_request()
    cap = _make_capture()
    cap.verifier_residual_mass = 1.5

    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="verifier_residual_mass"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            writer.write_requests([req])
            writer.write_capture_records([cap])


def test_residual_mass_inconsistent(tmp_path: Path) -> None:
    manifest = _make_manifest()
    req = _make_request()
    cap = _make_capture()
    cap.verifier_residual_mass = 0.5  # will not match actual residual

    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="inconsistent"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            writer.write_requests([req])
            writer.write_capture_records([cap])


def test_topk_probability_mass_exceeds_one(tmp_path: Path) -> None:
    manifest = _make_manifest()
    req = _make_request()
    cap = _make_capture()
    cap.verifier_topk_logprobs = [-0.1] * 5

    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="Top-k probability mass"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            writer.write_requests([req])
            writer.write_capture_records([cap])


def test_topk_ids_not_unique(tmp_path: Path) -> None:
    manifest = _make_manifest()
    req = _make_request()
    cap = _make_capture()
    cap.verifier_topk_ids = [1, 1, 2, 3, 4]

    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="unique"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            writer.write_requests([req])
            writer.write_capture_records([cap])


def test_top1_not_in_topk(tmp_path: Path) -> None:
    manifest = _make_manifest()
    req = _make_request()
    cap = _make_capture()
    cap.verifier_top1_id = 99999

    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="not in verifier_topk_ids"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            writer.write_requests([req])
            writer.write_capture_records([cap])


def test_top1_not_max_logprob(tmp_path: Path) -> None:
    manifest = _make_manifest()
    req = _make_request()
    cap = _make_capture()
    # Swap top1_id to the second-highest logprob token.
    sorted_indices = sorted(
        range(len(cap.verifier_topk_logprobs)),
        key=lambda i: cap.verifier_topk_logprobs[i],
        reverse=True,
    )
    cap.verifier_top1_id = cap.verifier_topk_ids[sorted_indices[1]]

    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="maximum logprob"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            writer.write_requests([req])
            writer.write_capture_records([cap])


def test_illegal_logprob(tmp_path: Path) -> None:
    manifest = _make_manifest()
    req = _make_request()
    cap = _make_capture()
    cap.verifier_topk_logprobs[0] = float("inf")

    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="Invalid verifier logprob"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            writer.write_requests([req])
            writer.write_capture_records([cap])


def test_negative_timing(tmp_path: Path) -> None:
    manifest = _make_manifest()
    req = _make_request()
    cap = _make_capture()
    cap.stage_0_cuda_time_ms = -1.0

    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="stage_0_cuda_time_ms"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            writer.write_requests([req])
            writer.write_capture_records([cap])


def test_topk_length_mismatch(tmp_path: Path) -> None:
    manifest = _make_manifest()
    req = _make_request()
    cap = _make_capture()
    cap.verifier_topk_logprobs = cap.verifier_topk_logprobs[:-1]

    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="length mismatch"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            writer.write_requests([req])
            writer.write_capture_records([cap])


def test_unknown_schema_version(tmp_path: Path) -> None:
    manifest = _make_manifest(schema_version="v0")
    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="Unknown schema version"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            pass


def test_hidden_shape_dtype_mismatch(tmp_path: Path) -> None:
    manifest = _make_manifest(hidden_size=8, dtype="bfloat16")
    req = _make_request()
    cap = _make_capture(hidden_size=4)

    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="shape"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            writer.write_requests([req])
            writer.write_capture_records([cap])

    cap2 = _make_capture(hidden_size=8, dtype=torch.float32)
    out_dir2 = tmp_path / "artifact2"
    with pytest.raises(DVIArtifactError, match="dtype"):
        with DVIArtifactWriter(out_dir2, manifest) as writer:
            writer.write_requests([req])
            writer.write_capture_records([cap2])


def test_eval_accepted_count_must_match_common_prefix(tmp_path: Path) -> None:
    manifest = _make_manifest(draft_length=4)
    req = _make_request()

    ev = _make_eval(accepted_count=2, request_id="r0")
    ev.accepted_count = 3  # lie about accepted count

    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="common prefix"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            writer.write_requests([req])
            writer.write_eval_records([ev])


def test_eval_advancement_formula(tmp_path: Path) -> None:
    manifest = _make_manifest(draft_length=4)
    req = _make_request()

    # Partial accept: accepted_count=2 -> advancement must be 3.
    ev_bad = _make_eval(accepted_count=2, request_id="r0")
    ev_bad.advancement = 2

    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="advancement"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            writer.write_requests([req])
            writer.write_eval_records([ev_bad])

    # Full accept: advancement must be 4.
    ev_good = _make_eval(accepted_count=4, request_id="r0")
    assert ev_good.advancement == 4


def test_manifest_base_checkpoint_must_be_hash(tmp_path: Path) -> None:
    manifest = _make_manifest(base_checkpoint_hash="")
    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="base_checkpoint_hash"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            pass


def test_atomic_write_failure_cleanup(tmp_path: Path) -> None:
    manifest = _make_manifest()
    out_dir = tmp_path / "artifact"
    writer = DVIArtifactWriter(out_dir, manifest)
    writer.write_requests([_make_request()])
    temp_dir = writer._temp_dir
    assert temp_dir.exists()
    writer.__exit__(RuntimeError, RuntimeError("boom"), None)
    assert not temp_dir.exists()
    assert not out_dir.exists()


def test_empty_eval_ok(tmp_path: Path) -> None:
    manifest = _make_manifest()
    out_dir = tmp_path / "artifact"
    with DVIArtifactWriter(out_dir, manifest) as writer:
        writer.write_requests([_make_request()])
        writer.write_capture_records([_make_capture()])

    reader = DVIArtifactReader(out_dir)
    assert list(reader.iter_eval_records()) == []



def test_negative_request_token_id(tmp_path: Path) -> None:
    manifest = _make_manifest()
    req = _make_request()
    req.prompt_token_ids = [1, -1, 3]

    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="non-negative"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            writer.write_requests([req])


def test_response_exceeds_max_length(tmp_path: Path) -> None:
    manifest = _make_manifest(max_response_length=4)
    req = _make_request(response_len=5)

    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="max_response_length"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            writer.write_requests([req])


def test_eval_negative_timing(tmp_path: Path) -> None:
    manifest = _make_manifest(draft_length=4)
    req = _make_request()
    ev = _make_eval(request_id="r0")
    ev.draft_cuda_time_ms = -0.5

    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="draft_cuda_time_ms"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            writer.write_requests([req])
            writer.write_eval_records([ev])


def test_eval_nonfinite_timing(tmp_path: Path) -> None:
    manifest = _make_manifest(draft_length=4)
    req = _make_request()
    ev = _make_eval(request_id="r0")
    ev.verify_cuda_time_ms = float("nan")

    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="verify_cuda_time_ms"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            writer.write_requests([req])
            writer.write_eval_records([ev])


def test_eval_alignment_check_passed_must_be_bool(tmp_path: Path) -> None:
    manifest = _make_manifest(draft_length=4)
    req = _make_request()
    ev = _make_eval(request_id="r0")
    ev.alignment_check_passed = "yes"  # type: ignore[assignment]

    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="alignment_check_passed"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            writer.write_requests([req])
            writer.write_eval_records([ev])


def test_orphan_capture_safetensors_rejected(tmp_path: Path) -> None:
    from vllm.v1.worker.gpu.split_dvi.artifact import _sha256_file

    manifest = _make_manifest()
    out_dir = tmp_path / "artifact"
    with DVIArtifactWriter(out_dir, manifest) as writer:
        writer.write_requests([_make_request()])

    # Add an orphan capture safetensors file that has no jsonl pair.
    orphan = out_dir / "capture-00001.safetensors"
    orphan.write_bytes(b"orphan")

    # Patch manifest checksums to list the orphan with its real hash.
    manifest_path = out_dir / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["file_checksums"][orphan.name] = _sha256_file(orphan)
    manifest_path.write_text(json.dumps(data, indent=2))

    with pytest.raises(DVIArtifactError, match="Missing jsonl pair"):
        DVIArtifactReader(out_dir)


def test_symlink_rejected(tmp_path: Path) -> None:
    from vllm.v1.worker.gpu.split_dvi.artifact import _sha256_file

    manifest = _make_manifest()
    out_dir = tmp_path / "artifact"
    with DVIArtifactWriter(out_dir, manifest) as writer:
        writer.write_requests([_make_request()])

    # Symlink listed in the manifest under an allowed eval name so that it is
    # not caught as an extra file; the reader should still reject it.
    target = tmp_path / "external.txt"
    target.write_text("secret")
    link = out_dir / "eval-00000.jsonl"
    link.symlink_to(target)

    manifest_path = out_dir / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["file_checksums"][link.name] = _sha256_file(link)
    manifest_path.write_text(json.dumps(data, indent=2))

    with pytest.raises(DVIArtifactError, match="regular files"):
        DVIArtifactReader(out_dir)


def test_close_failure_cleans_up_temp_dir(tmp_path: Path) -> None:
    manifest = _make_manifest()
    out_dir = tmp_path / "artifact"
    writer = DVIArtifactWriter(out_dir, manifest)
    writer.write_requests([_make_request()])
    temp_dir = writer._temp_dir

    # Simulate a final-dir race by creating the target directory before close.
    out_dir.mkdir()
    with pytest.raises(DVIArtifactError, match="Refusing to overwrite"):
        writer.close()
    assert not temp_dir.exists()
    assert out_dir.exists()



def test_manifest_vocab_size_round_trip(tmp_path: Path) -> None:
    manifest = _make_manifest(vocab_size=32000)
    req = _make_request()
    cap = _make_capture(vocab_size=32000)

    out_dir = tmp_path / "artifact"
    with DVIArtifactWriter(out_dir, manifest) as writer:
        writer.write_requests([req])
        writer.write_capture_records([cap])

    reader = DVIArtifactReader(out_dir)
    assert reader.manifest.vocab_size == 32000


def test_request_token_exceeds_vocab_size(tmp_path: Path) -> None:
    manifest = _make_manifest(vocab_size=100)
    req = _make_request()
    req.response_token_ids = [50, 99, 100]

    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="vocab_size"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            writer.write_requests([req])


def test_capture_topk_id_exceeds_vocab_size(tmp_path: Path) -> None:
    manifest = _make_manifest(vocab_size=50)
    req = _make_request()
    cap = _make_capture(vocab_size=100)
    # _make_capture uses top-k from vocab_size=100, which may all be <50 or not.
    # Force an out-of-vocab id to guarantee the failure path.
    cap.verifier_topk_ids[0] = 99
    cap.verifier_top1_id = 99

    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="vocab_size"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            writer.write_requests([req])
            writer.write_capture_records([cap])


def test_manifest_vocab_size_must_be_positive(tmp_path: Path) -> None:
    manifest = _make_manifest(vocab_size=0)
    out_dir = tmp_path / "artifact"
    with pytest.raises(DVIArtifactError, match="vocab_size must be positive"):
        with DVIArtifactWriter(out_dir, manifest) as writer:
            pass



def _make_v2_evaluation_records(
    manifest: DVIArtifactManifest,
    request_id: str = "r0",
    *,
    rejected: bool = False,
) -> list[DVIEvaluationRecord]:
    proposal_count = manifest.draft_length - 1
    committed = [10, 11] if rejected else [10, 11, 12, 13]
    accepted_count = 1 if rejected else proposal_count
    advancement = len(committed)
    records: list[DVIEvaluationRecord] = []
    for row_index in range(proposal_count + 1):
        is_proposal = row_index < proposal_count
        logits = torch.full((manifest.vocab_size,), -10.0)
        logits[20 + row_index] = 10.0
        topk_ids, topk_lps, residual, top1_id = _logits_to_topk(logits, 2)
        records.append(
            DVIEvaluationRecord(
                request_id=request_id,
                cycle_id=3,
                generation_id=7,
                row_index=row_index,
                absolute_position=row_index,
                row_kind="proposal" if is_proposal else "bonus",
                num_proposals=proposal_count,
                draft_token_id=100 + row_index if is_proposal else None,
                draft_support_token_ids=[100 + row_index, 200 + row_index]
                if is_proposal
                else [],
                draft_support_logits=[1.0, 0.0] if is_proposal else [],
                stage_0_hidden=torch.zeros(
                    manifest.hidden_size, dtype=torch.float32
                ),
                verifier_topk_ids=topk_ids,
                verifier_topk_logprobs=topk_lps,
                verifier_residual_mass=residual,
                teacher_probs_on_draft_support=[0.9, 0.0]
                if is_proposal
                else [],
                accepted=(
                    True if is_proposal and row_index < accepted_count else
                    False if is_proposal and row_index == accepted_count else None
                ),
                accepted_count=accepted_count,
                committed_token_ids=committed if row_index == 0 else [],
                correction_token_id=(
                    committed[-1] if rejected and row_index == 0 else None
                ),
                terminal_token_id=(
                    committed[-1] if row_index == 0 else None
                ),
                advancement=advancement if row_index == 0 else 0,
            )
        )
    return records


def _make_v2_manifest() -> DVIArtifactManifest:
    return _make_manifest(
        schema_version="v2",
        capture_mode="evaluation",
        policy_version="policy5",
        num_proposals=3,
        bonus_token=True,
        dtype="float32",
    )


def test_v2_evaluation_round_trip_preserves_bonus_contract(tmp_path: Path) -> None:
    manifest = _make_v2_manifest()
    request = _make_request(response_len=4)
    records = _make_v2_evaluation_records(manifest)
    out_dir = tmp_path / "evaluation-artifact"

    with DVIArtifactWriter(out_dir, manifest) as writer:
        writer.write_requests([request])
        writer.write_evaluation_records(records)

    reader = DVIArtifactReader(out_dir)
    loaded = list(reader.iter_evaluation_records())
    assert len(loaded) == 4
    assert loaded[-1].row_kind == "bonus"
    assert loaded[-1].draft_token_id is None
    assert loaded[-1].draft_support_token_ids == []
    assert loaded[0].advancement == 4
    assert loaded[0].correction_token_id is None
    assert loaded[0].committed_token_ids == [10, 11, 12, 13]


def test_v2_evaluation_reject_records_store_correction_token(
    tmp_path: Path,
) -> None:
    manifest = _make_v2_manifest()
    request = _make_request(response_len=4)
    records = _make_v2_evaluation_records(manifest, rejected=True)
    out_dir = tmp_path / "evaluation-artifact"

    with DVIArtifactWriter(out_dir, manifest) as writer:
        writer.write_requests([request])
        writer.write_evaluation_records(records)

    loaded = list(DVIArtifactReader(out_dir).iter_evaluation_records())
    assert loaded[0].accepted_count == 1
    assert loaded[0].advancement == 2
    assert loaded[0].correction_token_id == 11


def test_v2_bonus_row_rejects_draft_fields(tmp_path: Path) -> None:
    manifest = _make_v2_manifest()
    request = _make_request(response_len=4)
    records = _make_v2_evaluation_records(manifest)
    records[-1].draft_token_id = 999

    with pytest.raises(DVIArtifactError, match="bonus rows"):
        with DVIArtifactWriter(tmp_path / "evaluation-artifact", manifest) as writer:
            writer.write_requests([request])
            writer.write_evaluation_records(records)
