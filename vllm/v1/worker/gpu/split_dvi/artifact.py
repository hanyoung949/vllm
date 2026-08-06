# SPDX-License-Identifier: Apache-2.0
"""File-based schema-versioned artifact for DVI L0 offline evaluation.

Disk layout::

    artifact_dir/
      manifest.json
      requests.jsonl
      capture-00000.jsonl
      capture-00000.safetensors
      eval-00000.jsonl

Rules:
- Tensors are stored in safetensors shards, metadata lives in JSONL.
- File checksums (sha256) are recorded in the manifest and verified on read.
- Writes are atomic: all data is written to a temp directory and renamed.
- Unknown schema versions fail-fast.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors


KNOWN_SCHEMA_VERSIONS = {"v1", "v2"}
CAPTURE_SHARD_SIZE = 1024  # number of capture records per shard
EVALUATION_SHARD_SIZE = 128


class DVIArtifactError(ValueError):
    """Raised when an artifact fails validation or I/O consistency checks."""


@dataclass
class DVIArtifactManifest:
    schema_version: str = "v1"
    base_checkpoint: str = ""
    base_checkpoint_hash: str = ""  # immutable revision / file hash
    head_adapter_hash: str = ""
    tail_adapter_hash: str = ""
    tokenizer_revision: str = ""
    split_stage_0_size: int = 0
    split_stage_2_size: int = 0
    hidden_size: int = 0
    vocab_size: int = 0
    draft_length: int = 4
    dtype: str = "bfloat16"
    sampling_config: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    max_response_length: int = 256
    file_checksums: dict[str, str] = field(default_factory=dict)

    # KL / teacher semantics metadata. Must be explicit so experiments are
    # comparable and so the current L0 forward-KL variant is not confused
    # with the paper's reverse-KL objective.
    kl_direction: str = "forward"  # "forward" | "reverse" | "ce" | "pg" | ...
    teacher_distribution_kind: str = "processed_topk_residual"
    # "raw_temperature_scaled" | "processed_topk_residual" | ...
    teacher_temperature: float | None = None
    # Raw draft-head logits are sampled independently of teacher temperature.
    draft_temperature: float = 1.0
    topk_mass_coverage: float | None = None  # low -> truncated-KL approximation unreliable
    capture_mode: str = "train"  # "train" | "evaluation"
    policy_version: str = ""
    num_proposals: int | None = None
    bonus_token: bool = False

    def validate(self) -> None:
        if self.schema_version not in KNOWN_SCHEMA_VERSIONS:
            raise DVIArtifactError(
                f"Unknown schema version {self.schema_version!r}. "
                f"Known versions: {KNOWN_SCHEMA_VERSIONS}"
            )
        if not self.base_checkpoint_hash:
            raise DVIArtifactError("base_checkpoint_hash must be non-empty")
        if not self.head_adapter_hash:
            raise DVIArtifactError("head_adapter_hash must be non-empty")
        if not self.tail_adapter_hash:
            raise DVIArtifactError("tail_adapter_hash must be non-empty")
        if not self.tokenizer_revision:
            raise DVIArtifactError("tokenizer_revision must be non-empty")
        if self.split_stage_0_size <= 0:
            raise DVIArtifactError("split_stage_0_size must be positive")
        if self.split_stage_2_size <= 0:
            raise DVIArtifactError("split_stage_2_size must be positive")
        if self.hidden_size <= 0:
            raise DVIArtifactError("hidden_size must be positive")
        if self.vocab_size <= 0:
            raise DVIArtifactError("vocab_size must be positive")
        if self.draft_length <= 0:
            raise DVIArtifactError("draft_length must be positive")
        if not isinstance(self.sampling_config, dict):
            raise DVIArtifactError("sampling_config must be a dict")
        if self.capture_mode not in {"train", "evaluation"}:
            raise DVIArtifactError(
                f"capture_mode must be 'train' or 'evaluation', got "
                f"{self.capture_mode!r}"
            )
        if self.schema_version == "v2":
            if not math.isfinite(self.draft_temperature) or self.draft_temperature <= 0:
                raise DVIArtifactError(
                    "schema v2 draft_temperature must be positive and finite"
                )
            if self.capture_mode != "evaluation":
                raise DVIArtifactError(
                    "schema v2 artifacts must use capture_mode='evaluation'"
                )
            if self.draft_length < 2:
                raise DVIArtifactError(
                    "schema v2 evaluation artifacts require draft_length >= 2"
                )
            if not self.policy_version:
                raise DVIArtifactError(
                    "schema v2 artifacts require a non-empty policy_version"
                )
            expected = self.draft_length - 1
            if self.num_proposals not in (None, expected):
                raise DVIArtifactError(
                    f"num_proposals must be {expected} for draft_length="
                    f"{self.draft_length}, got {self.num_proposals}"
                )
            self.num_proposals = expected
            if not self.bonus_token:
                raise DVIArtifactError(
                    "schema v2 evaluation artifacts require bonus_token=True"
                )


@dataclass
class DVIRequestRecord:
    request_id: str = ""
    prompt_token_ids: list[int] = field(default_factory=list)
    response_token_ids: list[int] = field(default_factory=list)

    def validate(self, manifest: DVIArtifactManifest) -> None:
        if not self.request_id:
            raise DVIArtifactError("request_id must be non-empty")
        for name, tokens in (
            ("prompt_token_ids", self.prompt_token_ids),
            ("response_token_ids", self.response_token_ids),
        ):
            if not all(isinstance(t, int) for t in tokens):
                raise DVIArtifactError(f"{name} must be a list of ints")
            if any(t < 0 for t in tokens):
                raise DVIArtifactError(
                    f"{name} must be non-negative integers"
                )
            if any(t >= manifest.vocab_size for t in tokens):
                raise DVIArtifactError(
                    f"{name} contains token id >= vocab_size "
                    f"{manifest.vocab_size}"
                )
        if len(self.response_token_ids) > manifest.max_response_length:
            raise DVIArtifactError(
                f"response_token_ids length "
                f"{len(self.response_token_ids)} exceeds "
                f"manifest max_response_length "
                f"{manifest.max_response_length}"
            )


@dataclass
class DVICaptureRecord:
    request_id: str = ""
    position: int = 0
    stage_0_hidden: torch.Tensor | None = None
    verifier_topk_ids: list[int] = field(default_factory=list)
    verifier_topk_logprobs: list[float] = field(default_factory=list)
    verifier_residual_mass: float = 0.0
    verifier_top1_id: int = 0
    stage_0_cuda_time_ms: float = 0.0
    verify_cuda_time_ms: float = 0.0
    serialization_time_ms: float = 0.0
    network_time_ms: float = 0.0

    def validate(
        self,
        request_index: dict[str, DVIRequestRecord],
        manifest: DVIArtifactManifest,
    ) -> None:
        if self.request_id not in request_index:
            raise DVIArtifactError(
                f"Capture record references unknown request_id {self.request_id!r}"
            )
        req = request_index[self.request_id]
        response_len = len(req.response_token_ids)
        if not (0 <= self.position <= response_len):
            raise DVIArtifactError(
                f"Invalid position {self.position} for request {self.request_id!r} "
                f"with response length {response_len}"
            )
        if len(self.verifier_topk_ids) != len(self.verifier_topk_logprobs):
            raise DVIArtifactError(
                "verifier_topk_ids and verifier_topk_logprobs length mismatch"
            )
        if not self.verifier_topk_ids:
            raise DVIArtifactError("verifier_topk_ids must not be empty")

        # Token IDs must be valid ints, unique within top-k, and in vocabulary.
        for tid in self.verifier_topk_ids:
            if not isinstance(tid, int) or tid < 0:
                raise DVIArtifactError(
                    f"Invalid verifier_topk_id {tid!r}; must be non-negative int"
                )
            if tid >= manifest.vocab_size:
                raise DVIArtifactError(
                    f"verifier_topk_id {tid} exceeds manifest vocab_size "
                    f"{manifest.vocab_size}"
                )
        if len(set(self.verifier_topk_ids)) != len(self.verifier_topk_ids):
            raise DVIArtifactError("verifier_topk_ids must be unique")

        # Logprobs must be finite and form a sub-distribution with residual.
        for lp in self.verifier_topk_logprobs:
            if not isinstance(lp, float) or math.isnan(lp) or math.isinf(lp):
                raise DVIArtifactError(
                    f"Invalid verifier logprob {lp!r}; must be finite float"
                )
        topk_mass = sum(math.exp(lp) for lp in self.verifier_topk_logprobs)
        if topk_mass > 1.0 + 1e-4:
            raise DVIArtifactError(
                f"Top-k probability mass {topk_mass} exceeds 1.0"
            )
        if not (0.0 <= self.verifier_residual_mass <= 1.0):
            raise DVIArtifactError(
                f"verifier_residual_mass must be in [0, 1], got "
                f"{self.verifier_residual_mass}"
            )
        expected_residual = max(0.0, 1.0 - topk_mass)
        if abs(self.verifier_residual_mass - expected_residual) > 1e-3:
            raise DVIArtifactError(
                f"verifier_residual_mass {self.verifier_residual_mass} inconsistent "
                f"with top-k mass {topk_mass}; expected ~{expected_residual}"
            )

        # top-1 token must be among top-k and correspond to max logprob.
        if self.verifier_top1_id not in self.verifier_topk_ids:
            raise DVIArtifactError(
                f"verifier_top1_id {self.verifier_top1_id} not in verifier_topk_ids"
            )
        max_lp = max(self.verifier_topk_logprobs)
        top1_idx = self.verifier_topk_ids.index(self.verifier_top1_id)
        if self.verifier_topk_logprobs[top1_idx] != max_lp:
            raise DVIArtifactError(
                "verifier_top1_id does not correspond to the maximum logprob"
            )

        # Timing fields must be finite and non-negative.
        for name, value in (
            ("stage_0_cuda_time_ms", self.stage_0_cuda_time_ms),
            ("verify_cuda_time_ms", self.verify_cuda_time_ms),
            ("serialization_time_ms", self.serialization_time_ms),
            ("network_time_ms", self.network_time_ms),
        ):
            if (not isinstance(value, (int, float))
                    or math.isnan(value)
                    or math.isinf(value)
                    or value < 0):
                raise DVIArtifactError(
                    f"{name} must be a finite non-negative number, "
                    f"got {value!r}"
                )

        if self.stage_0_hidden is None:
            raise DVIArtifactError("stage_0_hidden must be provided")
        # Normalize tensor to CPU contiguous for stable storage.
        self.stage_0_hidden = self.stage_0_hidden.detach().cpu().contiguous()
        if self.stage_0_hidden.shape != (manifest.hidden_size,):
            raise DVIArtifactError(
                f"stage_0_hidden shape {tuple(self.stage_0_hidden.shape)} does not "
                f"match manifest hidden_size {manifest.hidden_size}"
            )
        if str(self.stage_0_hidden.dtype) != f"torch.{manifest.dtype}":
            raise DVIArtifactError(
                f"stage_0_hidden dtype {self.stage_0_hidden.dtype} does not match "
                f"manifest dtype {manifest.dtype}"
            )


@dataclass
class DVIEvalRecord:
    request_id: str = ""
    position: int = 0
    draft_token_ids: list[int] = field(default_factory=list)
    verifier_top1_ids: list[int] = field(default_factory=list)
    accepted_count: int = 0
    advancement: int = 0
    alignment_check_passed: bool = False
    draft_cuda_time_ms: float = 0.0
    verify_cuda_time_ms: float = 0.0

    def validate(
        self,
        request_index: dict[str, DVIRequestRecord],
        manifest: DVIArtifactManifest,
    ) -> None:
        if self.request_id not in request_index:
            raise DVIArtifactError(
                f"Eval record references unknown request_id {self.request_id!r}"
            )
        req = request_index[self.request_id]
        response_len = len(req.response_token_ids)
        if not (0 <= self.position <= response_len):
            raise DVIArtifactError(
                f"Invalid eval position {self.position} for request "
                f"{self.request_id!r} with response length {response_len}"
            )
        k = manifest.draft_length
        if len(self.draft_token_ids) != k:
            raise DVIArtifactError(
                f"draft_token_ids length {len(self.draft_token_ids)} does not match "
                f"manifest draft_length {k}"
            )
        if len(self.verifier_top1_ids) != k:
            raise DVIArtifactError(
                f"verifier_top1_ids length {len(self.verifier_top1_ids)} does not "
                f"match manifest draft_length {k}"
            )
        for tid in self.draft_token_ids + self.verifier_top1_ids:
            if not isinstance(tid, int) or tid < 0:
                raise DVIArtifactError(
                    f"Token id {tid!r} must be a non-negative int"
                )
            if tid >= manifest.vocab_size:
                raise DVIArtifactError(
                    f"Token id {tid} exceeds manifest vocab_size "
                    f"{manifest.vocab_size}"
                )

        # accepted_count must equal the longest common prefix of draft and verifier.
        common_prefix = 0
        for d, v in zip(self.draft_token_ids, self.verifier_top1_ids):
            if d == v:
                common_prefix += 1
            else:
                break
        if self.accepted_count != common_prefix:
            raise DVIArtifactError(
                f"accepted_count {self.accepted_count} does not match common prefix "
                f"length {common_prefix} between draft_token_ids and verifier_top1_ids"
            )
        if not (0 <= self.accepted_count <= k):
            raise DVIArtifactError(
                f"accepted_count {self.accepted_count} out of range [0, {k}]"
            )
        # Greedy / no-bonus advancement formula.
        if self.accepted_count == k:
            expected_advancement = k
        else:
            expected_advancement = self.accepted_count + 1
        if self.advancement != expected_advancement:
            raise DVIArtifactError(
                f"advancement {self.advancement} does not match greedy/no-bonus "
                f"formula; expected {expected_advancement} for accepted_count="
                f"{self.accepted_count} and k={k}"
            )

        if not isinstance(self.alignment_check_passed, bool):
            raise DVIArtifactError(
                f"alignment_check_passed must be bool, got "
                f"{self.alignment_check_passed!r}"
            )

        for name, value in (
            ("draft_cuda_time_ms", self.draft_cuda_time_ms),
            ("verify_cuda_time_ms", self.verify_cuda_time_ms),
        ):
            if (not isinstance(value, (int, float))
                    or math.isnan(value)
                    or math.isinf(value)
                    or value < 0):
                raise DVIArtifactError(
                    f"{name} must be a finite non-negative number, "
                    f"got {value!r}"
                )


@dataclass
class DVIEvaluationRecord:
    """One dense evaluation row from a real stochastic DVI cycle.

    Stage-0 and stage-2 produce complementary halves of this record. The
    offline merger joins them by cycle identity, so replay never reconstructs
    a population by requiring contiguous positions in a sparse capture.
    """

    request_id: str = ""
    cycle_id: int = 0
    generation_id: int = 0
    row_index: int = 0
    absolute_position: int = 0
    rng_position: int | None = None
    request_seed: int | None = None
    row_kind: str = "proposal"  # "proposal" | "bonus"
    num_proposals: int = 0
    draft_token_id: int | None = None
    draft_support_token_ids: list[int] = field(default_factory=list)
    draft_support_logits: list[float] = field(default_factory=list)
    stage_0_hidden: torch.Tensor | None = None
    verifier_topk_ids: list[int] = field(default_factory=list)
    verifier_topk_logprobs: list[float] = field(default_factory=list)
    verifier_residual_mass: float = 0.0
    teacher_probs_on_draft_support: list[float] = field(default_factory=list)
    accepted: bool | None = None
    accepted_count: int = 0
    committed_token_ids: list[int] = field(default_factory=list)
    correction_token_id: int | None = None
    terminal_token_id: int | None = None
    advancement: int = 0

    def validate(
        self,
        request_index: dict[str, DVIRequestRecord],
        manifest: DVIArtifactManifest,
    ) -> None:
        if manifest.schema_version != "v2":
            raise DVIArtifactError(
                "DVIEvaluationRecord requires an artifact schema v2 manifest"
            )
        if self.request_id not in request_index:
            raise DVIArtifactError(
                f"Evaluation record references unknown request_id "
                f"{self.request_id!r}"
            )
        if self.cycle_id < 0 or self.generation_id < 0 or self.row_index < 0:
            raise DVIArtifactError(
                "cycle_id, generation_id, row_index must be non-negative"
            )
        if self.row_kind not in {"proposal", "bonus"}:
            raise DVIArtifactError(f"invalid evaluation row_kind {self.row_kind!r}")
        expected_proposals = manifest.draft_length - 1
        if not 0 <= self.num_proposals <= expected_proposals:
            raise DVIArtifactError(
                f"num_proposals {self.num_proposals} is outside "
                f"[0, {expected_proposals}]"
            )
        if self.row_kind == "proposal" and self.row_index >= self.num_proposals:
            raise DVIArtifactError("proposal row_index is outside num_proposals")
        if self.row_kind == "bonus" and self.row_index != self.num_proposals:
            raise DVIArtifactError("bonus row must follow all proposal rows")
        if self.row_kind == "bonus":
            if self.draft_token_id is not None:
                raise DVIArtifactError("bonus rows must not contain a draft token")
            if self.draft_support_token_ids or self.draft_support_logits:
                raise DVIArtifactError(
                    "bonus rows must not contain draft support data"
                )
        elif self.draft_token_id is None:
            raise DVIArtifactError("proposal rows require a draft token")

        req = request_index[self.request_id]
        if not (0 <= self.absolute_position <= len(req.response_token_ids)):
            raise DVIArtifactError("evaluation absolute_position is out of range")
        if self.rng_position is not None and self.rng_position < 0:
            raise DVIArtifactError("evaluation rng_position must be non-negative")
        if self.request_seed is not None and not isinstance(self.request_seed, int):
            raise DVIArtifactError("evaluation request_seed must be an integer")
        if self.draft_token_id is not None and not (
            isinstance(self.draft_token_id, int)
            and 0 <= self.draft_token_id < manifest.vocab_size
        ):
            raise DVIArtifactError("invalid draft_token_id")
        if len(self.draft_support_token_ids) != len(self.draft_support_logits):
            raise DVIArtifactError("draft support ids/logits length mismatch")
        if len(self.teacher_probs_on_draft_support) not in {
            0, len(self.draft_support_token_ids)
        }:
            raise DVIArtifactError(
                "teacher_probs_on_draft_support must align with draft support"
            )
        if len(set(self.draft_support_token_ids)) != len(
            self.draft_support_token_ids
        ):
            raise DVIArtifactError("draft support ids must be unique")
        for token_id in self.draft_support_token_ids:
            if not isinstance(token_id, int) or not 0 <= token_id < manifest.vocab_size:
                raise DVIArtifactError("invalid draft support token id")
        for value in self.draft_support_logits + self.verifier_topk_logprobs:
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise DVIArtifactError("evaluation logits/logprobs must be finite")
        for value in self.teacher_probs_on_draft_support:
            if not isinstance(value, (float, int)) or not 0.0 <= value <= 1.0:
                raise DVIArtifactError(
                    "teacher support probabilities must be in [0, 1]"
                )
        if len(self.verifier_topk_ids) != len(self.verifier_topk_logprobs):
            raise DVIArtifactError("verifier top-k ids/logprobs length mismatch")
        if len(set(self.verifier_topk_ids)) != len(self.verifier_topk_ids):
            raise DVIArtifactError("verifier top-k ids must be unique")
        for token_id in self.verifier_topk_ids:
            if not isinstance(token_id, int) or not 0 <= token_id < manifest.vocab_size:
                raise DVIArtifactError("invalid verifier top-k token id")
        topk_mass = sum(math.exp(float(lp)) for lp in self.verifier_topk_logprobs)
        if topk_mass > 1.0 + 1e-4:
            raise DVIArtifactError("verifier top-k mass exceeds one")
        if not 0.0 <= self.verifier_residual_mass <= 1.0:
            raise DVIArtifactError("verifier residual mass must be in [0, 1]")
        if abs(self.verifier_residual_mass - max(0.0, 1.0 - topk_mass)) > 1e-3:
            raise DVIArtifactError(
                "verifier residual mass is inconsistent with top-k"
            )
        if self.accepted is not None and not isinstance(self.accepted, bool):
            raise DVIArtifactError("accepted must be bool or None")
        if self.accepted is not None and self.row_kind == "bonus":
            raise DVIArtifactError("bonus rows do not have proposal acceptance")
        if self.row_kind == "proposal":
            if self.row_index < self.accepted_count:
                expected_accepted: bool | None = True
            elif self.row_index == self.accepted_count:
                expected_accepted = False
            else:
                expected_accepted = None
            if self.accepted is not expected_accepted:
                raise DVIArtifactError(
                    "proposal accepted flag does not match accepted prefix semantics"
                )
        if not 0 <= self.accepted_count <= self.num_proposals:
            raise DVIArtifactError("accepted_count is outside proposal range")
        for token_id in self.committed_token_ids:
            if not isinstance(token_id, int) or not 0 <= token_id < manifest.vocab_size:
                raise DVIArtifactError("invalid committed token id")
        if self.row_index == 0:
            if not self.committed_token_ids:
                raise DVIArtifactError(
                    "row 0 must contain the committed token sequence"
                )
            if self.advancement != len(self.committed_token_ids):
                raise DVIArtifactError(
                    "advancement must equal committed_token_ids length"
                )
        elif (
            self.committed_token_ids
            or self.correction_token_id is not None
            or self.terminal_token_id is not None
            or self.advancement != 0
        ):
            raise DVIArtifactError(
                "cycle summary fields may only be populated on row 0"
            )
        if self.terminal_token_id is not None and not (
            isinstance(self.terminal_token_id, int)
            and 0 <= self.terminal_token_id < manifest.vocab_size
        ):
            raise DVIArtifactError("invalid terminal_token_id")
        if self.row_index == 0 and self.terminal_token_id != self.committed_token_ids[-1]:
            raise DVIArtifactError(
                "terminal_token_id must equal the final committed token"
            )
        if self.correction_token_id is not None and not (
            isinstance(self.correction_token_id, int)
            and 0 <= self.correction_token_id < manifest.vocab_size
        ):
            raise DVIArtifactError("invalid correction_token_id")
        if self.committed_token_ids:
            expected_advancement = self.accepted_count + 1
            if self.accepted_count == self.num_proposals:
                expected_advancement = self.num_proposals + 1
                if self.correction_token_id is not None:
                    raise DVIArtifactError(
                        "fully accepted cycles must not have a correction token"
                    )
            elif self.correction_token_id != self.committed_token_ids[-1]:
                raise DVIArtifactError(
                    "correction_token_id must be the committed token after rejection"
                )
            if self.advancement != expected_advancement:
                raise DVIArtifactError(
                    "advancement does not match stochastic prefix/bonus semantics"
                )
            if len(self.committed_token_ids) != expected_advancement:
                raise DVIArtifactError(
                    "committed token sequence length does not match advancement"
                )
        if self.stage_0_hidden is None:
            raise DVIArtifactError("evaluation stage_0_hidden must be provided")
        self.stage_0_hidden = self.stage_0_hidden.detach().cpu().contiguous()
        if self.stage_0_hidden.shape != (manifest.hidden_size,):
            raise DVIArtifactError(
                "evaluation hidden shape does not match manifest"
            )
        if str(self.stage_0_hidden.dtype) != f"torch.{manifest.dtype}":
            raise DVIArtifactError(
                "evaluation hidden dtype does not match manifest"
            )


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonl_dump(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _jsonl_load(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class DVIArtifactWriter:
    """Write a DVI artifact directory atomically."""

    def __init__(
        self,
        final_dir: str | Path,
        manifest: DVIArtifactManifest,
        shard_size: int = CAPTURE_SHARD_SIZE,
    ) -> None:
        self.final_dir = Path(final_dir)
        self.manifest = manifest
        self.manifest.validate()
        if shard_size <= 0:
            raise DVIArtifactError(f"shard_size must be positive, got {shard_size}")
        self.shard_size = shard_size

        self._temp_dir = Path(tempfile.mkdtemp(
            prefix=self.final_dir.name + ".",
            dir=str(self.final_dir.parent),
        ))
        self._request_records: list[DVIRequestRecord] = []
        self._capture_records: list[DVICaptureRecord] = []
        self._evaluation_records: list[DVIEvaluationRecord] = []
        self._eval_records: list[DVIEvalRecord] = []
        self._closed = False
        self._capture_shard_count = 0
        self._evaluation_shard_count = 0
        self._request_ids: set[str] = set()

    def _flush_capture_shard(self) -> None:
        if not self._capture_records:
            return
        shard_idx = self._capture_shard_count
        jsonl_path = self._temp_dir / f"capture-{shard_idx:05d}.jsonl"
        st_path = self._temp_dir / f"capture-{shard_idx:05d}.safetensors"

        tensor_dict: dict[str, torch.Tensor] = {}
        json_records: list[dict[str, Any]] = []
        for rec_idx, rec in enumerate(self._capture_records):
            key = f"cap-{shard_idx:05d}-{rec_idx:05d}-h"
            if rec.stage_0_hidden is None:
                raise DVIArtifactError("stage_0_hidden missing in capture record")
            tensor_dict[key] = rec.stage_0_hidden
            d = asdict(rec)
            d.pop("stage_0_hidden")
            d["stage_0_hidden_key"] = key
            json_records.append(d)

        _jsonl_dump(jsonl_path, json_records)
        save_safetensors(tensor_dict, str(st_path))
        self._capture_records = []
        self._capture_shard_count += 1

    def _flush_evaluation_shard(self) -> None:
        if not self._evaluation_records:
            return
        shard_idx = self._evaluation_shard_count
        jsonl_path = self._temp_dir / f"evaluation-{shard_idx:05d}.jsonl"
        st_path = self._temp_dir / f"evaluation-{shard_idx:05d}.safetensors"
        tensor_dict: dict[str, torch.Tensor] = {}
        json_records: list[dict[str, Any]] = []
        for rec_idx, rec in enumerate(self._evaluation_records):
            key = f"eval-{shard_idx:05d}-{rec_idx:05d}-h"
            if rec.stage_0_hidden is None:
                raise DVIArtifactError(
                    "stage_0_hidden missing in evaluation record"
                )
            tensor_dict[key] = rec.stage_0_hidden
            data = asdict(rec)
            data.pop("stage_0_hidden")
            data["stage_0_hidden_key"] = key
            json_records.append(data)
        _jsonl_dump(jsonl_path, json_records)
        save_safetensors(tensor_dict, str(st_path))
        self._evaluation_records = []
        self._evaluation_shard_count += 1

    def write_requests(self, requests: list[DVIRequestRecord]) -> None:
        if self._closed:
            raise DVIArtifactError("Writer is closed")
        for req in requests:
            req.validate(self.manifest)
            if req.request_id in self._request_ids:
                raise DVIArtifactError(
                    f"Duplicate request_id {req.request_id!r}"
                )
            self._request_ids.add(req.request_id)
        self._request_records.extend(requests)

    def write_capture_records(self, records: list[DVICaptureRecord]) -> None:
        if self._closed:
            raise DVIArtifactError("Writer is closed")
        request_index = {r.request_id: r for r in self._request_records}
        for rec in records:
            rec.validate(request_index, self.manifest)
        self._capture_records.extend(records)
        while len(self._capture_records) >= self.shard_size:
            batch = self._capture_records[: self.shard_size]
            self._capture_records = self._capture_records[self.shard_size :]
            self._flush_capture_shard_with_records(batch)

    def _flush_capture_shard_with_records(
        self, records: list[DVICaptureRecord]
    ) -> None:
        """Flush a specific batch without touching self._capture_records."""
        old = self._capture_records
        self._capture_records = records
        self._flush_capture_shard()
        self._capture_records = old

    def write_eval_records(self, records: list[DVIEvalRecord]) -> None:
        if self._closed:
            raise DVIArtifactError("Writer is closed")
        request_index = {r.request_id: r for r in self._request_records}
        for rec in records:
            rec.validate(request_index, self.manifest)
        self._eval_records.extend(records)

    def write_evaluation_records(
        self, records: list[DVIEvaluationRecord]
    ) -> None:
        if self._closed:
            raise DVIArtifactError("Writer is closed")
        if self.manifest.schema_version != "v2":
            raise DVIArtifactError(
                "evaluation records require an artifact schema v2 manifest"
            )
        request_index = {r.request_id: r for r in self._request_records}
        for rec in records:
            rec.validate(request_index, self.manifest)
        self._evaluation_records.extend(records)
        while len(self._evaluation_records) >= EVALUATION_SHARD_SIZE:
            batch = self._evaluation_records[:EVALUATION_SHARD_SIZE]
            self._evaluation_records = self._evaluation_records[EVALUATION_SHARD_SIZE:]
            old = self._evaluation_records
            self._evaluation_records = batch
            self._flush_evaluation_shard()
            self._evaluation_records = old

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._flush_evaluation_shard()
            self._flush_capture_shard()

            # Write requests.
            requests_path = self._temp_dir / "requests.jsonl"
            _jsonl_dump(requests_path, [asdict(r) for r in self._request_records])

            # Write eval shard.
            if self._eval_records:
                eval_path = self._temp_dir / "eval-00000.jsonl"
                _jsonl_dump(eval_path, [asdict(r) for r in self._eval_records])

            # Compute checksums and update manifest.
            file_checksums: dict[str, str] = {}
            for child in sorted(self._temp_dir.iterdir()):
                if child.is_file() and child.name != "manifest.json":
                    rel = child.name
                    file_checksums[rel] = _sha256_file(child)
            self.manifest.file_checksums = file_checksums

            manifest_path = self._temp_dir / "manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self.manifest), f, indent=2, ensure_ascii=False)

            # Atomic publish: refuse to overwrite an existing artifact.
            if self.final_dir.exists():
                raise DVIArtifactError(
                    f"Refusing to overwrite existing artifact directory: "
                    f"{self.final_dir}"
                )
            os.rename(self._temp_dir, self.final_dir)
        except Exception:
            if self._temp_dir.exists():
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._closed = True
            raise
        self._closed = True

    def __enter__(self) -> "DVIArtifactWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            if exc_type is None:
                self.close()
        finally:
            if not self._closed and self._temp_dir.exists():
                shutil.rmtree(self._temp_dir, ignore_errors=True)
                self._closed = True


class DVIArtifactReader:
    """Read and validate a DVI artifact directory."""

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        if not self.artifact_dir.is_dir():
            raise DVIArtifactError(
                f"Artifact directory not found: {self.artifact_dir}"
            )

        manifest_path = self.artifact_dir / "manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.manifest = DVIArtifactManifest(**data)
        self.manifest.validate()

        # Verify checksums first and detect extra/missing files.
        checksum_files = set(self.manifest.file_checksums.keys())
        actual_files = {
            p.name for p in self.artifact_dir.iterdir() if p.is_file()
        } - {"manifest.json"}
        missing = checksum_files - actual_files
        extra = actual_files - checksum_files
        if missing:
            raise DVIArtifactError(
                f"Missing files referenced by manifest: {sorted(missing)}"
            )
        if extra:
            raise DVIArtifactError(
                f"Unexpected files not listed in manifest checksums: {sorted(extra)}"
            )
        for rel_path, expected in self.manifest.file_checksums.items():
            file_path = self.artifact_dir / rel_path
            actual = _sha256_file(file_path)
            if actual != expected:
                raise DVIArtifactError(
                    f"Checksum mismatch for {rel_path}: expected {expected}, "
                    f"got {actual}"
                )
        if "requests.jsonl" not in checksum_files:
            raise DVIArtifactError(
                "requests.jsonl must be listed in file_checksums"
            )

        # Enforce regular files only and a strict file naming scheme.
        for child in self.artifact_dir.iterdir():
            if child.name == "manifest.json":
                continue
            if child.is_symlink() or child.is_dir():
                raise DVIArtifactError(
                    f"Artifact directory must contain only regular files; "
                    f"found {child.name}"
                )

        capture_jsonl: set[str] = set()
        evaluation_jsonl: set[str] = set()
        evaluation_st: set[str] = set()
        capture_st: set[str] = set()
        for name in checksum_files:
            if name == "requests.jsonl":
                continue
            if name.startswith("capture-"):
                if name.endswith(".jsonl"):
                    capture_jsonl.add(name)
                elif name.endswith(".safetensors"):
                    capture_st.add(name)
                else:
                    raise DVIArtifactError(
                        f"Illegal capture file name in manifest: {name}"
                    )
            elif name.startswith("evaluation-"):
                if name.endswith(".jsonl"):
                    evaluation_jsonl.add(name)
                elif name.endswith(".safetensors"):
                    evaluation_st.add(name)
            elif name.startswith("eval-") and name.endswith(".jsonl"):
                continue
            else:
                raise DVIArtifactError(
                    f"Illegal file listed in manifest checksums: {name}"
                )
        for name in capture_jsonl:
            pair = name.replace(".jsonl", ".safetensors")
            if pair not in capture_st:
                raise DVIArtifactError(
                    f"Missing safetensors pair for capture shard {name}"
                )
        for name in capture_st:
            pair = name.replace(".safetensors", ".jsonl")
            if pair not in capture_jsonl:
                raise DVIArtifactError(
                    f"Missing jsonl pair for capture shard {name}"
                )
        for name in evaluation_jsonl:
            pair = name.replace(".jsonl", ".safetensors")
            if pair not in evaluation_st:
                raise DVIArtifactError(
                    f"Missing safetensors pair for evaluation shard {name}"
                )
        for name in evaluation_st:
            pair = name.replace(".safetensors", ".jsonl")
            if pair not in evaluation_jsonl:
                raise DVIArtifactError(
                    f"Missing jsonl pair for evaluation shard {name}"
                )

        # Load requests.
        requests_path = self.artifact_dir / "requests.jsonl"
        raw_requests = _jsonl_load(requests_path)
        self.requests: list[DVIRequestRecord] = [
            DVIRequestRecord(**r) for r in raw_requests
        ]
        seen: set[str] = set()
        for req in self.requests:
            req.validate(self.manifest)
            if req.request_id in seen:
                raise DVIArtifactError(
                    f"Duplicate request_id {req.request_id!r} in requests.jsonl"
                )
            seen.add(req.request_id)
        self.request_index = {r.request_id: r for r in self.requests}

    def iter_capture_records(self) -> Iterator[DVICaptureRecord]:
        """Yield capture records one at a time, loading one shard at a time."""
        shard_names = sorted(
            name for name in self.manifest.file_checksums
            if name.startswith("capture-") and name.endswith(".jsonl")
        )
        for jsonl_name in shard_names:
            jsonl_path = self.artifact_dir / jsonl_name
            st_name = jsonl_name.replace(".jsonl", ".safetensors")
            st_path = self.artifact_dir / st_name
            if st_name not in self.manifest.file_checksums:
                raise DVIArtifactError(
                    f"Missing checksum entry for safetensors shard {st_name}"
                )
            tensors = load_safetensors(str(st_path))
            try:
                raw_records = _jsonl_load(jsonl_path)
                for raw in raw_records:
                    key = raw.pop("stage_0_hidden_key")
                    if key not in tensors:
                        raise DVIArtifactError(
                            f"Tensor key {key!r} not found in {st_name}"
                        )
                    raw["stage_0_hidden"] = tensors[key]
                    rec = DVICaptureRecord(**raw)
                    rec.validate(self.request_index, self.manifest)
                    yield rec
            finally:
                del tensors

    def iter_eval_records(self) -> Iterator[DVIEvalRecord]:
        """Yield eval records one at a time."""
        eval_names = sorted(
            name for name in self.manifest.file_checksums
            if name.startswith("eval-") and name.endswith(".jsonl")
        )
        for jsonl_name in eval_names:
            jsonl_path = self.artifact_dir / jsonl_name
            for raw in _jsonl_load(jsonl_path):
                rec = DVIEvalRecord(**raw)
                rec.validate(self.request_index, self.manifest)
                yield rec

    def iter_evaluation_records(self) -> Iterator[DVIEvaluationRecord]:
        """Yield dense v2 evaluation rows with their hidden tensors."""
        names = sorted(
            name for name in self.manifest.file_checksums
            if name.startswith("evaluation-") and name.endswith(".jsonl")
        )
        for jsonl_name in names:
            jsonl_path = self.artifact_dir / jsonl_name
            st_name = jsonl_name.replace(".jsonl", ".safetensors")
            st_path = self.artifact_dir / st_name
            tensors = load_safetensors(str(st_path))
            try:
                for raw in _jsonl_load(jsonl_path):
                    key = raw.pop("stage_0_hidden_key")
                    if key not in tensors:
                        raise DVIArtifactError(
                            f"Tensor key {key!r} not found in {st_name}"
                        )
                    raw["stage_0_hidden"] = tensors[key]
                    rec = DVIEvaluationRecord(**raw)
                    rec.validate(self.request_index, self.manifest)
                    yield rec
            finally:
                del tensors
