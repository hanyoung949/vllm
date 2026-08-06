# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference stochastic sampler for Stage-DVI.

This module defines the sampling contract independently of the split runtime.
It intentionally favors explicit, testable semantics over kernel performance.

``P`` is the processed target-policy distribution and ``Q`` is the processed
draft distribution.  The draft distribution has bounded sparse support so a
split packet only needs to carry support token ids and probabilities, not a
full vocabulary-sized tensor.  For every proposed token ``x`` the verifier:

1. accepts with probability ``min(1, P(x) / Q(x))``;
2. on rejection, samples one correction from ``normalize(max(P - Q, 0))``;
3. stops the block after the correction;
4. records ``log P(token)`` for every committed token.

Production stochastic DVI uses the standard ``k - 1`` proposals plus one
target bonus row.  If every proposal is accepted, the verifier samples and
commits the bonus from the final target row.  The reference helper retains an
explicit no-bonus option for focused block tests.

Random draws are keyed by request seed, generation, absolute token position,
and draw kind.  Consequently batching and speculative block boundaries do not
change the logical random stream.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import torch

_MASK64 = (1 << 64) - 1
_MIX_GENERATION = 0x9E3779B97F4A7C15
_MIX_POSITION = 0xBF58476D1CE4E5B9
_MIX_DRAW_KIND = 0x94D049BB133111EB


class DVIDrawKind(enum.IntEnum):
    """Independent logical random streams used by stochastic DVI."""

    PROPOSAL = 1
    ACCEPTANCE = 2
    CORRECTION = 3
    TARGET_REFERENCE = 4


def _splitmix64(value: int) -> int:
    value = (value + _MIX_GENERATION) & _MASK64
    value = ((value ^ (value >> 30)) * _MIX_POSITION) & _MASK64
    value = ((value ^ (value >> 27)) * _MIX_DRAW_KIND) & _MASK64
    return value ^ (value >> 31)


def stateless_uniform(
    request_seed: int,
    generation_id: int,
    absolute_position: int,
    draw_kind: DVIDrawKind,
) -> float:
    """Return a deterministic float64 uniform in the open interval ``(0, 1)``.

    The function is the executable RNG contract for the reference sampler.
    A future GPU implementation must match these keys, though it may use a
    different documented counter-based transform if its tests are updated as
    an intentional contract change.
    """
    if generation_id < 0:
        raise ValueError("generation_id must be non-negative")
    if absolute_position < 0:
        raise ValueError("absolute_position must be non-negative")
    counter = request_seed & _MASK64
    counter ^= (generation_id * _MIX_GENERATION) & _MASK64
    counter ^= (absolute_position * _MIX_POSITION) & _MASK64
    counter ^= (int(draw_kind) * _MIX_DRAW_KIND) & _MASK64
    random_bits = _splitmix64(counter) >> 11
    return (random_bits + 0.5) / float(1 << 53)


@dataclass(frozen=True)
class SparseDraftDistribution:
    """A normalized, bounded-support draft proposal distribution."""

    token_ids: torch.Tensor
    probabilities: torch.Tensor

    def validate(self, vocab_size: int) -> None:
        if self.token_ids.ndim != 1 or self.probabilities.ndim != 1:
            raise ValueError("draft support token ids and probabilities must be 1-D")
        if self.token_ids.numel() == 0:
            raise ValueError("draft support must not be empty")
        if self.token_ids.shape != self.probabilities.shape:
            raise ValueError("draft support token ids/probabilities shape mismatch")
        if self.token_ids.dtype != torch.int64:
            raise ValueError("draft support token ids must use int64")
        if bool(((self.token_ids < 0) | (self.token_ids >= vocab_size)).any()):
            raise ValueError("draft support contains an out-of-vocabulary token")
        if torch.unique(self.token_ids).numel() != self.token_ids.numel():
            raise ValueError("draft support contains duplicate token ids")
        if not bool(torch.isfinite(self.probabilities).all()):
            raise ValueError("draft support contains non-finite probabilities")
        if bool((self.probabilities <= 0).any()):
            raise ValueError("draft support probabilities must be positive")
        total = float(self.probabilities.sum())
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(f"draft support probabilities sum to {total}, not 1")

    def probability(self, token_id: int) -> float:
        matches = self.token_ids == token_id
        if not bool(matches.any()):
            return 0.0
        return float(self.probabilities[matches][0])


@dataclass(frozen=True)
class DVIStochasticVerificationResult:
    """Committed prefix and target-policy metadata for one request block."""

    sampled_token_ids: list[int]
    target_logprobs: list[float]
    accepted_count: int
    stopped_by_rejection: bool

    @property
    def num_sampled(self) -> int:
        return len(self.sampled_token_ids)


def probabilities_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Apply temperature/top-k/top-p and return normalized probabilities.

    This is a reference implementation for plain stochastic sampling.  The
    initial runtime gate must continue to reject penalties, structured output,
    logit bias, and other history-dependent processors until the production
    path supplies identically processed ``P`` and ``Q`` distributions.
    """
    if logits.ndim != 1:
        raise ValueError(f"logits must be 1-D, got shape {tuple(logits.shape)}")
    if logits.numel() == 0:
        raise ValueError("logits must not be empty")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and greater than zero")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if not bool(torch.isfinite(logits).any()):
        raise ValueError("logits must contain at least one finite value")

    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    processed = logits.to(dtype=dtype) / temperature
    vocab_size = processed.numel()

    if 0 < top_k < vocab_size:
        keep = torch.topk(processed, top_k).indices
        remove = torch.ones_like(processed, dtype=torch.bool)
        remove[keep] = False
        processed = processed.masked_fill(remove, -torch.inf)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(processed, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative - sorted_probs >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        filtered = torch.full_like(processed, -torch.inf)
        filtered.scatter_(0, sorted_indices, sorted_logits)
        processed = filtered

    probabilities = torch.softmax(processed, dim=-1)
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("processed distribution contains non-finite probabilities")
    return probabilities


def make_sparse_draft_distribution(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float = 1.0,
) -> SparseDraftDistribution:
    """Create the bounded-support ``Q`` distribution sent with a proposal."""
    if top_k <= 0:
        raise ValueError("draft top_k must be positive to bound packet size")
    probabilities = probabilities_from_logits(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    token_ids = torch.nonzero(probabilities > 0, as_tuple=False).squeeze(-1)
    support_probs = probabilities[token_ids]
    support_probs = support_probs / support_probs.sum()
    result = SparseDraftDistribution(
        token_ids=token_ids.to(dtype=torch.int64),
        probabilities=support_probs,
    )
    result.validate(logits.numel())
    return result


def _sample_categorical(probabilities: torch.Tensor, uniform: float) -> int:
    if probabilities.ndim != 1 or probabilities.numel() == 0:
        raise ValueError("categorical probabilities must be a non-empty vector")
    if not 0 < uniform < 1:
        raise ValueError("categorical uniform must be in the open interval (0, 1)")
    cumulative = torch.cumsum(probabilities, dim=0)
    draw = torch.tensor(uniform, dtype=cumulative.dtype, device=cumulative.device)
    index = int(torch.searchsorted(cumulative, draw, right=False).item())
    return min(index, probabilities.numel() - 1)


def sample_sparse_draft_token(
    distribution: SparseDraftDistribution,
    *,
    request_seed: int,
    generation_id: int,
    absolute_position: int,
) -> int:
    """Sample one proposal token from sparse ``Q`` using the proposal stream."""
    distribution.validate(int(distribution.token_ids.max()) + 1)
    uniform = stateless_uniform(
        request_seed,
        generation_id,
        absolute_position,
        DVIDrawKind.PROPOSAL,
    )
    support_index = _sample_categorical(distribution.probabilities, uniform)
    return int(distribution.token_ids[support_index])


def verify_stochastic_block(
    target_logits: torch.Tensor,
    draft_token_ids: list[int],
    draft_distributions: list[SparseDraftDistribution],
    absolute_positions: list[int],
    *,
    request_seed: int,
    generation_id: int,
    target_temperature: float,
    target_top_k: int = 0,
    target_top_p: float = 1.0,
    bonus_token: bool = False,
) -> DVIStochasticVerificationResult:
    """Verify one proposed path with exact stochastic accept/correction."""
    if target_logits.ndim != 2:
        raise ValueError("target_logits must have shape [block_size, vocab_size]")
    block_size, vocab_size = target_logits.shape
    if block_size == 0:
        raise ValueError("stochastic block must contain at least one row")
    proposal_count = len(draft_token_ids)
    if not (
        proposal_count
        == len(draft_distributions)
        == len(absolute_positions)
    ):
        raise ValueError("draft proposal metadata lengths must match")
    expected_rows = proposal_count + int(bonus_token)
    if block_size != expected_rows:
        raise ValueError(
            "target block size must equal proposal count plus the optional "
            f"bonus row, got {block_size} and {proposal_count}"
        )
    if bonus_token and proposal_count == 0:
        raise ValueError("bonus-token verification requires at least one proposal")

    committed: list[int] = []
    target_logprobs: list[float] = []
    accepted_count = 0
    stopped_by_rejection = False

    for row, (proposal, draft, position) in enumerate(
        zip(draft_token_ids, draft_distributions, absolute_positions)
    ):
        draft.validate(vocab_size)
        if not 0 <= proposal < vocab_size:
            raise ValueError(f"proposal token {proposal} is out of vocabulary")
        q_proposal = draft.probability(proposal)
        if q_proposal <= 0:
            raise ValueError("proposal token is absent from its draft distribution")

        target_probs = probabilities_from_logits(
            target_logits[row],
            temperature=target_temperature,
            top_k=target_top_k,
            top_p=target_top_p,
        )
        p_proposal = float(target_probs[proposal])
        acceptance_probability = min(1.0, p_proposal / q_proposal)
        acceptance_uniform = stateless_uniform(
            request_seed,
            generation_id,
            position,
            DVIDrawKind.ACCEPTANCE,
        )

        if acceptance_uniform < acceptance_probability:
            token_id = proposal
            accepted_count += 1
        else:
            residual = target_probs.clone()
            support_ids = draft.token_ids.to(device=residual.device)
            residual[support_ids] -= draft.probabilities.to(
                device=residual.device, dtype=residual.dtype
            )
            residual.clamp_(min=0)
            residual_mass = float(residual.sum())
            if residual_mass <= 0 or not math.isfinite(residual_mass):
                raise RuntimeError("rejected proposal has no valid residual mass")
            residual /= residual_mass
            correction_uniform = stateless_uniform(
                request_seed,
                generation_id,
                position,
                DVIDrawKind.CORRECTION,
            )
            token_id = _sample_categorical(residual, correction_uniform)
            stopped_by_rejection = True

        token_probability = float(target_probs[token_id])
        if token_probability <= 0:
            raise RuntimeError("stochastic verifier selected a zero-probability token")
        committed.append(token_id)
        target_logprobs.append(math.log(token_probability))
        if stopped_by_rejection:
            break

    if bonus_token and not stopped_by_rejection:
        bonus_position = absolute_positions[-1] + 1
        target_probs = probabilities_from_logits(
            target_logits[-1],
            temperature=target_temperature,
            top_k=target_top_k,
            top_p=target_top_p,
        )
        bonus_uniform = stateless_uniform(
            request_seed,
            generation_id,
            bonus_position,
            DVIDrawKind.TARGET_REFERENCE,
        )
        token_id = _sample_categorical(target_probs, bonus_uniform)
        token_probability = float(target_probs[token_id])
        if token_probability <= 0:
            raise RuntimeError("stochastic verifier selected a zero-probability bonus")
        committed.append(token_id)
        target_logprobs.append(math.log(token_probability))

    return DVIStochasticVerificationResult(
        sampled_token_ids=committed,
        target_logprobs=target_logprobs,
        accepted_count=accepted_count,
        stopped_by_rejection=stopped_by_rejection,
    )
