# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for Stage-DVI stochastic sampling."""

from __future__ import annotations

import math

import torch

from vllm.v1.worker.gpu.split_dvi.stochastic_sampler import (
    DVIDrawKind,
    make_sparse_draft_distribution,
    probabilities_from_logits,
    sample_sparse_draft_token,
    stateless_uniform,
    verify_stochastic_block,
)


def test_rng_is_independent_of_batch_and_block_boundaries():
    positions = list(range(20, 28))
    whole = [
        stateless_uniform(7, 3, p, DVIDrawKind.ACCEPTANCE) for p in positions
    ]
    split = [
        stateless_uniform(7, 3, p, DVIDrawKind.ACCEPTANCE)
        for chunk in (positions[:3], positions[3:])
        for p in chunk
    ]
    assert split == whole
    assert whole != [
        stateless_uniform(7, 3, p, DVIDrawKind.CORRECTION) for p in positions
    ]


def test_verifier_output_is_invariant_to_request_order_and_batch_partition():
    target = torch.log(
        torch.tensor(
            [
                [0.52, 0.28, 0.15, 0.05],
                [0.15, 0.55, 0.20, 0.10],
                [0.10, 0.20, 0.60, 0.10],
                [0.35, 0.15, 0.10, 0.40],
            ],
            dtype=torch.float64,
        )
    )
    draft_values = torch.tensor(
        [0.10, 0.60, 0.20, 0.10], dtype=torch.float64
    )
    draft = make_sparse_draft_distribution(
        torch.log(draft_values), temperature=1.0, top_k=4
    )

    def verify(request_seed: int):
        positions = [31, 32, 33]
        proposals = [
            sample_sparse_draft_token(
                draft,
                request_seed=request_seed,
                generation_id=2,
                absolute_position=position,
            )
            for position in positions
        ]
        return verify_stochastic_block(
            target,
            proposals,
            [draft] * 3,
            positions,
            request_seed=request_seed,
            generation_id=2,
            target_temperature=1.0,
            bonus_token=True,
        )

    seeds = [101, 202, 303, 404, 505]
    whole_batch = {seed: verify(seed) for seed in seeds}
    partitioned = {
        seed: verify(seed)
        for batch in (seeds[:2], seeds[2:4], seeds[4:])
        for seed in reversed(batch)
    }
    assert partitioned == whole_batch


def test_top_k_top_p_distribution_is_normalized_and_bounded():
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64)
    probs = probabilities_from_logits(
        logits, temperature=0.7, top_k=3, top_p=0.8
    )
    assert math.isclose(float(probs.sum()), 1.0, abs_tol=1e-12)
    assert torch.count_nonzero(probs).item() <= 3
    assert probs[-1] == 0


def test_top_k_is_strictly_bounded_when_logits_tie():
    probs = probabilities_from_logits(
        torch.zeros(8, dtype=torch.float64),
        temperature=1.0,
        top_k=3,
    )
    assert torch.count_nonzero(probs).item() == 3


def test_accept_reject_mixture_equals_target_distribution_exactly():
    target = torch.tensor([0.52, 0.28, 0.15, 0.05], dtype=torch.float64)
    draft = torch.tensor([0.10, 0.60, 0.20, 0.10], dtype=torch.float64)
    accepted_mass = torch.minimum(target, draft)
    rejection_mass = 1.0 - accepted_mass.sum()
    residual = torch.clamp(target - draft, min=0)
    residual /= residual.sum()
    output = accepted_mass + rejection_mass * residual
    torch.testing.assert_close(output, target, rtol=0, atol=1e-12)


def test_block_size_one_records_target_not_draft_logprob():
    target = torch.log(torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64))
    draft_logits = torch.log(torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64))
    draft = make_sparse_draft_distribution(
        draft_logits, temperature=1.0, top_k=3
    )
    proposal = sample_sparse_draft_token(
        draft, request_seed=11, generation_id=0, absolute_position=5
    )
    result = verify_stochastic_block(
        target,
        [proposal],
        [draft],
        [5],
        request_seed=11,
        generation_id=0,
        target_temperature=1.0,
    )
    token_id = result.sampled_token_ids[0]
    assert result.num_sampled == 1
    assert math.isclose(
        result.target_logprobs[0], math.log([0.6, 0.3, 0.1][token_id])
    )


def test_forced_rejection_uses_positive_residual_and_stops_block():
    # Q puts excess mass on token 0. Any rejection must correct to token 1.
    target = torch.log(
        torch.tensor(
            [[0.2, 0.8], [0.2, 0.8], [0.2, 0.8], [0.2, 0.8]],
            dtype=torch.float64,
        )
    )
    draft_logits = torch.log(torch.tensor([0.9, 0.1], dtype=torch.float64))
    draft = make_sparse_draft_distribution(
        draft_logits, temperature=1.0, top_k=2
    )

    # Find a position whose proposal is token 0 and acceptance uniform exceeds
    # P(0)/Q(0), making the rejection deterministic for this contract test.
    position = next(
        p
        for p in range(1000)
        if sample_sparse_draft_token(
            draft, request_seed=19, generation_id=0, absolute_position=p
        )
        == 0
        and stateless_uniform(19, 0, p, DVIDrawKind.ACCEPTANCE) > 0.2 / 0.9
    )
    result = verify_stochastic_block(
        target,
        [0, 0, 0, 0],
        [draft] * 4,
        [position, position + 1, position + 2, position + 3],
        request_seed=19,
        generation_id=0,
        target_temperature=1.0,
    )
    assert result.sampled_token_ids == [1]
    assert result.accepted_count == 0
    assert result.stopped_by_rejection
    assert math.isclose(result.target_logprobs[0], math.log(0.8))


def test_all_accepted_block_four_has_no_bonus_token():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64))
    draft = make_sparse_draft_distribution(logits, temperature=1.0, top_k=3)
    positions = [40, 41, 42, 43]
    proposals = [
        sample_sparse_draft_token(
            draft, request_seed=23, generation_id=1, absolute_position=p
        )
        for p in positions
    ]
    result = verify_stochastic_block(
        logits.repeat(4, 1),
        proposals,
        [draft] * 4,
        positions,
        request_seed=23,
        generation_id=1,
        target_temperature=1.0,
    )
    assert result.sampled_token_ids == proposals
    assert result.accepted_count == 4
    assert result.num_sampled == 4
    assert not result.stopped_by_rejection


def test_all_accepted_three_proposals_commits_target_bonus():
    values = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    logits = torch.log(values)
    draft = make_sparse_draft_distribution(logits, temperature=1.0, top_k=3)
    positions = [40, 41, 42]
    proposals = [
        sample_sparse_draft_token(
            draft, request_seed=29, generation_id=2, absolute_position=p
        )
        for p in positions
    ]
    result = verify_stochastic_block(
        logits.repeat(4, 1),
        proposals,
        [draft] * 3,
        positions,
        request_seed=29,
        generation_id=2,
        target_temperature=1.0,
        bonus_token=True,
    )
    bonus_uniform = stateless_uniform(
        29, 2, 43, DVIDrawKind.TARGET_REFERENCE
    )
    expected_bonus = int(
        torch.searchsorted(torch.cumsum(values, dim=0), bonus_uniform).item()
    )
    assert result.sampled_token_ids == proposals + [expected_bonus]
    assert result.accepted_count == 3
    assert result.num_sampled == 4
    assert not result.stopped_by_rejection
    assert math.isclose(result.target_logprobs[-1], math.log(values[expected_bonus]))


def test_single_position_monte_carlo_matches_target_distribution():
    target_values = torch.tensor([0.52, 0.28, 0.15, 0.05], dtype=torch.float64)
    draft_values = torch.tensor([0.10, 0.60, 0.20, 0.10], dtype=torch.float64)
    target_logits = torch.log(target_values).unsqueeze(0)
    draft = make_sparse_draft_distribution(
        torch.log(draft_values), temperature=1.0, top_k=4
    )
    counts = torch.zeros(4, dtype=torch.int64)
    num_trials = 20_000
    for request_seed in range(num_trials):
        proposal = sample_sparse_draft_token(
            draft,
            request_seed=request_seed,
            generation_id=0,
            absolute_position=0,
        )
        result = verify_stochastic_block(
            target_logits,
            [proposal],
            [draft],
            [0],
            request_seed=request_seed,
            generation_id=0,
            target_temperature=1.0,
        )
        counts[result.sampled_token_ids[0]] += 1

    observed = counts.to(torch.float64) / num_trials
    total_variation = 0.5 * torch.abs(observed - target_values).sum()
    assert total_variation < 0.015, (observed, target_values)



def test_multi_position_and_bonus_monte_carlo_match_target_distributions():
    target_values = torch.tensor(
        [
            [0.52, 0.28, 0.15, 0.05],
            [0.15, 0.55, 0.20, 0.10],
            [0.10, 0.20, 0.60, 0.10],
            [0.35, 0.15, 0.10, 0.40],
        ],
        dtype=torch.float64,
    )
    draft_values = torch.tensor(
        [0.10, 0.60, 0.20, 0.10], dtype=torch.float64
    )
    target_logits = torch.log(target_values)
    draft = make_sparse_draft_distribution(
        torch.log(draft_values), temperature=1.0, top_k=4
    )
    counts = torch.zeros((4, 4), dtype=torch.int64)
    observations = torch.zeros(4, dtype=torch.int64)
    num_trials = 30_000

    for request_seed in range(num_trials):
        positions = [0, 1, 2]
        proposals = [
            sample_sparse_draft_token(
                draft,
                request_seed=request_seed,
                generation_id=0,
                absolute_position=position,
            )
            for position in positions
        ]
        result = verify_stochastic_block(
            target_logits,
            proposals,
            [draft] * 3,
            positions,
            request_seed=request_seed,
            generation_id=0,
            target_temperature=1.0,
            bonus_token=True,
        )
        for position, token_id in enumerate(result.sampled_token_ids):
            counts[position, token_id] += 1
            observations[position] += 1

    assert observations[0] == num_trials
    for position in range(4):
        assert observations[position] > 500
        observed = counts[position].to(torch.float64) / observations[position]
        total_variation = 0.5 * torch.abs(
            observed - target_values[position]
        ).sum()
        assert total_variation < 0.035, (
            position,
            observations[position],
            observed,
            target_values[position],
        )
