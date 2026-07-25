# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-DVI greedy block verifier (runs on split stage_2).

Semantics (greedy, no bonus token):

    target_ids[j] = argmax(logits of block row j)   for j in 0..L_r-1
    accepted      = longest prefix of draft_ids == target_ids
    accepted < draft_length: num_sampled = accepted + 1
                             sampled = target_ids[:accepted + 1]   (prefix + correction)
    accepted == draft_length == L_r: num_sampled = L_r
                             sampled = target_ids[:L_r]            (all drafts accepted)

Requests with ``draft_lengths[r] == 0`` (prefill rows) take the plain greedy
path: a single argmax token is committed.

Committing any prefix of ``target_ids`` is exactly what sequential greedy
decoding would produce, so correctness is independent of draft quality; the
drafts only affect how many tokens one cycle commits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SplitDVIVerificationResult:
    req_ids: list[str]
    sampled_token_ids: list[list[int]]
    accepted_counts: list[int]
    num_sampled: list[int]

    def to_padded_tensors(
        self,
        max_sample_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (sampled_token_ids [R, max_sample_len] int64,
        num_sampled [R] int32) in the V2 runner's native layout."""
        num_reqs = len(self.req_ids)
        sampled = torch.zeros(num_reqs, max_sample_len, dtype=torch.int64)
        for i, ids in enumerate(self.sampled_token_ids):
            assert len(ids) <= max_sample_len, (
                f"request {self.req_ids[i]!r} sampled {len(ids)} tokens > "
                f"max_sample_len={max_sample_len}"
            )
            sampled[i, : len(ids)] = torch.tensor(ids, dtype=torch.int64)
        num_sampled = torch.tensor(self.num_sampled, dtype=torch.int32)
        return sampled.to(device), num_sampled.to(device)


class SplitDVIGreedyBlockVerifier:
    """Stateless greedy block verifier."""

    def verify(
        self,
        target_logits: torch.Tensor,
        req_ids: list[str],
        draft_token_ids: list[int],
        draft_lengths: list[int],
        cu_num_logits: list[int],
    ) -> SplitDVIVerificationResult:
        """Verify one DVI block.

        ``target_logits``: ``[total_logits, vocab]`` for the block rows.
        ``draft_token_ids`` / ``draft_lengths``: flat per-request proposals.
        ``cu_num_logits``: ``[num_reqs + 1]`` logit-row boundaries per request
        (num_logits_r = number of block rows for request r).
        """
        num_reqs = len(req_ids)
        if len(cu_num_logits) != num_reqs + 1:
            raise ValueError(
                f"cu_num_logits has {len(cu_num_logits)} entries for "
                f"{num_reqs} requests"
            )
        if len(draft_lengths) != num_reqs:
            raise ValueError(
                f"draft_lengths has {len(draft_lengths)} entries for "
                f"{num_reqs} requests"
            )
        if sum(draft_lengths) != len(draft_token_ids):
            raise ValueError(
                f"draft_token_ids has {len(draft_token_ids)} entries, "
                f"draft_lengths sum to {sum(draft_lengths)}"
            )

        target_ids = target_logits.argmax(dim=-1).tolist()
        if len(target_ids) != cu_num_logits[-1]:
            raise ValueError(
                f"target_logits has {len(target_ids)} rows but cu_num_logits "
                f"ends at {cu_num_logits[-1]}"
            )

        sampled_token_ids: list[list[int]] = []
        accepted_counts: list[int] = []
        num_sampled: list[int] = []
        draft_offset = 0
        for r in range(num_reqs):
            row_start, row_end = cu_num_logits[r], cu_num_logits[r + 1]
            num_logits_r = row_end - row_start
            draft_len_r = draft_lengths[r]
            if num_logits_r < 1:
                raise ValueError(f"request {req_ids[r]!r} has zero logit rows")
            if draft_len_r > num_logits_r:
                raise ValueError(
                    f"request {req_ids[r]!r}: draft_length {draft_len_r} > "
                    f"num_logits {num_logits_r}"
                )
            row_targets = target_ids[row_start:row_end]
            proposals = draft_token_ids[draft_offset : draft_offset + draft_len_r]
            draft_offset += draft_len_r

            accepted = 0
            for proposal, target in zip(proposals, row_targets):
                if proposal != target:
                    break
                accepted += 1

            if draft_len_r == 0:
                # Prefill row: plain greedy single token.
                n_sampled = 1
            elif accepted == draft_len_r == num_logits_r:
                # All proposals accepted (no bonus token in v1).
                n_sampled = num_logits_r
            else:
                n_sampled = accepted + 1

            sampled_token_ids.append(row_targets[:n_sampled])
            accepted_counts.append(accepted)
            num_sampled.append(n_sampled)

        return SplitDVIVerificationResult(
            req_ids=list(req_ids),
            sampled_token_ids=sampled_token_ids,
            accepted_counts=accepted_counts,
            num_sampled=num_sampled,
        )
