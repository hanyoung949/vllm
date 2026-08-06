# Stage-DVI Stochastic Sampling Contract

## Scope

This contract is the correctness boundary between Stage-DVI and an RL
rollout. The existing greedy verifier remains unchanged until this stochastic
path passes its unit, runtime, and one-step GRPO gates.

The initial stochastic mode supports temperature, top-k, and top-p sampling.
Penalties, structured output, allowed-token masks, logit bias, beam search,
and multi-output requests remain fail-closed until both target and draft paths
apply them identically.

## Distribution Authority

- `pi_theta` is the raw full-model policy.
- `P_theta` is the target distribution after all supported sampling transforms.
- `Q_phi` is the external draft distribution after its sampling transforms.
- Tokens committed by DVI must be distributed exactly as `P_theta`.
- Every rollout logprob is `log P_theta(token | committed_history)`.

Draft probabilities, acceptance probabilities, and residual-correction
probabilities are never written as GRPO policy logprobs.

## Sparse Draft Distribution

Stage 0 constructs a strictly bounded top-k `Q_phi` for each proposed token.
It samples the proposal from that sparse distribution and sends:

- proposal token id;
- support token ids;
- processed support logits (after draft temperature);
- absolute token position.

Stage 2 normalizes the transmitted support logits. Tokens outside the support
have exactly zero draft probability.
This makes correction exact without transferring a vocabulary-sized draft
tensor. The support size must never exceed the configured draft top-k, even
when logits tie.

## Acceptance And Correction

For proposal `x` at a position:

```text
accept_probability = min(1, P_theta(x) / Q_phi(x))
```

If accepted, commit `x` and continue. If rejected, sample one token from:

```text
normalize(max(P_theta - Q_phi, 0))
```

Commit the correction and stop the block. Production stochastic DVI uses a
block of `k - 1` draft proposals plus one target-only bonus row. If every
proposal is accepted, sample and commit the bonus from that final target row.
Thus each cycle commits between one and `k` tokens while preserving the target
autoregressive distribution.

## RNG Contract

Every logical draw is keyed by:

```text
(request_seed, generation_id, absolute_position, draw_kind)
```

Draw kinds are proposal, acceptance, correction, and sequential-target
reference. Batch order, batch size, speculative block size, and fallback
boundaries must not change a request's logical random stream.

Bitwise equality with ordinary rollout sampling is not required. Distribution
equality with the target policy is required. Tests that compare baseline and
DVI training must therefore compare token/logprob/reward/advantage
distributions rather than token hashes.

## Proposed Packet Fields

The stochastic packet extends existing request-major DVI metadata with:

```text
sampling_mode = "stochastic_v1"
draft_token_ids
draft_support_offsets
draft_support_token_ids
draft_support_logits
draft_positions
generation_ids
policy_version
draft_version
```

The bounded support arrays should travel as tensors in the split data plane,
not as Python lists in host metadata. Metadata carries shapes, versions, and
protocol identity only.

## Delivery Gates

1. Reference sampler: exact mixture checks, no-bonus focused blocks, standard
   `k - 1` proposal plus target-bonus blocks, RNG partition invariance,
   target-logprob checks, and Monte Carlo distribution checks.
2. Split runtime: sequential-reference GPU smoke, correction/KV rollback,
   mixed requests, and target logprob side output.
3. GRPO vertical slice: baseline versus DVI one-step token, logprob, reward,
   advantage, and policy-update distributions on GPU 2/3/4.
4. Maintenance: only after gate 3, add live capture, warm candidates,
   validation-gated promotion, and amortized wall-clock accounting.
