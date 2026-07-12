---
title: Flash Attention
date: 2026-07-04
---

Standard attention isn't slow because of the math — it's slow because it writes a
giant intermediate matrix to memory and reads it back. FlashAttention computes
*exactly* the same result without ever materializing that matrix, making it
faster and linear in memory.

## Attention is memory-bound

For queries, keys, and values $Q, K, V \in \mathbb{R}^{N \times d}$, attention is:

$$S = \frac{QK^\top}{\sqrt d}, \qquad O = \mathrm{softmax}(S)\,V$$

The scores $S$ and probabilities $P = \mathrm{softmax}(S)$ are $N \times N$. The
naive kernel writes $S$ to HBM, reads it back to softmax it, writes $P$, then
reads it again for $PV$ — three round-trips of an $N^2$ matrix.

The catch is that a GPU's arithmetic throughput dwarfs its memory bandwidth (on
an A100, ~19 TFLOP/s vs ~1.5 TB/s). So those $N^2$ HBM round-trips, not the
FLOPs, are the bottleneck. Kill them and you win.

## Online softmax

The obstacle: softmax needs a whole row of $S$ at once to normalize. Online
softmax removes that constraint by computing a numerically-stable softmax in one
streaming pass, keeping a running max $m$ and running denominator $\ell$.

When a new block with local max $\tilde m$ arrives, the max may grow. Everything
accumulated under the old $m$ is rescaled by $e^{m - m'}$ to stay consistent:

$$m' = \max(m, \tilde m), \qquad \ell' = e^{m - m'}\,\ell + \textstyle\sum e^{x - m'}$$

That single correction factor is the whole trick — subtracting a constant from
every exponent just multiplies the accumulator, so past work is fixed up without
revisiting it.

## The tiled algorithm

FlashAttention loads blocks of $Q, K, V$ into fast SRAM and, for each query
block, streams over all key/value blocks while maintaining $m$, $\ell$, and an
unnormalized output $O$:

```
for each query block Q_i:
    m ← -∞ ; ℓ ← 0 ; O ← 0
    for each key/value block K_j, V_j:
        S_ij  = Q_i K_jᵀ / √d
        m'    = max(m, rowmax(S_ij))
        P_ij  = exp(S_ij - m')
        α     = exp(m - m')            # rescale old state
        ℓ     = α·ℓ + rowsum(P_ij)
        O     = α·O + P_ij V_j
        m     = m'
    O_i = O / ℓ                        # normalize once, at the end
```

The full $N \times N$ matrix is never allocated: each tile $S_{ij}$ lives in SRAM
and is discarded after it's merged. Activation memory drops from $O(N^2)$ to
$O(N)$, and HBM traffic from $O(N^2)$ to $O(N^2/M)$ for SRAM size $M$.

## Exact, and trainable

This is not an approximation. The rescaling $e^{m - m'}$ is an algebraic
identity, so the final $O$ equals $\mathrm{softmax}(QK^\top/\sqrt d)\,V$ to
floating-point rounding — a free drop-in with no retraining.

The backward pass needs $P$, which we refused to store. Instead of writing it out,
FlashAttention keeps only $O$ and the per-row stats $(m, \ell)$ — all $O(N)$ — and
**recomputes** each $S_{ij}$ tile on the fly. Since the kernel is memory-bound,
those extra FLOPs are nearly free.

## Results

| Metric               | Standard  | FlashAttention |
|----------------------|-----------|----------------|
| Activation memory    | $O(N^2)$  | $O(N)$         |
| HBM accesses         | $O(N^2)$  | $O(N^2/M)$     |
| Exact?               | yes       | yes            |
| Wall-clock speedup   | 1x        | 2–4x           |

The linear memory matters even more than the speed: it's what makes long context
windows feasible at all. **FlashAttention-2** improved GPU utilization for
another ~2x; **FlashAttention-3** added Hopper-specific overlap and FP8 — but the
core idea never changes: keep the $N \times N$ matrix off HBM, softmax online,
stay exact.
