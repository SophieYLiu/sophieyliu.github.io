---
title: Flash Attention
date: 2026-07-04
---

Attention is the workhorse of the Transformer, but the textbook implementation
is quietly wasteful. It spends most of its time not on math, but on shuffling a
giant intermediate matrix back and forth to memory. FlashAttention is the
observation that if you never materialize that matrix, you can compute *exactly*
the same result far faster and with linear memory. This post walks through why
standard attention is slow, the online-softmax trick that makes the fix
possible, the tiled algorithm itself, and why the answer it produces is
bit-for-bit the real thing rather than an approximation.

## The operation we're computing

Given queries `Q`, keys `K`, and values `V`, each a matrix of shape `[N, d]`
(sequence length `N`, head dimension `d`), attention is:

```
S = Q Kᵀ / √d        # scores,        [N, N]
P = softmax(S)        # probabilities, [N, N]   (softmax over each row)
O = P V               # output,        [N, d]
```

The whole thing is three matrix operations. The trouble is the middle two: `S`
and `P` are `N × N`. For a sequence of 8K tokens that's 64M entries *per head,
per layer* — hundreds of megabytes that exist only to be immediately consumed by
the next step.

## Why standard attention is slow: it's memory-bound

Here's the part that's easy to miss. A modern GPU can do far more arithmetic per
second than it can move bytes to and from its main memory (HBM). The A100, for
example, does ~19 TFLOP/s of FP16 matmul but only ~1.5 TB/s of HBM bandwidth —
roughly a 10:1 ratio. On-chip SRAM is ~10x faster than HBM but tiny (tens of KB
per streaming multiprocessor).

Now trace what the standard implementation actually does with memory:

1. Compute `S = QKᵀ`, **write** all `N²` scores to HBM.
2. **Read** `S` back, compute `softmax`, **write** `P` (`N²`) back to HBM.
3. **Read** `P` and `V`, compute `PV`, **write** `O`.

The `softmax` step in particular is nearly pure data movement: for every one of
the `N²` elements it reads a float and writes a float, doing a trivial amount of
arithmetic in between. The GPU's expensive matmul units sit idle while the chip
waits on memory. Attention is **memory-bound**, and the thing it's bound on is
reading and writing that `N × N` matrix.

> The key realization: the FLOPs aren't the problem. The `N²` round-trips to HBM
> are. If we never write `S` or `P` to HBM at all, we remove the bottleneck.

But softmax seems to *need* the whole row of `S` at once — you can't normalize
until you've seen every score. That's the knot online softmax unties.

## The enabling trick: online (streaming) softmax

Recall the numerically-stable softmax over a vector `x`. You subtract the max
before exponentiating so nothing overflows:

```
m = max(x)
softmax(x)_i = exp(x_i - m) / Σ_j exp(x_j - m)
```

This looks like it requires two full passes over `x`: one to find `m`, one to
sum. Online softmax computes it in a **single streaming pass**, updating a
running max and a running sum as each new element (or block) arrives.

Suppose we've processed some elements and hold `m` (max so far) and `l` (sum of
`exp(x - m)` so far). A new block of scores arrives with local max `m_blk`. The
running max becomes `m_new = max(m, m_blk)`. The catch: our accumulated `l` was
computed relative to the *old* `m`. To keep it consistent with `m_new`, we
rescale it by `exp(m - m_new)`:

```
m_new = max(m, m_blk)
l_new = exp(m - m_new) · l  +  Σ exp(x_blk - m_new)
```

That correction factor `exp(m - m_new)` is the whole idea. Because subtracting a
constant from every exponent multiplies the whole sum by a constant, we can fix
up work we already did without revisiting it. When the max grows, everything
computed under the old max gets down-weighted by exactly the right amount.

The same correction extends to the output accumulator, and that's what lets us
fold the `PV` matmul into the same streaming loop.

## Tiling: computing attention block by block

FlashAttention splits `Q`, `K`, `V` into blocks that fit in fast SRAM, and
computes the output one query-block at a time. For a fixed block of queries
`Q_i`, it streams over all key/value blocks `K_j, V_j`, maintaining three
running quantities per query row:

- `m` — the running row-max of the scores seen so far
- `l` — the running denominator (sum of exponentials)
- `O` — the running, unnormalized weighted sum of value vectors

For each key/value block `j` we do a local attention computation and merge it
into the accumulators using the online-softmax correction:

```
S_ij   = Q_i K_jᵀ / √d                 # this block's scores  [Br, Bc]
m_blk  = rowmax(S_ij)
m_new  = max(m, m_blk)

P_ij   = exp(S_ij - m_new)             # unnormalized probs for this block
l      = exp(m - m_new) · l  +  rowsum(P_ij)
O      = exp(m - m_new) · O  +  P_ij V_j    # rescale old output, add new
m      = m_new
```

After the last key block, normalize once: `O_i = O / l`. That single division
turns the accumulated unnormalized sum into the true softmax-weighted output.

Notice what never happened: we never allocated an `N × N` matrix. `S_ij` is only
`Br × Bc` (a small tile), lives in SRAM, and is discarded after it's merged.
`Q_i`, `K_j`, `V_j`, and the output block are the only things touched in HBM, and
each is read a bounded number of times. Memory traffic drops from `O(N²)` to
`O(N²/M)` HBM accesses (where `M` is the SRAM size), and the memory *footprint*
for activations drops from `O(N²)` to `O(N)`.

## The full algorithm

Putting the two loops together:

```
for each block of queries  Q_i:                 # outer loop over rows
    m ← -∞ ; l ← 0 ; O ← 0                       # init accumulators (in SRAM)
    for each block of keys/values  K_j, V_j:     # inner loop, streams over cols
        S_ij  = Q_i K_jᵀ / √d
        m_new = max(m, rowmax(S_ij))
        P_ij  = exp(S_ij - m_new)
        α     = exp(m - m_new)                   # rescale factor for old state
        l     = α · l + rowsum(P_ij)
        O     = α · O + P_ij V_j
        m     = m_new
    O_i = O / l                                  # normalize once, at the end
    write O_i to HBM
```

Everything inside the inner loop stays in SRAM. The clever ordering — one kernel,
fused matmul → softmax → matmul, with the softmax computed incrementally — is why
it's a single fast pass instead of three memory-bound ones.

## It's exact, not an approximation

This is worth stressing, because "faster attention" often means "approximate
attention" (low-rank, sparse, or kernelized variants that trade accuracy for
speed). FlashAttention does **not**. Online softmax is an algebraic identity, not
an approximation: rescaling by `exp(m_old - m_new)` reconstructs precisely the
sum you'd have gotten from a single global max. The final `O_i` is the same
`softmax(QKᵀ/√d)V` you'd compute the slow way, down to floating-point rounding
(and even the rounding is comparable, since the same values get summed).

So FlashAttention is a pure systems win: identical numerics, dramatically less
memory traffic. You can drop it into an existing model with no retraining and no
accuracy change.

## The backward pass: recomputation beats storage

Training needs gradients, and the backward pass needs `P` — the very `N × N`
matrix we refused to store. Writing it out would reintroduce the `O(N²)` memory
we just eliminated.

FlashAttention's answer is **recomputation**: during the forward pass it saves
only the output `O` and the per-row softmax statistics (`m` and `l`), which are
`O(N)`. In the backward pass it recomputes each `S_ij` tile on the fly from `Q`
and `K`, using the saved statistics to reconstruct `P_ij` exactly. This trades a
modest amount of extra arithmetic for a huge memory saving — and since the kernel
is memory-bound anyway, the recomputed FLOPs are nearly free. It's the classic
gradient-checkpointing trade-off, applied at exactly the right granularity.

## What you get

The headline results from the original work:

| Metric                     | Standard attention | FlashAttention |
|----------------------------|--------------------|----------------|
| Activation memory          | `O(N²)`            | `O(N)`         |
| HBM accesses               | `O(N²)`            | `O(N²/M)`      |
| Exact result?              | yes                | yes            |
| Typical wall-clock speedup | 1x                 | 2–4x           |

The linear memory is arguably the bigger deal than the speed: it's what makes
long context windows practical at all. Storing the full score matrix for a
64K-token sequence is hopeless; streaming over it in SRAM tiles is routine. A
large fraction of the "long context" progress in modern LLMs rests on this.

## Beyond v1

The idea kept improving. **FlashAttention-2** reworked the loop ordering and
work partitioning to keep the GPU's matmul units busier and cut non-matmul
FLOPs, roughly doubling throughput again. **FlashAttention-3** specialized for
Hopper (H100) hardware, overlapping computation with data movement and
exploiting FP8. The through-line is unchanged: keep the `N × N` matrix off HBM,
compute softmax online, and stay exact.

## Takeaways

- Standard attention is **memory-bound**, not compute-bound — its cost is the
  `N²` round-trips of the score matrix to HBM, not the arithmetic.
- **Online softmax** computes a numerically-stable softmax in one streaming pass
  by rescaling running accumulators with `exp(m_old - m_new)`.
- **Tiling** fuses the two matmuls and the softmax into a single SRAM-resident
  kernel, dropping activation memory from `O(N²)` to `O(N)`.
- The result is **exact**, so it's a free swap — and **recomputation** carries
  the same trick into the backward pass.
