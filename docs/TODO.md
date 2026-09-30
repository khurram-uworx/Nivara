# Plan: #454 — fully-masked attention row under `finfo.min`: backward diverges from PyTorch, `dk`/`dv` contaminated

Tracking issue: [#454](https://github.com/khurram-uworx/Nivara/issues/454)

## Problem

Under HuggingFace's `torch.finfo(dtype).min` mask convention, a **fully-masked** query row
saturates in the forward: at float32 the ULP at 3.4e38 is ~2e31, so `score + finfo.min`
collapses every cell of the row to one bit-identical constant. The row softmax therefore
becomes a uniform `1/L` average of V, and `dq` for that row is meaningless.

The issue records the divergence but stops at "three candidate answers, no verified ground
truth". Grounding resolved the ground truth:

**None of the three is correct. The exact derivative is zero.**

| tensor | `-inf` vs PyTorch | `finfo.min` vs PyTorch |
|---|---|---|
| forward, all rows | exact | exact |
| `dq` rows 0..6 | exact | exact |
| `dq` row 7 (fully masked) | exact (both zero) | PyTorch x8 |
| `dk`, all rows | exact | differs (max 6.1) |
| `dv`, all rows | exact | differs (max 2.5) |

### Finding 1 — the row's output is locally constant, so `dq[7]` is exactly 0

All 8 post-mask scores in row 7 are bit-identical to `finfo.min` (verified, head 0):
`[-3.402823e+38 x8]`. The forward output of that row is `mean(V)` and does not depend on
`q_7` at all. Central finite differences in float64 — the test that distinguishes a
*locally constant* function from a merely *small-gradient* one — return bit-identical losses:

```
finfo(f64).min           FD dq[7] head0 = [0. 0. 0. 0.]
finfo(f32).min in f64    FD dq[7] head0 = [0. 0. 0. 0.]
moderate -1e9            FD dq[7] head0 = [-0.021992 -0.092760 -0.886148 -0.176155]
no mask                  FD dq[7] head0 = [-0.087649 -0.073051 -0.829177 -0.235991]
torch fixture                        = [-7.761303  2.982784  0.828112 -0.568959]
```

The `-1e9` row is the load-bearing contrast: a **non**-saturating finite fill lets the
additive constant cancel inside the softmax, so the row is the ordinary unmasked attention
row and has a real gradient. The answer therefore depends on fill magnitude versus
precision — and `finfo.min` is by construction always in the saturating regime, so
`dq[7] = 0` holds for every `finfo.min` fill, at any precision.

### Finding 2 — the 8x is a missing `1/L`, not a difference of principle

Writing `a_j = dO_7·V_j`, `A = Σa_j`, `s = Σ a_j K_j`, `T = Σ K_j`:

- Nivara: `scale·(1/8)·(s − (A/8)·T)` — softmax-backward with the **normalized** `P = 1/8`
- PyTorch: `scale·(s − (A/8)·T)` — softmax-backward with **unnormalized** `P = 1`

i.e. PyTorch computes `dS = dP − mean(dP)` where Nivara computes `P·(dP − dot)`. The 8 is
just the row's `1/L`, dropped on one side and not the other. Verified component-wise
against the fixture. **Neither is right**: both apply the softmax VJP as though the
pre-softmax score still depended on `q_7`, and under a saturating fill it does not.

This makes the fixtures unusable as a parity target, and it means "match PyTorch" was
never an available resolution — PyTorch's `dq[7]` is 8x away from PyTorch's own exact
derivative.

### Finding 3 — exact closed forms for `dk` and `dv`

Independent of which candidate you pick, and **bit-exact** in float32:

- `dq[7] = 0`
- `dk == dk(-inf)` — row 7's `dS` row is zero, and rows 0..6 have identical `P` under both fills
- `dv == dv(-inf) + dout[7]/L` on every key (per head) — the V path still sees `p·dO_7`

Measured by replicating the op in numpy (validated against the existing `-inf` fixture to
5e-7) and applying the row-zeroing:

```
saturated rows detected (head 0)                : [F F F F F F F T]   (same under -inf)
dq[7] exactly zero                               : True
dk_fixed vs -inf fixture dk   -> BIT-EXACT       : max|diff| = 0.0
dv_fixed vs (-inf dv + dout[7]/8) -> max|diff|   : 0.0
```

Bit-exactness is not luck: the arithmetic is the same kernels in the same order, and
`P = 0` on the `-inf` row already made the cleared `dS` a no-op there. So the corrected
`dk` is bit-for-bit the existing `-inf` fixture, and the tests can be zero-tolerance gates
rather than tolerance bands.

## Proposed changes

### 1. Kernels — `src/Nivara/AutoDiff/Operations/AttentionKernels.cs`

`ApplyMask` is the only place that sees both the pre-mask score and the fill, and it
already computes `score + mask`, so the annihilation test costs one comparison:

```csharp
public static void ApplyMask(Span<T> scores, ReadOnlySpan<T> mask,
                             Span<bool> scoreIndependentRows, int rows, int cols)
```

- A cell is **dead** (its value no longer reflects the score) iff `mask != 0 && (score + mask) == mask`.
- A row is flagged iff it contains no live cell.
- `-inf` cells satisfy the same test (`-inf == -inf`), so no special case.
- `NaN` mask cell: `NaN == m` is false → live → row not flagged. Correct.
- The existing 2-argument overload delegates with `Span<bool>.Empty`, so all 18 existing
  `ApplyMask` tests and both existing call sites keep their current contract.

`SoftmaxBackwardRows` gains the matching optional span; a flagged row's `dS` is cleared
after the per-row dot is computed:

```csharp
public static void SoftmaxBackwardRows(ReadOnlySpan<T> weights, Span<T> dS, int rows, int cols,
                                       ReadOnlySpan<bool> scoreIndependentRows = default)
```

**One kernel change covers both AD directions**: the forward-mode JVP at
`ForwardGradOperations.cs:1682` calls the same method, and clearing `tScores` there also
kills the `vTan` term at `:1689`.

### 2. Call sites — 4 forward/backward/JVP pairs

| Site | Forward (`ApplyMask`) | Consume (`SoftmaxBackwardRows`) |
|---|---|---|
| `ReverseGradOperations.cs` `MultiHeadAttention<T>` | :579 | :637 |
| `ReverseGradOperations.cs` `BatchedMultiHeadAttention<T>` | :821 | :917 |
| `ForwardGradOperations.cs` `MultiHeadAttention<T>` | :1617 | :1682 |
| `ForwardGradOperations.cs` `BatchedMultiHeadAttention<T>` | same shape | same shape |

Saved state is `bool[numHeads * qLen]`, allocated only under the existing `shouldTrack` /
`trackTangent` gate beside `savedWeights`. **Per head**: a score can be annihilated in one
head and not another, so the flag is not head-invariant. Inference pays nothing.

`MultiheadAttention`, `LlamaCausalAttention`, `ModernBertAttention` (`Nn/`) need no change —
they route through the reverse op. The GPU `AttentionKernels.BatchedAttention` and
`DecodeAttention` are forward-only and have no gradient path.

### 3. Tests

- `tests/Nivara.Tests/AutoDiff/AttentionKernelsTests.cs` — kernel level: the new overload
  against an annihilating fill, an ordinary mask, a mixed row, `-inf`, and a NaN cell; and
  that `SoftmaxBackwardRows` zeroes a flagged row.
- `tests/Nivara.Tests/NivaraTorch/BandedAttentionTests.cs` — replace
  `BandedAttention_FinfoMinFill_Backward_FiniteButDivergentFromPyTorch` with the three
  exact assertions at `absTol: 0, relTol: 0`: `dq[7]` exactly zero, `dk` bit-for-bit the
  `-inf` fixture, `dv` bit-for-bit `-inf dv + dout[7]/8`. Keep the `dq` rows 0-6 and
  finite-ness assertions unchanged.
- `tests/Nivara.Tests/AutoDiff/ForwardGradOperationsTests.cs` — a saturated row contributes
  an exactly-zero tangent.
- `tests/Nivara.Tests/AutoDiff/BatchedMultiHeadAttentionTests.cs` — the same closed forms per
  batch element, so the batched path is not silently exempt.

The removed test asserted `differs == true` against PyTorch. It would have kept passing
after the fix (`0` vs `-7.76`) — it pinned the defect rather than the contract.

### 4. Fixture provenance — keep the `.bin` files, reclassify

`attn_band_padding_minfill_{dq,dk,dv}.bin` stay on disk. The reason they are not a parity
target is recorded in:

- `samples/data/torch-comparison/manifest.json` (`attn_band_padding_minfill` entry)
- `samples/NivaraTorch/README.md` (:397, :418)
- `samples/NivaraTorch/gen_reference.py` (:1681-1688)
- the mask-contract note in `CHANGELOG.md`, and the #481 group in `AttentionMaskTests.cs`

### 5. Regression gates built into the change

- The `-inf` path must stay **bit-for-bit**: the detector flags the same row under both
  conventions and `P = 0` there already, so clearing is a no-op.
  `BandedAttention_NegInfFill_ForwardAndBackward_MatchPyTorch` and
  `BatchedMultiHeadAttention_Backward_SelfAttention_MatchesPerSequenceGradients` therefore
  become zero-tolerance proofs that the fix is inert on the shipped convention.
- No timing assertions introduced, so no `[Category("Performance")]` concern.
- No `dotnet test` inside the change itself.

## Verification steps

1. `dotnet build Nivara.slnx` after each commit.
2. `dotnet test -c Release` (ask first) on the targeted fixtures:
   `BandedAttentionTests`, `AttentionKernelsTests`, `AttentionMaskTests`,
   `BatchedMultiHeadAttentionTests`, `ForwardGradOperationsTests`.
3. `dotnet test -c Release --filter "Category!=Performance"` for the full suite.
4. Confirm the `-inf` fixtures are unchanged byte-for-byte (`git diff --stat` over
   `samples/data/torch-comparison` must be empty).

## Planned commits

1. `docs: plan #454 saturated attention row gradient in TODO.md`
2. `feat: report score-independent attention rows from ApplyMask` — kernel overloads + unit tests
3. `fix: zero the score gradient of a saturated attention row` — 4 call sites + exactness tests
4. `docs: record #454 as the non-parity reason for the minfill backward fixtures`

## Blast radius

**Narrow, and inert on the shipped convention.**

- `AttentionKernels<T>.ApplyMask` / `SoftmaxBackwardRows` — 2 overloads added, existing
  signatures delegate unchanged. 18 direct `ApplyMask` tests unaffected.
- `ReverseGradOperations.MultiHeadAttention<T>` / `BatchedMultiHeadAttention<T>` and the
  two `ForwardGradOperations` twins — forward, backward, and JVP paths. All existing
  attention parity fixtures use benign masks (no saturated row) and must be unaffected.
- Downstream consumers of those two ops, all of which inherit the fix without edits:
  `Nn/MultiheadAttention`, `Nn/LlamaCausalAttention`, `Nn/ModuleHelpers.CreateBlockDiagonalMask`,
  `samples/Nivara.Samples` (BertModel, DistilBertModel, ModernBertModel, LlamaForCausalLM).
  ModernBERT is inference-only and uses `-inf`, so inert.
- `savedWeights` grows by `bool[numHeads * qLen]` **only when tracking**. Not on the
  inference path.
- Test surface: `BandedAttentionTests` (one test replaced by three), plus additions to
  `AttentionKernelsTests`, `AttentionMaskTests`, `BatchedMultiHeadAttentionTests`,
  `ForwardGradOperationsTests`.
- Documentation/manifest: fixture provenance, CHANGELOG, `samples/NivaraTorch/README.md`.

## GitHub issues log

- [ ] #489 — partially-saturated mask row: annihilated cells still receive a spurious `dS`
  (created while planning #454). A per-row flag cannot express per-cell death, so this needs
  per-cell data and a fixture with a fill that kills some cells in a row and not others.
  Explicitly **out of scope here** — same root cause, different fix shape.
