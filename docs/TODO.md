# Plan: #492 — reconcile the Laya CPU forward with its attention leg profile

Issue: https://github.com/khurram-uworx/Nivara/issues/492

Branch: `khurram/492` (confirmed).

This branch measures and records. It does not change a kernel in `src/`. If a measurement
confirms a kernel defect, file it and stop — a fix is a separate decision, not part of this plan.

## Problem

`docs/LAYA.md` records a ~4.1 s Laya CPU forward (`laya benchmark`, four questions padded to
`max_len` 512, median 3910–4243 ms). The `--mask` leg profile, scaled by 28 layers, put
attention alone at 9.1–13.4 s. Those cannot both be the cost of one forward, and until they
are reconciled no percentage-of-forward from that harness is defensible. `docs/LAYA.md` §3
excludes attention from the CPU-vs-GPU decision on a MAC-share argument (0.13%). MAC share is
not time share.

The issue's "11× `MatMulTransposedB` vs `MatMul` layout gap" was already retracted by
`76c795fc`: the isolated-leg design timed softmax on a converged buffer and timed `QK^T`
cache-hot. The corrected gap is ~1.8×, and the README forbids citing the 11× figure. The
reconciliation itself was explicitly left open. That is what this branch closes.

## What is already wrong, before any new probe

1. **The leg profile does not call the matmul the way production does.**
   `ApplyMaskProbe.LegChain.Execute` passes the full multi-head buffers (`heads * rows * headDim`)
   into `GradKernels.MatMulTransposedB` / `MatMul` while declaring one head's dimensions.
   Production (`ReverseGradOperations.MultiHeadAttention`, ~line 570) slices `qh` / `kh` / `vh`
   per head first. `TensorsHelper.MultiplyCoreFloat` rents, copies and `clearArray`-returns by
   `b.Length` and `a.Length`, so the probe moves 16× the buffer production moves, then the
   accounting multiplies that one call by 16 heads. The published per-call milliseconds are not
   a production head. Fix this before trusting any reconciliation number, and retract the
   milliseconds those calls produced. Do not replace them until a corrected run exists.

2. **The profile's mask is not the forward's mask.** `--mask` builds `MaskRegime.Causal`
   (upper triangle `-inf`, every row has a finite diagonal). `laya benchmark` calls
   `ModernBertEncoder.Forward(ids, validLength)` with valid lengths 39–119 padded to 512.
   `ModernBertMasks.Build` suppresses every key at or past `validLength`, and 18 of 28 layers
   are sliding (`local_attention` 128 → band 64). A padding query row is entirely `-inf`, and
   `GradKernels.SoftmaxSingle` returns at `max == -inf` without `Exp`. The profile has no such
   row, so it times the slow softmax on all 512 rows. A faithful mirror has to run the
   production mask (full band and sliding band, at the recorded valid lengths and at
   `validLength == 512`) or the comparison is two different workloads.

3. **The residual ~1.8× is not yet shown to be layout.** `QK^T` is 512×64→512×512 and `PV` is
   512×512→512×64: same MAC count, 8× more output elements and 8× shorter reductions on the
   `QK^T` side. `--cpu-gemm` already measures the pure layout effect at GEMM shapes, and there
   `MatMulTransposedB` is the *faster* leg (2414 vs 2791 ms), because `MatMul` pays a per-call
   transpose. Settle it with a same-run 4-cell matrix at the attention shapes, not by citing
   either recorded ratio.

## Probe lifecycle

Checked before writing anything new:

- `tests/Nivara.PerformanceTests`: `--mask` (the leg profile, #480), `--cpu-gemm` (layout at
  GEMM shapes), `--gemm-legs` (GPU leg split, #440). None times a CPU forward and its legs in
  one process.
- `tests/Nivara.SimdProbe`: kernel-vs-BCL probes. Not a forward.

No temp harness. The slice fix lands in `--mask`. The reconciliation and the 4-cell matrix
land as a new `--laya-forward` mode in `tests/Nivara.PerformanceTests`, which is the home for
one-off A/B flags. `docs/LAYA.md` is not edited until three runs of that mode exist.

## Proposed changes

### 1. Fix the `--mask` leg profile's spans

In `LegChain.Execute`, slice per head exactly as `MultiHeadAttention` does, and time the head
loop rather than one call times 16. `PackHeads` stays once per layer.

```csharp
for (int h = 0; h < heads; h++)
{
    var qh = qHeads.AsSpan(h * qLen * headDim, qLen * headDim);
    var kh = kHeads.AsSpan(h * kvLen * headDim, kvLen * headDim);
    var vh = vHeads.AsSpan(h * kvLen * headDim, kvLen * headDim);
    // time MatMulTransposedB(qh, kh, ...), scale, mask, softmax, MatMul(scores, vh, ...), scatter
}
```

Guard: if a timed span's length is not `qLen * headDim` / `kvLen * headDim`, the run prints
VOID and does not print a number. The multiplier table stops inferring 16 calls from one.

The README's published per-call table (7.74 / 4.31 and the two later runs) was measured on
the oversized spans. Retract those milliseconds in the same commit, with the reason. Do not
invent replacements. The absolute mask delta (96–133 ms/forward) was measured on `scores`,
which is correctly sized, so it is not retracted by this fix; say that explicitly so a reader
does not throw out the one number the fix does not touch.

### 2. `--laya-forward` mode

New file `tests/Nivara.PerformanceTests/LayaForwardProbe.cs`, wired like `--mask` in
`Program.ParseArgs` (early return, mutually exclusive with the other modes). Probe, not a
gate: exit 0 on a healthy run, asserts only structural guards (span lengths, finite hidden
state, checkpoint-or-synthetic label). Prints its own coverage.

Three blocks, one process, so the ratio carries no cross-run load:

**A. Forward total.** Resolve `samples/data/laya/model.safetensors` from cwd, walking up.
Present on this machine (842 MB). Load via `SafeTensorsLoader.Read` and
`ModernBertEncoder<float>.LoadWeights(..., prefix: "encoder")`, then `Eval()` is already
called by `LoadWeights`. Time `Forward(ids, validLength)` at seq 512 for the recorded valid
lengths (40, 119, 48, 39) plus 512. Protocol matches `laya benchmark`: 1 untimed warmup, 3
timed, median. Also time `LayaDecisionHead<float>` on one question so encoder and head are
separate lines — the recorded 4.1 s is encoder-plus-head, and the reconciliation has to say
which piece it is comparing.

Ids: random in `[0, vocab)` with the checkpoint's pad id past `validLength`. Values do not
change the GEMM shape; they do change score magnitude. Real weights are what makes that the
recorded workload.

Synthetic fallback, only if the file is absent: `new ModernBertEncoder<float>(config)`
(Kaiming-initialized `Linear` weights, so the matmul is not a zero-skip). Printed as
`SYNTHETIC` on every line, and **never** used as the denominator against the 4.1 s figure.
A missing checkpoint is not a failed reconciliation; it is an unmeasured forward, labeled.

**B. Same-run leg profile**, after the slice fix, under the masks production actually builds:

| cell | band | validLength | why |
|---|---|---|---|
| causal (today's profile) | n/a, upper triangle | 512 | the number the README currently quotes, so the before/after of the slice fix is visible |
| full attention | -1 | 512 | a global layer with no padding |
| full attention | -1 | 40 and 119 | the recorded questions' padding |
| sliding | 64 | 40 and 512 | 18 of 28 layers |

Build the mask with `ModernBertMasks.Build<float>` and time `ApplyMask` / `SoftmaxRows` on
that buffer, not on `MaskRegime.Causal`. Count fully-masked rows and print them, so an
early-out is evidence rather than a story.

Roll the layer mix up (10 full + 18 sliding, from `samples/data/laya/encoder/config.json`)
at validLength 40 and at 512, and print `attention roll-up` next to `measured forward`.
The roll-up is a lower bound on attention only; it is not a forward. Print the residual
(`forward − attention roll-up`) and do not call it GEMM unless block A and a GEMM projection
were timed in this same run. They are not. Do not divide by 2414 or by 4100.

**C. Layout matrix.** Four cells, interleaved, same buffers, median of 30 after warmup:

| shape | `bTransposed` |
|---|---|
| QK^T 512×64×512 | true (B is `[512 × 64]`, no transpose) |
| QK^T 512×64×512 | false (B is `[64 × 512]`, transposed per call) |
| PV 512×512×64 | true |
| PV 512×512×64 | false |

Spans are exactly `aRows * aCols` and `aCols * bCols`. A `QK^T`/`PV` ratio that survives
the transposed-B flag being held constant is a shape gap. A ratio that appears only when the
flag flips is the per-call transpose. Print both ratios and do not name either "layout"
unless the flag is what moved it.

**Diagnostics pass, optional and labeled.** One extra forward with
`DiagnosticsTracker.IsEnabled` and `AutoDiffDiagnostics.Enabled`, grouped by `OperationType`.
Shares only. `DiagnosticsTracker` takes a lock per op, so its durations are not the
denominator and are not printed as one.

**What the mode refuses to print.** A percentage of the recorded 4.1 s or of the recorded
2414 ms. It prints this run's forward, this run's attention roll-up, and the residual. If
the hidden state is non-finite, the numeric block is VOID on its own path.

### 3. Record only what three runs support

Run `--laya-forward` three times, same power state, Release. Then:

- `tests/Nivara.PerformanceTests/README.md`: document `--laya-forward`, replace the retracted
  per-call table with the corrected runs, and state whether attention-alone still exceeds the
  same-run forward. If the three runs disagree by more than the noise the mode itself prints,
  publish the band and not a point.
- `docs/LAYA.md`: the issue forbids editing it on one probe run in either direction. Touch it
  only after the three runs, and only the sentences the runs actually bear on — the 4.1 s vs
  2414 ms "not a contradiction" paragraph (line 129) and the §3 MAC-share exclusion (line 184).
  If the runs do not close the gap, the edit is a pointer to the unresolved band, not a new
  percentage. Do not restate #440, #447 or #458.

## Verification

- `dotnet build tests/Nivara.PerformanceTests/Nivara.PerformanceTests.csproj -c Release` before
  each commit that touches the harness. Debug numbers are void; every probe run is `-c Release`.
- Reproduce first, on this branch, before changing the probe: `laya benchmark` and `--mask`,
  so the "before" is this machine and not the 2026-10-01 README.
- After the slice fix: `--mask` once, to confirm the guard holds and to see the direction of
  the per-call move. Not a published baseline by itself.
- After `--laya-forward` exists: three Release runs. That is the result.
- No `dotnet test` of the full suite. This branch does not change `src/`. A targeted run is
  not useful unless a later step adds an NUnit test, in which case ask first.
- Power state: print it (the harness already warns on battery in the GEMM modes). Do not
  compare a battery figure to the AC 4.1 s as if they were the same measurement.

## Planned commits

1. `docs: plan #492 reconciliation in TODO.md` — this file.
2. `fix(probe): time --mask attention matmuls on per-head slices` — the span fix, the VOID
   guard, and the retraction of the milliseconds those calls produced. No replacement numbers.
3. `test: add --laya-forward same-run reconciliation` — the mode, its README section (what it
   measures and what it refuses to print; no results yet).
4. `docs: record the #492 reconciliation` — README numbers and the `docs/LAYA.md` sentences
   the three runs support. Skipped, not emptied, if the runs were not made.

## Blast radius

| change | affects | callers | tests |
|---|---|---|---|
| `LegChain.Execute` span fix | `--mask` leg-profile numbers only | `ApplyMaskProbe.Run`, manual | none in NUnit; the in-probe length guard is the pin |
| `--laya-forward` flag | `Program.ParseArgs` early-return chain | manual invocation | none; a probe, not a gate |
| README / LAYA.md | recorded claims | readers of the decision paragraph | the three runs are the evidence |

No symbol in `src/Nivara` or `src/Nivara.Extensions` changes. `ModernBertEncoder.Forward`,
`ModernBertMasks.Build`, `ReverseGradOperations.MultiHeadAttention` and
`TensorsHelper.MultiplyCore` are read, not edited. Downstream of those symbols is therefore
untouched.

## GitHub issues log

- [ ] #473 — CPU attention is still dense (`ModernBertMasks.MaxDenseLength`); referenced, not
  duplicated, if the roll-up shows the full `QK^T` dominating a sliding layer. Do not file a
  second issue for that.

Created during execution (none yet):

<!-- append here as `gh issue create` happens; do not hold deferred work in memory -->
