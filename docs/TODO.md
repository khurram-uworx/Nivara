# Plan: BFloat16 transformer inference demo (NivaraInference sample)

## Problem

Nivara now supports `BFloat16` across the **AutoDiff domain** (issue #137, merged
to `main` after `v1.4.0`) and the **column/query layer** (POLARS-ROADMAP Phase 2,
this branch — see `docs/BFLOAT16.md`). DistilBERT / MiniLM are the canonical BF16
transformer workloads, yet the `NivaraInference` sample:

1. Hard-codes every model to `float` (`DistilBertForSequenceClassification<float>`,
   `Module<float>`, `Linear<float>`, …).
2. Its README still claims the SafeTensors loader "throws `NotSupportedException`
   for non-F32 tensors (F16, BF16)" — **stale**: the core
   `Nivara.Samples.SafeTensorsLoader` has supported F16/BF16 since SAFETENSORS
   Phase 4 (one day after the README was written).

So we are *capable* of BF16 inference but not *demonstrating* it — and DistilBERT
is exactly the BF16 poster child. This plan closes that gap.

## Goal

Demonstrate BFloat16 inference end-to-end on the three text models
(DistilBERT base, DistilBERT SST-2 fine-tuned, MiniLM) and quantify it against
the **existing F32 HuggingFace reference fixtures** already in `samples/data/`:

- **Precision** — run in `BFloat16`, diff against the F32 reference
  (max-abs-diff, mean-abs-diff, argmax agreement, cosine). BERT-family is famously
  BF16-robust, so expect tiny diffs — a credible, reproducible proof point.
- **Memory** — BF16 halves weight memory (DistilBERT 255 MB → ~128 MB).
- **Speed** — explicitly NOT a win claim: on .NET CPU, BFloat16 matmul runs the
  scalar `TensorPrimitives.Dot` path (per `docs/TENSORS.md`: "correct but not
  hardware-accelerated"). Frame any timing as parity, never a speedup.

## Grounding (microsoft-learn)

`System.Numerics.BFloat16` is a first-class .NET 11 type in `System.Runtime.dll`
implementing the full `IFloatingPointIeee754<BFloat16>` interface (trig/hyperbolic/
root/exp functions, explicit conversions, `BinaryPrimitives.ReadBFloat16LittleEndian`).
This confirms the `TypeValidator` admission and the generic-kernel strategy used
across AutoDiff and the column/query layer.

## Approach — sample-only, no engine changes

The engine already supports BFloat16:

- `SafeTensorsLoader.Read<T>(path)` is generic — `Read<BFloat16>` widens the F32
  weights on disk to `BFloat16` (F32→BF16 truncation, simulating a BF16-distributed
  checkpoint).
- `DistilBertForSequenceClassification<T>.LoadWeights<TWeight>` is generic over the
  weight element type, so `LoadWeights<BFloat16>` builds `BFloat16` parameters
  directly.
- Every transformer module (`Linear<T>`, `LayerNorm<T>`, `BertSelfAttention<T>`,
  `GeluExact`) is generic over `IFloatingPointIeee754<T>`.
- `GradientUtils.Constant<T>(T[])` is generic — `Constant<BFloat16>(BFloat16[])`
  builds BFloat16 input tensors from token ids / attention masks.

Add a `bf16` run mode for the three text models that:
1. Loads weights as BFloat16: `SafeTensorsLoader.Read<BFloat16>(modelPath)`.
2. Builds the `<BFloat16>` model + `LoadWeights<BFloat16>`.
3. Runs the existing compare sentences through inference in BFloat16.
4. Widens the BFloat16 output to `float32` and diffs it against the existing F32
   PyTorch reference fixture (reusing `DistilBertSst.PrintCompareDiff` machinery).

## Files

- `samples/NivaraInference/DistilBertSst.cs`
  - `LoadBFloat16(Dictionary<string,(BFloat16[] Data,int[] Shape)> tensors, string modelDir)`
    → `DistilBertForSequenceClassification<BFloat16>`.
  - `PredictLogitsBFloat16(...)` → `ReverseGradTensor<BFloat16>` (mirror `PredictLogits`).
  - `SaveBFloat16CompareOutput(...)` → mirrors `SaveCompareOutput` but runs the
    `<BFloat16>` model and widens logits to `float[]` when serializing (so it drops
    into the existing `.bin` compare format). Add `using System.Numerics;`.
- `samples/NivaraInference/Program.cs`
  - Add `using System.Numerics;`.
  - `Main`: add `bf16` mode; when set, load `SafeTensorsLoader.Read<BFloat16>(modelPath)`
    and dispatch to the BFloat16 run methods.
  - `RunDistilBertSstBFloat16(...)`, `RunDistilBertBFloat16(...)`,
    `RunMiniLMBFloat16(...)` — mirror the existing `Inference`/`Compare` methods but
    with `<BFloat16>` and a diff against the F32 reference fixture.
- `samples/NivaraInference/README.md`
  - Fix the stale "throws for non-F32 (F16, BF16)" claim (loader supports F16/BF16).
  - Add a "BFloat16 inference" subsection: the `bf16` mode for the three text
    models, measured precision (max-abs-diff / argmax agreement) + memory, and a
    cross-link to `docs/BFLOAT16.md`.

## Blast radius — UPDATED during execution

Sample changes as planned, **plus one engine change** that the demo exposed:

- **Engine fix (committed):** BFloat16/Half transformer inference was actually
  *broken* — `Embedding<T>.Forward(ReverseGradTensor<T>)` recovered token IDs via
  `int.CreateChecked(input.Data[i])`, but BF16 cannot represent vocab indices
  (~30k) exactly (exact only to 256), so the embedding lookup fetched wrong rows
  and output diverged catastrophically from F32 (DistilBERT SST-2 max logit diff
  ~7.4 vs F32's ~1e-6). Fixed by adding exact-`int[]` `Forward(int[])` overloads to
  `Embedding<T>`, `BertEncoder<T>`, `MiniLMDistilled<T>` and
  `DistilBertForSequenceClassification<T>`; the existing `ReverseGradTensor<T>`
  overloads are retained (F32/F64 unaffected), so it is non-breaking.
  Commits: `baac328` (fix), `301e5ef` (regression test).
- **Sample:** `DistilBertSst.cs` + `Program.cs` `bf16` mode now passes token IDs as
  exact `int[]` (not `BFloat16` tensors); the attention mask stays a `BFloat16`
  tensor (its 0/1 values round-trip exactly).

## Verification

- `dotnet build samples/NivaraInference -c Release` (compile check after each commit).
- `dotnet run --project samples/NivaraInference -c Release -- distilbert_sst bf16`
  (and `distilbert bf16`, `minilm bf16`) — confirm argmax agreement with the F32
  reference and small max-abs-diff; capture the numbers for the README.
- The sample has no unit-test coverage; verification is build + manual run.

## Planned commits

1. `docs: plan BFloat16 inference demo in TODO.md`
2. `feat(samples): BFloat16 load + compare for DistilBERT SST-2`
3. `feat(samples): BFloat16 inference + compare for base DistilBERT`
4. `feat(samples): BFloat16 inference + compare for MiniLM`
5. `docs: fix stale BF16 loader claim + add BFloat16 inference section (README)`
6. `docs: remove TODO.md — BFloat16 inference demo executed`

## GitHub issues log

- Engine bug found & fixed in-branch (no separate issue needed): BFloat16/Half
  transformer token-ID corruption in `Embedding<T>` embedding lookups.
  Commits `baac328` (fix) + `301e5ef` (regression test `EmbeddingBFloat16Tests`).
