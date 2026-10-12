# Plan: extend `--silk` from MiniLM to the DistilBERT family

Branch: `khurram/silk-distilbert`

> **Process reminder.** As each task executes, if deferred work or a concern surfaces (a limitation,
> follow-up, refactor, or red flag), create a GitHub issue immediately with
> `gh issue create --repo khurram-uworx/Nivara` and record its number in the **GitHub issues log**
> at the bottom of this file. Do not rely on memory or wait until the plan finishes — compaction
> during execution can lose it.

## Problem

`--silk` (the hand-authored OpenCL C backend on `Silk.NET.OpenCL`, `samples/Nivara.Samples/Gpu/`)
is MiniLM-only. The encoder runner `SilkBertEncoderRunner` hardcodes `BertGpuNaming.Bert` in two
places, and it rejects any checkpoint that ships classification-head weights
(`pre_classifier`/`classifier`), because the six-kernel set carries no gather and no ReLU GEMM
epilogue.

The ILGPU backend (`BertEncoderGpuRunner`) is already naming-driven and runs both DistilBERT and
DistilBERT SST-2 through `BertGpuNaming.DistilBert`. The Silk runner is the only piece that does not
generalize. DistilBERT is a genuine extension of the frontier:

| Property | MiniLM | DistilBERT |
|---|---|---|
| Weight keys | BERT-style (`embeddings.*`) | `distilbert.*` |
| Hidden / intermediate | 384 / 1536 | 768 / 3072 |
| headDim | 32 | **64 == `SilkKernels.MaxHeadDim`** (the boundary) |
| Token-type embedding | present | absent |
| Head | none (CLS pooling on host) | `pre_classifier → ReLU → classifier` |

## Decisions (confirmed with the human before implementation)

1. **Scope: both stages.** Ship `distilbert --silk` (encoder) and `distilbert_sst --silk` (head).
2. **Head implementation: device kernels.** Add `gather_rows` (CLS extraction, exact copy) and
   `gemm_row4_relu` (third epilogue over the shared `GEMM_ROW4_ACCUMULATE` macro). This keeps the
   runner device-pure and the byte-identity gate meaningful. The host-round-trip alternative was
   rejected (adds a sync, breaks the "same code, different `Forward`" seam).

## Proposed changes

### Stage 1 — `distilbert --silk` (encoder)

`samples/Nivara.Samples/Gpu/SilkBertEncoderRunner.cs`
- Add a `BertGpuNaming naming` ctor parameter; store it in a field; use it at both sites that
  currently hardcode `BertGpuNaming.Bert` (ctor line ~88 and `UploadLayer` line ~273).
- Keep the head rejection for Stage 1 (the `distilbert` base checkpoint has no head, so it never
  fires on the intended input); it is removed in Stage 2.
- Update the doc comment's MiniLM-only / six-kernel wording once Stage 2 lands.

`samples/NivaraInference/Program.cs`
- `OpenEncoder` delegate (line ~1409) gains a `BertGpuNaming naming` parameter. `OpenIlgpuEncoder`
  passes it through (currently hardcodes `Bert`); `OpenSilkEncoder` passes it to the new ctor. The
  three MiniLM bodies pass `BertGpuNaming.Bert`.
- Refactor `RunDistilBertGpuInference` / `BenchmarkDistilBertGpu` / `RunDistilBertGpuCompare`
  (lines ~1922, ~1969, ~2127) into backend-agnostic bodies parameterized by
  `(string backend, OpenEncoder open, BertGpuNaming naming, tensors)`, mirroring the MiniLM trio,
  with `*Gpu` and `*Silk` wrappers. They use `DistilBertConfig.ToBertConfig()` and `DistilBert`.
- `SilkModes` (line ~50): add `["distilbert"] = ["", "benchmark", "compare"]`; rewrite the
  MiniLM-only comment above it.
- Dispatch `case "distilbert"` (line ~561): add `if (useSilk) { … RunDistilBertSilk* }` before the
  `useGpu` branch.
- Guards/help: model-scope message (line ~358) → `(Silk models: minilm, distilbert, distilbert_sst)`;
  help text (lines ~252–255); F32 message (line ~462) "six-kernel Silk stack" → "hand-authored Silk
  kernel stack".

`tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs`
- Update `Silk_OnModelWithoutSilkKernels_Rejected` expected string (line ~263) and its doc comment
  (lines ~254–259).

`tests/Nivara.PerformanceTests/SilkKernelsGate.cs`
- Add a DistilBERT-shaped attention cell at headDim 64 (numeric bound), exercising the
  `MaxHeadDim` boundary the MiniLM dims do not.

End-to-end: `distilbert --silk compare` — GPU vs same-process CPU
(`|gpu−cpu| ≤ 1e-3·(1+|cpu|)`). `last_hidden_state_py.bin` is absent, so GPU-vs-CPU only (the
situation ModernBERT was in).

### Stage 2 — `distilbert_sst --silk` (head)

`samples/Nivara.Samples/Gpu/SilkKernels.cs`
- Add `gather_rows` — exact port of `ElementwiseKernels.Gather` (pure copy → byte-identity):
  `__kernel void gather_rows(__global const float* table, __global const int* ids, __global float* out, const uint hidden, const uint total)`.
- Add `gemm_row4_relu` — third epilogue over the existing `GEMM_ROW4_ACCUMULATE` macro
  (`c = max(acc + bias, 0)`; pure arithmetic → byte-identity, like `gemm_row4_bias`).
- Add both to `KernelNames` (line ~68) and update the "six kernels" doc comment. `SilkRuntime`
  compiles the whole translation unit and looks up each name in `KernelNames`, so these two edits
  are all that is required host-side.

`SilkBertEncoderRunner.cs`
- Replace the head rejection with the ILGPU pattern
  `hasHead = tensors.ContainsKey(PreWKey) && tensors.ContainsKey(ClsWKey)`; upload
  `preW`/`preB`/`clsW`/`clsB` (GEMM weights transposed) and allocate `clsOut` (`8*hidden`),
  `logits` (`8*2`), `clsIds`.
- In `Forward`, after the layer stack: upload `clsIds[b] = b*seqLen`;
  `Gather → gemm_relu(clsOut→h) → gemm_bias(h→logits, bCols=2)`; read back and return
  `new BertEncoderGpuResult(x, logits)`. Mirrors `BertEncoderGpuRunner.Forward` lines ~305–320.
- Add `LaunchGather` and a `LaunchGemmRelu` path (the existing `LaunchGemm` helper already takes a
  kernel name, so `gemm_row4_relu` needs no new code beyond passing the name).

`Program.cs`
- Add `["distilbert_sst"]` to `SilkModes`; refactor `RunDistilBertSstGpuInference` /
  `BenchmarkDistilBertSstGpu` / `RunDistilBertSstGpuCompare` (lines ~2000, ~2043, ~2184) into
  backend-agnostic bodies with `*Silk` wrappers; add the `distilbert_sst` `if (useSilk)` dispatch
  branch; update the model-scope string.

`SilkKernelsGate.cs`
- Add byte-identity cells for `gather_rows` and `gemm_row4_relu`.

End-to-end: `distilbert_sst --silk compare` — logits vs CPU + argmax 8/8, and the PyTorch 3-way
(`compare_distilbert_sst_py.bin` is present).

### Docs (fold in after the code runs)

- `docs/ACCELERATION.md` — line 18 status sentence and §1 "Second OpenCL backend": scope MiniLM +
  DistilBERT + SST-2; kernel set six → eight; file-table row; gate coverage. Add the DistilBERT
  informal A/B only if measured on AC (else "not measured / non-quotable", as the MiniLM battery
  note does).
- `docs/DISTELBERT.md` — "GPU path": add a `--silk` paragraph/row + verification-table rows.
- Consistency (same branch, own commit): `samples/NivaraInference/README.md` quick start,
  `CHANGELOG.md` Unreleased bullet, and the `Program.cs` help/scope strings. `docs/SILK.md` is the
  GpuProbe leg, not the model path — no change.

## Verification steps

- `dotnet build Nivara.slnx`
- `dotnet test -c Release --filter "FullyQualifiedName~InferenceCliArgumentTests"` (targeted;
  ask before the full suite)
- `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --silk-kernels`
- `dotnet run -c Release --project samples/NivaraInference -- distilbert --silk compare`
- `dotnet run -c Release --project samples/NivaraInference -- distilbert_sst --silk compare`
- Benchmarks (`… distilbert --silk benchmark`, `… distilbert_sst --silk benchmark`) — **Release,
  AC power only** for numbers that go in the docs.

## Blast radius

- **Sample + gate only.** No `src/Nivara` changes; the core library, its public API and its tests
  are untouched.
- `samples/Nivara.Samples/Gpu/SilkKernels.cs` — shared by `--silk` (sample) and `--silk-kernels`
  (gate); adding kernels is additive.
- `samples/Nivara.Samples/Gpu/SilkBertEncoderRunner.cs` — implements `IBertEncoderRunner`; the
  MiniLM path is exercised by `minilm --silk {,benchmark,compare}` and must stay green.
- `samples/NivaraInference/Program.cs` — CLI surface; the `(model, mode)` allow-lists and guard
  messages are pinned by `InferenceCliArgumentTests`, which must move in lockstep.
- `tests/Nivara.PerformanceTests/SilkKernelsGate.cs` — standalone harness (`--silk-kernels`), not
  part of the CI unit suite.
- Downstream callers: none outside the sample (the `IBertEncoderRunner` seam is sample-internal).

## Planned commits

1. `Add bert-family weight naming to the Silk encoder runner` — Stage 1 code + `distilbert`
   routing/scope + CLI test update.
2. `Add gather_rows and gemm_row4_relu Silk kernels` — Stage 2 kernels + `KernelNames` + gate cells.
3. `Port the DistilBERT classification head to the Silk runner` — head support + `distilbert_sst`
   routing/scope.
4. `document --silk support for the DistilBERT family` — ACCELERATION.md, DISTELBERT.md, README,
   CHANGELOG.

## GitHub issues log

- [ ] (none yet)
