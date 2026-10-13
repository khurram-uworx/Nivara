# Silk promotion: move the Silk OpenCL backend and the model layer into Nivara.Extensions

Branch: `khurram/silk` (off `main` @ `b230115d`).

## Problem

Silk is the long-term GPU bet, but it lives in `samples/Nivara.Samples/` — a non-packable sample
library. The Silk kernels, runtime, runners and sessions should ship from `Nivara.Extensions`
(the third-party-dependency home per AGENTS.md), and the sample runners should consume them from
there. ILGPU stays in `samples/Nivara.Samples` for now; it is not being dropped.

Three layers are involved, only the first of which is Silk-specific:

| Layer | Files (`samples/Nivara.Samples/`) | Notes |
|---|---|---|
| Silk kernels + runtime | `Gpu/SilkKernels.cs`, `Gpu/SilkRuntime.cs` | dependency-free device layer |
| Silk runners + sessions | `Gpu/SilkBertEncoderRunner.cs`, `Gpu/SilkModernBertGpuRunner.cs`, `Gpu/SilkLayaHeadGpuRunner.cs`, the Silk halves of `Gpu/ModernBertGpuSession.cs` / `Gpu/LayaGpuSession.cs` | depend on the model types |
| Model layer | `BertModel.cs`, `ModernBertModel.cs`, `DistilBertModel.cs`, `LayaHeadModel.cs`, `LayaHeadScoring.cs`, `LayaPromptBuilder.cs`, `Gpt2BpeTokenizer.cs`, `SafeTensorsLoader.cs`, `StateDictLoader.cs`, `FrameBuilder.cs` | shared by the CPU path, ILGPU and Silk |

## Confirmed decisions (with the human)

| Area | Decision |
|---|---|
| Boundary | One-shot full move: device layer + model zoo + three Silk runners + sessions, in this branch |
| Package | Fold into the existing `Nivara.Extensions` package (its nuspec gains `Silk.NET.OpenCL` 2.23.0) |
| Accessibility | Move types **as they are** — public types stay public so production consumers (samples/CLI) can use them. Test-only access goes through `InternalsVisibleTo`, never by widening the public surface. `SilkKernels` stays `internal` |
| Namespace | Silk backend → `Nivara.Silk` in `src/Nivara.Extensions/Silk/`; model layer → `Nivara.AI` in `src/Nivara.Extensions/AI/` |
| Breadth | Compiler-forced closure only. `LlamaForCausalLM`, `TextClassifierModel`, `TokenClassifierModel`, `FrameBuilder`, `Incident/*` stay in samples unless the compiler drags one in |
| Launcher | Add a typed dispatch launcher now (the three Silk runners carry identical copies of the `Launch*` helpers) |

## Target layout

```
src/Nivara.Extensions/Silk/          namespace Nivara.Silk
  SilkKernels.cs                     (internal) OpenCL C source + geometry
  SilkRuntime.cs                     (public)  device select, build, buffers, dispatch
  SilkLauncher.cs                    (new)     typed dispatch shared by the three runners
  WeightLayout.cs                    (new)     Transposed / QkvPacked / Req, extracted from GpuBuffers
  SilkBertEncoderRunner.cs           (public)
  SilkModernBertGpuRunner.cs         (public)
  SilkLayaHeadGpuRunner.cs           (public)
  SilkModernBertSession.cs           (moved out of ModernBertGpuSession.cs)
  SilkLayaSession.cs                 (moved out of LayaGpuSession.cs)

src/Nivara.Extensions/AI/            namespace Nivara.AI
  NivaraEmbeddingGenerator.cs        (already there)
  BertModel.cs, ModernBertModel.cs, DistilBertModel.cs, LayaHeadModel.cs,
  LayaHeadScoring.cs, LayaPromptBuilder.cs, Gpt2BpeTokenizer.cs,
  SafeTensorsLoader.cs, StateDictLoader.cs, FrameBuilder.cs   (compiler-closure)
  IBertEncoderRunner.cs              (the shared seam)
  ModernBertGpuSession.cs            (interface + delegate only; ILGPU session class stays in samples)
  LayaGpuSession.cs                  (interface + delegate only; ILGPU session class stays in samples)
  BertEncoderGpuTypes.cs             (new) BertGpuNaming / BertGpuKeys / BertEncoderGpuResult,
                                     split out of Gpu/BertEncoderGpuRunner.cs
```

`BertEncoderGpuResult`, `BertGpuNaming` and `BertGpuKeys` currently live inside
`Gpu/BertEncoderGpuRunner.cs` (which stays in samples), so that file must be split.

### Circular-reference guard

The Silk runners call `GpuBuffers.Transposed` / `QkvPacked` / `Req`, but `GpuBuffers` is
ILGPU-coupled and stays in samples. Move those three pure host-math helpers into
`Nivara.Silk` (`WeightLayout.cs`) and have `GpuBuffers` delegate to them, so Extensions never
references samples.

### Accessibility note

`SilkKernels` is `internal`; its consumers (the Silk runners, `SilkRuntime`) are now same-assembly,
so that is fine. The two tests that reach it — `SilkKernelsCatalogTests` and the `--silk-kernels`
gate — get there through `InternalsVisibleTo`. `Nivara.Extensions.csproj` currently grants IVT to
`Nivara.Tests` only; add `Nivara.PerformanceTests`. `WeightLayout` is `public` because `GpuBuffers`
(samples) is a production consumer.

## Blast radius

- **Moved code:** ~2,264 lines of Silk + ~3,400 lines of model code (compiler-closure subset), across
  ~20 files.
- **Projects touched:** `src/Nivara.Extensions`, `samples/Nivara.Samples`, `samples/NivaraInference`,
  `samples/NivaraFineTuning`, `tests/Nivara.Tests`, `tests/Nivara.PerformanceTests`.
- **Downstream callers to rewire:** `NivaraInference` (`Program.cs`, `DistilBertSst.cs`, `Laya.cs`,
  `ModernBert.cs`), `NivaraFineTuning` (`Program.cs`, `Sst2Dataset.cs`), ~20 test files that
  `using Nivara.Samples;` / `using Nivara.Samples.Gpu;`.
- **Docs carrying the old paths:** `SILK.md` (lines 14, 359, 371–381, 387, 432, 445, 474, 506–513,
  535, 562–576), `ACCELERATION.md` (184, 194–200), `ILGPU.md` (30, 281, 284, 297), `LAYA.md` (82–85),
  `DISTELBERT.md` (143), the `AGENTS.md` implementation map, and the `--silk` help string in
  `NivaraInference/Program.cs:257`.
- **Tests covering the moved code:** `tests/Nivara.Tests/Gpu/SilkKernelsCatalogTests.cs`,
  `tests/Nivara.PerformanceTests/SilkKernelsGate.cs`, the model tests under
  `tests/Nivara.Tests/AutoDiff/` and `tests/Nivara.Tests/NivaraTorch/`.
- **Risk of a mechanical closure miscount:** the code-memory relationship index resolves by name and
  reports false edges (e.g. `BertLayer.Forward` "calling" the GPU kernel `ElementwiseKernels.GeluExact`
  when it actually calls `ReverseGradOperations.GeluExact`). The closure is therefore driven by the
  **compiler**, not the index. Verified false: the model files do **not** depend on
  `Gpu/ElementwiseKernels.cs`.

## Proposed changes

### Step 1 — Silk device layer + host math + launcher (commit 2)

- Move `Gpu/SilkKernels.cs` and `Gpu/SilkRuntime.cs` to `src/Nivara.Extensions/Silk/`;
  `namespace Nivara.Silk`.
- Extract `Transposed` / `QkvPacked` / `Req` from `Gpu/GpuBuffers.cs` into
  `src/Nivara.Extensions/Silk/WeightLayout.cs`; `GpuBuffers` calls the new helper.
- Add `SilkLauncher` absorbing the duplicated helpers: `LaunchGemm`, `LaunchLayerNorm`,
  `LaunchAttention`, `LaunchAdd`, `LaunchAddBias`, `LaunchGather`, `LaunchSplit`, `LaunchRotary`,
  `LaunchGeGlu`, `RunLinear`, `UploadPlain`, `UploadTransposed`, `Upload`, `Release`.
- Rewire the three Silk runners (still in samples for this commit) to call `SilkLauncher`.
- Add `Silk.NET.OpenCL` 2.23.0 to `Nivara.Extensions.csproj`; add IVT to
  `Nivara.PerformanceTests`; drop the direct pin from `Nivara.Samples.csproj` (now transitive).
- Backend-neutralize the `--silk …` strings inside the moved files (a shipped library should not
  name a CLI flag); reword cross-assembly `<see cref>` targets (they become errors under
  `TreatWarningsAsErrors`).

### Step 2 — Silk runners + sessions (commit 3)

- Move `SilkBertEncoderRunner.cs`, `SilkModernBertGpuRunner.cs`, `SilkLayaHeadGpuRunner.cs` to
  `Silk/`; move the Silk halves of `ModernBertGpuSession.cs` / `LayaGpuSession.cs` into
  `Silk/SilkModernBertSession.cs` / `Silk/SilkLayaSession.cs`.
- Move the shared seam (`IBertEncoderRunner`, `IModernBertGpuSession`, `ILayaGpuSession` and the
  `OpenModernBertSession` / `OpenLayaSession` delegates) to `AI/`; the ILGPU session classes stay in
  `samples/Nivara.Samples/Gpu/` and implement the `Nivara.AI` interfaces.
- Split `BertGpuNaming` / `BertGpuKeys` / `BertEncoderGpuResult` out of
  `Gpu/BertEncoderGpuRunner.cs` into `AI/BertEncoderGpuTypes.cs`.
- Rewire the ILGPU runners/sessions and the CLI.

### Step 3 — Model layer (commit 4)

- Move the model files to `src/Nivara.Extensions/AI/`, `namespace Nivara.AI`, seeded by
  `BertConfig`, `ModernBertConfig` and the Laya types; follow the compiler to the true closure
  (expected: `BertModel`, `ModernBertModel`, `DistilBertModel`, `LayaHeadModel`, `LayaHeadScoring`,
  `LayaPromptBuilder`, `LayaCalibration`, `Gpt2BpeTokenizer`, `SafeTensorsLoader`, `StateDictLoader`).
- Rewire every consumer's `using`.

### Step 4 — Tests (commit 5)

- Update `using Nivara.Samples;` → `using Nivara.AI;` and `using Nivara.Samples.Gpu;` → the
  appropriate `Nivara.Silk` / `Nivara.AI` / remaining `Nivara.Samples.Gpu` namespace across the test
  projects. `SharedMemoryAllocationTests` keeps scanning `Nivara.Samples.Gpu` (ILGPU kernels stay).

### Step 5 — Docs (commit 6)

- Update the paths in `SILK.md`, `ACCELERATION.md`, `ILGPU.md`, `LAY A.md`, `DISTELBERT.md`, the
  `AGENTS.md` implementation map, `CHANGELOG.md`, the `NivaraInference` help string, and the
  `Nivara.Extensions` package `Description` (now ships the model layer + a GPU backend).

## Verification steps

1. `dotnet build Nivara.slnx` (warnings are errors — the `<see cref>` sweep is load-bearing).
2. Targeted, device-free: `dotnet test -c Release --filter "FullyQualifiedName~SilkKernelsCatalogTests"`
   plus the model tests under `Nivara.Tests/AutoDiff` and `Nivara.Tests/NivaraTorch`.
3. GPU-dependent (ask before running, it is minutes long): the `--silk-kernels` Release gate, and
   `minilm --silk compare` / `modernbert --silk compare` / `laya --silk compare`.
4. Full filtered suite only if the human asks (`Category!=Performance`).

## Planned commits

1. `docs: plan Silk promotion into Nivara.Extensions in TODO.md`
2. `refactor: extract host math and move the Silk device layer into Nivara.Extensions`
3. `refactor: move the Silk runners and sessions into Nivara.Extensions`
4. `refactor: move the BERT-family model layer into Nivara.Extensions`
5. `test: rewire Silk and model tests to the Nivara.Extensions namespaces`
6. `docs: update Silk and model docs for the Nivara.Extensions move`
7. (G2) `docs: remove TODO.md — plan executed`

## Non-goals

- ILGPU is not dropped and stays in `samples/Nivara.Samples` (kernels, runtime, runners, sessions).
- `tests/Nivara.GpuProbe/OpenCl/SilkLeg.cs` is untouched (its own probe leg, its own Silk.NET pin).
- No decision on the public API surface beyond moving the types as-is.
- The full `dotnet test` suite is not run unprompted.

## GitHub issues log

- (none yet — create an issue at discovery time via
  `gh issue create --repo khurram-uworx/Nivara` and record its number here)
