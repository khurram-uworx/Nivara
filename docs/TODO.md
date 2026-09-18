# DistilBERT GPU first scenario — `distilbert --gpu` via ILGPU (branch `khurram/distilbert-gpu`)

Status: **In progress — plan committed; assessment committed in
[docs/DISTILBERT-GPU.md](DISTILBERT-GPU.md) (`cb28485`).**

> Reminder: as each task executes, if you find deferred work or a concern
> (known limitations, follow-ups, refactors) that is outside the current plan,
> create a GitHub issue immediately
> (`gh issue create --repo khurram-uworx/Nivara`) and record its number in the
> GitHub issues log below — don't rely on memory or wait until the plan
> finishes, as compaction during execution can lose important items.

## Problem

DistilBERT inference in `samples/NivaraInference` is CPU-only (185 ms / 128-tok
forward vs PyTorch CPU 35 ms — ~5×). The GPU probe
([tests/Nivara.GpuProbe](../../tests/Nivara.GpuProbe/README.md)) proved that
ILGPU (pure-managed C#→OpenCL JIT, NuGet-only, 3/3 correctness gates PASS on
this Arc 140T machine) is the lowest-friction managed backend, but its measured
gemv win (1536×576, 133.4 µs ≈ 6.6 GMAC/s naive) does **not** transfer to
DistilBERT's batch-128 GEMM shapes without a tiled kernel (naive ≈ 825 ms —
slower than CPU). We want the first **end-to-end, gated, measured** GPU sample
scenario: `distilbert --gpu` / `distilbert_sst --gpu`, F32-only, with the
research/backend decision deferred to a later library-design phase.

Full reasoning, numbers, and the honest estimate live in
[docs/DISTILBERT-GPU.md](DISTILBERT-GPU.md). The keystone risk there is the
**tiled GEMM** (decision gate: ≥ ~0.3 T MAC/s on Arc 140T f32 or step back to
the OpenVINO path).

## Scope (locked)

- ILGPU package reference added in **`samples/Nivara.Samples` only**.
- **No** `src/Nivara`, **no** `src/Nivara.Extensions`, **no** `src/Nivara.Gpu`
  project, **no** core/kernel GPU branches, **no** public API changes.
- F32-only: `--gpu` + `--precision bf16|fp16` → clear rejection message.
- No SmolLM/Qwen GPU, no KV-cache/decode GPU, no training on GPU, no bf16 GPU.

## Grounding (G1) — verified before implementation

- **ILGPU official docs** (ilgpu.net tutorials + master samples, fetched 2026-09-19):
  - Implicitly-grouped kernels **cannot** use shared memory / group intrinsics — the
    tiled GEMM **must** be an explicitly-grouped kernel (`LoadKernel` /
    `LoadStreamKernel` + launch config `(numGroups, groupSize)`, indices via
    `Grid.GlobalIndex.XY`, `Group.IdxX`/`Group.IdxY`, `Group.Barrier()`).
  - Shared memory on OpenCL confirmed: `SharedMemory.Allocate2D<float,
    Stride2D.DenseX>(new Index2D(size, size), new Stride2D.DenseX(size))`.
  - Canonical reference pattern: official `Samples/MatrixMultiply/Program.cs`
    (tiled kernel TILE staging + zero-fill bounds checks) — saved to
    `%TEMP%\opencode\ilgpu_matrixmultiply_Program.cs` during grounding.
  - Buffer path precedents (probe-verified): `Allocate1D`/`Allocate2DDenseX`,
    `CopyFromCPU`, views, `AsContiguous().GetAsArray()` / `GetAsArray2D`,
    `CLDeviceType.GPU` + Intel vendor assert, `XMath.Exp` from ILGPU.Algorithms.
- **`XMath` has NO `erf`** (verified in ILGPU master `Src/ILGPU.Algorithms/XMath.cs`).
  GELU erf is therefore not a "fallback" — the A–S 7.1.26 polynomial port with
  `XMath.Exp` (identical formula to CPU `GradKernels.Erf<float>`) is the primary
  path; this makes CPU/GPU GELU parity *tighter* than the 1e-6+1e-5·|x| gate.
- **CPU gate kernel confirmed** (code-memory MCP): `MatMulTransposedB<T>(
  ReadOnlySpan<T>, ReadOnlySpan<T>, T[], int aRows, int aCols, int bCols)` in
  `src/Nivara/AutoDiff/Nn/LlamaFusedKernels.cs` — the production f32 reference
  for the tiled-GEMM temp harness.
- **MS Learn**: no ILGPU coverage (expected — not a Microsoft tool); adjacent
  authoritative source is the Windows OpenCL dev-notes page (API reference is
  Khronos, OpenCL.dll ships with Windows). Nothing else to ground there.
- Blast radius unchanged from §below.

## DistilBERT kernel inventory (what the GPU forward runs)

From the assessment (§3): 6 layers × (`q/k/v/o` `[128,768]·[768,768]`×4, fused
attention core (12 heads × QKᵀ + row-softmax + ·V), `lin1`
`[128,768]·[768,3072]`, GELU(erf), `lin2` `[128,3072]·[3072,768]`, LayerNorm ×2,
bias/residual adds) ≈ 5.44 GMAC + embeddings (two gathers) + head. Weights:
66.9M params F32 = 255.5 MB (iGPU shares DRAM — upload is memcpy-class).
CPU gate targets: encoder hidden states vs CPU path ~1e-6 class; SST-2 argmax
8/8; `|gpu − cpu| ≤ 1e-6 + 1e-5·|cpu|` (probe gate contract).

## Planned commits (one logical change each)

1. ✅ `docs: add DistilBERT GPU assessment (DISTILBERT-GPU.md)` — **done** (`cb28485`)
2. `docs: plan DistilBERT GPU first scenario in TODO.md` — this commit
3. `samples: add ILGPU 1.5.3 references to Nivara.Samples` — `ILGPU` +
   `ILGPU.Algorithms` (same versions as the probe); restores now pull ILGPU
   transitively for Nivara.Tests / PerformanceTests / GpuProbe / other samples
   (pure-managed, build-harmless; no code change)
4. `samples: add ILGPU runtime + tiled GEMM kernel (Nivara.Samples/Gpu)` —
   sample-scoped GPU scaffolding under `samples/Nivara.Samples/Gpu/`:
   - `IlgpuRuntime.cs` — context/accelerator/stream lifecycle, device select
     (`CL_DEVICE_TYPE_GPU` + Intel vendor, **asserted — no CPU fallback**),
     persistent `ArrayView` buffer upload/download helpers (probe patterns from
     `tests/Nivara.GpuProbe/Ilgpu/IlgpuLeg.cs`)
   - `GemmKernels.cs` — the keystone **tiled GEMM** (grouped-kernel pattern,
     local-memory staging, `Index2D`/`GroupedIndex2D`; operands plain row-major
     — weights pre-transposed once at upload so `C = A·Bt` is coalesced, per
     assessment §3 layout note)
   - Before the full kernel set: **measure the tiled GEMM** at DistilBERT shapes
     ([128,768]·[768,768], [128,768]·[768,3072], [128,3072]·[3072,768]) via a
     **temp probe harness** under `%TEMP%\opencode\` that references Nivara.Samples
     and gates vs `LlamaFusedKernels.MatMulTransposedB<float>` (production CPU
     kernel, tolerance 1e-6+1e-5·|ref|) and reports GMAC/s. **Decision gate:**
     < ~0.3 T MAC/s → stop and escalate to the human (OpenVINO fallback);
     ≥ 0.3 → delete temp harness, record the measurement in
     `docs/DISTILBERT-GPU.md`, continue
5. `samples: add DistilBERT GPU forward — attention, LayerNorm, GELU, gather` —
   `Gpu/AttentionKernels.cs` (fused 12-head score+scale+mask+row-softmax+weighted-V,
   `XMath.Exp`), `Gpu/ElementwiseKernels.cs` (LayerNorm row-reduce, GELU via
   direct port of `GradKernels.Erf` A-S 7.1.26 poly with `XMath.Exp`, bias/residual
   adds, embedding gather), `Gpu/DistilBertGpuRunner.cs` (uploads weights
   (transposed) by the exact `DistilBertLoader` key set, runs the §-inventory
   forward, returns readback `float[]` per stage for gating)
6. `samples: wire --gpu into distilbert/distilbert_sst + benchmark` —
   `Program.cs` gains `--gpu`; `distilbert --gpu` prints output stats + timing;
   `distilbert --gpu benchmark` (3 warmup + 10 timed); `distilbert_sst --gpu`
   prints the 8-sentence argmax table; `--precision bf16|fp16` + `--gpu` →
   clear "GPU is f32-only in this phase" error; CPU modes untouched
7. `samples: gate GPU forward vs CPU/PyTorch (hidden states, SST-2 argmax)` —
   same-process CPU reference forward, per-stage diff
   (`|gpu − cpu| ≤ 1e-6 + 1e-5·|cpu|`), SST-2 argmax parity 8/8, and diff vs
   `last_hidden_state_py.bin` / `compare_distilbert_sst_py.bin` when fixtures
   exist
8. `docs: record measured DistilBERT GPU numbers` — update
   `docs/DISTILBERT-GPU.md` (§4 estimate → measured table) and
   `samples/NivaraInference/README.md` (GPU rows, `--gpu` usage)

## Verification (each gated step)

- `dotnet build Nivara.slnx` after every commit (build is quick; `dotnet test`
  full NUnit suite only **after explicit human confirmation** per AGENTS.md —
  sample-only changes should not affect it, snapshot runs as needed)
- Tiled-GEMM temp harness: gates + GMAC/s table → recorded in
  `docs/DISTILBERT-GPU.md` before the model kernel set is written
- `dotnet run --project samples/NivaraInference -c Release -- distilbert --gpu`
  (weights already at `samples/data/distilbert`) — CPU-parity gate prints PASS
- `dotnet run --project samples/NivaraInference -c Release -- distilbert --gpu benchmark`
  — per-forward median ms (target ~10–30 ms vs 185 ms CPU; headline number)
- `dotnet run --project samples/NivaraInference -c Release -- distilbert_sst --gpu`
  — 8/8 argmax vs CPU, fixture diff when present
- CPU regression sanity: `distilbert` (no `--gpu`) timing/numerics unchanged

## Blast radius

- **New files**: `samples/Nivara.Samples/Gpu/*.cs` (5–6 files, sample-scoped,
  like `Gpt2BpeTokenizer`/`LlamaLoader` conventions)
- **Edited**: `samples/Nivara.Samples/Nivara.Samples.csproj` (+2 ILGPU refs —
  restore ripple to Nivara.Tests/PerformanceTests/GpuProbe/5 samples, build-only;
  no behavior change), `samples/NivaraInference/Program.cs` (`--gpu` flag +
  distilbert/distilbert_sst wiring), `samples/NivaraInference/README.md`,
  `docs/DISTILBERT-GPU.md`
- **Untouched**: `src/Nivara`, `src/Nivara.Extensions`, `tests/Nivara.Tests`
  (only restore ripple), `tests/Nivara.GpuProbe`, other samples (restore ripple
  only)
- Tests covering touched files: Nivara.Inference sample has no NUnit coverage;
  CPU parity is verified by the sample's own gate + the existing compare
  fixtures

## GitHub issues log

- (empty — create issues during execution for anything deferred, e.g. bf16 GPU,
  `src/Nivara.Gpu` promotion decision, tiled-GEMM regression gate into the probe)