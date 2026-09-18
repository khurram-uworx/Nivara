# DistilBERT GPU first scenario — `distilbert --gpu` via ILGPU (branch `khurram/distilbert-gpu`)

Status: **Complete — all commits landed and all gates PASSED** (correctness on
battery, benchmarks on AC power, 2026-09-19). Commit 8 (docs) below; final G2
review then delete this file.

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
CPU gate targets: encoder hidden states vs CPU path; SST-2 argmax 8/8;
per-element `|gpu − cpu| ≤ 1e-3·(1 + |cpu|)` on final hidden state + logits
(**revised 2026-09-19 from the probe's `1e-6 + 1e-5·|cpu|`** — two different
f32 summation orders can't meet that; measured GEMM floor is 4.4e-5–1.6e-4 at
K=768–3072, see DISTILBERT-GPU.md §4.3/§6).

## Planned commits (one logical change each)

1. ✅ `docs: add DistilBERT GPU assessment (DISTILBERT-GPU.md)` — **done** (`cb28485`)
2. ✅ `docs: plan DistilBERT GPU first scenario in TODO.md` — **committed**
   (`de63a5a`)
3. ✅ `samples: add ILGPU 1.5.3 references to Nivara.Samples` — **committed**
   (`87f592e`) — `ILGPU` +
   `ILGPU.Algorithms` (same versions as the probe); restores now pull ILGPU
   transitively for Nivara.Tests / PerformanceTests / GpuProbe / other samples
   (pure-managed, build-harmless; no code change)
4. ✅ `samples: add ILGPU runtime + tiled GEMM kernel (Nivara.Samples/Gpu)` —
   **committed** (`349b481`):
   - `IlgpuRuntime.cs` — context/accelerator/stream lifecycle, device select
     (`CL_DEVICE_TYPE_GPU` + Intel vendor, **asserted — no CPU fallback**),
     persistent `ArrayView` buffer upload/download helpers (probe patterns from
     `tests/Nivara.GpuProbe/Ilgpu/IlgpuLeg.cs`)
   - `GemmKernels.cs` — the keystone **tiled GEMM** (grouped-kernel pattern,
     local-memory staging, `Index2D`/`GroupedIndex2D`; operands plain row-major
     — weights pre-transposed once at upload so `C = A·Bt` is coalesced, per
     assessment §3 layout note)
   - ✅ **Measured via temp harness** (`%TEMP%\opencode\gemm-measure\`, then
     deleted): gates vs `LlamaFusedKernels.MatMulTransposedB<float>`-class double
     truth (maxAbs 4.4e-5..1.6e-4 at K=768..3072, tight enough); **Row4 gate
     PASS** 303–379 GMAC/s on all shapes (recorded in DISTILBERT-GPU.md §4.3);
     OpenVINO fallback **not** triggered
5. ✅ `samples: add DistilBERT GPU forward — attention, LayerNorm, GELU, gather` —
   **committed** (`c52eea2`):
   `Gpu/AttentionKernels.cs` (fused 12-head score+scale+mask+row-softmax+weighted-V,
   `XMath.Exp`), `Gpu/ElementwiseKernels.cs` (LayerNorm row-reduce, GELU via
   direct port of `GradKernels.Erf` A-S 7.1.26 poly with `XMath.Exp`, bias/residual
   adds, embedding gather), `Gpu/DistilBertGpuRunner.cs` (uploads weights
   (transposed) by the exact `DistilBertLoader` key set, runs the §-inventory
   forward, returns readback `float[]` per stage for gating)
6. ✅ `samples: wire --gpu into distilbert/distilbert_sst + benchmark` —
   **committed** (`736fd79`):
   `Program.cs` gains `--gpu`; `distilbert --gpu` prints output stats + timing;
   `distilbert --gpu benchmark` (3 warmup + 10 timed, `ReportGpuTiming` mirror);
   `distilbert_sst --gpu` prints the 8-sentence argmax table; `--precision bf16|fp16`
   + `--gpu` → clear "GPU is f32-only in this phase" error; CPU modes untouched;
   GPU `compare`/`predict` defer to commit 7
7. ✅ `samples: gate GPU forward vs CPU/PyTorch (hidden states, SST-2 argmax)` —
   **committed** (`ce19bb3`):
   `distilbert --gpu compare` runs the same-process CPU reference
   (`BertEncoder.Forward` path, identical tokenization) vs the GPU runner,
   per-stage diff (`|gpu − cpu| ≤ 1e-3·(1+|cpu|)` on final hidden state + logits
   — revised from `1e-6 + 1e-5·|cpu|`, see §4.3), plus diff vs
   `last_hidden_state_py.bin` when present; `distilbert_sst --gpu compare` = 8
   sentences logits parity + argmax 8/8 + GPU-vs-PyTorch diff vs
   `compare_distilbert_sst_py.bin` when present; explicit GATE PASS/FAIL verdict
   lines — **executed 2026-09-19, both gates PASS** (see Verification below)
8. `docs: record measured DistilBERT GPU numbers` — update
   `docs/DISTILBERT-GPU.md` (§4 estimate → measured table, §4.4) and
   `samples/NivaraInference/README.md` (GPU rows, `--gpu` usage) — **committed**
   (docs commit, this file's sibling in the same change)

## Verification (each gated step)

- `dotnet build Nivara.slnx` after every commit (build is quick; `dotnet test`
  full NUnit suite only **after explicit human confirmation** per AGENTS.md —
  sample-only changes should not affect it, snapshot runs as needed)
- Tiled-GEMM temp harness: gates + GMAC/s table → recorded in
  `docs/DISTILBERT-GPU.md` before the model kernel set is written
- ✅ **Correctness gates PASSED (battery, weights downloaded from HF cache
  misses — the checkpoints were not pre-provisioned; ~511 MB via `hf download`
  into `samples/data/distilbert{,_sst}` on 2026-09-19)**:
  - `distilbert --gpu compare` → **GATE PASS** — hidden [128,768] parity vs CPU
    `maxAbs 1.53e-5, maxRel 3.24e-6, 0/98304 violations` (tol 1e-3·(1+|r|));
    `last_hidden_state_py.bin` absent (fixture optional)
  - `distilbert_sst --gpu compare` → **GATE PASS** — logits parity vs CPU
    `maxAbs 3.8e-6, maxRel 6.7e-7, 0/16`, argmax **8/8**; vs PyTorch fixture
    `compare_distilbert_sst_py.bin` `maxAbs 3.3e-6, maxRel 5.8e-7`, argmax
    **8/8** (CPU's own doc'd bound vs HF is 9.5e-7 — same class)
  - `distilbert --gpu` → 110–135 ms fwd (battery), stats sane
  - `distilbert_sst --gpu` → 8 sensible argmax (4 POS / 4 NEG), per-sentence
    88–163 ms (battery)
- **Fix during verification** (`5ad36e5`): ILGPU array `CopyFromCPU(T[])`
  requires `data.Length ≥ view.Length` (fills the whole view) — 128-element
  payloads into 1024-cap workspace threw `ArgumentOutOfRangeException('data')`.
  Payload copies now use explicit-length `View.SubView(0, n)` on `runtime.Stream`
  (ordered with kernel launches); ctor ends with a device sync so default-stream
  weight uploads are visible to kernel launches
- ✅ **Perf runs (AC power, 2026-09-19)** — 3 warmup + 10 timed each, same
  session per row (GPU vs CPU columns recorded together):
  - `distilbert --gpu benchmark` → **GPU 65.3 ms avg (62–73)** vs CPU `distilbert
    benchmark` **194.7 ms avg (152–232)** → **~3.0×**
  - `distilbert_sst --gpu benchmark` → **GPU 64.0 ms avg (61–71)** vs CPU
    `distilbert_sst benchmark` **166.4 ms avg (134–208)** → **~2.6×**
  - `--precision bf16|fp16` + `--gpu` rejection verified (clear F32-only
    message, exit 1)
  - Numbers recorded in docs/DISTILBERT-GPU.md §4.4 + README; plan §4 estimated
    ~15–25 ms — the ~65 ms gap is launch/dispatch overhead at 128-row shapes (one
    kernel per op, ~100 dispatches/forward), not GEMM throughput (flagged follow-up:
    lazy stream + per-op fusion). Battery reference: ~110–135 ms/forward.
- ✅ CPU regression sanity: `distilbert` (no `--gpu`) timing/numerics unchanged
  (194.7 ms avg matches the pre-GPU ~185 ms class; remote HT uses same path)

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

- **#435** — promote tiled-GEMM correctness+perf harness into a lasting
  regression gate (probe or sample bench); created while executing plan commit
  4 — the temp harness's double-precision-truth gate caught a real Row4
  `colBase` bug during the keystone run (maxAbs ~40–77) before the harness was
  deleted
- (create issues during execution for anything deferred, e.g. bf16 GPU,
  `src/Nivara.Gpu` promotion decision)