# TODO — #468: remove the exposed `SharedMemory.Allocate2D` pattern from the live GEMM path

Branch: `khurram/468` (off `main` @ `eafb793`).
Issue: <https://github.com/khurram-uworx/Nivara/issues/468> — *ILGPU 1.5.3 OpenCL: two
`SharedMemory.Allocate2D` calls with differing extents mis-place the second tile*.

## Problem

Two `SharedMemory.Allocate2D` calls in one ILGPU kernel whose extents disagree get the second
tile mis-placed by ILGPU 1.5.3's OpenCL lowering. The kernel reads a neighbouring tile's staged
data and returns plausible wrong values (`maxAbs` 20–119, the right order of magnitude for a real
dot product against the wrong elements, so not an OOB smear). One variant fails earlier, at
`LoadKernel`, with a `CLException`.

**The mitigation landed on the wrong four kernels.** `docs/ACCELERATION.md` lesson 19 and
`samples/Nivara.Samples/Gpu/GemmKernels.cs:167-171` record the fix — one
`SharedMemory.Allocate<float>(n)` with hand-computed 2D offsets — and it is applied to the four
#440 geometries (`GemmKernels.cs:190, 265, 331, 422`). But the kernels every GPU encoder path
actually runs still use two `Allocate2D` calls with disagreeing extents:

| kernel | lines | A tile | B tile | production callers |
|---|---|---|---|---|
| `TiledGemmKernelRow4` | 79-82 | `16x16` `DenseX(16)` | `16x64` `DenseX(64)` | `ModernBertGpuRunner.cs:120`, `TiledGemm` (`GemmKernels.cs:811-813`), `GemmLegBenchmark.cs:122` |
| `TiledGemmKernelRow4Bias` | 505-508 | same | same | `BertEncoderGpuRunner.cs:224`, `LayaHeadGpuRunner.cs:155` |
| `TiledGemmKernelRow4Gelu` | 573-576 | same | same | `BertEncoderGpuRunner.cs:226`, `LayaHeadGpuRunner.cs:156` |
| `TiledGemmKernelRow4Relu` | 641-644 | same | same | `BertEncoderGpuRunner.cs:228`, `LayaHeadGpuRunner.cs:157` |
| `TiledGemmKernelRow4Qkv` | 723-726 | same | same | `BertEncoderGpuRunner.cs:230`, `LayaHeadGpuRunner.cs:158` |
| `TiledGemmKernel` (1x1) | 34-37 | `16x16` `DenseX(16)` | `16x16` `DenseX(16)` | `TiledGemm` (`OneToOne` variant) |

So DistilBERT, MiniLM, ModernBERT and Laya all run the pattern the issue says is unsafe.

**"Square A escapes" is not a rule.** The issue's note that the incumbent was safe because its A
tile is `16x16` is contradicted by the issue's own `1x8@KT16` row: A tile `16x16`, and it failed at
`LoadKernel`. The only evidence-backed safe case in the table is `2x2@KT32`, where the two extents
were *identical* (`32x32` / `32x32`). `Row4` stages `16x16` against `16x64` — disagreeing — so it
is in the exposed class and is correct only by accident of this ILGPU build + Arc driver. Same for
the 1x1 kernel, whose two tiles match only by coincidence.

**The gate was dropping the error code.** `ILGPU.Runtime.OpenCL.CLException` exposes a public
`CLError Error` (verified by reflection over `ILGPU 1.5.3`), but
`tests/Nivara.PerformanceTests/GemmBenchmark.cs:204-205` prints only
`ex.GetType().Name + ex.Message`. The issue's "bare `CLException` carrying no error code" is
partly our own formatting, not an ILGPU omission.

## The invariant this establishes

> **A kernel body may contain at most one shared-memory allocation.** Two tiles are staged through
> one `SharedMemory.Allocate<float>(n)` with hand-computed 2D offsets.

This is lesson 19's fix and lesson 15's prescribed shape, stated so it can be checked mechanically.
Deliberately *not* "the two extents must match": matching extents was correct in one observed case,
which is an accident of one build, not a property. The invariant does not encode an accident.

Consequence: `TiledGemmKernel` (1x1) migrates too, or the invariant needs an exemption for the one
geometry that happens to work. Migrating it is the plan (an exemption list is a rot vector).
`AttentionKernels.cs:93` keeps its single `Allocate2D` — one tile, not the pattern.

## Blast radius

**Touched**

- `samples/Nivara.Samples/Gpu/GemmKernels.cs` — 6 kernel bodies (addressing only) + the section
  banner, whose "the incumbent Row4 escapes only because its A tile is 16x16" comment becomes false
  reasoning and must go.
- `tests/Nivara.PerformanceTests/GemmBenchmark.cs` — fingerprint capture/compare, load-failure
  diagnostics, one extra failure reason in the exit code, `CLDevice.DeviceVersion` in the header.
- `tests/Nivara.PerformanceTests/gemm-f32-baseline.json` — new, generated pre-migration.
- `tests/Nivara.Tests/Gpu/SharedMemoryAllocationTests.cs` — new guard.
- `docs/ACCELERATION.md` (lesson 19), `tests/Nivara.PerformanceTests/README.md`,
  `docs/ILGPU.md` (§6 gotchas row, pin-bump re-check).

**Not touched, and why it stays correct**

- `GpuBuffers.GemmGeometry` / `SharedBytes` / `GemmCfg` — geometry, group size and shared footprint
  are unchanged. `Row4Geometry.SharedBytes` = `(16*16 + 16*64) * 4` = 5120 B, and the single
  allocation is `16*16 + 16*64` = 1280 floats = the same 5120 B. No new memory, no change to
  `ValidateGemmSharedMemory`'s contract.
- The four #440 geometries — already on the single-allocation shape; they stay as the measured
  negative baseline, and their byte-identity check against Row4 keeps its meaning (Row4 is
  unchanged in results, which is exactly what the fingerprint proves).
- `AttentionKernels` — one shared allocation already.
- `GemmLegBenchmark` — profiling, not a gate; loads `TiledGemmKernelRow4` and inherits the fix.
- No CHANGELOG entry: GPU/ILGPU sample work is not tracked there (verified — no `ILGPU`, `#435` or
  `#440` entries in `CHANGELOG.md`).

**Verification coverage**

- `tests/Nivara.PerformanceTests --gemm` — 171 cells (23 shapes × 3-4 plain + 4 #440), tolerance
  `1e-3` vs double-precision truth, byte-identity of the #440 cells against Row4, and (new) an
  f32 bit-exact fingerprint of every cell. GPU + AC power required.
- `tests/Nivara.Tests` — the new IL-scan guard. No GPU required.
- Not covered: the ILGPU pin/driver matrix (one device: Intel Arc iGPU, OpenCL). Any claim here is
  scoped to ILGPU 1.5.3 + this driver, and the fingerprint is what makes a future bump detectable.

## Proposed changes

### 1. Bit-exact f32 baseline for the `--gemm` gate (no kernel changes)

`GemmBenchmark.cs` hashes each cell's read-back with FNV-1a 64 over
`BitConverter.SingleToInt32Bits`, keyed `{variant, shape}`, into
`tests/Nivara.PerformanceTests/gemm-f32-baseline.json`. `--write-gemm-baseline` regenerates it
loudly; a normal run only ever compares, so the gate can never rewrite its own reference. A
*missing* baseline entry gets its own counter and its own sentence, kept off the numeric tolerance
counter so a structural gap is never reported as a number — the separation the gate already uses
for unhostable-skip vs compile-failure. An unconditional summary line follows the existing
`#440 coverage:` precedent: `fingerprint: 171/171 cells bit-exact`, or the failing cells by name.
Mismatches join the exit code beside the existing four reasons.

Generated from **unmodified** kernels, so it is true by construction, and it is what makes the
migration in step 2 falsifiable.

### 2. Stage every GEMM tile pair through one allocation

For the Row4 family (`KT = TileSize = 16`, `TileCols = TileSize * BlockCols = 64`):

```csharp
const int ASz = TileSize * KT;                                  // 256
var smem = SharedMemory.Allocate<float>(ASz + KT * TileCols);    // 1280 floats = 5120 B
// stage: aTile[x, y]               -> smem[x * KT + y]
//        bTile[x, y * BC + w]       -> smem[ASz + x * TileCols + (y * BC + w)]
// read:  aTile[x, k]               -> smem[x * KT + k]
//        bTile[k, bBase + i]        -> smem[ASz + k * TileCols + bBase + i]
```

`TiledGemmKernel` (1x1) is the same with `KT = TileSize` and `TileCols = TileSize` →
`SharedMemory.Allocate<float>(512)`.

Same allocation size as before, identical barrier placement, identical bounds guards, identical
strictly-ascending-K accumulation. **Only the shared-memory address expressions change**, so the
output must be bit-identical — which step 1's fingerprint asserts, cell by cell. Tolerance-green
is *not* the evidence for this step; a fingerprint mismatch is the failure signal that matters.

If any GMAC/s cell moves outside run-to-run noise, the 2026-09-29 ratio table in
`tests/Nivara.PerformanceTests/README.md` is superseded and is updated **in this same commit**.

### 3. Stop dropping the OpenCL error code on a kernel-load failure

One `Describe(Exception)` helper that appends `CLError` for `CLException`, used by the LOAD-FAIL
line (`GemmBenchmark.cs:205`) and the `loadFailures` list (`:204`). Add `CLDevice.DeviceVersion` to
the gate header so a load failure is attributable to a driver version. Diagnostics only.

### 4. Guard: at most one shared allocation per kernel method

`tests/Nivara.Tests/Gpu/SharedMemoryAllocationTests.cs`, beside the existing
`GpuElementwiseParityTests`. Scans the **compiled IL**, not the source: for each method in the
`Nivara.Samples.Gpu` types, read `MethodBody.GetILAsByteArray()`, walk `call`/`callvirt` tokens
through `Module.ResolveMethod`, and fail on any method with more than one call whose declaring
type is `ILGPU.SharedMemory`. Chosen over a source-text scan because it is immune to reformatting
and comments, needs no repo-path walking, and needs no ILGPU compile-time reference (match on
`DeclaringType.FullName`), so it survives a version bump. Failure message names type, method and
the offending call count, and cites #468 + lesson 19.

Stated limitation, in the test's doc comment: a kernel that moved its allocation into a helper
called twice from one kernel would not be caught. No such shape exists in this tree.

### 5. Records, and one correction

- `docs/ACCELERATION.md` lesson 19 — record that the live Row4 family migrated too, and retract the
  "square A escapes" heuristic, pointing at the issue's own `1x8@KT16` row that contradicts it.
- `tests/Nivara.PerformanceTests/README.md` — the baseline file, how to regenerate it
  deliberately, the six migrated kernels.
- `docs/ILGPU.md` — §6 gotchas row gains the `Allocate2D` entry; the pin/driver-bump re-check
  (§7 item 8) names the fingerprint gate.
- `GemmKernels.cs` section banner — drop the false "escapes because its A tile is 16x16" reasoning.

### 6. Upstream report — drafted for the human to send

A self-contained report as a draft in the hand-off: staging-only repro kernel (no GEMM math, one
group of 256, deterministic), the four-geometry table, ILGPU 1.5.3, device / driver /
`MaxSharedMemoryPerGroup` / `MaxNumThreadsPerGroup`, the single-`Allocate` control, the
`1x8@KT16` `LoadKernel` failure **with its `CLError` value** (needs step 3 run first), and whether
one-`Allocate` is upstream's recommended shape. Target: NCSA/ILGPU. **The human posts it** — no
agent writes to a third-party tracker.

## Verification steps

Ask before each GPU or test run; capture the process exit status, never a pipeline's.

1. `dotnet build Nivara.slnx` — clean.
2. **Pre-migration** `--gemm` on AC power: 10 kernels loaded, 0 tolerance failures, 0 byte-identity
   failures, 0 load failures, `#440 coverage: all 4 tile geometries loaded`, exit 0. A green
   pre-migration run is what makes the fingerprints meaningful.
3. Step 1: `--gemm --write-gemm-baseline`, then a normal `--gemm` compare run green.
4. Step 2: `--gemm` green **with `fingerprint: 171/171 cells bit-exact`**. Also diff the AC-power
   GMAC/s table pre/post.
5. Step 4: targeted `dotnet test` for the new fixture.
6. `git diff main...khurram/468` reviewed against this file before removal (G2).

## Planned commits

1. `docs: plan the #468 Allocate2D removal in TODO.md`
2. `Add a bit-exact f32 baseline to the --gemm gate`
3. `Stage every GEMM tile pair through one shared allocation`
4. `Report the OpenCL error code when a GEMM kernel fails to load`
5. `Test: at most one shared allocation per GPU kernel method`
6. `Record the #468 migration and retract the square-A heuristic`

## GitHub issues log

- [x] #468 — ILGPU OpenCL `Allocate2D` lowering bug (this branch; upstream, mitigated in-repo)

Reminder: as each task executes, if deferred work or a concern appears outside this plan, create the
issue immediately with `gh issue create --repo khurram-uworx/Nivara` and record the number above —
do not rely on memory.
