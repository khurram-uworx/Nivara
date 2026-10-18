# Plan — Silk cached decode: fuse the projection norms and remove the idle-lane geometry (#670)

Branch: `khurram/670`. Base: `main` @ `91241820`.

As each task executes, if deferred work or a concern surfaces, create a GitHub issue immediately
(`gh issue create --repo khurram-uworx/Nivara`) and record its number in the GitHub issues log below.
Do not rely on memory — compaction during execution loses it.

## Problem

Issue #670: after #657, Silk's *cached* (single-token) decode still runs one norm-fused kernel per
projection.

- Q/K/V launch three fused kernels (`rms_norm_gemm_bias` on Qwen, `rms_norm_lm_head` on SmolLM),
  each recomputing `RMSNorm(x)` and re-reading `x` from global.
- gate/up launch two fused kernels (`rms_norm_lm_head`), each recomputing the same post-attention
  `RMSNorm`.

So #657's "norm weights absorbed into a fused entry" is only half realized: the norm is absorbed
per projection, but recompute and repeated `x` reads remain, and each projection costs its own
enqueue + set-arg + status round trip.

Second, larger problem found while mapping the call sites and confirmed by reading the kernels:
**every one of these launches is `aRows == 1`, and the kernel geometry is built for 16 rows.**
`GemmGlobalRows(1) == 16` over a local size of `[16, 16]`, so one work-group carries 16 rows of
which exactly one is live. The other 15 still execute the whole K-loop against zeroed A tiles and
cannot return early (they share barriers with the live lane). Lane utilization is 16/256 = **6.25%**.
The same shape wastes the tied LM head, which is the largest single decode GEMM.

### Volume at Qwen 2.5-0.5B (24 layers, hidden 896, inter 4864, vocab 151 936), per token

| decode GEMM | MACs | current kernel | after this plan |
|---|---|---|---|
| Q/K/V | 25 M | `rms_norm_gemm_bias` (Qwen) / `rms_norm_lm_head` (SmolLM) | `rms_norm_gemv`, 3 slots |
| gate + up | 210 M | `rms_norm_lm_head` | `rms_norm_gemv`, 2 slots |
| tied head | 136 M | `rms_norm_lm_head` (Qwen) / `RmsNorm`+`Gemm` (SmolLM) | `rms_norm_gemv`, 1 slot |
| down | 105 M | `gemm_row4_add` | unchanged — follow-up issue |
| o_proj | 19 M | `gemm_row4_add` | unchanged — follow-up issue |

Per-layer commands in the cached loop: 13 today (3 QKV + 2 gate/up + 2 rotary + 2 KV-cache copy +
1 attention + 1 o_proj add + 1 SwiGLU + 1 down add) -> **10** after the fusion — which is 11 -> 8
kernel launches, because two of those commands are `runtime.CopyBuffer` calls, not launches. Head
launches per token: Qwen 1 -> 1 (kernel swapped), SmolLM 2 -> 1 (the `RmsNorm` + `Gemm` pair
collapses into the fused call).

## Design

### New kernel: `rms_norm_gemv`

One decode-shaped (row-1) kernel with up to three output slots. 1-D launch; each lane owns
`BLOCK = 4` consecutive output columns; K is visited strictly ascending with a single accumulator per
column — the same contraction the tile kernels perform, which is what makes the result bit-identical
(see below).

```
__kernel void rms_norm_gemv(
    __global const float* x, __global const float* gamma,
    __global const float* w0, __global const float* b0, __global float* out0,
    __global const float* w1, __global const float* b1, __global float* out1,
    __global const float* w2, __global const float* b2, __global float* out2,
    const uint aCols, const uint cols0, const uint cols1, const uint cols2,
    const uint hasBias, const float eps)
```

- `normScale` computed from the row exactly as `rms_norm` / `rms_norm_lm_head` do (ascending-k single
  accumulator, `1.0f / sqrt(ssq / aCols + eps)`).
- Group -> `(slot, colBase)` mapping through per-slot group counts, so no work-group straddles two
  slots: the group index walks slot 0's groups, then slot 1's, then slot 2's. There is no `nSlots`
  argument — a slot is reachable iff its column count is positive, and a separate count could
  disagree with the widths it is meant to describe.
- The decode kernel drops `aRows` (the launcher asserts it is 1 and the tile-shaped C1 body's
  `outRow < aRows` guards were the only use). The *launcher* signature keeps `aRows`, so the call
  sites do not move between C1 and C2.
- Per group, `32 lanes x 4 columns = 128 columns`, which divides every Qwen width exactly
  (896/128 = 7 + 1 + 1 = 9 groups for QKV, 4864/128 = 38 for gate/up, 151936/128 = 1187 for the
  head; SmolLM 576 and 192 produce a tail group, whose surplus lanes the `col >= bCols` early return
  discards).
- Slot selection uses the existing pointer pattern; `hasBias == 0` means the store is the plain
  accumulator (no `acc + 0.0f`, which would canonicalize `-0.0f` and break identity).
- `hasBias` precedent: `conv2d` already carries this exact flag
  (`SilkKernels.cs` conv2d arg 18) and `SilkResNet18GpuRunner.cs:111` passes `0` for an absent bias
  buffer, so a null mem-arg on an unused slot is already proven on this runtime.

### Bit-identity argument (the evidence a tolerance cannot give)

Read from the source:

- `rms_norm` computes `rms = sqrt(sumSq / (float)cols + eps); inv = 1.0f / rms;` then
  `y = x * inv * gamma[c]`. `rms_norm_lm_head` computes
  `normScale = 1.0f / sqrt(ssq / (float)aCols + eps)` and stages
  `x[...] * normScale * gamma[aCol]`. Same `sqrt`, same reciprocal, same left-to-right product
  order => the same float values.
- Both reduce `ssq` sequentially over ascending `k` in one accumulator, and accumulate each output
  element in one register over ascending `k`, with `#pragma OPENCL FP_CONTRACT OFF` at the top of the
  translation unit keeping `a * b + c` unfused.
- `rms_norm_gemv` keeps exactly that: one accumulator per output column, ascending `k`, identical
  value text. Only *which lane* computes an element changes — not the arithmetic proving it.

Consequences: the exact-equality tests can assert **bit-pattern** equality
(`BitConverter.SingleToInt32Bits`), and the `--silk-kernels` byte-identity posture is untouched.
If a measurement ever shows a difference, it is a defect, not rounding.

### Norm-reduction placement (open micro-choice, benchmark-decided)

Per-lane redundant reduction (no barrier, `x` L1-hot, ~25% extra issue) versus one lane reducing and
broadcasting through `__local` (also bit-identical, adds a serial `aCols`-long add chain per group).
Start per-lane; if the C2 benchmark shows the norm dominating, hoist it — that is a C2 follow-up, not
a plan change.

### Removals (only non-additive part)

- `SilkKernels.RmsNormGemmBias` constant, its `KernelNames` entry, and `SilkLauncher.RmsNormGemmBias`
  — the Qwen cached Q/K/V calls were its only call sites.
- The `rms_norm_gemm_bias` row in `docs/SILK.md` § kernel table.
- Fix the now-false clause in the `rms_norm_lm_head` row ("also used for bias-free cached decode
  projections") — SmolLM's cached Q/K/V move to the new kernel.

## Commits

| # | Commit | Content | Guard |
|---|---|---|---|
| 1 | `docs: plan the Silk cached-decode projection fusion in TODO.md` | this file | — |
| 2 | `feat(silk): fuse cached-decode QKV and gate/up norms into one entry each` | `rms_norm_gemv` (tile-shaped: the existing body text plus a per-slot group mapping), `SilkLauncher.RmsNormGemv`, the 8 call sites in the 4 cached methods, remove `rms_norm_gemm_bias`, SILK.md table, CHANGELOG, exact-equality fixture | new bit-exactness test + catalog test |
| 3 | `perf(silk): decode-shape the fused projection kernel` | the launcher signature is unchanged; the kernel body and grid become the row-1 GEMV, and the launcher asserts `aRows == 1`; device-free geometry test | commit 2's bit-exactness test (must stay green unchanged) |
| 4 | `perf(silk): decode-shape the cached tied-LM-head projection` | Qwen cached heads -> new kernel; SmolLM cached head `RmsNorm`+`Gemm` -> one call; head case in the fixture; SILK.md wording; CHANGELOG | fixture head case |
| 5 | `docs: remove TODO.md — plan executed` | after both G2 reviews clear | — |

Note on commit 2: it is deliberately tile-shaped — the same body and the same total work-group count
as the launches it replaces, with only the enqueue count reduced (3 -> 1 for QKV, 2 -> 1 for
gate/up). Commit 3 then changes the geometry, which is the lane-utilization win. Keeping them apart is
what attributes each win to its own benchmark round; merging them would report one number for two
effects. One kernel throughout (not a `rms_norm_qkv`/`rms_norm_gate_up` pair that a later commit
deletes): the multi-slot factoring is identical in both shapes, and a name that lives for one commit
is a name nobody can find again.

## Stage log

- **Stage 2 (C1, `78cb8b35`)**: tile-shaped `rms_norm_gemv`, eight cached call sites, exact-equality
  fixture, SILK.md table, CHANGELOG. Verified: Release build clean (0 warnings); `dotnet format`
  clean; Gpu+Qwen 29 passed / 2 skipped; the new fixture 3/3 on a real OpenCL device; Docs 67/67.
  Round 1 measured (see Round log).
- **Stage 3 (C2)**: decode-shaped. New `GEMVLANES` 32 / `GEMVCOLS` 128 macros, the
  `GemvGroupSize` / `GemvBlockCols` / `GemvGroupCols` constants and `GemvGlobalCols`; the kernel
  dropped `aRows` and the shared-memory tile; the launcher rejects `aRows != 1`. The launcher
  signature is unchanged, so the eight call sites and the C1 fixture's assertions did not move.
  - *Deviation from this plan, recorded*: the fixture did gain two tests. C1's widths are all whole
    lanes (96 / 24 / 40 / 200), so nothing exercised the per-lane `avail` bound that keeps a lane
    whose span runs past the width in bounds. Rather than leave that path unverified the fixture
    gained a biased 25/26/27 case and a bias-free 27 case; its three C1 assertions are untouched and
    still pass bit-identical. The fixtures' `<remarks>` paragraph was rewritten because it named C1's
    guard expressions (`outCol < bCols` and friends), which C2 removed.
  - *Choice, recorded*: the `avail` guards were kept rather than replaced by a `cols % BLOCK == 0`
    precondition on the launcher. The tile kernel they replace guards each of its four columns the
    same way, so this is the house pattern, and three loop-invariant predicates cost nothing in a
    kernel that reads four floats per multiply-add.
  - New device-free `SilkGemvGeometryTests`: macro/constant agreement, whole-group arithmetic,
    zero idle lanes on every Qwen width, the SmolLM tail, no `aRows` in the fused kernel's signature
    against `aRows` still in the tied-head tile kernel's, and the batch rejection.
    `DocReferenceTests.ExpectedFileCount` 711 -> 712 with the arithmetic extended.
- Found while editing `docs/SILK.md` in stage 3: two of its `SilkKernels.cs:` citations are stale on
  `main` (`:69` for `KernelNames`, which stage 3 moves to 117; `:331` for `GEMM_ROW4_ACCUMULATE`, now
  381). The citation gate only requires a cited line to *exist*, so it cannot see this. Fixed in its
  own commit rather than folded into stage 3.

## Verification

Run by the agent (no prompting needed for targeted runs):

```
dotnet build -c Release Nivara.slnx
dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore
dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Gpu|FullyQualifiedName~Nivara.Tests.Qwen"
dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"
```

The Docs run is the whole namespace because `docs/SILK.md` changes (six pinned counts live in two
fixtures). The new SILK.md rows carry no `File.cs:NN` citation and no `docs/….md` reference, so no
pinned count should move; if one does, that is a finding, not a bump.

Not to be run by the agent without asking: the full `dotnet test` suite.

## Speed gate (human-run)

Per `docs/QWEN.md` § "Benchmark commands for agents (human-run, too slow for agents)", quoted
verbatim and run from the repo root in PowerShell, **once after commit 2, once after commit 3, once
after commit 4**. Each run deletes `data\qwen-bench` first, so the files read afterwards are from that
run only.

```powershell
Remove-Item -Recurse -Force "data\qwen-bench" -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path "data\qwen-bench" | Out-Null
dotnet build -c Release Nivara.slnx
dotnet samples/NivaraInference/bin/Release/net11.0/NivaraInference.dll qwen benchmark --plain > data\qwen-bench\cpu_plain.txt 2>&1
dotnet samples/NivaraInference/bin/Release/net11.0/NivaraInference.dll qwen benchmark > data\qwen-bench\cpu_tools.txt 2>&1
dotnet samples/NivaraInference/bin/Release/net11.0/NivaraInference.dll qwen --silk benchmark > data\qwen-bench\silk_tools.txt 2>&1
```

Consumed afterwards (read, never re-run): `silk_tools.txt` for decode ms/token; `cpu_*.txt` is the
unchanged control. The agent never polls or re-runs. If a round shows a regression, the staged commits
exist so the offending stage can be held out.

### Round log

The 2026-10-17 figures in `docs/QWEN.md` § Results are **not** a usable baseline for this machine
state: in round 1 the control moved by the same factor as the measured path, which no Silk change can
touch. Rounds are therefore compared with each other, in the same conditions.

| Round | Commit | Silk plain decode | Control: CPU plain decode | Notes |
|---|---|---|---|---|
| 1 | `78cb8b35` (C1) | 1398 ms / 7 tokens, 199.7 ms/token (runs 207 / 196 / 199) | 161.9 ms/token | model build 11,347 ms |

Round 1 detail:

- `silk_tools.txt` is a misnomer: `qwen --silk benchmark` reports `Prompt tokens: 36`, the plain
  prompt — the surface `docs/QWEN.md:343` already labels "plain prompt". Only the file name says
  "tools". Recorded as a follow-up below.
- 199.7 ms/token is not evidence of a regression *or* of an improvement. The CPU control, which no
  Silk change can touch, is 1.41x slower today than the 2026-10-17 record (114.7 -> 161.9 ms/token)
  and the CPU plain total is 1.57x slower; Silk's model build — weight upload and program
  compilation, untouched by this change — is 2.19x slower (5,178 -> 11,347 ms). The Silk/control
  ratio is 1.23x today against 1.13x recorded, i.e. the same band. The valid comparison is round 2
  against round 1.
- The three generated-token count (7) and the 36-token prompt match the recorded row, so the shape
  is comparable even though the machine state is not.

## Blast radius

Source:

- `src/Nivara.Extensions/Silk/SilkKernels.cs` — constants, `KernelNames`, `Source` (new kernel, remove
  one), geometry helper.
- `src/Nivara.Extensions/Silk/SilkLauncher.cs` — new launcher(s), remove one, new grid helper.
- `src/Nivara.Extensions/Silk/SilkQwenGpuRunner.cs` — cached paths only (`ForwardCached`,
  `ForwardCachedArgmax`); prefill/`Forward`/`ForwardLast*` untouched.
- `src/Nivara.Extensions/Silk/SilkSmolLMGpuRunner.cs` — same.

Downstream callers: `SilkQwenSession`, `SilkSmolLMSession` (public surface unchanged), the CLI
`qwen --silk` path in `samples/NivaraInference`. No public API change.

Tests:

- `tests/Nivara.Tests/Gpu/SilkKernelsCatalogTests.cs` — automatically requires each new name to appear
  as `__kernel void <name>(`.
- New: `tests/Nivara.Tests/Gpu/SilkRmsNormGemvTests.cs` (device-gated, `Assert.Ignore` without a
  device) — bit-pattern equality against the single-projection kernels, including tail widths. (The
  plan first named this file `SilkDecodeProjectionTests.cs`; it took the kernel's name instead.)
- New: device-free geometry assertions for the per-slot group/column arithmetic — added with the
  decode-shaped geometry (commit 3), since the tile-shaped kernel uses the existing `GemmGlobalCols`.
- `tests/Nivara.Tests/Qwen/SilkQwenParityTests.cs`, `SilkQwenKvCacheTests.cs`,
  `SilkQwenToolsWeatherLoopTests.cs` — must stay green (self-skip without a device).
- `tests/Nivara.PerformanceTests/SilkKernelsGate.cs` `MeasureSmolLMKernels` — two `MeasureCpuReference`
  cells for the new kernel (human-run harness).

Docs: `docs/SILK.md` kernel table, `CHANGELOG.md` Unreleased/Changed.

Risk: the identity argument is only as strong as the fixture — tail-width shapes must be in it, or a
column-index guard bug hides. `tests/Nivara.Tests` already has internals access
(`InternalsVisibleTo` in `Nivara.Extensions.csproj`), so the fixture can drive `SilkLauncher` directly.

## Out of scope (tracked, not deferred silently)

- Row4 `gemm_row4_add` decode variant (down + o_proj, ~125 M MACs/token still at 6%): separate issue —
  it touches the family carrying the ILGPU byte-identity claim and the `--silk-kernels` gate.
- Qwen prefill single-row heads (`SilkQwenGpuRunner.cs` prefill/`Forward` heads) keep the tile kernel at
  6%; #657 explicitly says keep prefill individual.
- Pre-existing doc drift: the `KernelNames` quote in `docs/SILK.md` § 11.1 is stale (SmolLM/vision
  kernels missing). Untouched.
- Pre-existing doc drift found in round 1: the `docs/QWEN.md` block names the third output
  `silk_tools.txt`, but `qwen --silk benchmark` measures the **plain** prompt (36 tokens) — the same
  surface the results table labels "plain prompt". Either the file name or the table row is wrong;
  the file name is the cheaper fix. Untouched: it is a docs-block change outside #670's scope.
- Pre-existing message drift found while checking the Silk invocation was legal:
  `samples/NivaraInference/Program.cs:460` omits `qwen` from the `--silk` model list although
  `SilkModes["qwen"]` accepts it (`Program.cs:63`). Untouched.

## Probe and harness lifecycle

No new probe or harness. Step 0 of the lifecycle was satisfied: the question ("is the fused
decode-shaped kernel correct and faster") is already answered by existing homes — the device-gated
NUnit fixture for correctness/identity and `tests/Nivara.PerformanceTests/SilkKernelsGate.cs`
(`--silk-kernels`, `MeasureSmolLMKernels` cells) for the CPU-reference measurement. Both are extended
in place; no temp project is created.

## GitHub issues log

- [ ] #681 — Silk: decode-shaped `gemm_row4_add` for down/o_proj (remaining 6%-utilization decode GEMMs) (created while working on #670)
