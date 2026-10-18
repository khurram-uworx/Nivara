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
a plan change. The benchmark never reached the point of deciding it: rounds 2 and 3 cannot attribute a
change to a commit on this machine ("Speed gate outcome" below), so the per-lane form ships as written
and the hoist stays open for whoever can measure it in one session.

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
  - Round 2 measured (see Round log): Silk plain decode 199.7 -> 157 ms/token against a control that was
    1.72x slower than round 1's; this round's three files are hash- and timestamp-verified as coming from
    the single run. The 1.27x ratio is **not attributed** to this commit — round 3 superseded that
    inference ("Speed gate outcome").
- Found while editing `docs/SILK.md` in stage 3: two of its `SilkKernels.cs:` citations are stale on
  `main` (`:69` for `KernelNames`, which stage 3 moves to 117; `:331` for `GEMM_ROW4_ACCUMULATE`, now
  381). The citation gate only requires a cited line to *exist*, so it cannot see this. Fixed in its
  own commit rather than folded into stage 3.
- Not part of any stage: `InferenceCliArgumentTests` spawned the sample with the repository as its
  working directory, and the sample resolves `samples/data/<model>/model.safetensors` against that
  directory — so on a machine keeping the gitignored qwen checkpoint (988 MB, dated 2026-10-17) two
  of its tests loaded the weights and ran real generation where they expect the model-file check:
  2 failed / 45 passed in 3 m 11 s, the first killed by the fixture's 30 s bound. The fixture now runs
  every invocation in an empty temporary directory, which is 47/47 in 10 s. Fixed in its own commit;
  the defect predates the branch and the checkpoint is what exposed it.
- **Stage 4 (C3, `ff78f9a2`)**: the cached tied-LM-head projection. Qwen's two cached heads
  (`ForwardCachedArgmax`, `ForwardCached`) drop `rms_norm_lm_head` for a one-slot `rms_norm_gemv`;
  SmolLM's two drop `rms_norm` + `gemm` for the same one slot, which is one launch fewer per token
  and removes the last write of the normalized row to memory in the cached path. Nothing is removed
  this time: `rms_norm_lm_head` still serves the prefill head and the two `ForwardLast*` heads, which
  take a row view instead of being a decode step. Verified: Release build 0 warnings; `dotnet format`
  clean; Gpu + Qwen 30 passed / 2 skipped / 0 failed, up from 29 / 2, with all six fixture cases
  running on device rather than skipping; Docs 67/67 with the unrelated untracked
  `docs/plan/TYPESAFE-LAYA.md` stashed aside for the run and restored SHA-256-identical.
  - New fixture case `RmsNormGemv_SmolLmCachedHead_MatchesRmsNormThenGemm` at widths 24, 96 and 200
    replays what SmolLM's head replaced — `rms_norm` writing the normalized row, then `gemm` reading
    it back — as two launches and asserts the one-slot fused call is bit-identical. Qwen's side of the
    head was already covered by the one-bias-free-slot case, which compares against
    `rms_norm_lm_head` itself. The identity holds because both norm kernels reduce the same way: each
    accumulates `sumSq += v * v` over ascending columns in one thread and never splits the reduction,
    and `1.0f / sqrt(z)` is `1.0f / rms` for a correctly-rounded `sqrt` — which is also why the fused
    kernel can apply the scale inside the contraction instead of to a stored row.
  - *Disclosure, recorded*: `docs/SILK.md`'s `rms_norm_gemv` row has listed the tied head among its
    fused shapes since stage 2 (`78cb8b35`), a commit before any tied-head call site existed. That is
    a capability list — the kernel takes one to three slots, and the one-slot form was implemented and
    tested in that commit — but it reads as a usage claim, and this stage is what makes it one. The
    row is left as it stands; the row that had gone stale was `rms_norm_lm_head`'s.
  - Speed gate: round 3 measured (see Round log) — 1296 ms / 185 ms/token, with the two controls
    disagreeing about the round's state. C3's Qwen-side change keeps the launch count at 1 and swaps the
    geometry, so this block is not the instrument that can see it; the launch-count reduction C3 makes is
    on SmolLM, which the block does not run. See "Speed gate outcome" for what that means for the
    branch's speed claim.

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

`docs/plan/TYPESAFE-LAYA.md` is untracked, belongs to another workstream, and carries two dangling
`docs/PLAN.md` references, so the Docs run fails on it (3 failed / 64 passed) independently of this
branch. Stash it aside for the run — `git stash push -u -- docs/plan/TYPESAFE-LAYA.md`, run the
suite, `git stash pop` — and confirm the file returns SHA-256-identical; that is how the 67/67 runs
above were taken. Do not edit or commit that file from this branch.

Not to be run by the agent without asking: the full `dotnet test` suite.

## Speed gate (human-run)

Per `docs/QWEN.md` § "Benchmark commands for agents (human-run, too slow for agents)", quoted
verbatim and run from the repo root in PowerShell, **once after commit 2, once after commit 3, once
after commit 4**. Each run deletes `data\qwen-bench` first, so the files read afterwards are from that
run only. All three rounds were run by the agent, at the user's explicit request — `docs/QWEN.md`
otherwise treats this block as human-run. What the three rounds do and do not support is in "Speed
gate outcome" below; it is less than the plan assumed when it was written.

```powershell
Remove-Item -Recurse -Force "data\qwen-bench" -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path "data\qwen-bench" | Out-Null
dotnet build -c Release Nivara.slnx
dotnet samples/NivaraInference/bin/Release/net11.0/NivaraInference.dll qwen benchmark --plain > data\qwen-bench\cpu_plain.txt 2>&1
dotnet samples/NivaraInference/bin/Release/net11.0/NivaraInference.dll qwen benchmark > data\qwen-bench\cpu_tools.txt 2>&1
dotnet samples/NivaraInference/bin/Release/net11.0/NivaraInference.dll qwen --silk benchmark > data\qwen-bench\silk_tools.txt 2>&1
```

Consumed afterwards (read, never re-run without being asked): `silk_tools.txt` for decode ms/token;
`cpu_*.txt` is the unchanged control. The agent does not poll a running round, and does not re-run one on
its own initiative — each of the three rounds below was run by it at the user's explicit request. If a
round shows a regression, the staged commits exist so the offending stage can be held out.

### Round log

The 2026-10-17 figures in `docs/QWEN.md` § Results are **not** a usable baseline for this machine
state: in round 1 the control moved by the same factor as the measured path, which no Silk change can
touch. Rounds were therefore compared with each other, each read against its own control — and round 3
shows that this does not work here: the round's two controls disagree with each other about its state,
and the measured path moves *opposite* to the control in both intervals. What the table supports is a
band, not a per-commit delta. See "Speed gate outcome" below, which supersedes the round-2
attribution.

| Round | Commit | Silk plain decode | Control: CPU plain decode | Notes |
|---|---|---|---|---|
| 1 | `78cb8b35` (C1) | 1398 ms / 7 tokens, 199.7 ms/token (runs 207 / 196 / 199) | 161.9 ms/token | model build 11,347 ms; CPU tools decode 220.3 ms/token |
| 2 | `9ab34e22` (C2) | 1102 ms / 7 tokens, 157 ms/token (runs 166 / 157 / 157) | 278.9 ms/token | model build 12,448 ms; CPU tools decode 422.6 ms/token |
| 3 | `ff78f9a2` (C3) | 1296 ms / 7 tokens, 185 ms/token (runs 195 / 177 / 185) | 243.7 ms/token | model build 11,068 ms; CPU tools decode 219.4 ms/token |

Round 1 detail:

- `silk_tools.txt` is a misnomer: `qwen --silk benchmark` reports `Prompt tokens: 36`, the plain
  prompt — the surface `docs/QWEN.md:343` already labels "plain prompt". Only the file name says
  "tools". Recorded as a follow-up below.
- 199.7 ms/token is not evidence of a regression *or* of an improvement. The CPU control, which no
  Silk change can touch, is 1.41x slower today than the 2026-10-17 record (114.7 -> 161.9 ms/token)
  and the CPU plain total is 1.57x slower; Silk's model build — weight upload and program
  compilation, untouched by this change — is 2.19x slower (5,178 -> 11,347 ms). The Silk/control
  ratio is 1.23x today against 1.13x recorded, i.e. the same band — itself a ratio reading, and the round
  log now treats ratios here as uncontrolled ("Speed gate outcome"). The comparison this plan intended
  was round 2 against round 1; why that method did not hold is the outcome section's subject.
- The three generated-token count (7) and the 36-token prompt match the recorded row, so the shape
  is comparable even though the machine state is not.

Round 2 detail (its attribution is superseded by round 3 — see "Speed gate outcome"):

- Provenance: this is the first round whose three files are verifiably from one run. All four commands
  reported exit 0, and the run window is `16:18:43`–`16:31:46` (build 16:19, `cpu_plain` 16:20,
  `cpu_tools` 16:31, `silk_tools` 16:31); the three SHA-256 hashes differ from round 1's. Round 1's did
  not have that property: its `cpu_plain.txt` was stamped 00:18 while its `cpu_tools.txt` and
  `silk_tools.txt` were stamped 15:31 and 15:32, so round 1's control figure came from a different run
  than its Silk figure. Round 1's numbers still stand as measured, but its control is weaker evidence
  than it reads, and a round is only readable once this check has been made.
- Control moved *against* the measured path: CPU plain decode 161.9 -> 278.9 ms/token (1.72x slower),
  CPU tools decode 220.3 -> 422.6 ms/token (1.92x slower). C2 (`9ab34e22`) changes
  `src/Nivara.Extensions/Silk/`, `docs/SILK.md` and its tests only, and neither control command passes
  `--silk`, so this movement cannot be attributed to the branch. Rounds 1 and 2 are therefore not
  like-for-like either — but the host got slower, not faster, so the direction of the Silk change is
  still readable even though the size is not.
- Measured: Silk plain decode 199.7 -> 157 ms/token (median total 1398 -> 1102 ms, 1.27x); repeat sets
  207/196/199 and 166/157/157. The inference drawn here at the time — that the control's 1.72x slowdown
  makes 1.27x a floor rather than an effect size, because per-token cost is GPU-resident so host drift
  can move it by little more than the launch overhead — is **superseded**: round 3 moved the control the
  other way and Silk moved the other way with it. The number stands as measured; the attribution drawn
  from it does not.
- The model build (11,347 -> 12,448 ms, 1.10x) and the CPU full-forward medians moved the *other* way
  (plain 22,955 -> 20,796 ms, tools 198,115 -> 180,207 ms). A uniform machine slowdown would move all
  of these together; they disagree in sign, so this is drift between runs rather than a step change in
  machine state.
- Silk/control ratio 1.23x in round 1 (against 1.13x recorded) and 0.56x in round 2 (157 vs 278.9). The
  ratio moved mostly because its denominator moved. Ratios of this kind are recorded, not used: round 3
  shows the denominator is not a proxy for the conditions that set the numerator, so a ratio here is
  uncontrolled too.
- Attribution: commit 3 (C2) was the only Silk change between the two runs — the decode-shaped geometry
  (`GEMVLANES = 32` x `BLOCK = 4` = 128 columns/group, flat 1-D launch) in place of the tile shape. The
  attribution of the 1.27x to that geometry is **superseded** by round 3, which does not reproduce it
  under a control matching round 1's.
- What this round cannot say, restated after round 3: whether the 1.27x was the change or the machine.
  The block answers neither, by construction.

Round 3 detail:

- Provenance: all four commands reported exit 0, and the three files are stamped `16:49` (`cpu_plain`),
  `16:59` (`cpu_tools`) and `17:00` (`silk_tools`) with hashes `31D1D40B8367` / `FEA27363CCF0` /
  `A5D745127531`, none of which matches round 2's. Round 2's raw files were copied out to
  `%TEMP%\qwen-bench-round2\` before the run deleted `data\qwen-bench`, with the copy hash-verified
  against the source first, so the folder again holds one run's files and round 2's figures remain
  checkable against what produced them.
- Measured: Silk plain decode 157 -> 185 ms/token (median total 1102 -> 1296 ms, 1.18x slower), runs
  195 / 177 / 185 — an 18 ms spread, wider than round 2's 166 / 157 / 157 at 9 ms. Both are far tighter
  than the ~40 ms the median moves between rounds, which is the sample saying the within-round spread is
  not the source of the between-round differences.
- Controls, both from this run: CPU plain decode 278.9 -> 243.7 ms/token (1.13x faster than round 2, still
  1.51x round 1's 161.9; KV-cache median 3047 ms for 7 tokens, prefill 1342 ms) and CPU tools decode
  422.6 -> 219.4 ms/token (1.93x faster than round 2 and 1.00x round 1's 220.3; KV-cache median 8191 ms
  for 19 tokens, prefill 8647 ms). The two controls disagree about this round's state: the tools control
  says the decode conditions are back to round 1's, the plain control says they are still 1.51x off.
  Nothing else was recorded about the host, so nothing breaks the tie.
- The sign is wrong for control-normalization again: in round 1 -> 2 the control got slower and Silk got
  faster, in round 2 -> 3 the control got faster and Silk got slower. Two intervals is thin evidence, but
  it points away from the assumption the round-2 record relied on — that the CPU control tracks whatever
  sets the Silk number. Full-forward medians moved as well (plain 20,796 -> 19,334 ms, tools 180,207 ->
  185,525 ms), in opposite directions to each other.
- Scope: this block runs Qwen only. C3's Qwen-side change is a geometry swap with **no** change in launch
  count (the head was already one launch on Qwen), while C3's launch-count reduction (SmolLM's head, 2 -> 1
  per layer) is on a model this block never runs. Round 3 therefore has little to say about C3's mechanism
  even on a quiet machine — and round 3's job was to measure C3.
- What this round cannot say: whether the 1.18x slowdown is C3, the machine, or a mix. Given rounds 1-2,
  "the machine" is the better-supported of the three.

### Speed gate outcome

Three rounds, one per stage, all run by the agent at the user's request:

| Round | Commit | Silk plain decode | Control: CPU plain | Control: CPU tools |
|---|---|---|---|---|
| 1 | `78cb8b35` (C1) | 199.7 ms/token | 161.9 ms/token | 220.3 ms/token |
| 2 | `9ab34e22` (C2) | 157 ms/token | 278.9 ms/token | 422.6 ms/token |
| 3 | `ff78f9a2` (C3) | 185 ms/token | 243.7 ms/token | 219.4 ms/token |

- **No per-commit attribution is supported.** Silk varies by 1.27x across rounds; the controls vary by up
  to 1.93x; the two controls disagree with each other about round 3; and the measured path moves opposite
  to the control in both intervals. On this machine the block cannot separate a ~10-30% effect from state
  it does not measure. Round 2's 1.27x is recorded as measured and no longer as attributable.
- **No regression is observed.** Every round's Silk decode lies in a 157-200 ms/token band, and no round's
  Silk figure is worse than round 1's while its own control was also worse. That is the strongest
  statement the data carries, and it is a weak one.
- **The mechanism claim does not rest on the block.** C1 takes the cached layer from 11 kernel launches to
  8 (`SilkQwenGpuRunner`/`SilkSmolLMGpuRunner`) and C3 takes SmolLM's head from 2 to 1; both are readable
  in the source, and the fused kernel reproduces the kernels it replaced bit for bit (the fixture plus the
  `--silk-kernels` byte-identity posture). The Speed gate was to put a number on that. It did not.
- **The instrument that can attribute it** is a same-session A/B: the competing kernels timed against each
  other in one process, interleaved and repeated, so state drift hits both sides. `rms_norm_lm_head` (tile)
  and `rms_norm_gemv` (flat, one slot) coexist after C3, so C3's geometry change is directly measurable
  that way — the same pair the fixture already asserts is bit-identical. C1/C2's replacements no longer
  have a counterpart in the tree (`rms_norm_gemm_bias` was deleted), so their geometry would have to be
  re-measured from `main` or left unmeasured. Adding that cell to
  `tests/Nivara.PerformanceTests/SilkKernelsGate.cs` and running it is a human decision, as is whether
  #670 ships with a timing number at all — this plan is the only place one was promised.

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
