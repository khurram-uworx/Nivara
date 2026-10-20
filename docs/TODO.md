# Plan: #682 — Silk decode-shape kernel geometry as same-session A/B

## Problem

#670 fused the Silk cached-decode projections into one `rms_norm_gemv` launch per projection and
then decode-shaped that kernel. The plan's Speed gate ran the `docs/QWEN.md` benchmark block once
per stage, three rounds in all, and could not attribute any of the three stages: the controls did
not track the measured path, the block runs Qwen only, and C3's launch-count reduction is on
SmolLM, which the block never runs.

What #670 could state structurally: 11 kernel launches per cached layer down to 8 (C1), SmolLM's
head from 2 to 1 (C3), and bit-identity against the kernels replaced, asserted by
`tests/Nivara.Tests/Gpu/SilkRmsNormGemvTests.cs` on a real device. No timing number accompanies it.

## Proposed changes

Add a same-session A/B cell to `tests/Nivara.PerformanceTests/SilkKernelsGate.cs`:

- Time `rms_norm_lm_head` (tile form, `aRows = 1`) against `rms_norm_gemv` (flat form, one slot)
  in one process, at the Qwen head shape (hidden 896, vocab 151936) and at a couple of small
  widths.
- Interleave the two routes and alternate which one is measured first, so load drift cannot
  systematically penalise the first-measured route; assert on the median of the per-round ratios
  rather than on best-of-N taken separately per route (the pattern already used by
  `TransposeKernelProbe`).
- The pair is safe to compare: `RmsNormGemv_OneBiasFreeSlot_MatchesSingleProjectionKernel`
  already asserts the two produce identical bits, so any timing difference is a scheduling
  difference, not a correctness one.

C1/C2's replacements no longer have a counterpart in the tree (`rms_norm_gemm_bias` was deleted by
#670), so their geometry can only be re-measured from `main` or left unmeasured. C3's change is
the one directly measurable this way.

## Implementation

In `SilkKernelsGate.Run(string[])`, after `MeasureSmolLMKernels(silk, buffers)` and before
`return Report(legs);`, call a new `MeasureDecodeShapeAb(silk, buffers)`. The method prints its
own section and adds no `Leg`s (timing-only; gate exit code stays correctness-only).

Method design (mirror `ApplyMaskProbe` / `TransposeKernelProbe`):

- Shapes: `(hidden=896, width=151936)` (Qwen head), `(896, 96)`, `(896, 24)`.
- For each shape:
  - Allocate `x[hidden]`, `gamma[hidden]`, `weight[hidden*width]`, `outTile[width]`,
    `outFlat[width]` via `silk.Allocate`/`Upload` (registered in `buffers` for cleanup).
  - `runTile`: `SilkLauncher.RmsNormLmHead(silk, sx, sg, sw, outTile, 1, hidden, width, eps)` + `Finish()`.
  - `runFlat`: `SilkLauncher.RmsNormGemv(silk, sx, sg, [new GemvSlot(sw, 0, outFlat, width)], false, 1, hidden, eps)` + `Finish()`.
  - Warm both routes `Warmups=5` (untimed).
  - `Rounds=10` interleaved; alternate first-measured route each round.
  - Per round record `Stopwatch` ticks; per-round ratio = tileTicks/flatTicks; skip zero-denom.
  - Report median of per-round ratios, plus median ms per route.
- Use existing private `Time(Action)` and `Median(double[])`; fully-qualify `System.Diagnostics.Stopwatch`.

## Blast radius

- `tests/Nivara.PerformanceTests/SilkKernelsGate.cs` only. No source, no doc/ADR/count pins.
- Exit code unaffected (correctness-only).

## Verification steps

- `dotnet build Nivara.slnx` (Release).
- `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --silk-kernels` on a GPU machine.

## Planned commits

1. `docs: plan #682 in TODO.md`
2. `test(perf): add decode-shape A/B cell to SilkKernelsGate (#682)`
3. `docs: remove TODO.md — plan executed`

## GitHub issues log

- (none created yet)

## Open question (from issue)

Whether #670 needed a timing number at all, or mechanism + bit-identity is the right bar. This change
supplies the number but sets no threshold; verdict left to human.