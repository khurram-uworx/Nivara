# Plan: perf(silk): Phase C - update Silk tests to BFloat16 helpers (#1142)

Follow-up to #1138 (phase C). Branch: khurram/1142 (base khurram/1141).

## Problem
Silk tests still use hand-rolled BF16 conversions (`>> 16` / `<< 16`) instead of the
.NET 11 `System.Numerics.BFloat16` / `BitConverter` helpers. This risks drift between
test expectations and the canonical BFloat16 conversion.

## Proposed changes
- `tests/Nivara.Tests/Gpu/SilkBf16BackwardProbeTests.cs`: `ToBf16`/`FromBf16` use
  `BitConverter.BFloat16ToUInt16Bits((BFloat16)f)` and
  `(float)BitConverter.UInt16BitsToBFloat16(u)`.
- `tests/Nivara.Tests/AutoDiff/SafeTensorsLoaderBf16Tests.cs`: three `<< 16` sites
  use `(float)BitConverter.UInt16BitsToBFloat16(...)`.

## Verification
- Build: `dotnet build Nivara.slnx`
- Targeted tests (Release): SilkBf16BackwardProbeTests, SafeTensorsLoaderBf16Tests
- Parity tests remain bit-identical (no tolerance relaxation).

## Blast radius
- Test-only; no source/runtime changes. No docs pins affected.

## GitHub issues log
- (none yet)
