# Plan: perf(silk): Phase A — SilkRuntime BFloat16 overloads + SilkBertWeights widen centralization (#1140)

## Problem
Issue #1140 (follow-up to #1138). The true BF16 path stores device weights as `ushort[]` and converts at boundaries using ad-hoc bit shifts (`(uint)u << 16`) and `MemoryMarshal.Cast`. .NET 11 provides `System.Numerics.BFloat16` plus `BitConverter` helpers; centralize BFloat16 handling in `SilkRuntime` and use the helpers for clarity, with no numeric change.

## Proposed changes

### SilkRuntime (`src/Nivara.Extensions/Silk/SilkRuntime.cs`)
- Add `using System.Numerics;` and `using System.Runtime.InteropServices;`.
- `Upload(nint, ReadOnlySpan<BFloat16>)` — delegates to the ushort overload via `MemoryMarshal.Cast`.
- `DownloadBFloat16(nint, int)` returning `BFloat16[]` — read ushorts then cast.
- `AllocateBFloat16(int)` aliasing `AllocateUshort`.
- `EnsureBFloat16(SilkBuffer, int)` aliasing `EnsureUshort`.

### SilkBertWeights (`src/Nivara.Extensions/Silk/SilkBertWeights.cs`)
- Narrow constructor widen lambda: BF16 uses `(float)BitConverter.UInt16BitsToBFloat16(u)` instead of `BitConverter.UInt32BitsToSingle((uint)u << 16)`.
- `ExportCore` narrow-matrix branch: BF16 uses `DownloadBFloat16` + `(float)`; FP16 keeps `DownloadUshort` + `Int16BitsToHalf`. No manual shift.
- Document `NewParamNarrow` storage (ushort holds BF16/FP16 bit-pattern).

### Trainer (`src/Nivara.Extensions/Silk/SilkDistilBertTrainer.Bf16True.cs`)
- Document the blittable cast site in `ToUshort`.

## Verification
- `dotnet build Nivara.slnx`
- Targeted Silk tests: `SilkDistilBertTrainerTrueBf16SmokeTests`, `SilkBertNarrowParityTests`, `SilkDistilBertTrainerBf16ParityTests`.

## Planned commits
1. docs: plan issue #1140 in TODO.md
2. feat(silk): add BFloat16 overloads to SilkRuntime
3. refactor(silk): use BitConverter BFloat16 helpers in SilkBertWeights
4. docs(silk): document blittable cast in trainer ToUshort
5. docs: remove TODO.md — plan executed

## Blast radius
- `SilkRuntime` public API additions (additive, no break).
- `SilkBertWeights` internal; widen results bit-identical (verified: 0 mismatches across all 65536 ushort patterns).
- Trainer internal conversion unchanged behavior.

## GitHub issues log
(none)
