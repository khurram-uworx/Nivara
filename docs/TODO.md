# Plan: remove vision Silk BF16 runner shims (#1282)

## Problem

`SilkMobileNetV2GpuRunner.Bf16.cs` and `SilkResNet18GpuRunner.Bf16.cs` define public
constructors accepting `BFloat16[]` tensors, but both just call `ToFloat(...)` and delegate to
the F32 runner. There is no true narrow BF16 vision lane (CPU vision is F32-only, and the CLI
already rejects `--precision bf16/fp16` for vision models). The shims advertise a capability
that does not exist.

## Blast radius

- `src/Nivara.Extensions/Silk/SilkMobileNetV2GpuRunner.Bf16.cs` (constructor only)
- `src/Nivara.Extensions/Silk/SilkResNet18GpuRunner.Bf16.cs` (constructor only)
- `samples/NivaraInference/Program.cs` (dead `bf16 ? WidenBFloat16Tensors(tensorsBf16) : F32()` branches for mobilenet_v2 / resnet18 Silk paths)
- `docs/SILK.md` (prose implying bf16 routes for vision)
- `docs/plan/NIVARAINFERENCE-SILK-GAPS.md` (Gap N3 status)

No test references the removed constructors (verified via code-memory impact analysis: zero
downstream dependencies). No other source/sample references them.

## Proposed changes

1. Delete the two `*.Bf16.cs` shim files.
2. In `Program.cs`, replace the dead conditional with `F32()` for both vision Silk branches.
3. Clarify `docs/SILK.md` that vision runners are F32-only.
4. Update `docs/plan/NIVARAINFERENCE-SILK-GAPS.md` Gap N3 to note removal.

## Verification steps

- `dotnet build Nivara.slnx`
- `dotnet build samples/NivaraInference/NivaraInference.csproj`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~InferenceCliArgumentTests"`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Docs snippet gate: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~EveryGatedBlock_CompilesWithoutErrors"`

## Planned commits

1. `docs: plan #1282 in TODO.md`
2. `refactor(silk): remove vision BF16 runner shims that only widen to float`
3. `refactor(inference): drop dead BF16 widen branches for vision Silk paths`
4. `docs: clarify vision Silk runners are F32-only`
5. `docs: remove TODO.md — plan executed`

## GitHub issues log

- (none created yet)
