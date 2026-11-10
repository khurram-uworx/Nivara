# Plan: test(silk) — pin BF16/F16 head kernel registration against silent widening (#1113)

## Problem

Follow-up to #1013. `SilkLayaHeadGpuRunner.LaunchGemm` maps the four head GEMM
kernels (`GemmQkv`/`GemmGelu`/`GemmRelu`/`GemmBias`) to narrow/activation-level
variants via an inline `switch`. If a case is dropped during a refactor, `mapped`
silently stays the base (F32) kernel name while `b` is a `ushort` buffer, reverting
to F32/widened behaviour with no compile-time signal. Nothing pins that the BF16/F16
variants remain in `SilkKernels.KernelNames` (registered/compiled) and reachable from
dispatch.

## Proposed changes

1. `src/Nivara.Extensions/Silk/SilkLayaHeadGpuRunner.cs`
   - Extract the inline mapping into `internal static string ChooseKernel(string kernel, bool activationBf16 = false, bool narrowBf16 = false, bool narrowF16 = false)` plus name aliases `MapKernel`/`MapGemmKernel`/`MapHeadGemmKernel`/`DispatchKernel`.
   - Simplify `LaunchGemm` to call `ChooseKernel(...)`.
2. `tests/Nivara.Tests/Gpu/SilkLayaHeadKernelRegistrationTests.cs` (new, device-free)
   - Catalog: `KernelNames` contains and `Source` defines every
     `Gemm{Qkv,Gelu,Relu,Bias}` × `{Bf16,F16,Bf16Act}` variant.
   - Dispatch: `ChooseKernel` maps base→base (F32), base→`…Bf16` (narrow BF16),
     base→`…F16` (narrow F16), base→`…Bf16Act` (activation BF16).
   - Every mapped result is a member of `KernelNames`.

## Verification steps

- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkLayaHeadKernelRegistration"`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Targeted build of Nivara.Extensions.

## Planned commits

1. `docs: plan #1113 head kernel registration pin in TODO.md`
2. `refactor(silk): extract head GEMM kernel dispatch into ChooseKernel`
3. `test(silk): pin BF16/F16 head kernel registration and dispatch`
4. `docs: remove TODO.md — plan executed` (after G2)

## Blast radius

- `SilkLayaHeadGpuRunner.LaunchGemm` (internal dispatch) only; behaviour unchanged.
- New internal static method; tests reach via InternalsVisibleTo.
- New test file; no docs pins affected.

## GitHub issues log

(none yet)