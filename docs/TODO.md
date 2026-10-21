# Plan: Torch parity for Silk (OpenCL) kernels — Phase A (vision)

## Problem

`NivaraTorch` (`samples/NivaraTorch/` + `tests/Nivara.Tests/NivaraTorch/`) verifies
**CPU AutoDiff layers only** against the PyTorch fixtures in
`samples/data/torch-comparison/`. Silk kernels are currently verified by:

- `tests/Nivara.GpuProbe` (dot16 / silu / gemv vs CPU reference),
- `tests/Nivara.PerformanceTests/SilkKernelsGate` (Silk vs ILGPU for BERT/decode kernels),
- model-level `--silk compare` (GPU forward vs same-process CPU).

The vision family (`conv2d`, `batchnorm_eval`, `relu`, `relu6`, `maxpool2d`,
`adaptive_avgpool2d`) has **no kernel-level gate at all** — only the end-to-end
MobileNetV2/ResNet-18 `compare`. The other families are at least cross-checked
against ILGPU or the CPU.

Goal of this phase: add per-kernel Torch-parity tests for the vision family so the
Silk kernels are pinned against an independent reference (PyTorch), not just
against our own ILGPU/CPU paths.

## Scope (Phase A only)

In scope:
- `conv2d` (5 fixture cases)
- `batchnorm_eval` (3 cases)
- `relu` / `relu6`
- `maxpool2d` (2 cases)
- `adaptive_avgpool2d` (2 cases)

Out of scope (later phases):
- Phase B: elementwise/linear/norm/activation kernels (`silu`, `geglu`, `layernorm`,
  `add`, `add_bias`, `gemm`/`gemm_bias`).
- Phase C: attention kernels.
- Phase D: decode kernels (no Torch fixtures; keep ILGPU/CPU cross-checks).

## Proposed changes

Create `tests/Nivara.Tests/Gpu/SilkVisionTorchParityTests.cs`:

- `[TestFixture] public sealed class SilkVisionTorchParityTests`.
- Device gate via `SilkRuntime` constructor wrapped in try/catch → `Assert.Ignore`
  when no OpenCL device is available (existing pattern in `SilkGemmGemvAddTests`).
- For each case: load `.bin` fixtures via `Nivara.Tests.NivaraTorch.TestHelpers.LoadBin`,
  upload to device buffers, invoke `Nivara.Silk.SilkLauncher.<Kernel>(...)`, download,
  assert with `TestHelpers.AssertTensorEqual(expected, actual, 1e-5f, 1e-4f)`.
- Conv weight is already in PyTorch `[outC, inC/groups, kH, kW]` layout, which
  matches the kernel's weight indexing — no transpose.
- BatchNorm: eps = 1e-5, hasAffine = true, running stats loaded from fixtures.
- Dimensions hardcoded per case, mirroring `Conv2dTests` / `BatchNorm2dTests`.

Tolerance: 1e-5 abs / 1e-4 rel (matches the CPU Torch parity tests). Not
bit-exact, because conv/bn accumulate in a different order than PyTorch's BLAS.

## Blast radius

Files added:
- `tests/Nivara.Tests/Gpu/SilkVisionTorchParityTests.cs` (new).

Files read but not changed: `TestHelpers`, `SilkLauncher`, `SilkRuntime`, fixture
`.bin` files under `samples/data/torch-comparison/`.

Tests affected: new tests only. No existing tests touched. If a Silk vision
kernel diverges from PyTorch beyond tolerance, that is a real bug and must be
fixed in the kernel (not loosened in the test).

No docs pins affected (no docs files changed, no fixtures added/removed).

## Verification steps

1. `dotnet build tests/Nivara.Tests/Nivara.Tests.csproj -c Release` (with
   OpenCL present) — must compile.
2. Run `dotnet test ... --filter "FullyQualifiedName~SilkVisionTorchParity"` on a
   machine with an OpenCL GPU — all cases must pass.
3. On a no-GPU machine (CI) — tests must report inconclusive (Assert.Ignore),
   not failures.
4. `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.

## Planned commits

1. `docs: plan silk torch parity phase A in TODO.md`
2. `test: add silk vision kernel torch parity tests`
3. (if needed) `fix: ...` for any kernel divergence found
4. `docs: remove TODO.md — plan executed` (after G2 review)

## GitHub issues log

- [ ] #774 — Silk Torch parity: Phase B elementary/activation/norm kernels (created while working on Phase A)
- [ ] #775 — Silk Torch parity: Phase C attention kernels (created while working on Phase A)

## Decisions for the human (Gate G1)

1. Tolerance vs bit-exact: conv/bn use floating-point accumulation in a
   different order than PyTorch, so tolerance (1e-5/1e-4) is the correct bar;
   bit-exact would be impossible and would only add flake. Bit-exact is reserved
   for pure mul-add kernels (gemm/add/add_bias) in Phase B.
2. Device availability in CI: tests `Assert.Ignore` on no-GPU runners. They will
   not fail CI, but they will also not exercise the kernels in CI. Accepted trade-off
   until a software OpenCL implementation is available in the test environment.
3. Phase scope limited to vision family in this PR.