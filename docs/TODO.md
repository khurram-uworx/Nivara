# Plan: Issue #1225 — Silk Laya native (F16 storage, F32 compute) on the device path

## Problem

`laya` defaults to native loading on CPU (PR #1224): F16 storage, F32 compute, ~843 MB
resident instead of ~1.69 GB. On Silk, Laya stays F32 (or explicit `--precision bf16`)
because there is no Half device path: `--silk --precision fp16` is rejected for every
model, and `laya --silk` with default/native precision prints a one-line note and runs
F32. A user who picks `laya --silk` for speed loses the footprint win entirely.

## Goal

`laya --silk` with default/native precision runs F16 storage on the device (widening
in-register), matching the CPU native path, instead of falling back to F32.

## Proposed changes

### A. CLI routing (`samples/NivaraInference/Program.cs`)

- Remove the `if (useSilk && isLaya && native)` note block (line ~670-673).
- In the `case "laya":` + `useSilk` branch (lines ~815-827), add a native/default
  branch that loads `SafeTensorsLoader.Read<Half>` (reuse the same `tensorsHalf` path
  already used for CPU native/fp16) and opens the F16 device session, mirroring the
  CPU `layaNativeCpu -> Run<float,Half>(tensorsHalf, ...)` path.
  - For default mode: `Laya.RunGpu(tensorsHalf, modelDir)` (Half overload in
    `Laya.Narrow.cs`) which calls `GpuSeamFactories.LayaFp16` — already wired.
  - For benchmark: `Laya.RunBenchmarkGpu(tensorsHalf, modelDir, GpuSeamFactories.LayaFp16)`.
  - For compare: needs a device-vs-CPU-native gate; existing `CompareGpu` takes float
    tensors + `OpenLayaSession`. A Half-vs-CPU-native compare needs a new overload or
    a native-aware compare. See step D.
- Keep `--silk --precision fp16` rejected (`--silk is F32-only` guard stays).
- Keep `--silk --precision bf16` working (existing BF16 branch unchanged).
- Keep `--precision native` accepted for laya (already accepted).

### B. Sample helpers (`samples/NivaraInference/Laya.cs`, `Laya.Narrow.cs`)

- `RunGpu(Half,...)` already exists and uses `LayaFp16` factory + prints
  "Half native". Good — may just need label/precision messaging consistency.
- Add a `CompareGpu` overload accepting `Dictionary<string,(Half[] Data,int[] Shape)>`
  and an `OpenLayaSessionFp16` open delegate (or resolve factory inside), comparing
  device (F16 storage) against CPU native (`Run<float,Half>` reference) using the
  same `GateRelTol = 1e-3` bound and `max|diff| <= 1e-3*(1+|ref|)`.

### C. Silk stack (Extensions)

- `SilkLayaSession` Half ctor/Open already present (verified via code-memory SQL +
  file read). `SilkModernBertGpuRunner.Fp16.cs` Half ctor present. `SilkLayaHeadGpuRunner.Half.cs`
  present with narrowF16 dispatch + F16 GEMM kernels.
- Verify no gaps: activation path for Half (currently only `activationBf16` is handled
  in `Forward`/`EnsureWorkspace`/launchers). For "F16 storage, F32 compute" (native),
  activations compute in F32 on device, so `activationF16` should be false and the F32
  (non-activation-narrow) path is used for compute buffers. Confirm `narrowF16=true`
  (weights stored as ushort/F16, widened in kernels) with `activationF16=false`
  (compute buffers F32) is a supported combination in the runner. Audit
  `ChooseKernelImpl`, `EnsureWorkspace`, `Forward`, launchers for `narrowF16`-only
  (no activation) handling. If missing, add F32-compute buffers while keeping narrow
  weight upload as F16 — analogous to CPU native (F16 storage, F32 compute).
- Note: the head runner's `activationF16` field exists but is only consumed in a few
  places (checked). If native uses F32 compute, the F32 buffer path (else branch) is
  taken, which is correct; narrow weights still use F16 kernels. Good.

### D. Parity gate

- Add a real-checkpoint Silk parity gate (Gpu category) mirroring
  `SilkLayaNarrowParityTests` but over real checkpoint / F16 storage: device vs CPU
  native agree within `max|diff| <= 1e-3*(1+|ref|)`. Could be a sample `compare`
  path (CPU-native reference vs Silk F16 device) or a new Gpu test. Issue asks for a
  Silk parity gate (Gpu category) mirroring `SilkLayaNarrowParityTests` but over the
  real checkpoint / F16 storage. Implement as a new `[Category("Gpu")]` test that
  loads the real checkpoint if present (else `Assert.Ignore`), runs CPU native
  (`Run<float,Half>` equivalent) and Silk F16 device, compares logits with the bound.
  Keep synthetic `SilkLayaNarrowParityTests` unchanged.

### E. Tests / DocsCoveragePins

- Update `InferenceCliArgumentTests.NativePrecision_OnNonSmolLMModel_Rejected`:
  remove/adjust the assertion that `laya --silk` prints the F32 device-path note;
  instead assert it reaches the model-file lookup (no note) — mirroring
  `QwenSilk_NativeDefault_RunsBf16DevicePath`.
- Keep `SilkLaya_Fp16_Rejected` / `SilkLaya_Fp16_Benchmark_Rejected` (fp16 still
  rejected on Silk).
- Review `DocsCoveragePins` (citation counts) if any doc changes; update counts if
  docs edited. Likely add a note in `docs/LAYA.md` / `docs/SILK.md` about native Silk
  path; update pin counts accordingly.

## Verification steps

1. Build: `dotnet build Nivara.slnx` (Release).
2. Run targeted tests: `SilkLayaNarrowParityTests`, `SilkLayaHeadKernelRegistrationTests`,
   `InferenceCliArgumentTests` (ask first per AGENTS.md; targeted runs allowed).
3. CLI seam: `laya --silk` no longer prints F32 device-path note; reaches model load.
4. Formatting: `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
5. Snippet gate: `EveryGatedBlock_CompilesWithoutErrors` (if docs changed).

## Planned commits

1. `docs: plan issue #1225 in TODO.md`
2. `feat(samples): route laya --silk native/default to F16 device session`
3. `feat(samples): add CompareGpu Half-vs-CPU-native parity path`
4. `feat(silk): ensure narrowF16 storage with F32 compute on device (if audit finds gap)`
5. `test: update CLI seam expectations for laya --silk native`
6. `test: add Gpu real-checkpoint Silk Laya F16 parity gate`
7. `docs: document laya --silk native F16 path` (+ pin updates)
8. `docs: remove TODO.md — plan executed`

## Blast radius

- `samples/NivaraInference/Program.cs` (CLI dispatch + note) — affects inference sample
  behavior and CLI arg tests.
- `samples/NivaraInference/Laya.cs`, `Laya.Narrow.cs` — sample helpers, compare/gate.
- `src/Nivara.Extensions/Silk/SilkLayaHeadGpuRunner*.cs` (audit only; changes if gap) —
  affects GPU parity tests and kernel registration tests.
- `src/Nivara.Extensions/Silk/SilkLayaSession.cs` (already supports Half; likely none).
- `src/Nivara.Extensions/AI/GpuSeam.cs` (already has LayaFp16 factory; likely none).
- `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs` — expectations.
- `tests/Nivara.Tests/Gpu/*` — new parity gate.
- `docs/*`, `DocsCoveragePins` — citation/doc counts.

## GitHub issues log

- (none yet; create if deferred work surfaces during implementation)
