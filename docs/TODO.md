# Plan: optimize Silk attention_backward_dkv (issue #1002)

## Problem

`attention_backward_dkv` is the leading attention contributor in the DistilBERT
training backward chain (profile #1001, B=8): 424 ms of a 748 ms attention
backward and 1551.6 ms total backward. Each lane is serial over `seqLen * headDim`
(128 * 64 = 8192) and the kernel is launched over `batch * numHeads * seqLen`
items. Issue acceptance requires either a measurable reduction (>=20% of
baseline backward ms) or a documented conclusion that the kernel is already
memory-bandwidth bound, with `SilkDistilBertTrainerTests` (3 tests) green.

## Blast radius

- Primary: `src/Nivara.Extensions/Silk/SilkKernels.cs` kernel source (constant
  `AttentionBackwardDkv` + supporting helpers).
- Launcher: `src/Nivara.Extensions/Silk/SilkLauncher.cs`
  (`AttentionBackwardDkv` dispatch signature stays stable if possible).
- Trainer parity gate:
  `tests/Nivara.Tests/Gpu/SilkDistilBertTrainerTests.cs` (3 GPU tests) +
  `tests/Nivara.PerformanceTests/SilkDistilBertTrainingProfileProbe.cs`
  (regression measurement).
- No public API surface beyond the (internal) kernel/launcher.

## Proposed changes (draft, to be grounded/amended in G1)

1. Micro-optimisations with bit-identical accumulation order:
   - hoist `rowIdx`, `qBase`, `dAttnBase` address arithmetic;
   - guard the `p == 0` skip before the `attn_backward_dotv` call;
   - hoist `delta[rowIdx]` once.
2. If micro-opts land flat, try register-pressure relief: remove the `dk[64]`/
   `dv[64]` private arrays and accumulate directly into `dQkv`, keeping the
   same `i`-outer order so rounding stays inside epsilon.
3. If that is also flat, try loop restructuring (transpose) only behind a
   measurement showing a win; otherwise record the kernel as bandwidth-bound and
   update docs/SILK.md accordingly.
4. Re-run `--silk-distilbert-train-profile` after each candidate and keep only
   the winning change set (one logical change per commit).

## Verification steps

- Build: `dotnet build src/Nivara.Extensions -c Release`.
- Parity: `dotnet test tests/Nivara.Tests -c Release
  --filter "FullyQualifiedName~SilkDistilBertTrainerTests"` (3 tests, OpenCL).
- Measurement: `dotnet run -c Release --project tests/Nivara.PerformanceTests --
  --silk-distilbert-train-profile` (B=4/B=8 backward ms vs the #1001 baseline).
- Formatting: `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.

## Planned commits

- (added as landed) optimization commit(s) + docs update.

## GitHub issues log

- #1002 — this issue (in progress on this branch).
- Follow-ups discovered during work (if any) will be created at discovery time.

## Conclusion

The `dkv` kernel was exercised with (a) fusing the `dot` and `dv` passes and
(b) hoisting the fixed `V` row out of the `i` loop. Neither moved `AttnDkv`
measurably (424 ms -> 419 ms, <2% at B=8, within run-to-run noise) and
`SilkDistilBertTrainerTests` stayed 3/3 green, so the kernel is treated as
memory-bandwidth bound and no kernel edit is retained. The geometric-fixed
attempt confirms the #1001 reading: LN/fc GEMM backward (763 ms) remains the
larger target for the next optimisation pass.