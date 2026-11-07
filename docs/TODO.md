# Plan: perf(silk) attention backward fusion and prefetch (#1011)

## Problem
Attention backward contributes ~742 ms at B=8 (~48% of backward chain). Its
`attention_backward_prep`, `dq`, and `dkv` kernels repeatedly read q/k/v/probs/mask
and launch separately. Goal: measurable >=20% reduction or documented
memory-bandwidth conclusion; `SilkDistilBertTrainerTests` parity stays green.

## Blast radius
- `src/Nivara.Extensions/Silk/SilkKernels.cs`: kernel constants, `KernelNames`, OpenCL source string.
- `src/Nivara.Extensions/Silk/SilkLauncher.cs`: new launcher(s).
- `src/Nivara.Extensions/Silk/SilkDistilBertTrainer.cs`: attention-backward call sequence and profile keys.
- `tests/Nivara.Tests/Gpu/SilkBackwardKernelTests.cs`: parity case(s).
- Optional docs only if conclusion is memory-bandwidth.
- Laya inference attention touched only after a positive result.

## Proposed changes
1. Prep private caching: load q row, dAttn row, mask row once; cache Q.K score once per (query,key) in a private score buffer; reuse for max/sum/probability passes; OpenCL prefetch for key rows.
2. Dq private caching of query dAttn row; Dkv private caching of key V row and prefetch of next query row.
3. Fuse Dq and Dkv into one `attention_backward_dq_dkv` kernel (same global id performs both private reductions; disjoint output columns; no atomics).
4. Trainer uses prep + fused kernel; profile keys updated (sum of prep + fused).
5. Parity tests call the fused launcher and retain tolerance + device-vs-device exactness.

## Planned commits
1. docs: plan #1011 in TODO.md
2. feat(silk): add attention_backward_dq_dkv launcher + kernel
3. perf(silk): private caching + prefetch in attention backward kernels
4. perf(silk): trainer uses fused attention backward path and updated profile keys
5. test(silk): parity coverage for fused attention backward kernel
6. docs: update profile/result caption only if needed

## Verification
- Build Extensions + Gpu tests (`dotnet build`).
- Targeted tests: `SilkBackwardKernelTests.AttentionBackward_MatchesCpuReplay_AndIsDemonisticOnDevice` and `SilkDistilBertTrainerTests` on GPU runner (human-run).
- Human-run profile probe before/after: `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --silk-distilbert-train-profile`.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` before final.

## GitHub issues log
- [ ] #1011 — perf(silk): probe attention backward fusion and prefetch for BERT/Laya (parent)

## Notes
No pushes during session. Decisions beyond trivial blast radius escalated.