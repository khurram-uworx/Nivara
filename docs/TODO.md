# Plan: optimize Silk `DecodeGqaAttention` at large `kvLen` (#671)

## Problem

After the #657/#670 projection fusion, Silk cached decode still scales poorly at large
`kvLen` because `DecodeGqaAttention` evaluates the score loop token-by-token with one
thread per query head. Only `numHeads` (14 for Qwen) lanes are active and the rest of
each 256-wide work-group returns immediately, so the GPU is effectively
single-threaded per head and `kvLen`-bound. This matters for longer Qwen/SmolLM
contexts (kvLen 512/1024/4096).

## Proposed changes

1. Restructure `decode_gqa_attention` in `src/Nivara.Extensions/Silk/SilkKernels.cs` from
   one work item per head to one work-group per head: split `kvLen` across the group's
   lanes, compute per-lane online-softmax partials `(mx, sum, accv[])`, then merge with
   the associative `(mx, sum, accv)` combination. Dot-product order stays ascending `d`.
2. Update `SilkLauncher.DecodeGqaAttention` to dispatch `numHeads * AttentionGroupSize`
   global / `AttentionGroupSize` (64) local.
3. Update `docs/SILK.md` kernel row to reflect the new geometry.
4. (Optional) add an OpenCL parity check against `AttentionKernels.DecodeAttention<float>`
   for the 14/2/64 shape at small and large kvLen.

## Blast radius

- `SilkKernels.cs` kernel source (string) — affects both Qwen and SmolLM cached decode.
- `SilkLauncher.DecodeGqaAttention` — sole dispatch site for the kernel.
- Runners `SilkQwenGpuRunner` / `SilkSmolLMGpuRunner` call the launcher unchanged.
- Parity gates: `SilkQwenParityTests`, `SilkQwenKvCacheTests` (synthetic).
- Performance: `QwenDecodeAttentionProbe` (human-run).

## Verification

- `dotnet build Nivara.slnx` (Release) for Extensions + tests.
- Targeted: `SilkQwenParityTests`, `SilkQwenKvCacheTests`.
- Docs suite if `docs/` changes: `dotnet test ... --filter "FullyQualifiedName~Nivara.Tests.Docs"`.
- Human-run: `QwenDecodeAttentionProbe` at kvLen 512/1024/4096 before/after.

## Planned commits

1. docs: plan #671 in TODO.md
2. perf(silk): parallelize decode attention over kvLen with work-group reduction
3. docs: update SILK.md decode_gqa_attention geometry
4. (optional) test: Silk decode attention parity vs CPU

## GitHub issues log

- (none yet)
