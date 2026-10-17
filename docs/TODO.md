# Plan — Issue #651: Fuse Silk decode/prefill GQA attention (single pass)

## Problem

`SilkKernels.cs` kernels `decode_gqa_attention` and `causal_gqa_attention` recompute `q·k`
three times per (query, key) — max pass, sum pass, and weighted-V pass. This is ~3× the
necessary MACs and is the largest self-inflicted bottleneck on the cached decode path.
SmolLM (`SilkSmolLMGpuRunner.ForwardCached`) and prefill both hit it. Same pattern as the
ILGPU reference `BatchedAttention`, but the reference is a separate code path used only by
the gate; Silk is the production path being optimised.

## Proposed changes

Online-softmax fusion. One traversal per query head: compute each score once, keep a running
max (`mx`), rescale the unnormalised accumulators (`sum`, `accv[0..headDim)`) in place on
every max update, then divide by `sum`.

Components:

1. `src/Nivara.Extensions/Silk/SilkKernels.cs` — `causal_gqa_attention` body (single pass).
2. `src/Nivara.Extensions/Silk/SilkKernels.cs` — `decode_gqa_attention` body (single pass).

Host launcher (`SilkLauncher.cs`) unchanged: same 9 args in the same order, so no
`clSetKernelArg` mismatch risk. Launch geometry unchanged (`LinearGlobal` / `LinearGroupSize`).

## Code sketch (decode; causal differs only in the j loop bound `j <= qPos`)

```c
uint h = ...
if (h >= numHeads) return;
... bases ...
if (kvLen == 0u) { zeros; return; }

float accv[64];
for (d) accv[d] = 0.0f;
float mx = -INFINITY;
float sum = 0.0f;
for (j) {
    float acc = 0; for (dd) acc += q[qBase+dd] * kCache[kBase+dd];
    float s = acc * scale;
    if (s > mx) {
        if (mx != -INFINITY) {
            float corr = exp(mx - s);
            sum *= corr;
            for (d) accv[d] *= corr;
        }
        mx = s;
    }
    float w = exp(s - mx);
    sum += w;
    for (d) accv[d] += w * vCache[... + d];
}
if (mx == -INFINITY) { zeros; return; }
float inv = 1.0f / sum;
for (d) out[qBase + d] = accv[d] * inv;
```

## Verification steps

- Build: `dotnet build Nivara.slnx`
- Silk kernel gate (BIT-EXACT / bound): `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --silk-kernels`
  (worst ratio must stay ≤ 1.0 against `1e-3 * (1 + |x|)`)
- Host-side smoke if OpenCL available; else gate is the verification and the absence must be
  reported explicitly (no silent green).

## Blast radius

- Only the two C source strings in `SilkKernels.cs` change. Their byte layout, argument count,
  launch geometry, and macro surface are unchanged, so `SilkLauncher`, `SilkRuntime`,
  `SilkSmolLMGpuRunner`, the gate, and the two ModernBERT/Laya runners are untouched.
- Numerical output changes accumulation order slightly (online rescaling vs clean passes); the
  gate's tolerance bound is the contract. `accv[64]` private storage uses the existing
  `MAXHEADDIM` constant (host already rejects `headDim > 64` in `EnsureAttentionLocalMemory`).

## Planned commits

1. `docs: plan #651 (fuse Silk GQA attention) in TODO.md`
2. `Fuse causal_gqa_attention into single online-softmax pass`
3. `Fuse decode_gqa_attention into single online-softmax pass`
4. `docs: remove TODO.md — plan executed` (only after G2 reviews)

## GitHub issues log

- (none yet)

## Notes / deferred work

- Row4-style shared-memory tiling for the q·k dot products (also suggested by the issue) is
  deferred to a follow-up perf commit; this plan targets the 3× MAC removal, which is the
  self-inflicted bottleneck called out. If tiling lands later it can amend this change.
