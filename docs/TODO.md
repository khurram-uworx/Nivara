# Plan: Issue #658 Milestone 2 — fused online-softmax decode attention (CPU)

Branch: `khurram/658-milestone-2`
Base: `khurram/658-milestone-1`

## Problem

`AttentionKernels<T>.DecodeAttention` currently performs three passes per query head:
1. compute and store every score (`scale * Dot(q, k)`) in a rented `scores` buffer,
2. softmax that score row in place,
3. weight the V rows into the output.

This reads the KV cache multiple times, allocates a scores buffer per call, and diverges
from the Silk `causal_gqa_attention` online-softmax contract.

## Proposed changes

Rewrite `DecodeAttention` as a single fused pass per query head:

- Track `m` (running max) and `sum` (unnormalised denominator).
- For each KV row `j`:
  - compute `s = scale * Dot(qHead, kRow)`,
  - rescale prior accumulators when `s > m` using `corr = Exp(m - s)`,
  - set `m = s`, add `w = Exp(s - m)`, and accumulate `outHead[d] += w * vRow[d]`.
- Clear the output on an all-`-∞` prefix (no valid max).
- Divide the accumulated output by `sum` once at the end.

Keep the public signature unchanged. Remove the now-unused `scores` rent and
`VPhaseFloat*` helpers from this class; inline the vectorized V accumulation into
the new single pass.

Float + hardware SIMD path:
- broadcast `w` into `Vector512<float>`/`Vector<float>`,
- FMA the vector-weighted V row into an in-place accumulator block spanning `headDim`.

Generic / non-SIMD path:
- use the same scalar online update with `TensorPrimitives.Dot` and `T.Exp`.

## Sketch

```csharp
output.Clear();
for (qh)
{
    outHead.Clear();
    T m = T.NegativeInfinity, sum = T.Zero;
    for (j)
    {
        T s = scale * TensorPrimitives.Dot(qHead, kRow);
        if (s > m)
        {
            if (m != T.NegativeInfinity)
            {
                T corr = T.Exp(m - s);
                sum *= corr;
                for (d) outHead[d] *= corr; // scalar or vector
            }
            m = s;
        }
        T w = T.Exp(s - m);
        sum += w;
        // accumulate w * vRow into outHead (vector when float + SIMD)
    }
    if (m == T.NegativeInfinity)
        outHead.Clear();
    else
        for (d) outHead[d] /= sum;
}
```

## Blast radius

Primary:
- `src/Nivara/AutoDiff/Operations/AttentionKernels.cs`
  - `DecodeAttention`
  - `VPhaseFloat512`
  - `VPhaseFloat`

Callers (unchanged):
- `src/Nivara/AutoDiff/Nn/LlamaCausalAttention.ForwardCached` / fused decode
- `samples/Nivara.Samples/LlamaForCausalLM.*`

Tests:
- `tests/Nivara.Tests/AutoDiff/LlamaCausalAttentionTests.cs`
  - `DecodeAttention_FusedMatchesMultiHeadAttention_AcrossGqaRatios`
  - `DecodeAttention_GqaMapping_ReusesSharedKeyValueHead`
  - `DecodeAttention_SteadyState_AllocatesNothing`
- `tests/Nivara.Tests/AutoDiff/LlamaDecoderBlockTests.cs`
- `tests/Nivara.Tests/AutoDiff/LlamaCausalKVCacheTests.cs`

## Verification steps

1. `dotnet build Nivara.slnx`
2. `dotnet test -c Release --filter "FullyQualifiedName~LlamaCausalAttentionTests"`
3. `dotnet test -c Release --filter "FullyQualifiedName~LlamaDecoderBlockTests|LlamaCausalKVCacheTests"`
4. `dotnet test -c Release --filter "FullyQualifiedName~EveryGatedBlock_CompilesWithoutErrors"`
5. `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
6. Human-run: `dotnet run -c Release --project samples/NivaraInference -- smollm benchmark`
7. If attention outputs drift, run SmolLM reference generation and inspect parity tests.

## Planned commits

1. `docs: plan #658 milestone 2 online-softmax decode in TODO.md`
2. `refactor(attention): make DecodeAttention a fused online-softmax pass`
3. `perf(attention): vectorise V accumulation in fused decode path`

## GitHub issues log

- #658 — parent: CPU tiled multithreaded GEMM and cache-friendly SmolLM decode
- #458 — CPU-side performance deficit tracker

## Notes / red flags

- Online softmax changes floating-point rescaling order; outputs may drift beyond bit
  exactness but should remain inside existing 1e-5 attention tolerances.
- Removing the `scores` buffer should lower the allocation guard; verify no hidden
  ArrayPool rent remains.
- Generic `T.Exp` depends on `IFloatingPointIeee754<T>` (Half, BFloat16) supporting it.