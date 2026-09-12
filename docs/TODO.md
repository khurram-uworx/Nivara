# AVX-512 scalar hot-path probe for Qwen decode

Branch: `khurram/avx512` (off `main`).

## Problem

Qwen2.5-0.5B F32 decode runs ~441 ms/token on this machine (11th Gen Core
i5-1135G7, Tiger Lake) vs Ollama Q4_K_M ~28 ms/token — not like-for-like, but the
F32 path should still be as fast as .NET allows. Question: does any Qwen decode
hot-path kernel still run **scalar** where a hand-rolled **AVX-512** branch could
help? Grounded direction from the human: only add optimized branches where
`TensorPrimitives` is **not** already used — `TensorPrimitives` is itself
vectorized (AVX-512-capable) inside the .NET runtime (MS Learn:
"Use SIMD and hardware intrinsics in .NET", dotnet/standard SIMD doc), so
replacing it would be reinventing the wheel.

## Audit result (done — no `src/Nivara` change yet)

Per-token path is overwhelmingly TensorPrimitives-backed. Already vectorized:

| Kernel | Where | Today |
|---|---|---|
| All Linear projections (QKV/O, gate/up/down, LM head) | `TensorsHelper.MultiplyCore` | `TensorPrimitives.Dot` per row |
| SiLU | `GradKernels.Silu` | `TensorPrimitives.Sigmoid` + `Multiply` |
| RMSNorm kernel | `RMSNormKernel` | `TensorPrimitives.Dot` + `Multiply`/`MultiplyAdd` |
| Softmax of scores | `GradKernels.SoftmaxSingle` | `TensorPrimitives.Subtract/Exp/Sum/Divide` (max scan scalar, tiny) |
| Attention scores (decode) | `AttentionKernels.DecodeAttention:123` | `TensorPrimitives.Dot` |

Scalar (no TensorPrimitives) — the only AVX-512 candidates:

1. **RoPE forward** — `GradKernels.RotaryForward` (`GradKernels.cs:250`):
   scalar pair rotation `out[i]=x0·c−x1·s; out[i+p]=x0·s+x1·c`. Called per
   (head, position) from `RotaryEmbedding.cs:107`. per token: 24 layers ×
   (14 Q + 2 K) heads × 32 pairs.
2. **Decode-attention V-weighted accumulation** — `AttentionKernels.DecodeAttention`
   (`AttentionKernels.cs:126-132`): scalar `out[d] += w * vRow[d]` for 14 heads ×
   kvLen × 64. Called from `LlamaCausalAttention.ForwardCached:225`.

Micro-fix note (not AVX-512, out of scope here): RMSNorm forward gamma multiply
(`RMSNorm.cs:148-153`) is a scalar multiply that could simply be a
`TensorPrimitives.Multiply` call — see issues log.

Both real candidates are ~0.1% of the ~1.05 GFLOP per-token budget (RoPE ≈
0.05 MFLOP, V-accum ≈ 0.67 MFLOP at kvLen 376), so the expected outcome is
"already AVX-512 optimized; no kernel changes." We probe to confirm with data.

## Approach: probe first (`tests/Nivara.SimdProbe`, `--scalar` mode)

New `ScalarKernelProbe.cs` (float path, matches the F32 Qwen pipeline):

1. **Current-path baseline** — `TensorPrimitives.Dot`-per-row GEMV at the four
   Qwen shapes (lm_head/gate-up/qkv-o/down): ms, weight-read GB/s, GFLOPS.
   Evidence for "already vectorized / memory-bound".
2. **RoPE** — scalar (verbatim copy of today's `RotaryForward`) vs AVX-512
   pair-rotation kernel (2× `Vector512` FMA chains) at p=32 (per-head rotation),
   bulk loops for the per-token volume (384 rotations) and prefill (×216).
3. **Decode-attention V-accum** — scalar (verbatim copy of today's inner loop)
   vs AVX-512 d-blocked broadcast-FMA kernel at headDim=64, kvLen ∈ {216, 376},
   14 heads / 2 KV heads.
4. **Correctness gate** — AVX-512 variants vs scalar references, including a
   K/headDim not multiple of 16 (tail handling).
5. `Program.cs`: add `"scalar" => ScalarKernelProbe.Run()`. README: mode +
   real-machine results (README host description is stale Arrow Lake; this box
   is Tiger Lake with AVX-512 F/DQ/L/BW/VL/VNNI present).

Decision rule (per human direction): promote into `src/Nivara` **only if** a
kernel can plausibly move ≥ ~1% of per-token time **and** AVX-512 gives ≥ 2× on
it. Otherwise conclude "already optimized, no change" and report the numbers.
Probable promotion targets if the rule fires: `GradKernels.RotaryForward` and
`AttentionKernels.DecodeAttention`, each behind `Avx512F.IsSupported` with the
existing scalar path as fallback.

## Verification steps

- `dotnet build tests/Nivara.SimdProbe` to compile-check.
- `dotnet run -c Release --project tests/Nivara.SimdProbe -- scalar` (long run —
  **ask the human before running**).
- Existing `Nivara.Tests` suite untouched unless promotion happens (then run the
  RoPE/attention/AD test groups — also ask first).

## Planned commits

1. `docs: plan AVX-512 scalar hot-path probe in TODO.md`
2. `probe: add --scalar mode (ScalarKernelProbe.cs) to SimdProbe`
3. `docs: record --scalar probe results and decision in SimdProbe README`

## Blast radius

- Additive only: `tests/Nivara.SimdProbe/{Program.cs,ScalarKernelProbe.cs,README.md}`,
  `docs/TODO.md`, plus this branch. No `src/Nivara` edits in this step.
- If promotion fires (unlikely per the audit), it touches `GradKernels.RotaryForward`
  and `AttentionKernels.DecodeAttention` only, behind `Avx512F.IsSupported` with
  scalar fallback; covered by existing RoPE/attention tests.

## GitHub issues log

- [ ] (none yet — any deferred work surfaced by the probe, e.g. the RMSNorm
      gamma `TensorPrimitives.Multiply` micro-fix, will be filed here when/if it
      comes up; do not hold items in memory)