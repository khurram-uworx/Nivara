# Plan: deeper fused attention/MLP blocks for Silk decode (#672)

## Problem

After multi-output norm fusion (#670) and large-kvLen decode-attention work (#671), Silk cached decode
still has kernel-launch and intermediate-buffer overhead at residual attention/MLP boundaries:

- `decode_gqa_attention` writes the full `context` buffer (qWidth = numHeads*headDim, 896 for Qwen),
  then `gemm_row4_add` reads it for the O-projection + residual. The tile GEMM with aRows=1 also
  over-launches 16 rows for 1 useful row.
- MLP uses separate `rms_norm_gemv` (gate/up), `swiglu`, and `gemm_row4_add` (down+residual) launches.

Goal: fold O-projection into the decode-attention kernel (avoid context write/read) and add a single
MLP block kernel (gate, up, SwiGLU, down, residual), each gated on parity + measured per-token win.
Reject changes that only reduce launch counts without a measured win.

## Proposed changes

### Stage A1 — flat single-row O-proj GEMV + residual (low risk)

Add a flat GEMV kernel `gemm_gemv_add`:

```
__kernel void gemm_gemv_add(
    __global const float* a,   // [aCols]
    __global const float* b,   // [aCols * bCols] (B row-major, Bt layout not needed: single row)
    __global float* c,         // [bCols] output
    __global const float* residual, // [bCols]
    const uint aCols,
    const uint bCols)
```

One thread per BLOCK (4) output columns, single accumulator over k ascending, store `a0 + residual`.

Dispatch `GemvGlobalCols(bCols)` threads. Wire into both runners replacing
`GemmAdd(context, Wo, x, x, 1, qDim, hiddenDim)`.

Wait: the existing Wo is stored transposed (Bt layout) for the tile GEMM (a @ B where B is [aCols,bCols]
row-major in tile form). For a single-row GEMV we need a consistent layout. The tile GEMM computes
c[row,outCol] = sum_k a[row,k]*B[k,outCol]. Wo is uploaded transposed (WeightLayout.Transposed) so B[k,outCol]
= Wo_original[outCol,k]? Let me verify: tile GEMM c = a @ B with B stored as [aCols, bCols] (K x N). The
checkpoint Wo is [out=hidden, in=qDim] row-major; transposed gives [in=qDim, out=hidden] = [aCols,bCols].
So B[k,outCol] = Wo[outCol,k]. Good — same as current GemmAdd. So the flat GEMV uses the same Wo buffer
with B[k,outCol] = Wo_transposed[k,outCol]. No layout change needed.

### Stage A2 — fold O-proj into decode attention (true fusion)

New kernel `decode_gqa_attention_oproj`:

Inputs: q, kCache, vCache, Wo (qWidth x hidden, transposed layout [qWidth,hidden]), residual (hidden),
out (hidden), kvLen, numHeads, numKvHeads, headDim, scale.

Design: one work-group per query head is insufficient for the cross-head O-proj. Use a single dispatch
of `numHeads * AttentionGroupSize` threads grouped into `numHeads` work-groups of `AttentionGroupSize`,
each group computes its head's context into a shared `context` tile (size headDim), then a barrier, then
all groups cooperate? Cross-group barrier requires global memory + a second pass.

Pragmatic design (two-phase, device-wide):

Phase 1 (per head, as today): each work-group of AttentionGroupSize threads splits kvLen, computes its
head's context vector into global `context[h*headDim .. (h+1)*headDim]` (or a shared tile if single
work-group). Barrier (global, via separate kernel or in-kernel with CLK_GLOBAL_MEM_FENCE + atomic).

Phase 2 (O-proj): launch a flat GEMV over hidden columns reading the `context` buffer. This is just
A1 chained — i.e. A2's "fusion" is really launching phase1+phase2 in one host call. To genuinely avoid
the buffer we'd need a single work-group holding all heads' context in shared memory, which requires
work-group size >= numHeads*AttentionGroupSize (896 for Qwen) — exceeds typical iGPU max work-group
(256). So true single-buffer fusion is device-limited.

Decision: implement A2 as a combined launcher `DecodeGqaAttentionOproj` that runs the existing
`decode_gqa_attention` kernel followed by the A1 flat GEMV, but exposed as one logical op so the runner
no longer names the context buffer. This does not remove the buffer but removes the over-launch and the
tile staging; the buffer remains an internal scratch. If device max work-group allows, additionally
prototype a single-work-group variant behind a capability check.

Actually, given the device limit, the honest outcome is: A2's measurable win comes from A1's flat GEMV;
the "avoid writing/reading context" claim is not realizable on this hardware. Record this as a finding
and keep A1. Do not fabricate a kernel that cannot run.

### Stage B — single MLP block kernel (optional, measurement-gated)

Add `mlp_block` kernel: inputs normed (hidden), Wgate, Wup, Wdown (transposed), x (residual, hidden),
out (hidden), intermediate, hidden, eps? Norm is already done (post-norm) before MLP in the runner, so
no norm in this kernel. Operations:
- gate[i] = sum_k normed[k]*Wgate[k,i]  (i in [0,intermediate))
- up[i] = sum_k normed[k]*Wup[k,i]
- swiglu[i] = (gate[i]/(1+exp(-gate[i])))*up[i]
- out[j] = residual[j] + sum_i swiglu[i]*Wdown[i,j]

This is two GEMMs of different shapes serialized; the intermediate (4864) accumulator is per-thread
over output columns. Dispatch: for the down projection we need hidden output columns; each thread owns
BLOCK hidden columns and loops over intermediate. The gate/up GEMMs produce intermediate columns; we
can either (a) compute gate/up on the fly per output down-column (recomputing gate/up hidden→inter
for each down output column — O(hidden * intermediate * hidden) — too much), or (b) materialize
swiglu into an intermediate buffer then down-project (current approach, just fused into one kernel
launch but still needs the intermediate buffer internally).

Conclusion: a true buffer-free fusion is not feasible without either recomputing or shared memory
holding the whole intermediate (4864 floats ~19KB, fits in shared memory on many devices, but the
gate/up matmuls need hidden-wide reduction per intermediate column). A single work-group holding
intermediate in shared memory and doing gate/up/down serially would serialize all intermediate columns
through one group — worse parallelism.

Decision: implement B as a two-launch fusion? That's just current. So Stage B likely yields no win;
still measure A1 alone and, only if there is headroom, prototype a shared-memory MLP block behind a
capability check. Default: file a follow-up issue and skip B unless measurement shows win.

## Blast radius

- `SilkKernels.cs`: new constants + kernel source + `KernelNames` entries.
- `SilkLauncher.cs`: new dispatch methods.
- `SilkQwenGpuRunner.cs`, `SilkSmolLMGpuRunner.cs`: cached-decode O-proj call site (and MLP if B lands).
- Tests: `SilkDecodeAttentionParityTests`, `SilkRmsNormGemvTests`, `SilkKernelsCatalogTests`, new parity tests.
- Docs: `docs/SILK.md` kernel table + `KernelNames` listing (pinned by docs scan tests).

## Verification

- `dotnet build Nivara.slnx -c Release`
- Targeted tests (ask before running): SilkDecodeAttentionParityTests, SilkQwenParityTests, SilkQwenKvCacheTests, SilkRmsNormGemvTests, SilkKernelsCatalogTests.
- Docs suite if docs touched.
- Human-run Qwen benchmark block; compare to baseline.

## Planned commits

1. docs: plan #672 in TODO.md
2. (grounding notes appended)
3. perf(silk): flat single-row O-proj GEMV for cached decode (A1)
4. docs(silk): update SILK.md for gemm_gemv_add
5. perf(silk): expose combined decode-attention+O-proj launcher (A2)
6. (conditional) perf(silk): fused MLP block kernel (Stage B) — only if measured win
7. docs: record #672 measurement outcome
8. docs: remove TODO.md — plan executed

## GitHub issues log

- (none yet)
