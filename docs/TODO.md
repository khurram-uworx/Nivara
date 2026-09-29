# TODO — #462: wire the Laya decision head onto the GPU encoder runner

Branch: `khurram/462` (off `main` at 17d5a5f)

## Problem

`--gpu` is rejected for `laya` (`samples/NivaraInference/Program.cs:271-274`). The ModernBERT
encoder already runs on the OpenCL iGPU via `ModernBertGpuRunner`, but the decision head added by
#460 runs on CPU, and so does `laya compare`. The head is unblocked and needs **zero new kernels**:
pre-norm, biased LayerNorm, ReLU, fused biased QKV, no RoPE, no GeGLU, no sliding-window band.
`BertSelfAttention<T>` already has the CPU shape; the GPU side should reuse the trunk rather than
grow a second attention module.

## Grounding

Kernel availability verified against `samples/Nivara.Samples/Gpu/` — every head op already has a
kernel, so the issue's "zero new kernels" claim holds:

| Head op (`LayaHeadModel.cs`) | Existing kernel |
|---|---|
| `h + type_emb[qtype]` broadcast over rows | `ElementwiseKernels.AddBias` |
| pre-norm **biased** LayerNorm, eps 1e-5 | `ElementwiseKernels.LayerNorm1D` (real beta) |
| fused **biased** QKV `[3d, d]` | `GemmKernels.TiledGemmKernelRow4Qkv` |
| attention, no RoPE, no band | `AttentionKernels.BatchedAttention` + `GpuBuffers.GlobalAttentionBand` |
| `out_proj` / `linear2` / `scorer.3` + bias | `GemmKernels.TiledGemmKernelRow4Bias` |
| residual add | `ElementwiseKernels.Add` |
| `linear1` + bias + **ReLU** | `GemmKernels.TiledGemmKernelRow4Relu` |
| scorer LN; scorer hidden + exact GELU | `LayerNorm1D`; `GemmKernels.TiledGemmKernelRow4Gelu` |
| marker row gather; type row gather | `ElementwiseKernels.Gather` |
| act head LN-free `[d+4, 256]` + GELU, then `[256, n_act]` | `TiledGemmKernelRow4Gelu`; `TiledGemmKernelRow4Bias` |

Config facts to pin in code:

- Head `headDim = 1024 / max(1, 1024/64) = 64`, which equals the encoder's headDim. So
  `GpuBuffers.MaxHeadDim` (64) and `ValidateAttentionLocalMemory` are satisfied — but the head
  runner must validate **its own** headDim rather than inheriting the encoder's.
- The head's LayerNorm eps is `1e-5` (the `LayaDecisionHead` ctor default), **not**
  `config.NormEps`. Reusing the encoder eps implicitly is the easiest way to be subtly wrong.

### The one place "zero new kernels" has to bend

The act head's 4 features are `top1`, `top1 - top2`, normalized entropy, and `max(2,k)/255` —
all derived from a softmax over the marker logits. Computing those on-device would require a
softmax + top-k + entropy kernel, i.e. new kernels, contradicting the issue. So: keep the act
GEMMs on device, compute the 4 features on the host exactly as `LayaDecisionHead` does, and upload
the assembled `[d+4]` row. This must be a named comment, not a silent divergence.

## Blast radius

Confirmed by code-memory SQL over `SymbolRecord` / `RelationshipRecord`, not by inspection:

- `ModernBertGpuRunner` has exactly **3** referrers, all in `samples/NivaraInference/ModernBert.cs`:
  `RunGpu` (519), `BenchmarkGpu` (567), `CompareGpu` (672). An **additive** seam leaves all three
  untouched.
- `LayaDecisionHead` is referenced only by its own file and
  `tests/Nivara.Tests/AutoDiff/LayaDecisionHeadTests.cs` (the CPU reference). Those tests must keep
  passing unmodified — they are the gate's reference, not a casualty.
- `Laya` (mode class) is referenced only from `Program.cs`.
- No `src/Nivara` change. All work is sample-side, consistent with #460.

## Open decisions (escalate, do not assume)

1. **Encoder → head seam — DECIDED: (A) additive in-place seam** (human, this session). The head
   consumes the final-normed device buffer; only the `k` marker logits and `n_act` act logits read
   back. Matches repo precedent — `BertEncoderGpuRunner.Forward` already runs its SST-2 head in
   place on device and reads back only logits. `ArrayView<float>` is a ref struct, so this cannot
   be a returned view: expose the buffer + row count as properties and add a no-readback forward.
   The three existing `ModernBert.cs` call sites must keep compiling and behaving identically.
2. **Is the 1e-3 bound a hard gate? — DECIDED: hard fail at 1e-3, print the measured `maxRel`**
   (human, this session). `docs/ACCELERATION.md:146` records the ModernBERT *encoder* gate at
   `maxRel 5.577E-004` — already ~56% of the budget, from accumulated F32 reduction-order drift
   over 28 layers. Two more head layers plus scorer and act sit on top, so the combined gate is the
   tightest in the repo. It is still a hard gate: no silent widening. The concession is that every
   failure prints the measured `maxRel` next to the bound, so a near-bound result is visibly
   attributable to arithmetic drift rather than silently absorbed. The encoder's 1e-3-vs-1e-5
   discussion in `CompareGpu` is the precedent to follow in wording.

## Proposed changes

1. **`samples/Nivara.Samples/Gpu/LayaHeadGpuRunner.cs` (new).** Device-side head, mirroring
   `LayaHeadLayer<T>.Forward` / `LayaDecisionHead<T>.Forward` operation for operation. Ctor uploads
   `type_emb.weight`, `scorer.0/1/3`, `act_head.0/2`, and per layer `norm1`/`norm2`
   (weight+bias), fused `in_proj_weight`/`in_proj_bias` sliced into q/k/v, `out_proj`, `linear1`,
   `linear2`. Own LayerNorm eps 1e-5; call `ValidateAttentionLocalMemory` for its own headDim.
   Forward contract mirrors the CPU: `Forward(ArrayView<float> hidden, int rows,
   LayaQuestionType type, int[] markerPositions, int validLength)`.
2. **Seam on `ModernBertGpuRunner` (per decision 1).** Additive only.
3. **`samples/NivaraInference/Laya.cs`.** Add `RunGpu`, `BenchmarkGpu`, `CompareGpu`. `CompareGpu`
   gates against the in-process CPU head (the user's call: CPU is the correctness reference), over
   the same `FixtureQuestions` / `FixtureState` #460 pinned byte-exact against the wheel. No new
   Python fixtures and no second wheel gate — the CPU head is transitively pinned, so the GPU head
   inherits the pin. Mirror `ModernBert.CompareGpu`'s structure, including the non-finite "mask
   clamp" check (load-bearing for the fully-masked-row hazard #448 that the head inherits through
   its dense `[L, L]` mask).
4. **`samples/NivaraInference/Program.cs`.** Replace the `--gpu` reject at 271-274 with the same
   dispatch shape as `modernbert` (286-296). The `--gpu` + bf16/fp16 reject at 174-178 already
   covers the third acceptance criterion globally.
5. **Docs.** `docs/LAYA.md` (status line 3, GPU-path section 132-187, "what's next" item 2 at
   204), `docs/ACCELERATION.md` (§1b line 73, item 10 line 143, item 5 line 447),
   `samples/NivaraInference/README.md` (add `laya --gpu` to lines 50-54).

## Verification steps

- `dotnet build Nivara.slnx` (repo uses `.slnx`).
- `tests/Nivara.Tests/AutoDiff/LayaDecisionHeadTests.cs` stays green — it is the reference the gate
  compares against, and nothing in it may change.
- `laya --gpu compare` passes the `1e-3*(1+|x|)` bound on marker logits and act probability.
- `laya --gpu` and `laya --gpu benchmark` run encoder + head on the iGPU, F32 only.
- `laya --gpu --precision bf16|fp16` still rejects.
- Ask the human before any `dotnet test` or long-running run.

## Test strategy

No new NUnit file. The repo gates GPU parity through the sample's `compare` mode, not NUnit —
`tests/Nivara.Tests/Gpu/GpuElementwiseParityTests.cs` is the only GPU test file and it exercises a
device-free scalar. The existing CPU head tests are the reference and stay unmodified.

## Planned commits

1. `docs: plan #462 Laya GPU head in TODO.md`
2. `Add LayaHeadGpuRunner for the decision head on the accelerator`
3. `Expose the encoder's final hidden state on device from ModernBertGpuRunner` (if decision 1 = A)
4. `Add laya --gpu, benchmark, and compare modes`
5. `Wire laya --gpu through the CLI dispatch`
6. `docs: record the Laya GPU head in LAYA.md, ACCELERATION.md, and the sample README`

## GitHub issues log

- [ ] #NNN — placeholder; created at discovery time as work proceeds.

> As each task executes, if you find deferred work or a concern (known limitation, follow-up,
> refactor) outside this plan, create it immediately with
> `gh issue create --repo khurram-uworx/Nivara` and record the number above. Do not rely on memory
> or wait until the plan finishes — compaction can lose it.
