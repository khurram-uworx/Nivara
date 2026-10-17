# Plan: issue #628 — `--silk` for SmolLM-135M (slices 1–3)

Branch: khurram/628
Issue: https://github.com/khurram-uworx/Nivara/issues/628

## Problem

`--silk` (and `--ilgpu`) do not run SmolLM. SmolLM is a Llama-family causal LM (30
layers, hidden 576, intermediate 1536, 9 Q / 3 KV heads, headDim 64, SiLU gated FFN,
RMSNorm eps 1e-5, half-split RoPE theta 10000, tied embeddings, vocab 49152, BF16 on
disk but Silk is F32-only). The Silk translation unit has no RMSNorm, SiLU, gated SiLU
FFN, or causal GQA attention kernels, and no runner. Slices 1–3 land the kernels, a
F32 prefill runner, and the CLI/test wiring. Slice 4 (KV-cached decode) is tracked
separately on #445.

## Proposed changes

### Slice 1 — kernels + launcher + gate

`src/Nivara.Extensions/Silk/SilkKernels.cs`:
- `RmsNorm` (`"rms_norm"`): per row `rms = sqrt(mean(x²)+eps)`, `y = x/rms * gamma`. No mean subtraction, no beta. Alias-safe.
- `Silu` (`"silu"`): elementwise `x*sigmoid(x)`.
- `SwiGlu` (`"swiglu"`): fused `silu(gate)*up`, half-convention matching `geglu` (first `intermediate` = gate, second = up). Do not reuse `geglu`.
- `CausalGqaAttention` (`"causal_gqa_attention"`): causal (j≤qPos), GQA KV-repeat, headDim≤64; inputs `q[seq,numHeads*hd]`, `k/v[seq,numKvHeads*hd]`; launch `seq*numHeads`.
- Add all four to `KernelNames`.

`src/Nivara.Extensions/Silk/SilkLauncher.cs`: typed dispatchers `RmsNorm`, `Silu`, `SwiGlu`, `CausalGqaAttention`.

`tests/Nivara.PerformanceTests`: add `SilkSmolLMKernelsGate` (or cells) comparing Silk vs CPU reference
(`RMSNormKernel.PerRowRMSNormForwardKernel`, `GradKernels.Silu`, `AttentionKernels.BatchedAttention`) at
SmolLM dims, with existing exactness discipline (byte-identity for pure mul-add, 1e-3 bound for
transcendentals).

### Slice 2 — prefill runner + session

New `src/Nivara.Extensions/Silk/SilkSmolLMGpuRunner.cs` (F32-only, `IDisposable`):
- Constructor `(SilkRuntime, LlamaConfig, tensors)`.
- Uploads `embed_tokens.weight` (tied LM head), `model.norm.weight`, per-layer transposed weights
  (q/k/v/o, gate/up/down) and RMSNorm gammas. No bias (canonical Llama); bias-free `gemm_row4`.
- RoPE cos/sin via `RotaryEmbedding<float>` cached to `maxPositionEmbeddings`.
- Workspace buffers sized to seqCap.
- `Forward(int[] tokenIds) -> float[]` returning `seq*vocab` logits (gather embedding → per layer
  RMSNorm/QKV/RoPE/causal-attn/O/residual/post-norm/gate-up/swiglu/down/residual → final
  RMSNorm → tied LM head GEMM).
- Expose `ForwardLastLogits(int[] tokenIds) -> float[vocab]` to avoid materializing `seq*vocab`
  during generation (only the last position is consumed).

New `src/Nivara.Extensions/Silk/SilkSmolLMSession.cs` mirroring `SilkModernBertSession`.

`samples/NivaraInference/Program.cs`:
- Add `"smollm"` to `SilkModes` (default/benchmark; `generate`/`ab` stay CPU — `ab` toggles widen SIMD).
- When `useSilk`, construct `SilkSmolLMSession` and route through a Silk-aware path so
  `RunSmolLMGeneration`'s `ArgMaxLastRow` logic can be reused.

### Slice 3 — CLI tests

- Update `InferenceCliArgumentTests` expectations: `smollm` now in Silk model list; add precision/mode
  rejection pins analogous to modernbert.
- Add a parity smoke test (gated on checkpoint availability) comparing Silk vs CPU argmax on a short prompt.

## Verification steps

- `dotnet build Nivara.slnx` clean (Extensions + sample).
- `SilkKernelsCatalogTests` green.
- Targeted CLI arg tests green.
- Kernel gate runs on an OpenCL GPU and all cells within bound.

## Planned commits

1. docs: plan #628 in TODO.md
2. feat(silk): add RMSNorm/SiLU/SwiGlu/causal-GQA kernels + launchers
3. test(perf): add SilkSmolLM kernel gate
4. feat(silk): add SilkSmolLMGpuRunner + session
5. feat(sample): wire --silk for smollm (prefill path)
6. test: update CLI arg expectations for smollm silk
7. docs: remove TODO.md — plan executed

## Blast radius

- `src/Nivara.Extensions/Silk/SilkKernels.cs`, `SilkLauncher.cs`: new kernels; `SilkKernels.KernelNames`
  consumers (`SilkRuntime` constructor builds all named kernels; catalog test).
- New runner/session: only consumed by sample when wired.
- `samples/NivaraInference/Program.cs`: dispatch + mode allow-list; tests in
  `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs` pin exact messages.
- No `src/Nivara` changes.

## GitHub issues log

- [ ] #445 — SmolLM KV-cached decode on ILGPU; slice 4 of this plan must reconcile with it
  (comment to be added after slices 1–3 land).

## Notes / risks

- `headDim=64 == MaxHeadDim` drives the attention tile to its compiled bound; keep
  `EnsureAttentionLocalMemory`.
- Tied LM head GEMM is `seq × vocab` (49152); prefer `ForwardLastLogits` during generation.
- RoPE: theta=10000, half-split — reuse existing `rotary` kernel; only tables differ.
- BF16 on disk but Silk is F32-only: runner takes F32 tensors (sample already widens).
- Slice 4 deferred; do not promise PyTorch token match.
