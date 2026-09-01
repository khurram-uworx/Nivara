# NivaraTorch Parity Coverage — Fill Gaps for New AutoDiff Building Blocks

## Problem

The `samples/NivaraTorch/` harness provides PyTorch↔Nivara parity for AutoDiff layers via
`gen_reference.py` (PyTorch `.bin` fixtures) + NUnit classes in `tests/Nivara.Tests/NivaraTorch/`.
The last fixture generation was 2026-08-16 (Pow). Several building blocks were added **after** that
and have **no PyTorch parity fixture and no NivaraTorch test class** — they only have hand-rolled
unit tests under `tests/Nivara.Tests/AutoDiff/`, which cannot catch subtle math drift the way an
independent PyTorch oracle can.

Affected (confirmed no `silu_*`/`rope`/`rmsnorm_module`/`llama_*` fixtures in
`samples/data/torch-comparison/`):

1. `Activation.Silu` / `ReverseGradOperations.Silu` (gated SiLU FFN activation).
2. `RotaryEmbedding<T>` (RoPE, half-split `rotate_half` layout).
3. `RMSNorm<T>` module **with affine gamma** (only the no-gamma `PerRowRMSNorm` op is parity-tested).
4. `LlamaCausalAttention<T>` (GQA KV-repeat + RoPE + causal mask).
5. `LlamaDecoderBlock<T>` (pre-norm gated SiLU FFN with residual adds).
6. `LlamaForCausalLM` (tied LM head) — end-to-end logits parity.
7. `DepthwiseSeparableConv2d` (pre-existing, no parity).
8. `TransformerBlock` (pre-existing, no parity).
9. `SparseEmbedding` module (pre-existing, no parity; ops `SparseEmbeddingBag` are covered).

Stale docs: `samples/NivaraTorch/README.md` and `gen_reference.py` RMSNorm comments say "Nivara's
RMSNorm has no learnable weight" — outdated since the affine-gamma module.

## Proposed changes

### Phase A — Fixture generation (`samples/NivaraTorch/gen_reference.py`)
Append new cases **at the end** of the generation stream, each on a **dedicated** `torch.Generator`
with a fresh seed (preserving the file's bit-stable RNG contract). New fixtures:
- `silu_1d`, `silu_4d` (forward + input grad via `.sum().backward()`).
- `rope_1head`, `rope_2head` (RoPE forward, no grad; implement Llama `rotate_half` half-split with
  torch cos/sin).
- `rmsnorm_module_2d` (affine gamma; forward + grad w.r.t. input and gamma via
  `nn.RMSNorm(..., elementwise_affine=True)`).
- `llama_attn` (GQA: Q/K/V Linear proj → RoPE → KV-repeat → scaled-dot causal attention → O proj;
  forward + input grad).
- `llama_decoder_block` (pre-norm attn residual + pre-norm gated-SiLU FFN; forward + input grad).
- `llama_lm` (tied-hidden LM head logits; forward only, small vocab).
- `dsc` (depthwise separate conv via groups=inCh + ReLU + 1×1; forward + grad).
- `transformer_block_rms`, `transformer_block_ln` (forward only).
- `sparse_embedding` (EmbeddingBag mode=sum; forward + weight grad).

Run `python samples/NivaraTorch/gen_reference.py`, commit the whole
`samples/data/torch-comparison/` tree (manifest + `.bin` files) as one unit.

### Phase B — New NivaraTorch test classes (`tests/Nivara.Tests/NivaraTorch/`)
Match the existing pattern (TestHelpers.LoadBin / AssertTensorEqual, `[SetUp] GradientUtils.Grad()`):
- `SiluTests.cs` — extend existing `ActivationTests` or new class w/ forward + grad.
- `RotaryEmbeddingTests.cs` — forward parity (2 layouts).
- `RMSNormModuleTests.cs` — module forward + backward (input **and** gamma grads).
- `LlamaAttentionTests.cs`, `LlamaDecoderBlockTests.cs`, `LlamaForCausalLMTests.cs`.
- `DepthwiseSeparableConvTests.cs`, `TransformerBlockTests.cs`, `SparseEmbeddingTests.cs`.

Update `samples/NivaraTorch/README.md` fixture table + fix stale RMSNorm text.

### Phase C — Forward-mode JVP parity (if forward ops exist)
Add JVP cross-checks to `tests/Nivara.Tests/AutoDiff/ForwardParityTests.cs` for `Silu`, and RoPE /
`GqaRepeatKV` only if they have `ForwardGradOperations` support. **Verify forward-op existence
first; skip silently if reverse-only.**

## Verification
- `python samples/NivaraTorch/gen_reference.py` (available: torch 2.13.0+cpu, py 3.12.8).
- `dotnet test --filter "FullyQualifiedName~NivaraTorch"` (ask before running).
- `dotnet test --filter "FullyQualifiedName~NivaraTorch"` again after any fixes.

## Planned commits
1. `docs: plan NivaraTorch parity gap fill in TODO.md` (this file).
2. `test(nivara-torch): add PyTorch parity fixtures for Llama-family + new building blocks`
   (gen_reference.py + `samples/data/torch-comparison/` tree).
3. Per-phase test class commits:
   - `test: Silu + RMSNorm module + RotaryEmbedding parity tests`
   - `test: LlamaCausalAttention + LlamaDecoderBlock + LlamaForCausalLM parity tests`
   - `test: DepthwiseSeparableConv2d + TransformerBlock + SparseEmbedding parity tests`
4. `docs: update NivaraTorch README fixture table + fix stale RMSNorm note`
5. (Phase C, if forward ops exist) `test: forward-mode JVP parity for Silu/RoPE/GqaRepeatKV`
6. Include the pre-staged `docs/BFLOAT16-TRANSFORMER.md` doc update (marks Phase 2 done) as a
   separate commit under this branch.

## Blast radius
- **gen_reference.py**: appended cases only; existing fixtures bit-stable (dedicated RNGs).
- **New test files only** under `tests/Nivara.Tests/NivaraTorch/` + `AutoDiff/ForwardParityTests.cs`
  (additive). **No changes to `src/Nivara/**` product code** unless a parity test reveals a bug.
- **Fixture tree**: regenerating commits ~28+ new `.bin` files; no existing fixture changes.
- Downstream: `dotnet test --filter ...NivaraTorch` runs. No public API changes.

## GitHub issues log

- (empty — populate as deferred work is discovered)
