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
6. `DepthwiseSeparableConv2d` (pre-existing, no parity).
7. `TransformerBlock` (pre-existing, no parity).
8. `SparseEmbedding` module (pre-existing, no parity; ops `SparseEmbeddingBag` are covered).

Deferred (human decision): `LlamaForCausalLM` (tied LM head) full-model parity **not in scope** for this
branch — it is a sample-side composition wrapper, and composition correctness is covered by existing
unit tests + decoder-block parity. Captured as a GitHub issue (see issues log).

Stale docs: `samples/NivaraTorch/README.md` and `gen_reference.py` RMSNorm comments say "Nivara's
RMSNorm has no learnable weight" — outdated since the affine-gamma module.

## Proposed changes

### Phase A — Fixture generation (`samples/NivaraTorch/gen_reference.py`)
Append new cases **at the end** of the generation stream, each on a **dedicated** `torch.Generator`
with a fresh seed. New fixtures:
- `silu_1d`, `silu_4d` (forward + input grad via `.sum().backward()`).
- `rope_1head`, `rope_2head` (RoPE forward + input grad; Llama `rotate_half` half-split with
  torch cos/sin).
- `rmsnorm_module_2d` (affine gamma; forward + grad w.r.t. input and gamma via
  `nn.RMSNorm(..., elementwise_affine=True)`).
- `llama_attn` (GQA: Q/K/V Linear proj → RoPE → KV-repeat → scaled-dot causal attention → O proj;
  forward + input grad).
- `llama_decoder` (pre-norm attn residual + pre-norm gated-SiLU FFN; forward + input grad).
- `dsc` (depthwise separate conv via groups=inCh + ReLU + 1×1; forward + grad).
- `transformer_block_rms`, `transformer_block_ln` (forward + input grad).
- `sparse_embedding` (sum-bag with padding index skipped; forward + weight grad).

> **Determinism fix (departs from the original "existing fixtures bit-stable" blast-radius claim):**
> `gen_reference.py` never seeded the global torch RNG, so `nn.Conv2d`/`nn.Linear` module weights
> (which consume it) changed on every run. The harness now calls `torch.manual_seed(42)` up front and
> the **full** fixture tree (276 files) was regenerated and verified byte-stable across two runs. The
> 33 previously-affected conv/linear fixture values changed as a result; the C# tests are data-driven
> (load from `.bin`), so this is safe. `llama_lm` was dropped (deferred with `LlamaForCausalLM`).

Run `python samples/NivaraTorch/gen_reference.py`, commit the whole
`samples/data/torch-comparison/` tree (manifest + `.bin` files) as one unit.

### Phase B — New NivaraTorch test classes (`tests/Nivara.Tests/NivaraTorch/`)
Match the existing pattern (TestHelpers.LoadBin / AssertTensorEqual, `[SetUp] GradientUtils.Grad()`):
- `SiluTests.cs` — forward + grad.
- `RotaryEmbeddingTests.cs` — forward + grad (2 layouts).
- `RMSNormModuleTests.cs` — module forward + backward (input **and** gamma grads).
- `LlamaCausalAttentionTests.cs`, `LlamaDecoderBlockTests.cs`.
- `DepthwiseSeparableConv2dTests.cs`, `TransformerBlockTests.cs`, `SparseEmbeddingTests.cs`.

(`LlamaForCausalLM` deferred — see issues log.)

Update `samples/NivaraTorch/README.md` fixture table + fix stale RMSNorm text (op vs module).

### Phase C — Forward-mode JVP parity (only forward ops that exist)
Add JVP cross-checks to `tests/Nivara.Tests/AutoDiff/ForwardParityTests.cs` for `Silu` and
`GqaRepeatKV` (both confirmed to have `ForwardGradOperations`): reverse-gradient parity +
finite-difference JVP. RoPE is **reverse-only** (no `ForwardGradOperations.Rotary*`) — skipped.

## Verification
- `python samples/NivaraTorch/gen_reference.py` (done, 276 fixtures, byte-stable across two runs).
- `dotnet build tests/Nivara.Tests` — clean (0 warn / 0 err).
- `dotnet test --filter "FullyQualifiedName~NivaraTorch"` (ask before running).
- `dotnet test --filter "FullyQualifiedName~NivaraTorch"` again after any fixes.

## Planned commits
1. `docs: plan NivaraTorch parity gap fill in TODO.md` (this file). ✅ (5129581)
2. `test(nivara-torch): add PyTorch parity fixtures for Llama-family + new building blocks`
   (gen_reference.py + `samples/data/torch-comparison/` tree). ✅ (6ac5863)
   - Wording adjusted: "plus a determinism fix in the fixture harness" (global-RNG `torch.manual_seed`).
3. `test(nivara-torch): add PyTorch parity + JVP tests for the new building blocks` ✅ (80a7acc)
   - Committed as one unit (silu/rope/RMSNorm-module/Llama-attn/Llama-decoder/DSC/TransformerBlock/
     SparseEmbedding + forward-mode `Silu`/`GqaRepeatKV` JVP), not the originally-planned
     per-module splits.
4. `docs: update NivaraTorch README fixture table + fix stale RMSNorm note` (this commit).
5. The pre-staged `docs/BFLOAT16-TRANSFORMER.md` doc update was committed separately as
   `docs: mark SmolLM Phase 2 complete in BFLOAT16 transformer doc` ✅ (77a20f7).

## Blast radius
- **gen_reference.py**: appended new cases; added `torch.manual_seed(42)` (determinism fix). Existing
  conv/linear fixtures were **regenerated** (see Phase A note) — full tree committed as one unit.
- **New test files only** under `tests/Nivara.Tests/NivaraTorch/` + `AutoDiff/ForwardParityTests.cs`
  (additive). **No changes to `src/Nivara/**` product code** unless a parity test reveals a bug.
- **Fixture tree**: regenerating committed 276 files (updated manifest + `.bin` tree).
- Downstream: `dotnet test --filter ...NivaraTorch` runs. No public API changes.

## GitHub issues log

- [ ] #372 — `LlamaForCausalLM` full-model PyTorch parity (tied LM head + N decoder blocks + final
  RMSNorm) — deferred from the core-block parity branch by human decision.
