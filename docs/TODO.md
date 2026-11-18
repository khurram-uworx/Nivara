# Plan: #1312 — ToSilkWeights() on BatchedTransformer

## Problem
`Silk*GpuRunner` constructors take `Dictionary<string, (float[] Data, int[] Shape)>` with conventional keys, but `BatchedTransformer` stores parameters under `Module_0.Weight`-style keys via `Module.StateDict()`. A runner needs a stable mapping with conventional, documented keys. Add `ToSilkWeights()` that copies data out (no alias) and reports shapes.

## Proposed changes
- `BatchedTransformer<T>.ToSilkWeights()` → dictionary with keys:
  - `tok_emb.weight`
  - `final_norm.weight`, `final_norm.bias`
  - per layer `l`: `layers.{l}.ln1.weight/bias`, `ln2.weight/bias`, `q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `o_proj.weight`, `mlp_fc1.weight/bias`, `mlp_fc2.weight/bias`
- Helper `BatchedTransformer<T>.TensorToFloatArray` converts `T`→`float` (identity for float).
- `BatchedTransformerBlock<T>.PopulateSilkWeights(int, dict)` fills per-layer entries.
- Data is copied (not aliased); shapes cloned.

## Verification
- Build `dotnet build Nivara.slnx`.
- Targeted test: small model, key count `3 + 12*nLayer`, shapes.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.

## Planned commits
1. docs: plan #1312 in TODO.md
2. feat: add ToSilkWeights() export helper
3. test: add unit test for key set/shapes
4. docs: remove TODO.md — plan executed

## GitHub issues log
- (none yet)

## Blast radius
- `samples/NivaraChat/Transformer/BatchedTransformer.cs` (BatchedTransformer + BatchedTransformerBlock).
- No change to training/serving behavior.