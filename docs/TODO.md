# Plan: #642 CPU measurement & benchmark tidy-up for #476

## Problem
#476 removed the dense `[L, L]` mask from the CPU ModernBERT/Laya attention paths (band + validLength
carried as metadata), but the tidy-up is incomplete:
- The CPU benchmark path still skips lengths above `ModernBertMasks.MaxDenseLength` and the CLI still
  rejects `--seq` values past it, even though the CPU path no longer allocates a dense mask.
- `CpuBenchmarkSeqLengths` / comments still describe the obsolete cap.
- The CPU measurement called for by #476 acceptance #5 (L=128/256, 0.9% ceiling) is not recorded
  with the ceiling stated up front.
- Docs still describe the CPU path as capped at 2048.

## Proposed changes
1. `samples/NivaraInference/ModernBert.cs`: remove the `MaxDenseLength` skip in `RunBenchmark`; keep
   `CpuBenchmarkSeqLengths = [128, 256]` but fix the XML doc; add a `MaxPositionEmbeddings` guard.
2. `samples/NivaraInference/Program.cs`: drop the CPU dense-mask rejection in `TryParseSeqLengths`
   (remove the now-unused `useGpu` parameter); update help text.
3. `src/Nivara.Extensions/AI/ModernBertModel.cs`: clarify `MaxDenseLength` constrains only the dense
   reference helper `ModernBertMasks.Build`.
4. `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs`: replace the obsolete dense-mask
   reject test with a reach-the-model-lookup check for a large CPU `--seq`.
5. Docs: `docs/MODERNBERT.md`, `docs/LAYA.md`, `docs/ACCELERATION.md`,
   `samples/NivaraInference/README.md`, `samples/NivaraInference/Python/modernbert_benchmark.py`.

## Verification
- Build solution.
- Targeted: `InferenceCliArgumentTests`.
- Manual (weights absent in this checkout): `modernbert benchmark` (128/256) and `modernbert --ilgpu compare`.

## Planned commits
1. docs: plan #642 in TODO.md
2. Remove dense-mask cap from CPU benchmark path (ModernBert.cs + Program.cs)
3. Clarify MaxDenseLength scope (ModernBertModel.cs)
4. Update CLI arg test for lifted CPU cap
5. Docs tidy-up for #642
6. docs: remove TODO.md — plan executed

## GitHub issues log
(none yet)

## Blast radius
- `TryParseSeqLengths` signature change (sample-only, one call site).
- `ModernBert.RunBenchmark` no longer skips; behavior change for any caller passing >2048.
- CLI accepts larger CPU `--seq`; test contract changes.
- No `src/Nivara` public API change.