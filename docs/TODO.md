# Plan: #1277 — extend selected-weight BF16 storage to MiniLM / DistilBERT / SST Silk runners

## Problem

Silk (OpenCL) BERT-family inference is opt-in narrow precision. ModernBERT and Laya already store
selected matmul weights (QKV, FFN, tokEmb? no—tokEmb F32) as BF16 on device with F32 activations.
MiniLM, DistilBERT and DistilBERT SST-2 currently widen BF16 weights to F32 at load on the Silk path
(`Program.cs`: `miniSilkTensors = bf16 ? WidenBFloat16Tensors(tensorsBf16) : F32();`), so they get no
memory benefit and no true narrow lane.

Core runner already supports BF16 (`SilkBertEncoderRunner` BFloat16 constructor → `isBf16` path;
`SilkBertWeights` stores QKV/O/FFN as `ushort`, biases/LN/embeddings F32). Parity tests pass.
Gap is sample wiring.

## Proposed changes

1. `samples/NivaraInference/Program.cs` — add `OpenSilkEncoderBf16` helper.
2. Dispatch for `minilm`/`distilbert`/`distilbert_sst` under `useSilk`:
   - `bf16`: default inference (+ similarity/predict) → narrow methods with BFloat16 tensors;
     benchmark/compare → widened F32 (existing pattern).
   - else → `F32()` unchanged.
3. Add narrow default-inference methods (BFloat16 tensors, `OpenSilkEncoderBf16`).
4. `silkNarrowModel` allow-list: add the three models.

## Verification

- `dotnet test … SilkBertNarrowParityTests` (green).
- Build solution.
- CLI: `minilm/distilbert/distilbert_sst --silk --precision bf16` reaches model-file lookup.
- End-to-end on real checkpoint (human-run).

## Planned commits

1. docs: plan #1277 in TODO.md
2. feat(sample): wire BF16 narrow lane for MiniLM/DistilBERT/SST Silk runners
3. docs: remove TODO.md — plan executed

## Blast radius

- `samples/NivaraInference/Program.cs` dispatch + new helpers/methods.
- No core (Nivara.Extensions) changes.
- CLI tests may need `silkNarrowModel` update; existing InferenceCliArgumentTests unaffected.

## GitHub issues log

(none yet)
