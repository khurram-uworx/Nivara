# Plan: Silk parity wiring — NivaraChat, NivaraInference, cleanup

Branch: `khurram/silk` (base `khurram/chat-smollm-silk`).
Headline: Silk is the next-release GPU backend; every user-facing entry point that can run on Silk should honour `--silk` (or be deliberately out of scope with a reason).

## Problem

Silk.NET/OpenCL coverage is already wide in `NivaraInference`, `NivaraClassifier`, `NivaraIncident`, `NivaraChat` RAG/multiview, and `NivaraChat` SmolLM. The gaps are:

- `NivaraChat --embed` and `--intent` call `MiniLMEmbeddingGenerator.Create` directly and ignore `ctx.UseSilk` (`EmbeddingMode.cs:29`, `IntentMode.cs:47`).
- `NivaraChat` RAG in-memory vector-store fallback (`RagSearch.cs`, `useFrame == false`) also instantiates the CPU generator, so an in-memory store cannot use Silk.
- `NivaraInference` rejects two user-facing modes under `--silk`: `minilm … similarity` and `distilbert_sst … predict`. Both can run on the existing Silk encoder result (`Hidden` / `Logits`) without new kernels.
- `NivaraInference/README.md` and help text still describe `--ilgpu` as a live path, although the ILGPU sample half was removed with #814 and `Program.cs` no longer parses `--ilgpu` (it is F32-only Silk + CPU now).

## Proposed changes

### 1. MiniLMEmbedder: add a generator factory

`src/Nivara.Samples/MiniLMEmbedder.cs` is the owner of the "CPU or Silk" embedding factory.
Add `public static IEmbeddingGenerator<string, Embedding<float>> CreateGenerator(string modelDir, bool useSilk)`.

- CPU delegates to `MiniLMEmbeddingGenerator.Create(modelDir)` (unchanged).
- Silk builds `Handle` and wraps `Embed`/`EmbedBatch` in `NivaraEmbeddingGenerator<string>`,
  using `config.HiddenSize` as the dimension and `Nivara-MiniLM (Silk)` as provider name.

### 2. NivaraChat wiring

- `EmbeddingMode.Run()` → `EmbeddingMode.Run(ModeContext ctx)` and use `MiniLMEmbedder.CreateGenerator(minilmDir, ctx.UseSilk)`. `Program.cs` dispatch updated to `EmbeddingMode.Run(ctx)`.
- `IntentMode.Run(ctx)` already has `ctx`: replace the hardcoded generator with `MiniLMEmbedder.CreateGenerator(minilmDir, ctx.UseSilk)`.
- `RagSearch.cs` in-memory fallback: build the generator the same way, reading `ctx.UseSilk`.
- `Program.cs`: extend the `--silk` help line to `--embed`/`--intent`.

### 3. NivaraInference parity modes on Silk

- `RunMiniLmSilkSimilarity`: encode the five fixed sentences through `OpenSilkEncoder(..., BertGpuNaming.Bert)`, take the first `hiddenSize` floats of each `result.Hidden` as the CLS vector, L2-normalize, print the cosine matrix (same output shape as `RunMiniLMSimilarity`).
- `RunDistilBertSstSilkPredict`: interactive loop, single encode per line, `runner.Forward`, `result.Logits`, softmax, print sentiment — same as `RunDistilBertSstPredict` but on the backend runner.
- Add `"similarity"` to `SilkModes["minilm"]` and `"predict"` to `SilkModes["distilbert_sst"]`, plus dispatch branches before the generic fallthrough.
- Update `InferenceCliArgumentTests` Silk mode matrices in the same commit (AGENTS.md coupling).

### 4. Cleanup of stale `--ilgpu` documentation

- `samples/NivaraInference/README.md`: replace "iGPU, `--ilgpu`" phrasing with "iGPU (Silk, `--silk`)" where the number is Silk-derived, and mark ILGPU-only commands as historical.
- `Program.cs` help line: drop `--ilgpu` from the usage template; mention Silk.

### 5. docs/SILK.md

- Add a short "Wiring coverage" subsection listing NivaraChat (`--embed`, `--intent`, `--rag*`, `--smollm`) and NivaraInference (models/modes) and the deliberately-out-of-scope items.

## Verification

- `dotnet build Nivara.slnx` (core + Extensions)
- `dotnet build samples/NivaraChat/NivaraChat.csproj -c Release`
- `dotnet build samples/NivaraInference/NivaraInference.csproj -c Release`
- targeted: `InferenceCliArgumentTests`
- manual smoke: `NivaraChat --embed --silk`, `--intent --silk`, `minilm --silk similarity`, `distilbert_sst --silk predict` (only if model dirs present).

## GitHub issues log

- (none yet)

## Blast radius

- Samples only; no `src/Nivara` core changes, no public API changes.
- `MiniLMEmbedder.CreateGenerator` is the only new public API addition (samples assembly).
- `InferenceCliArgumentTests` must change in lockstep (coupling).
- DocsCoveragePins: review after commit; expected no count change, will verify.