# Plan: Wire `--synthetic-weights` for SmolLM (issue #1254)

## Problem

SmolLM has synthetic benchmark helpers (`RunSmolLMSyntheticBenchmark`, `RunSmolLMSyntheticAb`, `RunSmolLMSyntheticBenchmarkSilk`) but they are dead code: `--synthetic-weights` is rejected for every model except qwen, and no CLI path reaches them. SmolLM should gain the same checkpoint-free synthetic surface as Qwen so all scenarios (CPU/Silk, f32/bf16/native, single-shot + benchmark) run on a clean checkout.

## Proposed changes

### A. Guards (samples/NivaraInference/Program.cs)
- Allow `--synthetic-weights` for `smollm` (not just qwen).
- Extend the mode guard (default/benchmark only) to smollm.
- Help text: `(qwen only)` → `(qwen/smollm)`.

### B. Dispatch (before model-file lookup)
Add smollm synthetic block mirroring Qwen's:
- Silk → `RunSmolLMSyntheticBenchmarkSilk(plain, bf16 || native)`
- native → `RunSmolLMSyntheticBenchmark<float, BFloat16>()`
- bf16 → `RunSmolLMSyntheticBenchmark<BFloat16, BFloat16>()`
- else → `RunSmolLMSyntheticBenchmark<float, float>()`

### C. Synthetic runner refactor
- `RunSmolLMSyntheticBenchmark<T>()` → `RunSmolLMSyntheticBenchmark<TModel, TWeight>()` (mixed): synthesize in `TWeight`, load `LlamaLoader.Load<TModel, TWeight>`, native precision label when mixed.
- `RunSmolLMSyntheticBenchmarkSilk()` → `RunSmolLMSyntheticBenchmarkSilk(bool plain, bool bf16)`: synthesize float or BFloat16 and open matching `SilkSmolLMSession` overload.

### D. Tests
- Update guard message expectation in `InferenceCliArgumentTests`.
- Add smollm beyond-benchmark-mode reject.
- Add accepted smollm synthetic cases.

## Verification steps

- `dotnet build Nivara.slnx`
- Targeted: `InferenceCliArgumentTests`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Manual runs of `smollm --synthetic-weights` (CPU + Silk, f32/bf16/native).

## Planned commits

1. docs: plan synthetic-weights SmolLM wiring in TODO.md
2. feat: allow --synthetic-weights for smollm (guards + help)
3. feat: refactor SmolLM synthetic runners for mixed dtype + BF16 silk
4. feat: dispatch smollm synthetic path
5. test: CLI coverage for smollm synthetic
6. docs: update SMOLLM.md synthetic section (if present)

## GitHub issues log

- (none yet)

## Blast radius

- `samples/NivaraInference/Program.cs`: guards, dispatch, two synthetic helpers.
- `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs`: guard message + new cases.
- No library changes; no test count pins affected except CLI gate.

Reminder: as each task executes, if you find deferred work or a concern, create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its number here.
