# Plan: SmolLM/Qwen CPU viability, synthetic parity confirmation, and timing diagnostics (#1443)

## Problem

Issue #1443 as filed is "LoRA: record post-merge CPU ft-lora smoke evidence" (OPEN). The requested work on this branch, as stated by the human, is different:

- Qwen on CPU is effectively unusable because it takes too long.
- CPU/Silk parity has been validated using synthetic weights with tighter control so it can "just work at least".
- Try SmolLM with more logging to determine what is happening and where time is spent.
- If SmolLM CPU is still not viable, rely on synthetic-weight confirmation as the confidence gate.

**Scope decision:** human confirmed (option 2) repurposing #1443. Issue retitled to "CPU inference viability: SmolLM/Qwen diagnostics and synthetic parity fallback" and body updated to match this work. Proceed on branch `khurram/1443`.

## Blast radius

| Area | Files/symbols | Risk | Coverage |
|------|---------------|------|----------|
| SmolLM CPU generation timing | `samples/NivaraInference/Program.cs` — `RunSmolLMGeneration<TModel,TWeight>`, `BenchmarkSmolLMCore`, `RunSmolLMSyntheticBenchmark`, `RunSmolLMGenerate`, `ConsumeGenerated` | Low-medium: sample CLI output/timing only; no core/runtime behavior change | Existing inference CLI arg tests; manual/targeted smoke; synthetic parity tests unchanged |
| Silk SmolLM timing parity output | `samples/NivaraInference/Program.cs` — `RunSmolLMGenerationSilk`, `BenchmarkSmolLMSilk`, `RunSmolLMSyntheticBenchmarkSilk` | Low: sample output only | Existing Silk parity tests |
| Synthetic parity confidence | `tests/Nivara.Tests/SmolLM/SilkSmolLMParityTests.cs`, `tests/Nivara.Tests/Qwen/SilkQwenParityTests.cs`, `tests/Nivara.Tests/SmolLM/SmolLMTestWeights.cs`, `tests/Nivara.Tests/Qwen/QwenTestWeights*.cs` | Low: tests only, GPU-category where device needed | Targeted NUnit runs |
| CPU-only synthetic determinism/smoke (if added) | New/extended test under `tests/Nivara.Tests/SmolLM/` or `Qwen/` | Low: must be fast/deterministic; no GPU | Targeted NUnit runs |
| Docs | `docs/QWEN.md`, possibly `docs/SILK.md` or sample README; DocsCoveragePins if citations/blocks/counts change | Low-medium: snippet/citation gates sensitive | `Nivara.Tests.Docs` namespace |
| CLI surface | Only if adding explicit flag/env; `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs` coupling | Medium if flag added | `InferenceCliArgumentTests` |

No core `src/Nivara` changes planned. No public library API changes planned. Sample/CLI and tests/docs only unless grounding reveals a real instrumentation hook needed in core.

## Grounding (G1)

- [ ] Ground via microsoft-learn MCP: .NET `Stopwatch`/timing guidance as needed, `Console` logging patterns only if relevant; confirm no unstable .NET API assumptions.
- [ ] Ground via code-memory MCP: confirm SmolLM/Qwen generation call graph, synthetic weight synthesis, Silk session API timing boundaries (`ForwardPrefill`, `ForwardPrefillArgmax`, `ForwardCached*`).
- [ ] State blast radius (above) and confirm no hidden core dependency.
- [ ] Escalate any decision/red flag to human before implementation.

## Proposed changes

### A. Coarse timing/logging for SmolLM CPU path (done)

- [x] `RunSmolLMGeneration<TModel,TWeight>`: split `ForwardPrefill` and decode loop timing; return tuple includes `PrefillMs`/`DecodeMs`.
- [x] `BenchmarkSmolLMCore`, `RunSmolLMSyntheticBenchmark`, `RunSmolLMGenerate`/`ConsumeGenerated`, AB paths: print prefill/decode split and decode throughput as `tok/s`.
- [x] Default output coarse; no new CLI flag added (avoids `InferenceCliArgumentTests` surface change).

### B. Silk SmolLM timing parity (done)

- [x] `RunSmolLMGenerationSilk`: split timing around `ForwardPrefillArgmax` and decode loop; report prefill/decode/total with `tok/s`.
- [x] CPU/Silk comparison uses same prompt/max tokens/precision fields.

### C. Synthetic-weight confirmation / tighter control (done)

- [x] Existing GPU-gated parity tests remain confidence gate: `SilkSmolLMParityTests`, `SilkQwenParityTests`.
- [x] Added CPU-only deterministic synthetic smoke tests: `SmolLMSyntheticSmokeTests`, `QwenSyntheticSmokeTests` (fast, no GPU, no checkpoint, deterministic ids + finite logits).
- [x] Added GPU-gated Silk synthetic CPU/Silk timing comparison tests in `SilkSmolLMParityTests` and `SilkQwenParityTests` to report prefill/decode `tok/s` for future planning.
- [x] No tolerance loosening; existing `1e-3 * (1 + max|cpu|)` parity bound unchanged.

### D. Documentation / evidence (done)

- [x] Updated `docs/QWEN.md` Interpretation with CPU viability fallback: synthetic-weight CPU/Silk parity gate and SmolLM as smaller diagnostic path with prefill/decode `tok/s` logging.
- [x] No invented benchmark numbers.
- [x] No new citations, fenced csharp blocks, or doc files; DocsCoveragePins unchanged.

## Verification steps

- [ ] `dotnet build Nivara.slnx -c Release`
- [ ] `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` (before commit/PR)
- [ ] Targeted tests (ask before running): `InferenceCliArgumentTests` if CLI touched; SmolLM/Qwen parity/smoke tests added or touched; any affected sample tests.
- [ ] `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` if docs changed (ask before running — but docs namespace is targeted; AGENTS allows targeted runs).
- [ ] Snippet gate covered by Docs namespace run; confirm no new gated blocks broken.
- [ ] Do **not** run Qwen real-checkpoint benchmarks; do not run full test suite without explicit human confirmation.
- [ ] Human-run only if desired: SmolLM synthetic/real CPU benchmark commands, recorded outside agent polling.

## Planned commits

1. `docs: plan SmolLM CPU diagnostics and synthetic parity fallback`
2. `feat: split SmolLM CPU prefill/decode timing logs` (sample CLI timing/output; tests if needed)
3. `test: add deterministic CPU synthetic smoke for SmolLM/Qwen` (if cheap and useful)
4. `docs: record CPU viability fallback and synthetic parity gate` (if docs updated)
5. `docs: remove TODO.md — plan executed` (only after G2 passes)

## GitHub issues log

- [ ] #1443 — "CPU inference viability: SmolLM/Qwen diagnostics and synthetic parity fallback" (repurposed per human confirmation; tracks this branch's work)

## Notes

- As each task executes, if deferred work or concerns appear, create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record it here.
- Do not use skills other than the one explicitly requested by the user for this workflow; do not push without confirmation.
- Benchmarks/real-checkpoint runs are human-run per AGENTS.md.
