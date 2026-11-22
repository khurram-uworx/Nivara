# Plan: SmolLM/Qwen CPU viability, synthetic parity confirmation, and timing diagnostics (#1443)

## Problem

Issue #1443 as filed is "LoRA: record post-merge CPU ft-lora smoke evidence" (OPEN). The requested work on this branch, as stated by the human, is different:

- Qwen on CPU is effectively unusable because it takes too long.
- CPU/Silk parity has been validated using synthetic weights with tighter control so it can "just work at least".
- Try SmolLM with more logging to determine what is happening and where time is spent.
- If SmolLM CPU is still not viable, rely on synthetic-weight confirmation as the confidence gate.

**Scope decision:** proceed with the human-requested SmolLM/Qwen CPU viability + synthetic parity + diagnostics work on branch `khurram/1443`. Record the #1443 mismatch in PR description; do not silently close #1443 unless the human confirms the issue should be repurposed.

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

### A. Coarse timing/logging for SmolLM CPU path

Add prefill/decode split timing to SmolLM CPU generation, matching Qwen's existing `GenerateCore` split:

- `RunSmolLMGeneration<TModel,TWeight>`: measure `ForwardPrefill` separately from decode loop; return/include prefill/decode ms (extend return tuple or add an overload/helper that preserves existing call sites).
- `BenchmarkSmolLMCore`, `RunSmolLMSyntheticBenchmark`: print `Prefill: X ms`, `Decode: Y ms (Z ms/token)`, total/tok/s.
- `ConsumeGenerated` / `RunSmolLMGenerate`: print prefill/decode split in normal output.
- Keep default output coarse and useful; avoid per-op/per-layer spam. If deeper diagnostics needed, gate behind an explicit opt-in (env var `NIVARA_VERBOSE_TIMING=1` or CLI flag — decide during G1/implementation; CLI flag requires `InferenceCliArgumentTests` update).

### B. Silk SmolLM timing parity

- `RunSmolLMGenerationSilk`: split timing around `ForwardPrefillArgmax` and decode loop where measurable. Report prefill/decode/total.
- Keep CPU/Silk comparison apples-to-apples for same prompt/max tokens/precision.

### C. Synthetic-weight confirmation / tighter control

- Confirm existing synthetic parity tests remain the confidence gate: `SilkSmolLMParityTests`, `SilkQwenParityTests` (GPU category, skip when no OpenCL).
- Add/strengthen CPU-only deterministic synthetic smoke where cheap: e.g. synthesize SmolLM/Qwen weights, run small prefill+few decode steps on CPU, assert deterministic generated ids/argmax for fixed seed/prompt/config and no NaN/Inf. Must be fast, no GPU, no real checkpoint.
- If adding tests, keep them focused and not brittle; prefer extending existing test classes/files only when natural.

### D. Documentation / evidence

- Update `docs/QWEN.md` or relevant doc section with a concise note: Qwen real-checkpoint CPU can be impractical; SmolLM is the smaller diagnostic path; synthetic-weight CPU/Silk parity is the fallback confidence gate when real-checkpoint CPU is too slow. Include exact commands/conditions as appropriate.
- Do not invent benchmark numbers. Only record numbers actually observed in this session/human-run artifacts.
- Review `DocsCoveragePins`/doc gates if citations/snippet blocks/counts change.

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

- [ ] #1443 — "LoRA: record post-merge CPU ft-lora smoke evidence" (existing; branch name matches but issue scope differs from requested work — confirm disposition with human)
- [ ] Create new issue if this work needs its own tracker separate from #1443.

## Notes

- As each task executes, if deferred work or concerns appear, create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record it here.
- Do not use skills other than the one explicitly requested by the user for this workflow; do not push without confirmation.
- Benchmarks/real-checkpoint runs are human-run per AGENTS.md.
