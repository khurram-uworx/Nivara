# TODO — TypeSafeAI News Assessment PoC

## Problem

Probe/PoC: use TypeSafeAI NuGet + Ollaya (local decision-model server) to classify and assess AI/ML news about Laya/Jev/System 1 models. This is unrelated to Nivara core — it's a standalone scenario to learn the TypeSafeAI API and validate the Ollaya local endpoint.

## Proposed Changes

1. **New mode `--typesafe` in NivaraChat** — add a `TypesafeMode` class under `samples/NivaraChat/Modes/`
2. **Configure `TypeSafeClient`** to point at Ollaya (`http://localhost:11435`, model `laya:en`, dummy API key)
3. **Design a `QuestionSet`** for AI/ML news assessment:
   - `Choice` — Topic category: `ModelRelease`, `ResearchPaper`, `Benchmark`, `Opinion`, `Other`
   - `Noul` — "Does this describe a model release or announcement?"
   - `Noul` — "Does this mention benchmark results?"
   - `Score` — "How significant is this development?" (`minor`, `moderate`, `major`, `breakthrough`)
   - `Score` — "How relevant is this to production AI/ML engineering?" (`low`, `medium`, `high`)
   - `Noul` — "Does this contain actionable technical details?"
4. **Fabricate 5–6 realistic AI/ML news snippets** about Laya/Jev/System 1 models
5. **Wire into `Program.cs`** — add `--typesafe` case to the mode switch
6. **Output**: Print each news item with structured assessment (category, significance, relevance, confidence)

## Verification

- Build: `dotnet build samples/NivaraChat/NivaraChat.csproj`
- Run: `dotnet run --project samples/NivaraChat -- --typesafe`
- Validate output shows structured assessments for each news item

## Planned Commits

1. `docs: plan TypeSafeAI news assessment PoC in TODO.md`
2. `feat: add TypesafeMode with Ollaya endpoint and news assessment questions`
3. `feat: wire --typesafe mode into Program.cs`

## Blast Radius

- **Files touched**: `samples/NivaraChat/Modes/TypesafeMode.cs` (new), `samples/NivaraChat/Program.cs` (one new case)
- **No impact on Nivara core** — purely additive to the NivaraChat sample
- **No existing tests affected** — new code only

## GitHub issues log

- (none yet)
