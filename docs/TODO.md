# TODO: Issue #1416 — LoRA behavioral worklog SFT dataset and deterministic generator

Branch: `khurram/1416` (off `khurram/lora`)
Issue: https://github.com/khurram-uworx/Nivara/issues/1416

## Problem

Issue #1416 needs a deterministic synthetic behavioral SFT dataset for the worklog/LogTime showcase (Llama-family LoRA fine-tuning, headline Qwen2.5-0.5B, SmolLM generic verification). Core dataset/generator/evaluator/CLI pieces exist on `khurram/lora`, but #1416 is not complete: the generator does not cover relative-date phrasing required by scope/acceptance, fraction/seed-data conventions need an explicit deterministic contract, CLI split/provenance behavior needs tightening, and the acceptance tests are missing.

## Blast radius

- `samples/NivaraFineTuning/WorklogSftDataset.cs` — generator, split/hash, save/load, tokenization contract.
- `samples/NivaraFineTuning/WorklogSftEvaluator.cs` — parsing/metrics; may need small contract fixes only if tests expose bugs.
- `samples/NivaraFineTuning/Program.cs` — `ft-generate` split/provenance/CLI options; minimal.
- `samples/data/worklog/*` — seed data/provenance may be regenerated for deterministic contract; README updated.
- `tests/Nivara.Tests/Nivara.Tests.csproj` — add ProjectReference to `samples/NivaraFineTuning/NivaraFineTuning.csproj`.
- `tests/Nivara.Tests/FineTuning/WorklogSftDatasetTests.cs` (new) — dataset/generator/split/overwrite/format/token-mask tests.
- `tests/Nivara.Tests/FineTuning/WorklogSftEvaluatorTests.cs` (new) — evaluator metric/parsing/clarification/convention tests.
- Docs/pins: no `docs/*.md` added/removed expected; if any citation-bearing doc changes, revise DocsCoveragePins. `docs/TODO.md` removed before PR (iterative-work G2).

Downstream callers: `WorklogLoraSftTrainer` and `ft-lora`/`ft-eval` CLI consume dataset/evaluator APIs; keep public API compatible or update call sites in same commit.

## Proposed changes

1. **Contract lock / generator fixes**
   - Add deterministic relative-date coverage in `GenerateSynthetic`: fixed base date policy (default `2026-07-19`, explicit in metadata), seed-deterministic relative phrasing (`today`, `yesterday`, `tomorrow`, weekday phrasing as needed) resolved to ISO `YYYY-MM-DD`.
   - Codify explicit duration/fraction rule: stated decimal/integer hours as-is; minute durations converted by deterministic rule (`minutes / 60m`, rounded to 2 decimals, invariant formatting, trim unnecessary trailing zeros) and covered by tests.
   - Keep JSONL schema `{"messages":[system,user,assistant]}`; preserve tool-call format `<tool_call><function>LogTime</function>...`.
   - Ensure generated dataset covers: all-day/full-day=8h, half-day=4h, explicit durations, multi-ticket separate calls, ambiguous/non-worklog clarification, relative dates.

2. **Split/provenance/CLI**
   - Keep `SplitDeterministic` deterministic train/val/test behavior and provenance hash stable for same inputs.
   - Extend `ft-generate` to optionally emit `val.jsonl` (default 0 for backcompat unless changed) and include richer provenance metadata (format, seed, baseDate, counts, ratios, hash, generator, fraction rule).
   - Keep overwrite refusal without `--force` for generated outputs.

3. **Tests (acceptance)**
   - Dataset: determinism same seed/count; different seed differs; category coverage; schema/ISO date/tool-call parseability; split/provenance stability; partition/no-overlap; SaveJsonl overwrite refusal; Load/Save roundtrip; tokenization prompt-mask/answer labels.
   - Evaluator: parse single/multi/malformed calls; clarification detection; perfect prediction metrics; multi-ticket exactness; convention all-day/half-day scoring; clarification scoring.

4. **Data/provenance refresh (if needed)**
   - Regenerate `samples/data/worklog` via generator with recorded seed/count/ratios and update README/provenance. Only if contract changes require it; keep deterministic and documented.

## Planned commits

1. `docs: plan issue 1416 worklog SFT dataset completion`
2. `test(ft): add WorklogSft dataset and evaluator coverage for #1416`
3. `feat(ft): cover relative dates and deterministic worklog generation contract for #1416`
4. `chore(ft): refresh worklog seed data and provenance for #1416` (if needed)
5. `docs: remove TODO.md — plan executed` (after G2 clears, before push/PR)

## Verification

- `dotnet build samples/NivaraFineTuning/NivaraFineTuning.csproj -c Release`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~WorklogSft"` (ask before running; targeted)
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` before push
- Smoke: `dotnet run --project samples/NivaraFineTuning/NivaraFineTuning.csproj -c Release -- --mode ft-generate --out-dir <tmp> --seed 42 --max-examples 20 --force` twice, compare hashes/outputs.
- If docs change: targeted Docs tests as needed; revise DocsCoveragePins only if required by rules.

## GitHub issues log

- [ ] #1416 — LoRA: behavioral worklog SFT dataset and deterministic generator (this branch)
- [ ] (add discoveries here immediately during execution)

## Grounding / decisions

- Grounded in codebase: `WorklogSftDataset.cs`, `WorklogSftEvaluator.cs`, `Program.cs`, `samples/data/worklog/*`, `docs/plan/LORA.md`, issue #1416 body/comments. No new .NET/BCL tensor API assumptions requiring microsoft-learn verification; work is internal sample/data/test code.
- Decision: keep JSONL schema stable; put prompt-mask metadata in tokenization API/tests, not persisted JSONL fields, unless owner requests schema change.
- Decision: relative-date base default is fixed and recorded in provenance; generator remains deterministic given same seed/count/base/ratios.
- Decision: deterministic minute fraction rule codified and tested; seed data regenerated if it diverges from contract.
- Red flag to watch: regenerating seed data changes existing train/test files - document provenance and avoid silent drift.
