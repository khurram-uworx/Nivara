# TODO: #1442 clarify LoRA report memory footprint contract

## Problem

Issue #1442 (`khurram-uworx/Nivara#1442`, open): LoRA fine-tuning eval/report memory footprint is approximate/advisory or unspecified, limiting usefulness for capacity planning.

Evidence from issue:
- `docs/plan/LORA.md` known gaps note "Memory footprint estimate still approximate".
- `ft-eval` report sets `MemoryBudgetMode` as `advisory` when `--max-memory` is provided, otherwise `unspecified`; it does not report measured peak/actual memory footprint.

Acceptance criteria (issue):
- Report contract for memory is explicit and documented.
- If "actual measured" is chosen, report includes measured value and tests cover schema/behavior.
- If "advisory only" is chosen, docs/field names make that unambiguous.

## G1 decision (cleared by human)

Contract direction: **Advisory-only, clarified** (human confirmed). We will not add measured peak memory. We will make the requested-budget/advisory-only contract explicit and unambiguous in report schema docs, field descriptions, markdown labels, CLI help, user docs, and tests. JSON property names remain stable (`memoryBudgetBytes`, `memoryBudgetMode`) to avoid unnecessary schema churn; semantics are clarified.

## Grounding / blast radius

- Code-memory navigation: `semantic_search` + `sql_query` on `SymbolRecord` for `MemoryBudget*`/report/eval symbols; `find_related_code` on `LoraSftRunReport`.
- Current report schema: `samples/NivaraFineTuning/LoraSftRunReport.cs` defines `MemoryBudgetBytes` (long?, requested), `MemoryBudgetMode` ("advisory"|"unspecified"), markdown "Memory budget" section with "advisory only; not a hard runtime cap".
- CLI wiring: `samples/NivaraFineTuning/Program.cs` parses `--max-memory` (default 0), help says advisory/not hard cap, sets both fields in all 4 report sites (ft-lora CPU, ft-lora Silk, ft-eval CPU, ft-eval Silk).
- Docs: `docs/LORA.md` limitation #1442 says contract needs clarification; `docs/plan/LORA.md` tracks G3/#1442.
- Tests: `tests/Nivara.Tests/FineTuning/LoraSftRunReportTests.cs` round-trips memory fields and asserts markdown contains memory budget/unspecified; `FtEvalCliDispatchTests.cs` unrelated to memory contract.
- No measured memory capture exists in NivaraFineTuning (`GC.GetTotalMemory`/`Process`/peak not used).

Blast radius: small, localized to NivaraFineTuning report contract + docs/tests.
- `samples/NivaraFineTuning/LoraSftRunReport.cs` — property XML docs/comments, markdown labels/text.
- `samples/NivaraFineTuning/Program.cs` — `--max-memory` help text / console wording only where needed.
- `tests/Nivara.Tests/FineTuning/LoraSftRunReportTests.cs` — assert clarified contract strings/semantics.
- `docs/LORA.md` — update limitation/status text for #1442.
- `docs/plan/LORA.md` — mark G3/#1442 resolved/clarified.
- No core/runtime API changes; no new dependencies; no measured-memory behavior added.

## Proposed changes

1. **Clarify report schema semantics** (`LoraSftRunReport.cs`)
   - Add XML doc comments on `MemoryBudgetBytes` and `MemoryBudgetMode` stating: requested advisory budget in bytes supplied via `--max-memory`; not measured runtime footprint; not a hard cap/enforcement. `MemoryBudgetMode` indicates whether a budget was requested (`advisory`) or not (`unspecified`).
   - Update markdown section to be unambiguous, e.g. "## Memory budget (advisory, requested only)" / labels "Requested advisory budget (bytes)" and "Mode: advisory (requested budget only; not measured runtime footprint; not a hard cap)". Keep JSON property names unchanged.

2. **Clarify CLI contract** (`Program.cs`)
   - Tighten `--max-memory` help text to explicitly say "requested advisory memory budget... reported in run report; not measured footprint; not a hard runtime cap".
   - Keep console/report assignment behavior unchanged.

3. **Docs**
   - `docs/LORA.md`: replace limitation #1442 with clarified contract note (advisory requested budget only; no measured peak).
   - `docs/plan/LORA.md`: update G3/#1442 status and known-gap text to reflect clarified advisory-only contract.

4. **Tests** (`LoraSftRunReportTests.cs`)
   - Extend markdown assertions to pin clarified wording (advisory/requested/not measured/not hard cap as appropriate).
   - Add/adjust round-trip case confirming advisory-only contract fields serialize as before.

5. **Verification / review**
   - Ask before running tests. Run targeted FineTuning tests.
   - Run docs snippet gate / DocsCoveragePins if docs touched as needed (check counts; revise only if required).
   - G2 branch review before deleting TODO.

## Planned commits

1. `docs: plan #1442 advisory memory report contract in TODO.md`
2. `fix: clarify advisory-only memory budget contract in LoRA reports`
3. `test: pin LoRA report advisory memory contract`
4. `docs: record #1442 advisory memory contract clarification` (if docs changes not bundled; may bundle with fix/test as appropriate by logical unit)
5. `docs: remove TODO.md - plan executed` (only after G2 clears)

## GitHub issues log

- [ ] #1442 — clarify LoRA report memory footprint contract (this plan)

## Verification

- [ ] `dotnet build samples/NivaraFineTuning/NivaraFineTuning.csproj -c Release`
- [ ] Targeted tests (ask before running): `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~LoraSftRunReportTests|FullyQualifiedName~FtEvalCliDispatchTests"`
- [ ] If docs citations/counts change, run relevant DocsCoverage gates; revise DocsCoveragePins only if required.
- [ ] G2: branch review as a whole + against this TODO before deleting TODO.

## Notes

- No push/PR without explicit human confirmation (will follow user request when work is ready).
- As each task executes, create GitHub issues immediately for any deferred/out-of-scope concern and record here.
