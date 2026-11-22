# #1444 - Consolidate LoRA docs

## Problem

Issue #1444: `docs/LORA.md` and `docs/plan/LORA.md` overlap and their roles drifted after LoRA work landed. The plan doc is no longer an active handoff tracker; keeping it creates contradictory/stale status. User confirmed direction: delete `docs/plan/LORA.md` and keep `docs/LORA.md` as the stable public document, enriched with end-to-end run guidelines and durable engineering notes from the plan.

## Grounding (G1)

- **Blast radius (small, docs-only):** `docs/LORA.md` (rewrite), `docs/plan/LORA.md` (delete), `samples/NivaraChat/README.md` (1 ref), `samples/NivaraChat/ARCHITECTURE.md` (1 ref), `tests/Nivara.Tests/Docs/DocsCoveragePins.cs` (comment/pin arithmetic may change). No runtime code changes.
- **Doc gate contracts checked:** `DocSnippetExtractor.ScanExcludedDocuments` excludes only `docs/TODO.md` (`tests/Nivara.Tests/Docs/DocSnippet.cs:148`); `docs/plan/LORA.md` is currently scanned, so deleting it affects discovered doc-reference/citation counts. Top-level `docs/*.md` count is unaffected because `docs/plan/LORA.md` is not top-level (`DocsCoveragePins.ExpectedDocumentCount` stays 63).
- **Citations in deleted file:** `docs/plan/LORA.md:112` contains `samples/NivaraFineTuning/Program.cs:1362+` (code citation). Removing/moving it affects citation pins; prefer dropping brittle line citation unless it adds real value, then update pin arithmetic with reason.
- **No new .NET/Tensor APIs**; microsoft-learn grounding not required. Code-memory/sql used to confirm LoRA symbol landscape; no code behavior changes.
- **Decision resolved by user:** delete plan, consolidate into `docs/LORA.md`. No open red flags.

## Proposed changes

1. **Rewrite `docs/LORA.md`** as stable public guide:
   - Role note: single public LoRA doc (no plan/handoff file).
   - Status: landed capabilities; CPU reference; Silk trainer/eval; synthetic GPU-gated parity confirmed; real-checkpoint forward+loss smokes `[Explicit, Category("Gpu")]`; honest limitations.
   - End-to-end runbook: prerequisites/data, `ft-generate`, `ft-lora` CPU, `ft-lora --silk`, `ft-eval` CPU/Silk, NivaraChat adapter consumption, expected artifacts.
   - Design decisions/scope: D1-D5 condensed from plan.
   - Adapter formats & compatibility: `nivara-lora-adapter-v1` vs `nivara-lora-sft-adapters-v1`, no bridge by design.
   - Parity, evidence, guardrails: tolerances, no silent CPU fallback, no false iGPU claim, advisory memory contract.
   - Engineering notes/observations: 0.5B CPU backward slow for smoke, device-specific smoke evidence not release certification, shared bundle schema, report fields.
   - Key files (paths/symbols, no brittle line citations).
2. **Delete `docs/plan/LORA.md`**.
3. **Update references**:
   - `samples/NivaraChat/README.md` - replace `docs/plan/LORA.md` reference with `docs/LORA.md`.
   - `samples/NivaraChat/ARCHITECTURE.md` - same.
4. **Reconcile docs gate pins** if needed after running docs tests:
   - `tests/Nivara.Tests/Docs/DocsCoveragePins.cs` - update ExpectedDocumentReferenceCount / citation pins only if counts changed; update arithmetic comment with reason. Do not bump blindly.
5. **Verify**:
   - `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
   - `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` (ask before running? user explicitly asked to test; run targeted docs tests as normal verification).
   - Review branch diff before PR.

## Planned commits

1. `docs: plan #1444 LoRA docs consolidation in TODO.md`
2. `docs: consolidate LoRA plan into public docs/LORA.md (#1444)`
3. `test/docs: revise DocsCoveragePins after LoRA doc consolidation (#1444)` (only if needed)
4. `docs: remove TODO.md - plan executed` (after G2 review clears)

## GitHub issues log

- [ ] #1444 — LoRA docs: clarify/consolidate docs/LORA.md vs docs/plan/LORA.md roles (this work)

## Verification checklist

- [ ] No references to `docs/plan/LORA.md` remain (except git history).
- [ ] `docs/LORA.md` has clear role, end-to-end runbook, durable engineering notes, no contradictory status.
- [ ] `docs/plan/LORA.md` deleted.
- [ ] DocsCoveragePins arithmetic/comments accurate if changed.
- [ ] `dotnet format whitespace --verify-no-changes` passes.
- [ ] Docs tests pass in Release.
- [ ] G2 branch review (whole + against TODO) clear before TODO removal.
