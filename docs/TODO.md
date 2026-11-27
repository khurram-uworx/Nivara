# TODO: Docs refresh for release v1.5.4 (Issue #1518)

## Problem

Issue #1518 tracks the full RELEASING Step 2/Step 6 docs refresh for release v1.5.4. The required doc set must be reviewed with named owners/status, source-of-truth checks where applicable, and explicit justification for any out-of-scope item. #1507 (closed) is the standing enforcement policy.

Tag `v1.5.4` already exists; `git diff --name-only v1.5.3..v1.5.4` shows only `README.md` and `docs/AUTODIFF.md` changed in the inventory, so other inventory docs may have drifted or need verification against current source.

## Blast radius

Docs-only change set (no runtime/API behavior changes expected):

| File / area | Risk | Notes |
|---|---|---|
| `README.md` | Low | Public-facing; verify tooling/deps/figures/samples/links against source. |
| `GETTING-STARTED.md` | Low | Verify APIs/examples/commands against current code and docs gates. |
| `ARCHITECTURE.md` | Low | Verify architecture claims against current implementation. |
| `docs/AUTODIFF.md` | Low | Verify API/behavior drift; check examples/citations. |
| `EXAMPLES.md` | Low | Verify examples against current APIs/samples; gate-sensitive. |
| `samples/README.md` | Low | Verify sample inventory/commands/descriptions. |
| `AGENTS.md` | Low | Verify tooling/publishing/dependency/measured-figure claims and process rules. |
| `RELEASING.md` | Low | Verify release/publish/tooling mechanics against `.github/workflows/*`, `*.csproj`, artifacts. |
| `docs/adr/` | Low | Review ADRs for release applicability; add/update only if a decision changed. |

Potential test/gate impact if `docs/` changes: `Nivara.Tests.Docs` suite, snippet gate, citation/reference pins, ADR index. `DocsCoveragePins` may need revision only if document/citation counts change; do not bump pins without arithmetic justification.

## Grounding (G1 - required before implementation)

- [x] Branch confirmed/created: `khurram/1518` (user-confirmed).
- [x] Code-memory MCP grounding: located doc-gate symbols `Nivara.Tests.Docs.DocsCoveragePins` (`tests/Nivara.Tests/Docs/DocsCoveragePins.cs`), `DocsCoverageTests`, `DocSnippetExtractor`, `DocSnippetCompiler` via `sql_query`. Confirmed docs gate/test coupling in AGENTS.
- [x] Source-of-truth checks (initial): `.github/workflows/ci.yml` (ubuntu-latest, .NET 11.0.x, whitespace format verify, build Release, test filter `Category!=Performance&Category!=ReleaseReadiness&Category!=Gpu`, TRX artifact, coverage upload), `.github/workflows/cd.yml` (tag `v*`/workflow_dispatch, .NET 11.0.x, build Release, test filter `Category!=Performance&Category!=Gpu`, pack/push Nivara + Nivara.Extensions + snupkg via NuGet OIDC login), `.github/workflows/release-readiness.yml` (workflow_dispatch, ReleaseReadiness category). `src/Nivara/Nivara.csproj` and `src/Nivara.Extensions/Nivara.Extensions.csproj` target `net11.0`, `PackageVersion` 1.5.5-preview; Extensions deps exact-pinned (Streamix 1.2.3, System.Numerics.Tensors 11.0.0, Apache.Arrow 23.0.0, CsvHelper 34.1.0, Microsoft.ML 6.0.0, Microsoft.ML.Tokenizers 3.0.0, Silk.NET.OpenCL 2.23.0, Parquet.Net 6.1.1-pre.1, Google.Protobuf 3.36.2, Grpc.Tools 2.85.0, Microsoft.Extensions.AI.Abstractions 11.0.1, SourceLink 11.0.100).
- [x] Drift signals identified: `RELEASING.md` contains stale hardcoded "3200+ tests passing" PR-description requirement (line ~234); sample inventory in `samples/README.md` should be checked against actual `samples/` directories; docs should be reviewed for stale measured/version claims. No blocking product decision/red flag requiring human input before docs review; any ambiguous claim will be verified against source or called out explicitly.

## Proposed changes / status

1. **README.md** - **updated** (@khurram-uworx). Fixed stale measured claim (NivaraTorch 55 -> 101 functional tests, verified against `tests/Nivara.Tests/NivaraTorch` `[Test]` count). Reviewed package/dependency/tooling/sample/feature claims; no other source-of-truth drift found.
2. **GETTING-STARTED.md** - **updated** (@khurram-uworx). Fixed stale NivaraTorch test-count claim (55 -> 101). Reviewed API/sample/command drift; gate annotations preserved.
3. **ARCHITECTURE.md** - **updated** (@khurram-uworx). Fixed stale NivaraTorch test-count claim (55 -> 101). Reviewed architecture claims against current layout; no structural drift requiring change.
4. **docs/AUTODIFF.md** - **reviewed** (@khurram-uworx). Scanned for stale measured figures/API drift markers; no release-driven drift identified.
5. **EXAMPLES.md** - **reviewed** (@khurram-uworx). Reviewed for API/sample drift; gate annotations preserved; no stale claims found.
6. **samples/README.md** - **updated** (@khurram-uworx). Removed duplicate NivaraClassifier ONNX section and folded ONNX export/parity note into main NivaraClassifier entry. Verified sample inventory against `samples/` directories.
7. **AGENTS.md** - **reviewed** (@khurram-uworx). Verified tooling/dependency/workflow claims against `.github/workflows/*` and `*.csproj`; no drift found.
8. **RELEASING.md** - **updated** (@khurram-uworx). Replaced stale hardcoded "3200+ tests passing" PR-description requirement with CI-reported required-test-count wording. Verified Step 2/Step 6 and template/workflow references against current files.
9. **docs/adr/** - **reviewed, no ADR change needed** (@khurram-uworx). ADR-001 through ADR-010 reviewed; no v1.5.4 release decision required a new/updated ADR. AGENTS ADR index remains complete.

## Verification

- [x] Ask before running tests / long-running verification (user explicitly requested test/PR flow).
- [x] `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` clean after docs edits.
- [ ] If `docs/` changed, targeted docs gate tests: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` (run as part of requested PR test plan).
- [x] Review `DocsCoveragePins`/doc counts only if required by actual doc/citation changes; no `docs/*.md` added/removed, no citation/reference count changes expected from these edits.
- [x] Manual markdown/table rendering sanity check for touched docs.
- [ ] Branch review G2 before deleting `docs/TODO.md`: (a) branch work as whole, (b) against this plan.

## Planned commits

1. `docs: plan v1.5.4 docs refresh in TODO.md` (done)
2. `docs: record G1 grounding notes in TODO.md` (done)
3. `docs: refresh v1.5.4 release docs for drift` (done)
4. `docs: update TODO with v1.5.4 docs refresh status` (this update)
5. `docs: remove TODO.md — plan executed` (after G2 clears)

## GitHub issues log

- [ ] #1518 — [release v1.5.4] Docs refresh (Step 2/Step 6) (tracking issue; this plan implements it)

Reminder: as each task executes, if you find deferred work or a concern outside this plan, create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its number here. Don't rely on memory or wait until plan end; compaction can lose it.
