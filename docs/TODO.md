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
- [ ] Ground plan via microsoft-learn MCP (relevant .NET/docs/tooling claims) and code-memory MCP (symbols/relationships for docs-gate enforcement: `DocsCoveragePins`, `DocSnippetExtractor`, doc tests).
- [ ] Verify source-of-truth claims called out by #1518: `.github/workflows/*`, `*.csproj`, current test/build artifacts where applicable.
- [ ] Identify any decision/red flag/assumption needing human input; stop and ask before implementation if found.

## Proposed changes

1. **README.md** - Review for drift. Verify package/dependency/tooling claims, sample inventory, links, measured/perf claims, and feature descriptions against current source. Keep claims source-backed.
2. **GETTING-STARTED.md** - Review for API/sample drift. Ensure examples compile under snippet gate conventions where gated; verify commands and package names.
3. **ARCHITECTURE.md** - Review for architecture drift against current code layout and behavior; update only where stale/inaccurate.
4. **docs/AUTODIFF.md** - Review for API/behavior drift; verify examples/citations and type/behavior claims.
5. **EXAMPLES.md** - Review for drift against current APIs/samples; preserve gate annotations and compilability.
6. **samples/README.md** - Review sample inventory/command drift against `samples/` directory contents.
7. **AGENTS.md** - Verify tooling/publishing/dependency/measured-figure claims against source of truth; keep process guidance accurate.
8. **RELEASING.md** - Verify release/publish/tooling mechanics against `.github/workflows/*`, `*.csproj`, and current build/test artifacts; ensure Step 2/Step 6 and template references remain accurate.
9. **docs/adr/** - Review ADRs applicable to v1.5.4 changes; record reviewed/added/none. Add ADR only if a real decision changed.

For each file, record owner/status/source-of-truth-checked in the PR description (and reflect in issue/PR workflow as requested).

## Verification

- [ ] Ask before running tests / long-running verification.
- [ ] `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` (after docs edits).
- [ ] If `docs/` changed, targeted docs gate tests: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` (ask before running).
- [ ] Review `DocsCoveragePins`/doc counts only if required by actual doc/citation changes; update with arithmetic justification.
- [ ] Manual markdown/table rendering sanity check for touched docs.
- [ ] Branch review G2 before deleting `docs/TODO.md`: (a) branch work as whole, (b) against this plan.

## Planned commits

1. `docs: plan v1.5.4 docs refresh in TODO.md` (this file)
2. One or more logical docs-refresh commits by area/file group (only files relevant to each step), with verification before each commit.
3. `docs: remove TODO.md — plan executed` (only after G2 clears)

## GitHub issues log

- [ ] #1518 — [release v1.5.4] Docs refresh (Step 2/Step 6) (tracking issue; this plan implements it)

Reminder: as each task executes, if you find deferred work or a concern outside this plan, create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its number here. Don't rely on memory or wait until plan end; compaction can lose it.
