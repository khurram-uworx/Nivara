# Exclude `release/` from the widened doc-reference scan (#1508)

## Problem

Issue [#1508](https://github.com/khurram-uworx/Nivara/issues/1508) tracks mechanical
`DocsCoveragePins` / doc-coverage churn causing CI drift. Release retros/reviews already moved to
`release/vX.Y.Z/RELEASE-REVIEW.md` (not `docs/`) to avoid narrow-scope pin churn, but the widened
doc-reference scan still picks up `README.md`/`AGENTS.md` anywhere under the repo root. That
currently includes `release/README.md`, so edits there can move
`ExpectedDocumentReferenceCount` / `ExpectedCodeCitationCount`.

## Grounding (G1)

- Codebase (code-memory/reads): `DocReferenceExtractor.ScanFiles()` collects `README.md`/`AGENTS.md`
  recursively from repo root (`tests/Nivara.Tests/Docs/DocReferences.cs::ScanFiles()`), and
  `Collect()` already enforces `ExcludedPathPrefixes`. `DocSnippetExtractor.MarkdownFiles()` scans
  only repo root `*.md` + `docs/**/*.md`, so `release/` is already outside the narrow
  snippet/citation/document-count scope (`DocSnippet.cs::MarkdownFiles()`).
- `DocsCoveragePins.cs` already records release/README.md impact (+1 #1506/#1507 comments), confirming
  widened-scope participation.
- No microsoft-learn/Tensor/Vector API grounding required: this change is doc-gate scope/test policy,
  not a .NET tensor/vector/runtime API behavior change.

**Decision/red flag (accepted by human in conversation):** excluding `release/` is an intentional
coverage reduction. Broken markdown refs / `File.cs::Symbol` citations inside `release/` will no
longer be caught by the widened doc gate. Accepted because `release/` is process/artefact tooling,
not reader-facing product docs; the tradeoff is documented in code comments and this plan.

## Blast radius

| Area | Impact |
|---|---|
| `tests/Nivara.Tests/Docs/DocReferences.cs` | Add `"release/"` to `ExcludedPathPrefixes`; update comments to document exclusion and tradeoff. |
| `tests/Nivara.Tests/Docs/DocsCoveragePins.cs` | Update widened-scope pins to the actual post-change values; replace stale release/README.md delta comments with exclusion rationale. |
| `tests/Nivara.Tests/Docs/*` tests | Docs suite must pass; narrow-scope pins should be unchanged. |
| `release/**` content | No longer scanned by widened doc-reference gate; no runtime/product code impact. |

No production source changes. No public API changes. No docs/ product doc changes expected.

## Proposed changes

1. **Exclude release/** — add `"release/"` to `DocReferenceExtractor.ExcludedPathPrefixes` and update
   the XML comment to state `release/` is process/artefact tooling excluded to avoid mechanical pin
   churn, with the intentional coverage-reduction tradeoff.
2. **Revise pins** — run targeted Docs tests, capture the real delta, and update
   `DocsCoveragePins` widened-scope expected values/comments accordingly. Do not guess numbers.
3. **Branch review (G2)** — review branch as a whole and against this plan before deleting TODO.

## Verification

- Ask before running tests.
- Targeted: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`
- Format gate before PR: `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Confirm no unintended pin drops without rationale; narrow-scope pins unchanged.

## Planned commits

1. `docs: plan #1508 release scan exclusion`
2. `test(docs): exclude release from widened doc reference scan`
3. `docs: remove TODO.md — plan executed` (only after G2 clears)

## GitHub issues log

- [ ] #1508 — Reduce mechanical docs coverage/pin churn causing CI drift (this work partially addresses the release/README.md churn source; broader scope-split/policy remains open unless resolved)

## Reminder

As each task executes, if you find deferred work or a concern, create a GitHub issue immediately
(`gh issue create --repo khurram-uworx/Nivara`) and record its number in the log — don't rely on
memory or wait until the end of the plan, as compaction during execution can lose important items.
