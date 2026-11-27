# Docs refresh task template

Use this template when creating a per-release docs-refresh GitHub issue (RELEASING Step 2/Step 6). It makes the full doc review set, named owners, source-of-truth checks, and scope justifications explicit and auditable. See #1507 for the standing enforcement policy.

## How to use

1. Replace placeholders: `<X.Y.Z>`, `<YYYY-MM-DD>`, `<owner>`, `<issue-url>`.
2. Confirm the doc set below matches `RELEASING.md` Step 2/Step 6. If the release changed the required set, update the template in the same PR and note why.
3. Create the issue:
   ```text
   gh issue create --repo khurram-uworx/Nivara --title "[release vX.Y.Z] Docs refresh (Step 2/Step 6)" --body-file <temp-body>.md
   ```
4. Link the issue from the release PR. Close it only when every listed doc is reviewed/updated or explicitly scoped out with justification.

## Issue body template

```md
## Goal

Track the full Step 2/Step 6 docs refresh for release v<X.Y.Z>. This is a per-release follow-up issue; #1507 is the standing enforcement policy.

## Rules

- Full Step 2/Step 6 doc set required; no silent scope drop.
- Named owner per file. Incomplete = release-prep incomplete unless an explicit maintainer waiver/justification is recorded in this issue and the release PR.
- Verify tooling/publishing/dependency/measured-figure claims against source of truth where applicable (`.github/workflows/*`, `*.csproj`, current test/build artifacts).
- Release retrospective/review belongs in `release/v<X.Y.Z>/RELEASE-REVIEW.md`, not `docs/`.

## Doc inventory

| Doc | Owner | Status (`pending/reviewed/updated/out-of-scope`) | Source-of-truth checked | Notes / justification |
|---|---|---|---|---|
| `README.md` | <owner> | pending | no | Verify tooling/dependency/measured-figure claims against current source of truth. |
| `GETTING-STARTED.md` | <owner> | pending | no | Review for drift against current APIs/samples. |
| `ARCHITECTURE.md` | <owner> | pending | no | Review for drift against current architecture. |
| `docs/AUTODIFF.md` | <owner> | pending | no | Review for API/behavior drift; check examples/citations as applicable. |
| `EXAMPLES.md` | <owner> | pending | no | Review for drift against current samples/APIs. |
| `samples/README.md` | <owner> | pending | no | Review for sample inventory/command drift. |
| `AGENTS.md` | <owner> | pending | no | Verify tooling/publishing/dependency/measured-figure claims against current source of truth. |
| `RELEASING.md` | <owner> | pending | no | Verify release/publish/tooling mechanics against `.github/workflows/*`, `*.csproj`, test/build artifacts. |
| `docs/adr/` (as applicable) | <owner> | pending | no | List ADRs reviewed/added/none; record if no ADR change matched the release. |

## Verification

- [ ] Every listed doc has owner + status.
- [ ] Out-of-scope rows have explicit justification.
- [ ] Tooling/publishing/dependency/measured-figure claims checked against source of truth where applicable.
- [ ] `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` clean if docs touched.
- [ ] If `docs/` changed, doc coverage/gate tests considered as needed (`dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` when applicable).
- [ ] PR links this issue and includes doc-refresh confirmation listing each file with named owner/status.

## PR test plan

- [ ] `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` (if doc/code formatting scope requires)
- [ ] Targeted doc/gate tests as needed
- [ ] Manual markdown/table rendering sanity check
```

## Notes

- Refs #1507. Per-release issues created from this template are evidence of compliance; they do not close the standing enforcement issue unless a maintainer explicitly decides #1507 is satisfied.
- If a doc is added to or removed from the RELEASING Step 2/Step 6 set, update this template, `RELEASING.md`, and the release PR checklist in the same change. Adding/removing a `docs/*.md` file also affects `DocsCoveragePins` (see `AGENTS.md`).
