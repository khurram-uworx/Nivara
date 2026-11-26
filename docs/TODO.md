# Plan: Per-release docs-refresh template and RELEASING wiring (#1507)

## Problem

Issue #1507 (`khurram-uworx/Nivara#1507`, open) asks for a consistent full docs refresh (RELEASING Step 2/Step 6) every release, with named owners recorded in the PR. The benchmark refresh flow already has a canonical per-release template (`release/TASK-TEMPLATE-BENCHMARKS.md`, wired by `e084c2af`, "1506-like task"), but docs refresh does not:

- `RELEASING.md` Step 2 lists the doc set and source-of-truth verification, but has no per-release tracking-issue/template hook and no named-owner capture at review start.
- `RELEASING.md` Step 6 creates a docs-refresh issue inline, but does not reference a canonical template, so scope/checklist/verification is ad hoc.
- `release/README.md` lists only `TASK-TEMPLATE-BENCHMARKS.md`.
- There is no `release/TASK-TEMPLATE-DOCS.md`.

Goal: add `release/TASK-TEMPLATE-DOCS.md` and wire it into `RELEASING.md` and `release/README.md`, making per-release docs-refresh follow-up issues deterministic and auditable, similar to the benchmark template pattern.

## Scope / non-goals

- In scope: new docs template, RELEASING wiring, release README wiring, plan/review hygiene.
- Out of scope: closing #1507 automatically (it is a standing policy/enforcement issue; this work provides the enabling template/process — PR should `Refs #1507` unless maintainer decides otherwise), changing docs coverage pins unless a `docs/` file is added/removed, changing benchmark flow.

## Proposed changes

### 1. `release/TASK-TEMPLATE-DOCS.md` (new)

Mirror `release/TASK-TEMPLATE-BENCHMARKS.md` structure/strictness, adapted for docs review (not benchmark artefacts).

Planned sections:
- **How to use** — placeholders `<X.Y.Z>`, `<YYYY-MM-DD>`, `<owner>`, `<issue-url>`; create command:
  ```text
  gh issue create --repo khurram-uworx/Nivara --title "[release vX.Y.Z] Docs refresh (Step 2/Step 6)" --body-file <temp-body>.md
  ```
  Link from release PR; close only when full set reviewed or explicitly scoped out with justification.
- **Goal / Rules** — full Step 2/Step 6 set required; named owner per file; incomplete = release-prep incomplete unless explicit maintainer waiver/justification recorded; no silent scope drop; verify tooling/publishing/dependency/measured-figure claims against source of truth (`.github/workflows/*`, `*.csproj`, current test/build artifacts) where applicable.
- **Doc inventory table** (source of truth = RELEASING Step 2/Step 6):

  | Doc | Owner | Status (`pending/reviewed/updated/out-of-scope`) | Source-of-truth checked | Notes / justification |
  |---|---|---|---|---|
  | `README.md` | | pending | | |
  | `GETTING-STARTED.md` | | pending | | |
  | `ARCHITECTURE.md` | | pending | | |
  | `docs/AUTODIFF.md` | | pending | | |
  | `EXAMPLES.md` | | pending | | |
  | `samples/README.md` | | pending | | |
  | `AGENTS.md` | | pending | | |
  | `RELEASING.md` | | pending | | |
  | `docs/adr/` (as applicable) | | pending | | List ADRs reviewed/added/none |

- **Verification** — every listed doc has owner + status; out-of-scope rows justified; source-of-truth checks recorded; `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` clean if docs touched; if `docs/` changed, consider doc coverage/gate tests; PR links issue and includes doc-refresh confirmation.
- **Notes** — release retrospective/review belongs in `release/vX.Y.Z/RELEASE-REVIEW.md`, not `docs/`; refs #1507 as standing enforcement issue (per-release issue is evidence of compliance, not closure).

### 2. `RELEASING.md`

- **Step 2**: add issue-creation hook mirroring Step 5 style, referencing `release/TASK-TEMPLATE-DOCS.md`. Create the issue at the start of docs review (earlier tracking), title `[release vX.Y.Z] Docs refresh (Step 2/Step 6)`. Keep existing source-of-truth verification sentence.
- **Step 6**: change inline issue-creation sentence to reference template and state issue created in Step 2 (or created/linked before PR). Keep named-owner/link requirement.
- **PR description**: tighten wording to name both templates: benchmark refresh from `release/TASK-TEMPLATE-BENCHMARKS.md` and docs refresh from `release/TASK-TEMPLATE-DOCS.md`.

Sketch (Step 2 addition, after existing review list/verification paragraph):

```md
**Always create a GitHub issue** to track docs refresh for the release (use `gh issue create --repo khurram-uworx/Nivara`). Use `release/TASK-TEMPLATE-DOCS.md` as the canonical issue body/checklist template so the full Step 2/Step 6 review set, named owners, source-of-truth checks, and scope justifications are consistent. Title it clearly, e.g. `[release vX.Y.Z] Docs refresh (Step 2/Step 6)`, and link it from the PR. If the full review is completed before opening the PR, still create the issue and close it with a note (or reference it in the PR) so the release process always produces this tracking issue. See #1507 for the standing enforcement policy.
```

Sketch (Step 6 replacement):

```md
**Always create/link the GitHub issue** to track docs refresh for the release (created in Step 2 from `release/TASK-TEMPLATE-DOCS.md`, use `gh issue create --repo khurram-uworx/Nivara` if not already created). Title it clearly, e.g. `[release vX.Y.Z] Docs refresh (Step 2/Step 6)`, listing the required review set and named owners in the issue body; link it from the PR.
```

### 3. `release/README.md`

Update item 3 to include docs template:

```md
3. **Task templates** — `TASK-TEMPLATE-BENCHMARKS.md` for 1506-like benchmark README refresh issues, and `TASK-TEMPLATE-DOCS.md` for per-release docs refresh issues (Step 2/Step 6, see #1507).
```

## Blast radius

| File | Change | Risk | Downstream impact |
|---|---|---:|---|
| `release/TASK-TEMPLATE-DOCS.md` | New file | Low | Used by humans/agents during release prep; referenced by RELEASING/release README. No code/runtime impact. |
| `RELEASING.md` | Add template refs + Step 2 issue hook + Step 6 clarification | Low | Agentic release instructions change; PR/release checklist expectations become more explicit. No build/test/runtime impact. |
| `release/README.md` | List docs template | Low | Release folder documentation only. |
| `docs/TODO.md` | Plan file, deleted before PR merge-ready conclusion per G2 | None after deletion | N/A |

No `src/`, `tests/`, `docs/` (published docs), `.csproj`, or workflow files changed. No DocsCoveragePins impact expected because no `docs/*.md` added/removed and no citations changed. `release/` is outside doc coverage pin scope.

## Grounding (G1)

- Inspected #1507 and #1506 via `gh issue view --repo khurram-uworx/Nivara`.
- Reviewed `RELEASING.md`, `release/README.md`, `release/TASK-TEMPLATE-BENCHMARKS.md`.
- Reviewed pattern commit `e084c2af docs: add benchmark refresh task template and wire RELEASING`.
- Used code-memory MCP `sql_query` on `ChunkRecord`/`SymbolRecord` to confirm relevant files are indexed as text chunks; no code symbols involved.
- Confirmed current Step 6 already requires docs-refresh issue and named owners in PR; gap is missing canonical template + Step 2 creation hook.

Decisions/red flags: none requiring human input identified. The naming `release/TASK-TEMPLATE-DOCS.md` matches user suggestion and benchmark template naming. Issue #1507 remains open as standing policy; PR should use `Refs #1507`.

## Verification

- `git diff --check`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` (touched markdown files; CI-enforced)
- Review branch diff against plan (G2a/G2b)
- Confirm DocsCoveragePins revision not required: no `docs/*.md` added/removed, no citations changed, no ADR added; `release/` files are outside doc-coverage scope.
- If any `docs/` file were touched, run `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` — not expected.
- Manual markdown sanity check: tables render, commands/paths correct, placeholders consistent.

## Planned commits

1. `docs: plan docs refresh template and RELEASING wiring (#1507)` — this file.
2. `docs: add docs refresh task template and wire RELEASING` — add `release/TASK-TEMPLATE-DOCS.md`, update `RELEASING.md`, update `release/README.md`. Refs #1507.
3. `docs: remove TODO.md — plan executed` — only after G2 clears.

## GitHub issues log

Reminder: as each task executes, if you find deferred work or a concern outside this plan, create a tracked issue immediately via `gh issue create --repo khurram-uworx/Nivara` and record its number here. Do not rely on memory or wait until plan end.

- [ ] #1507 — standing policy issue; this plan implements enabling template/process. PR should `Refs #1507` (do not auto-close unless maintainer confirms).
