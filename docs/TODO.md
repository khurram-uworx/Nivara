# Plan: issue #1178 — documentation drift and gate coverage

## Problem
GitHub issue #1178 (milestone 1.7, medium/review): `AGENTS.md`, `RELEASING.md`, and `README.md` contain claims/links the repository contradicts, `docs/RELEASE15-REVIEW.md` retains unfiled/open items, and the docs reference gate cannot see root-relative markdown links/root-doc targets. The gate’s coverage is over-trusted because it resolves only `docs/...md` references and a limited scanned set.

## Goals / acceptance criteria (from #1178)
- [ ] `RELEASING.md` describes the mechanism the workflow actually uses (OIDC, no `NUGET_API_KEY`), and every file it names exists.
- [ ] No `AGENTS.md` statement about dependency pinning, toolchain, or measured figures is contradicted by the repository; measured figures are date-stamped with a re-measure command.
- [ ] Root-relative markdown links are resolved by a gate, and a deliberately broken link fails it (prove the gate can fail, not just that it exists).
- [ ] README claims match shipped behaviour, with a test or a doc assertion where the claim is mechanically checkable.
- [ ] `docs/RELEASE15-REVIEW.md` items are either filed or explicitly closed as "no action".
- [ ] `RELEASING.md` step 2 lists `AGENTS.md` and `README.md`.

## Proposed changes

### 1) Factual doc corrections (AGENTS.md, RELEASING.md)
- `RELEASING.md`
  - Prerequisites: remove `NUGET_API_KEY` secret requirement; describe OIDC publish path used by `.github/workflows/cd.yml` (`permissions: id-token: write`, `NuGet/login@v1` trusted/federated credential → temp API key). Note repo/account-side NuGet trusted publishing setup is required (no long-lived secret in repo settings).
  - Remove `global.json` mention in SDK prerequisite (file does not exist); keep `.csproj` `TargetFramework` as source.
  - Step 2 review list: explicitly add `AGENTS.md` and `README.md` and require checking claims against `.github/workflows/*` and `*.csproj`.
- `AGENTS.md`
  - Replace “deliberately unpinned / latest unpinned” dependency statements with exact-pinned reality (versions in `src/Nivara/Nivara.csproj`, `src/Nivara.Extensions/Nivara.Extensions.csproj`, `Directory.Build.props` as applicable); state version bumps are reviewed commits; keep “read the csproj” guidance.
  - Fix broken link `AGENTS.md:39`: `[docs/AUTODIFF.md](AUTODIFF.md)` → `[docs/AUTODIFF.md](docs/AUTODIFF.md)`.
  - Measured figures (`AGENTS.md:103`, `AGENTS.md:142`): convert to dated snapshots with explicit re-measure commands and authoritative artifact pointers (trx / CI `test-durations` artifact), so policy/threshold text does not rely on silently drifting numbers.

### 2) Reference gate extension (root-relative markdown links / root docs)
- `tests/Nivara.Tests/Docs/DocReferences.cs` (`DocReferenceExtractor`)
  - Extend markdown reference discovery/resolution to include root-relative `.md` targets (not only `docs/...md`), so links like `[GUIDELINES.md](GUIDELINES.md)` / `[README.md](...)` / `[RELEASING.md](...)` etc. are checked for existence.
  - Update scan scope to include root documentation files required by issue (`README.md`, `AGENTS.md`, `RELEASING.md`, `CHANGELOG.md`, `GUIDELINES.md`, `EXAMPLES.md`, `GETTING-STARTED.md`, `ARCHITECTURE.md`) and ensure `ScanFiles`/discovery behavior matches widened intent without regressing existing `docs/` behavior.
  - Keep exclusions/tolerations intentional; resolve relative targets correctly (repo-root-relative and existing `../docs/...` normalization behavior).
- `tests/Nivara.Tests/Docs/DocReferenceTests.cs`
  - Add/adjust assertions proving a deliberately broken root-relative markdown link fails the gate (negative control) while resolvable root-relative links pass.
- `tests/Nivara.Tests/Docs/DocsCoveragePins.cs` (and any dependent docs coverage tests/fixtures)
  - Recalculate pinned counts/arithmetic after scope/reference-count changes in the same commit as the gate change (drop treated as reduced coverage needing reason only if real; here scope intentionally widened).

### 3) README reconciliation
- Remove/editing artifact: replace bare URL bullet under “Why Nivara” (`README.md:11`) with descriptive text/link.
- Reconcile headline claims with shipped behavior:
  - “Explicit null handling — no NaN-based semantics or hidden behavior”
  - “Schema-aware query planning — errors surface early, not at runtime”
  - Action: verify current behavior (including cases analogous to issue’s H1/H2) against implementation/tests; either correct behavior (if clearly intended and low-risk) or soften claims to reflect current state + add known-limitations link/note. Prefer doc/test-backed accuracy; avoid asserting unverified runtime fixes in docs-only change. Add a mechanically checkable assertion/test where feasible (or document limitation explicitly if not).

### 4) Release retrospective cleanup
- `docs/RELEASE15-REVIEW.md`: convert open “Fix options/Suggestion” items to either filed issue references or explicit “closed as no action / lesson learned” status so no unfiled actionable suggestions remain. Keep historical record clear.

## Blast radius
- **Docs/content:** `AGENTS.md`, `RELEASING.md`, `README.md`, `docs/RELEASE15-REVIEW.md` (and any root docs found broken once gate widens).
- **Docs gate/tests:** `tests/Nivara.Tests/Docs/DocReferences.cs`, `DocReferenceTests.cs`, `DocsCoveragePins.cs`, possibly `DocsCoverageTests.cs` / `DocCitations.cs` / fixtures in `tests/Nivara.Tests/Docs/Fixtures/` depending on scope/count changes.
- **Downstream impact:** CI/local docs integrity checks (`Nivara.Tests.Docs` namespace), release checklist consumers, contributor/agent guidance. No runtime library/API behavior changes intended except if a narrowly-scoped, test-backed README-claim correction is required (decision point).
- **Risk areas:** widened scan/regex may surface existing dangling root-relative links and will move pinned counts; negative-control test must prove failure mode; relative-path resolution must not create false positives.

## Verification
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Targeted docs gate: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` (ask before running long/full-suite commands per repo rules)
- If README behavior claims are reconciled via code/test: run targeted tests for affected area(s) (to be identified during implementation) before commit.
- Manual spot-check: `RELEASING.md` matches `cd.yml` (OIDC) and names existing files only; `AGENTS.md` link/claims accurate; widened gate catches an intentionally broken root-relative link.

## Planned commit list
1. `docs: plan issue 1178 in TODO.md`
2. `docs: correct AGENTS and RELEASING drift for issue 1178` (facts, links, measured-claim handling, release checklist updates)
3. `test/docs: extend doc reference gate to root-relative markdown links` (gate + negative control + pin/count updates)
4. `docs: reconcile README claims and clean release retrospective for issue 1178` (and any targeted test/doc assertion if needed)
5. Fix commits only if review/verification finds gaps
6. `docs: remove TODO.md — plan executed` (after G2 clears)

## Grounding (G1) notes / questions for human
- Ground implementation details against official docs where relevant via microsoft-learn MCP (e.g. GitHub Actions/NuGet OIDC publishing concepts if wording needs precision) and codebase via code-memory MCP (done for docs gate symbols/relationships).
- **Decision/red flag:** README reconciliation item 7 (H1/H2) is underspecified in this session without reproducing current behavior. I plan to verify against current code/tests and prefer softening claims + known-limitations link if behavior does not match, unless a trivial/test-backed correction is clearly intended. If you want a code fix to make README claims literally true now, please confirm scope/expected behavior before I implement that part.
- **Red flag:** widening the gate may expose existing broken root-relative links outside the files named in the issue; fixing them is necessary for a green gate and should be included, but I will report any unexpected blast radius before proceeding past trivial fixes.

## GitHub issues log

- [ ] (none yet) — create immediately during execution if deferred/follow-up work is discovered outside this plan (`gh issue create --repo khurram-uworx/Nivara`) and record issue number here.

## Iterative-work reminders
- Commit after each logical change unit; never push.
- Ask before running `dotnet test` or other long-running verification commands.
- Create GitHub issues immediately for deferred work found during execution and log them above.
- G2: review branch as a whole and against this TODO before deleting it; only delete when both clear.
