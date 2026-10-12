# Plan: enforce `.editorconfig` whitespace in CI (issue #593)

## Problem

CI (`.github/workflows/ci.yml`) has no formatting step, so `.editorconfig` rules —
in particular `insert_final_newline` and `end_of_line` — have no gate. #577 stripped
the final newline from eight files and it merged unnoticed; #592 fixed those eight by
hand.

The issue reports **365 diagnostics across 107 files** (`ENDOFLINE` 255, `FINALNEWLINE`
105, `CS8618` 5). Those numbers **do not reproduce on the current tree** and the
`ENDOFLINE` volume is environment-dependent, so the issue's framing has to be corrected
before acting on it.

## Verified ground truth (this tree, `main` = 70e41652)

```
dotnet format Nivara.slnx --verify-no-changes
  -> 108 diagnostics / 105 files = FINALNEWLINE 103 + CS8618 5, ENDOFLINE 0

dotnet format Nivara.slnx whitespace --verify-no-changes
  -> 103 FINALNEWLINE, exit 2

dotnet format Nivara.slnx style --verify-no-changes
  -> 0 diagnostics, exit 0
```

All 103 `FINALNEWLINE` are `.cs`; no `TRAILINGWHITESPACE`/`INDENT` were reported.

### Why `ENDOFLINE` is 0 here but will fire on CI

- `.gitattributes` is only `* text=auto` — no `eol=` — so **all text blobs are LF**
  (`git ls-files --eol` -> `i/lf` for every text file).
- `.editorconfig` says `end_of_line = crlf`.
- Checkout line endings are therefore decided by `core.autocrlf`. Locally it is `true`,
  so the worktree is CRLF (813 `i/lf w/crlf`) and `dotnet format` is satisfied.
- GitHub's `ubuntu-latest` has `core.autocrlf` unset (`false`), so git checks the LF
  blobs out **as LF** -> `dotnet format` flags `ENDOFLINE` on essentially every text
  file. The policy mismatch is the real blocker, and on CI it is larger than 255.

### Why the 5 `CS8618` must not be "fixed"

`dotnet build` is clean (0 warnings, Debug and Release). The diagnostic build log shows
the warnings are suppressed at build time:

```
info SP0001: Diagnostic 'CS8618 ...' was programmatically suppressed by a
DiagnosticSuppressor with suppression ID 'NUnit3002' and justification
'Field/Property is initialized in SetUp or OneTimeSetUp method'
```

`dotnet format` does not run diagnostic suppressors, so it re-reports them (and
`TreatWarningsAsErrors=true` shows them as errors). The build is right; `format` is
wrong. Adding `= null!;` would paper over a tool limitation and mutate test fixtures.
The gate is scoped to `whitespace` so these are out of scope by construction.

### Bonus hygiene finding

`tests/Nivara.Tests/Expressions/ColumnExpressionsTests.cs` (plural) is a tracked
**0-byte** file, a leftover from `23a9957d`, next to the real `ColumnExpressionTests.cs`.

## Decisions (confirmed with the human)

1. **Line endings: LF everywhere.** Blobs are already LF, so this needs no blob rewrite.
2. **Gate scope: `dotnet format whitespace` only** (analyzers excluded).
3. **Delete the 0-byte stray file.**
4. **Gate lives in `ci.yml` only.**

## Proposed changes

### 1. LF policy

`.editorconfig`:

```
-end_of_line = crlf
+end_of_line = lf
```

`.gitattributes`:

```
-* text=auto
+* text=auto eol=lf
```

`eol=lf` forces LF on Windows checkouts too (it overrides `core.autocrlf`), so local
and Linux CI agree. `*.bin binary` is untouched.

### 2. Normalize the 103 missing final newlines

One whitespace-only commit produced by `dotnet format Nivara.slnx whitespace`
(+1 line per file; every hunk is `\ No newline at end of file` -> `+`).

### 3. Delete the stray file

`git rm tests/Nivara.Tests/Expressions/ColumnExpressionsTests.cs`

### 4. CI gate (`ci.yml`, after `Restore dependencies`, before `Build`)

```yaml
    - name: Verify formatting
      run: dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore
```

`dotnet format` exits `2` when changes are needed, so the step fails correctly.

## Verification steps

1. Re-checkout so the worktree picks up `eol=lf` (`git add --renormalize . && git reset --hard`),
   then confirm `git ls-files --eol` reports `w/lf` for text files. This makes the
   Windows worktree byte-identical to the Linux CI checkout, so a local pass is real
   evidence for CI.
2. `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` -> exit 0.
3. `dotnet build Nivara.slnx -c Release` -> 0 warnings, 0 errors (unchanged).
4. Optionally `dotnet format Nivara.slnx style --verify-no-changes` -> still 0.

No runtime/API change, so no targeted tests are required. The full suite is not run
without explicit human confirmation.

## Blast radius

- `.editorconfig`, `.gitattributes` — repo-wide policy; affects checkout line endings
  for every text file but **not** committed blob content (already LF).
- 103 `.cs` files — final-newline only; no semantic change.
- `tests/Nivara.Tests/Expressions/ColumnExpressionsTests.cs` — deleted (empty, module
  contains no compiled symbol).
- `.github/workflows/ci.yml` — one added step.

No `src/Nivara` or `src/Nivara.Extensions` behaviour is touched. Nothing downstream
depends on line endings at compile/runtime.

## Planned commits

1. `chore: switch line-ending policy to LF` (`.editorconfig`, `.gitattributes`)
2. `style: restore the final newline on the 103 remaining files`
3. `chore: delete the empty ColumnExpressionsTests.cs leftover`
4. `ci: gate whitespace formatting with dotnet format`

Plus `docs: plan enforce .editorconfig whitespace in CI in TODO.md` and the final
`docs: remove TODO.md — plan executed`.

## GitHub issues log

- [ ] #622 — document the whitespace-formatting workflow and why the gate excludes analyzers (created during #593)
