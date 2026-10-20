# Plan: Document whitespace-formatting workflow (issue #622)

## Problem

#593 added a whitespace-formatting gate to CI:

```
dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore
```

The scope of that gate — whitespace only, not the full `dotnet format` — is deliberate but undocumented, so the reasoning lives only in a commit message. Two things should be written down for contributors and agents:

1. What to run before committing: `dotnet format whitespace` is the local equivalent of the gate.
2. Why the gate excludes analyzers: the default `dotnet format` (and `dotnet format analyzers`) reports five `CS8618` findings in test fixtures. These are false positives: every field is assigned in `[SetUp]`, and NUnit.Analyzers' `DiagnosticSuppressor` (`NUnit3002`) suppresses them at build time. `dotnet format` does not run diagnostic suppressors, so with `TreatWarningsAsErrors=true` they show as errors. A contributor running `dotnet format` (or `--verify-no-changes`) locally sees red that CI never sees, with no explanation.

Minor second point: `.gitattributes` is now `* text=auto eol=lf`, which applies to every text file. There are no `.bat`/`.cmd` files today, but if one is added it would be checked out with LF.

## Proposed changes

- `CONTRIBUTING.md` (Build Configuration section): add a `### Formatting` subsection with the local command, the CI gate, the analyzer false-positive explanation (listing the five fixtures), and a line-endings note.
- `AGENTS.md` (Code Style section): add two bullets mirroring the above so agents see the rule where coding standards live.

No code, tests, or behavior changes. No fenced ```csharp blocks added (snippet gate unaffected). No `File.cs:NN` citations added (citation/reference pinned counts unaffected).

## Blast radius

- Files changed: `CONTRIBUTING.md`, `AGENTS.md`, `docs/TODO.md` (plan), then removed.
- Downstream: documentation tests (`DocCitationTests`, `DocReferenceTests`, `AdrIndexTests`) are unaffected because no citations/ADRs/files counts change.
- No source symbols affected.

## Verification steps

- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` passes after edits.
- Targeted docs tests: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` (optional, run only if asked).

## Planned commits

1. `docs: plan whitespace-formatting workflow documentation in TODO.md`
2. `docs: document whitespace-formatting workflow in CONTRIBUTING.md`
3. `docs: document whitespace-formatting workflow in AGENTS.md`
4. `docs: remove TODO.md — plan executed`

## GitHub issues log

- (none created yet)