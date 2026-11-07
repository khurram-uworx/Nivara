# Plan: Refresh top-level documentation (README, GETTING-STARTED, EXAMPLES, ARCHITECTURE, AGENTS)

## Problem

The top-level documentation still reflects pre-1.5 / pre-1.4-era assumptions in several places, while AGENTS.md and CHANGELOG.md already describe v1.5 outcomes. Specifically:

- `ARCHITECTURE.md` advertises **.NET 10.0** even though every project targets `net11.0`.
- `ARCHITECTURE.md` contains duplicated legacy sections (a second copy of null handling, storage architecture, query engine architecture, file I/O architecture, testing strategy) that drift from the canonical first copies.
- Extension product surface is under-documented: Silk/OpenCL GPU backend (`Nivara.Silk`), BERT/ModernBERT/Laya model layer (`Nivara.AI`), embeddings convention + Microsoft.Extensions.AI glue, ONNX interop.
- Streaming, Arrow zero-copy, and seedable AutoDiff initializers are not surfaced in `GETTING-STARTED.md`.
- `README.md` sample list misses several shipped samples and mis-states extension scope.
- `AGENTS.md` live GPU metric wording is a stale-prone dangling figure.

## Proposed changes

1. `ARCHITECTURE.md`: fix target framework; delete duplicated sections; add Silk/AI/embeddings/Arrow-zero-copy notes in the relevant places.
2. `README.md`: update samples list; broaden optional-extensions description; add doc links (EMBEDDINGS, SILK); add capability bullets for embeddings glue and seedable initializers.
3. `GETTING-STARTED.md`: add Arrow `UseZeroCopy` paragraph; streaming memory-budget/context paragraph; embeddings/ME.AI section; ONNX note; seedable initializers note; Csv-install note.
4. `EXAMPLES.md`: mark Python snippets illustrative where applicable; add embeddings/Silk mentions or links.
5. `AGENTS.md`: trim stale GPU metric phrasing.

## Verification steps

- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` (citation / snippet / ADR gates)
- Build: `dotnet build Nivara.slnx -c Release` (docs are not code, but keep the tree green)

## Planned commits

1. `docs: plan documentation refresh in TODO.md` (this file)
2. `docs(architecture): correct target framework and remove duplicated sections`
3. `docs(architecture): surface Silk/AI/embeddings/Arrow zero-copy extensions`
4. `docs(readme): refresh sample list, extension scope, and doc links`
5. `docs(getting-started): add Arrow zero-copy, streaming budget, embeddings, ONNX, seedable notes`
6. `docs(examples): markup illustrative snippets and add embeddings/Silk links`
7. `docs(agents): trim stale GPU metric phrasing`
8. `docs: remove TODO.md — plan executed`

## Blast radius

- Documentation only. No source changes expected; build unaffected.
- Gate coupling: any newly added fenced `csharp` blocks or `File.cs:NN` citations would change the `DocsCoveragePins` counts and the ADR index gate. This plan deliberately limits additions to fenced-free prose plus reusing existing fenced blocks; only the ARCHITECTURE additions that cite symbols will be reviewed against `ExpectedCitationCount`.

## GitHub issues log

- (none yet — any deferred work found during execution will be logged here and tracked via `gh issue create --repo khurram-uworx/Nivara`)
