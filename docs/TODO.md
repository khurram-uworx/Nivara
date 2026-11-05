# Plan: keep `--embed` as the primitive baseline; document `--rag` as the #867 convention

## Problem

Issue #919 proposed migrating `--embed` to the #867 embedding convention. The code already
contains the convention example in `--rag` (#912): one `NivaraFrame` (metadata + `Embedding`),
`FilterByMask` ACL, `RowCosineSimilarity` + `TopKDescending`, `--roundtrip`,
`--silk` parity, and a Parquet cache in the dimension-frame form. `--embed` is the
deliberately primitive "before" state (parallel arrays + `TensorPrimitives` loop).
Migrating it would erase the contrast that makes the improvement visible.

## Proposed changes

- `docs/EMBEDDINGS.md`: add a "Worked contrast" section (primitive vs convention) anchored
  to `samples/NivaraChat/Modes/EmbeddingMode.cs::Run()` and `RagMode.cs::Run(ModeContext)`,
  plus a "before" code fence; add an honest "what the convention does not remove" paragraph;
  correct the false persistence bullet (Arrow/Parquet reject `float[]`; persist the
  dimension frame via `RagIndexCache`).
- `samples/NivaraChat/README.md`: `--embed` as deliberate primitive baseline; `--rag` as
  the convention example (default frame, `--store inmemory` legacy); fix the stale
  InMemoryVectorStore description; API table + tree annotations.
- `samples/NivaraChat/Modes/EmbeddingMode.cs`: extend XML summary (no behaviour change).
- Pin updates: `DocsCoveragePins.cs` (citations +2 doc refs; +5 symbol citations), and
  `DocumentationSnippetTests.cs` (gated 110->111, total 256->257).
- Issue #919: repurpose to track this docs work (or close as not planned).

## Verification

- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`
- `dotnet build samples/NivaraChat/NivaraChat.csproj -c Release` (code touch only)

## Planned commits

1. docs: plan (this file)
2. docs(EMBEDDINGS): add worked contrast + correct persistence bullet
3. docs(NivaraChat): mark --embed primitive, --rag convention
4. samples(NivaraChat): extend EmbeddingMode XML summary
5. tests(Docs): update citation/snippet pins for the new fence and links
6. docs: remove TODO.md

## GitHub issues log

- [ ] #919 — repurpose to docs: keep --embed as primitive baseline (created 2026-11-05)
