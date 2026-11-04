# Plan — #867 AISTACK Phase 2: embeddings as a flat `[rows, dim]` frame convention

Branch: `khurram/867`
Issue: [#867](https://github.com/khurram-uworx/Nivara/issues/867)
Follow-on: [#912](https://github.com/khurram-uworx/Nivara/issues/912) (RAG sample — consumes this API)

## Problem

Embeddings have no documented home in Nivara. The AI-stack re-scope
(`docs/plan/AISTACK-ROADMAP.md` Phase 2) settled the direction: **a flat
`[rows, dim]` frame convention, not a new column kind**. A column of
`NivaraColumn<float[]>` is the storage/interop form; a frame of `E0..E{n-1}`
`float` columns is the form the existing row-scoring kernels understand. Nothing
in the product documents that convention, and there is no glue between it and
`Microsoft.Extensions.AI`, so every sample reinvents parallel arrays and manual
scoring loops (`samples/NivaraChat/Modes/EmbeddingMode.cs`, the chess samples).

The row-scoring kernels already exist and are correct:
`NivaraFrame.RowCosineSimilarity` / `RowDot` (mask-first null semantics,
`src/Nivara/NivaraFrame.cs`) and `NivaraSeries.TopKDescending`. #867 is
additive glue plus a compiled contract. Core is not touched.

## Non-goals (explicit, to resist scope drift)

- No new `IColumn` implementation, no `NivaraColumn<Embedding<float>>`.
- No change to `NivaraFrame`, `NivaraColumn`, `NivaraSeries`, `TensorsHelper`.
- No ARROW Phase B / zero-copy embedding columns.
- No double/Half embeddings (float-only, matching the `IRootFunctions<float>` kernels).
- No ANN / vector index / persistence format.
- No BLAS, MKL or custom tensor math.

## Design decisions (agreed with the human before implementation)

1. **Query parameter is `NivaraSeries<float>`, plus a `float[]?` convenience
   overload.** The issue comment proposed `ReadOnlySpan<float>`, which cannot
   express the comment's own "null query → all nulls" criterion. The series
   overload is pure delegation (null mask and labels pass straight through).
2. **Null handling is a caller-supplied policy, never a library guess.**
   Two named delegates:
   - `NullEmbeddingReplacement(int rowIndex, int dimension) -> float[]` for a null
     row crossing into MEAI, which has no null vector.
   - `PartialEmbeddingRepair(int rowIndex, float[] values, int[] nullDimensions)
     -> float[]?` for a matrix row whose dimension cells are only partly null
     (`null` return = collapse the row to a null embedding).
   The parameter is optional in the signature but **required whenever it is
   reachable**: with nulls present and no policy supplied the method throws,
   naming the row and the parameter. No zero-fill, no skip, no silent collapse.
   Row *selection* stays a frame decision (`FilterByMask` before conversion), so
   positional alignment is never lost.
3. **Docs are a new gated `docs/EMBEDDINGS.md` plus a prose pointer from
   `docs/TENSORS.md`.** `TENSORS.md` is in `UngatedDocuments`, so its fences are
   never compiled; the convention people copy from must be gated.
4. **`GenerateFrameAsync` ships now** — #912's ingestion is generate → frame in
   one step.
5. **Scoring goes through the matrix conversion** (one tested path, reuses the
   existing kernel, matches the issue's "scored with the existing kernels").
   Cost is 2x embedding bytes live at once; #912 measures and only then does a
   direct `float[]`-column scorer become justified.
6. `ToNivaraFrame` keeps the comment's `(idColumnName, ids)` pair; #912 composes
   further metadata with `WithColumn`.

## Proposed changes

### New file: `src/Nivara.Extensions/AI/NivaraEmbeddingExtensions.cs`

Namespace `Nivara.AI`, `public static class NivaraEmbeddingExtensions`.
Public-API-only against core (`NivaraColumn<T>.CreateFromSpans(values, nullMask)`
with `true = null`, `CreateForReferenceType`, `NivaraSeries<T>(column, indexColumn)`,
`NivaraFrame.Create`). `Nivara.Extensions` has `InternalsVisibleTo` for the test
projects only, so nothing internal may leak into this file.

```csharp
public delegate float[] NullEmbeddingReplacement(int rowIndex, int dimension);
public delegate float[]? PartialEmbeddingRepair(int rowIndex, float[] values, int[] nullDimensions);

public static NivaraFrame ToNivaraFrame(
    this GeneratedEmbeddings<Embedding<float>> generated,
    string embeddingColumnName = "Embedding",
    string? idColumnName = null,
    IEnumerable<object>? ids = null);

public static GeneratedEmbeddings<Embedding<float>> ToGeneratedEmbeddings(
    this NivaraFrame frame,
    string embeddingColumnName = "Embedding",
    NullEmbeddingReplacement? nullReplacement = null,
    int? dimension = null);

public static NivaraFrame ToEmbeddingMatrix(
    this NivaraFrame frame,
    string embeddingColumnName = "Embedding",
    string columnPrefix = "E",
    int? dimension = null);

public static NivaraFrame ToEmbeddingFrame(
    this NivaraFrame matrixFrame,
    string embeddingColumnName = "Embedding",
    string columnPrefix = "E",
    int? dimension = null,
    PartialEmbeddingRepair? repair = null);

public static NivaraSeries<float> RowCosineSimilarity(
    this NivaraFrame frame, string embeddingColumnName,
    NivaraSeries<float> query, IColumn? labels = null);
public static NivaraSeries<float> RowCosineSimilarity(
    this NivaraFrame frame, string embeddingColumnName,
    float[]? query, IColumn? labels = null);
public static NivaraSeries<float> RowDot(... same two overloads ...);

public static Task<NivaraFrame> GenerateFrameAsync<TInput>(
    this IEmbeddingGenerator<TInput, Embedding<float>> generator,
    IEnumerable<TInput> values,
    string embeddingColumnName = "Embedding",
    CancellationToken cancellationToken = default);
```

Null contract:

| Situation | Behaviour |
|---|---|
| null `float[]` row → matrix | all `E*` cells null (faithful, no policy) |
| null query element | all scores null (inherited kernel semantics) |
| null `float[]` query (array overload) | all scores null |
| null row → `ToGeneratedEmbeddings`, policy supplied | policy vector, validated against `dimension` |
| null row → `ToGeneratedEmbeddings`, no policy | throws, naming row and `nullReplacement` |
| policy returns wrong-length vector | throws, naming row and both lengths |
| fully-null matrix row → `ToEmbeddingFrame` | null `float[]` (faithful, no policy) |
| partially-null matrix row, repair supplied | repair vector, or `null` → null embedding |
| partially-null matrix row, no repair | throws, naming row and `repair` |
| heterogeneous vector lengths | throws, naming row and both lengths |
| all-null column, no `dimension` | throws — dimension unknowable even with a policy |

### New file: `tests/Nivara.Tests/AI/NivaraEmbeddingExtensionsTests.cs`

NUnit 4.x, `Method_Scenario_ExpectedBehavior`, AAA. ~30 tests:

- conversion in: carries vectors, id column, id count mismatch, empty input
- matrix: `E0..E{n-1}` + prefix, null row → all-null row, heterogeneous lengths
  throw with row index, all-null without/with `dimension`
- round-trip: bit-exact float equality both directions, no tolerance
- policy: each row of the null contract table above, including
  policy-not-invoked-when-clean and repair-returns-null
- scoring parity: wrapper vs scalar `TensorPrimitives.CosineSimilarity` per row, exact;
  `RowDot` likewise
- nulls: null row → null score, null query → all null, `float[]?` null → all null
- labels: labels attached; `TopKDescending` positional labels recoverable as indices
  (pins the stringification and de-risks #912)
- ACL path: `FilterByMask` on an embedding frame preserves vectors and nulls

### New file: `docs/EMBEDDINGS.md` (gated)

Two storage forms and when to use each; MEAI round-trip; the null table; the
policy delegates in use; Top-K + label recovery; access-control filtering before
conversion; "Nivara does not own the math". Fences carry their own `using`
directives (the gate hoists them; `Nivara.AI` and `Microsoft.Extensions.AI` are
not in `BaseUsings`) and define a deterministic `Func<string, float[]>` so they
compile standalone.

### Docs gate updates

| Pin | Now | After |
|---|---|---|
| `DocsCoveragePins.ExpectedDocumentCount` | 56 | 57 |
| `DocsCoveragePins.ExpectedCitedDocumentCount` | 19 | 20 |
| `DocsCoveragePins.ExpectedCitationCount` | 113 | 113 + C |
| `DocsCoveragePins.ExpectedCodeCitationCount` | 27 | 27 (symbol-form citations only) |
| `DocsCoveragePins.ExpectedDocumentReferenceCount` | 232 | 232 + R |
| `DocSnippet.GatedDocuments` | 4 | 5 |
| `DocumentationSnippetTests` gated | 106 | 106 + N |
| `DocumentationSnippetTests` all | 252 | 252 + N |
| `DocumentationSnippetTests` excluded / ungated | 2 / 144 | 2 / 144 |

`C` = citations written, `N` = csharp fences written, `R` = pointers to
`docs/EMBEDDINGS.md` added elsewhere. `ExpectedDocumentCount` is verified against
the code as 9 root `*.md` + 47 under `docs/` recursively, minus the excluded
`docs/TODO.md`.

### `docs/TENSORS.md`

Fix the stale bullet at L267 ("Embedding columns as a first-class data type" —
contradicts the re-scope) and add a prose pointer to `docs/EMBEDDINGS.md`. Prose
only: adding a ```csharp fence to an ungated doc would ship uncompiled code.

## Blast radius

| Area | Impact |
|---|---|
| `src/Nivara.Extensions/AI/NivaraEmbeddingExtensions.cs` | new, additive |
| `src/Nivara.Extensions/AI/NivaraEmbeddingGenerator.cs` | doc comments only |
| `src/Nivara/*` core | none |
| `tests/Nivara.Tests/AI/*` | new fixture, no existing test touched |
| `tests/Nivara.Tests/Docs/*` | pin + allowlist edits only |
| `docs/EMBEDDINGS.md`, `docs/TENSORS.md` | new gated doc + prose pointer |
| Consumers | #912 is the only downstream; nothing else calls these APIs yet |

Risk is concentrated in null-mask expansion (a missed mask yields a finite false
score) — covered by the parity and property-style null tests.

## Verification

```
dotnet build Nivara.slnx
dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.AI"
dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"
dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore
```

The full suite is not run without asking (AGENTS.md). The Docs filter is 66
tests / ~25 s and is mandatory because `docs/` changes.

## Planned commits

1. `docs: plan #867 embedding convention in TODO.md`
2. `Add embedding convention glue between frames and Microsoft.Extensions.AI`
3. `Add tests for the embedding convention glue`
4. `Document the embedding convention in docs/EMBEDDINGS.md`
5. `Update docs coverage pins for docs/EMBEDDINGS.md`
6. `docs: remove TODO.md — plan executed`

## GitHub issues log

- [ ] (none yet)
