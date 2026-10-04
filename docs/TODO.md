# Plan: #532 — resolve the duplicate `SchemaValidationException` public type

## Problem

Two distinct public exception types ship under the same short name in the same
assembly, so a consumer that imports both namespaces gets CS0104 at every use
site:

| Type | Namespace | Declared at | Thrown from |
|---|---|---|---|
| `SchemaValidationException` | `Nivara.Exceptions` | `src/Nivara/Exceptions/QueryEngineExceptions.cs:100` | ~60 query-engine sites (`JoinOperation.cs`, `NivaraFrameExtensions.cs`, `WindowOperations.cs`, `TypeCompatibilityValidator.cs`, `TypedRowFactory.cs`, `ColumnExpression.cs`, …) |
| `SchemaValidationException` | `Nivara.IO` | `src/Nivara/IO/IOExceptions.cs:100` (derives `NivaraIOException`) | `NivaraParquetReader.cs:363`, `NivaraParquetWriter.cs:246,270`; caught in the writer's filters at `NivaraParquetWriter.cs:78,131` |

The source never trips over this itself: `NivaraParquetWriter.cs` and
`NivaraParquetReader.cs` are declared `namespace Nivara.IO;` and import no
`Nivara.Exceptions`, so the bare name resolves there. The clash only reaches
*consumer* code — the normal shape of code that both runs a query and reads a
Parquet file.

### The evidence chain, in order

1. **#532 was filed from real gate output.** `tests/Nivara.Tests/Docs/DocSnippetCompiler.cs:52,56`
   puts both `Nivara.Exceptions` and `Nivara.IO` in `BaseUsings`, so every gated
   snippet sees both.
2. **The gate caught it as CS0104.** Commit `9b45818d` ("Add negative control
   proving the gate catches cross-namespace ambiguity") shipped a fixture,
   `tests/Nivara.Tests/Docs/Fixtures/AmbiguousSchemaException.md`, whose two
   types collide exactly as the library's do. Its negative-control test asserts
   CS0104 appears in the output.
3. **The library defect was never fixed; the doc was.** Commit `fb47dbe8`
   (#539) changed `GETTING-STARTED.md` to `catch (Nivara.Exceptions.SchemaValidationException ex)`
   and added a note at `GETTING-STARTED.md:777-780` telling readers the name is
   ambiguous and to catch the fully-qualified one. That silenced the gate
   without removing the defect.
4. **The gate is coupled to the document.** `EveryGatedBlock_CompilesWithoutErrors`
   compiles snippets out of the markdown *as it currently stands*. So when the
   rename lands, that workaround stops compiling — CS0246 — and the gate will
   demand the doc be updated. Rename and doc are one change, not two.

### Red flag found while grounding (out of scope, tracked separately)

`main` is currently **red**: 53 CS0104 diagnostics from
`EveryGatedBlock_CompilesWithoutErrors`, all rooted in
`GETTING-STARTED.md:35`'s snippet colliding with `Nivara.Linq.QueryFrame`
(CI run 30061467490, and the same 53 locally). Main last built clean at
`2f2fb0e4`, before #528–#532 landed.

This is orthogonal to the rename — a different type pair in a different
namespace — so it does not block this work, but it means **"the gate is green"
cannot be used as the verification for this branch.** Verification below is
scoped to the diagnostics this change can actually affect.

## Decision

Take #532's option 2: rename both types so the name carries its origin.

| Before | After | Namespace |
|---|---|---|
| `Nivara.Exceptions.SchemaValidationException` | `QuerySchemaValidationException` | unchanged |
| `Nivara.IO.SchemaValidationException` | `DataSchemaValidationException` | unchanged |

Namespaces and base classes are unchanged, so `catch` filters, `is` patterns and
existing qualification still behave; only the short name moves.

**No `[Obsolete]` shim is possible.** Both classes are `sealed`, so C# cannot
alias them, and re-introducing a `SchemaValidationException` in any form would
restore the exact ambiguity being removed. This is a clean breaking change.

## Proposed changes

### 1. Source rename (already applied in the working tree — 33 files, +132/−132)

Declaration, ~60 query-engine throw/catch sites, `<exception cref>` tags, the 3
Parquet throw sites and 2 Parquet catch filters, and the `Assert.Throws<>` /
`Throws.TypeOf<>` call sites in tests.

### 2. Source XML docs the mechanical pass skipped

The classes' own `<summary>` lines still name the old type:

- `src/Nivara/Exceptions/QueryEngineExceptions.cs:103,110`
- `src/Nivara/IO/IOExceptions.cs:103,110`

Also sharpen `src/Nivara/Exceptions/QueryEngineExceptions.cs:98` — "Exception
thrown when schema validation fails" is the vagueness the rename exists to
remove; it should say query planning / expression validation. The IO summary at
`src/Nivara/IO/IOExceptions.cs:98` already says "during I/O operations" and is
fine as is.

### 3. Test names, and one assertion that was too loose to notice the rename

Eleven method names across six files still say `ThrowsSchemaValidationException`,
plus `#region SchemaValidationException Tests` at
`tests/Nivara.Tests/IO/IOExceptionTests.cs:128`.

The one that matters: `tests/Nivara.Tests/IO/IOExceptionTests.cs:364` asserts
`Does.Contain("SchemaValidationException")` against `GetType().Name`. It passes
today only because that string is a **substring** of `DataSchemaValidationException`
— which is exactly why the rename went unnoticed there. Tighten to the exact name.

Also: `ParquetWriterTests.cs:199`, `ColumnExpressionTests.cs:52,62`,
`QueryPlanTests.cs:158`, `TypedLinqTests.cs:61,75`, `SortOperationTests.cs:113`.

### 4. Present-tense prose in tests

`tests/Nivara.Tests/MixedTypeIntegrationTests.cs:199,331` (inline comments) and
`tests/Nivara.Tests/Docs/Preambles/SnippetStubs.cs:65`.

### 5. NEW gate — no two public type names may collide across namespaces

`tests/Nivara.Tests/Exceptions/TypeNameUniquenessTests.cs`

The point of the rename is that this cannot recur. Scoped to **all types**, not
just exceptions:

- Reflect over `typeof(Exception).Assembly` **and** the `Nivara.Extensions`
  assembly (the test project already references both). This also catches a
  collision *between* the two shipped packages, which a per-file check misses.
- Take non-nested types with a non-null namespace. Nested types cannot cause
  CS0104. Include `internal`: two internal types in one assembly in different
  namespaces clash internally just as hard.
- Group by `Type.Name` — which carries the `` `1 ``/`` `2 `` arity suffix, so
  generic overload pairs (`IColumn` / `IColumn<T>`) are correctly *not* flagged —
  and assert each name maps to exactly one `FullName`.

**Negative control.** Put the scan in a pure function and test *that function*
against synthetic input (two same-named types in different namespaces must be
reported). Otherwise the gate could pass because the reflection silently
returned nothing — which is the failure mode the repo's existing
`NegativeControl_TheGateStillRejectsABrokenSnippet` exists to prevent.

**Feasibility, verified.** All 299 top-level type declarations under `src/` were
parsed for name collisions. **Zero cross-namespace collisions exist today**, so
this gate passes on day one with no allowlist and no collateral renames. The only
repeated names are six generic-arity pairs (`IColumn`/`IColumn<T>`,
`NivaraColumn`/`NivaraColumn<T>`, `IQueryOperation`/`IQueryOperation<T>`,
`IQueryNodeVisitor`/`IQueryNodeVisitor<T>`, `IQueryPlanVisitor`/`IQueryPlanVisitor<T>`)
plus one genuine `partial` (`NivaraFrameExtensions`).

**Coverage to be stated in the file's doc comment**, so a green run is not
mistaken for a broad one: covers type *names* in the two shipped assemblies
only. Does not cover `samples/` (not shipped; sample-local helpers would produce
false positives), member or method names, or generic arity collisions.

### 6. Documentation

Undo the CS0104 workaround, then update the rest:

- `GETTING-STARTED.md:771,827` — drop the `Nivara.Exceptions.` qualification.
  The workaround is no longer needed and teaching readers to work around a
  library defect is the wrong lesson.
- `GETTING-STARTED.md:777-780` — **delete the note.** It asserts "Nivara ships two
  types named `SchemaValidationException`", which becomes false.
- `EXAMPLES.md:328` — the comment naming the thrown type.
- `ARCHITECTURE.md:756` — the exception-hierarchy bullet.
- `docs/LINQ.md:58,274,346` — prose references.

### 7. CHANGELOG

An `[Unreleased] → ### Changed` entry: breaking public API rename, both
directions, and why. Required precisely because no shim is possible.

## What the documentation gate can and cannot see

Recorded because it bounds the verification. `GatedDocuments` is only
`docs/AGENT-CODE-EXAMPLES.md`, `docs/STREAMING.md`, `EXAMPLES.md`,
`GETTING-STARTED.md`. `ARCHITECTURE.md` and `docs/LINQ.md` are in
`UngatedDocuments` — explicitly never compiled.

| Site | Gated? | Real code or comment? | Gate catches it? |
|---|---|---|---|
| `GETTING-STARTED.md:771` | yes | `catch` clause | **yes** |
| `GETTING-STARTED.md:827` | yes | `catch` clause | **yes** |
| `EXAMPLES.md:328` | yes | `//` comment | no — compiles either way |
| `ARCHITECTURE.md:756` | **no** | bullet | no |
| `docs/LINQ.md:58,274,346` | **no** | prose | no |

So the gate covers 2 of the 6 sites. `rg -w SchemaValidationException` remains
necessary for the other four, and **a green gate is not evidence they are
correct.**

## Blast radius

| Area | Impact |
|---|---|
| `Nivara` (core) | `QuerySchemaValidationException` renamed — public API break. ~60 internal sites updated. |
| `Nivara.IO` | `DataSchemaValidationException` renamed — public API break. 3 throw sites, 2 catch filters in `Nivara.Extensions`. |
| `Nivara.Extensions` | Parquet reader/writer updated; namespaces and base classes unchanged. |
| `GETTING-STARTED.md` | 2 gated snippets + 1 note; citation-gate risk if the note is moved rather than deleted (see Verification). |
| `EXAMPLES.md`, `ARCHITECTURE.md`, `docs/LINQ.md` | prose only, no snippets compiled. |
| Test assemblies | 11 test names, 1 assertion, 3 comments. |
| New file | `tests/Nivara.Tests/Exceptions/TypeNameUniquenessTests.cs`. |

No behaviour change: same types, same messages, same throw sites, same catch
semantics. Only the short names move.

## Verification

1. `dotnet build Nivara.slnx` — green.
2. Targeted gate, **diffed against the baseline** (the tree is red on `main`):
   ```
   dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --nologo `
     --filter "FullyQualifiedName~EveryGatedBlock_CompilesWithoutErrors"
   ```
   The gate is doc-coupled, so renaming without updating the doc fails it with
   CS0246 at `GETTING-STARTED.md`. After the doc update it must return to
   **exactly the same 53 baseline diagnostics** — no more, no fewer. Any other
   delta is a regression. (Needs the full diagnostic list saved first; the
   summary line truncates.)
3. Full suite in Release — `dotnet test -c Release --filter "Category!=Performance"`.
   Release per `AGENTS.md`: a Debug run compares unoptimized `Nivara.dll`
   against ReadyToRun framework code. Expect the 53 pre-existing failures plus any
   other pre-existing breakage on `main`; the bar is **no new** failures.
4. `TypeNameUniquenessTests` green, including its negative control.
5. `rg -w SchemaValidationException` must return exactly two intentional
   survivors: `CHANGELOG.md:312` and `tests/Nivara.Tests/Incident/IncidentSurfaceTests.cs:12,31`.
   Both are past-tense narratives of prior incidents — the same "the *before* side
   is correct in context" precedent the CHANGELOG sets for the #222 removal entry.

## Planned commits

1. `docs: plan #532 rename in TODO.md`
2. `Rename the two clashing SchemaValidationException types` — source rename,
   XML docs, test names, the tightened assertion, prose comments, the new gate.
3. `Update the docs for the renamed schema-validation exceptions` — the four
   markdown documents + CHANGELOG.

## GitHub issues log

- [ ] #532 — resolve the duplicate `SchemaValidationException` public type *(this branch)*
- [ ] TBD — `main` is red: 53 CS0104 diagnostics in `EveryGatedBlock_CompilesWithoutErrors`
      from `GETTING-STARTED.md:35` colliding with `Nivara.Linq.QueryFrame`. Introduced
      after `2f2fb0e4`. Unrelated to #532 but blocks using "gate is green" as a
      verification signal. *(created while grounding #532)*
- [ ] TBD — `DataFrameSchemaValidationException` (`src/Nivara/Exceptions/DataFrameExceptions.cs:91`)
      is dead public API: never thrown by production code, only constructed in
      `tests/Nivara.Tests/Exceptions/DataFrameExceptionTests.cs`. A third
      schema-validation exception in the same namespace as the renamed
      `QuerySchemaValidationException`. No name clash, so #532 is unaffected;
      removing it is a separate breaking change. *(created while grounding #532)*

**While executing:** if further deferred work or concerns surface, create the
issue immediately with `gh issue create --repo khurram-uworx/Nivara` and record
the number above — do not rely on memory or wait for the end of the plan, as
compaction during execution can lose items.