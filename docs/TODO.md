# Plan — #515: compile-gate fenced csharp blocks in documentation

Branch: `khurram/515`, branched off `khurram/516` at `8e0d5d8b`.

## Problem

No fenced code block in any Nivara document is compiled. A snippet naming a type or
parameter that does not exist ships unnoticed and teaches a shape that cannot be copied.
#510, #502, #501 and #498 were all found by reading documentation against the source
rather than by any tool. The root cause in every case is a rename or signature change
that documentation never followed.

## Scope measured on this branch

Fenced ```csharp blocks, repo-wide: **233 across 21 documents**. The issue's table
covers 110 across 7 (47% of blocks, 33% of documents). First-stage allowlist per the
issue: `docs/STREAMING.md` (11) + `docs/AGENT-CODE-EXAMPLES.md` (3) = 14 blocks.

Documents with blocks: GETTING-STARTED.md 63, docs/AUTODIFF.md 59, docs/LINQ.md 22,
EXAMPLES.md 15, docs/research/AGENT-FRAMEWORK.md 15, docs/STREAMING.md 11,
docs/BFLOAT16.md 7, ARCHITECTURE.md 6, docs/INTEGERS.md 5,
docs/blog/3-self-attention-implemented-the-dirty-details.md 5,
docs/blog/1-the-workhorses-from-words-to-numbers.md 5,
docs/blog/4-the-bypass-wire-the-rectifier-and-what-i-learned.md 4,
docs/ACCELERATION.md 4, docs/ILGPU.md 3, docs/AGENT-CODE-EXAMPLES.md 3,
docs/blog/2-self-attention-the-soft-crossbar-switch.md 1, docs/SAFETENSORS.md 1,
docs/RETRAINING.md 1, docs/QWEN.md 1, docs/TENSORS.md 1, README.md 1.

## Premise corrections to the issue (evidence)

1. **There is no `Nivara.Gpu` assembly.** Only `Nivara` and `Nivara.Extensions` ship.
   The name in the issue comes from `docs/ACCELERATION.md:22`, which cites
   `src/Nivara.Gpu` — a path that does not exist. Tracked separately (#517).
2. **Only 9 of 169 blocks in those 8 documents declare a type** (5%); 13 use `await`;
   31 carry top-level `using` directives that must be hoisted. So "every gated block
   becomes self-contained" is a rewrite of ~94% of the corpus, and GETTING-STARTED.md
   is deliberately progressive (block N depends on block N-1's `column`/`frame`).
   Hence the preamble-reference contract below, not self-containment.
3. **`docs/STREAMING.md` is harder than "triageable".** 9/11 use `await`; 8 reference
   undefined locals; 2 need `Streamix.AspNetCore` + ASP.NET Core. Handled below.
4. Prose defects a compile gate structurally cannot see: `docs/LINQ.md:5-6,27` calls
   `QueryFrame` internal (it is `public sealed class`, `src/Nivara/Query/QueryFrame.cs:14`);
   `docs/ACCELERATION.md:22` cites the nonexistent `src/Nivara.Gpu` (#517).

## Proposed changes

### 1. `tests/Nivara.Tests/Nivara.Tests.csproj`

Add `Microsoft.CodeAnalysis.CSharp` 5.0.0 (4.0.0 and 5.0.0 are both in the local
NuGet cache; 5.0.0 matches the .NET 11 SDK). No CI change is needed — CI already
builds and runs this project with `--filter "Category!=Performance"`.

### 2. Wrapping contract — preamble references

Each gated block is preceded by an HTML comment naming a context, and that context is
a real `.cs` file in the test project (the `tests/Nivara.Tests/IO/FileHandleProbe.cs`
precedent the issue cites — real fixtures, not mocks):

```markdown
<!-- gate: preamble=streaming-as-stream -->
```csharp
await using var query = Csv.ScanAsQueryFrame("telemetry.csv") ...
```

The generated compilation unit is `preamble + block body verbatim`, as **top-level
statements** — one wrapper that covers `await`, `await using` and `await foreach`
uniformly. Block text is never edited to make it compile. Preambles are themselves
compiled C#, so they cannot rot; a dangling preamble name is a hard failure, which is
what makes "must not silently skip" structural rather than aspirational.

Second mode `file` for blocks that are whole compilation units: the preamble
contributes only `using` directives.

### 3. `tests/Nivara.Tests/Docs/DocSnippet.cs` — extractor + registry

- Repo root by walking up to `Nivara.slnx` (more robust than the fixed 5-level `..`
  walk at `tests/Nivara.Tests/AutoDiff/CrossFrameworkParityTests.cs:9`).
- Extract only ` ```csharp ` fences. Records each block's 0-based start line.
- Classify every discovered block into exactly one bucket: `Gated`,
  `Excluded(reason)`, or `Ungated(doc)`. Allowlist and exclusions are **explicit
  paths, never path globs**.
- Two lists: `GatedDocuments` (STREAMING.md, AGENT-CODE-EXAMPLES.md) and
  `ExcludedDocuments` with per-block reasons.

### 4. `tests/Nivara.Tests/Docs/DocSnippetCompiler.cs` — Roslyn

- References from `AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES")`, split on
  `Path.PathSeparator`, deduped by simple name, filtered to existing files. That is
  the real runtime closure (`Nivara`, `Nivara.Extensions`, `Streamix 1.2.3`, NUnit,
  System.*) on any OS — no hardcoded framework list, no `Nivara.Gpu` question.
- `CSharpParseOptions(LanguageVersion.Latest)`.
- `GetDiagnostics()` only, **not** analyzer runs, so NUnit.Analyzers cannot turn this
  into the style linter the issue forbids. Severity `Error` only; warnings ignored.
- Line mapping by manual offset, so every message reads `docs/STREAMING.md:246`.

### 5. `tests/Nivara.Tests/Docs/Preambles/*.cs`

| Preamble | Blocks (doc:line) | Supplies |
|---|---|---|
| `streaming-as-stream` | STREAMING:133 | `CancellationToken ct`, `void Report(double, int)` |
| `streaming-flux-basic` | STREAMING:246, 434 | `ISink sink` |
| `streaming-flux-window` | STREAMING:259 | `IPager pager` |
| `streaming-flux-training` | STREAMING:280 | `QueryFrame query`, `int featureCount` |
| `streaming-flux-reverse` | STREAMING:311 | — (self-contained) |
| `streaming-flux-frame` | STREAMING:321 | `QueryFrame queryFrame` |
| `streaming-flux-publish` | STREAMING:342, 366 | `dashboard`, `archival`, `liveUI` |
| `agent-tensor-kernel` | AGENT:7 | (after the `AsTensorSpan` doc fix) |
| `agent-null-tensor` | AGENT:16 | `T`, `len`, `values`, `hasNulls` |

### 6. `tests/Nivara.Tests/Docs/DocumentationSnippetTests.cs` — the gate

Asserted numbers, not printed ones (AGENTS.md: "a gate must state its own coverage in
every summary, and reduced coverage is a failure, not a neutral event"):

- `Gated == 12` (9 STREAMING + 3 AGENT-CODE-EXAMPLES)
- `Excluded == 2`
- `Gated + Excluded + Ungated == 233` — the test discovers the true total by scanning,
  so "12 of 233 checked" is a claim it can back, not a printout.
- The registry names all 21 snippet-bearing documents, so a new document containing
  snippets forces an explicit decision instead of drifting into the skip bucket.
- Exclusion list asserted **by value**: exactly `docs/STREAMING.md` × 2, reason
  "requires Streamix.AspNetCore + ASP.NET Core", not a glob.

### 7. `tests/Nivara.Tests/Docs/Fixtures/Broken.md` — negative control

Deliberately outside `docs/`, so it needs no exclusion and cannot mislead a reader.
One broken block, gated through the identical extract→compile path, asserting a real
diagnostic that names the fixture's line. Plus an in-memory broken snippet for the
compiler path alone. This is acceptance criterion 5: a green run is known to mean
something.

### 8. Documentation edits

- Add the one-line `<!-- gate: preamble=... -->` comment above each of the 14 blocks.
- Fix `docs/AGENT-CODE-EXAMPLES.md:9`: `tensorStorage.AsTensorSpan()` "returns
  TensorSpan<T>" — `AsTensorSpan()` is the BCL `Tensor<T>` method returning `Span<T>`
  (`src/Nivara/Tensors/TensorInteropExtensions.cs:54`); the Nivara zero-copy path is
  `AsTensorView()` (`src/Nivara/NivaraColumn.cs:1239`). `MyKernels` needs a real target.
- Mark the 2 ASP.NET Core blocks (STREAMING:385, STREAMING:411) with the exclusion
  reason, so the skip is visible in prose as well as in the test summary.

## Blast radius

- **New files only** under `tests/Nivara.Tests/Docs/`, plus one `PackageReference` in
  `tests/Nivara.Tests/Nivara.Tests.csproj`. Nothing under `src/` changes.
- `Nivara.Tests` gains a Roslyn dependency: slower restore and a larger test assembly.
  Affects every local `dotnet test` in this repo, and CI's test step.
- Markdown edits are additive comments, except the `AsTensorSpan` fix in
  `docs/AGENT-CODE-EXAMPLES.md`, which is a real content change to a document
  `AGENTS.md` points agents at.
- **No downstream callers.** The gate is a test; nothing in `src/` or `samples/`
  depends on it. The only production-visible artefact is the `AsTensorSpan` doc fix.
- Existing tests are untouched; `GateEvaluatorTests` is the closest precedent for
  the asserted-coverage style and is not affected.
- CI needs no edit — `Nivara.Tests` already runs under
  `--filter "Category!=Performance"`, and the gate is not a timing test, so it must
  NOT carry `[Category("Performance")]`.

## Verification

1. `dotnet build Nivara.slnx -c Release` — the preambles are compiled C#; a broken
   preamble fails the build itself.
2. `dotnet test -c Release --filter "FullyQualifiedName~DocumentationSnippetTests"`
   — green, and reporting `12 gated / 2 excluded / 233 total`.
3. Negative control: confirm `Fixtures/Broken.md` fails with a diagnostic naming its
   line, i.e. the gate is not vacuously green.
4. Temporarily introduce a bogus member into a gated block and confirm the failure
   message reads `docs/STREAMING.md:<line>`, then revert.
5. Full suite `dotnet test -c Release --filter "Category!=Performance"` to confirm no
   regression from the new dependency.

### Verification status as executed

| Step | Result |
|---|---|
| 1 | Partial. `Nivara.Tests` builds clean in Release. `Nivara.slnx` as a whole was not built. |
| 2 | Done. 21 passed / 0 failed across both doc fixtures in Release (19 at the time of the gate commit, 21 after the G2 fixes). |
| 3 | Done. Negative control reports `Fixtures/Broken.md:13 CS1061`. |
| 4 | Done. Injected `ThisMemberDoesNotExist` into `docs/STREAMING.md`; the gate reported `docs/STREAMING.md:179 CS1061`, the exact injected line. Reverted. |
| 5 | **Not run** — declined at the human's direction on 2026-10-04. The new `Microsoft.CodeAnalysis.CSharp` 5.0.0 dependency is therefore unverified against the rest of the suite. CI's `--filter "Category!=Performance"` run on ubuntu is the first check. |

## Planned commits

1. `docs: plan #515 in TODO.md`
2. `test: add the doc snippet extractor and compile gate infrastructure`
3. `test: gate docs/STREAMING.md and docs/AGENT-CODE-EXAMPLES.md snippets`
4. `docs: add gate preambles and fix the AsTensorSpan snippet`
5. `docs: remove TODO.md - plan executed`

## Drift from the plan, and what the plan got wrong

Recorded rather than rewritten, so the comparison above stays honest.

1. **§2's preamble shape was not implementable as written.** The plan put locals in a compiled
   preamble file. Locals only exist inside a method or at top-level statements, and a preamble
   file carrying top-level statements would collide with `Nivara.Tests`' own entry point
   (CS8802, one compilation unit may hold them). Landed instead: preamble *types* are compiled
   in `Preambles/SnippetStubs.cs`, and the *locals* are spliced verbatim from a `locals:`
   line in the gate comment. The anti-rot property still holds where it matters — every type a
   snippet names is compiled C# — and the gate comment now uses a multi-line form so `locals`
   can hold arbitrary code.
2. **Four wrap modes, not two.** `TopLevel` and `File` as planned, plus `GenericLocal` because
   `docs/AGENT-CODE-EXAMPLES.md` block 2 uses the type parameter `T` at top level, and
   `MemberDecl` because block 3 is a method declaration, not a type declaration.
   `File` currently has no consumer among the gated blocks; its only use is its own test. It is
   kept because stage 2 documents declare types, and it should be deleted if stage 2 does not
   need it.
3. **§8's `AsTensorSpan` diagnosis was wrong.** `Tensor<T>.AsTensorSpan()` returns
   `TensorSpan<T>`, not `Span<T>`, so the original comment's return type was right; the real
   defects were the undeclared `tensorStorage` variable and the fictional `MyKernels`. The
   landed fix uses `NivaraColumn<T>.TryGetSpan(out ReadOnlySpan<T>)`, the actual zero-copy path.
4. **§8 found a second, larger defect.** `NivaraColumn<int>.CreateFromNullable` does not exist;
   the static is on the non-generic class with `T` inferred. Fixed in the gated document and
   tracked as #520 for the 16 other call sites. So blast radius's "the only production-visible
   artefact is the `AsTensorSpan` doc fix" understated it: two API shapes in
   `docs/AGENT-CODE-EXAMPLES.md` changed, still markdown-only.
5. **Planned commit 4 was folded into commit 3.** The gate comments and the `AsTensorSpan`
   fix are one logical unit, since the gate is what proved the defect. An unplanned fifth
   commit carries the G2 review fixes.

## Risks and open items — as resolved

- **The Streamix operator risk did not materialise.** `Named`, `Retry`, `Checkpoint`, `Trace`,
  `Log`, `Filter`, `Publish`, `Replay`, `Connect`, `ForEachAsync`, `WindowByTime`, `FlatMap`
  and `Where` all resolved from Streamix 1.2.3, so no gated block needed a doc fix on account
  of them. All 12 gated blocks compile.
- **Defect yield was 2, not 1**: `CreateFromNullable` (#520) and the zero-copy snippet. Stage 1
  found nothing in `docs/STREAMING.md` — its 9 gated blocks are correct as written. The defect
  value concentrates in stage 2 (GETTING-STARTED.md 63, AUTODIFF.md 59, EXAMPLES.md 15), which
  is why the 233/219 pin is the assertion that carries the gate's credibility.
- `Microsoft.AspNetCore.App.Ref` is present locally, so only `Streamix.AspNetCore` blocks
  11/11 later — one `<FrameworkReference>` plus one `PackageReference`.

## GitHub issues log

- [x] #517 — `docs/ACCELERATION.md:22` cites a nonexistent `src/Nivara.Gpu`; issue
      #515's proposed direction assumed a `Nivara.Gpu` assembly that does not exist
- [x] #518 — `docs/LINQ.md:5-6,27` calls `QueryFrame` internal; it is
      `public sealed class` (`src/Nivara/Query/QueryFrame.cs:14`)
- [x] #520 — `NivaraColumn<T>.CreateFromNullable` does not exist; 16 documented call
      sites use the wrong shape (GETTING-STARTED.md ×9, EXAMPLES.md ×5, AGENTS.md ×1,
      CHANGELOG.md ×1). The static is `NivaraColumn.CreateFromNullable<T>(T?[])`
      (`src/Nivara/NivaraColumn.Factory.cs:19`) and `T` is inferred. Found by this
      gate; stage 2 will resurface it. AGENTS.md's helper list is the highest-value fix.

Reminder: as each task executes, if you find deferred work or a concern, create a
tracked issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record
its number in the log above — do not rely on memory or wait until the plan finishes,
as compaction during execution can lose important items.