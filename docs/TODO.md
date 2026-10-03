# Plan: #510 — doc snippets call APIs and name types that do not exist

Branch: `khurram/510`, created off `khurram/507` (per human instruction), which already
carries #502's executed plan (`840801f8 docs: remove TODO.md - #502 plan executed`).

## Problem

Documentation teaches shapes that cannot be compiled. `AGENTS.md` points agents at
`docs/AGENT-CODE-EXAMPLES.md` and `docs/STREAMING.md` as the sources for how to write Nivara
code, so a snippet naming a nonexistent type teaches an uncopyable shape — and a reader who
trusts it will not notice the name does not exist.

Two root causes, both renames/behaviour that documentation never followed:

1. **`ParquetReader` → `NivaraParquetReader`** (`4ca14bf2`, `R088`). Every doc mention of a
   `Parquet.*` Parquet entry point, and `Nivara.IO.ParquetReader`, names a type that stopped
   existing at that rename. `Nivara.IO` contains `Csv`, `Json`, `NivaraParquetReader` — no
   `Parquet`. (`Parquet` *is* a namespace, from Parquet.Net: `Parquet.ParquetRowGroupReader`
   at `ParquetDataSource.cs:114`. That is almost certainly how the bad name got written.)
2. **A design proposal presented as configuration.** `docs/ACCELERATION.md:169`'s "Streaming
   Memory-Budget Enforcement" section is Phase 1 of #325 — all ten acceptance criteria
   unchecked (`:445-454`). It shows `BudgetEnforcement`, a `MemoryBudget` class, and
   `AsStream(enforced: true)`. None of the three exists in `src/`.

## What #502 already fixed (do not re-touch)

Verified against this base, not against `khurram/502`:

- `docs/STREAMING.md:135` and `docs/ACCELERATION.md:215` — the `.Filter("status", "OK")` calls
  are gone; both now use `Filter(ColumnExpressions.Col("status") == "OK")`
  (`417b5c63`).
- `docs/ACCELERATION.md:216` — `.AsStream(enforced: true)` is gone from the snippet.
- `GETTING-STARTED.md:534` and `docs/STREAMING.md:136` — `.Select("Name", "Age")` and
  `.Select("timestamp", "value")` are **valid and must not be "fixed"**:
  `NivaraFrame.Select(params string[])` (`NivaraFrameExtensions.cs:168`) and
  `QueryFrame.Select(params string[])` (`QueryFrame.cs:121`) are both `params`. #502's G2
  reached the same conclusion and reverted a draft that had changed them (`a153cb4e`).

## Verified reality (grounding for every edit below)

| Documented | Reality |
|---|---|
| `Csv.ScanAsQueryFrame(path, chunkSize)` | `ScanAsQueryFrame(string filePath, CsvOptions? options = null)` — `CsvExtensions.cs:63`, `Nivara.IO.Csv` |
| `Json.ScanAsQueryFrame(path, chunkSize)` | `ScanAsQueryFrame(string filePath, JsonOptions? options = null)` — `JsonExtensions.cs:63`, `Nivara.IO.Json` |
| `Parquet.ScanAsQueryFrame(path, chunkSize)` | `ScanAsQueryFrame(string filePath, ParquetReadOptions? options = null)` — `NivaraParquetReader.cs:133`, `Nivara.IO.NivaraParquetReader`. **No `chunkSize` parameter exists on any of the three.** |
| `.AsStream(enforced: true)` | `AsStream(int chunkSize = 10000, CancellationToken ct = default)` — `QueryFrame.cs:461` |
| `Nivara.IO.ParquetReader` | `Nivara.IO.NivaraParquetReader` (`NivaraParquetReader.cs:10`) |
| `src/Nivara.Extensions/IO/ParquetReader.cs` | `NivaraParquetReader.cs`; and `ParquetWriter.cs` is `NivaraParquetWriter.cs` |

Where the chunk size actually goes: the factories take options only, and row count per chunk
is set by `AsStream(chunkSize:)` — row-oriented sources honour it, Parquet treats it as
advisory and aligns to row groups (`QueryFrame.cs:446-454`).

### Why no "here is how you set it today" snippet is possible

The issue offers a fallback for `AsStream(enforced: true)`: *"show how to set
`BudgetEnforcement` on a `NivaraExecutionContext` directly."* That cannot be written
truthfully, and the reason is the substance of the fix:

- `QueryFrame.AsStream` (`QueryFrame.cs:461-478`) constructs its own
  `NivaraExecutionContext(ExecutionStrategy.Streaming)` inline and sets only `CancellationToken`,
  `ChunkSize`, `ExecutionDiagnostics`. `MemoryBudget` is left at the 1 GB constructor default
  (`NivaraExecutionContext.cs:17`) and **no overload accepts a context**.
- `StreamingExecutionStrategy` is **internal** (`StreamingExecutionStrategy.cs:8`), so its
  `public StreamChunksAsync(QueryPlan, NivaraExecutionContext, CancellationToken)` (`:498`) is
  unreachable from user code.
- Public `IExecutionStrategy` (`ExecutionEngine.cs:37-70`) has no streaming member.
- The only reachable consumer of a custom `MemoryBudget` is `ExecutionEngine.Execute(plan,
  context)` (`ExecutionEngine.cs:117`) with `QueryFrame.ToQueryPlan()` (`:1057`) — which
  **materializes a whole frame** and yields no chunk stream. There the budget only feeds
  `StreamingBudgetTracker` (advisory warning) and `CalculateChannelCapacity` (`:94-101`).

So there is **no public knob for a streaming memory budget at all**, which is why
`ACCELERATION.md:194`'s "Mode is set via `NivaraExecutionContext` or at strategy construction
time" is itself the misleading sentence: the object a reader would construct never reaches
`AsStream`, and `StreamingExecutionStrategy`'s constructor takes no arguments at all.

## Proposed changes

### Commit A — `docs: correct the Parquet scan-factory name and signatures`

One reason: the rename at `4ca14bf2` was never followed in the docs, and the factory list
also documents a `chunkSize` parameter that exists nowhere.

| File | Line | Change |
|---|---|---|
| `ARCHITECTURE.md` | 319 | `Parquet.ScanAsQueryFrame` → `NivaraParquetReader.ScanAsQueryFrame` |
| `ARCHITECTURE.md` | 356-358 | replace the three bullets with the real signatures + options types, and say where `chunkSize` goes |
| `ARCHITECTURE.md` | 1024 | `Parquet.ScanAsQueryFrame` → `NivaraParquetReader.ScanAsQueryFrame` |
| `docs/STREAMING.md` | 12 | `Parquet.ScanAsQueryFrame(string, ParquetReadOptions?)` → `NivaraParquetReader.ScanAsQueryFrame(...)`; the parameter list was already right, only the type prefix was wrong |
| `samples/NivaraIncident/README.md` | 83, 167 | `Parquet.ScanAsQueryFrame` → `NivaraParquetReader.ScanAsQueryFrame` (the sample's own code already calls it that way — `Ingestion.cs:9`) |
| `samples/NivaraFineTuning/README.md` | 90, 339 | `Nivara.IO.ParquetReader` → `Nivara.IO.NivaraParquetReader` (`Sst2Dataset.cs:87`) |
| `docs/plan/ARROW-ROADMAP.md` | 157 | `ParquetReader.cs`, `ParquetWriter.cs` → `NivaraParquetReader.cs`, `NivaraParquetWriter.cs` |

### Commit B — `docs: correct the Parquet factory name in the v1.4.0 changelog entry`

`CHANGELOG.md:444` names `Parquet.ScanAsQueryFrame` in the v1.4.0 release note. Kept as its
own commit because editing a historical release note is a judgment call distinct from fixing
reference docs, and it should be visible on its own in review.

### Commit C — `docs: mark the streaming budget-enforcement section as an unimplemented design`

Per the human's decision (G1): **banner plus rewrite**, not banner-only.

1. **Status banner** on `# Streaming Memory-Budget Enforcement` (`docs/ACCELERATION.md:169`),
   in the house style already used at `docs/ROADMAP-SUGGESTION.md:3` — naming the section a
   design proposal for #325 Phase 1, not shipped API, and stating that `BudgetEnforcement`,
   `MemoryBudget`, and `AsStream(enforced:)` do not exist yet. This covers all ten mentions
   (`:190, 201, 208, 223-225, 416, 425, 437-438, 450`) with one edit rather than ten.
2. **Rewrite `### Configuration` (`:192-224`)** so it describes today's behaviour: the budget
   is not user-settable through `AsStream`; chunk size is `AsStream(chunkSize:)`; the bounded
   channel is the real backpressure, with the advisory `StreamingBudgetTracker` warning —
   matching the shipped contract already written down at `docs/STREAMING.md:120-129`. Delete
   the two snippets that assign `BudgetEnforcement`.
3. **Fix the comment #502 introduced at `:219`** — "the budget caps how big one gets" describes
   Enforced-mode behaviour that does not exist; `AsStream`'s chunk size is its `chunkSize`
   argument.
4. **Fix `:425`** — the "What stays the same" row claims `QueryFrame.AsStream()` has
   `enforced: true` as an optional parameter. It has no such parameter; the row becomes a
   statement that this proposal would add one.
5. **Reconcile against #325's actual body.** The section says "Phase 1 of issue #325", but
   #325 proposes **spill-to-disk** for boundary operators (`SpillDirectory`, spill/reconstruct).
   The section describes an in-memory byte-budget gate instead. Say so rather than leaving a
   design doc pointing at an issue whose text it does not match.

Acceptance criteria at `:445-454` stay unchecked — they are a proposal's own checklist, and
the banner now tells the reader so.

## Verification

Documentation only. There is no mechanical gate and I am not adding one in this branch: no
test reads `.md` (grepped `tests/` — every hit is a comment citing a doc), and a real
snippet-compiler is out of scope per the issue. Sizing for the follow-up: 110 fenced
`csharp` blocks across the seven main docs (`GETTING-STARTED.md` 63, `docs/LINQ.md` 22,
`docs/STREAMING.md` 11, `ARCHITECTURE.md` 6, `docs/ACCELERATION.md` 4,
`docs/AGENT-CODE-EXAMPLES.md` 3, `README.md` 1), and most are fragments rather than programs.

So each corrected signature is **re-read from source and cited by `file:line` in the commit
message**, which is what makes the claim checkable. Before each commit I re-grep the file to
confirm no other line in it still carries the old name.

No `dotnet build` is needed (no `.cs` touched). No test run is needed either; I will ask
before running anything long regardless.

## Blast radius

- **Commit A** — six markdown files, zero runtime blast radius. Every edit moves a document
  *toward* the code; none asserts behaviour that does not exist. The risk is a typo making a
  signature *less* accurate, which is why each is cited by `file:line`.
- **Commit B** — one changelog line. Historical-record edit; called out separately so it can
  be dropped at review without unpicking the rest.
- **Commit C** — one section of one document. The rewrite removes shipped-API claims and adds
  a "not implemented" banner; it cannot mislead a reader into calling something that does not
  exist. The one thing to watch in review is that the rewrite stays **descriptive** — it must
  not start specifying the design (that is #325's job when it is planned properly), or this
  commit acquires a second reason.
- **Explicitly out of scope:** implementing `BudgetEnforcement` / `MemoryBudget` /
  `AsStream(enforced:)`; adding a doc-snippet compile gate; any `src/` or `tests/` change.

## GitHub issues log

Created during execution, recorded here as they are filed:

- [ ] #NNN — no public knob for a streaming memory budget: `AsStream` hard-codes 1 GB and
  takes no context; `StreamingExecutionStrategy` is internal (so `StreamChunksAsync` is
  unreachable); `NivaraExecutionContext.MemoryBudget` cannot reach the chunk path. The
  capability gap behind the `ACCELERATION.md` rewrite — filed when the rewrite lands.
- [ ] #NNN — doc snippets are never compiled; ~110 fenced `csharp` blocks, no gate. Explicitly
  out of scope per #510; needs a design for fragment-wrapping before it is cheap.

Reminder: as each task executes, if further deferred work or a concern appears, create the
issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record the number
above — do not rely on memory or wait until the plan finishes, as compaction during execution
can lose it.