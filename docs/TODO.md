# #516 — Docs say the default streaming memory budget is 256 MB; the core pipeline default is 1 GB

Branch: `khurram/516` (off `main` @ `695f0b5a`)

## Problem

Two documents state the core query pipeline's default streaming memory budget is 256 MB. It is
1 GB, and that 1 GB is pinned by an executable test assertion.

| Site | Claim | Actual |
|---|---|---|
| `AGENTS.md:123` | "Default memory budget for streaming: 256 MB (configurable)" | 1 GB ❌ |
| `docs/STREAMING.md:105` | "(default budget 256 MB → `(25,600,000) / 100` → 256,000 → clamped to 100,000)" | 1 GB ❌ |
| `ARCHITECTURE.md:733` | `// 1GB` | ✅ already correct |
| `docs/ACCELERATION.md:211`, `:459` | "1 GB constructor default (`NivaraExecutionContext.cs:17`)" | ✅ already correct |
| `docs/ACCELERATION.md:249` | `MemoryBudget = 256 * 1024 * 1024` | inside a snippet already marked *"Proposed only. Does not compile"* — not a defect, leave alone |
| `docs/spec/SPEC-307-308.md:93` | "MemoryBudget = 1GB" | ✅ already correct |

The authoritative value is `src/Nivara/Execution/NivaraExecutionContext.cs:17`
(`MemoryBudget = 1024 * 1024 * 1024`), asserted by
`tests/Nivara.Tests/ExecutionContextTests.cs:18`. So the docs are wrong against a test, not
merely against a comment.

The 256 MB constant is real but lives in a different subsystem:
`src/Nivara.Extensions/IO/StreamingBufferManager.cs:31`. That class's own XML remark
(lines 12–19) and `docs/STREAMING.md:126-129` both state it is intentionally not wired into
`StreamingExecutionStrategy`. The number only leaked into the *core pipeline* narrative at the
two sites above.

### Why it survived

Both inputs land on the same clamped output, so the worked example cannot discriminate its
input. The issue's own table compounds this by **mixing unit bases**: `256 MB → 25,600,000`
is decimal (256 × 10⁶) while `1 GB → 107,374,182` is binary. The code constant is binary. The
issue's proposed replacement ("64 MB → 64,000 rows") only holds for decimal MB; against the
code's convention 64 MiB → 67,108. Any corrected example must be denominated in **bytes**.

### Additional findings beyond the issue

1. **On the `AsStream` path the derivation never runs at all.** `QueryFrame.cs:470` always
   sets `ChunkSize` (parameter default 10,000), and `resolveChunkSize` is
   `context.ChunkSize ?? calculateChunkSize(...)` (`StreamingExecutionStrategy.cs:37`).
   `ChunkSize` is assigned in exactly one place in `src/`. So the budget-derived fallback is
   reachable only via `ExecutionEngine.Execute(plan, context)` with `Strategy = Streaming` and
   `ChunkSize` null. The 1 GB default's only effects on `AsStream` are channel capacity and
   the tracker threshold.

2. **Those effects are also clamped** — the same invisibility trap one level down. At the 1 GB
   default with 10,000-row chunks: `CalculateChannelCapacity` = `clamp(1,073,741,824 /
   1,000,000, 2, 16)` = **16**, and 256 MB would also give 16. `StreamingBudgetTracker` warns
   at 2× budget = **2 GB** (256 MB would give 512 MB). Stating the concrete values is what
   makes the default's real consequence checkable.

3. **Latent overflow in both budget→sizing formulas** (the human elected to fix this in the
   same change). `calculateChunkSize` does `(int)(chunkMemory / estimatedBytesPerRow)` with no
   overflow guard, and no project sets `CheckForOverflowUnderflow`, so the narrowing is
   unchecked. Above ~2.1 TB the truncation wraps and the derivation **inverts**: the largest
   possible budget produces the *smallest* chunk size. `CalculateChannelCapacity` has the
   identical defect. `StreamingStrategy_MemoryBudgetMaxValue_Works`
   (`ExecutionEdgeCaseTests.cs:172-187`) passes because it only asserts `RowCount == 100` —
   the test named for that case does not cover the thing that breaks.

   Verified before/after (every normal value byte-identical; only the wrapping region changes):

   | Input | Old | New |
   |---|---|---|
   | `CalculateChunkSize(1 GB)` | 100,000 | 100,000 |
   | `CalculateChunkSize(67,108,864)` | 67,108 | 67,108 |
   | `CalculateChunkSize(1)` / `(0)` | 1,000 | 1,000 |
   | **`CalculateChunkSize(long.MaxValue)`** | **1,000** | **100,000** |
   | `CalculateChannelCapacity(1 GB, 10_000)` | 16 | 16 |
   | **`CalculateChannelCapacity(long.MaxValue, 1_000)`** | **2** | **16** |

## Proposed changes

### 1. Clamp in `long` space before narrowing (`src/Nivara/Execution/StreamingExecutionStrategy.cs`)

Both bounds sets ([1_000, 100_000] and [2, 16]) fit comfortably in `int`, so clamping in
`long` space and *then* narrowing makes the unchecked cast provably safe.

```csharp
// was: static int calculateChunkSize(long memoryBudget)
internal static int CalculateChunkSize(long memoryBudget)
{
    const long estimatedBytesPerRow = 100;
    var chunkMemory = memoryBudget / 10;
    var calculatedChunkSize = chunkMemory / estimatedBytesPerRow;
    // Clamp in long space: narrowing first wraps above ~2.1 TB and inverts the result.
    return (int)Math.Clamp(calculatedChunkSize, 1000, 100000);
}

internal static int CalculateChannelCapacity(long memoryBudget, int chunkSize)
{
    const long estimatedBytesPerRow = 100;
    var bytesPerChunk = (long)chunkSize * estimatedBytesPerRow;
    if (bytesPerChunk <= 0) return 2;
    // Clamp in long space: narrowing first wraps and collapses capacity to the floor.
    var capacity = memoryBudget / bytesPerChunk;
    return (int)Math.Clamp(capacity, 2, 16);
}
```

`calculateChunkSize` → `CalculateChunkSize` and `private static` → `internal static`, mirroring
its sibling `internal static CalculateChannelCapacity`; internals are already visible to
`Nivara.Tests` via `InternalsVisibleTo` in `src/Nivara/Nivara.csproj`. Sole caller
(`resolveChunkSize`, line 37) updated. `resolveChunkSize` keeps its existing name — the class
already has camelCase privates (`isSuitableForStreaming`, `executeOperationsOnData`).

### 2. Tests pinning the derivation

New `tests/Nivara.Tests/Execution/StreamingChunkSizeTests.cs`
(`Nivara.Tests.Execution`, NUnit 4.x, exact `Assert.That`, no `[TestCase]`):

- `CalculateChunkSize_DefaultOneGigabyteBudget_ClampsToMaximum` → 1 GB = 100,000 — the direct
  gate on the number the docs now state.
- `CalculateChunkSize_BudgetInsideClampRange_MatchesFormulaExactly` → 67,108,864 = 67,108 —
  pins the docs' hand-checkable worked example exactly.
- `CalculateChunkSize_TinyBudget_ClampsToMinimum` → 1 and 0 = 1,000.
- `CalculateChunkSize_MaxValueBudget_ClampsToMaximum` → `long.MaxValue` = 100,000 (fails at
  1,000 without the fix).
- `CalculateChunkSize_GrowingBudget_NeverDecreasesChunkSize` → monotonic sweep across the wrap
  boundary, matching the existing `..._ShrinkingBudget_NeverIncreasesCapacity` style.

`tests/Nivara.Tests/Execution/StreamingBackpressureTests.cs` — add
`CalculateChannelCapacity_MaxValueBudget_ClampsToMaximumSixteen` (the twin regression).

`tests/Nivara.Tests/Execution/ExecutionEdgeCaseTests.cs` is deliberately **not** touched: it is
a `RowCount` smoke test through `Execute` that cannot observe chunk size, and the new unit test
is the real gate. Recorded here so the omission is a decision, not a gap.

### 3. Documentation corrections

**`AGENTS.md:123-124`** — state 1 GB with its `NivaraExecutionContext.cs:17` attribution, name
`StreamingBufferManager` as the 256 MB's owner, and split the chunk-size bullet so `AsStream`'s
10,000-row parameter default is not confused with the budget-derived fallback.

**`docs/STREAMING.md`**

- *§chunkSize semantics* — add that `AsStream` **always** supplies a chunk size, so its
  10,000-row default shadows the budget derivation; point at the fallback's one public route.
- *§Memory budget → chunk size* — state the 1 GB default with attribution and attribute 256 MB
  to `StreamingBufferManager`; replace the clamped decimal-MB example with a byte-denominated
  table (one row strictly inside the clamp, one showing the default honestly hitting the
  ceiling); note integer truncation and that the clamp is applied before the narrowing.
- *§Channel capacity* — add the concrete default-derived values (capacity **16**, tracker
  threshold **2 GB**) and the budgets at which they first move (< 16,000,000 bytes for capacity,
  2,000,000 for the floor).
- *§AC3 resolution* — add the #514 caveat that the derivation's input is unreachable from the
  public API today, cross-linking `docs/ACCELERATION.md:208-224`.

No `CHANGELOG.md` entry — matches the convention of the recent doc-accuracy fixes `b2f42b80`
and `d881740c`.

## Blast radius

- `StreamingExecutionStrategy.CalculateChunkSize` (was `calculateChunkSize`, private static →
  internal static). Callers: `resolveChunkSize` only, which feeds
  `ExecuteCore` (line 124), `executeCoreInternalAsync` (line 273) and `StreamChunksAsync`
  (line 521). All three are inside the same class. Behaviour is unchanged for every budget
  below ~2.1 TB; above it, the result changes from the 1,000-row floor to the 100,000-row
  ceiling, which is the intended monotonic reading.
- `StreamingExecutionStrategy.CalculateChannelCapacity` — callers `CreateBoundChannel`
  (line 105) and tests. Unchanged below the wrap region; `long.MaxValue` budgets go from
  capacity 2 to 16.
- Tests covering this area: `StreamingExecutionStrategyTests`, `StreamingBackpressureTests`,
  `StreamingBudgetDiagnosticTests`, `StreamingBudgetTrackerTests`, `ExecutionEdgeCaseTests`,
  `ExecutionIntegrationTests`, `AsyncStreamingTests`. `StreamingBudgetDiagnosticTests` sets
  `MemoryBudget = 1`, whose derived chunk size (1,000) is unchanged by the fix.
- Docs only: `AGENTS.md`, `docs/STREAMING.md`. No public API surface changes; no behavioural
  change on any path a caller can currently reach with a sane budget.

## Verification steps

1. `dotnet build Nivara.slnx` — clean build, no new warnings.
2. `dotnet test -c Release --filter "FullyQualifiedName~StreamingChunkSize|FullyQualifiedName~StreamingBackpressure"` — new + touched tests green. (Not timing tests, so no `[Category("Performance")]` needed.)
3. `dotnet test -c Release --filter "Category!=Performance"` — full suite green (catches any
   behaviour change from the overflow fix).
4. Re-read `AGENTS.md` + `docs/STREAMING.md` against `NivaraExecutionContext.cs:17` and
   `StreamingExecutionStrategy.cs` to confirm every number now matches the code.

**Ask the human before running any `dotnet test` / long-running verification command.**

## Planned commits

1. `docs: plan #516 in TODO.md`
2. `fix: stop the streaming budget derivations wrapping above ~2.1 TB` — `StreamingExecutionStrategy.cs`, new `StreamingChunkSizeTests.cs`, `StreamingBackpressureTests.cs`
3. `docs: correct the default streaming memory budget to 1 GB (#516)` — `AGENTS.md`, `docs/STREAMING.md`
4. `docs: remove TODO.md — plan executed`

Fix (2) lands before the docs (3) so the documented numbers describe final behaviour. Note the
numbers the docs state (67,108 / 100,000 / 16 / 2 GB) are **unchanged** by the fix, so the two
commits are genuinely independent reasons.

> As each task executes, if you find deferred work or a concern (known limitations, follow-ups,
> refactors) outside this plan, create a tracked issue immediately via
> `gh issue create --repo khurram-uworx/Nivara` and record its number in the log below — do not
> rely on memory or wait until the plan finishes, as compaction during execution can lose it.

## GitHub issues log

- [ ] #514 — no public knob for a streaming memory budget; already open, referenced not duplicated. The derivation's input is unreachable from the public API until it lands.