# REVIEW — Phases 1–3 (POLARS-ROADMAP) findings and issue tracking

**Status:** review complete · **Scope:** Phases 1–3 of `docs/plan/POLARS-ROADMAP.md` · **Date:** 2026-08-15
**Tracking:** each finding has a GitHub issue — see the [Issues](#issues) table. Work hand-off happens via those issues.

---

## 1. Purpose

Review of the delivered work for the first three roadmap phases before Phase 4 (async-first streaming) is planned:

- **Phase 1 — Unified typed expression engine** (delivered #153)
- **Phase 2 — Kernel fusion + generic-math collapse** (delivered, incl. kernel IR + chunked execution #167)
- **Phase 3 — Window functions** (delivered, core set #135, rank family #156, WindowSpec #162)

This review verifies the work against the roadmap acceptance criteria, records issues found, and links each to a tracked GitHub issue so the engineering team (or an agent) can pick them up independently.

## 2. Overall verdict

Phases 1–3 are in good shape and the architecture is sound. The typed/fused expression engine fully purged the boxed interpreter; the window kernels are shared between eager and lazy paths; semantics are documented; core null/`minPeriods` behavior is correct. **However, four high-priority correctness bugs were found** that should be fixed before Phase 4 planning, because two of them (streaming/parallel window execution) sit exactly on the Phase 4 chunked-execution path.

### Delivered highlights

- **Phase 1:** fused evaluator replaced the boxed interpreter — no `object?` per-element dispatch on the numeric/vectorizable query path; guardrail path counters exist (`FusedExpressionEvaluator.cs:30-33`).
- **Phase 2:** kernel IR (`KernelIR.cs`) + three backends (flat span interpreter, `TensorPrimitives` SIMD, offset-based compiled delegate); chunked execution is bit-identical to whole-column; `OrderBy` computed keys route through the fused evaluator; generic-math collapse complete in `NivaraColumn.cs`.
- **Phase 3:** rolling/cumulative/shift/rank delivered with eager/lazy parity, `WindowExpression` composition with the fused engine, and correct `minPeriods`/null semantics; `WindowSpec`/`Over()`/partitioned windows added (#162).

## 3. Issues (summary table)

| # | Severity | Area | Finding | Issue |
| --- | --- | --- | --- | --- |
| 1 | 🔴 High | Phase 3 + Phase 4 | Streaming/parallel execution of window expressions computes windows per chunk/slice — incorrect results at boundaries | [#245](https://github.com/khurram-uworx/Nivara/issues/245) |
| 2 | 🔴 High | Phase 2 | Compiled-delegate cache key collides across literal types (`0.1f` vs `0.1`, `1.1m` vs `1.1`) | [#246](https://github.com/khurram-uworx/Nivara/issues/246) |
| 3 | 🔴 High | Phase 2 | Backend divergence at masked positions: compiled path evaluates nulls (`DivideByZeroException`), span kernel short-circuits | [#247](https://github.com/khurram-uworx/Nivara/issues/247) |
| 4 | 🔴 High | Phase 3 | `RollingSum`/`RollingMean`/`CumulativeSum` prefix-sum accumulator silently overflows in element type `T` | [#248](https://github.com/khurram-uworx/Nivara/issues/248) |
| 5 | 🟡 Medium | Phase 2 | Literal-only fused plans (`Lit(2) * 2`) throw `NotSupportedException` instead of a constant column | [#249](https://github.com/khurram-uworx/Nivara/issues/249) |
| 6 | 🟡 Medium | Phase 2 | `CoerceLiteral` relies on fragile `Convert.ChangeType` chain (throws for `int→Half`/`Int128`/`nint`) | [#250](https://github.com/khurram-uworx/Nivara/issues/250) |
| 7 | 🟡 Medium | Phase 3 | Window kernels allocate heavily on hot paths (per-partition `OrderBy`, 4–6 full arrays, no pooled scratch) | [#251](https://github.com/khurram-uworx/Nivara/issues/251) |
| 8 | 🟡 Medium | Phases 2–3 | Test coverage gaps (streaming-vs-eager chunk equivalence, all-null, window>length, rolling min/max property tests) | [#252](https://github.com/khurram-uworx/Nivara/issues/252) |
| 9 | 🟢 Low | Phase 2 | `OperationFusionRule` is dead code: `FuseOperations` always returns `null` but rule is registered by default | [#253](https://github.com/khurram-uworx/Nivara/issues/253) |
| 10 | 🟢 Low | Phase 3 | `RowNumber` nulls rows with null order keys (Polars divergence); no Polars cross-validation exists | [#254](https://github.com/khurram-uworx/Nivara/issues/254) |
| 11 | 🟢 Low | Phase 3 | Synthetic `__window_<n>` hydration column can shadow a user column of the same name | [#255](https://github.com/khurram-uworx/Nivara/issues/255) |
| 12 | 🟢 Low | Docs | `POLARS-ROADMAP.md` cites `RankFunctions.cs`/`RankOperation.cs` (don't exist); AGENTS.md Int128 note outdated | [#256](https://github.com/khurram-uworx/Nivara/issues/256) |

## 4. Detailed findings

### 🔴 4.1 — Window expressions are evaluated per chunk/slice under streaming and parallel strategies (#245)

**Severity:** High · **Issue:** [#245](https://github.com/khurram-uworx/Nivara/issues/245)

**Symptom:** `Select`/`Filter` ops whose expressions contain a `WindowExpression` return incorrect values at chunk/slice boundaries.

- Streaming: `NonStreamableOperations` (`StreamingExecutionStrategy.cs:8`) does **not** include `Select`; `isSuitableForStreaming` (:10-16) inspects only the top-level op type, so a `Select` with a window is streamed and `SelectOperation.Execute(chunk)` runs windows per chunk (`SelectOperation.cs:93-107`).
- Parallel: `executeFilterParallelSync` runs `FilterOperation.Execute(subset)` per row-range slice (`ParallelExecutionStrategy.cs:48-55`), so filter predicates containing windows are evaluated per slice.

**Root cause:** `FusedExpressionEvaluator.ContainsWindowExpression` is private (`FusedExpressionEvaluator.cs:331-348`); the strategy layer cannot detect that an op's expression is not chunk-sliceable.

**Why it matters for Phase 4:** Phase 4 makes streaming async-native and chunk-capable end to end; this bug is the exact class of correctness failure Phase 4 would amplify. Must be fixed before/during Phase 4 planning.

### 🔴 4.2 — Compiled-delegate cache key collides across literal types (#246)

**Severity:** High · **Issue:** [#246](https://github.com/khurram-uworx/Nivara/issues/246)

**Symptom:** `FormatValue` (`ExpressionTypeInferer.cs:181-190`) formats literals with `ToString(null, CultureInfo.InvariantCulture)`, losing the runtime type — `0.1f` and `0.1` both become `"0.1"`, `1.1m` and `1.1` both become `"1.1"`. The compiled-delegate cache keyed by `plan.Signature` (`FusedExpressionEvaluator.cs:278`) then reuses a semantically wrong delegate when a query repeats the same textual shape with a different literal type.

### 🔴 4.3 — Backend divergence at masked positions (#247)

**Severity:** High · **Issue:** [#247](https://github.com/khurram-uworx/Nivara/issues/247)

**Symptom:** The compiled path evaluates every position then ORs masks separately (`FusedExpressionEvaluator.cs:300`); the span kernel short-circuits masked positions in one pass (`FusedKernel.cs`). So `decimalCol / intCol` where `intCol` has nulls → `DivideByZeroException` on the compiled path (masked positions read `default` = `0`), correct `null` on the span path. Same expression, different result per backend. Compiled path also stores real computed values at masked positions while the span kernel stores `default(T)`.

### 🔴 4.4 — Prefix-sum accumulator can silently overflow (#248)

**Severity:** High · **Issue:** [#248](https://github.com/khurram-uworx/Nivara/issues/248)

**Symptom:** `RollingSum`/`RollingMean` compute `windowSum = prefixSum[i] - prefixSum[lo-1]` in element type `T` (`WindowFunctions.cs:114,121,158`) via `buildPrefix` (:349-371); cumulative sum/product also unchecked (:337-340). A prefix sum accumulates every prior element, so it can wrap where a per-window sum would not. Existing property tests use values too small to catch it.

### 🟡 4.5 — Literal-only fused plans throw (#249)

**Severity:** Medium · **Issue:** [#249](https://github.com/khurram-uworx/Nivara/issues/249)

`Lit(2) * 2` routes to the TensorPrimitives backend, which requires a column leaf (`TensorPrimitivesKernel.cs:33-46`) → `NotSupportedException` → `QueryExecutionException`. Legacy produced a constant column.

### 🟡 4.6 — `CoerceLiteral` fragile conversion chain (#250)

**Severity:** Medium · **Issue:** [#250](https://github.com/khurram-uworx/Nivara/issues/250)

`CoerceLiteral<T>` (`FusedKernel.cs:283`) chains `Convert.ChangeType` + `(T)` + `T.CreateChecked`; `ChangeType` throws for cross-type conversions in the extended numeric domain (`int→Half`, `int→Int128`, `int→IntPtr`, `decimal→Half`, etc.). Mitigated in practice by `NumericPromoter`, but remains a landmine for uniform `nint`/`Int128`/`Half` plans.

### 🟡 4.7 — Window-kernel hot-path allocations (#251)

**Severity:** Medium · **Issue:** [#251](https://github.com/khurram-uworx/Nivara/issues/251)

Per-partition LINQ `OrderBy` in `RankKernel` (:93-95) and `PartitionedWindowEngine`; 4–6 full-array allocations per rolling-kernel call (`WindowFunctions.cs:274-307,349-371,377`); several full-column copies per `PartitionedWindowEngine` call (including reflection dispatch `ColumnFilterHelper.cs:35-40`); per-row `object?[]` keys in `GroupByOperation.CreateGroupsInternal` (:353). The roadmap's "pooled scratch" risk note is unaddressed.

### 🟡 4.8 — Test coverage gaps (#252)

**Severity:** Medium · **Issue:** [#252](https://github.com/khurram-uworx/Nivara/issues/252)

Missing: streaming-vs-eager chunk equivalence (would catch #245); all-null inputs; window-larger-than-data; Shift boundary periods (0/±length); rolling mean/min/max property tests; partitioned rolling property tests; float-vs-double/decimal-vs-double literal collision; heterogeneous division with null divisor; literal-only plans; masked-position backing values.

### 🟢 4.9 — `OperationFusionRule` is dead code (#253)

**Severity:** Low · **Issue:** [#253](https://github.com/khurram-uworx/Nivara/issues/253)

Registered as a default rule (`OptimizationRule.cs:253`) but `FuseOperations` always returns `null` (`OperationFusionRule.cs:109-115`) → no-op plan + misleading "fused" diagnostics. `GetReferencedColumns` is dead code with an always-true check.

### 🟢 4.10 — `RowNumber` Polars divergence and no cross-validation (#254)

**Severity:** Low · **Issue:** [#254](https://github.com/khurram-uworx/Nivara/issues/254)

`RowNumber` nulls rows with null order keys (`RankKernel.cs:37-38`); Polars `row_number` numbers every row. Also no automated Polars cross-validation exists anywhere in `tests/` (grep for "Polars" → zero matches); the NivaraTorch `gen_reference.py` pattern is available for reuse.

### 🟢 4.11 — Synthetic `__window_<n>` column name collision (#255)

**Severity:** Low · **Issue:** [#255](https://github.com/khurram-uworx/Nivara/issues/255)

Window hydration materializes into `__window_<n>` synthetic columns (`FusedExpressionEvaluator.cs:355-382`); a user column with the same name would be shadowed in the combined input dictionary.

### 🟢 4.12 — Docs drift (#256)

**Severity:** Low · **Issue:** [#256](https://github.com/khurram-uworx/Nivara/issues/256)

`POLARS-ROADMAP.md:98` cites `RankFunctions.cs`/`RankOperation.cs` (actual: `RankKernel.cs`/`RankOperations.cs`); AGENTS.md "boxed Int128/UInt128 are not IConvertible" is outdated for net10.0 self-conversion.

## 5. Suggested sequencing

Proposed order of work (each maps to a tracked issue; can be handed to engineering or an agent independently):

1. **#245** — Streaming/parallel window execution (correctness, blocks Phase 4). Add a chunk-equivalence property test.
2. **#247** — Masked-position backend divergence (correctness; cross-backend test).
3. **#246** — Literal-type cache collision (correctness; regression test).
4. **#248** — Prefix-sum overflow (correctness; regression test).
5. **#249 / #250** — Literal-only plans and `CoerceLiteral` hardening (Medium correctness).
6. **#252** — Test coverage gaps (tests that would have caught several of the above).
7. **#251** — Window-kernel allocation reduction (performance polish).
8. **#253 / #255 / #256** — Housekeeping (dead code, name collision, docs drift).
9. **#254** — Polars-parity polish (`RowNumber`, cross-validation fixtures).

After these are resolved (or at least the four High items), Phase 4 planning should proceed. The chunk-equivalence test in #245 doubles as the Phase 4 streaming acceptance criterion (`POLARS-ROADMAP.md:131`).

## Related documents

- `docs/plan/POLARS-ROADMAP.md` — the roadmap this review gates.
- `docs/LINQ.md` — plan-layer and query-engine specification (window semantics documented here).
- `AGENTS.md` — repo guidance; also the source of the outdated Int128 note (#256).
