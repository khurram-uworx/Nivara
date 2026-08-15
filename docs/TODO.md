# Plan: Fused-expression and rank-window bugfix batch (#249, #250, #254, #255)

Branch: `khurram/issues` (off `main`, HEAD `d4dc823` Merge PR #257).
Repo: `khurram-uworx/Nivara` (use `--repo khurram-uworx/Nivara` for `gh`).

## Objective

Address four open, bug-labeled GitHub issues as ONE branch/PR batch, mirroring the
existing `#245–#248` combined-fix precedent in this repo:

- **#249 (medium)** — literal-only fused plans (`Lit(2) * 2`) throw `NotSupportedException`
  wrapped in `QueryExecutionException` instead of producing a constant column.
- **#250 (medium)** — `CoerceLiteral<T>` (FusedKernel.cs:283) is a fragile
  `Convert.ChangeType` + `(T)` + `T.CreateChecked` chain; throws for cross-type pairs on the
  extended numeric domain (int→Half/IntPtr/Int128/UInt128, decimal→Half/Int128, Half→double).
  Also `NumericPromoter` lacks nint/nuint/Int128/UInt128 promotion rules, so `nintCol+2` and
  `Int128Col+5` resolve to `int` (wrong type, silently truncating).
- **#254 (low)** — `RankKernel.Compute` nulls rows with null order keys for ALL rank kinds
  including `RowNumber`; Polars `row_number` numbers every row. No Polars cross-validation
  exists. **Decision: include the minimal Polars fixture cross-validation in this PR.**
- **#255 (low)** — `HydrateWindows` names synthetic columns `__window_<synthetic.Count>`
  without checking user input keys, so a user column named e.g. `__window_0` is shadowed.

## Status

- [ ] Branch created (`khurram/issues`)
- [ ] Write this plan to `docs/TODO.md`
- [ ] Commit plan (`docs: plan fused-expression and rank-window bug batch in TODO.md`)
- [ ] Commit 1 — #249 literal-only fused plans → constant column
- [ ] Commit 2 — #250 CoerceLiteral + extended-domain NumericPromoter promotion
- [ ] Commit 3 — #255 collision-free synthetic window names
- [ ] Commit 4 — #254 RowNumber null-key semantics + docs + updated tests
- [ ] Commit 5 — #254 minimal Polars cross-validation fixtures + C# tests
- [ ] Commit 6 — CHANGELOG update
- [ ] Review docs/TODO.md, confirm all items done, remove it, commit removal
- [ ] Offer push + PR (human confirms; never push without confirmation)

## Grounding

- C# implicit-conversion rules verified via MS Learn (`int→nint`, `uint→nuint` implicit;
  `nint→long`, `nuint→ulong` implicit; `nint+uint→long`, `nint+long→long`; `nint+ulong` /
  `nint+nuint` / `Int128+UInt128` are compile errors → use repo's safe-superset `→double`
  convention, matching existing `ulong+signed→double`).
- net10.0 supports `Convert` self-type only; cross-type `Convert.ChangeType` still throws for
  extended types (AGENTS.md note "boxed Int128/UInt128 are not IConvertible" is outdated for
  net10 self-conversion but true cross-type).
- `GetNumericTypes()` (TypeCompatibilityValidator.cs:229-238) includes
  byte/sbyte/short/ushort/char/int/uint/long/ulong/nint/nuint/Int128/UInt128/float/double/
  Half/decimal — the promotion/coercion domain must cover all of these.
- NivaraTorch fixture pattern (to mirror for #254 cross-validation): Python gen script
  (`samples/NivaraTorch/gen_reference.py`) writes raw .bin + JSON `manifest.json` into
  `samples/data/torch-comparison/`; C# `tests/Nivara.Tests/NivaraTorch/TestHelpers.cs`
  resolves `samples/data/torch-comparison` relative to `TestContext.CurrentContext.TestDirectory`
  via `Path.Combine(..., "..","..","..","..","..","samples","data","torch-comparison")`.

## Background research (verified facts)

### #249 route (verified in current code)

- `Lit(2) * 2` builds `ScalarExpression(Multiply, LiteralExpression(2), 2)`.
- `ExpressionTypeInferer.TryInfer` returns `FusedExpressionPlan` (record, ExpressionTypeInferer.cs:17-22)
  with `Columns.Count == 0` for literal-only expressions (all leaves are literals).
- `FusedExpressionEvaluator.EvaluateCore` (FusedExpressionEvaluator.cs:237-245) throws for
  `plan == null || plan.Columns.Count == 0`:
  message "Expression '{...}' cannot run through the fused evaluator: unsupported operand combination",
  wrapped in `QueryExecutionException`.
  (The issue #249 text mentions a route through `EvaluateTensorPrimitives`/`IsDispatchable`
  — that is a stale snapshot of the code; the real current rejection point is EvaluateCore.)
- `LiteralExpression` passthrough exists at FusedExpressionEvaluator.cs:229-230 using
  `input.Values.FirstOrDefault()?.Length ?? 1` — this is the length convention to reuse.
- Compiled delegate (`EvaluateCompiled`) builds fine with zero leaf params
  (paramTypes = dest + 4 int params + null mask). So literal-only plans can run on the compiled path.
- `ValidateLeafLengths(plan)` (FusedExpressionEvaluator.cs:542) indexes `plan.Columns[0]` —
  must be guarded for zero-column plans.
- `TensorPrimitivesKernel.IsDispatchable` (TensorPrimitivesKernel.cs:192-201) requires a column
  leaf — intentional, do NOT loosen. Constants don't need SIMD.

### #250 facts

- `FusedKernel.CoerceLiteral<T>` at FusedKernel.cs:283-291:
  `Convert.ChangeType` → `(T)(object)` → `T.CreateChecked`. Throws for cross-type extended pairs.
- `FusedKernel.Execute<T>` calls `CoerceLiteral<T>` at FusedKernel.cs:151 (per leaf literal).
- `TensorPrimitivesKernel.TryEvaluate`/`RunChunk` call it at lines 45, 46, 110, 111.
- `NumericPromoter.GetPromotedType` (Helpers/NumericPromoter.cs:17-80) ends in
  `return typeof(int)` fall-through; no nint/nuint/Int128/UInt128 arms.
- 8 call sites (all inside expression subsystem — blast radius confined):
  - KernelLowerer.cs:90 (scalar), :116 (binary)
  - ExpressionTypeInferer.cs:132 (binary), :141 (scalar), :153 (comparison validation)
  - FusedExpressionEvaluator.cs:779 (compiled node), :842 (build comparison)
  - ColumnExpression.cs:494, :702
- `NumericPromoterTests.cs` has NO nint/nuint/Int128/UInt128 cases → adding rules breaks nothing.

### #255 facts

- `HydrateWindows` (FusedExpressionEvaluator.cs:355-380); synthetic name picked at line 360:
  `var name = SyntheticWindowPrefix + synthetic.Count;` with `SyntheticWindowPrefix` = "__window_" (:440).
- The merged dict at FusedExpressionEvaluator.cs:211 overwrites a user column of the same name.
- `MaterializeWindow` key names (:398, :405) go into a fresh internal `keyColumns` dict — NO collision there.
- Only the HydrateWindows site (line 360) needs uniqueness.

### #254 facts

- `RankKernel.Compute` (Tensors/RankKernel.cs:52-131) masks null-order-key rows at :80-95 for ALL kinds.
- All rank surfaces funnel through `RankKernel.Compute`:
  - eager `WindowFrameExtensions.cs:179-246` (helper calls RankKernel.Compute directly)
  - lazy `RankOperations.cs:232`
  - fused `MaterializeWindow` → `RankKernel.Compute` (FusedExpressionEvaluator.cs:410)
  A single kernel fix covers all three surfaces.
- `PartitionedWindowEngine` (Tensors/PartitionedWindowEngine.cs:15-16) keeps null-key rows per
  `NullOrdering` ("SQL-faithful"). That divergence STAYS; document it in #254 docs.
- MultiColumnComparer honors `NullOrdering` for null-key rows (SQL-faithful ordering).

## Planned changes

### Commit 1 — #249: literal-only fused plans → constant column

**File:** `src/Nivara/Expressions/FusedExpressionEvaluator.cs`

- In `EvaluateCore` (around :237-245): replace the blanket
  `if (plan == null || plan.Columns.Count == 0) throw` with:
  - `plan == null` → keep throw (unknown/unlowerable expression).
  - `plan.Columns.Count == 0` → route to `EvaluateCompiled` (or a helper that mirrors the
    LiteralExpression passthrough) with length `input.Values.FirstOrDefault()?.Length ?? 1`.
  - Guard `ValidateLeafLengths(plan)` so it returns early when `plan.Columns.Count == 0`.
- Do NOT touch `TensorPrimitivesKernel.IsDispatchable`.

**Tests** (`tests/Nivara.Tests/Query/FusedExpressionEvaluatorTests.cs`):
- `Lit(2) * 2` → int column, every element 4 (length matches input count / 1 for empty).
- Double literal plan `Lit(2.5) * 2` → double column 5.0.
- Literal-only comparison `Lit("a") == Lit("b")` → bool column.
- Existing `FusedPathEvaluationCount` / `SpanKernelPathEvaluationCount` guardrails: assert the
  literal-only plan lands on the compiled path (constant column), not the tensor-primitives path.

### Commit 2 — #250: CoerceLiteral + extended-domain promotion

**File A:** `src/Nivara/Expressions/FusedKernel.cs` (CoerceLiteral<T>, :283-291)

Rewrite as a runtime type-switch over the boxed literal type, dispatching to typed
`T.CreateChecked(v)` for: byte, sbyte, short, ushort, char, int, uint, long, ulong, nint, nuint,
Int128, UInt128, Half, float, double, decimal. Checked-conversion semantics; no `IConvertible`.
If the boxed value type is not in the domain, keep a clear exception.

**File B:** `src/Nivara/Helpers/NumericPromoter.cs` (GetPromotedType)

Add explicit arms BEFORE the `int` fall-through (C# §10.2.3 implicit-conversion matrix, using
repo safe-superset `→double` convention for compile-error pairs):

- nint + (sbyte/byte/short/ushort/int/char) → nint
- nint + uint → long; nint + long → long
- nint + ulong → double (superset); nint + nint → nint
- nint + nuint → double (superset)
- nint + (float/double/Half) → double (repo superset for float/Half); nint + decimal → decimal
- nuint + (sbyte/byte/short/ushort/int/char) → long (signed context, mirroring nuint+int→long? VERIFY: C# nuint+int → compile error; repo superset → long to preserve sign) — see note
- nuint + (uint/ulong/ushort/byte) → nuint; nuint + nuint → nuint
- nuint + long → double (superset); nuint + (float/double/Half) → double; nuint + decimal → decimal
- Int128 + (any integer/char) → Int128; Int128 + Int128 → Int128
- Int128 + (float/double/Half) → double; Int128 + decimal → decimal; Int128 + UInt128 → double (superset)
- UInt128 + (byte/ushort/uint/ulong) → UInt128; UInt128 + (sbyte/short/int/long/char) → Int128
- UInt128 + nint → Int128 (if nint fits — C#: nint+UInt128 error; use superset Int128 if representable, else double — prefer Int128 for fit) — see note
- UInt128 + (float/double/Half) → double; UInt128 + decimal → decimal; UInt128 + UInt128 → UInt128
- Half + (float/double) → double; Half + decimal → decimal; Half + Half → Half (verify current behavior)

> NOTE (fill exact table at implementation time with MS Learn §10.2.3. Verify each pair against
> C# semantics; deviations use the repo's documented safe-superset convention. If a cell is
> genuinely ambiguous, choose the WIDEST safe type and cover with a test.)

**Tests:**
- `tests/Nivara.Tests/Helpers/NumericPromoterTests.cs`: new extended-pair cases
  (nint+int→nint, Int128+int→Int128, UInt128+int→Int128, nint+uint→long, nint+nuint→double,
  Int128+UInt128→double, nint+long→long, nint+decimal→decimal, nint+Half→double, etc.).
- `tests/Nivara.Tests/Query/FusedExpressionEvaluatorTests.cs`: uniform nint/Int128/Half plans
  with int literals evaluate correctly; cross-type literal coercion no longer throws
  (e.g. `Int128Col + 5`, `nintCol + 2`, `HalfCol + 1`); assert result column types.

### Commit 3 — #255: collision-free synthetic window names

**File:** `src/Nivara/Expressions/FusedExpressionEvaluator.cs`

- Add a helper (e.g. `NextWindowName(IReadOnlySet<string> inputKeys, ...)`) that picks the first
  index `i >= 0` where `SyntheticWindowPrefix + i` is not already a user input key
  (OrdinalIgnoreCase compare) and not already used by a synthetic column in this plan.
- Use it at the HydrateWindows site (line 360). MaterializeWindow untouched.

**Tests** (`tests/Nivara.Tests/Query/WindowExpressionEvaluationTests.cs` or
`FusedExpressionEvaluatorTests.cs`):
- Frame with a user column literally named `__window_0`; expression uses both a window function
  and `Col("__window_0")`. Assert the user column keeps its values and the window column is
  distinct and correct (no shadowing).
- Variant where user column `__window_0` AND `__window_1` exist → synthetic picks `__window_2`.

### Commit 4 — #254: RowNumber null-key semantics + docs + updated tests

**File A:** `src/Nivara/Tensors/RankKernel.cs`

- In `Compute` (:80-95): when `kind == RankKind.RowNumber`, do NOT mask rows with null order keys
  and do NOT exclude them from numbering — include them in the sort (ordered per
  MultiColumnComparer.NullOrdering) and number sequentially, resetting per partition.
- Rank / DenseRank / PercentRank keep null-out semantics unchanged.
- Update the class/method doc comment (:36-38) to document the divergence explicitly.

**File B:** `docs/LINQ.md` (:246-256 and :300-302)
- Document: RowNumber numbers every row (null order keys included, SQL-faithful ordering);
  rank/dense_rank/percent_rank still null out rows with null order keys. Note the residual
  divergence between RankKernel and PartitionedWindowEngine.

**File C:** `docs/plan/POLARS-ROADMAP.md` (:98-109)
- Mark the row_number parity item done / update status line.

**Tests:**
- `tests/Nivara.Tests/Tensors/RankFunctionsTests.cs`:
  - Rewrite `Rank_NullOrderKey_NullOutput_ExcludedFromNumbering` (:141-151) — RowNumber now
    numbers null-key rows; rank family still nulls.
  - Update the null-key property test (~:194-253) accordingly.
  - Add dedicated null-key RowNumber tests: no partition (global numbering), per-partition reset,
    mixed null/non-null keys, null-first vs null-last ordering.
- `tests/Nivara.Tests/Tensors/RankFunctionsFrameTests.cs` and
  `tests/Nivara.Tests/Query/RankOperationTests.cs`: verify same new semantics on frame + query
  surfaces (they route through RankKernel.Compute).

### Commit 5 — #254: minimal Polars cross-validation

**New files:**
- `samples/NivaraWindow/gen_reference.py` — Python script using `polars` (mirrors
  `samples/NivaraTorch/gen_reference.py` structure, including the regeneration header comment)
  computing row_number/rank/dense_rank/percent_rank outputs over fixed input data
  (with partitions, order keys, nulls) → writes `samples/data/polars-window/manifest.json`
  (JSON manifest containing input arrays + null masks + expected output arrays per case).
- `samples/NivaraWindow/requirements.txt` — `polars`.
- `samples/data/polars-window/manifest.json` — committed generated fixtures.

**New test:** `tests/Nivara.Tests/Query/PolarsWindowCrossValidationTests.cs` (or under a new
`tests/Nivara.Tests/NivaraWindow/` folder):
- Load manifest via path resolution mirroring TestHelpers.cs
  (`Path.Combine(TestContext.CurrentContext.TestDirectory, "..","..","..","..","..","samples","data","polars-window")`).
- For each case: build the partition/order inputs, run `RankKernel.Compute`, compare output
  values (and null masks for rank family) against the Polars fixture.
- Keep the matrix small (~6-10 cases): global row_number, partitioned row_number, null keys in
  order-by (null-first and null-last), rank/dense_rank/percent_rank with nulls, all kinds on a
  simple sorted frame. No Python toolchain in the test run — fixtures are committed.

### Commit 6 — CHANGELOG

- Update `CHANGELOG.md` mirroring the prior
  "docs: changelog for fused-kernel and window fixes (#245-#248)" commit: summarize the batch
  (#249, #250, #254, #255) with behavior notes (RowNumber null-key semantics change is the one
  user-visible behavior change).

## Verification

- `dotnet build Nivara.slnx` before each commit (fast check).
- Ask the human before running `dotnet test` (AGENTS.md rule). When approved, run the affected
  fixtures: FusedExpressionEvaluatorTests, NumericPromoterTests, WindowExpressionEvaluationTests,
  RankFunctionsTests, RankFunctionsFrameTests, RankOperationTests, PolarsWindowCrossValidationTests,
  plus the broader Query suite.
- `git status` + `git diff` before each commit; stage only relevant files.

## Blast radius

- **#249/#250/#255** — confined to the fused-expression subsystem:
  - Files: FusedExpressionEvaluator.cs, FusedKernel.cs, KernelLowerer.cs, ExpressionTypeInferer.cs,
    TensorPrimitivesKernel.cs (no change), NumericPromoter.cs, ColumnExpression.cs (no change).
  - Call sites of GetPromotedType: 8 (all in expression subsystem — listed above).
  - Downstream: any query using literal expressions / scalar expressions / window expressions.
  - Tests covering: FusedExpressionEvaluatorTests, NumericPromoterTests, LinqQueryTests
    (SortByExpression(Lit(5)) at :289), WindowExpressionEvaluationTests.
- **#254** — RankKernel.cs; all three rank surfaces (eager/lazy/fused) route through it.
  - Tests: RankFunctionsTests, RankFunctionsFrameTests, RankOperationTests, RankKernelTests (if any).
  - Docs: docs/LINQ.md, docs/plan/POLARS-ROADMAP.md.
  - Behavior change visible to users: RowNumber now numbers rows with null order keys.
- **No overlapping edits across the four commits.**

## GitHub issues log

- [ ] #249 — literal-only fused plans throw (being fixed in this batch)
- [ ] #250 — CoerceLiteral + NumericPromoter extended domain (being fixed in this batch)
- [ ] #254 — RowNumber null-key semantics + Polars cross-validation (being fixed in this batch)
- [ ] #255 — synthetic __window_<n> shadowing (being fixed in this batch)
- (new issues created during execution will be appended here immediately via
  `gh issue create --repo khurram-uworx/Nivara`)

## Reminder (per iterative-work skill)

As each task executes, if you find deferred work or a concern (known limitations, follow-ups,
refactors) that is outside the current plan, create a tracked GitHub issue IMMEDIATELY via
`gh issue create --repo khurram-uworx/Nivara` and record its number in the GitHub issues log
above. Don't rely on memory or wait until the plan finishes — compaction can lose it.
