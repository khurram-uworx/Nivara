# TODO — Issue #137: BFloat16 AutoDiff kernels (net11 migration)

**Issue:** https://github.com/khurram-uworx/Nivara/issues/137 (`enhancement`, `future`)
**Branch:** `khurram/137`

## Problem

Issue #137 deferred BFloat16-typed AutoDiff work until the repo targeted .NET 11.
The net11 migration has landed (all projects target `net11.0`; Tensors
`11.0.0-preview.7`). On .NET 11, `System.Numerics.BFloat16` implements
`IBinaryFloatingPointIeee754<BFloat16>` (verified via Microsoft Learn), so it
natively satisfies the AutoDiff constraint `IFloatingPointIeee754<T>` — every
generic op already compiles against it. What remains is admitting it at the
runtime gates, testing it, and documenting the decision.

## Blast radius

- **Changed files (core):**
  - `src/Nivara/AutoDiff/Utilities/TypeValidator.cs` — add BFloat16 to `IsSupportedType`/`GetSupportedTypes`
  - `src/Nivara/AutoDiff/Extensions/NivaraAutoGradExtensions.cs` — BFloat16 arm in `ToReverseGradTensorsAuto`
- **Optional tuned arms (defer unless needed):** `RMSNormKernel.cs`, `Adam.cs`, `AdamW.cs` Half-style paths
- **Tests:** new `tests/Nivara.Tests/AutoDiff/BFloat16Tests.cs`; existing AutoDiff tests unaffected (no behavior change for float/double/Half)
- **Docs:** `docs/TENSORS.md`, `docs/AUTODIFF.md`, `AGENTS.md`, `CHANGELOG.md`, `ExpressionTypeInferer` XML comment
- **Downstream callers of the gates:** `TypeSafetyTests`, frame→tensor conversion paths. No public contract removed; purely additive.

## Plan / commit list

1. `docs: plan BFloat16 AutoDiff kernel support in TODO.md` ← this commit
2. **Gates:** admit BFloat16 in `TypeValidator` (+ `ToReverseGradTensorsAuto` arm, doc-comment updates). Verify build.
3. **Tests:** `BFloat16Tests.cs`:
   - TypeValidator accepts BFloat16; `GetSupportedTypes` contains it
   - forward/backward parity vs inline float references (bf16 ≈ 8-bit mantissa → rel tol ~1e-2): elementwise ops, Softmax/Mean, MatMul + Linear module
   - optimizer smoke: SGD + Adam step inside `using GradientUtils.Grad()`; loss decreases, grads finite
   - inference-default guard: no graph nodes outside `Grad()`
   - frame boundary: `NivaraFrame` BFloat16 column → `ToReverseGradTensorsAuto`
4. **Loader check:** verify `SafeTensorsLoader.ConvertBF16<BFloat16>` round-trips losslessly (BF16→F32→BF16 is exact) — expected zero code change; note in docs.
5. **Docs:** TENSORS.md type-support note, AUTODIFF.md type section, AGENTS.md known-issue entry, CHANGELOG, stale `ExpressionTypeInferer` XML comment wording.
6. Review branch vs issue acceptance criteria; remove TODO.md when all items done.

## Decisions

- BFloat16 **joins** the `IFloatingPointIeee754<T>` runtime-admitted set (it satisfies the interface natively at net11).
- Tuned RMSNorm/Adam/AdamW BF16 arms **deferred** behind a benchmark follow-up issue (generic double fallback is correct; no HW BF16 SIMD in TensorPrimitives yet).
- Out of scope (separate issues if ever wanted): fused-expression engine `IsGenericMathType`, DataFrame numeric promotion rules, Parquet/Arrow BF16 interop.

## Verification

- Build: `dotnet build Nivara.slnx`
- Tests: ask before running `dotnet test` (per AGENTS.md); target `Nivara.Tests` AutoDiff category first.

## GitHub issues log

- [ ] #137 — parent issue: BFloat16 AutoDiff kernels (this plan executes it)
- *(create follow-ups here as they are discovered during execution)*

---

## Deferred ledger — CodeMemory MCP issues (external filing)

**Do NOT act on this during the #137 plan.** After the plan executes, file these
experienced problems as one GitHub issue at https://github.com/khurram-uworx/CodeMemory,
then delete this section together with TODO.md. Recorded verbatim from the session
(2026-08-22, Nivara repo, v0.6.0+12f28b5c31f841e23383551e925b8ab979b):

1. **`find_related_code` silently returns `[]` for valid method symbol paths.**
   Call: `find_related_code({ symbolPath: "Nivara.AutoDiff.Utilities.TypeValidator.IsSupportedType", relationType: "references" })` → `[]`.
   The same class-level lookup (`get_edit_context` on `Nivara.AutoDiff.Utilities.TypeValidator`)
   *did* return caller/test relationships, so the data exists. Expected either results or a
   "symbol not found (+ suggestions)" diagnostic instead of a silent empty array.
2. **SQL parser fails on LIKE patterns containing `/`.**
   - `WHERE FilePath LIKE '%Optimizer/SGD.cs%' AND Kind='Method'` → `Parse error: Expected a SQL statement, found SymbolRecord, Line: 1, Col: 22`
   - `WHERE FilePath LIKE '%Nn/Module.cs%' AND Kind='Method'` → `Parse error: Expected Expected an expression, found: Identifier { Ident = FROM }`
   Nearly identical queries *without* `/` (e.g. `LIKE '%ReverseGradOperations.cs%'`) parse fine,
   so the slash inside the string literal seems to break the tokenizer/parser.
3. **`COUNT(*)` aggregates unsupported, with cryptic errors.**
   `SELECT COUNT(*) AS tpCalls FROM SymbolRecord WHERE FullName LIKE '%TensorPrimitives%'`
   → `Parse error: Expected (, found EOF`. Either support aggregates or return a clear
   "aggregates are not supported" message.
4. **Parser error messages are internal dumps** (`Expected Expected an expression, found:
   Identifier { Ident = FROM }`) — duplicated "Expected", no position context relative to the
   query text. Hard to self-correct a query from these.
5. **Improvement idea (semantic search):** searches for implementation code
   ("BFloat16 widening SafeTensorsLoader") returned mostly .md docs (TENSORS/AUTODIFF/
   CHANGELOG/AGENTS) and sample files; the actual kernel (`SafeTensorsLoader.ConvertBF16<T>`)
   only surfaced via SQL. A `codeOnly` filter or doc-downweighting would help agent workflows.

