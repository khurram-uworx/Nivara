# Plan: #167 Fused expression engine — kernel IR + chunked span execution

Branch: `khurram/167`. Source issue: https://github.com/khurram-uworx/Nivara/issues/167
Future-ideas context: https://github.com/khurram-uworx/Nivara/issues/171 (Phase 4 async streaming bridge, enabled by chunked span kernels).

## Problem

`FusedExpressionEvaluator` makes backend decisions inline against the `ColumnExpression` AST
(`FusedExpressionEvaluator.cs:201-252`), so the planner stays coupled to one execution
representation and there is no way to run a fused kernel chunk-at-a-time over span slices
(required by #171). The node-tree fallback (`FusedKernel`) re-walks the AST per element with a
dictionary + `ReadOnlyMemory.Span` lookups (`FusedKernel.cs:110-116`).

## Proposed changes

### 1. Kernel IR — `src/Nivara/Expressions/KernelIR.cs` + `KernelLowerer.cs` (new)

Single planning representation, decoupled from every backend (issue's intent). Post-order flat node list:

```csharp
internal enum KernelOp { Column, Literal, Scalar, Add, Subtract, Multiply, Divide, Modulo,
    Equal, NotEqual, GreaterThan, LessThan, GreaterThanOrEqual, LessThanOrEqual, And, Or, Not }

internal readonly record struct KernelNode(KernelOp Op, int Left, int Right, object? Value, Type ComputeType);
// Column:   Left = leaf index into plan.Columns
// Literal:  Value = original literal value (coerced per-backend)
// Scalar:   Left = operand node index, Value = scalar, ComputeType = promoted type
// Binary/Comparison: Left/Right = child node indices; Not: Left = operand

internal sealed class KernelPlan {
    IReadOnlyList<KernelNode> Nodes;                // post-order
    IReadOnlyList<FusedColumnBinding> Columns;      // leaves
    Type ResultType; bool HasNulls; string Signature;
    bool IsUniformNumeric;                          // generic-math result, != bool, all leaves == ResultType
    bool IsTensorPrimitivesDispatachable;           // IsUniformNumeric && single op in {Add,Subtract,Multiply,Divide}
    int MaxStackDepth;                              // for the span interpreter's stackalloc
}
```

`KernelLowerer.Lower(ColumnExpression, FusedExpressionPlan)` mirrors `BuildNode`'s promotion
logic (`NumericPromoter.GetPromotedType`); literals keep their original value so each backend
coerces once (kills per-eval `Convert.ChangeType` at `FusedKernel.cs:139`). Cache key stays
`plan.Signature`.

### 2. Compiled-from-IR, offset-based — `FusedExpressionEvaluator.BuildCompiledDelegate`

- `BuildNode` consumes `KernelNode`s (same promotion / `BuildComparison` / AndAlso/OrElse logic).
- Loop emits `dest[destStart + i] = value` over `i < count`, leaf access `leaf[i + start]`.
- Cached wrapper becomes `delegate void CompiledFusedInvoke(object[] leafArrays, Array dest, int start, int count, int destStart)`.
- Whole-column = `invoke(leaves, output, 0, length, 0)`; chunked = one call per chunk into a single shared output array (zero per-chunk allocation).
- Output assembly switches to internal `CreateFromOwnedArray` for null-free results (removes one `Create` copy).

### 3. Flat span interpreter — evolve `FusedKernel.cs`

- Keep public `Evaluate<T>(expression, leaves)` and `Execute<T>(expression, leaves, ReadOnlyMemory<T>[] inputs, ReadOnlyMemory<bool>[] masks, Span<T> output, bool[]? outputMask)` as thin adapters (raw-kernel tests keep working), internally lowering to `KernelPlan`.
- New core: `Execute<T>(KernelPlan, ReadOnlySpan<T>[] inputs, ReadOnlySpan<bool>[] masks, Span<T> output, Span<bool>? outputMask)` — hoisted literal coercions, leaf spans loaded once, per-element post-order eval on `stackalloc T[MaxStackDepth]`, inline mask OR. Replaces the recursive node-walk.

### 4. TensorPrimitives single-op backend

Null-free uniform numeric single-op dispatches to `NumericTensorKernels<T>` (generic
`TensorPrimitives.Add/Subtract/Multiply/Divide`; scalar-first Subtract/Divide use the existing
manual `INumber<T>` loops). `Modulo` has no generic `TensorPrimitives` overload in the pinned
10.0.11 package (`Ieee754Remainder` is not `%`) — stays on compiled/span backends.

### 5. Chunked execution

`EvaluateCore(expression, input, int? chunkSize)`; public `Evaluate` passes null (whole-column,
no behavior change); new `internal IColumn EvaluateChunked(expression, input, chunkSize)`.
Chunking = span slices / `start,count` args over the existing contiguous `ColumnStorage<T>`
buffers (issue decision: no Phase A storage rewrite). All three backends are chunk-capable.

Routing (replaces `EvaluateCore` inline decisions):
| Plan shape | Backend |
|---|---|
| null-bearing uniform numeric | span interpreter (mask fused) |
| null-free uniform numeric single-op (Add/Sub/Mul/Div) | TensorPrimitives |
| everything else (null-free chains, bool/comparison, heterogeneous, non-generic-math) | compiled-from-IR |

## Verification

- `dotnet build Nivara.slnx` after each step (ask before `dotnet test`).
- Existing `FusedExpressionEvaluatorTests` green except intended routing changes (see tests step).
- New chunked tests: chunk sizes {1, 2, 3, 511, 512, 1024, len-1, len} bit-identical to
  whole-column for null-free chain / null-bearing chain / Modulo / comparison / heterogeneous.
- Perf scenarios in `tests/Nivara.PerformanceTests/Program.cs`; compare via harness `--compare`.

## Planned commits

1. `docs: plan #167 kernel IR + chunked span execution in TODO.md`
2. `Add kernel IR and lowering for fused expression plans`
3. `Build compiled fused kernels from kernel IR with offset-based loop`
4. `Run uniform numeric plans through flat IR span interpreter`
5. `Dispatch single-op null-free numeric plans to TensorPrimitives`
6. `Execute fused plans chunk-at-a-time over span slices`
7. `Add chunked-execution and backend-routing guardrail tests`
8. `Add chunked and single-op perf scenarios; document #167 in roadmap/CHANGELOG/ADR-004`
9. `docs: remove TODO.md — plan executed`

## Blast radius

- `src/Nivara/Expressions/KernelIR.cs`, `KernelLowerer.cs` (new) — no existing callers.
- `src/Nivara/Expressions/FusedExpressionEvaluator.cs` — cache type changes from
  `Func<object[], int, Array>` to `CompiledFusedInvoke`; `BuildCompiledDelegate`/`BuildNode`
  signature changes; routing change in `EvaluateCore`. Called by `FilterOperation`,
  `SelectOperation` (through `Evaluate`/`EvaluateBoolean`), window materialization,
  `NivaraLinqExtensions`, and perf harness. Public surface unchanged.
- `src/Nivara/Expressions/FusedKernel.cs` — public raw-kernel signatures preserved; internals
  rewritten on IR.
- `tests/Nivara.Tests/Query/FusedExpressionEvaluatorTests.cs` — counter renames
  (`NodeTreePathEvaluationCount` → `SpanKernelPathEvaluationCount`) at lines 67, 331, 434;
  `Evaluate_NullFreeUniformPlan_StaysOnCompiledPath` (:440) now asserts TensorPrimitives
  dispatch (contract intentionally changed by #167).
- `tests/Nivara.PerformanceTests/Program.cs` — adds scenarios; no existing scenario changes.
- No changes to storage, execution strategies, query ops, or AutoDiff.

## Notes / risks

- `leaf[i + start]` bounds-check elimination vs today's `leaf[i]`: validate with the perf
  scenario; if the compiled path regresses, emit a start==0 fast-loop variant.
- `Min`/`Max` reductions exist in `NumericTensorKernels` but the element-wise `TensorPrimitives`
  overloads are not wrapped there — TP backend covers Add/Subtract/Multiply/Divide only.
- Guardrail `MaxAllocationFraction = 1.01`: chunked path must not allocate per chunk (uses one
  shared output array), or the chunked scenario needs its own baseline.

## GitHub issues log

- [ ] #171 — Phase 4 async streaming bridge (context for this plan; not worked here)
- [ ] #155 — span-capable compiled target (superseded in approach: offset-based compiled delegate
      is chunk-capable without spans; keep issue open for the span-interpreter path)
