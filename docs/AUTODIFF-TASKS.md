# AutoDiff Tasks

This is the implementation task list derived from
[`AUTODIFF-SUGGESTIONS.md`](AUTODIFF-SUGGESTIONS.md). Keep
[`IDEA-AUTODIFF.md`](IDEA-AUTODIFF.md) as historical context; use this file for
current actionable work.

## Ground Rules

- Keep AutoDiff limited to `float` and `double` unless there is a documented
  gradient policy for another type.
- Preserve null-mask semantics in every operation.
- Add tests with every behavior change: forward values, backward gradients,
  null masks, and shape/error cases where relevant.
- Keep examples grounded in APIs that exist. Do not document optimizer, layer,
  model, dataloader, or training-loop workflows until implemented.
- Do not introduce a Keras-like API as part of the near-term tasks.

## Task 1: Keep Documentation Current

Status: ongoing.

Scope:
- Keep `src/Nivara.Extensions/AutoDiff/README.md` aligned with implemented
  operations, utilities, integration helpers, and limitations.
- Keep `EXAMPLES.md` concise and grounded in existing APIs.
- Link broader follow-ups back to this file instead of adding aspirational code
  examples.

Validation:
- Confirm referenced files and links exist.
- Confirm examples compile by inspection against current public APIs.
- Run `dotnet test tests\Nivara.Tests\Nivara.Tests.csproj --filter FullyQualifiedName~AutoDiff --no-restore`
  when examples or documented API names change.

## Task 2: Improve Shape Ergonomics

Problem:
- `MatMul` and `Transpose` operate on flattened `GradTensor<T>` values with
  explicit dimension arguments.
- `GradTensor<T>` does not carry shape metadata, so matrix-heavy workflows are
  easy to misuse.

Design decision (Phase B scope):
- Add `Shape` (`int[]`) and `Rank` properties to `GradTensor<T>` directly.
- Refactor `MatMul`/`Transpose` to infer dimensions from tensor shapes, removing
  explicit `aRows, aCols, bCols` parameters.
- See plan: Phase B1 in execution plan.

Possible implementation paths (historical):
- Add a small shape-aware wrapper for AutoDiff matrix values.
- Or add named factory/helper methods that make flattened matrix conventions
  explicit without changing `GradTensor<T>`.

Acceptance criteria:
- Matrix examples clearly state row-major flattened layout.
- Invalid dimensions fail with clear error messages.
- Tests cover successful `MatMul`/`Transpose`, mismatched dimensions, backward
  gradients, and null-mask behavior.

Out of scope:
- Full tensor shape algebra.
- Implicit broadcasting.
- GPU or accelerator support.

## Task 3: Add Minimal Optimizer Primitives

Problem:
- Users can compute gradients, but parameter updates are still manual.

Initial scope:
- Add a minimal SGD-style update helper after gradient behavior and ownership
  semantics are settled.
- Keep it separate from layers, models, datasets, and training loops.

Design decision (Phase C scope):
- `SgdUpdate` returns new `GradTensor<T>` values (immutable-style, return-new-tensor).
  Consistent with C#/LINQ chainability and avoids side-effect surprises.
- Caller owns the new tensor and can dispose the old one.
- Null gradients: skip update positions where gradient is null (no change).
- See plan: Phase C2 in execution plan.

Open design points (historical):
- Decide whether updates mutate parameters, return new tensors, or operate on
  `NivaraColumn<T>` data through a new helper.
- Define disposal/resource ownership for updated parameters and gradients.
- Decide how null gradients affect updates.

Acceptance criteria:
- A tiny scalar-loss example can compute gradients and apply one explicit update
  step without introducing a model API.
- Tests cover `float` and `double`, null gradients, missing gradients, and
  repeated zero-grad/update cycles.

Out of scope:
- Adam, RMSProp, schedulers, parameter groups, mixed precision, and `Fit()`.

## Task 4: Reduce Allocation-Heavy Paths

Problem:
- Some AutoDiff internals allocate temporary arrays to extract column data or
  pass gradients through object columns.

Initial investigation:
- Review allocation sites in `GradOperations`, `ComputationGraph`, and
  `GradientUtils`.
- Identify hot paths where existing Nivara tensor/span or pooled-buffer patterns
  can be used safely.
- Preserve null-mask behavior before optimizing.

Acceptance criteria:
- Add focused benchmarks or diagnostics only where they will guide a concrete
  implementation choice.
- Replace high-impact temporary arrays with zero-copy or pooled-buffer paths
  when safe.
- Tests remain green for AutoDiff and null handling.

Out of scope:
- Rewriting core Nivara storage.
- Claiming AutoDiff is performance-final.

## Task 5: Expand Operation Coverage With Tests

Problem:
- The current operation set is useful but small.

Rules for adding an operation:
- Add forward-value tests.
- Add backward-gradient tests.
- Add null-mask propagation tests.
- Add shape/error tests when the operation has dimensional constraints.
- Update README and examples only after the API exists.

Candidate operations:
- Unary math operations where gradients are well-defined for `float` and
  `double`.
- Additional reductions with clear null and gradient policies.
- Matrix helpers only after shape ergonomics improve.

Out of scope:
- Broad numeric type support.
- Implicit broadcasting unless it is separately designed and tested.

## Task 6: Keep Type Support Explicit

Status: Phase A2 completed (June 2026).

Problem:
- Public generic constraints may suggest broader support than the runtime allows.

Phase A2 changes:
- `TypeValidator.ValidateNumericType<T>()` now throws `TypeValidationException` (was `AutoGradException`)
  — provides `ExpectedType`/`ActualType` for better diagnostics.
- Added type-safety tests for `long` and `decimal` to verify clear rejection messages.
- Existing error messages already explicitly state "only float, double supported".

Scope:
- Keep documentation explicit that only `float` and `double` are supported.
- Ensure error messages for unsupported types remain clear.
- Avoid examples using integral AutoDiff tensors.

Acceptance criteria:
- Type-safety tests cover unsupported numeric types.
- README and examples do not imply gradients for integral types.

## Task 7: Strengthen Null Gradient Policy

Status: Phase A1 completed (June 2026).

Problem:
- Null masks are preserved, but every new operation needs an explicit gradient
  policy for null positions.
- **`MatMul` and `Transpose` had NO null mask handling** — fixed in Phase A1.

Scope:
- Document operation-specific null behavior when it differs from simple mask
  propagation.
- Add null-mask tests for each operation added or changed.
- Keep boolean mask semantics authoritative; do not use NaN as null state.

Phase A1 fixes (June 2026):
- `MatMulVectorized`: result[i,j] is null if any contributing `a[i,k]` or
  `b[k,j]` is null (mask OR semantics across the k-sum).
- `TransposeVectorized`: result is null where original is null.
- Backward passes for both operations use the null-aware forward helpers.
- Tests added for forward null propagation and backward null handling.

Acceptance criteria:
- Existing and new operations preserve result null masks.
- Gradient utilities skip null positions consistently for norms and clipping.
- Tests assert both value behavior and `IsNull(index)` behavior.

## Deferred Non-Goals

- GPU or accelerator backends.
- Keras-like `Model.Compile()` or `Fit()`.
- Layers, dataloaders, full model abstractions, or metrics APIs.
- Implicit broadcasting.
- Broad numeric type support beyond `float` and `double`.
- Replacing `System.Numerics.Tensors` or core Nivara storage.
