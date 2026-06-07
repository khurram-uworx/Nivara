# Tensor Tasks — Backing Off

## Context

This document tracks the work items coming out of the decision documented in
[`TENSORS.md`](TENSORS.md): Nivara **backs off** from owning tensor math and
repositions as a tabular data layer that interops with BCL tensor APIs.

For now the scope is limited to backing off — not deciding where or how to take
things next. Future direction is deferred pending community feedback and
platform evolution.

## BCL tensor direction (research findings)

`System.Numerics.Tensors` in .NET 10 is span-oriented:

- `TensorPrimitives` static methods accept `ReadOnlySpan<T>` / `Span<T>` —
  generic overloads exist for any `T` implementing `INumber<T>` /
  `IRootFunctions<T>`, etc.
- `TensorSpan<T>` / `ReadOnlyTensorSpan<T>` are the primary surface for
  shaped tensor views — implicit conversion from `T[]` to `TensorSpan<T>`.
- `Tensor<T>` implements `ITensor<TSelf, T>` with `AsTensorSpan()`,
  `GetSpan()`, `TryGetSpan()` for zero-copy span extraction.
- Nearly 200 `TensorPrimitives` overloads in .NET 9, growing in .NET 10.

The trend: **spans are the currency** — not `Tensor<T>` allocations.
Nivara's interop should follow suit.

## Current interop snapshot

| API | Returns | Null-safe? |
|-----|---------|-----------|
| `NivaraColumn<T>.ToTensor()` | Copy | No (throws) |
| `NivaraColumn<T>.ToTensor(T default)` | Copy | Yes (fills nulls) |
| `NivaraSeries<T>.ToTensor()` / `ToTensor(T)` | Copy | No (throws) / Yes |
| `NivaraFrame.ToTensor<T>()` | Copy | No (throws on nulls) |
| `TensorInteropExtensions.ToTensorSpan<T>()` | Copy | No (throws on nulls) |
| `TensorInteropExtensions.FromTensor<T>()` | Copy | N/A |
| `TensorInteropExtensions.FromTensor<T>(Tensor, string[])` | Copy | N/A |
| `IColumnStorage<T>.AsSpan()` | View (internal) | No (raw data) |
| `TensorStorage.AsTensorSpanIfNoNulls()` | View (internal) | Yes (guards) |

Every public bridge copies defensively. The only zero-copy path
(`AsTensorSpanIfNoNulls`) is internal.

## Tasks

### T1. Audit and evaluate the current interop API surface

Catalog every public interop method, its null policy, copy semantics, and
whether it still earns its keep under the new direction. Identify which
methods to keep, redesign, or deprecate.

**Why:** Before removing anything ("backing off"), we need to ensure the
interop experience users will rely on instead is complete and ergonomic.

---

### T2. Redesign interop to align with BCL span conventions

Replace or supplement `ToTensor<T>()` (returns `Tensor<T>` copy) with
span-based accessors that feel native to modern .NET:

| Current | Target direction |
|---------|-----------------|
| `column.ToTensor()` → `Tensor<T>` copy | `column.AsReadOnlySpan()` → `ReadOnlySpan<T>` view (when no nulls) |
| `series.ToTensor(T default)` → `Tensor<T>` copy | `column.CopyTo(Span<T>, T default)` → fills user-provided span |
| `frame.ToTensor<T>()` → copy | `frame.AsRowMajorSpan()` → `ReadOnlySpan<T>` over pooled buffer |
| `FromTensor(Tensor<T>)` → copy | accept `ReadOnlyTensorSpan<T>` / `ReadOnlySpan<T>` directly |

Key design constraints:
- Immutability must be preserved — return `ReadOnlySpan<T>`, never `Span<T>`
- Null masks stay authoritative — expose `HasNulls` + `ReadOnlySpan<bool>?`
- Zero-copy only when column has no nulls; fall back to explicit fill API

---

### T3. Move tensor math wrappers to experimental namespace

The following APIs compete with `System.Numerics.Tensors` and should be
moved to `Nivara.Experimental.Tensors` or `Nivara.Extensions.Tensors`,
with an `[Obsolete]` pointing users to the BCL equivalent:

- `NivaraFrame.Dot(...)`
- `NivaraFrame.CosineSimilarity(...)`
- `NivaraFrame.ColumnNorms(...)`
- `NivaraFrame.RowNorms(...)`
- `NivaraSeries.DotProduct(...)`
- `NivaraSeries.Norm(...)`

Do not delete immediately — mark obsolete first, ship a transition window.

---

### T4. Add convenient column-to-span helpers

When nulls are present, users currently call `column.ToTensor(defaultValue)`
which allocates a `Tensor<T>`. Target experience:

```csharp
// Zero-copy when null-free
ReadOnlySpan<float> data = column.TryGetSpan(out var span) ? span : [];

// Explicit fill with rented buffer
float[] buffer = ArrayPool<float>.Shared.Rent(column.Length);
column.CopyTo(buffer, defaultValue: 0.0f);
// ... use buffer ...
ArrayPool<float>.Shared.Return(buffer);
```

This mirrors how `ReadOnlyTensorSpan<T>` exposes `TryGetSpan` in the BCL.

---

### T5. Update null mask interop

Null masks are central to Nivara's value. Ensure they interoperate cleanly
with span/tensor patterns:

- `column.NullMask.TryGetSpan(out ReadOnlySpan<bool> mask)` — existing
  internal; consider public exposure
- `column.CopyTo(Span<T> destination, T fillValue, Span<bool>? maskDestination)`
  — fill data and optionally write null positions to a mask span
- Document that `ReadOnlySpan<bool>` null masks mirror the `Tensor<bool>`
  approach Nivara already uses internally

---

### T6. Clean up deferred items from TENSORS-GAPS.md

Many items in [`TENSORS-GAPS.md`](TENSORS-GAPS.md) become irrelevant after
this decision. File-specific disposition:

| Item | Action |
|------|--------|
| P1 RowNorms optimization | Close — being deprecated |
| P2 TopKDescending threshold | Keep — Nivara-specific |
| E1 Row-wise cosine similarity | Close — Nivara won't add this |
| E2 Normalize helper | Keep — NivaraSeries op delegating to TensorPrimitives |
| E3 Dot vs DotProduct naming | Close — both being deprecated |
| E4 Nullable tensor conversion | Re-evaluate after T2 |
| E5 Frame tensor conversion metadata | Re-evaluate after T2 |
| E6 Axis enum | Close — no tensor math APIs to need it |
| E7 Collection expressions | Keep — Nivara ergonomics |
| E8 Shape constructors | Re-evaluate — keep as interop, not math |
| E9 TopKDescending result type | Keep — Nivara-specific |
| E10 Scalar tensor reducers | Close — being deprecated |
| T1 Benchmarks | Reframe — focus on interop paths |
| D1-D4 Design items | Close — not Nivara's concern |

---

### T7. Document the "back off" surface clearly

After implementation, update:
- `EXAMPLES.md` — already revised (shows Python → BCL first, Nivara only
  for tabular value-add)
- XML doc comments on deprecated members — include `<see
  cref="TensorPrimitives.CosineSimilarity"/>`-style redirects
- A migration guide if any code will break (currently none will — all APIs
  stay, just move namespace)

---

## Out of scope (for now)

- Deciding Option A/B/C from TENSORS.md (columnar analytics vs Polars vs AI
  data infra) — deferred
- Adding new Nivara tabular features — that is separate work not tracked
  here
- Removing the deprecated APIs — mark only, don't delete yet
