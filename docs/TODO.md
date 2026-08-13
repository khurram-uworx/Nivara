# Plan: Fix issues #201, #203, #206

Branch: `khurram/issues` (off `main`).

> As each task executes, if you find deferred work or a concern, create a GitHub issue
> immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its number in
> the GitHub issues log below — don't rely on memory or wait until the end of the plan,
> as compaction during execution can lose it.

## 1. Issue #201 — `Schema.areTypesCompatible` uses stale numeric type list

**Problem:** `Schema.areTypesCompatible` (`src/Nivara/Schema.cs:17-31`) hardcodes an
11-type numeric list omitting `Half`, `nint`, `nuint`, `Int128`, `UInt128`, `char`,
diverging from the authoritative `TypeCompatibilityValidator.GetNumericTypes()`
(17 types) and `TypeExtensions.IsNumericType()`. This causes spurious type-mismatch
errors in cross-type arithmetic / schema compatibility checks (REVIEW.md #13 pending).

**Change (decision confirmed: delete the private method entirely):**
- `src/Nivara/Schema.cs`
  - Add `using Nivara.Helpers;`
  - Delete private static `areTypesCompatible` (lines 11-31).
  - Replace the call at line 258 with `TypeCompatibilityValidator.AreTypesCompatible(thisType, otherType)`.
- `tests/Nivara.Tests/SchemaTests.cs`
  - Add `IsCompatibleWith_WithExtendedNumericTypes_NonExactMatch_ReturnsTrue`:
    Half↔float, Int128↔int, nint↔long, char↔ushort with `requireExactMatch: false`.

**Blast radius:** `areTypesCompatible` is private; only call site is
`Schema.IsCompatibleWith` (used by frame/query schema validation paths). `GetNumericTypes`
becomes the single authoritative definition — widening compatibility only (no previously
valid pair becomes invalid). Existing `SchemaTests.IsCompatibleWith_*` stay green
(int/long already compatible). `TypeCompatibilityValidatorTests` already asserts the
17-type domain.

## 2. Issue #203 — `MultiheadAttention.Forward` uses `new` instead of `override`

**Problem:** `MultiheadAttention.Forward(input, paddingMask)`
(`src/Nivara/AutoDiff/Nn/MultiheadAttention.cs:77`) hides base
`virtual Module<T>.Forward(input1, input2)` (`Nn/Module.cs:15-18`, throws
`NotSupportedException`). Dispatch through a `Module<T>` reference hits the base throw.
Signatures match exactly, so `override` is valid.

**Change:**
- `src/Nivara/AutoDiff/Nn/MultiheadAttention.cs:77` — `public new` → `public override`.
- `tests/Nivara.Tests/AutoDiff/NnTests.cs` — add test calling
  `Forward(input, paddingMask)` through a `Module<float>` reference; assert correct
  `[L, D]` shape and no `NotSupportedException`.

**Blast radius:** No production callers (TransformerBlock does not use this class; no
`new MultiheadAttention` in `src/` or `samples/`). Only `NnTests.cs` uses it. `VAE.Forward`
also uses `new` but is legitimate (nullable second param → distinct signature); left alone.

## 3. Issue #206 — `Sequential` constructor silently skips null modules

**Problem:** `Sequential<T>` ctor (`src/Nivara/AutoDiff/Nn/Sequential.cs:11-24`) skips
null module entries; `Append()` throws. Silent architecture corruption.

**Change (decision confirmed: also throw for null params array):**
- `src/Nivara/AutoDiff/Nn/Sequential.cs` — replace ctor body with
  `ArgumentNullException.ThrowIfNull(modules);` then in the loop
  `ArgumentNullException.ThrowIfNull(m);` before add + `RegisterModules(m)`, matching
  `Append()` and `Module.RegisterModules`.
- `tests/Nivara.Tests/AutoDiff/NnTests.cs` — add
  `Sequential_Constructor_NullModule_ThrowsArgumentNullException` (null in middle, plus
  null-array case).

**Blast radius:** No existing callers/tests pass null (grep confirmed). Normal
`params` invocations unaffected (compiler always passes an array).

## Verification

- `dotnet build Nivara.slnx` after each step.
- Targeted tests before final commit: `SchemaTests`, `NnTests` (MHA + Sequential),
  `TypeCompatibilityValidatorTests`. **Ask the human before running `dotnet test`.**
- Full suite only with human confirmation (AGENTS.md).

## Planned commits

1. `docs: plan fixes for issues #201/#203/#206 in TODO.md` (this file)
2. `fix(schema): consolidate areTypesCompatible onto TypeCompatibilityValidator (#201)` + tests
3. `fix(autodiff): override MultiheadAttention.Forward for virtual dispatch (#203)` + test
4. `fix(autodiff): throw on null modules in Sequential constructor (#206)` + test
5. Review `docs/TODO.md`; if complete, `git rm docs/TODO.md` →
   `docs: remove TODO.md — plan executed`

## GitHub issues log

- [ ] (empty — add entries here as issues are created during execution)
