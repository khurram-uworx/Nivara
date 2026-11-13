# Plan — Issue #1169: Fused expression caches unbounded + lossy cache key

**Branch:** `khurram/1169` (off `main` @ `09a91028`)
**Area:** `src/Nivara/Expressions/` (fused expression engine)
**Severity:** medium · **Found in:** 1.5.0 (`2881a923`) · **Review:** `docs/REVIEW-2026-11-12.md` (M6)

## Problem

Two distinct defects:

1. **Unbounded caches.** `FusedExpressionEvaluator` keeps eight static `ConcurrentDictionary` caches. One of them — `compiledKernelCache` (`ConcurrentDictionary<string, CompiledFusedInvoke>`) — is keyed by a structural signature and grows forever: one compiled delegate per distinct expression, literals baked in as `Expression.Constant`. The other seven are keyed by `Type`/`(Type, Type)` and are structurally bounded by the closed-generic type universe, so they do not grow per query. The genuinely unbounded one is the compiled cache.

2. **Lossy cache key.** `ExpressionTypeInferer.FormatValue` formats literals with `IFormattable.ToString(null, InvariantCulture)`, which renders `DateTime`/`DateTimeOffset`/`TimeOnly` without fractional seconds (`DateTime` → `"11/12/2026 10:00:00"`). Two expressions differing only by sub-second ticks therefore share `FusedExpressionPlan.Signature` and hence a compiled delegate; the second silently computes with the first literal baked in. This is the precision half of the #246 class (the type half was fixed by appending `value.GetType().FullName`).

   Measured (pwsh / .NET, 2026-11-13) — the issue body's two secondary claims are **wrong** on .NET 11:
   - `TimeSpan` default `ToString()` is already round-trip-safe (`"00:00:00.1000000"`).
   - `double`/`float` default `ToString()` is shortest-roundtrippable since .NET Core 3.0 (`0.1` vs `0.10000000000000002` already distinct).
   - `TimeOnly` default is *worse* than filed: it renders `"10:00"` — dropping seconds *and* sub-second ticks.

   There is also a third, unreported class: the signature concatenates literal text verbatim with no quoting or length-prefix, so a string literal containing the signature's own delimiters could forge another expression's key.

## Proposed changes (with sketches)

### A. `ExpressionTypeInferer.FormatValue` — exact, injective

Replace formatted text with **numeric components** (deterministic, culture-free, machine-independent — note: `DateTime.ToString("O")` for `Kind=Local` embeds the machine offset, so it is *not* the right answer):

```csharp
static string FormatValue(object? value)
{
    if (value == null) return "null";
    var text = value switch
    {
        DateTime dt       => $"{dt.Ticks}:{(int)dt.Kind}",
        DateTimeOffset dto => $"{dto.UtcTicks}:{dto.Offset.Ticks}",
        TimeOnly t        => $"{t.Ticks}",
        DateOnly d        => $"{d.DayNumber}",
        TimeSpan ts       => $"{ts.Ticks}",
        Guid g            => g.ToString("D", CultureInfo.InvariantCulture),
        float f           => $"{BitConverter.SingleToInt32Bits(f)}",
        double d          => $"{BitConverter.DoubleToInt64Bits(d)}",
        Half h            => $"{BitConverter.HalfToInt16Bits(h)}",
        decimal m         => m.ToString(CultureInfo.InvariantCulture),
        string s          => s,
        IFormattable f    => f.ToString(null, CultureInfo.InvariantCulture),
        _                 => value.ToString() ?? string.Empty,
    };
    // Length-prefix makes the concatenation injective: a literal whose text contains
    // the signature's own delimiters cannot forge another expression's key.
    return $"{text.Length}:{text}:{value.GetType().FullName}";
}
```

`BFloat16` (no `BitConverter` overload) falls through to `IFormattable`; its round-trip behaviour is pinned by the test, and an explicit arm is added if it fails.

### B. `src/Nivara/Expressions/BoundedCache.cs` — new internal helper

Two-generation `ConcurrentDictionary`-backed cache, capacity 1024, approximate-LRU promotion:

- Hit `current` → return.
- Hit `previous` → promote into `current`, return.
- Miss → `factory(key)`, insert into `current`, `RotateIfFull()`.
- `RotateIfFull()` swaps `previous = current; current = new()` under a lock (rotations ≈ once per 1024 inserts, so no hot-path lock); `Interlocked.Increment` an overflow counter.
- Hard ceiling `2 × capacity`. Exposes `Count`, `OverflowCount`, `Clear()`.
- Documented wart: under contention the factory can run twice and one delegate is discarded — same as `ConcurrentDictionary.GetOrAdd` today.

### C. `FusedExpressionEvaluator.cs`

- `static readonly BoundedCache<string, CompiledFusedInvoke> compiledKernelCache = new(CompiledKernelCacheCapacity);`
- `internal const int CompiledKernelCacheCapacity = 1024;`
- Internal test surface: `CompiledKernelCacheEntryCount`, `CompiledKernelCacheOverflowCount`, `ClearCompiledKernelCache()`.
- XML doc on **all eight** cache fields — the seven `Type`/`(Type, Type)`-keyed ones documented as "bounded by the closed-generic type universe; no eviction path needed" (honest answer to "eight unbounded caches").

### D. Tests — `FusedExpressionEvaluatorTests.cs`, new `#1169` section

1. `Signature_TemporalLiteralsDifferingBelowOneSecond_AreDistinct` — `DateTime` by ticks, `TimeOnly` by seconds and ticks, `DateTimeOffset` by ticks and offset.
2. `Evaluate_DateTimeLiteralsDifferingByTicks_UsesTheQueryLiteral` — **the real bug**: warm cache with `t0`, evaluate with `t1`, assert the boolean arrays differ and both match expected.
3. `Evaluate_TimeOnlyLiteralsDifferingBySeconds_UsesTheQueryLiteral` — same shape for the worse `TimeOnly` case.
4. `Signature_FloatingPointLiteralsDifferingByOneUlp_AreDistinct` — `double`, `float`, `Half`, `BFloat16` (already-green guard).
5. `Signature_StringLiteralContainingSignatureDelimiters_DoesNotCollide` — injectivity.
6. `CompiledKernelCache_Overflow_IsBoundedAndCounted` — clear in setup; evaluate `capacity + 64` distinct `Col("A") + i` over a 1-row column; assert `EntryCount <= 2 × capacity`, `OverflowCount == 1`, and a post-rotation evaluation still returns correct values.

#246's existing tests stay untouched and green.

## Planned commits

1. `docs: plan #1169 in TODO.md`
2. `fix(expressions): round-trip-safe, injective literal signature encoding` (ExpressionTypeInferer)
3. `feat(expressions): add bounded two-generation cache` (BoundedCache.cs)
4. `fix(expressions): bound compiledKernelCache with eviction counter` (FusedExpressionEvaluator)
5. `test(expressions): #1169 cache-collision and bound regression tests`
6. (G2) any fix commits from review
7. `docs: remove TODO.md — plan executed`

## Blast radius

- **Symbols:** `ExpressionTypeInferer.FormatValue`, `FusedExpressionEvaluator.compiledKernelCache` (field type changes from `ConcurrentDictionary<,>` to `BoundedCache<,>`; `GetOrAdd` call site unchanged), new `BoundedCache<,>`.
- **Callers:** `compiledKernelCache` has exactly one call site (`EvaluateCompiled`, line 325). `FormatValue` has two (`BuildSignature` for `LiteralExpression` and `ScalarExpression.Scalar`).
- **Tests:** `FusedExpressionEvaluatorTests` (63 tests, all must stay green). New tests added.
- **Risk:** the cache field's type changes from a public-ish `ConcurrentDictionary` to an internal helper; nothing external references it (internal class). `KernelPlan.Signature` keeps its `string` type — no breaking change to the IR.

## Verification (before each commit)

- `dotnet build Nivara.slnx -c Release`
- targeted `dotnet test … --filter "FullyQualifiedName~FusedExpressionEvaluatorTests"` (ask before running)
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` before push

## GitHub issues log

- (none created yet)

## Grounding notes (G1)

- Code-memory: signature chain traced `ExpressionTypeInferer.BuildSignature` → `FusedExpressionPlan.Signature` → `KernelLowerer.Lower` → `KernelPlan.Signature` → `compiledKernelCache` (single consumer). Seven of the eight caches are `Type`/`(Type,Type)`-keyed → structurally bounded.
- Microsoft Learn: verified `DateTime`/`TimeOnly`/`TimeSpan` format strings (see measured table above); confirmed `Expression.Constant` is the baked-literal mechanism; confirmed `ConcurrentDictionary` has no built-in LRU.
- Blast radius above.
- Decisions raised and resolved in conversation: key strategy = fixed string (injective); eviction = two-generation/1024; scope = compiled cache only + docs; docs = XML only. No outstanding red flags.
