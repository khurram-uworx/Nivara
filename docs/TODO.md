# TODO: Half/BFloat16 parity coverage (issue #1021)

Branch: khurram/1021 (base main)

## Problem

Issue #1021 asks for targeted parity coverage for `Half` / `BFloat16` across:
- core columns (null-mask arithmetic/comparisons, windows/cumulative ops, sorting,
  aggregation promote-to-double, fused expressions),
- AutoDiff native-type tests (relu/sigmoid/linear/attention + Adam step),
- SafeTensors loader (F16/BF16 read, widening/narrowing, `Read<BFloat16>()` fast path).

Current coverage is partial: `NivaraColumnTests` has shared arithmetic for both types but
dedicated null-mask only for BFloat16; `FusedExpressionEvaluatorTests` covers same-type and
promotion for both; `WindowFunctionsFrameTests` covers several Half/BF16 rolling cases; AutoDiff
has a full `BFloat16Tests` fixture but no `Half` equivalent and no attention parity; SafeTensors
has BF16->F32 widening tests but no in-memory F16/Half or `Read<BFloat16>()` fast-path test.

## Placement lessons from #1051 / #1052

- Keep `samples/NivaraInference` visitor-facing and minimal.
- Move synthetic-weight / evaluation / parity tooling into tests/performance harnesses.
- Tests must not depend on `NivaraInference.Qwen.SynthesizeTensors` unless explicitly CLI E2E.
- Use test-owned factories (pattern: `SilkQwenTestWeights`) for model-shaped data.
- New tests land as one consolidated fixture set per area (preferred by user).

## Proposed changes

### 1. Core columns
New consolidated fixture: `tests/Nivara.Tests/HalfBFloat16ColumnParityTests.cs`.
- `Arithmetic_NullMaskPropagation_Parity` for both types (scalar + column ops).
- `Comparison_SqlLikeNullMask_Parity` (equals/greater/less/top).
- `WindowAndCumulative_Parity` (rolling sum/mean/count, cumulative, shift, null inputs).
- `Sorting_Parity` (ascending/descending with nulls).
- `Aggregation_PromotesToDouble_Parity` (Sum/Mean/Quantile/StdDev/Variance).
- `FusedExpressions_Parity` (same type and `+ int` promotion to double).

Keep values small so Half range is not an issue; assertions exact where repr is exact.

### 2. AutoDiff
New fixture: `tests/Nivara.Tests/AutoDiff/HalfTests.cs`, mirroring `BFloat16Tests.cs`:
- Type admission.
- Elementwise forward + backward parity (relu, sigmoid, tanh, gelu, arithmetic).
- Softmax parity.
- MatMul parity.
- Linear SGD loss decrease.
- Linear Adam finite parameters.
- Inference mode (outside `Grad()`) no graph.
- `ToReverseGradTensorsAuto` conversion.
Add attention parity test to both `HalfTests.cs` and `BFloat16Tests.cs` using a tiny
deterministic `MultiHeadAttention` and float reference.

### 3. SafeTensors
New fixture: `tests/Nivara.Tests/AutoDiff/SafeTensorsLoaderHalfF16Tests.cs`:
- `Read<Half>` on in-memory F16 bytes equals expected Half.
- `Read<float>` on F16 bytes widens via Half.
- `Read<BFloat16>` on BF16 bytes equals raw reinterpret (all patterns spot/loop).
- `Read<float>` on BF16 bytes equals `SafeTensorsLoader.WidenBf16ToF32`.
- F32 -> Half/BF16 narrowing via `T.CreateChecked`.
- malformed header/length throws `ArgumentException`/`NotSupportedException`.
No model files required; synthetic bytes only.

## Blast radius

- Test-only changes; no `src` production code changes expected.
- No docs change unless snippet gate affected (not expected).
- Performance: small fixtures, no `[Category("Performance")]` needed because no timing assertions.
- If a test exposes a real production bug, address in same branch as additional fix commit with explicit note.

## GitHub issues log

- (none opened yet; create during execution if deferred work is found)

## Planned commits

1. `docs: plan 1021 in TODO.md`
2. `test(columns): add consolidated Half/BFloat16 parity coverage`
3. `test(autodiff): add HalfTests mirroring BFloat16 + attention parity`
4. `test(safetensors): add F16/Half + BF16 read/widening/narrowing coverage`
5. `docs: remove TODO.md - plan executed` (after G2 reviews)

## Verification

- Targeted: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~HalfBFloat16ColumnParity|FullyQualifiedName~HalfTests|FullyQualifiedName~SafeTensorsLoaderHalfF16"`
- Format check: `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Docs suite only if docs changed.

## Notes

Code-memory MCP returned null in-session; navigation was done with repo search + existing tests. Flag for follow-up if indexing must be confirmed.
