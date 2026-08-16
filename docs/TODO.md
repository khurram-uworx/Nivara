# Phase 2 — AutoDiff ADR-001 cleanup + SIMD (`khurram/incident-phase2`)

Source: `samples/Incident-PLAN.md` Phase 2 (tracked in issue #284). Work happens on
`khurram/incident-phase2` off `main` (Phase 1 branch `khurram/incident` is merged).

Goal: finish the ADR-001 non-null-domain cleanup inside the AutoDiff ops, and close the
remaining scalar-loop outliers by routing them through the shared `TensorPrimitives`
kernels. Small, isolated, high-value, low-risk. Every unit commits separately.

## Context (verified 2026-08-16, `main` @ HEAD)

- Line anchors in Incident-PLAN.md are stale; refreshed via code-memory (see per-unit file/line refs below).
- ADR-001 audit (grep `HasNulls|NullMask|nullMask|IsNull\(|WithoutNulls|TryGetNullMask` in
  `src/Nivara/AutoDiff/`) currently yields **7 matches, all at boundaries**:
  `ReverseGradTensor` ctors (x2), `ForwardGradTensor` ctors (x4), `TensorDataset` (x1).
  `GradTensor<T>.AsSpan()` (GradTensor.cs:167-171) throws on nulls. Interior must stay clean.
- BCL grounding (microsoft-learn, `net-10.0-pp` moniker): `TensorPrimitives.Pow<T>` and
  `MultiplyAdd<T>` generic overloads are available on .NET 10 (System.Numerics.Tensors
  10.0.10) and already compiled in-tree (ApplyPow / Adam kernels). No new API surface is
  introduced by any unit — we only route existing scalar loops through proven kernels.
  Do NOT reach for .NET 11 preview `Tensor<T>` methods; we target net10.
- Test baseline: 3028 green. Ask the human before running `dotnet test`.

## Planned commit list

1. `docs: plan phase 2 in TODO.md` — this file.
2. `perf(autodiff): drop dead nullable fallbacks in Gather/BroadcastGradient (ADR-001)` — 2.1.
3. `perf(autodiff): route reverse Pow through shared TensorPrimitives kernels` — 2.2 code.
4. `test(autodiff): NivaraTorch Pow parity fixture + reverse Pow backward tests` — 2.2 tests
   (gen_reference.py + fixture tree + tests committed as one unit, per gen_reference.py header
   "commit the full samples/data/torch-comparison/ tree as one unit").
5. `test(autodiff): double optimizer kernel routing + training parity` — 2.3.
6. `perf(autodiff): SIMD RMSNorm grad, broadcast per-channel runs, SGD momentum` — 2.4.
7. `docs: mark phase 2 complete + phase 3 handoff notes` — update samples/Incident-PLAN.md.
8. Remove docs/TODO.md — `docs: remove TODO.md — plan executed`.

---

## 2.1 — Dead branch removal inside the non-null domain

Blast radius: `ReverseGradOperations.Gather` + `GradOperationKernels.BroadcastGradient` are
internal AutoDiff helpers used only by the AutoDiff op surface. Downstream callers/tests:
`GradOperationsTests` (Gather block 1348-1462), `ForwardGradOperationsTests` (1161-1197),
`ForwardParityTests` (625-700), `DistilBertSequenceClassificationTests` (94,124),
`PerfTests.EmbeddingGather_OneHotMatMul_Vs_Gather`, `NnTests` (Broadcast block 1941-2123).

Rationale: domain tensors are constructed through ctors that throw on `HasNulls`, so
`NivaraColumn<T>.TryGetSpan` always succeeds for domain data. Both `else` branches are
unreachable nullable-store codepaths.

### 2.1a `Gather` — `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs:2423-2441`

Replace the `if/else` with the single span path (keep the `if` guard, delete the `else`
indexer loop `source.Data[srcOffset + j]`):

```csharp
if (source.Data.TryGetSpan(out var span))
{
    for (int i = 0; i < indices.Length; i++)
    {
        int srcOffset = indices[i] * stride;
        int dstOffset = i * stride;
        span.Slice(srcOffset, stride).CopyTo(resultValues.AsSpan(dstOffset, stride));
    }
}
```

Do NOT touch the backward gradFn (2452-2481) — it uses TryGetSpan + ArrayPool scatter with no
null branch. `using System.Buffers;` stays (backward rents gradBuf).

### 2.1b `BroadcastGradient` — `GradOperationKernels.cs:235-259`

Replace the `TryGetSpan` + `ArrayPool` fallback with the single `Array.Fill` path:

```csharp
if (scalarGrad.Length != 1)
    throw new ArgumentException($"Expected scalar gradient with length 1, got {scalarGrad.Length}");

scalarGrad.TryGetSpan(out var span);
var filled = new T[targetLength];
Array.Fill(filled, span[0]);
return NivaraColumn<T>.CreateFromOwnedArray(filled);
```

`using System.Buffers;` stays in GradOperationKernels.cs (ApplyDropout/ApplyDropoutGradient rent).

### Post-change audit (required)

Re-run the same grep. Expected: identical 7 boundary matches, zero new interior matches.

Verification: `dotnet build Nivara.slnx`; targeted suites `GradOperationsTests`,
`ForwardGradOperationsTests`, `ForwardParityTests`, `DistilBertSequenceClassificationTests`.

---

## 2.2 — `Pow` routed through the shared SIMD kernel

Blast radius: `ReverseGradOperations.Pow<T>` is public API. Callers: Nn modules/losses that
use Pow (VAE KL/sample math, MLP layers), `ForwardParityTests.Pow_ForwardTangent_EqualsBackwardGradient`
(443-455), `ForwardGradOperationsTests` (1346,1361). Behavior change: forward/backward now via
`TensorPrimitives.Pow` (SIMD) instead of scalar `Math.Pow`; identical math, tolerances in tests
already absorb last-ulp differences. Bonus: removes the forward-time `aArr` copy allocation.

### Rewrite — `ReverseGradOperations.cs:1633-1663`

Mirror `ForwardGradOperations.Pow` exactly:

```csharp
public static ReverseGradTensor<T> Pow<T>(ReverseGradTensor<T> a, double exponent) where T : struct, IFloatingPointIeee754<T>
{
    if (a == null) throw new ArgumentNullException(nameof(a));

    var resultTensor = new ReverseGradTensor<T>(
        GradOperationKernels.ApplyPow(a.Data, exponent),
        GradientUtils.ShouldTrackGrad(a),
        a.shape);

    if (GradientUtils.ShouldTrackGrad(a))
    {
        var gradFn = new OpNode<T>("Pow", [a], (typedGradOutput) =>
        {
            AccumulateGradient(a, GradOperationKernels.ApplyPowGradient(a.Data, typedGradOutput, exponent));
        });
        ComputationGraph.AddNode(resultTensor, gradFn);
    }

    return resultTensor;
}
```

`a.Data` read at backward time is safe (tensors immutable; only Grad mutates; matches
Gather/Slice/SparseEmbeddingBag closure pattern).

### Tests for 2.2

1. Fixture: extend `samples/NivaraTorch/gen_reference.py` with a dedicated
   `pow_rng = torch.Generator().manual_seed(404)` (keeps shared `ops_rng`/main streams
   bit-stable). Follow the AddBias block pattern (lines 781-802):
   - `pow_input.bin` = `torch.randn(8, generator=pow_rng)` (exponent 2.0; integer exponent
     over randn avoids NaN edges)
   - `pow_output.bin` = `pow_input.pow(2.0)`
   - `pow_grad.bin` = `pow_input.detach().requires_grad_(True)` → `.pow(2.0).sum().backward()`
     → `pow_input.grad`
   - manifest `"pow"` entry + print. Regenerate: `python samples/NivaraTorch/gen_reference.py`,
     commit the full `samples/data/torch-comparison/` tree.
2. `tests/Nivara.Tests/NivaraTorch/OperationTests.cs` — `Pow_MatchesPyTorch`: forward output vs
   `pow_output.bin`, then `Sum(output).Backward()` and compare `tensor.Grad` vs `pow_grad.bin`.
3. `tests/Nivara.Tests/AutoDiff/GradOperationsTests.cs` — `Pow_Backward_MatchesHandComputedGradient`
   (Pow(x,2.0) over {2,3,4}, Sum→Backward, assert grad == 2x) + fractional case (0.5 over
   positive input).

---

## 2.3 — Verify optimizer SIMD coverage (no core code change expected)

Blast radius: none (verification only) + one new test in `OptimizerTests.cs`.

Findings (source-verified): `Adam.ApplyAdamToSpan` dispatches float→`ApplyAdam_Kernel_Float` (83),
double→`ApplyAdam_Kernel_Double` (97), Half→`ApplyAdam_Kernel_Half` (111); trailing scalar loop
(125-139) is the defensive fallback. `AdamW.ApplyAdamWToSpan` same at 69/83/97. The earlier audit
claim that a new double kernel is required was INCORRECT — both double kernels exist. No code change.

Work:
- Kernel routing check: `Adam_Step_FloatAndDouble_ProduceEquivalentValues` (OptimizerTests.cs:814)
  already exercises the double kernel. Add a lightweight assertion that the double path (not the
  fallback) runs via `AutoDiffDiagnostics` capture (AutoDiff=AdamUpdate diagnostic) — optional;
  parity test may suffice.
- New `Adam_TrainingLoop_FloatAndDouble_ProduceEquivalentTrajectories` in OptimizerTests.cs:
  train small `Linear<float>` and `Linear<double>` on the same toy dataset ~10 steps (manual loop
  inside `GradientUtils.Grad()`: forward Linear + Sum loss, Backward, Step, ZeroGrad), assert
  parameters within 1e-6 each step.

---

## 2.4 — Secondary SIMD candidates (IN SCOPE per maintainer)

Blast radius: `GradOperationKernels.ApplyRMSNormGradient`, `ReverseGradOperations.BroadcastMultiply/
BroadcastAdd` (forward + backward), `ForwardGradOperations.BroadcastMultiply/BroadcastAdd`,
`SGD.stepNoMomentumInPlace/stepWithMomentumInPlace`. Existing coverage: `NnTests` (1941-2123),
`ForwardGradOperationsTests` (1510-1603), `ForwardParityTests` (458-582), `OptimizerTests`
SGD momentum (57, 718, 943).

All APIs already in-tree; no new BCL surface.

### 2.4a RMSNorm grad chain — `GradOperationKernels.cs:226-227`

Replace the element loop with two SIMD ops:

```csharp
TensorPrimitives.Multiply(gSpan, invRms, result);
TensorPrimitives.MultiplyAdd(inSpan, T.Negate(scale), result, result);
```

### 2.4b Broadcast per-channel-run SIMD

Data layout `[batch, channels, ...]`; channelStride = product of dims after channel dim. A run of
`channelStride` contiguous elements at offset `(b*c + ch)*channelStride` shares `scaleData[ch]` /
`biasData[ch]`.

- Reverse `BroadcastMultiply` (2631-2693):
  - forward: `TensorPrimitives.Multiply(inputRun, scaleData[ch], outputRun)` per (b, ch)
  - input grad: same per-run multiply by scaleData[ch]
  - scale grad: `scaleGrad[ch] += TensorPrimitives.Dot(gradRun, inputRun)` per batch
- Reverse `BroadcastAdd` (2703-2759):
  - forward: `TensorPrimitives.Add(inputRun, biasData[ch], outputRun)`
  - bias grad: `biasGrad[ch] += TensorPrimitives.Sum(gradRun)` per batch
- Forward `BroadcastMultiply` (1336-1393): primal per-run Multiply; input tangent per-run
  Multiply by scaleData[ch]; scale tangent per-run MultiplyAdd with scaleTanSpan[ch].
- Forward `BroadcastAdd` (1399-1464): primal per-run Add(scalar); aTan+bTan per-run
  Add(scalar); bTan-only per-run Array.Fill.

Keep validation/exception code untouched.

### 2.4c SGD momentum step — `SGD.cs:14-40`

`writable` aliases `dataSpan` (same underlying array); in-place TensorPrimitives destinations are
supported when they begin at the same location.

- `stepNoMomentumInPlace`, wd == 0: `writable = data - lr*grad`:
  ```csharp
  TensorPrimitives.Multiply(gradSpan, lr, temp);
  TensorPrimitives.Subtract(dataSpan, temp, writable);
  ```
- wd != 0: `writable = data - lr*(wd*data + grad)`:
  ```csharp
  TensorPrimitives.Multiply(dataSpan, wd, temp);
  TensorPrimitives.Add(temp, gradSpan, temp);
  TensorPrimitives.Multiply(temp, lr, temp);
  TensorPrimitives.Subtract(dataSpan, temp, writable);
  ```
- `stepWithMomentumInPlace`: `velocity = momentumT*velocity + lr*(wd*data + grad)` via
  `TensorPrimitives.MultiplyAdd(velocity, momentumT, temp, velocity)` then `writable = data - velocity`
  (Subtract with aliasing dataSpan/writable).

---

## Phase 3 handoff — `samples/Incident-PLAN.md` update (final step)

Update the Status header: Phase 2 (2.1-2.4) complete on `khurram/incident-phase2`; Phase 3
sample deferred. Replace the "Phase 2" section status with the completion note and add a
"Phase 2 → Phase 3 handoff notes" section capturing anything useful discovered while
implementing, e.g.:

- 2.2: Pow now routes through the shared `GradOperationKernels.ApplyPow/ApplyPowGradient`
  (SIMD) — any Phase 3/4 AutoDiff microbenchmark (Phase 4 item 4) should use Pow to show the
  SIMD impact; NivaraTorch `pow` fixture (forward + backward) exists for regression.
- 2.3: optimizer double/Half SIMD kernels confirmed — no new kernel work needed for Phase 4
  training microbenchmarks.
- 2.4: RMSNorm grad, Broadcast per-channel runs, and SGD momentum are SIMD — Phase 4
  kernel-selection visibility (% vectorized) should reflect these.
- Remaining stale-anchor caveat for future phases: re-grep before editing (line refs drift).
- ADR-001 audit remains clean (7 boundary matches only) after 2.1.

## GitHub issues log

- [ ] #284 — Phase 2+ deferred work (NivaraIncident plan; all of Phase 2 was tracked here).
- No new issues expected; if a unit surfaces a real concern (e.g. TensorPrimitives.Pow vs
  scalar Math.Pow divergence on target types, a broadcast op that still forces a copy), create
  `gh issue create --repo khurram-uworx/Nivara` immediately and record it here.

## Execution reminders

- Ask before running `dotnet test`; verify with `dotnet build Nivara.slnx` after each unit.
- Stage selectively; one logical change per commit; never push.
