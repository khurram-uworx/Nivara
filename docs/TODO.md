# Plan: generic narrow-precision benchmark (Approach B) for NivaraInference

## Problem

`--precision f32|bf16|fp16` added narrow-precision **compare** modes, but `benchmark` is F32-only.
Today `--precision bf16|fp16 benchmark` silently routes to the *compare* path (the switch checks
`bf16`/`fp16` before `benchmark`), so there's no way to time fp16/bf16 inference. The three F32
`Run*Benchmark` methods each duplicate the same 3-warmup + 10-pass timing loop, and the three
`Run*BFloat16` / `Run*Half` compare methods duplicate the narrow model builds. **Approach B**
unifies all of it into a single generic benchmark path over
`T : struct, IFloatingPointIeee754<T>` (float / Half / BFloat16), so one code path times all three
precisions and the 9 model×precision combinations route through it.

## Approach B design

**1. Shared generic timing/reporting helper** (in `Program.cs`) — the loop every benchmark needs,
generic over the tensor dtype:

```csharp
static void ReportTiming<T>(Func<ReverseGradTensor<T>> forward, int warmup = 3, int passes = 10)
    where T : struct, IFloatingPointIeee754<T>
{
    for (int i = 0; i < warmup; i++) forward();
    var times = new List<long>();
    for (int i = 0; i < passes; i++)
    {
        var sw = Stopwatch.StartNew();
        forward();
        sw.Stop();
        times.Add(sw.ElapsedMilliseconds);
    }
    Console.WriteLine($"  Average: {times.Average():F1} ms");
    Console.WriteLine($"  Min:     {times.Min():F1} ms");
    Console.WriteLine($"  Max:     {times.Max():F1} ms");
    Console.WriteLine();
}
```

**2. Three generic per-model benchmark methods** (in `Program.cs`), one per model, each generic
over `T`. Each builds the `<T>` model from `(T[] Data, int[] Shape)` tensors, uses the exact-int
token path (token IDs stay `int`; narrow mask via `GradientUtils.Constant(Array.ConvertAll(msk, x => (T)x))`),
reports params + weight MB (via `Unsafe.SizeOf<T>()` so F32=4 B, Half/BF16=2 B), then
`ReportTiming<T>(forward, 3, 10)`:

- `BenchmarkMiniLM<T>(Dictionary<string,(T[] Data,int[] Shape)> tensors, string precision)`
  → `MiniLMDistilled<T>.LoadWeights<T, T>(tensors, config)`, `model.Forward(intIds, mask, 1, intIds.Length)`.
- `BenchmarkDistilBert<T>(tensors, precision)`
  → `new BertEncoder<T>(config.ToBertConfig(), includeTokenTypeEmbedding: false)` +
  `DistilBertLoader.LoadEncoderWeights<T, T>(encoder, tensors, "distilbert")`,
  `encoder.ForwardWithMask(intIds, mask)`.
- `BenchmarkDistilBertSst<T>(tensors, precision)`
  → `DistilBertSst.Load<T>(tensors, modelDir)` + `DistilBertSst.PredictLogits<T>(model, tokenizer, text, 128)`,
  then a last-pass sentiment line via the float `Softmax`/`Label` (convert the `<T>` logits to `float[]`
  just for the label, mirroring the compare modes).

**3. Generic helpers in `DistilBertSst.cs`** — replace the per-dtype `Load`/`LoadBFloat16`/`LoadHalf`
and `PredictLogits`/`PredictLogitsBFloat16`/`PredictLogitsHalf` with generic versions (the typed
ones delegate to them, or call sites switch to the generics) so the SST-2 path no longer repeats
per-dtype code:

```csharp
public static DistilBertForSequenceClassification<T> Load<T>(
    Dictionary<string, (T[] Data, int[] Shape)> tensors, string modelDir)
    where T : struct, IFloatingPointIeee754<T>
{
    var config = DistilBertConfig.FromJson(File.ReadAllText(Path.Combine(modelDir, "config.json")));
    var model = new DistilBertForSequenceClassification<T>(config.ToBertConfig(), numClasses: 2);
    model.LoadWeights<T>(tensors);
    return model;
}

public static ReverseGradTensor<T> PredictLogits<T>(
    DistilBertForSequenceClassification<T> model, BertTokenizer tokenizer, string text, int maxLen)
    where T : struct, IFloatingPointIeee754<T>
{
    var (tokenIds, attnMask, _) = MiniLMTokenizer.Encode(tokenizer, text, maxLen);
    var intIds = Array.ConvertAll(tokenIds, x => (int)x);
    var mask = GradientUtils.Constant(Array.ConvertAll(attnMask, x => (T)x));
    return model.Forward(intIds, mask, 1, intIds.Length);
}
```

`Load<float>(float tensors)` is equivalent to the old `Load` (generic `LoadWeights<float>` covers
the head layers — confirmed in `DistilBertModel.cs:71`). `PredictLogits<T>` uses the exact-int path
for every `T` (correct for F32, Half, BF16 alike). The existing typed `Save*CompareOutput` methods
keep using the `Load<T>`/`PredictLogits<T>` generics (they stay, delegating through the generics;
no dead code).

**4. Switch routing** (`Main`) — route each model's `benchmark` (at all three precisions) through
the generic method; the F32 `Run*Benchmark` methods are removed (replaced by the generic wrappers)
to avoid dead code:

```csharp
case "minilm":
    if (bf16) return benchmark ? BenchmarkMiniLM(tensorsBf16, "BFloat16") : RunMiniLMBFloat16(tensorsBf16);
    if (fp16) return benchmark ? BenchmarkMiniLM(tensorsHalf, "Half") : RunMiniLMHalf(tensorsHalf);
    if (compare) return RunMiniLMCompare(tensors);
    bool similarity = mode == "similarity";
    return similarity ? RunMiniLMSimilarity(tensors) : benchmark ? BenchmarkMiniLM(tensors, "F32") : RunMiniLMInference(tensors);
```

Same pattern for `distilbert` and `distilbert_sst`. The compare/inference/`similarity`/`predict`
modes keep their existing `(float[],...)` methods unchanged.

## Verification

- `dotnet build samples/NivaraInference/NivaraInference.csproj -c Release` (after the refactor).
- Manual (weights present): `--precision f32|bf16|fp16 benchmark` for `minilm`, `distilbert`,
  `distilbert_sst` — each prints header + params + weight MB (4 vs 2 B) + avg/min/max ms.
- Confirm `--precision bf16|fp16` *without* `benchmark` still does compare; F32 `benchmark`,
  `compare`, `similarity`, `predict`, image paths unchanged (routing preserved).
- Ask before `dotnet test` (sample-only change; low test exposure, but confirm with human).

## Planned commits

1. `refactor(samples): genericize DistilBertSst Load/PredictLogits over the compute dtype` —
   `DistilBertSst.cs` generic helpers; typed methods delegate through generics.
2. `feat(samples): add generic F32/fp16/bf16 benchmark modes (Approach B)` — `ReportTiming<T>` +
   `BenchmarkMiniLM<T>`/`BenchmarkDistilBert<T>`/`BenchmarkDistilBertSst<T>` + routing; remove old
   F32 `Run*Benchmark` methods.
3. `docs: document narrow-precision benchmark usage in NivaraInference README` — update the
   "Speed — not yet separately benchmarked" note, add `--precision <bf16|fp16> benchmark` examples.

## Blast radius

- `samples/NivaraInference/Program.cs`, `samples/NivaraInference/DistilBertSst.cs`,
  `samples/NivaraInference/README.md`. No engine (`src/Nivara`), no `Nivara.Samples`, no tests.
- **Higher-touch than Approach A**: the three working F32 `Run*Benchmark` methods are removed and
  replaced by generic `Benchmark*<T>` calls; `DistilBertSst.Load*/PredictLogits*` are refactored to
  generic. Every call site of those removed/renamed methods must be updated (the switch + any
  remaining usages) — this is the B risk the human accepted.
- `DistilBertSst` is consumed only from `samples/NivaraInference/Program.cs` and its own file
  (verified via code-memory impact earlier), so the refactor stays sample-local.
- Behavior of documented invocations (`bf16` alone, `f32 benchmark/compare/similarity/predict`)
  is preserved; narrow `benchmark` is the new capability.
- Existing `PerfTests`/`EmbeddingBFloat16Tests` are unaffected (they test the engine, not the sample).

## GitHub issues log

- [ ] #363 — Narrow-precision (fp16/bf16) CPU inference is ~26x slower than F32 in benchmark mode
  (observed while verifying Approach B; document caveat + consider a vectorized narrow kernel later).
  <https://github.com/khurram-uworx/Nivara/issues/363>
