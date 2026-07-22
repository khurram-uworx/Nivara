# NivaraClassifier Implementation Plan

## Overview

Build a runnable text classification sample that exercises Nivara's autograd pipeline end-to-end: data generation → tokenization → embedding → mean pool → MLP → cross-entropy loss → training → inference. The implementation drives core library improvements and validates the full training pipeline with sequence data.

## Guiding Principles

1. **ADR-001: AutoDiff is a non-nullable domain.** All data entering the autograd graph must be null-free. Null boundary is enforced at `NivaraColumn<T>.Create(T[])` → `ReverseGradTensor<T>`. Strip nulls before AutoDiff entry; never introduce nulls into the computation graph.

2. **Embrace SIMD/Numerics/Tensors.** Use `TensorPrimitives` wherever spans are available — `Sum`, `Add`, `Multiply`, `Dot`. The `Embedding.Forward` already uses `Parallel.For` + `TensorPrimitives.Dot`. MeanPool should use `TensorPrimitives.Sum` for the reduce step.

3. **Find gaps and fix them.** Each implementation step should identify what's missing in the core library and either fix it or document it as a gap. Prioritize fixes that improve the developer experience for all users.

4. **Follow existing patterns.** Match NivaraChess CLI structure, NivaraGpt model patterns, and Nivara core conventions.

## Files to Create

| File | Purpose |
|------|---------|
| `samples/NivaraClassifier/NivaraClassifier.csproj` | Project file referencing Nivara core |
| `samples/NivaraClassifier/Program.cs` | CLI entry point, training, inference, wizard |
| `samples/NivaraClassifier/TextClassifierModel.cs` | `TextClassifierModel<T> : Module<T>` |
| `samples/NivaraClassifier/TextTokenizer.cs` | Reusable word-level tokenizer |
| `samples/NivaraClassifier/DataGenerator.cs` | Synthetic sentiment CSV generator |

## Core Library Changes (if needed)

| Change | File | Priority | Rationale |
|--------|------|----------|-----------|
| `LinearClassifier<T>` | `src/Nivara/AutoDiff/Nn/LinearClassifier.cs` | P2 (Gap F) | Convenience class: `Linear(in, hidden) → ReLU → Linear(hidden, out) → Softmax`. Reduces boilerplate for simple classifiers. |
| `MeanPool<T>` helper | Sample-local or `src/Nivara/AutoDiff/Nn/Functional/MeanPool.cs` | P3 | Mean-over-dimension reduction. Could be core if reusable, but currently only this sample needs it. |

## Implementation Steps

### Step 1: Project Structure

Create `NivaraClassifier.csproj` following the NivaraGpt pattern:

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net10.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
  <ItemGroup>
    <ProjectReference Include="..\..\src\Nivara\Nivara.csproj" />
  </ItemGroup>
</Project>
```

No external dependencies. Nivara core only.

### Step 2: TextTokenizer

Reusable word-level tokenizer. Goes in the sample initially; promote to `Nivara.Extensions.Text` if it proves useful.

**Design:**
- `TextTokenizer.FromDocuments(IEnumerable<string>, maxVocabSize, minFreq)` — static factory
- Builds vocab: word → frequency counting, min-freq filter, max-vocab truncation
- Special tokens: `<PAD>=0`, `<UNK>=1`, `<BOS>=2`, `<EOS>=3`
- `Encode(string text, int? fixedLength, bool addBosEos) → int[]` — word-level tokenization
- `Decode(ReadOnlySpan<int> tokens) → string` — round-trip for debugging
- `Save(string path)` / `Load(string path)` — vocab persistence via JSON

**Implementation notes:**
- Split on whitespace and punctuation (simple regex: `\w+`)
- Normalize to lowercase
- `Encode` returns `int[]` (not `List<int>`) for direct use with `TensorDataset<T>`
- Fixed-length padding uses `PadToken`; truncation drops from the end (or from the beginning for long尾部)

**Null boundary:** `Encode` always returns non-null `int[]`. Token IDs are validated at embedding lookup time.

### Step 3: DataGenerator

Synthetic sentiment CSV generator. Self-contained, no external data.

**Templates:**
- 7 positive templates: "i loved the {noun}", "the {noun} was absolutely fantastic", etc.
- 7 negative templates: "i hated the {noun}", "the {noun} was terrible and boring", etc.
- Nouns pool: ~40 common nouns (product, movie, book, service, restaurant, ...)
- Activities pool: ~20 activities (daily use, travel, work, gaming, ...)
- Adverb modifiers: optional ("very", "really", "extremely", "somewhat", "quite")

**API:**
- `DataGenerator.Generate(int count, int seed) → (string[] texts, int[] labels)`
- `DataGenerator.SaveCsv(string path, int count, int seed)` — write CSV directly

**Implementation notes:**
- Use `Random(seed)` for reproducibility
- Generate 1000 training + 200 test by default
- Output format: `text,label` CSV

### Step 4: TextClassifierModel

`TextClassifierModel<T> : Module<T>` — the core model.

**Architecture:**
```
Embedding(vocabSize, embedDim)     → [B, L, D]
MeanPool over L dimension          → [B, D]
Linear(D → hiddenDim) → ReLU      → [B, H]
Dropout(p) (optional)              → [B, H]
Linear(H → numClasses)             → [B, C]
```

**Forward implementation:**
```csharp
public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
{
    // 1. Embed: [B, L] → [B, L, D]
    var embedded = embedding.Forward(input);

    // 2. MeanPool: [B, L, D] → [B, D]
    var pooled = MeanPoolOverSequence(embedded, batchSize, seqLen, embedDim);

    // 3. MLP head
    var hidden = hiddenLayer.Forward(pooled);
    hidden = Activation.Relu(hidden);
    hidden = dropout?.Forward(hidden) ?? hidden;
    var output = outputLayer.Forward(hidden);

    return output;
}
```

**MeanPool implementation (sample-local):**
```csharp
ReverseGradTensor<T> MeanPoolOverSequence(ReverseGradTensor<T> embedded, int B, int L, int D)
{
    // For each batch element b, sum D elements across L positions, divide by L
    // Use existing ReverseGradOperations operations for autograd correctness

    var result = new T[B * D];
    for (int b = 0; b < B; b++)
    {
        for (int d = 0; d < D; d++)
        {
            T sum = T.Zero;
            for (int l = 0; l < L; l++)
                sum += embedded[b * L * D + l * D + d];
            result[b * D + d] = sum / T.CreateChecked(L);
        }
    }

    // This is NOT autograd-compatible! We need differentiable operations.
    // Alternative: use Slice + Mean per batch, or implement as a custom op.
}
```

**Better MeanPool approach (autograd-compatible):**

Since Nivara's `Mean` operates on the entire tensor (flattened), and we need per-batch-row mean, we have two options:

**Option A: Manual differentiable mean pool (recommended for this sample)**

Implement MeanPool as a `Module<T>`-level operation using existing differentiable ops:

```csharp
ReverseGradTensor<T> MeanPoolOverSequence(ReverseGradTensor<T> embedded, int B, int L, int D)
{
    // Reshape to [B*L, D] (already the shape from Embedding.Forward)
    // For each batch element b, Gather rows [b*L .. b*L+L-1], then Mean
    // This is O(B) Gather+Mean calls — simple but correct

    var batchMeans = new ReverseGradTensor<T>[B];
    for (int b = 0; b < B; b++)
    {
        var indices = new int[L];
        for (int l = 0; l < L; l++)
            indices[l] = b * L + l;
        var seq = ReverseGradOperations.Gather(embedded, indices, axis: 0); // [L, D]
        batchMeans[b] = ReverseGradOperations.Mean(seq); // scalar? No — we need row-wise mean
    }
    // Problem: Mean flattens everything. We need Mean over axis=0 of [L, D] → [D]
}
```

**Option B: Direct SIMD-accelerated mean pool with manual backward (best performance)**

Implement as a custom differentiable operation in the sample:

```csharp
// Forward: [B*L, D] → [B, D] by averaging L rows per batch
// Backward: distribute gradient equally across L rows

static ReverseGradTensor<T> MeanPoolBatched(ReverseGradTensor<T> input, int B, int L, int D)
{
    int inputRows = B * L;
    var inputData = input.Data;
    var resultData = new T[B * D];

    // SIMD-friendly: for each (b, d), sum over L rows
    for (int b = 0; b < B; b++)
    {
        for (int d = 0; d < D; d++)
        {
            T sum = T.Zero;
            int baseIdx = b * L * D + d;
            for (int l = 0; l < L; l++)
                sum += inputData[baseIdx + l * D];
            resultData[b * D + d] = sum / T.CreateChecked(L);
        }
    }

    var resultCol = NivaraColumn<T>.Create(resultData);
    var result = new ReverseGradTensor<T>(resultCol, GradientUtils.ShouldTrackGrad(input), [B, D]);

    if (GradientUtils.ShouldTrackGrad(input))
    {
        // Backward: gradient flows equally to all L positions
        // dInput[b*L+l, d] = dOutput[b, d] / L
        var gradFn = new OpNode<T>("MeanPoolBatched", [input], (gradOutput, sgn) =>
        {
            var inputGrad = new T[inputRows * D];
            var L_ = T.CreateChecked(L);
            for (int b = 0; b < B; b++)
            {
                for (int d = 0; d < D; d++)
                {
                    T gradVal = gradOutput[b * D + d] / L_;
                    for (int l = 0; l < L; l++)
                        inputGrad[b * L * D + l * D + d] = gradVal;
                }
            }
            var gradCol = NivaraColumn<T>.Create(inputGrad);
            AccumulateGradient(input, gradCol, sgn);
        });
        ComputationGraph.AddNode(result, gradFn);
    }

    return result;
}
```

**Decision:** Use Option B for performance. The manual backward pass is straightforward and SIMD-friendly. For `float`/`double`, the inner loop can use `TensorPrimitives.Add` on slices.

**SIMD optimization for MeanPool:**

For `float` type, the inner sum loop can be vectorized:
```csharp
// Instead of scalar sum over L rows for each (b, d):
// Copy D elements from each of L rows into contiguous buffers, then use TensorPrimitives.Add
var rowBuffer = ArrayPool<T>.Shared.Rent(D);
var accBuffer = ArrayPool<T>.Shared.Rent(D);
try
{
    for (int l = 0; l < L; l++)
    {
        // Copy row [b*L+l, :] into rowBuffer
        inputData.Slice(baseIdx + l * D, D).CopyTo(rowBuffer);
        // Accumulate: accBuffer += rowBuffer (SIMD via TensorPrimitives.Add)
        TensorPrimitives.Add(accBuffer.AsSpan(0, D), rowBuffer.AsSpan(0, D), accBuffer.AsSpan(0, D));
    }
    // Divide by L
    TensorPrimitives.Divide(accBuffer.AsSpan(0, D), T.CreateChecked(L), resultData.AsSpan(b * D, D));
}
finally { ArrayPool<T>.Shared.Return(rowBuffer); ArrayPool<T>.Shared.Return(accBuffer); }
```

This uses `TensorPrimitives.Add` (SIMD-accelerated) for the accumulate step. The `T` type parameter with `INumber<T>` constraint allows generic use, while `float`/`double` get SIMD paths.

### Step 5: Program.cs

Follow NivaraChess pattern: `Options` class, CLI parsing, `InteractiveWizard`, training function, inference function.

**CLI options** (see README.md for full list).

**Training flow:**
```
1. Parse options (or run wizard)
2. Generate/load data
3. Build TextTokenizer from training docs
4. Tokenize all texts → NivaraFrame
5. Create TensorDataset<float> + DataLoader<float>
6. Create TextClassifierModel<float>
7. Create CrossEntropyLoss<float> + AdamW<float>
8. Create TrainingLoop<float> and run
9. Evaluate on test data
10. Optionally save model
```

**Inference flow:**
```
1. Load model + tokenizer
2. Tokenize input text
3. model.Forward(tokenIds)
4. Softmax on logits → probabilities
5. Print class + confidence
```

**Wizard flow:**
Follow NivaraChess `InteractiveWizard.Run()` pattern:
```
What would you like to do?
  1. Train a model
  2. Predict sentiment
  3. Interactive REPL
  4. Generate data

Training settings:
  Epochs [10]:
  Batch size [32]:
  Learning rate [0.001]:
  ...
```

**Key implementation details:**

- Use `GradientUtils.Grad()` inside `TrainingLoop.Run()` (already handled by TrainingLoop)
- Use `CrossEntropyLoss<float>.Forward(logits, int[])` for integer labels
- Print epoch loss, timing, and accuracy after each epoch
- For accuracy: `model.Eval()`, iterate test batches, count correct predictions

**Null boundary in Program.cs:**
```csharp
// Data generation: TextTokenizer.Encode returns int[] — always non-null
// Tokenization to frame: NivaraColumn<int>.Create(tokenArray) — null-free
// TensorDataset.BuildTensor: handles null-stripping at boundary (already implemented)
// Model.Forward: receives ReverseGradTensor<float> from TensorDataset — null-free
// Loss: CrossEntropyLoss expects null-free tensors — guaranteed by data pipeline
```

### Step 6: Accuracy Evaluation

After training, evaluate on test data:

```csharp
static float EvaluateAccuracy(Module<float> model, DataLoader<float> loader)
{
    model.Eval();
    int correct = 0, total = 0;

    foreach (var batch in loader)
    {
        var logits = model.Forward(batch.Features);
        var predictions = GetPredictions(logits); // argmax over numClasses dimension
        var labels = batch.Labels; // [B, 1]

        for (int i = 0; i < predictions.Length; i++)
        {
            if (predictions[i] == (int)labels[i])
                correct++;
            total++;
        }
    }

    model.Train();
    return (float)correct / total;
}
```

**ArgMax implementation:** Since `ReverseGradTensor` doesn't have an ArgMax op, extract logits to array and compute argmax in C#:

```csharp
static int[] GetPredictions(ReverseGradTensor<float> logits)
{
    int batchSize = logits.shape[0];
    int numClasses = logits.shape[1];
    var predictions = new int[batchSize];

    for (int b = 0; b < batchSize; b++)
    {
        int bestClass = 0;
        float bestScore = float.MinValue;
        for (int c = 0; c < numClasses; c++)
        {
            float score = float.CreateChecked(logits[b * numClasses + c]);
            if (score > bestScore)
            {
                bestScore = score;
                bestClass = c;
            }
        }
        predictions[b] = bestClass;
    }

    return predictions;
}
```

### Step 7: ADR-001 Cleanup Opportunities

During implementation, audit the following for null-handling cleanup:

1. **`ReverseGradOperations.AccumulateGradient`** — the `stripGradientNulls` parameter and dual null paths. Per ADR-001, gradients should be null-free. The fast path (lines 1617-1626) already handles the common case. The else branch (lines 1628-1631) with `WithoutNulls()` calls is the fallback. **Opportunity:** If all callers guarantee null-free gradients, the `stripGradientNulls` parameter could default to `false` and the null-handling branches could be removed. However, this is a core library change — defer to a separate cleanup PR.

2. **`ModelSerializer.BuildParameterEntries`** — handles null masks in serialized tensors. **Keep as-is** — `NivaraColumn` stays nullable in storage; serialization must handle both cases.

3. **`Module.CloneTensor`** — handles null masks when cloning. **Keep as-is** — defensive, needed for storage compatibility.

4. **`TensorDataset.BuildTensor`** — already handles null-stripping at boundary (lines 60-93). **Keep as-is** — this IS the boundary enforcement point per ADR-001.

5. **`Embedding.Forward`** — creates null-free one-hot matrices and null-free result tensors. **Already clean.**

6. **`TextClassifierModel.Forward`** — must guarantee all intermediate tensors are null-free. The `Embedding.Forward` produces null-free output. `Linear.Forward` produces null-free output (MatMul + Add on null-free inputs). `Activation.Relu` is element-wise on null-free input. **Will be clean by construction.**

### Step 8: Testing Strategy

The sample itself serves as an integration test. Additional unit tests to add:

| Test | File | What it validates |
|------|------|-------------------|
| `TextTokenizer_VocabBuild_CorrectCounts` | tests/ | Min-freq filtering works |
| `TextTokenizer_EncodeDecode_RoundTrip` | tests/ | Encode → Decode preserves text |
| `TextTokenizer_SpecialTokens_CorrectIndices` | tests/ | PAD=0, UNK=1, BOS=2, EOS=3 |
| `TextClassifierModel_Forward_CorrectShape` | tests/ | Output shape [B, numClasses] |
| `TextClassifierModel_GradientFlows_AllParams` | tests/ | Every parameter gets a gradient |
| `MeanPoolBatched_CorrectValues` | tests/ | Average over L rows is correct |
| `MeanPoolBatched_GradientFlows` | tests/ | Gradient distributes evenly across L positions |
| `DataGenerator_CorrectLabelDistribution` | tests/ | ~50/50 positive/negative |
| `TrainingLoop_Converges_LossDecreases` | tests/ | Loss decreases over epochs |

### Step 9: Performance Profiling

After implementation, measure:

1. **Training throughput:** samples/second for batch size 32
2. **Embedding forward time:** dominated by one-hot + MatMul
3. **MeanPool time:** the new operation — compare scalar vs SIMD path
4. **End-to-end epoch time:** compare with NivaraGpt (different architecture, but useful baseline)

**Expected results:**
- 1000 samples, batch size 32, 10 epochs → ~5-15s total
- Embedding is the bottleneck (one-hot matrix creation + MatMul)
- MeanPool is cheap relative to Embedding
- TrainingLoop overhead is minimal for this model size

## Implementation Order

| Phase | Files | Effort | Dependencies |
|-------|-------|--------|--------------|
| 1 | `NivaraClassifier.csproj` | Tiny | None |
| 2 | `TextTokenizer.cs` | Small | None |
| 3 | `DataGenerator.cs` | Small | None |
| 4 | `TextClassifierModel.cs` (with MeanPool) | Medium | Embedding<T>, Linear<T>, ReverseGradOperations |
| 5 | `Program.cs` (CLI + training + inference) | Medium | All above + TrainingLoop, DataLoader, TensorDataset, CrossEntropyLoss, AdamW, ModelSerializer |
| 6 | Tests | Medium | All above |

## Risk Areas

1. **MeanPool backward correctness** — the manual backward pass must distribute gradient equally across L positions. Validate with gradient checking (finite differences).

2. **Embedding input type** — `Embedding<T>.Forward` expects `ReverseGradTensor<T>` where `T` is the numeric type. Token IDs are integers, but `T` is `float`. The `int.CreateChecked(input.Data[i])` conversion handles this (existing code in `Embedding.ForwardBatched`).

3. **TensorDataset column naming** — `TensorDataset<T>` expects feature columns in the frame. Token positions must be named `t0`, `t1`, ..., `t{L-1}`. Ensure consistent naming between tokenizer output and frame construction.

4. **CrossEntropyLoss with batch > 1** — the `Forward(logits, int[])` overload builds one-hot internally. Validate that batch dimension is handled correctly.

## Future Improvements (not in this implementation)

- **`LinearClassifier<T>` in core** — Gap F from `samples/README.md`
- **Subword tokenization** — BPE or WordPiece for real-world text
- **Attention-based classifier** — replace MeanPool with self-attention
- **Pre-trained embeddings** — GloVe/Word2Vec loading
- **Multi-class support** — extend beyond binary classification
- **Learning rate scheduling** — linear warmup + decay
