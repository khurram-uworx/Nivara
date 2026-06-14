# AutoDiff: Plan & Philosophy

## The drum machine

Nivara's AutoDiff is not a general-purpose deep learning framework.
It's a **drum machine with the best drum sounds** — not a general synth
that also does pianos, guitars, and orchestral hits.

The goal is a small, correct, tasteful set of gradient primitives that make
the 70–80% case feel complete. Not empty because we stopped too early.

## What's already shipped (the kick, snare, hi-hat)

These are proven by the cross-framework parity examples in `examples/README.md`:

| Piece | Status | Verified |
|-------|--------|----------|
| `ReverseGradTensor<T>` (backward-mode) | ✅ | 0.04% loss-curve parity, 50-epoch MLP training |
| `ForwardGradTensor<T>` (forward-mode) | ✅ | 6/6 JVP cases at 1e-5 tolerance |
| `ComputationGraph` + topo-sort backward | ✅ | Cycle detection, validation |
| Ops: Add, Sub, Mul, Div, Negate, MatMul, Transpose, Sum, Mean | ✅ | |
| Activations: ReLU, LeakyReLU, Sigmoid, Tanh, Exp, Log, Softmax, LogSoftmax | ✅ | |
| Regularization: Dropout, Abs, Clip | ✅ | |
| VAE: KLDivergence, SampleNormal (reparameterization) | ✅ | |
| `Module<T>`, `Parameter<T>`, `Sequential<T>` | ✅ | |
| `Linear<T>` layer | ✅ | |
| Loss functions: MSELoss, L1Loss, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss | ✅ | |
| Optimizers: SGD (momentum), Adam, AdamW | ✅ | |
| `TrainingLoop<T>`, `DataParallelTrainer<T>` | ✅ | |
| `TensorDataset<T>`, `DataLoader<T>`, `Batch<T>` | ✅ | |
| `ModelSerializer` (JSON+base64 checkpoint format) | ✅ | |
| Initializers: Kaiming, Xavier, Uniform, Normal (both Uniform/Normal variants) | ✅ | |
| Gradient clipping, gradient norm, detach, zero_grad | ✅ | |
| Null-mask propagation throughout all operations | ✅ | |

This is already a functional ML toolkit for CPU `.NET` — you can define,
train, save, load, and run inference with a multi-layer network. The parity
proofs confirm the gradients are correct.

## What we're adding (the ride cymbal, the aux in)

Two additions make the existing set feel **complete for practical workflows**.
Both are small, high-leverage additions that close the last common friction
points.

### 1. `NoGrad` scope

**Why it matters:**

Every inference pass today builds a computation graph and runs dropout in
training mode. Users must manually track `model.Train()`/`model.Eval()`
and accept wasteful graph construction. A `NoGrad` scope context manager:

- Suppresses gradient tracking inside the block
- Forces modules to eval mode (dropout disabled)
- Avoids unnecessary graph memory
- Makes inference code read naturally

**Design sketch:**

```csharp
using (GradientUtils.NoGrad())
{
    var pred = model.Forward(input);
    // No graph built, no gradients tracked
}
```

Implementation: ~50 lines. A `IDisposable` struct that saves/restores a
thread-local `_noGrad` flag. `ReverseGradTensor` constructors and
`OpNode` registration check the flag and skip graph wiring.

### 2. `state_dict()` / `load_state_dict()`

**Why it matters:**

The standard pattern for model surgery — freeze layers, fine-tune, inspect
weights, transfer learn — requires a well-known dictionary representation.
Today we have `Parameters()` (flat dictionary) and `ModelSerializer`
(JSON file round-trip) but no in-memory round-trip that handles
naming, filtering, and partial loading.

**Design sketch:**

```csharp
// Save
var state = model.StateDict();                    // Dictionary<string, ReverseGradTensor<T>>

// Load (full)
model.LoadStateDict(state);

// Load (partial — freeze all but last layer)
state.Remove("linear_2.weight");
state.Remove("linear_2.bias");
model.LoadStateDict(state);

// Serialize for wire/disk
var json = ModelSerializer.StateDictToJson(state);
var restored = ModelSerializer.JsonToStateDict(json);
```

Implementation: ~100 lines. `Module.StateDict()` walks named parameters
recursively. `Module.LoadStateDict()` matches keys and copies values with
shape validation. `ModelSerializer` gets two thin public wrappers.

## What we are NOT adding (and why)

| Feature | Reason excluded |
|---------|----------------|
| Conv1D/2D/3D | Spatial vision models are GPU territory. Enterprise CPU ML is tabular/sequence at most. |
| BatchNorm, LayerNorm | Needed for deep vision/text but almost never for the shallow-to-moderate tabular ML that targets CPU. |
| RNN, LSTM, GRU | Sequence models on CPU are niche. If needed, compose with Linear + manual state. |
| Transformer, Attention | Large language models run on GPU or via Microsoft.Extensions.AI + external API. |
| Embedding layer | Categorical features should be handled via feature engineering (one-hot, hashing) in the DataFrame layer, not via learned embeddings in the AutoDiff. |
| GELU, Swish, ELU, SELU, PReLU | Activations that matter for deep vision/transformer. Not needed for the tabular 80%. |
| LR schedulers | Nice-to-have but trivial to implement manually (`optimizer.ParameterGroups[0].LearningRate *= 0.95`). |
| Adagrad, RMSprop, Adadelta, Adamax, NAdam, RAdam | SGD + Adam + AdamW covers the vast majority of practice. CAPEX on more optimizers has decreasing returns. |
| `retain_graph` | Multi-loss scenarios exist but are rare. The added graph lifecycle complexity isn't worth it. |
| Mixed precision / GradScaler | GPU-only concern. |
| Int/long/half type support | Gradient computation lives in float/double. Integer tensors for indexing are the DataFrame's job. |
| ONNX export | If you need TorchScript / ONNX-level interop, you should use PyTorch. Nivara's serialization is for .NET-to-.NET. |
| `distributed` / DDP | Distributed training on CPU is not a realistic use case. |

## The test for every addition

Before adding a feature, ask:

1. Does a .NET developer need this to ship a gradient-based model to
   production on CPU?
2. Does this make the existing surface feel **more complete** rather than
   **more expansive**?
3. Can we implement it in <100 lines and with zero new dependencies?

If the answer to any of these is "no", it belongs in `Nivara.Extensions`
or not in Nivara at all.

## Implementation order

1. **NoGrad scope** — trivial, unlocks clean inference, unblocks nothing
2. **state_dict / load_state_dict** — small, unlocks model surgery pattern
3. Existing surface: lint, harden null-mask edge cases, improve diagnostics

After that, the AutoDiff surface is **complete for the 80%**. Further
investment goes into the columnar analytics engine (Arrow, Parquet, lazy
queries, joins, grouping, schema, null semantics) per `TENSORS.md`.

## Summary

```
   What we have: A complete CPU ML toolkit with proven gradient correctness
                 (Linear, optimizers, losses, training loop, serialization)

  What we add  : NoGrad scope    (10 lines of logic, 50 lines total)
                 state_dict/...  (100 lines, enables model surgery)

   What we ship: A drum machine with the best drum sounds.
                 Not a general synth.

   What's next : Columnar engine. See TENSORS.md.
```
