# Examples

This folder contains sample projects and documentation demonstrating Nivara's capabilities in .NET-native machine learning.

## [1-PyTorch.md](1-PyTorch.md) — Cross-Framework Parity: PyTorch ↔ Nivara

Nivara provides .NET developers correct autograd without leaving the ecosystem — no Python runtime, no 900 MB PyTorch install, no GPU required. These parity examples prove it: for CPU-based training, inference, and gradient computation, Nivara's forward and backward autograd produce effectively identical results to PyTorch.

The examples include:
- **Backward-mode (MLP FraudNet)**: Trains an identical 3-layer MLP in both frameworks and compares loss curves, validating reverse-mode autograd, optimizers, and training loop correctness.
- **Forward-mode (JVP Parity)**: Computes Jacobian-vector products for 6 canonical operations and compares, validating forward-mode autograd.

Results show <0.04% loss-curve divergence and 1e-5 JVP tolerance.

## [2-MicroGpt.md](2-MicroGpt.md) — Character-level Transformer on Nivara AutoDiff

A faithful per-position port of Andrej Karpathy's microgpt.py that trains a miniature GPT language model on the makemore names dataset (~32K names). This is the first Nivara showcase example, proving that Nivara's AutoDiff engine can train a real transformer — not just MLPs — with correct gradients, comparable performance to PyTorch (2.4× faster on CPU), and no external dependencies beyond the Nivara core library.

Key characteristics:
- Per-position forward/backward (not batched) — each token attends only to cached past tokens
- Weight tying by default (output projection reuses token embedding matrix)
- Uses `Embedding<T>`, `Linear<T>`, RMSNorm, SoftmaxList, and ConcatHeads via PadRight/PadLeft selection matrices

---

# Example Approach Capture & Gap Analysis

## 1. Approach Being Taken

### NivaraChatClient Example (`samples/NivaraChatClient.md`)

**Direction:** Nivara-trained domain-specific models as custom `Executor` subclasses in `Microsoft.Agents.AI.Workflows` graphs, mixed with LLM-backed `ChatClientAgent` nodes.

**Architecture (conceptual):**
```
Input → [NivaraSentimentExecutor] → [NivaraEntityExtractor] → [LLMAgent] → [NivaraValidator] → Output
```

Key characteristics of the approach:
- No `IChatClient` implementation — the prior standalone-chat-client direction was abandoned
- Training pipelines are separate from inference workflows (different projects/phases)
- Each domain model (sentiment, entity, validator) is a `Module<T>` subclass trained via `TrainingLoop<T>`
- Each is wrapped in an `Executor` subclass with `[MessageHandler]` for type-safe routing
- `ModelSerializer` bridges training output to inference input (JSON save/load)
- The example builds toward a hybrid workflow: deterministic Nivara nodes + stochastic LLM node

### MicroGpt Example (`samples/MicroGpt/`)

**Direction:** Per-position faithful C# port of Karpathy's microgpt.py, proving Nivara's AutoDiff engine can train a real transformer.

Key characteristics:
- Character-level GPT on the makemore names dataset (~32K names)
- Per-position forward/backward (not batched) — each token attends only to cached past tokens
- Weight tying by default (output projection reuses token embedding matrix)
- Explicit `List<Parameter<T>>` management instead of `Module<T>` inheritance
- Uses `Embedding<T>`, `Linear<T>`, RMSNorm, SoftmaxList (per-position softmax), ConcatHeads via PadRight/PadLeft selection matrices
- 2.4× faster than the Python original on CPU via SIMD tensor ops

### Relationship

They are complementary: MicroGpt proves the AutoDiff engine works at transformer scale; NivaraChatClient is the planned serving/integration showcase. They don't share code.

---

## 2. What We Could Do Better (Things We Know We Should Improve)

### 2.1 MicroGpt — Not a `Module<T>` subclass

**Current:** `MicroGptModel<T> : IDisposable` manages parameters via `allParams` list manually.

**Better:** Make it `Module<T>` so that `StateDict()`, `LoadStateDict()`, `ModelSerializer`, and `TrainingLoop<T>` work out of the box.

**Why we didn't:** At the time, `Embedding<T>` was not a `Module<T>` and couldn't participate in the module tree. Now that Gap 1 is fixed, MicroGptModel should be refactored to extend `Module<T>` and use `RegisterModules`/`RegisterParameters`.

### 2.2 Per-position softmax instead of batched CrossEntropyLoss

**Current:** `SoftmaxList()` — a manual list-based per-position softmax — plus manual NLL computation via one-hot + MatMul + Negate.

**Better:** Use `CrossEntropyLoss<T>.Forward()` for the batched loss.

**Why we didn't:** Shape compatibility issues at the time (`CrossEntropyLoss` expected specific shapes that didn't match the per-position output). Also, the per-position backward is a deliberate design choice (lower peak memory) — but the *loss computation* could still use `CrossEntropyLoss`.

### 2.3 ConcatHeads via expensive selection matrices

**Current:** `PadRight`/`PadLeft` create per-call identity matrices via `MatMul` with selection matrices.

**Better:** A dedicated `Concat` op that does in-place concatenation without materializing identity matrices.

**Why we didn't:** At MicroGpt scale (nEmbd ≤ 64) the overhead is negligible. A `Concat` op with a correct backward pass (splitting gradient along the concatenation axis) would be more efficient at scale and cleaner conceptually.

### 2.4 Embedding via one-hot + MatMul instead of Gather

**Current:** `Embedding<T>.Forward(int)` creates a one-hot vector and does `MatMul(oneHot, weight)`. Batched forward uses the same approach with a batched one-hot matrix.

**Better:** `ReverseGradOperations.Gather<T>` is now available and could replace the one-hot path for O(nEmbd) per lookup instead of O(vocabSize × nEmbd). For small vocabs (MicroGpt's ~55 chars), the current approach is adequate.

### 2.5 NivaraChatClient is documentation-only, not runnable code

**Current:** `NivaraChatClient.md` is a spec/plan, not a compilable example project.

**Better:** A real `samples/NivaraWorkflow/` project with `Program.cs`, `NivaraSentimentExecutor.cs`, training pipelines, and CSV data.

**Why we didn't:** The document was rewritten to capture the new direction after the IChatClient approach was replaced. Building the actual example is the next implementation step. Also requires Gap resolution (Embedding as Module, Gather, etc.) first.

### 2.6 No end-to-end integration test for the Agent Framework path

**Current:** No test validates that a Nivara `Executor` actually works inside a `WorkflowBuilder` graph.

**Better:** A test that constructs a minimal workflow with a Nivara executor, runs `InProcessExecution.RunAsync`, and asserts correct routing/output.

**Why we didn't:** Would require adding Agent Framework as a test dependency, which was deferred.

---

## 3. What's Missing from the Library (Gaps That Would Enable a More Complete/Perfect Implementation)

### Gap A: ~~Embedding\<T\> does not extend Module\<T\>~~ **RESOLVED**

`Embedding<T>` now extends `Module<T>`. It registers its weight via `RegisterParameters(weight)` in the constructor, inherits `StateDict()`/`LoadStateDict()`, `Train()`/`Eval()`, `GetParameters()`, and `Dispose()` from `Module<T>`. The `Forward(int tokenId)` convenience method is preserved; the abstract `Forward(ReverseGradTensor<T>)` override extracts a scalar int and delegates. MicroGptModel was updated to use `emb.GetParameters().Values` instead of the old `emb.Parameters` property. Batched `Forward(ReverseGradTensor<T> tokenIds)` uses one-hot + MatMul; Gather (resolved) could replace it for large vocabs.

### Gap B: ~~Embedding\<T\> has no batch/lookup Forward for token IDs~~ **RESOLVED**

`Embedding<T>.Forward(ReverseGradTensor<T> tokenIds)` now accepts batched input:
- `[L]` 1D tensor → `[L, nEmbd]` output
- `[B, L]` 2D tensor → `[B, L, nEmbd]` output
- Scalar (length=1) delegates to `Forward(int tokenId)` for backward compatibility

Implementation: builds a batched one-hot matrix `[totalTokens, vocabSize]`, performs a single MatMul against the weight matrix `[vocabSize, nEmbd]`. The MatMul kernel uses `Parallel.For` + `TensorPrimitives.Dot` (SIMD-accelerated). Gradients accumulate correctly for repeated tokens via autograd's `AccumulateGradient` in the MatMul backward pass. Token IDs are validated at lookup time with descriptive error messages.

For large vocabularies (32K–128K), this is still O(vocabSize × nEmbd) per batch due to one-hot expansion. The `Gather` operation is now available (Gap C resolved) and could replace this path.

### Gap C: ~~No Gather operation~~ **RESOLVED**

`ReverseGradOperations.Gather<T>(ReverseGradTensor<T> source, int[] indices, int axis)` is now implemented. Supports 1D and multi-dimensional source tensors. Forward copies stride-contiguous rows by index; backward scatters gradients back to source positions via accumulation (handles duplicate indices correctly). Null mask propagation is included. Shape metadata is correctly propagated for multi-dimensional sources. 8 tests cover forward correctness (1D, 2D), backward gradient accumulation, duplicate indices, out-of-range, nulls, empty indices, and sum-then-backward flow.

**Note:** Embedding's batched `Forward` still uses one-hot + MatMul. It could be refactored to use `Gather` internally for O(nEmbd) per lookup instead of O(vocabSize × nEmbd), but the current approach is functionally correct and adequate for small vocabs.

### Gap D: ~~No way to convert integer data to a differentiable tensor~~ **RESOLVED**

Gather's API accepts `int[]` indices directly (not a tensor), so integer tensors are unnecessary. The `Gather<T>` backward pass accumulates gradients at source positions by index — no autograd through the indices themselves. This is the standard ML framework approach (indices are never differentiated).

### Gap E: No batched transformer block (attention + MLP)

**Not previously documented.**

**Current:** MicroGptModel does per-position attention. A `TransformerBlock<T> : Module<T>` that handles batched multi-head attention with causal masking does not exist.

**Impact:** Anyone building a transformer must reimplement the entire block with per-position loops. No building block to compose with.

**Fix:** Add `TransformerBlock<T>` in `AutoDiff/Nn/`:
- Constructor: `(int nEmbd, int nHead, double dropout = 0.0)`
- Forward: `(ReverseGradTensor<T> x)` — batched `[B, L, nEmbd]` → `[B, L, nEmbd]`
- Internally: Pre-LN RMSNorm, QKV projections, scaled dot-product attention with causal mask, output projection + residual, MLP + residual, dropout

### Gap F: No LinearClassifier\<T\> convenience class

**Already documented** as Gap 4.

**Impact:** The simplest classifier (linear → softmax) requires `Sequential(Linear, Softmax)` each time. Not a correctness issue, but a developer-experience gap.

**Fix:** Add `LinearClassifier<T> : Module<T>` in `AutoDiff/Nn/`:
```csharp
public sealed class LinearClassifier<T> : Module<T> where T : struct, INumber<T>
{
    private readonly Linear<T> linear;
    public LinearClassifier(int inFeatures, int numClasses)
    {
        linear = new Linear<T>(inFeatures, numClasses);
        RegisterModules(linear);
    }
    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
        => ReverseGradOperations.Softmax(linear.Forward(input));
}
```

### Gap G: Vocabulary/feature extraction pipeline is not part of the library

**Not previously documented.**

**Current:** MicroGpt has `Tokenizer.cs` (char-level). The NivaraChatClient example mentions `TextTokenizer.cs` and `TextToFeature()` as ad-hoc example code.

**Impact:** Every example project must write its own tokenizer / feature extractor. No "load text → feature vector" pipeline exists in the library.

**Fix:** Not necessarily a library issue — this is application-level code. But a `Nivara.Extensions.Text` package with a simple bag-of-words or TF-IDF vectorizer would make the NivaraChatClient example dramatically more self-contained.

### Gap H: No CategoricalCrossEntropy accepting integer labels

**Not previously documented.**

**Current:** `CrossEntropyLoss<T>.Forward()` takes `(logits, targets)` where targets is a one-hot tensor. MicroGpt works around with manual `LogSoftmax` + `MatMul(one_hot)` + `Negate`.

**Better:** Overload that accepts integer targets (class indices) directly:
```csharp
public ReverseGradTensor<T> Forward(ReverseGradTensor<T> logits, int[] targets)
```

**Why:** All classification tasks use integer labels. Forcing one-hot conversion at the loss boundary is boilerplate.

### Gap I: No batched NLL loss

**Not previously documented.**

**Current:** Only per-position NLL (implicit in MicroGpt's manual loop). `CrossEntropyLoss<T>.Forward()` expects batched logits `[B, C]` and batched targets `[B, C]` (one-hot), but its formula (sum over batch/N) is correct.

**Impact:** CrossEntropyLoss already works for batched input — just providing the `int[]` targets overload would resolve this. No fix needed beyond Gap H.

### Gap J: No Dropout handling for Embedding

**Not previously documented.**

**Current:** `Dropout<T>` exists as a `Module<T>` but there's no `EmbeddingDropout` (a.k.a. word dropout) or attention dropout. The standard transformer uses dropout after embedding, after attention, after MLP, and on residual paths.

**Impact:** MicroGpt has no dropout. Real models need it for regularization.

**Fix:** Not a blocker — `Dropout<T>` can already be composed in a `Sequential` pipeline. But the NivaraChatClient example's training pipelines should use it.

### Gap K: No random sampling utilities for inference

**Not previously documented.**

**Current:** MicroGpt has inline sampling logic (softmax → sample from distribution). No reusable `CategoricalDistribution<T>` or `Sampler<T>` exists.

**Impact:** Every inference path (MicroGpt generation, entity extraction with beam search, etc.) reimplements sampling.

**Fix:** Add `Sampler<T>` in `AutoDiff/Nn/`:
- `Sample(ReverseGradTensor<T> logits, double temperature = 1.0)` → int
- `SampleTopK(...)` → int
- `SampleTopP(...)` → int

### Gap L: ModelSerializer cannot load into non-Module\<T\> models

**Not previously documented.**

**Current:** `ModelSerializer.Load<T>(Module<T> model, string path)` requires an already-instantiated `Module<T>` to load state dict into. `ModelSerializer.LoadModel(string path)` does not exist as a non-generic entry point.

**Impact:** The NivaraChatClient example uses a fictional `ModelSerializer.LoadModel("sentiment_model.json")` that doesn't exist in the library. The actual API is `ModelSerializer.Load<T>(model, path)`.

**What's missing:** A factory or convenience method that:
1. Reads the JSON header (format, type, architecture)
2. Instantiates the correct `Module<T>` subclass
3. Loads the state dict into it
4. Returns the typed module

This requires either a discriminated JSON format (architecture name stored in file) or the convention-based approach where the caller creates the model and calls `Load` (current approach).

### Gap M: Agent Framework packages not referenced anywhere

**Already documented** as Gap 5.

**Impact:** The example project (`samples/NivaraWorkflow/`) must reference `Microsoft.Agents.AI`, `Microsoft.Agents.AI.Workflows`, and `Microsoft.Extensions.AI`. Core stays clean.

**Fix:** Documentation/example concern only — no core change.

### Gap N: No example project for NivaraChatClient yet

**Not previously documented as a separate gap.**

**Impact:** The best showcase of Nivara's value proposition (deterministic model in an Agent Framework workflow) only exists as a markdown spec.

**Fix:** Build `samples/NivaraWorkflow/` as a real runnable project. This will also validate all the gaps above.

---

## Priority Order for Gaps

| Priority | Gap | Effort | Impact |
|----------|-----|--------|--------|
| ~~P0~~ | ~~A: Embedding as Module~~ | ~~Small~~ | **RESOLVED** |
| P0 | N: Build the actual example project | Large | Validates everything; biggest single improvement |
| ~~P1~~ | ~~C + D: Gather op~~ | ~~Medium~~ | **RESOLVED** — `int[]` indices, backward scatter-add, 8 tests |
| ~~P1~~ | ~~B: Batched embedding Forward~~ | ~~Small~~ | **RESOLVED** — one-hot + MatMul; Gather available for large vocab |
| P1 | E: Batched TransformerBlock | Medium | Building block for any sequence model |
| P2 | F: LinearClassifier | Tiny | Developer convenience |
| P2 | H: Integer-label CrossEntropyLoss | Small | Reduces boilerplate in all classification tasks |
| P2 | K: Sampler utilities | Small | Reusable inference sampling |
| P3 | G: Text feature extraction | Large | Application-level, could be an extension |
| P3 | J: Embedding dropout | Tiny | Already covered by Dropout<T> in Sequential |
| P3 | L: LoadModel factory | Small | Nice-to-have API convenience |
| P3 | M: Agent Framework refs | None | Example concern only |
