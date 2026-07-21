# MicroGPT on Nivara AutoDiff

A miniature GPT language model implemented using Nivara's tensor-based AutoDiff engine. Faithful per-position port of [Andrej Karpathy's microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), learning to generate human names character-by-character.

This is the first Nivara showcase example. It proves that Nivara's AutoDiff engine can train a real transformer — not just MLPs — with correct gradients, comparable performance to PyTorch (2.4× faster on CPU), and no external dependencies beyond the Nivara core library.

## How to run

```bash
cd samples/MicroGpt
dotnet run
```

With Karpathy's hyperparameters (for A vs C comparison):
```bash
dotnet run -- --block-size 16 --beta1 0.85 --beta2 0.99 --init-std 0.08 --no-weight-tying --lr-decay --temperature 0.5 --samples 20
```

## Architecture

```
Token Embedding → Position Embedding → N × Transformer Block → Output Projection (weight-tied by default)
```

Each Transformer Block:
- **RMSNorm → Multi-Head Self-Attention** (per-head dot products, KV cache) → Residual
- **RMSNorm → MLP** (expand 4×, squared ReLU, compress) → Residual

Key features:
- **Per-position forward** with KV cache — each token only attends to past tokens
- **Per-head attention** with differentiable softmax over cached positions
- **Weight tying** — output projection reuses the token embedding matrix (optional: separate `lm_head`)
- **Differentiable concatenation** via selection matrices (no non-differentiable ops)

## CLI Parameters

```
--n-embd <int>          Embedding dimension (default: 16)
--n-layer <int>         Number of transformer layers (default: 1)
--block-size <int>      Context window / block size (default: 8)
--n-head <int>          Number of attention heads (default: 4)
--steps <int>           Training steps (default: 1000)
--lr <float>            Learning rate (default: 0.01)
--beta1 <float>         Adam beta1 (default: 0.9)
--beta2 <float>         Adam beta2 (default: 0.95)
--init-std <float>      Weight init std dev (default: 0.02)
--no-weight-tying       Use separate lm_head instead of weight tying
--lr-decay              Linear LR decay to zero over steps
--temperature <float>   Sampling temperature (default: 1.0)
--seed <int>            RNG seed (default: 42)
--samples <int>         Number of generated samples (default: 5)
--help, -h              Show help
```

Karpathy defaults for A vs C comparison:
```bash
dotnet run -- --block-size 16 --beta1 0.85 --beta2 0.99 --init-std 0.08 --no-weight-tying --lr-decay --temperature 0.5 --samples 20
```

## Comparison Results

All runs: 1000 steps, same ~32K names dataset (makemore/names.txt).

### A vs B: Nivara vs AutoGrad-Engine (default arch)

Same architecture: nEmbd=16, nLayer=1, blockSize=8, nHead=4, weight tying, 3,648 params.

| Implementation | Time | vs A |
|---|---|---|
| **A) AutoGrad-Engine** (C#, scalar) | **58.7s** | 1.0× |
| **B) Nivara** (C#, tensor) | **34.9s** | **1.7× faster** |

### A vs C: Nivara Karpathy-arch vs native Karpathy Python

Same architecture: nEmbd=16, nLayer=1, blockSize=16, nHead=4, separate lm_head, initStd=0.08, β₁=0.85/β₂=0.99, LR decay, 4,224 params.

| Implementation | Time | vs C |
|---|---|---|
| **C) Karpathy's microgpt.py** (Python, scalar) | **97.2s** | 1.0× |
| **B) Nivara** matching Karpathy arch (C#, tensor) | **40.4s** | **2.4× faster** |

### Summary

| Implementation | Arch params | Time | Speedup vs baseline |
|---|---|---|---|
| AutoGrad-Engine (C#, scalar) | block=8, weight tying | 58.7s | 1.0× (A baseline) |
| Nivara (C#, tensor) | block=8, weight tying | 34.9s | 1.7× vs A |
| Nivara (C#, tensor) | block=16, lm_head, Karpathy hparams | 40.4s | 1.5× vs A |
| microgpt.py (Python, scalar) | block=16, lm_head, Karpathy hparams | 97.2s | 0.6× vs A |

Nivara's tensor operations (MatMul, TensorPrimitives kernels) are faster than scalar loops in both C# and Python. The advantage is larger against Python (2.4×) because of Python interpreter overhead and Nivara's SIMD-accelerated TensorPrimitives.

## File Map

| File | Purpose |
|---|---|
| `Program.cs` | Data loading, training loop, generation, loss function, CLI args, timing |
| `MicroGptModel.cs` | Full GPT model: embeddings, transformer blocks, multi-head attention, MLP, lm_head |
| `Tokenizer.cs` | Char-level tokenizer with BOS/EOS markers |
| `MicroGpt.csproj` | Project file referencing Nivara core |

## New AutoDiff Operations Added for MicroGPT

| Op | File | Purpose |
|---|---|---|
| `Pow` | `ReverseGradOperations.cs` | `x^exponent` with backward `exponent * x^(exponent-1) * grad` |
| `RMSNorm` | `ReverseGradOperations.cs` | Fused RMS normalization `x * rsqrt(mean(x²) + eps)` |
| `Slice` | `ReverseGradOperations.cs` | Differentiable sub-tensor extraction via selection matrix |
| `Embedding<T>` | `AutoDiff/Nn/Embedding.cs` | Token embedding lookup (one-hot + MatMul) |

## How the training loop works

```
for each step:
    pick a random name document
    tokenize → list of ints with BOS/EOS
    sequence length = min(doc length, blockSize - 1)

    using (GradientUtils.Grad()):
        new KV cache lists per layer
        for each position t in sequence:
            logits = model.Forward(token[t], t, keys, values)
            loss = NLL(logits, target_token[t+1])
            scaledLoss.Backward()         // per-position backward
            lossVal += scaledLoss.Data[0]
        optimizer.Step()
        optimizer.ZeroGrad()
```

The per-position backward pattern is specific to MicroGpt — it backprops each token's loss immediately rather than accumulating all positions. This matches the original AutoGrad-Engine approach and keeps peak memory low.

## How inference/generation works

- Start with `[BOS]` token
- At each position, run `model.Forward(token[pos], pos, keys, values)` with gradient tracking disabled
- Apply softmax with temperature to the output logits
- Sample from the probability distribution
- Append to token list; stop at `[EOS]` or blockSize
- Decode tokens (skip BOS/EOS) back to characters

## Design Decisions

1. **Per-position (faithful) approach** — matches AutoGrad-Engine exactly, processing one token at a time with KV cache. Not batched causal attention.

2. **Per-position backward** — each position's loss is backpropagated immediately (accumulating gradients on shared parameters). This matches AutoGrad-Engine's training loop and reduces peak memory vs. accumulating all positions then backpropagating once.

3. **Embedding via one-hot + MatMul** — avoids needing a dedicated Gather op while keeping the lookup differentiable.

4. **ConcatHeads via selection matrices** — differentiable head merging using PadRight/PadLeft with identity matrices. Not the most efficient but correct — fine at MicroGPT scale (nEmbd ≤ 64).

5. **Direct loss computation** — uses `LogSoftmax + MatMul(one_hot_selector) + Negate` instead of `CrossEntropyLoss<T>` to avoid shape compatibility issues and simplify the graph.

6. **No Module<T> inheritance** — MicroGptModel manages parameters explicitly via `List<Parameter<T>>` for clarity over the module system's `GetParameters()` dictionary.

## Relationship to other examples

MicroGpt is the **training** showcase — it proves Nivara can train a transformer. The NivaraChatClient example (`samples/NivaraChatClient.md`) is the **serving/integration** showcase — it shows how a trained Nivara model participates in a Microsoft Agent Framework workflow alongside an LLM. They are complementary: MicroGpt trains the model, NivaraChatClient puts it to work.
