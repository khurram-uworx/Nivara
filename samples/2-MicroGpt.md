# MicroGpt — Character-level Transformer on Nivara AutoDiff

## Purpose

Faithful per-position port of [Andrej Karpathy's microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). Trains a miniature GPT language model on the makemore names dataset (~32K names) to generate human names character-by-character.

This is the first Nivara showcase example. It proves that Nivara's AutoDiff engine can train a real transformer — not just MLPs — with correct gradients, comparable performance to PyTorch (2.4× faster on CPU), and no external dependencies beyond the Nivara core library.

## Architecture

```
Token Embedding → Position Embedding → N × Transformer Block → Output Projection
```

Each Transformer Block (per-position with KV cache):
- **RMSNorm → Multi-Head Self-Attention** (per-head dot products, KV cache) → Residual
- **RMSNorm → MLP** (expand 4×, squared ReLU, compress) → Residual

Key characteristics:
- **Per-position forward** — one token at a time, each token attends only to cached past tokens
- **Weight tying** (default) — output projection reuses the token embedding weight matrix
- **Differentiable concatenation** of attention heads via PadRight/PadLeft selection matrices
- **Per-position backward** — each token's NLL loss is backpropagated immediately, accumulating gradients on shared parameters (reduces peak memory vs. backpropagating all positions at once)

## Files

| File | What it is | Key details |
|------|------------|-------------|
| `MicroGpt/Program.cs` | Entry point | CLI arg parsing, data download, training loop, generation, benchmarking |
| `MicroGpt/MicroGptModel.cs` | GPT model class | `Embedding<T>` + `Linear<T>` for parameters; `Forward(tokenId, posId, keys, values)` per-position KV-cache forward pass; `ConcatHeads`, `PadRight`, `PadLeft`, `BroadcastScalar`, `SoftmaxList` helpers |
| `MicroGpt/Tokenizer.cs` | Char-level tokenizer | Builds vocab from dataset characters; adds `<BOS>`/`<EOS>` tokens; `Encode(text)` and `Decode(tokenId)` |
| `MicroGpt/MicroGpt.csproj` | Project file | References `Nivara` core only — no external packages |

## CLI

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
--help, -h              Show this help
```

Karpathy defaults for A vs C comparison:
```
dotnet run -- --block-size 16 --beta1 0.85 --beta2 0.99 --init-std 0.08 --no-weight-tying --lr-decay --temperature 0.5 --samples 20
```

## Performance (from README)

All runs: 1000 steps, same ~32K names dataset.

| Implementation | Arch params | Time | Speedup vs baseline |
|---|---|---|---|
| AutoGrad-Engine (C#, scalar) | block=8, weight tying | 58.7s | 1.0× |
| **Nivara** (C#, tensor) | block=8, weight tying | **34.9s** | **1.7×** |
| **Nivara** (C#, tensor) | block=16, Karpathy hparams | **40.4s** | **1.5×** |
| microgpt.py (Python, scalar) | block=16, Karpathy hparams | 97.2s | 0.6× |

Nivara is 2.4× faster than the Python original on CPU due to SIMD tensor ops.

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

## New ops added to Nivara for this example

| Op | File | Purpose |
|----|------|---------|
| `Pow` | `ReverseGradOperations.cs` | `x^exponent` with backward `exponent * x^(exponent-1) * grad` |
| `RMSNorm` | `ReverseGradOperations.cs` | Fused RMS normalization `x * rsqrt(mean(x²) + eps)` |
| `Slice` | `ReverseGradOperations.cs` | Differentiable sub-tensor extraction via selection matrix |
| `Embedding<T>` | `AutoDiff/Nn/Embedding.cs` | Token embedding lookup (one-hot + MatMul) |

## Design decisions worth knowing

1. **No Module\<T\> for MicroGptModel** — manages parameters explicitly via `List<Parameter<T>>` rather than inheriting from `Module<T>`. This was a choice for clarity and because `Embedding<T>` wasn't a `Module<T>` at the time.

2. **Embedding via one-hot + MatMul** — avoids needing a Gather op; keeps the lookup differentiable.

3. **ConcatHeads via selection matrices** — PadRight/PadLeft with identity matrices. Not the most efficient but correct at MicroGpt scale (nEmbd ≤ 64).

4. **Loss via LogSoftmax + MatMul(one_hot) + Negate** — instead of `CrossEntropyLoss<T>`, to avoid shape compatibility issues.

5. **Per-position, not batched** — no batch dimension. Each forward call is one token. This is the defining difference from a proper batched transformer.

## How to run

```bash
cd samples/MicroGpt
dotnet run
```

## Relationship to the NivaraChatClient example

MicroGpt is the **training** showcase — it proves Nivara can train a transformer. The NivaraChatClient example (`samples/NivaraChatClient.md`) is the **serving/integration** showcase — it shows how a trained Nivara model participates in a Microsoft Agent Framework workflow alongside an LLM. They are complementary: MicroGpt trains the model, NivaraChatClient puts it to work.
