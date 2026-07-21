# NivaraChess — Neural Chess Position Evaluator

A sample project demonstrating Nivara's autograd engine for a non-NLP domain. Trains a neural network to evaluate chess positions, progressing from simple material counting to approximating Stockfish's evaluation via knowledge distillation.

**Target audience:** Game developers, Unity/ML-adjacent devs exploring .NET-native ML.

## What it does

NivaraChess trains a model to predict centipawn scores for chess positions. It showcases:

- **Reverse-mode autograd** with real training loops (`TrainingLoop<T>`, `DataLoader<T>`)
- **Sparse embeddings** (`SparseEmbedding<T>`, `SparseEmbeddingBag<T>`) for NNUE-style feature transformers
- **Model serialization** (`ModelSerializer.Save`/`Load`) for save/load
- **`IEmbeddingGenerator<T>`** integration for position embeddings
- **Stockfish UCI integration** for training data generation (knowledge distillation)

## Phases

| Phase | Architecture | Training labels | Features | Key Nivara APIs |
|-------|-------------|----------------|----------|----------------|
| 1 | MLP (material → hidden → output) | Material piece counts | 12 categorical counts | `Linear<T>`, `Activation.Relu`, `MSELoss<T>`, `AdamW<T>` |
| 2 | NNUE halfKP (sparse embedding → hidden → output) | Material + PSQT heuristic | Sparse (king index + 63 piece slots) | `SparseEmbedding<T>`, `SparseEmbeddingBag<T>` |
| 3 | Same as Phase 2 | Stockfish static NNUE evaluation (`eval` command) | Same sparse features | `StockfishEvaluator` (UCI `eval` + `ucinewgame`), knowledge distillation |

### Phase 1 — Material Evaluator
Trains a simple MLP on 12 categorical features (piece counts for white/black). Fast to train (~12s for 10K positions, 20 epochs). Demonstrates the basic autograd training pipeline.

### Phase 2 — NNUE HalfKP
Uses `SparseEmbedding<T>` (wrapping `SparseEmbeddingBag<T>`) to learn from sparse king-relative features. ~50s/epoch with 10K positions. Demonstrates sparse embedding performance and the autograd infrastructure overhead tradeoff.

### Phase 3 — Stockfish Distillation
Replaces the heuristic training labels with Stockfish's static NNUE evaluation. The neural network learns to approximate Stockfish's position assessment — compressing a strong engine's knowledge into a smaller, faster model. Uses the `eval` command (not `go depth`) for fast, crash-free evaluation with `ucinewgame` synchronization between positions. Requires Stockfish installed locally.

## Quick start

```bash
# Interactive wizard (no args) — pick action, phase, and options
dotnet run --project samples/NivaraChess

# Phase 1: train a material evaluator
dotnet run --project samples/NivaraChess -- --phase 1 --epochs 5 --save material.json

# Phase 2: train an NNUE evaluator
dotnet run --project samples/NivaraChess -- --phase 2 --epochs 10 --feature-dim 128 --save nnue.json

# Phase 3: train with Stockfish labels (requires Stockfish)
dotnet run --project samples/NivaraChess -- --phase 3 --epochs 5 \
  --stockfish C:\bin\stockfish\stockfish-windows-x86-64-avx2.exe --save stockfish.json

# Evaluate a position
dotnet run --project samples/NivaraChess -- --load nnue.json --fen "start"

# Interactive REPL
dotnet run --project samples/NivaraChess -- --load nnue.json --interactive

# UCI engine mode (for GUI integration)
dotnet run --project samples/NivaraChess -- --load nnue.json --uci

# Embedding demo (compute position embeddings + cosine similarity)
dotnet run --project samples/NivaraChess -- --load nnue.json --embed
```

## CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--phase <int>` | 1 | Data generation phase: 1=material, 2=NNUE halfKP, 3=Stockfish |
| `--num-positions <int>` | 10000 | Number of random positions to generate for training |
| `--epochs <int>` | 20 | Training epochs |
| `--batch-size <int>` | 128 | Batch size |
| `--feature-dim <int>` | 256 | Feature transformer size (phase 2/3) |
| `--hidden-dim <int>` | 64 | Hidden layer size (phase 1) |
| `--lr <float>` | 0.001 | Learning rate |
| `--stockfish <path>` | — | Path to Stockfish executable (required for phase 3) |
| `--save <path>` | — | Save trained model to JSON |
| `--load <path>` | — | Load model from JSON |
| `--fen <fen-string>` | — | Evaluate a single FEN position |
| `--interactive` | — | Interactive FEN evaluation REPL |
| `--embed` | — | Demo position embeddings and cosine similarity |
| `--uci` | — | Minimal UCI evaluation mode for GUI integration |
| `--seed <int>` | 42 | RNG seed |
| `--help`, `-h` | — | Show CLI help |

## Modes of use

### Training
Trains the model on randomly generated positions. Outputs epoch loss, batch timing, and validation MAE. Save the model with `--save`.

### Evaluation (`--fen`)
One-shot evaluation of a FEN string. Prints the ASCII board, model score, baseline score (material or PSQT), and the error between them.

### Interactive REPL (`--interactive`)
Loops over user-supplied FEN strings, printing evaluations. Type `start` for the starting position, or paste any FEN. `quit` to exit.

### UCI mode (`--uci`)
Speaks just enough UCI protocol (`uci`, `isready`, `position`, `go`) to be recognized by chess GUIs like Arena or CuteChess. Returns `info score cp <score>` with the model's evaluation. Does not implement move search — `bestmove` returns a placeholder.

### Embedding demo (`--embed`)
Computes position embeddings for 5 hardcoded openings using `IEmbeddingGenerator<ChessBoard>`, prints the embedding vectors, and shows pairwise cosine similarity. Demonstrates Nivara's embedding infrastructure on non-NLP data.

## Architecture

```
NivaraChess/
├── Program.cs                  # CLI entry, training loop, eval orchestration
├── ChessEvalModel.cs           # ChessEvalModelBase, ChessEvalModel (Phase 1), NnueChessEvalModel (Phase 2)
├── ChessBoard.cs               # Board representation, FEN, material+PSQT eval, halfKP features
├── ChessDataGenerator.cs       # Training data generation (material positions + Stockfish scoring)
├── ChessEmbeddingGenerator.cs  # IEmbeddingGenerator<ChessBoard> implementation
├── ChessEvalConsole.cs         # Interactive REPL, UCI mode, evaluation display
├── StockfishEvaluator.cs       # UCI process wrapper: `eval` command + `ucinewgame` sync, crash-safe with retry
└── NivaraChess.csproj          # References Nivara core + Microsoft.Extensions.AI.Abstractions
```

### Model architectures

**Phase 1 (`ChessEvalModel`):**
```
Input[12] → Linear(12, hidden) → ReLU → Linear(hidden, embedding) → ReLU → Linear(embedding, 1)
```

**Phase 2/3 (`NnueChessEvalModel`):**
```
Input[sparse 128] → SparseEmbedding(featureCount, featureDim) → ReLU → Linear(featureDim, 32) → ReLU → Linear(32, 1)
```

## Nivara APIs demonstrated

| API | Where | Purpose |
|-----|-------|---------|
| `Module<T>` | `ChessEvalModel.cs` | Model base class with parameter registration |
| `Linear<T>` | `ChessEvalModel.cs` | Fully connected layers |
| `SparseEmbedding<T>` | `ChessEvalModel.cs` (Phase 2) | Sparse feature transformer |
| `SparseEmbeddingBag<T>` | `SparseEmbedding.cs` | Sum-pooled sparse embedding lookup |
| `Activation.Relu` | `ChessEvalModel.cs` | Non-linearity |
| `MSELoss<T>` | `Program.cs` | Regression loss |
| `AdamW<T>` | `Program.cs` | Optimizer with weight decay |
| `TrainingLoop<T>` | `Program.cs` | Training orchestration |
| `DataLoader<T>` | `Program.cs` | Batched data loading |
| `TensorDataset<T>` | `Program.cs` | Frame-backed dataset |
| `ModelSerializer.Save/Load` | `Program.cs` | JSON model persistence |
| `IEmbeddingGenerator<TInput, TEmbedding>` | `ChessEmbeddingGenerator.cs` | MS.Extensions.AI embedding interface |
| `TensorPrimitives.CosineSimilarity` | `Program.cs` | SIMD embedding comparison |

## Requirements

- .NET 10.0 SDK
- Nivara core library (`src/Nivara/Nivara.csproj`)
- Stockfish executable (only for phase 3 training)
- `Microsoft.Extensions.AI.Abstractions` 10.8.1

## Library gaps this example exposed and resolved

NivaraChess drove several core library additions and fixes. The original spec identified these gaps; all were resolved during implementation.

| Gap | Problem | Resolution |
|-----|---------|------------|
| **No Gather / sparse lookup** | NNUE needs to look up rows from a weight matrix by index and sum them. Building a one-hot matrix for each position is wasteful. | `SparseEmbedding<T>` and `SparseEmbeddingBag<T>` added to `src/Nivara/AutoDiff/Nn/SparseEmbedding.cs`. Forward uses `TensorPrimitives.Add` for sum pooling; backward uses scatter-add for gradient accumulation. |
| **No Reshape(-1) auto-inference** | Reshape validated all dimensions and didn't support -1 inference. Sparse embeddings need to reshape between flat and batched representations. | `GradTensor.Reshape(-1)` added at `src/Nivara/AutoDiff/GradTensor.cs:76-113`. Infers the missing dimension from total element count. |
| **Variable-length batch inputs** | Chess positions have varying numbers of pieces (0–32), but `TensorDataset<T>` assumes fixed `[B, F]` shapes. | Workaround: fixed-size bag of active features, padded to `MaxActiveFeatures` with sentinel -1 indices, masked out in the forward pass. |
| **No board representation utilities** | No chess-related utilities in Nivara or .NET standard libraries. | Built inline in `ChessBoard.cs`: FEN parsing/generation, 8×8 board representation, material scoring, piece-square tables, halfKP feature encoding. |

### Core library additions from this example

| New API | Location | Purpose |
|---------|----------|---------|
| `SparseEmbedding<T>` | `src/Nivara/AutoDiff/Nn/SparseEmbedding.cs` | Sparse lookup + sum pooling module, wraps `SparseEmbeddingBag<T>` |
| `SparseEmbeddingBag<T>` | `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs` | Sum-pooled sparse embedding: gathers rows by index, sums them. Forward and backward have TensorPrimitives SIMD paths for null-free tensors. |
| `GradTensor.Reshape(-1)` | `src/Nivara/AutoDiff/GradTensor.cs:76-113` | Auto-infer one dimension during reshape |

### Core library performance fixes driven by this example

The Phase 2 training loop (sparse embeddings, 10K positions, ~50s/epoch) exposed autograd infrastructure overhead as the real bottleneck. The following optimizations were applied to the core library:

| Fix | What changed | Impact |
|-----|-------------|--------|
| **`AccumulateGradient` fast path** | Bypasses `NivaraColumn.Add()` overhead (diagnostics recording, kernel selection, element-by-element storage indexer, null mask processing) for the common null-free case. Uses `TensorPrimitives.Add` on raw arrays directly. | Reduces per-gradient accumulation overhead — the hottest path in backward. |
| **`ComputationGraph.Backward` merged traversal** | Three separate DFS traversals (`ValidateGraph` + `BuildNodeToOutputMap` + `TopologicalSort`) merged into single `BuildBackwardPlan<T>()` method. Eliminates ~6 `HashSet`/`List` allocations per backward call. | ~10–20% reduction in Phase 2 epoch time. |
| **LINQ `Reverse()` → `for` loop** | Replaced `Reverse()` (allocates iterator) with reverse-index `for` loop in backward plan execution. | Zero-allocation reverse iteration. |
| **Typed `ZeroGrad` hash set** | `HashSet<ReverseGradTensor<T>>` instead of `HashSet<object>` — typed comparer, no boxing. | Minor, but eliminates type checks on hot path. |
| **`SparseEmbeddingBag` TensorPrimitives SIMD** | Forward and backward no-nulls paths use `TensorPrimitives.Add` replacing scalar `for dim` loops. Null paths retain scalar fallback. | Marginal per-call gain (small spans), but cumulative over batches. |

**Key insight:** `NivaraColumn<T>.AsWritableSpan()` allocates copies on both `TensorStorage` and `MemoryStorage` — in-place gradient accumulation without architectural changes to storage is not feasible. The fast path works by copying to arrays and using `TensorPrimitives.Add` directly, bypassing the full `NivaraColumn.Add()` pipeline.

## Performance

| Phase | Training time | Notes |
|-------|--------------|-------|
| 1 (MLP) | ~12s | 10K positions, 20 epochs. Fast baseline. |
| 2 (NNUE) | ~40–47s/epoch | 10K positions. Autograd infrastructure overhead dominates over arithmetic kernels. |
| 3 (Stockfish) | ~10ms/position | Stockfish `eval` command (static NNUE, no search). Much faster than `go depth`. |

Phase 2 epoch time breakdown (before optimization → after):
- Before: ~50s/epoch (scalar loops, triple DFS traversal, LINQ reverse)
- After: ~40–47s/epoch (merged backward plan, AccumulateGradient fast path, typed ZeroGrad)

The remaining overhead is the autograd graph infrastructure itself (node creation, dependency tracking, backward dispatch), not the arithmetic kernels. For small models (~10K params) with short forward passes, this fixed cost dominates.

## Limitations

- **No move search** — the model evaluates positions but does not play. The UCI mode returns placeholder `bestmove`.
- **Random position generation** — training data uses random material distributions with validity constraints (pawns on ranks 2–7, kings non-adjacent, at least 1 pawn per side). Positions are chess-valid but not from actual game play. Real applications would use opening books or game databases.
- **Phase 3 validation** compares against the PSQT heuristic (not Stockfish) since Stockfish is not available at inference time in this demo.
- **Single-threaded Stockfish** — `EvaluateBatch` processes positions sequentially. Parallel Stockfish instances would speed up phase 3 data generation.
- **Stockfish 18 network replica** — Stockfish 18 creates a child process for shared NNUE memory. The `eval` command avoids the search path that triggers crashes, but `go depth` may still crash on some setups.

## Future work: Expanding the embedding demo

The current `--embed` demo is a minimal proof of concept — 5 hardcoded openings with pairwise cosine similarity. Three directions for making it more useful:

**1. Nearest-neighbor search (most practical)**

Add a `--search` CLI mode that takes a query FEN and finds the K most similar positions from a pre-computed pool:
```
dotnet run -- --load nnue.json --search "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e3 0 1" --top-k 5
```

Implementation: generate N positions (e.g., 100–1000), compute embeddings, store in a flat array. For each query, compute its embedding, then score all stored embeddings via `TensorPrimitives.CosineSimilarity` and return the top K. This is the canonical embedding use case — making `--embed` a real search tool instead of a decorative demo.

**2. Larger position set + similarity matrix**

Expand the hardcoded 5 openings to a programmatically generated set covering different phases (openings, middlegames, endgames). Display a full similarity matrix rather than just pairwise comparisons. This makes the embedding quality visible at a glance — do similar positions cluster together?

**3. Embedding arithmetic / centroid analysis**

Group positions by opening type (Sicilian, King's Gambit, etc.), compute average embeddings per group (centroids), then:
- Find positions closest to a centroid (representative positions for each opening)
- Compute "style" embeddings (e.g., "aggressive" vs "positional" based on piece activity)
- Analogy-style operations (though chess position analogies are less intuitive than word analogies)

**4. Embedding persistence**

Save/load pre-computed embeddings to disk so the search pool doesn't need to be regenerated each time. Would pair well with `ModelSerializer` for full model + embedding persistence.
