# NivaraChess — Neural Chess Evaluation with Nivara AutoDiff

## Goal

Train a neural network to evaluate chess positions. The model learns a
board representation → position score (good/bad for the side to move).
The trained network is then used as a position evaluator, wrapped in a
UCI-compatible interface or interactive console opponent.

This showcases Nivara outside NLP — targeting the game dev / Unity
community, which is one of .NET's largest ecosystems outside enterprise.

```
Synthetic chess position generator (NNUE-style encoding)
    │
    v
NivaraFrame + TensorDataset<float> + DataLoader<float>
    │
    v
ChessEvalModel<float> : Module<float>
    ├── Feature encoder (NNUE-inspired halfKP encoding)
    ├── 2-3 hidden layers (Linear → ReLU)
    └── Single scalar output (position evaluation)
    │
    v
MSELoss (training against Stockfish evaluations)
    │
    v
TrainingLoop<float> + AdamW<float>
    │
    v
ModelSerializer.Save / Load
    │
    v
Interactive chess evaluator (FEN → score)
    OR
    IEmbeddingGenerator<BoardState, Embedding<float>>  (board → embedding vector)
```

## Why Chess?

| Angle | Why it works |
|---|---|
| **Game dev / Unity** | Chess evaluation is a concrete, non-trivial use case. Unity/C# is a huge .NET market. |
| **No external data needed** | Generate synthetic positions + evaluate with a simple heuristic (or Stockfish if desired). The minimal version uses material counting + piece-square tables as the target. |
| **Different architecture than NLP** | Exercises Nivara in a completely different domain: sparse inputs, board representation, feature engineering |
| **.NET ecosystem integration** | Can implement `IEmbeddingGenerator<BoardState, Embedding<float>>` — Microsoft's standard AI interface for embedding generation |
| **Scalable complexity** | Start with simple material-count regression, scale up to NNUE or ResNet-like architecture |
| **Visually appealing** | Console-based chess board + evaluation is inherently satisfying to demo |

## Architecture

### Phase 1: Simple Material Evaluator (MVP)

Target: Learn a material counting function. Input is a feature vector
of piece counts (12 dimensions: 6 piece types × 2 colors). Output is a
scalar evaluation in centipawns.

```
Input: [B, 12]  (piece counts: pawns, knights, bishops, rooks, queens, king × white/black)
    │
    v
Linear(12 → 32) → ReLU
    │
    v
Linear(32 → 16) → ReLU
    │
    v
Linear(16 → 1)  (scalar evaluation)
    │
    v
Output: [B, 1]  (positive = good for white, negative = good for black)
```

**Training data:** Generate random legal positions, compute evaluation
as `material_score = sum(piece_values) - sum(opponent_piece_values)`.
Piece values: P=100, N=320, B=330, R=500, Q=900, K=20000.

This is deliberately simple — it validates the pipeline end-to-end.
The model should learn weights approximating standard piece values.

### Phase 2: NNUE-Style Feature Encoder (Primary Target)

NNUE (Efficiently Updatable Neural Network) uses a sparse input
representation called "halfKP": for each of the king's squares (64),
encode the positions of all friendly and enemy pieces relative to it.

**Simplified feature encoding:**

```
Input board → 64 (king squares) × 64 (piece squares) × 12 (piece types)
           → ~49K features, extremely sparse (max ~32 non-zero per position)
```

For the example, we use a simpler but still effective encoding:

**HalfKP-style encoding:**
```
For each piece (max 32 pieces on board):
   featureIndex = pieceType * 64 * 64 + kingSquare * 64 + pieceSquare
   Set input[featureIndex] = 1
```

This creates a sparse binary vector of dimension ~12×64×64 = 49,152
(max 32 active entries).

**Network architecture:**
```
Sparse input [B, 49K]  (one-hot via embedding-like lookup; not dense MatMul)
    │
    v
Linear(256 → 32) → ReLU       (first layer is the "feature transformer")
    │
    v
Linear(32 → 32) → ReLU        (hidden layer)
    │
    v
Linear(32 → 1)                (evaluation head)
    │
    v
Scalar output [B, 1]
```

**Key challenge:** The input is 49K dimensions but very sparse. A dense
Linear(49K × 256) would be ~12.5M parameters. NNUE solves this by using
a **sparse lookup** instead of MatMul — each active input feature looks
up a row in a weight matrix, and the result is summed (effectively an
embedding with sum pooling).

**Gap this exposes:** Nivara doesn't have a sparse lookup / Gather op.
Building this efficiently reveals the gap and drives a fix.

### Phase 3 (Stretch): Implement `IEmbeddingGenerator<BoardState, Embedding<float>>`

Wrap the trained model so it implements Microsoft.Extensions.AI's standard
embedding interface. A chess position can be "embedded" into a vector
that represents its strategic features. This enables:
- Semantic similarity between positions (find similar positions)
- Clustering openings by embedding
- Dimensionality reduction for visualization

```csharp
public sealed class ChessEmbeddingGenerator : IEmbeddingGenerator<BoardState, Embedding<float>>
{
    private readonly ChessEvalModel<float> _model;

    public async Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<BoardState> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var results = new List<Embedding<float>>();
        foreach (var board in values)
        {
            var features = EncodeBoard(board);                 // → int[] activeFeatures
            var embedding = _model.ComputeEmbedding(features); // activation before output head
            results.Add(new Embedding<float>(embedding));
        }
        return new GeneratedEmbeddings<Embedding<float>>(results);
    }

    public object? GetService(Type serviceType, object? serviceKey) => ...;
    public void Dispose() => _model.Dispose();
}
```

## Gaps This Will Discover and Fix

### Gap 1: No Gather / sparse lookup operation

**Problem:** NNUE-style networks need to look up rows from a weight
matrix by index and sum them. The current `Embedding<T>.Forward(int)`
does this for one token via one-hot + MatMul, but for 32 sparse features
per position, building a 32 × 49K one-hot matrix each time is wasteful.

**Fix:** Add `ReverseGradOperations.Gather<T>(weight, indices, dim)`:
```csharp
// weight: [numRows, embedDim]
// indices: [numIndices]
// Returns: [numIndices, embedDim] — the gathered rows
public static ReverseGradTensor<T> Gather<T>(
    ReverseGradTensor<T> weight,
    int[] indices);
```

Or more concretely, a `SparseEmbedding<T>` module:
```csharp
public sealed class SparseEmbedding<T> : Module<T> where T : struct, INumber<T>
{
    public SparseEmbedding(int numEmbeddings, int embeddingDim);

    // indices: [batchSize, numActiveFeatures] — variable-length per row
    // Returns: [batchSize, embeddingDim] — summed embeddings
    public ReverseGradTensor<T> Forward(ReverseGradTensor<T> indices);
}
```

The backward pass for Gather is a ScatterAdd: each row of the gradient
is added back to the weight matrix at the gathered indices.

### Gap 2: Variable-length batch inputs

**Problem:** Chess positions have varying numbers of pieces (0–32).
`TensorDataset<T>` and `DataLoader<T>` assume fixed `[B, F]` shapes.

**Workaround:** Use a fixed-size bag of active features (e.g., pad to
32 with sentinel -1 indices, mask them out in forward). The `-1` index
can be treated as a zero row.

### Gap 3: No Reshape with -1 inference

**Problem:** `Reshape` validates all dimensions and doesn't support -1
auto-inference. For a sparse embedding layer, we may need to reshape
between flat and batched representations.

**Fix:** Add `-1` dimension inference to `Reshape`. This is a small
change to `GradTensor.Reshape`.

### Gap 4: No board representation utilities

**Problem:** There are no chess-related utilities in Nivara or .NET
standard libraries. We need:
- FEN parsing / generation
- Board state representation (8×8 with piece types)
- Move generation (for generating training positions)
- Legal position validation

**Approach:** Build minimal chess utilities inline in the example.
A full chess library is outside scope — we provide what's needed for
data generation. If the example proves popular, these could be extracted
to a separate package.

### Gap 5: NNUE feature transformer is a new architecture pattern

The NNUE's "sparse lookup → sum → dense layers" pattern is different
from standard MLP or transformer architectures. It exercises:
- Embedding-like forward without MatMul
- Sum pooling of gathered vectors
- Sparse gradient propagation (only touched rows receive gradients)

This is a good stress test for Nivara's autograd engine.

### Gap 6: Training data generation without Stockfish

Stockfish evaluation is the gold standard for chess position evaluation,
but it requires an external binary. The example should work without it:

| Complexity | Training data source | Quality |
|---|---|---|
| **Phase 1** | Material counting (hand-crafted formula) | Simple but correct signal |
| **Phase 2** | Material + piece-square tables | Reasonable positional understanding |
| **Phase 3** | Stockfish evaluations (optional, if installed) | Near-arbitrary accuracy |
| **Phase 4** | Self-play with MCTS (AlphaZero-style) | Full reinforcement learning |

Each phase is a valid example on its own. The example ships with Phase 1
(pure material) working out of the box; users can opt into higher phases.

## Files

```
samples/NivaraChess/
├── Program.cs                    # Entry point, CLI, all modes
├── ChessBoard.cs                 # Minimal board representation, FEN I/O
├── ChessDataGenerator.cs         # Random position generator + target evaluator
├── ChessEvalModel.cs             # Neural network model (Module<T>)
├── SparseEmbedding.cs            # Sparse lookup module (NEW gap fix)
├── ChessEvalConsole.cs           # Interactive chess evaluation REPL
└── NivaraChess.csproj            # Project referencing Nivara
```

### NivaraChess.csproj

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

## CLI Interface

```
--generate <int>      Generate N random positions with evaluations
--epochs <int>        Training epochs (default: 20)
--batch-size <int>    Batch size (default: 128)
--hidden-dim <int>    Hidden layer size (default: 64)
--lr <float>          Learning rate (default: 0.001)
--save <path>         Save trained model
--load <path>         Load trained model
--fen <fen-string>    Evaluate a single FEN position
--interactive         Interactive FEN evaluation REPL
--uci                 UCI protocol mode (for chess GUI integration)
--phase <int>         Data generation phase: 1=material, 2=PSQT, 3=Stockfish (default: 1)
--num-positions <int> Number of positions to generate (default: 10000)
--stockfish <path>    Path to Stockfish executable (phase 3)
--seed <int>          RNG seed (default: 42)
--help, -h            Show this help
```

### Modes

**Default:** Generate positions (Phase 1), train, show accuracy on
held-out test set, save to disk.

**`--fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1":**
Evaluate a starting position. Print the score in centipawns.

**`--interactive`:** REPL: type FEN strings, get evaluations. Also
optionally display an ASCII board. Type `quit` to exit.

**`--uci`:** UCI protocol. A chess GUI (Arena, CuteChess, etc.) can
connect to this as an engine. The engine evaluates positions using the
trained network. This is the "enterprise" integration mode.

### UCI Mode

UCI (Universal Chess Interface) is the standard protocol for chess
engines. The NivaraChess UCI mode implements a minimal subset:
```
UCI
id name NivaraChess
id author Nivara
uciok

isready
readyok

position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
go movetime 100
bestmove 0000  (no move — evaluation only engine)
```

This lets chess GUIs use NivaraChess as a position analysis tool.
(Stockfish-level search is not the goal — this is a static evaluator.)

## What This Exercises vs. MicroGpt

| Feature | MicroGpt | NivaraChess |
|---|---|---|
| **Domain** | Text / NLP | Board games / game AI |
| **Input representation** | Dense token IDs | Sparse board features |
| **Architecture** | Transformer | MLP / NNUE feature transformer |
| **Gather / sparse ops** | Not needed | **Essential — reveals gap** |
| **Training data** | Text corpus (names) | Synthetic positions + heuristics |
| **Data generation** | HTTP download | **Included generator** |
| **TrainingLoop<T>** | No (manual loop) | Yes |
| **Loss type** | NLL (classification) | MSE (regression) |
| **Output type** | Vocabulary distribution | Single scalar (evaluation) |
| **Model size** | ~3.5K params | ~10K–500K params depending on phase |
| **Inference API** | Generate text | FEN → score, UCI protocol |
| **IEmbeddingGenerator** | No | Optional (board → embedding vector) |
| **Interactive mode** | Generate samples | FEN REPL, UCI GUI |
| **Streaming** | Console.WriteLine | UCI protocol over stdin/stdout |
| **Core library changes** | None | Gather op, SparseEmbedding Module, Reshape -1 |

## Core Library Changes

After validation from this example, these would be promoted to `src/Nivara/`:

| New API | File | Description |
|---|---|---|
| `ReverseGradOperations.Gather<T>` | `src/Nivara/AutoDiff/Operations/` | Index-based row selection from matrices |
| `SparseEmbedding<T>` module | `src/Nivara/AutoDiff/Nn/` | Sparse lookup + sum pooling |
| `GradTensor.Reshape(-1)` support | `src/Nivara/AutoDiff/GradTensor.cs` | Auto-infer dimension in reshape |
| (Potentially) `ScatterAdd` op | `src/Nivara/AutoDiff/Operations/` | Gradient for Gather (scatter grad to indices) |

## Comparison: NivaraChatClient vs NivaraChess

| Aspect | NivaraChatClient | NivaraChess |
|---|---|---|
| **Target audience** | Enterprise .NET devs, AI/ML | Game devs, Unity, chess programmers |
| **Ecosystem integration** | IChatClient (Microsoft standard) | UCI protocol (chess standard), optional IEmbeddingGenerator |
| **Primary gap revealed** | LayerNorm, Concat, batched embedding | Gather, sparse ops, Reshape -1 |
| **Data dependency** | External text corpus (TinyShakespeare) | Self-contained synthetic data generator |
| **Library scope of changes** | ~5 new ops/modules | ~2 new ops + 1 module |
| **"Wow" factor** | Chat with a .NET-native AI | Evaluate chess positions interactively |
| **DI showcase** | services.AddChatClient() | Not directly (UCI is its own protocol) |
| **Difficulty** | Higher (transformer is complex) | Lower (MLP is simpler, but sparse ops are new) |
| **Relation to MicroGpt** | Direct lineage (both are transformers) | Different domain entirely |
