# AutoDiff Plan — Move to Core + Parallel CPU Training

## Motivation

Move AutoDiff from `Nivara.Extensions` into the main `Nivara` library and
weaponise the execution engine's data-parallel primitives to prove that modern
multi-core CPUs can handle quick model training without GPUs.

**Core thesis:** More workers with big shovels. An 8–16 core CPU with data-parallel
gradient computation closes the gap with GPU training for small-to-medium models
on columnar data.

---

## Phase 0 — Pre-move cleanup

### What

Remove four files that are defined but never used anywhere in the codebase:

| File | Reason |
|------|--------|
| `IAutoGradNumeric.cs` | Interface defined, zero implementations referenced |
| `AutoGradNumericTypes.cs` | Structs (`Float32`, `Float64`) defined, zero usage |
| `IGradOperation.cs` | Interface defined, `GradOperations` uses `OpNode` lambdas instead |
| `README.md` | Design notes belong in `docs/`, not source tree |

### Non-goal

No functional changes. These are straightforward deletions.

---

## Phase 1 — Move AutoDiff to core library

### File layout

Copy these files from `src/Nivara.Extensions/AutoDiff/` → `src/Nivara/AutoDiff/`:

```
src/Nivara/AutoDiff/
├── GradTensor.cs
├── ReverseGradTensor.cs
├── OpNode.cs
├── ComputationGraph.cs
├── Exceptions/
│   └── AutoGradExceptions.cs
├── Operations/
│   └── GradOperations.cs
├── Optimizer/
│   └── SgdOptimizer.cs
├── Utilities/
│   ├── GradientUtils.cs
│   ├── TypeConverter.cs
│   └── TypeValidator.cs
└── Extensions/
    └── NivaraAutoGradExtensions.cs
```

### Namespace changes

| Current | New |
|---------|-----|
| `Nivara.Extensions.AutoDiff` | `Nivara.AutoDiff` |
| `Nivara.Extensions.AutoDiff.Operations` | `Nivara.AutoDiff.Operations` |
| `Nivara.Extensions.AutoDiff.Optimizer` | `Nivara.AutoDiff.Optimizer` |
| `Nivara.Extensions.AutoDiff.Utilities` | `Nivara.AutoDiff.Utilities` |
| `Nivara.Extensions.AutoDiff.Extensions` | `Nivara` (extension methods merge into Nivara namespace) |

### Project changes

| File | Change |
|------|--------|
| `src/Nivara/Nivara.csproj` | No new NuGet deps needed (already has `System.Numerics.Tensors`) |
| `src/Nivara.Extensions/Nivara.Extensions.csproj` | Remove the moved files; remaining I/O code still references Nivara |
| `src/Nivara.Extensions/AutoDiff/` | Delete entire directory after move |

### Tests

AutoDiff tests live in `tests/Nivara.Extensions.Tests/AutoDiff/`. After the move:
- Copy test files to `tests/Nivara.Tests/AutoDiff/`
- Update namespace references
- Remove originals from Extensions tests
- Keep existing NUnit test structure

### Non-goal

No API surface changes. Only namespace moves and file relocations.

---

## Phase 2 — Data-parallel training infrastructure

### New file: `src/Nivara/AutoDiff/Training/DataParallelTrainer.cs`

```csharp
namespace Nivara.AutoDiff.Training;

public sealed class DataParallelTrainer<T>
    where T : struct, INumber<T>
{
    public DataParallelTrainer(
        NivaraFrame data,
        int maxDegreeOfParallelism);

    public TrainingResult Run(
        Func<Dictionary<string, ReverseGradTensor<T>>, ReverseGradTensor<T>> modelBuilder,
        Dictionary<string, ReverseGradTensor<T>> parameters,
        int epochs,
        T learningRate);
}
```

### How it works

For each epoch:

```
Split data rows into N chunks (N = maxDop × 2 for work-stealing)
  ↓
Parallel.ForEach(chunks):
  ├─ CreateRowSubset(data, chunk.Start, chunk.Length)
  ├─ Build local computation graph (shared params by ref)
  ├─ Forward → loss
  ├─ Backward() → local partial gradients
  └─ Return local gradient columns
  ↓
Sum partial gradients across chunks:
  grad_i = chunkGrads.Sum(g => g[i])   // NivaraColumn<T>.Add
  ↓
SgdOptimizer.SgdUpdate(params_i, lr)
  ↓
ZeroGrad(params)
  ↓
Record epoch diagnostics (timing, loss, gradient norm)
```

### Key design choices

- **Parameters shared by reference**: `ReverseGradTensor<T>` wraps immutable
  `NivaraColumn<T>`. Concurrent reads from all chunks are safe.
- **Local gradient accumulation**: Each chunk produces its own gradient columns.
  No shared mutable state during parallel section. Sum after `Parallel.ForEach`.
- **Reuses `ParallelExecutionHelper`**: Chunk sizing (`CalculateOptimalChunkSize`),
  DOP capping (`GetRecommendedParallelism`), and row subset slicing (`CreateRowSubset`).
- **Diagnostics integration**: Per-epoch timing, per-chunk timing, strategy info.
  Pattern matches `ExecutionDiagnostics.GenerateReport()`.

### New file: `src/Nivara/AutoDiff/Training/TrainingResult.cs`

```csharp
public sealed class TrainingResult
{
    public IReadOnlyList<EpochResult> Epochs { get; }
    public Dictionary<string, ReverseGradTensor<T>> TrainedParameters { get; }
    public void PrintSummary();  // console-friendly report
}

public sealed class EpochResult
{
    public int Epoch { get; }
    public double Loss { get; }
    public TimeSpan Elapsed { get; }
    public int Workers { get; }
    public int Chunks { get; }
    public double GradientNorm { get; }
}
```

---

## Phase 3 — Demonstrate CPU capability

### Update `EXAMPLES.md` Act 5

Current Act 5 shows single AutoDiff snippet. Expand to:

**5a**: Linear regression with AutoDiff (renamespaced from current code)
**5b**: Same model, data-parallel training — show wall-clock speedup across core counts
**5c**: Closing frame — "Modern CPUs with 8–16 cores can train small-to-medium
models in seconds. Nivara's columnar storage + data-parallel AutoDiff turns
every core into a worker with a shovel — no GPU required."

### Performance baseline to target

| Cores | Batch rows | Epochs | Target wall time |
|-------|-----------|--------|-----------------|
| 1 (baseline) | 4096 | 100 | ~2-3s |
| 4 | 4096 | 100 | ~600ms |
| 8 | 4096 | 100 | ~350ms |
| 16 | 4096 | 100 | ~200ms |

Target: near-linear scaling up to 8 cores, ~80% efficiency at 16.

### Key architectural metrics to validate

- Overhead of per-chunk graph construction (should be negligible vs row iteration)
- Gradient summation cost (N columns × Add per parameter)
- Memory usage scaling (N copies of gradient storage during parallel region)

---

## Non-goals (out of scope for this work)

| Feature | Rationale |
|---------|-----------|
| GPU backend | CPU-first. Prove modern CPUs are enough. |
| Adam/RMSProp optimisers | Keep SGD only. Add stateful optimisers when training loops require them. |
| Layer/model abstractions | AutoDiff stays as a low-level gradient layer. |
| Operator overloads (`+`, `-`, `*`) | `GradOperations.Add(...)` stays. Ergonomics deferred. |
| Broadcast / shape inference | Current shape validation is strict. Broadcasting is a separate effort. |
| ExecutionEngine IQueryOperation integration | Training doesn't fit the column-transform model. Integration is via `ParallelExecutionHelper` utilities, not as a custom operation. |

---

## Timeline (order of implementation)

1. Phase 0 — Pre-move cleanup (remove dead files) ~15 min
2. Phase 1 — Move to core + namespace update + test migration ~2-3h
3. Phase 2 — DataParallelTrainer + TrainingResult ~4-5h
4. Phase 3 — EXAMPLES.md update + docs ~1h

Total: ~8-9h engineering time.
