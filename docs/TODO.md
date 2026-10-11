# Plan: #577 — seedable AutoDiff initializers so init reproducibility can be gated

- **Branch:** `khurram/577` (off `khurram/574`, which carries the #574 deterministic-training-tests fix)
- **PR target:** to be confirmed at the end (`khurram/574` for a stacked PR; the human's earlier
  instruction said `khurram/571` — with the base now `khurram/574`, asking at conclusion is required)
- **Issue:** https://github.com/khurram-uworx/Nivara/issues/577

## Problem

The AutoDiff initializers (`KaimingUniformInitializer<T>`, `KaimingNormalInitializer<T>`,
`XavierUniformInitializer<T>`, `XavierNormalInitializer<T>`, `NormalInitializer<T>`,
`UniformInitializer<T>`, `PyTorchDefaultInitializer<T>`) all draw from `Random.Shared` at
`Initialize` time and expose no public way to supply a `Random`/seed, so `Linear<T>`'s default
constructor cannot be made reproducible by a caller.

Consequences:

- A caller wanting `--seed`-reproducible init must duplicate the distribution math in its own
  assembly: `samples/NivaraInference` carries a private `SeededKaimingUniform`
  (`Qwen.cs:990-1015`) that mirrors `bound = sqrt(6/fanIn)`, added by #570 precisely because the
  library offered no seedable path.
- The duplicate lives in an exe no test project references, so the reproducibility contract has no
  NUnit gate. The only alternative gate would be a live two-run trace diff needing the unshipped
  Qwen2.5-0.5B checkpoint.
- #574's flake (`TrainingLoop_Run_CompletesEpochs`) was the same root cause (unseeded Kaiming init)
  and was fixed on `khurram/574` with a test-only `DeterministicInit<T>` — that branch is this
  plan's base, so no overlap.

## Proposed changes

### 1. Shared RNG helper — `src/Nivara/Helpers/RandomGeneration.cs`

Add (class is already `internal static class RandomGeneration`, namespace `Nivara.Helpers`):

```csharp
/// <summary>Returns a seeded RNG, or <see cref="Random.Shared"/> when <paramref name="seed"/> is null.</summary>
public static Random Create(int? seed) => seed.HasValue ? new Random(seed.Value) : Random.Shared;
```

Route the existing `GenerateStandardNormal<T>(n, int? seed)` through `Create` and `Sampler<T>`'s
ctor (`src/Nivara/AutoDiff/Nn/Sampler.cs:17-20`) through it too — the exact
`seed.HasValue ? new Random(seed.Value) : Random.Shared` line is currently triplicated (AGENTS
rule #8). Behavior-identical.

### 2. Seven initializers — `src/Nivara/AutoDiff/Nn/Initializers/*.cs`

Each gains `readonly Random random;`, created **once in the constructor** (never inside
`Initialize` — re-creating per call would restart the stream and make later layers prefix-copies of
earlier ones, the exact defect #570 avoided), and `Initialize` uses the field instead of
`Random.Shared`.

Both overloads per type, seed delegating to the `Random` form:

```csharp
// Kaiming/Xavier/PyTorch (currently parameterless + static Instance):
public KaimingUniformInitializer(int? seed = null) : this(RandomGeneration.Create(seed)) { }
public KaimingUniformInitializer(Random random)
{
    this.random = random ?? throw new ArgumentNullException(nameof(random));
}
```

Same shape for `KaimingNormalInitializer`, `XavierUniformInitializer`, `XavierNormalInitializer`,
`PyTorchDefaultInitializer`. `Instance = new()` still compiles (resolves to the seed overload).

```csharp
// NormalInitializer — parameterless ctor is REPLACED by (int? seed = null) — keeping both would
// make `new NormalInitializer<T>()` ambiguous at zero args:
public NormalInitializer(int? seed = null) : this(T.Zero, T.One, seed) { }
public NormalInitializer(Random random) : this(T.Zero, T.One, random) { }
public NormalInitializer(T mean, T std, int? seed = null) : this(mean, std, RandomGeneration.Create(seed)) { }
public NormalInitializer(T mean, T std, Random random) { this.mean = mean; this.std = std; this.random = ...; }
```

`UniformInitializer` mirrors this with `(-T.One, T.One)` / `(T lower, T upper)`.

Known ambiguity: `new X<T>(null)` does not compile (`int?` vs `Random`); callers must use
`seed:`/`random:` named arguments or omit. Document in XML remarks. XML docs state null ⇒
`Random.Shared` (behavior unchanged); blank-line doc comment hygiene per `.editorconfig`.

### 3. Core determinism tests — `tests/Nivara.Tests/AutoDiff/NnTests.cs`

`[TestCaseSource]` over all seven initializers as `Func<int, IInitializer<float>>` factories:

- `Initializer_SameSeed_ProducesIdenticalWeights` — two fresh instances, same seed ⇒ identical arrays.
- `Initializer_DifferentSeed_ProducesDifferentWeights` — seeds 1 vs 2 ⇒ different arrays.
- A small `SnapshotWeights` helper builds a `Parameter<float>` (12 elements),
  `param.Tensor.Reshape(3, 4)` (so the rank<2 no-op path of Kaiming/Xavier/PyTorch is exercised),
  runs `Initialize`, copies values to a `float[]`.
- Two shared-stream pins:
  - one `Random` instance handed to two initializers ⇒ both draw from the same advancing stream
    (second Initialize differs from a fresh replay);
  - one initializer instance reused across two `Initialize` calls ⇒ the second call continues the
    stream (different weights), pinning "created once in ctor, not per Initialize".

`Parameter<float>` does not need disposal here (house precedent in the same file).

### 4. Sample de-duplication — `samples/NivaraInference/Qwen.cs`

- Delete `SeededKaimingUniform` (lines 990-1015) and its explanatory comment.
- `SentimentMLP(IInitializer<float> init)` / `LinearBaseline(IInitializer<float> init)`;
  `RegisterModules` unchanged.
- `RunDistill` (lines 746-747, 769): replace `var initRandom = new Random(seed);` + two ctor calls
  with a single `var init = new KaimingUniformInitializer<float>(seed);` shared across student
  Hidden, student Head, and baseline Head. Same formula and draw order ⇒ `--seed 42` output is
  preserved bit-for-bit (draw order: Hidden weight, Head weight, baseline weight — identical to the
  current sample).

### 5. Docs & changelog

- `CHANGELOG.md` `[Unreleased]` → `### Added`: seed/`Random` constructors on the seven
  initializers, default unchanged (`Random.Shared`), contract "same seed ⇒ identical weights".
- `CHANGELOG.md` → `### Fixed`: `NivaraInference` drops `SeededKaimingUniform` for the library
  initializer. The #570 entry's "cannot be seeded without a new public constructor" becomes false
  after this lands — per the supersession convention, the #570 entry gets a bracketed note pointing
  at the #577 entry rather than being silently rewritten.
- `docs/AUTODIFF.md` §Initializers (line 845): one sentence that each initializer accepts
  `int? seed` (nullable — null ⇒ `Random.Shared`) and `Random`. The example fence at line 863 is
  ungated, so no ADR-005 snippet-gate work.

## Blast radius

- **New public surface:** 7 initializer classes gain constructors (additive; `Instance` singletons
  and existing ctors preserved; `Normal`/`Uniform` parameterless ctors replaced by
  `(int? seed = null)`, source-compatible at `new X<T>()`).
- **Internal:** `RandomGeneration.Create` (new), `GenerateStandardNormal` + `Sampler<T>` ctor
  redirect to it (behavior-identical).
- **Downstream callers:** `Linear<T>` (`Linear.cs:62`), `Embedding`/`SparseEmbedding`/
  `MultiheadAttention`/`TransformerBlock` construct `initializers without seeds — unaffected.
  `Nivara.Tests` uses `Instance` and existing ctors — unchanged. `samples/NivaraInference`
  (only sample referencing random init) switches to the seeded library initializer.
- **Tests:** `Nivara.Tests` does reference the library, so the new determinism tests run in CI
  (this is the whole point of #577). The sample exe is still unreachable from tests — unchanged
  limitation, now documented in the #577 CHANGELOG note.

## Verification

- **Build:** `dotnet build src/Nivara -c Release` and `dotnet build samples/NivaraInference -c Release`
  (expect 0 warnings each).
- **Targeted tests (run autonomously per AGENTS.md):**
  `dotnet test -c Release --filter "FullyQualifiedName~NnTests"`.
- **Full suite:** ask the human first; always `--filter "Category!=Performance"`.
- No timing/allocation assertions → no `[Category("Performance")]` needed anywhere.

## Planned commits

1. `docs: plan #577 seedable initializers in TODO.md`
2. `docs: record #577 G1 grounding in TODO.md` (G1 gate)
3. `feat(autodiff): add seed/Random constructors to the AutoDiff initializers (#577)` — src + CHANGELOG Added
4. `test(autodiff): pin initializer same-seed/different-seed determinism (#577)` — NnTests
5. `refactor(samples): use library seedable Kaiming init in qwen distill (#577)` — Qwen.cs + CHANGELOG Fixed
6. `docs: document seeding on initializers in AUTODIFF.md (#577)` — AUTODIFF
7. `docs: remove TODO.md — plan executed` (G2 gate, only after both reviews clear)

## G1 grounding

_Run immediately after the plan commit; record findings here._

## GitHub issues log

- (none yet)

> As each task executes, any deferred work or concern is filed immediately via
> `gh issue create --repo khurram-uworx/Nivara` and recorded here — not held in memory, because
> compaction during execution can lose it.