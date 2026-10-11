# Plan — #570: `--seed` is accepted but unused, so a `qwen distill` run is not reproducible

## Problem

`NivaraInference`'s `qwen distill` mode accepts `--seed N`, stores it, and prints it:

```csharp
Console.WriteLine($"Seed (accepted for future use; Kaiming init is unseeded): {seed}");
```

but nothing consumes it. `samples/NivaraInference/Qwen.cs` has no RNG use in the distill
path, so the student's Kaiming initialisation is driven by `Random.Shared` and every run
differs. An operator passing `--seed 7` expects a reproducible run and silently does not get
one — the same class of defect as #479 (a value the operator asked for is quietly not the
input the run used), which is now closed and only made this gap more visible.

Root cause, verified:

- `Program.cs:42` — `int seed = 42;`
- `Program.cs:87,104` — `--seed`/`--teacher-examples` share `TryParseCount`; `seed` assigned.
- `Program.cs:444` — `Qwen.RunDistill(tensors, modelDir, teacherExamples, force, seed)`.
- `Qwen.cs:707-712` — `RunDistill(..., int seed)`; the parameter is read only at `Qwen.cs:844`
  in the status print.
- `Qwen.cs:745` `new SentimentMLP()` and `:767` `new LinearBaseline()` -> `Linear<float>`
  default ctor (`src/Nivara/AutoDiff/Nn/Linear.cs:62`) -> `KaimingUniformInitializer<float>.Instance`
  (`src/Nivara/AutoDiff/Nn/Initializers/KaimingUniformInitializer.cs:25`) -> `Random.Shared`.

The teacher pass is greedy and the label cache (`qwen_distill_labels.json`) is deterministic,
so the student/baseline init is the only non-deterministic input. `Adam` and `CrossEntropyLoss`
add no randomness, and `SentimentMLP` has no dropout.

## Decision (cleared with the human before execution)

- **Seed the init** (the issue's preferred direction), not drop the flag.
- **Sample-only** implementation: no `src/Nivara` change. The sample carries its own seeded
  Kaiming-uniform initializer because the library's `KaimingUniformInitializer<T>` draws from
  `Random.Shared` and cannot be seeded without a new public constructor (that was the
  alternative, A1, and was declined in favour of A2).
- **Always seeded, default 42**: every distill run is reproducible; `--seed` selects a
  different stream. Help text states the default.

## Proposed changes

### 1. `samples/NivaraInference/Qwen.cs`

Add `using Nivara.AutoDiff.Nn.Initializers;`. Add a private seeded initializer that mirrors
the library's Kaiming-uniform distribution (`bound = sqrt(6 / fanIn)`, uniform in
`[-bound, bound]`) but draws from a caller-supplied `Random`:

```csharp
/// <summary>Seeded Kaiming-uniform init: the same distribution as the library's
/// KaimingUniformInitializer&lt;float&gt; (uniform in [-bound, bound], bound = sqrt(6/fanIn))
/// but drawn from the run's seed so --seed fixes the student weights. Kept in the sample
/// because the library initializer draws from Random.Shared and cannot be seeded.</summary>
sealed class SeededKaimingUniform : IInitializer<float>
{
    readonly Random random;
    public SeededKaimingUniform(Random random) => this.random = random;

    public void Initialize(Parameter<float> parameter)
    {
        var tensor = parameter.Tensor;
        if (tensor.Shape.Length < 2) return;
        var bound = MathF.Sqrt(6f / tensor.Shape[1]);
        var data = new float[tensor.Length];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(random.NextDouble() * 2.0 - 1.0) * bound;
        parameter.Tensor = new ReverseGradTensor<float>(
            NivaraColumn<float>.CreateFromOwnedArray(data), tensor.RequiresGrad, tensor.Shape);
    }
}
```

- `SentimentMLP(Random random)` — one `SeededKaimingUniform(random)` shared by both `Linear`
  layers, passed as `weightInitializer`, so no double-init and no reliance on the default.
- `LinearBaseline(Random random)` — same, one `Linear(FeatureDim, 2)`.
- `RunDistill`: create one `var initRandom = new Random(seed);` and pass it to both models, so
  the whole run is a single deterministic stream from `seed` (avoids the baseline weights being
  a prefix-copy of the student's first layer that two separate `Random(seed)` instances produce).
- Replace `Qwen.cs:844` with the new seed status line naming what it now drives.

### 2. `samples/NivaraInference/Program.cs:175` (help)

```
--seed N           Distill: init seed, default 42 (same seed reproduces the run)
```

The README distill line does not mention `--seed`, so no README change.

### 3. `CHANGELOG.md` — `### Fixed`

Entry naming #570, in the style of the adjacent #568/#479 entries.

## G1 grounding (recorded before implementation)

- **Microsoft Learn** — `System.Random(Int32)`: "the same sequence is generated when the
  Random object is created again with the constructor and seed parameter" (reproducible within a
  runtime; only cross-major-version sequence stability is not guaranteed), and "If a negative
  number is specified, the absolute value of the number is used". Both support a seeded
  `Random` as the determinism source.
- **code-memory** — relationship graph confirms `SentimentMLP`, `LinearBaseline`, and
  `RunDistill` have no referrers outside `samples/NivaraInference/Qwen.cs` (only the `Head`/
  `Hidden` fields and the two constructors reference `Linear`); `RunDistill` is called from
  `Program.cs:444` only. No other sample or `src/Nivara` symbol is affected.
- **Red flag (escalated, accepted)** — with A2 the fix has no automated NUnit gate
  (`RunDistill` needs the absent teacher checkpoint; the student types are private to the
  sample exe). Recorded under "Known coverage limitation" above. The human chose A2 with this
  known.

## Blast radius

- `samples/NivaraInference/Qwen.cs` — new private nested type; two nested class constructors
  gain a parameter; `RunDistill` body. All callers are inside `Qwen.cs` / `Program.cs`.
- `samples/NivaraInference/Program.cs` — one help string.
- `CHANGELOG.md` — one entry.
- No `src/Nivara` symbol is touched. No test project references `NivaraInference`, so the
  existing `QwenDistillStudentParityTests` (which constructs `Linear<float>` with fixed fixture
  weights, bypassing init) is unaffected.

## Verification

- `dotnet build samples/NivaraInference -c Release` — must succeed.
- End-to-end reproducibility gate (needs the ~989 MB Qwen checkpoint, **absent from this
  checkout**): `dotnet run -c Release --project samples/NivaraInference -- qwen distill
  --teacher-examples 12 --seed 7` twice and diff the printed epoch-loss/accuracy trace; then
  `--seed 8` and confirm it differs.

### Known coverage limitation (honest statement)

With the A2 (sample-only) choice the fix has **no automated gate**: `RunDistill` needs the
teacher checkpoint, and `SentimentMLP`/`LinearBaseline` are private to the sample exe, which no
test project references. The only strong gate is the live two-run diff, which is not runnable
here. This was surfaced to the human during planning and A2 was chosen anyway; the changelog
will record the coverage limitation rather than imply the run was reproduced.

## Planned commits

1. `fix(samples): seed the qwen distill student/baseline init from --seed (#570)` —
   `Qwen.cs` + `Program.cs` help + `CHANGELOG.md` (mirrors the #568 commit shape).
2. `docs: plan #570 seed the qwen distill student init in TODO.md` (this file) — committed first,
   before implementation.
3. `docs: remove TODO.md — plan executed` — only after both G2 reviews clear.

## GitHub issues log

- [ ] (none yet — capture any deferred work discovered during execution here, immediately)
