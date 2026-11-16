# Plan: AutoDiff deferred Linear/Embedding init on MiniLM/DistilBERT load path (#1237)

Follow-up to #1226: apply the deferred-weight (`Uninitialized`/`BindWeight`) load path to
`MiniLMDistilled` and `DistilBertForSequenceClassification`, so their Linear/Embedding weights
are not allocated and Kaiming/Normal-initialised only to be replaced by the checkpoint.
Mirrors #1226 commits 3-4.

## Problem

`BertSelfAttention` already supports skip-init; `BertLayer`, `BertEncoder`, `MiniLMDistilled`
do not, and `DistilBertForSequenceClassification` does not. On the load path these still
allocate weight-sized placeholders and run initialisers whose output is immediately overwritten
by the checkpoint, doubling (or more) allocation for the encoder.

## Proposed changes

1. `src/Nivara.Extensions/AI/BertModel.cs`
   - `BertLayer`: public ctor delegates to internal `(..., false)`; internal ctor
     `(hiddenSize, intermediateSize, numHeads, eps, bool skipWeightInitialization)` threads
     flag into `BertSelfAttention` and uses `Linear.Uninitialized` for `fc1`/`fc2` when skip.
   - `BertEncoder`: public ctor delegates to internal `(config, includeTokenTypeEmbedding, false)`;
     internal ctor defers `wordEmbed`/`posEmbed`/`tokenTypeEmbed` via `Embedding.Uninitialized`
     and threads flag into `BertLayer`.
   - `MiniLMDistilled`: public ctor delegates to internal `(config, false)`; internal ctor
     `(config, bool skipWeightInitialization)` passes skip to `BertEncoder(config, true, skip)`.
   - `MiniLMDistilled.LoadWeights`: construct `new MiniLMDistilled<TModel,TWeight>(config, true)`.
   - Sealed aliases `BertSelfAttention<T>`, `BertLayer<T>`, `BertEncoder<T>`,
     `MiniLMDistilled<T>` gain internal skip ctors.
2. `samples/Nivara.Samples/DistilBertModel.cs`
   - `DistilBertForSequenceClassification<T>`: public ctor delegates to internal
     `(config, numClasses, false)`; internal ctor `(config, numClasses, bool skipWeightInitialization)`
     defers encoder (`BertEncoder<T>(config, includeTokenTypeEmbedding:false, skip)`) and
     `preClassifier`/`classifier` via `Linear<T>.Uninitialized`.
   - `DistilBertLoader.LoadEncoder`: construct encoder with skip.
3. Load-path construction sites use the skip ctor:
   - `samples/NivaraInference/DistilBertSst.cs` `Load<T>`
   - `samples/NivaraInference/Program.cs` (3 sites)
4. `dotnet format Nivara.slnx whitespace`.

## Verification

- `dotnet build Nivara.slnx` (Release)
- Targeted tests: DistilBert/MiniLM/Bert suites
- Docs snippet gate `EveryGatedBlock_CompilesWithoutErrors`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits

1. docs: plan #1237 in TODO.md
2. feat(Extensions): defer Linear/Embedding init on BertLayer/BertEncoder/MiniLMDistilled load path
3. feat(Samples): defer Linear/Embedding init on DistilBert load path
4. feat(Inference): use deferred ctors on DistilBert load sites
5. style: whitespace formatting
6. docs: remove TODO.md — plan executed

## Blast radius

- `BertModel.cs`: `BertLayer`, `BertEncoder`, `MiniLMDistilled` constructors (new internal overloads only; public signatures unchanged).
- `DistilBertModel.cs`: `DistilBertForSequenceClassification` constructor (new internal overload); `DistilBertLoader.LoadEncoder`.
- Inference load sites.
- Downstream callers unaffected (public ctors preserved).
- Tests: existing DistilBert/MiniLM tests construct with public ctors (init path) unchanged.

## GitHub issues log

(none yet)
