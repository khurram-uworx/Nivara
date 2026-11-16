# Plan: Linear/Embedding — skip weight initialisation on the load path (#1226)

**Issue:** #1226 — Linear/Embedding: skip weight initialisation on the load path (load-time transient)
**Milestone:** 1.5.2
**Branch:** khurram/1226 (off main @ 6b7fa0f4)

## Problem

`ModernBertEncoder<TModel,TWeight>.LoadWeights` and `LayaDecisionHead<TModel,TWeight>.LoadWeights`
construct `Linear`/`Embedding` modules whose constructors allocate a weight array and
Kaiming/Normal-initialise it, which `LoadWeights` then replaces with the checkpoint's own
array. That is short-lived garbage, measured today as ~839.8 MB of *weight payload* for Laya
Half (686.3 MB projections + 50.3 MB head + 103.2 MB embedding) and ~1.68 GB on the F32
path. The residency gates force a full GC so it doesn't fail them, but load time and peak
memory still pay it.

Documented in `docs/plan/LAYA-HONEST-WEIGHTS.md` §4.3 as a non-goal of the native-loading
PRs because it touches every model's constructor. This issue closes that gap.

## Key findings (verified against source, not the issue body)

1. **The transient is ~2× the quoted figure.** `KaimingUniformInitializer.Initialize` and
   `NormalInitializer.Initialize` allocate a *second* array and then replace `parameter.Tensor`
   (`KaimingUniformInitializer.cs:49-63`, `NormalInitializer.cs:74-86`). Each ctor therefore
   allocates its weight twice, with both arrays live during the swap. The issue's 839.8 MB is
   the single-copy payload; the cumulative allocation is ≈ 1.68 GB (Half) / ≈ 3.4 GB (F32).
   The gate will measure the real number rather than restate the doc figure.
2. **A silent-gradient trap in the loaders.** The five mixed-target bind sites derive the new
   tensor's gradient flag from the existing parameter:
   ```csharp
   var requiresGrad = linear.Weight != null && linear.Weight.Tensor.RequiresGrad;   // → false once Weight is deferred
   ```
   (`StateDictLoader.cs:210, 226, 267, 301, 209`). A deferred `Weight` would bind with
   `RequiresGrad = false`, so the model still runs but stops accumulating weight gradients.
   Mitigation: `?? true` at every site.
3. **The acceptance gate cannot run in CI.** Any test calling `GC.GetAllocatedBytesForCurrentThread`
   must carry `[Category("Performance")]` (enforced by `SlowTestCategorisationTests`), which
   CI filters out. So the allocation gate is local-only evidence; CI's real guardian is a
   *structural* test — "a deferred `Linear.Weight` is null and never allocates a weight-sized
   array" — which needs no measurement. Ship both.

## Proposed design — defer the `Parameter`, do not use a placeholder

**Rejected alternative:** keep the `Parameter` but back it with an empty `TWeight[]` plus the
true `[out, in]` shape. `GradTensor` does not validate shape against length, so it works
mechanically, and `StateDictLoader`'s five sites need no change. But it makes `Weight`
**never null**, and the repo already uses null-ness as the "not loaded" signal
(`ModernBertModel.cs:714` `tokenEmbedding.Weight is null`, `ModelSpan`/`ParamSpan` at
`628-642`, the ONNX exporter's `Weight?.Tensor.Shape ?? …`). Every one of those would start
silently reading empty spans. That sweep is far more fragile than the chosen approach's cost.

**Chosen:** when deferring, there is no `Parameter` at all.

- `Linear`/`Embedding`: the `weight` field is no longer `readonly` and becomes nullable.
  The public `Weight` property is *already* `Parameter<TWeight>?`, so no public signature
  changes and every existing consumer compiles unchanged.
- New `BindWeight(ReverseGradTensor<TWeight> weight)` (public factory's companion):
  - validates `shape == [outFeatures, inFeatures]` / `[numEmbeddings, embeddingDim]`;
  - creates `Parameter<TWeight>("Weight", …)`, registers it (`RegisterParameterObject` when
    `TWeight == TModel`, else `RegisterMixedParameters`), assigns the field;
  - if already bound, replaces `.Tensor` (one method serves both deferred and existing paths,
    so a reload still works).
- `LoadStateDict` materialises the parameter when unbound: same-dtype path clones (preserving
  today's copy semantics); mixed path keeps the existing `TypeConverter` conversion. This covers
  `LoadLinearWidened`/`LoadLinear<TModel>` (Qwen F32, MiniLM) without touching them.
- `Forward`, `Embedding.ForwardCore`/`GatherRows`, `StateDict`, `ExportToOnnx` get a guard
  throwing `InvalidOperationException` naming "weight has not been loaded" — no NREs, no
  silent empty spans.
- Bias is **not** deferred: it's 1-D (`outFeatures` elements), already zero unless a
  `biasInitializer` is supplied, and negligible. Only the 2-D weight matters.

**Known, accepted wart:** on the deferred path the bias is registered before the weight, so
`GetParameters()`/`Parameters()` enumerate `[Bias, Weight]` instead of `[Weight, Bias]`.
Nothing in the repo depends on that order (state dicts are dictionaries, optimizers are
order-independent). Stated in the changelog/PR rather than hidden.

## Public surface: `Uninitialized` factory (decided at G1 — see §Decisions)

- `Linear<TModel,TWeight>.Uninitialized(int inFeatures, int outFeatures, bool bias = true)`
- `Embedding<TModel,TWeight>.Uninitialized(int numEmbeddings, int embeddingDim)`
- same-dtype aliases `Linear<T>.Uninitialized` / `Embedding<T>.Uninitialized`.
- Internal skip ctors for the model classes thread the flag down.

Rationale: 1.5.2 is a patch release; adding an optional parameter to `Linear`'s existing
ctor is a binary break for compiled consumers even though it's source-compatible. A factory
adds a capability with no signature change. A factory is only useful with a public binder, so
`BindWeight` is public too.

## Scope — files to touch

Core (`src/Nivara`):
- `AutoDiff/Nn/Linear.cs` — field, ctor guard, `Uninitialized` factory, `BindWeight`,
  `LoadStateDict`, `Forward`, `StateDict`, `ExportToOnnx`.
- `AutoDiff/Nn/Embedding.cs` — same shape.
- `AutoDiff/Nn/LlamaCausalAttention.cs`, `LlamaDecoderBlock.cs`, `LlamaAliases.cs` —
  internal skip ctor.

Extensions (`src/Nivara.Extensions`):
- `AI/ModernBertModel.cs` — `ModernBertAttention`, `ModernBertMlp`, `ModernBertLayer`,
  `ModernBertEncoder` internal skip ctor; `LoadWeights` uses it.
- `AI/BertModel.cs` — `BertSelfAttention` internal skip ctor.
- `AI/LayaHeadModel.cs` — `LayaHeadLayer`, `LayaDecisionHead` internal skip ctor;
  `LoadWeights` uses it.
- `AI/ModernBertAliases.cs`, `AI/LayaAliases.cs` — pass-through.
- `AI/StateDictLoader.cs` — five bind sites → `BindWeight`, `?? true`.

Samples (`samples/Nivara.Samples`):
- `LlamaForCausalLM.cs` — `LlamaForCausalLM` internal skip ctor; `LlamaLoader.Load` uses it.

Tests:
- New `tests/Nivara.Tests/AutoDiff/LinearLoadPathTests.cs`,
  `EmbeddingLoadPathTests.cs`.
- Extend `ModernBertWeightLoadingTests`, `LayaDecisionHeadTests`,
  `SynthesizeTensorsBiasGatingTests` with reference-identity checks.
- Extend `LayaNativeCheckpointTests` with the allocation gate (Performance-categorised).

Docs (optional, pin-neutral):
- `CHANGELOG.md` 1.5.2 entry; one prose paragraph in `docs/AUTODIFF.md`.

## Planned commits

1. Gate first, red: allocation measurement + structural null-until-bound tests. Run in
   Release on the base tree, record the real number in the commit body.
2. `Linear`/`Embedding` deferral + `BindWeight` + guards + unit tests. Call sites unchanged;
   everything still initialises by default.
3. Extensions loaders: internal skip ctors, `StateDictLoader` bind sites, `?? true`.
4. Samples Llama: internal skip ctor through the chain.
5. *(optional)* MiniLM/DistilBERT loaders.
6. *(optional)* `CHANGELOG.md` + `docs/AUTODIFF.md` prose.

## Blast radius

- `Linear`/`Embedding` are the public AutoDiff modules (`src/Nivara/AutoDiff/Nn/`); the
  `Weight` property type is unchanged (already nullable), so all consumers compile.
- `Module.LoadStateDict` is unchanged; `Linear`/`Embedding` overrides gain a materialise-if-unbound
  branch, so public `LoadStateDict` users get a weight adopted from the supplied tensor.
- `StateDictLoader` is `Nivara.Extensions`-internal-ish but public; the five sites switch from
  `Weight!.Tensor = …` to `BindWeight(…)`. `Nivara.Extensions` has IVT to Nivara ✅.
- `LlamaForCausalLM` (Samples) uses the internal skip ctor; `Nivara.Samples` has IVT only to
  `Nivara.Tests`/`Nivara.PerformanceTests`, not to `Nivara` — hence the public `Uninitialized`
  factory for `Linear`/`Embedding` and the *internal* skip ctor for the model classes, which
  are in the same assembly as their loaders. No new IVT needed.
- Tests that assert `Weight is null` after construction would break only if they constructed
  with `Uninitialized`; none do today.

## Verification

- `LinearLoadPathTests` / `EmbeddingLoadPathTests`: null weight + absent parameter before
  binding; `Forward`/`StateDict`/`GatherRows` throw "not loaded"; `BindWeight` is zero-copy
  via `Unsafe.AreSame` on the span's reference; `RequiresGrad` stays true when binding an
  unbound module; shape-mismatch throws; `LoadStateDict` materialises for both same-dtype and
  `float`/`Half`.
- Extend `ModernBertWeightLoadingTests`, `LayaDecisionHeadTests`,
  `SynthesizeTensorsBiasGatingTests` with reference identity on a tiny synthetic dict.
- Local, Release: the Laya allocation gate (`< 16 MB`, from a measured ~1.68 GB before) plus
  `LayaNativeCheckpointTests` in full.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`; targeted
  `dotnet test -c Release --filter …`; Docs namespace filter only if docs change.
- Doc pins: a prose paragraph and a CHANGELOG entry move no pins (no `path::Symbol` citation,
  no new doc file).

## GitHub issues log

- [ ] #1237 — AutoDiff: defer Linear/Embedding init on MiniLM/DistilBERT load path (#1226 follow-up) (created while working on #1226)

## Decisions (Gate G1 — awaiting human clearance)

1. **Naming.** `Uninitialized(...)` factory vs `skipWeightInitialization` bool vs issue's
   `skipInitialize`. Recommendation: `Uninitialized` factory.
2. **Public surface.** Factory + public `BindWeight` (no binary break) — recommended. Alt:
   trailing optional `bool` on the four public ctors; or internal-only + new IVT for Samples.
3. **Scope.** `Linear`+`Embedding` only — recommended. Alt: also `LayerNorm`/`RMSNorm`; or
   also MiniLM/DistilBERT now.
4. **Initialiser + skip together.** Throw `ArgumentException`, or ignore? Moot under the
   factory (no initialiser parameters).
5. **Docs/CHANGELOG.** 1.5.2 entry + one prose paragraph now, or leave out of this PR?
6. **Red measurement.** Run the Laya checkpoint gate locally in Release (~1.7 GB peak, fixture
   present) as commit 1, or quantify the "before" another way?

## Grounding (to be performed at G1)

- microsoft-learn MCP: confirm `IFloatingPointIeee754<T>` / `Half` / tensor API availability
  (no new BCL types needed here; `Unsafe.AreSame` and `MemoryMarshal.GetReference` are
  existing patterns in `LayaNativeCheckpointTests`).
- code-memory MCP: re-verify the five bind sites and the model ctor chains before editing.
- Blast radius stated above; confirm no other `Weight!.Tensor =` sites outside the five
  (grep showed exactly those five in `StateDictLoader`).

## Notes

- `LoadLinearWidened` (F32-from-F16) still copies once — it must, it converts dtype — so
  that path halves rather than zeroes. Only the ctor garbage goes away.
- `Parameter.Version` has no consumers, so late binding is safe. No transposed-weight cache
  exists inside `Linear`.
- The iterative-work skill requires committing `docs/TODO.md` as its own logical unit before
  implementation; that commit is this file.
