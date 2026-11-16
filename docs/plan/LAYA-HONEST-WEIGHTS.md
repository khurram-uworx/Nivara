# Laya — default to native loading (F16 storage, F32 compute)

**Status:** Plan (draft, 2026-11-16)
**Issue:** #1218 — "Laya: default to native loading (F16 storage, F32 compute)"
**Phase:** Phase 2 of `docs/plan/MODEL-LOADING-HONESTY.md`
**Owner:** (unassigned)
**Related:** PR #1215 (`khurram/smollm`, SmolLM native default) and PR #1220 (Qwen native default) — both merged; issue #1219 (Silk BF16 decode crash); `docs/LAYA.md`, `docs/MODERNBERT.md`, `docs/HALF-F16.md`, `docs/SILK.md`, `docs/SMOLLM.md`.

---

## 1. Goal

Make a bare `laya` run hold the checkpoint's own F16 weights and compute in F32, instead of
widening every weight to F32 at load:

| | today (`--precision f32`) | after (`native`, the new default) |
|---|---|---|
| Storage dtype | F32 (widened at load) | **F16**, exactly as shipped |
| Compute dtype | F32 | F32 (unchanged) |
| Weight bytes resident | 1,685,175,320 B (1.569 GiB) | **842,587,654 B (803.6 MiB)** payload + 176,658 B of widened 1-D tensors |
| Payload arrays | reinterpreted into new F32 arrays | **the checkpoint's own arrays** |
| `--precision f32` | — | still available, unchanged |

The claim is **footprint, not CPU speed**. As with SmolLM (`docs/SMOLLM.md`), the native path is
not promised to be faster; the promise is that a byte of the checkpoint is never rewritten into a
second resident copy. Where SmolLM's win was BF16, Laya's is F16 — the same mechanism with a
different narrow dtype.

This is *not* the same change as flipping `--precision fp16` (pure-Half compute). F32 compute stays,
because Laya's activations, RoPE tables and logits are all validated in F32 by the wheel gate, and
Half compute would silently move that gate's meaning. The memory win comes from *storage*.

---

## 2. What the checkpoint actually is

Verified against the local snapshot (`samples/data/laya/model.safetensors`,
842,609,210 bytes, 206 tensors, 421,293,830 parameters):

| grouping | count | dtype |
|---|---|---|
| `encoder.*` | 170 | F16 |
| `head.*` | 24 | F16 |
| `scorer.*` | 6 | F16 |
| `act_head.*` | 4 | F16 |
| `type_emb.weight` `[3, 1024]` | 1 | F16 |
| `temperature` `[3]` | 1 | **F32** |

205 of 206 tensors are F16; the single F32 tensor is a 3-element buffer. So the model is
**effectively uniform F16**, and the "one `TWeight` for the whole model" decision recorded in
`MODEL-LOADING-HONESTY.md` §4 holds — no per-tensor dtype plumbing is needed.

Encoder config (`samples/data/laya/encoder/config.json`): 28 layers, `hidden_size` 1024,
`intermediate_size` 2624, 16 heads of 64, `vocab_size` 50368, `layer_types` +
`rope_parameters` (the newer ModernBERT layout), no biases anywhere under `encoder.*`.

Parameter split, which every residency number below is built from. Each row is summed from the
safetensors header and the rows add up to the total exactly:

| role | elements |
|---|---|
| encoder projections (28 × 12,255,232) | 343,146,496 |
| token embedding (`encoder.embeddings.tok_embeddings`) | 51,576,832 |
| encoder LayerNorm gammas (57 × 1024) | 58,368 |
| head (`head.*`) | 25,192,448 |
| scorer (`scorer.*`) | 1,052,673 |
| act head (`act_head.*`) | 263,938 |
| type embedding (`type_emb.weight`) | 3,072 |
| `temperature` (F32) | 3 |
| **total** | **421,293,830** |

That is 1,685,175,320 B as F32 (1.569 GiB) vs 842,587,654 B as F16 (803.6 MiB), the same
1.685 GB device figure `docs/LAYA.md` §"GPU path" records. The file itself is 842,609,210 B; the
21,556 B difference is 21,544 B of safetensors framing (8-byte length prefix + 21,536-byte JSON
header, padding included) plus the 12-byte F32 `temperature`. Role grouping is not prefix grouping:
the token embedding is nested under `encoder.*`, which is 170 tensors / 394,781,696 elements on its
own.

The encoder has no biases — every 1-D tensor under `encoder.*` is a `*_norm.weight` — and layer 0
has no attention norm at all: `ModernBertLayer`'s constructor creates `attnNorm` only when
`layerIndex != 0` (`src/Nivara.Extensions/AI/ModernBertModel.cs::ModernBertLayer`), and the
checkpoint accordingly ships 27 `attn_norm.weight` tensors, not 28. Encoder LayerNorms are
therefore 57: `embeddings.norm`, `final_norm`, 27 `attn_norm` and 28 `mlp_norm`.

The only tensors widened to F32 in memory on the native path are the checkpoint's 1-D tensors —
**88,323 elements**, 58,368 encoder gammas plus 29,955 head/scorer/act-head norms and biases —
because `LoadLayerNorm<TModel, TWeight>` and the mixed `LoadLinear` convert them at load. That is
`88,323 × 4 = 353,292 B` of F32 copies, three orders of magnitude below the weights and not worth a
second storage dtype. (The F16 originals stay in the loader dictionary, so the two are additive
while it is alive, not a replacement.)

---

## 3. Why this is not a flag flip

`Laya.Run<TModel, TWeight>` and `LayaDecisionHead<TModel, TWeight>` already accept independent
compute and storage dtypes, and `Linear<TModel, TWeight>.Forward` already routes through
`GradKernels.MixedMatMulTransposedB<TModel, TWeight>`
(`src/Nivara/AutoDiff/Nn/Linear.cs::Linear.Forward`), which has a real `float × Half` arm
(`src/Nivara/AutoDiff/Operations/GradKernels.cs::MixedMatMulTransposedB` →
`src/Nivara/Primitives/NarrowFloatKernels.cs::MatMulTransposedB(ReadOnlySpan<float>, ReadOnlySpan<Half>, …)`).
The **decision head is therefore already honest today** — it never widens a weight.

Four things in the *encoder* and the *loader* are not, and each is a memory defect rather than a
numerical one. `Laya.Run<float, Half>` compiles and produces correct numbers right now; it is
simply unusable, because it allocates more than the F32 path it was meant to replace.

### 3.1 The fused encoder forward widens every projection, every call

`ModernBertLayer<TModel, TWeight>.ForwardFused`
(`src/Nivara.Extensions/AI/ModernBertModel.cs::ModernBertLayer.ForwardFused`) reads its weights
through `WeightSpan<TParam>(Parameter<TParam>)`, whose non-matching branch allocates a fresh
`TModel[]` and scalar-converts into it:

```csharp
var widened = new TModel[sourceSpan.Length];
for (int i = 0; i < widened.Length; i++)
    widened[i] = TModel.CreateChecked(sourceSpan[i]);
return widened.AsSpan();
```

`ForwardFused` calls it 7 times per layer (`qProj`, `kProj`, `vProj`, `oProj`, `inputProj`,
`gateProj`, `downProj`), so a 28-layer forward is 196 allocations in which **343,146,496 projection
elements = 1.3726 GB** of F32 weights are converted and immediately thrown away. (The LayerNorm
gammas do not allocate: `LayerNorm<TModel>` stores them already widened at load, so `WeightSpan`
takes its same-type branch — and layer 0 has no attention norm, so it reads only one gamma.) The
`Linear`-based paths do not have this problem; only the fused one does.

Fix: keep the `TWeight` span and use the mixed kernel. `WeightSpan` splits into two narrow helpers
that cannot silently widen:
`ParamSpan(Parameter<TWeight>)` — the storage span straight from
`Parameter.Tensor.Data.TryGetSpan`, no conversion — for the seven projection weights, and
`ModelSpan(Parameter<TModel>)` — the same read, no conversion branch — for the LayerNorm gammas,
which `LoadLayerNorm<TModel, TWeight>` already widened at load. After the edit there is no widening
path left in the fused forward at all; the seven projection calls become
`GradKernels.MixedMatMulTransposedB<TModel, TWeight>(…)`.

The per-head attention matmuls inside `ForwardFused` — `qh × khᵀ` and `scores × vh` — keep their
existing same-dtype kernels: both operands are already `TModel` and neither is a weight.

### 3.2 The encoder widens the whole embedding table to read `seqLen` rows

`ModernBertEncoder<TModel, TWeight>.Forward(int[], int)`'s inference path starts with:

```csharp
var weightData = WeightSpan(tokenEmbedding.Weight);   // widens all 51,576,832 elements
for (int i = 0; i < seqLen; i++)
    weightData.Slice(id * hiddenSize, hiddenSize).CopyTo(hiddenBuf.AsSpan(i * hiddenSize, hiddenSize));
```

That is a **206.3 MB F32 allocation per forward** to gather at most 512 rows (2.1 MB at
`max_len` 512).

Fix: gather rows in the storage dtype and convert per element, which is exactly what
`Embedding<TModel, TWeight>.ForwardCore` already does for the decision head
(`src/Nivara/AutoDiff/Nn/Embedding.cs::Embedding.ForwardCore`). Add an internal
`Embedding<TModel, TWeight>.GatherRows(int[] tokenIds, Span<TModel> destination)` that reads the
`TWeight` span and converts `seqLen × hiddenSize` elements, and call it from
`ModernBertEncoder.Forward`. For `TWeight == TModel` it degenerates to a `CopyTo` per row, so the
F32 path keeps its current cost.

### 3.3 The loader copies every fused block

`StateDictLoader.LoadLinearSlice<TModel, TWeight>(Linear<TModel, TWeight>, …)` and
`LoadFusedLinearSlice<TModel, TWeight>(Linear<TModel, TWeight>, …)`
(`src/Nivara.Extensions/AI/StateDictLoader.cs`) split ModernBERT's fused `Wqkv` and `Wi` into
row blocks with `Array.Copy` into a fresh `TWeight[]`:

```csharp
var slice = new TWeight[rowCount * inFeatures];
Array.Copy(w.Data, rowOffset * inFeatures, slice, 0, slice.Length);
linear.Weight!.Tensor = ReverseGradTensor<TWeight>.FromMatrix(slice, rowCount, inFeatures, requiresGrad);
```

For Laya that is `28 × (3·1024² + 2·2624·1024) = 238,551,040 elements = 477.1 MB` of F16
duplication sitting next to the payload it was copied from — the single biggest reason native would
otherwise land at ~1.32 GB instead of ~0.85 GB. The non-fused projections are already zero-copy
(`LoadLinear`'s mixed overload assigns `ReverseGradTensor<TWeight>.FromMatrix(w.Data, …)` over the
loader's own array), so the fused blocks are the whole of the gap.

Fix: a zero-copy row view. `ColumnStorage<T>.Slice` is a documented true zero-copy view
(`src/Nivara/Storage/ColumnStorage.cs::Slice`), and `NivaraColumn<T>.Slice` exposes it publicly
(`src/Nivara/NivaraColumn.cs::Slice`); `Dispose` only flags the instance, never the shared array.
So the loader can build the block as:

```
NivaraColumn<TWeight>.CreateFromOwnedArray(w.Data)   // shares the payload array
    .Slice(rowOffset * inFeatures, rowCount * inFeatures)
    → new ReverseGradTensor<TWeight>(column, requiresGrad, new[] { rowCount, inFeatures })
```

using the existing internal shape-taking constructor
(`src/Nivara/AutoDiff/ReverseGradTensor.cs::ReverseGradTensor(NivaraColumn<T>, bool, int[])`).
`Nivara.Extensions` already has `InternalsVisibleTo` on `Nivara` (`src/Nivara/Nivara.csproj`), so
no visibility change is needed.

Two constraints this must respect, both already true:

- The loader's dictionary arrays must not be mutated afterwards. That is already the contract
  (`ReverseGradTensor.FromMatrix` documents it), and the snapshot is read-only by construction.
- `Linear`'s `Parameter<T>.Tensor` setter bumps `Version`, invalidating cached derived views. The
  slice assignment goes through the same setter, so nothing changes there.

**Measurement honesty:** slices are visible to the residency gate as "the payload array's own
memory at an offset", so the gate must assert the *base array* identity
(`Unsafe.AreSame` on the element reference at `rowOffset·inFeatures`), not just element equality.
Otherwise a copy passes.

### 3.4 `MixedMatMul` has no `Half` arm, so mixed-dtype backward widens per call

`GradKernels.MixedMatMul<TModel, TWeight>` (the `gradOutput @ W` direction used for the *input*
gradient in `Linear.Backward`) has a `BFloat16` fast arm but no `Half` arm, so `float × Half` falls
through to `Array.ConvertAll(b.ToArray(), x => TModel.CreateChecked(x))` — widening the entire
weight matrix on every backward call. Its sibling `MixedMatMulTransposedB` *does* have the Half arm
(`GradKernels.cs`, the `typeof(TWeight) == typeof(Half)` branch), and
`NarrowFloatKernels.MatMul(ReadOnlySpan<float>, ReadOnlySpan<Half>, …)` already exists, so this is
an asymmetry, not a missing kernel.

Fix: add the symmetric 12-line branch. **No new "mixed-dtype training is unsupported" guard is
needed for Laya**, and inventing one would be wrong: the module path is dtype-generic and correct,
and the only inference-only surface — `ForwardFused` — already throws inside
`GradientUtils.Grad()` (`ModernBertModel.cs::ModernBertLayer.ForwardFused`, the
`GradientUtils.IsGradEnabled` guard). Laya native is inference-default, not
inference-only-by-accident; a future Half-storage head fine-tune works and is now fast.

---

## 4. Design

### 4.1 CLI and dispatch (`samples/NivaraInference/Program.cs`)

1. **Storage dtype is per model, not per flag.** Today `if (bf16 || native)` loads
   `SafeTensorsLoader.Read<BFloat16>` for *every* native run (`Program.cs:718`), and `if (fp16)`
   loads `Read<Half>` into `tensorsHalf` (`Program.cs:728`). Add the Laya case, and add the `isLaya`
   local next to `isQwen` / `isSmolLM` (`Program.cs:144`):

   ```csharp
   bool isLaya = modelType == "laya";
   bool layaNative = native && isLaya;            // F16 storage, F32 compute
   bool layaNativeCpu = layaNative && !useSilk;   // Silk has no Half device path (§4.6)
   if (bf16 || (native && !layaNative)) { … Read<BFloat16> … }
   if (fp16 || layaNativeCpu)           { … Read<Half> …     }
   ```

   | run | map loaded |
   |---|---|
   | `smollm` / `qwen`, `native` | `Read<BFloat16>` |
   | `laya`, `native` (CPU) | `Read<Half>` |
   | `laya`, `--silk` + default/native | F32 only — no Half map, and no BF16 map either |
   | `laya`, explicit `--precision bf16` | `Read<BFloat16>` |

   The `--silk` case is the reason for two booleans rather than one: a bare `native` run on the
   Silk path consumes F32 (the device path is F32, §4.6), so loading either narrow map would hold a
   ~400 MB array nothing reads. That is the exact defect this plan exists to remove, so do not
   reintroduce it in the load block.

   `Read<Half>` over an F16 checkpoint is a zero-hop native read
   (`src/Nivara.Extensions/AI/SafeTensorsLoader.cs::ConvertF16` — the `typeof(T) == typeof(Half)`
   branch returns the payload bits), and the single F32 `temperature` tensor goes through
   `ConvertF32<Half>` (3 conversions, harmless — see §4.6).

   Reading Laya as `BFloat16` instead would be a silent disaster: every one of the 421 M elements
   would go through `ConvertF16<BFloat16>`'s per-element `CreateChecked`, and the resident bytes
   would be BF16-roundtripped F16 — neither the snapshot's dtype nor its precision.

2. **Default rule.** `isLaya` joins the existing default at `Program.cs:334`:
   `if ((isSmolLM || isQwen || isLaya) && !precisionSet) precision = "native";`.
   Both of Laya's reachable modes (`default`, `benchmark`) consume narrow weights, so no mode gate
   is needed. Do **not** build this on `ConsumesNarrow`'s
   `"laya" or "modernbert" => mode is not ("compare" or "compare_diag")` clause: the sample no
   longer exposes a `laya compare` mode (`SupportedModes["laya"]` is `["", "benchmark"]`, so the
   mode allow-list rejects it before `ConsumesNarrow` is consulted), and the wheel gate lives in
   `tests/Nivara.PerformanceTests --inference-compare laya`, which loads F32 through the
   non-generic `SafeTensorsLoader.Read(path)` and is untouched by this change.

3. **The native guard.** `if (native && !(isSmolLM || isQwen))` at `Program.cs:626` becomes
   `if (native && !(isSmolLM || isQwen || isLaya))`, and the message — currently "`--precision
   native` is a 'smollm'/'qwen' option only: these keep their BF16 snapshot weights and compute in
   F32" — is rewritten to name each model's storage dtype. The comment above it already records
   "Laya ships F16 weights", so update it rather than deleting it. Keep the `qwen distill` native
   rejection untouched.

4. **Dispatch.**
   ```csharp
   if (layaNativeCpu) return Laya.Run<float, Half>(tensorsHalf, modelDir, mode, "native (F16 storage, F32 compute)");
   if (fp16)          return Laya.Run<Half, Half>(tensorsHalf, modelDir, mode, "Half");
   ```
   `Laya.Run<TModel, TWeight>` already dispatches `benchmark` vs. bare inference and prints the
   precision label, so no new entry point is required — unlike SmolLM, which needed a separate
   `RunSmolLMNative` because it has generation modes.

5. **`--silk` overrides native.** Laya Silk is `bf16` or `F32` on the device; there is no Half
   device path (`Program.cs:656` rejects `--silk --precision fp16` for every model). Mirror Qwen's
   `#1219` note (`Program.cs`, the `qwen` + `native` silk branch):
   ```csharp
   if (native) Console.WriteLine("Note: laya --silk with native/default precision runs the F32 device path (no Half device path); use --precision bf16 for BF16 Silk.");
   ```
   `layaNativeCpu` is false for `--silk`, so no Half map is loaded and the existing `useSilk`
   branches take `F32()`. Print the note in the **argument-validation block** (next to the
   `useSilk && fp16` guard at `Program.cs:656`), not in the dispatch: `#589` requires every
   observable CLI behaviour to be reachable before the first disk touch, and that is what lets the
   CLI gate assert the note from an empty working directory.

6. **Help text.** `Program.cs:408-413` currently reads "Default for smollm (BF16 weights, F32
   activations, ~269 MB resident instead of ~1.05 GB); rejected for every other model, whose
   checkpoints differ or whose default is F32." It must list all three defaults with their storage
   dtypes and residency — SmolLM BF16 (~269 MB vs ~1.05 GB), Qwen BF16, Laya F16 (~843 MB vs
   ~1.69 GB) — and stop implying native is SmolLM-only. Two adjacent lines are already stale and
   should be fixed in the same edit: the `--precision fp16` note says "rejected where the run is
   F32-only: the vision models and laya/modernbert 'compare'", but neither `laya` nor `modernbert`
   exposes a `compare` mode any more (§4.1(2)). Also update the `--precision fp16` line to say
   Laya's *default* is now F16 storage / F32 compute, so `fp16` remains the pure-Half compute
   opt-in.

### 4.2 Encoder fused forward — widen in-register, never per call

`ModernBertLayer.ForwardFused` keeps its structure and its `[Category]`-free inference contract.
Changes are mechanical:

- Replace `WeightSpan(Parameter<TParam>)` with two helpers that cannot silently widen:
  `ParamSpan(Parameter<TWeight>)` for the projection weights — the raw storage span, asserting
  null-free as `WeightSpan` does today — and `ModelSpan(Parameter<TModel>)` for the LayerNorm
  gammas, which `LoadLayerNorm<TModel, TWeight>` already widened at load. `ModelSpan` keeps no
  conversion branch, so a `TModel` read can never allocate. The existing `attnNorm != null` guard
  (layer 0) and the `Bias == null` empty-span case stay as they are.
- The seven projection matmuls (`qProj`, `kProj`, `vProj`, `oProj`, `inputProj`, `gateProj`,
  `downProj`) call `GradKernels.MixedMatMulTransposedB<TModel, TWeight>` with the `TWeight` span.
  The output buffers, `ArrayPool` rentals, diagnostics records and `KernelType` reporting are
  unchanged (the kernel type is still determined for `TModel`; add the weight dtype to the
  diagnostics payload so a native run is distinguishable from an F32 run in a trace).
- `ModernBertEncoder.Forward`'s `WeightSpan(tokenEmbedding.Weight)` becomes
  `tokenEmbedding.GatherRows(tokenIds, hiddenBuf)` (§3.2); `embedNorm` / `finalNorm` read through
  `ModelSpan`.

This makes `ModernBertLayer.ForwardFused` genuinely `TWeight`-generic for the first time. The
existing F32 path executes the same code with `TWeight == TModel`, so it stays bit-identical:
`MixedMatMulTransposedB` short-circuits to `MatMulTransposedB<TModel>` when the two types match.

### 4.3 Loader — zero-copy fused-block views

Add a private helper in `StateDictLoader` and use it from both mixed-target slice overloads:

```csharp
static ReverseGradTensor<TWeight> RowBlockView<TWeight>(
    TWeight[] source, int rowOffset, int rowCount, int inFeatures, bool requiresGrad)
```

- Builds `NivaraColumn<TWeight>.CreateFromOwnedArray(source)` once per call and slices it. (A
  per-key cache is unnecessary — the column is a 40-byte wrapper over the shared array — but if a
  profile shows the wrapper cost, cache it in a `Dictionary<string, NivaraColumn<TWeight>>` keyed by
  tensor name.)
- Keeps every existing validation (`rowOffset`, `rowCount`, fused-row bounds, `inFeatures` /
  `OutFeatures` agreement) *before* the view is taken, so a bad block still throws with the same
  message.
- Biases stay copies: they are 1024–2624 elements, and `LoadFusedLinearSlice` also converts them to
  `TModel`, which is the intended widened set.

`LoadLinear` / `LoadEmbed` need no change — they already assign the loader's array by reference.

**Load-time transient (documented, not fixed here):** `Linear<TModel,TWeight>` and
`Embedding<TModel,TWeight>` constructors allocate and Kaiming/Normal-initialise a weight array that
`LoadWeights` immediately replaces. For Laya Half that is ~839.8 MB of short-lived garbage
(686.3 MB projections + 50.3 MB head + 103.2 MB embedding), and ~1.68 GB on the F32 path. It is an
existing cost of *every* model load in this repo, so it is not a regression and not a blocker — but
the residency gate must measure **after a forced full GC**, and a follow-up can add a
`Linear(..., skipInitialize: true)` load path if load time becomes the complaint. Do not fix it in
this change; it touches every model's constructor.

### 4.4 The single F32 `temperature` buffer

`temperature [3]` is F32 on disk and is read into `Half[]` by `SafeTensorsLoader.Read<Half>`
(`ConvertF32<Half>`), then consumed once in `LayaDecisionHead.LoadWeights`
(`src/Nivara.Extensions/AI/LayaHeadModel.cs::LayaDecisionHead.LoadWeights`) into
`TModel[] checkpointTemperature`.

Decision: **leave it in the single-dtype map.** The buffer is 3 elements, `LayaCalibration` does not
apply it, and `docs/LAYA.md` already records that it is loaded and not used (the shipped values
clamp). Half rounding of 0.1006-style values is ~1e-4 relative — irrelevant for a value that is not
applied, and the printed raw value is documented as raw. The alternative (a second `float[]`
reader for one 12-byte tensor) costs a loader overload and a model-API change for nothing. Record
the rounding in `docs/LAYA.md` §"Temperature is not in the head" so the next reader does not
re-derive it.

### 4.5 NivaraChat and the adaptive head

`NivaraChat` uses `LayaDecisionHead<float, float>` for adaptive training
(`samples/NivaraChat/Helpers/AdaptiveTraining.cs`, `AdaptiveFeedbackCollector.cs`,
`AdaptiveHeadPersistence.cs`). That stays widened F32: it *trains*, it persists F32 state dicts, and
`Linear.StateDict()` converts the weight to `TModel` anyway. No change. Do not propagate the native
default into the chat sample's head.

### 4.6 Silk

Unchanged and explicitly documented as an asymmetry, mirroring SmolLM's CPU-only `ab` mode:

- `laya --silk` stays `F32` (or explicit `--precision bf16`) on the device path.
- `--silk --precision fp16` stays rejected (already is, for every model).
- `--silk` + default/native prints the one-line note in §4.1(5) and runs F32.

`docs/SILK.md` gets the sentence, and `docs/LAYA.md` §"GPU path" already quotes the F32 device
arithmetic (1.685 GB device buffers) — that stays true and is now the *reason* the device path is
F32.

---

## 5. Residency arithmetic

| role | elements | F32 (today) | native (target) |
|---|---|---|---|
| 2-D weights, kept by reference | 421,205,504 | 1,684,822,016 B | **842,411,008 B** |
| 1-D tensors, widened at load | 88,323 | 353,292 B | 353,292 B (F32) + 176,646 B (F16 originals still in the map) |
| `temperature` (F32 on disk) | 3 | 12 B | 12 B (widened in the head) |
| **model-referenced total** | **421,293,830** | **1,685,175,320 B** | **842,764,312 B** |
| fused-block copies | — | 0 (nothing fused is stored) | **0** (was 477.1 MB before §4.3) |
| `Linear`/`Embedding` ctor garbage | — | ~1.68 GB transient | ~839.8 MB transient (GC'd) |

The native total is 176,658 B above the checkpoint payload (842,587,654 B) because the encoder
gammas plus the head norms and biases are copied to F32 while the loader map keeps the F16
originals: the widened set is duplicated, not replaced, until the map is dropped. Both numbers are
pinned in §7.2.

- 2-D weights are `421,205,504` elements: encoder projections `343,146,496`, token embedding
  `51,576,832`, head `25,165,824`, scorer `1,049,600`, act head `263,680`, `type_emb` `3,072`.
- 1-D tensors are `88,323` elements: encoder gammas `58,368`, head norms and biases `26,624`,
  scorer `3,073`, act head `258`.

---

## 6. Acceptance criteria

1. A bare `laya` run loads `SafeTensorsLoader.Read<Half>` and runs `Laya.Run<float, Half>`; the
   banner reads `Device: CPU (.NET …), precision native (F16 storage, F32 compute)`.
   Neither `RunInference` nor `RunBenchmark` prints a parameter line today — the only
   `ParameterLine` overloads live in `samples/NivaraInference/Laya.Narrow.cs` and serve the GPU
   paths. Fold those two into one `ParameterLine<TWeight>` that derives its label and bytes per
   element from `typeof(TWeight)` / `Unsafe.SizeOf<TWeight>()`, and call it from all four paths.
   The native line then reads `Parameters: 421,293,830 (803.6 MB as Half / 1607.1 MB as F32)`
   (existing MiB arithmetic, `MB` label), so the footprint claim is visible in run output rather
   than only in a test.
2. Every projection, fused block and the embedding table the model holds is **the loader's own
   array memory** — full-array identity for `oProj`/`Wo`/`embedding`, base-array-offset identity for
   `Wqkv`/`Wi` blocks. No copy, no widened duplicate.
3. One forward pass allocates **no weight-sized** buffer: the freed/allocated bytes attributable to
   widening are zero (measured, not argued — §7.3).
4. The `--inference-compare laya` probe is unchanged: still `Read<float>`, still
   `InferenceCompare.LayaCompare`, same bound.
5. Native vs `--precision f32` logits on the four fixture questions agree to float
   summation-order noise, pinned by a documented bound (§7.2).
6. `--precision f32` still runs the old path and still costs ~1.7 GB; `--precision fp16` still runs
   pure-Half compute; `--precision bf16` still runs BFloat16.
7. `laya compare` / `compare_diag` stay rejected by the mode allow-list (unchanged message), and
   `laya --silk` still runs F32 with the note.
8. `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` green; the Docs test
   namespace green; targeted inference tests green.

---

## 7. Verification

### 7.1 The wheel gate (unchanged, and why)

The gate is `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --inference-compare
laya` (`tests/Nivara.PerformanceTests/InferenceCompareProbe.cs` →
`samples/NivaraInference/InferenceCompare.cs::LayaCompare`). It loads the checkpoint through the
non-generic `SafeTensorsLoader.Read(path)`, which is `Read<float>`
(`src/Nivara.Extensions/AI/SafeTensorsLoader.cs::Read(string)`), and the Python side
(`samples/NivaraInference/Python/laya_compare.py`) builds the model with
`common.build_model(cfg, encoder_dir=…)`, `load_state_dict(weights, strict=True)`, `model.eval()`
and calls `.float()` on outputs — **F32 compute**, no `autocast`, no dtype override. So the wheel
reference is an F32-compute reference, which is exactly what native runs. The existing bound
(`|diff| <= 1e-3·(1 + |ref|)`, observed max 1.3e-5) carries over unmodified and needs no re-pinning.

State this in the PR: the gate compares F32 compute against F32 compute, so it does **not** pin
F16 *compute*. That is the reason `--precision fp16` remains opt-in, and why nothing here changes
the gate's meaning.

### 7.2 New checkpoint gate: `LayaNativeCheckpointTests`

New fixture under `tests/Nivara.Tests/Inference/`, modelled on
`SmolLMNativeCheckpointTests` (same `[Category("Performance")]` + skip-with-reason-when-absent
structure; the 421 M-parameter build is well over the 2-second rule). Three properties:

1. **`LoadedTensors_AreTheCheckpointPayloadBitForBit`** — 205 tensors F16, 1 tensor F32, 206 total,
   `421,293,830` parameters; `MemoryMarshal.AsBytes(data).SequenceEqual(payload)` for every tensor
   read as `Half` except `temperature`, which must round-trip through `ConvertF32<Half>` and is
   asserted as 3 elements. A re-upload at F32 or BF16 fails here, naming the tensor.
2. **`NativeModel_KeepsTheCheckpointArrays_AndOnlyTheOneDimensionalTensorsAreWidened`** — walk the
   encoder and head; `Unsafe.AreSame` on the element reference for every projection, every
   `Wqkv`/`Wi` row block (at `rowOffset · inFeatures`), the embedding and `type_emb`; then assert
   the by-reference set is `421,205,504` elements / `842,411,008 B` and the widened set is exactly
   `88,323` elements / `353,292 B` (58,368 encoder gammas + 26,624 head + 3,073 scorer + 258
   act head) — every 1-D tensor in the file and nothing else. Element equality alone would pass a
   copy, so identity is the assertion; the named 1-D total is what fails if a `LayerNorm` or a
   `Linear` bias starts round-tripping a 2-D block.
3. **`NativeForward_MatchesTheF32Path`** — encoder forward + head forward on the four fixture
   questions, `float × Half` vs `float × float`, assert
   `max|native − f32| <= 1e-4·(1 + |f32|)` and identical argmax / decision / temperature bucket.
   The bound is summation order (`NarrowFloatKernels` accumulates 8-wide in `Vector128<float>`;
   `TensorPrimitives` uses a different reduction), not precision loss — Half→float is exact, so this
   is a structural check on the kernel swap, not a tolerance band over a defect.

### 7.3 Allocation bound (the property that makes §3.1/§3.2 impossible to regress)

In the same fixture, with `GradientUtils` off and warm-up excluded:
`GC.GetAllocatedBytesForCurrentThread()` around one `encoder.Forward(ids, valid)`. Assert the
allocation is below the activation-buffer envelope (output + rented scratch are pooled, so the
measured value is small). The two failure modes this catches, with their sizes, are stated in the
assertion message: a per-forward weight widen is ≥1.37 GB and an embedding-table widen is ≥206 MB;
the bound sits at a few MB, so either regression fails loudly rather than by a factor.

Do the same for `Linear<float, Half>.Backward` (weight `RequiresGrad = true`) to pin §3.4: the
`Array.ConvertAll` fallback widens the whole matrix per call, the Half arm does not.

### 7.4 CLI surface

`tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs` currently asserts
`AssertRejected(Run("laya", "--precision", "native"), "--precision native is a 'smollm'/'qwen' option only")`.
That assert inverts:

- `laya --precision native` is **accepted**: with the gate's empty working directory the deepest
  reachable point becomes `Model file not found` (`ModelLookupMessage`), so assert that and not a
  guard message. The fixture's class `<para>` claims "the `native` precision's `smollm`-only scope";
  rewrite it with the new scope in the same commit, per the coverage-in-every-summary rule.
- `minilm`, `mobilenet_v2`, `modernbert` still reject, with the updated message naming all three
  models.
- `laya compare` / `compare_diag` stay rejected by the mode allow-list (`Unsupported mode 'compare'
  for 'laya'`, already pinned in `InferenceCliArgumentTests`) — that rejection is not the `native`
  guard and does not move. There is no `ConsumesNarrow` guard to name here: the flag guard at
  `Program.cs:665` covers only `bf16 || fp16`, and `native` is gated by the allow-list alone.
- `laya --silk` with default precision still runs F32 and prints the note; because §4.1(5) puts the
  note in the validation block, assert it on the `Model file not found` path in the empty working
  directory rather than needing a checkpoint.

---

## 8. Docs and pins

Files to change:

- **`docs/LAYA.md`**
  - §"Model overview": replace "widened to F32 at load, so the resident set is about 1.7 GB" with
    the native/F32 split (842.6 MB F16 storage vs 1.685 GB F32 weights), and keep `--precision f32`
    as the documented wide option.
  - Add a **"Native loading"** section mirroring `docs/SMOLLM.md` §"Native loading": storage dtype
    = snapshot dtype, compute = F32, widening happens in `NarrowFloatKernels`' dot kernels, the
    `temperature` rounding note, and the Silk asymmetry.
  - §"The load path" fenced diagram: `SafeTensorsLoader.Read<float>` → `Read<Half>` with a note that
    `--precision f32` still reads `<float>`.
  - §"Performance": add a native vs F32 row only when a **human-run** measurement exists; do not
    quote a derived number. Update the "Weight bind was 6.6 s …" sentence once native load time is
    measured. Protocol stays: one untimed pass, three timed, median, AC, Release, F32 compute.
- **`docs/MODERNBERT.md`**: note that `ModernBertLayer.ForwardFused` is now `TWeight`-generic and
  that the encoder gathers embedding rows in the storage dtype.
- **`docs/HALF-F16.md`**: the `MixedMatMul` Half arm (symmetry with BF16) and the "widen
  in-register, never at load and never per call" rule with Laya's before/after numbers.
- **`docs/SILK.md`**: one line under the BERT-family section that Laya Silk stays F32/bf16 and why.
- **`samples/NivaraInference/README.md`**: the precision table row for `laya`.

**Pins.** `docs/plan/` is excluded from every `DocsCoveragePins` scan, so *this* file moves nothing.
The `docs/` edits above do: adding `file.cs::Symbol` citations and cited documents moves
`ExpectedCitationCount` (122), `ExpectedCitedDocumentCount` (20) and possibly
`ExpectedCodeCitationCount` (53). Extend each pin's own arithmetic comment rather than bumping the
number, per `AGENTS.md`. No new `docs/*.md` file is added, so `ExpectedDocumentCount` (60) should
not move — if it does, something is wrong.

---

## 9. Commit breakdown

One reason per commit, in dependency order. Commits 1–4 are dtype-neutral: they change F32 numerics
not at all and can be reviewed (and merged) before the CLI default flips.

1. **`GradKernels.MixedMatMul`: add the Half arm.** Mirror `MixedMatMulTransposedB`'s branch; test
   `Linear<float, Half>` backward is allocation-bounded and numerically equal to the widen path.
2. **`Embedding.GatherRows` + `ModernBertEncoder.Forward` gather.** Removes the 206 MB per-forward
   table widen; F32 path unchanged (covered by the existing ModernBERT tests).
3. **`ModernBertLayer.ForwardFused` mixed-dtype projections.** `ParamSpan`/`ModelSpan`,
   `MixedMatMulTransposedB<TModel,TWeight>`, weight dtype added to the diagnostics payload; test the
   `float × Half` forward equals `float × float` within the §7.2 bound and allocates no
   weight-sized buffer.
4. **`StateDictLoader` zero-copy fused-block views.** Both mixed-target slice overloads; identity
   assertions in a loader unit test (a tiny synthetic fused tensor is enough — this one does not
   need the real checkpoint).
5. **CLI: Laya native default + `Read<Half>` + guards + help text + `ParameterLine<TWeight>` +
   `InferenceCliArgumentTests`.** Includes the `--silk` note and the per-model map selection (§4.1(1)).
6. **`LayaNativeCheckpointTests`** (§7.2, §7.3).
7. **Docs + pins** (§8), and `docs/LAYA.md`'s performance table update if the human-run
   measurement is available by then.

Optional follow-up (separate issue, not this PR): `Linear`/`Embedding` load-path constructors that
skip initialisation when a state dict is about to replace the weight (load-time transient only).

---

## 10. Risks and non-goals

- **Residency ≠ working set.** The gate proves the *model's* storage is the payload; it does not
  watch the process working set, so a third copy made elsewhere escapes it. Say so in the test's
  `<remarks>`, as `SmolLMNativeCheckpointTests` does.
- **F16 launch-range risk on other checkpoints.** This plan is defined against *this* snapshot,
  whose 205 F16 tensors are all within Half's range. A future Laya re-upload in BF16 or F32 must
  re-derive `native`'s storage dtype, not inherit it. The dtype is asserted in the gate, so a
  re-upload fails there rather than silently running the wrong path.
- **Load-time transient garbage** stays (~839.8 MB Half, ~1.68 GB F32). Documented in §4.3; not a
  regression; bounded by a forced-GC measurement.
- **No speed claim.** SmolLM's own data showed mode medians overlapping from sweep to sweep. The
  same discipline applies: quote footprint, and quote latency only as a distribution once a
  human-run sweep exists.
- **Non-goals:** Half *compute* as the default; a Half/Mixed Silk device path; per-tensor storage
  dtypes; touching `NivaraChat`'s F32 adaptive head; fixing the `Linear` ctor transient here.

---

## 11. Commands

```text
# build + whitespace gate
dotnet build Nivara.slnx -c Release
dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore

# the new checkpoint gate (skips with a reason when the snapshot is absent)
dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~LayaNativeCheckpointTests"

# CLI surface + docs
dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~InferenceCliArgumentTests"
dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"

# the wheel gate (unchanged meaning; requires the laya wheel + tokenizer)
dotnet run -c Release --project tests/Nivara.PerformanceTests -- --inference-compare laya

# the default path and its F32 counterpart
dotnet run --project samples/NivaraInference -c Release -- laya
dotnet run --project samples/NivaraInference -c Release -- laya --precision f32
```

**Benchmark figures are human-run.** If §8's performance row is to be filled, the human runs
`laya benchmark` and `laya benchmark --precision f32` (AC, Release, same session, alternating
order) and reports the medians; the agent reads the output rather than launching the runs.

---

## 12. References

- Issue #1218 — "Laya: default to native loading (F16 storage, F32 compute)".
- `docs/plan/MODEL-LOADING-HONESTY.md` §3 Phase 2 (this work) and §4 "What NOT to expand".
- `docs/SMOLLM.md` §"Native loading" — the shipped precedent for the same mechanism on BF16.
- `docs/LAYA.md` — the wheel gate, the load path, the 1.7 GB claim, the performance table.
- `src/Nivara/AutoDiff/Nn/Linear.cs` (`Linear.Forward`, `Linear.Backward`),
  `src/Nivara/AutoDiff/Operations/GradKernels.cs` (`MixedMatMulTransposedB`, `MixedMatMul`),
  `src/Nivara/Primitives/NarrowFloatKernels.cs` (the `float × Half` kernels).
- `src/Nivara.Extensions/AI/ModernBertModel.cs` (`ForwardFused`, `ModernBertEncoder.Forward`,
  `WeightSpan` → `ParamSpan`/`ModelSpan`), `src/Nivara.Extensions/AI/StateDictLoader.cs` (mixed-target
  slice overloads), `src/Nivara.Extensions/AI/SafeTensorsLoader.cs` (`Read<T>`, `ConvertF16`,
  `ConvertF32`).
- `tests/Nivara.Tests/Inference/SmolLMNativeCheckpointTests.cs` — the gate shape to copy.
