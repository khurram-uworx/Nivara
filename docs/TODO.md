# MiniLM GPU (`minilm --gpu`, ILGPU, F32) — branch `khurram/minilm-gpu`

## Status

**Implementing.** Branch `khurram/minilm-gpu` off `khurram/distilbert-gpu` HEAD (PR #436 unmerged
— the MiniLM work builds on the GPU infra that only exists there; when PR #436 merges, this PR
retargets to `main`). Commits so far: plan doc; `BertEncoderGpuRunner` rename + config/naming-driven
generalization; `minilm --gpu` inference/benchmark/compare + CLS/L2 pooling + gates; token-type row
0 fix (gate caught the MiniLM-vs-DistilBERT embedding difference — see commit 3d25e6b).

Correctness gates run: `minilm --gpu compare` **GATE PASS** (hidden maxRel 1.0e-5, 0/245760
violations; pooled-embedding cosine 1.000000), `minilm --gpu` (L2 norm 1.000000), f32-only reject,
`distilbert --gpu` + `distilbert_sst --gpu` regression PASS, `minilm` CPU regression unchanged.
Remaining: AC-power `minilm --gpu benchmark` (human-gated), README/reflection/roadmap docs,
G2, TODO.md deletion, push/PR offer.

## Problem

`minilm` (all-MiniLM-L6-v2, 6-layer 384-dim BERT encoder, sentence embeddings) is CPU-only in
`samples/NivaraInference` (~64 ms/128-token forward). The DistilBERT GPU infrastructure proven in
PR #436 — `IlgpuRuntime`, Row4 tiled GEMM, BatchedAttention, LayerNorm/embed/GELU kernels, the
config-driven weight-uploading runner, `--gpu` wiring with F32-only reject, and the
`|gpu − ref| ≤ 1e-3·(1+|ref|)` gate — applies to MiniLM unchanged: same kernel set, same
`BertEncoder` compute graph, smaller dims (384/1536/12/6), BERT-style weight keys (already
prefix-stripped in the provisioned file: `embeddings.*`, `encoder.layer.N.*`, no head).

MiniLM GPU is the cheap second model that proves the "one config-driven runner, N encoder models"
shape. It is a **correctness/generalization milestone, not a speed story**: at ≈1.36 GMAC (¼ of
DistilBERT) the GEMM legs drop to ~4–6 ms but the ~100 kernel-launch overhead per forward stays,
so the honest expectation versus the 64 ms CPU is only ~1.3–1.8× until M2 (kernel fusion / lazy
stream) lands — a separate tracked follow-up that lifts both models.

## Scope (locked, mirrors DistilBERT)

- `samples/NivaraInference` + `samples/Nivara.Samples` only. **No `src/Nivara`, `Nivara.Extensions`,
  `src/Nivara.Gpu` changes.**
- `--gpu` is F32-only; `--gpu --precision bf16|fp16` → existing global reject (Program.cs line ~174)
  already covers `minilm` — no new reject code; update the help text "GPU (distilbert /
  distilbert_sst only, this phase)" to include minilm.
- GPU surface: `minilm --gpu` (embedding inference), `minilm --gpu benchmark`, `minilm --gpu compare`
  (3-way gate vs CPU + PyTorch fixture when present). `similarity` mode stays CPU-only on GPU (clear
  error, matching distilbert's unsupported-mode pattern).
- Pooling replication: host-side after hidden readback — take row 0 of the `[1·128, 384]` hidden
  (CLS token), L2-normalize. This is exactly `MiniLMDistilled.ForwardWithMask` =
  `ExtractRow(hidden, 0, HiddenSize)` + `L2Normalize`, which `Python/minilm_compare.py` confirms is
  CLS + L2 norm. Runner stays GPU-pure (returns hidden only; logits null for MiniLM).
- PyTorch fixture `compare_minilm_embeddings_py.bin` is **optional**: absent locally (graceful skip,
  same as DistilBERT's missing `last_hidden_state_py.bin`). `Python/minilm_compare.py` can
  regenerate it (5 sentences, 5×384 f32 — same sentence list as the CPU compare).
- Assumed from discussion (recommended defaults): **correctness-first** — no fusion/lazy-stream in
  this pass; M2 is its own plan (issue to be created). No MiniLM training, no bf16/fp16.

## Findings during implementation

- MiniLM (BERT-style keys) differs from DistilBERT in one unexpected way: **token-type embeddings
  are used**. The CPU reference (`MiniLMDistilled` → `BertEncoder includeTokenTypeEmbedding: true`
  default) and PyTorch both add `token_type_embeddings[0]` (all-zero segment ids) at every position;
  DistilBERT (`includeTokenTypeEmbedding: false` via `DistilBertLoader`) has no token-type at all.
  The runner now uploads the token-type table when the key is present and broadcasts row 0 via the
  existing `addBias` kernel (key-presence driven, so DistilBERT is untouched). The gate's failure
  signature (cosine ~0.95 but ~99% tolerance violations) diagnosed it — see commit 3d25e6b.

## Proposed changes

### 1. Generalize `samples/Nivara.Samples/Gpu/DistilBertGpuRunner.cs` → `BertEncoderGpuRunner`

Rename file/class/result record; make the runner neutral:

- Ctor takes `BertConfig` (distilbert/sst call sites pass `config.ToBertConfig()`) + a
  `BertGpuNaming naming` enum (`DistilBert` | `Bert`) for key resolution. Dims already read from
  config: `hiddenDim = config.HiddenSize`, `intermediateDim = config.IntermediateSize`,
  `numHeads = config.NumAttentionHeads`, `numLayers = config.NumHiddenLayers`, `eps = config.LayerNormEps`.
- Add a key-resolver (small static helper, e.g. `BertGpuKeys.Resolve(naming, i)` or a
  `BertGpuNaming`-switched prefix builder) mapping the 3 embed keys + 12 per-layer keys + optional
  head keys to the two naming styles:

  | role | DistilBert | Bert (MiniLM) |
  |---|---|---|
  | embed prefix | `distilbert.embeddings` | `embeddings` |
  | layer prefix | `distilbert.transformer.layer.{i}` | `encoder.layer.{i}` |
  | q/k/v | `attention.{q,k,v}_lin` | `attention.self.{query,key,value}` |
  | out | `attention.out_lin` | `attention.output.dense` |
  | ffn1/ffn2 | `ffn.lin1` / `ffn.lin2` | `intermediate.dense` / `output.dense` |
  | ln1/ln2 | `sa_layer_norm` / `output_layer_norm` | `attention.output.LayerNorm` / `output.LayerNorm` |
  | head (optional) | `pre_classifier.weight/.bias`, `classifier.weight/.bias` | n/a |

  `hasHead` stays key-presence driven (false for MiniLM). Same `Forward` signature and result record
  shape (`Hidden`, `Logits?`). All kernels, workspace, launch helpers untouched.

### 2. `samples/NivaraInference/Program.cs`: `minilm --gpu`

- `case "minilm"` gains a `useGpu` branch mirroring distilbert: benchmark → `BenchmarkMiniLmGpu`,
  compare → `RunMiniLmGpuCompare`, similarity → unsupported-mode error, else `RunMiniLmGpuInference`.
- `RunMiniLmGpuInference`: config from `samples/data/minilm/config.json`, `BertEncoderGpuRunner` build,
  single sentence → `Forward(intIds, attnMask, 1, 128)` → host-side CLS+L2-norm embedding → print
  stats + L2 norm (~1.0), mirroring `RunMiniLMInference`.
- `BenchmarkMiniLmGpu`: `ReportGpuTiming(() => runner.Forward(...))`, mirrors `BenchmarkDistilBertGpu`.
- `RunMiniLmGpuCompare`: over the same 5 sentences as `Python/minilm_compare.py` / CPU compare:
  - CPU reference: `MiniLMDistilled<float>.LoadWeights(tensors, config)`; encoder hidden via
    `model.encoder.ForwardWithMask(input, mask)` (`[128,384]`), sentence embeddings via
    `model.ForwardWithMask(input, mask)` (5×384, normalized).
  - GPU: one runner, one `Forward` per sentence → hidden `[128,384]` → host-side embedding.
  - Gates: (1) hidden parity GPU vs CPU encoder hidden (maxRel ≤ 1e-3); (2) embedding parity
    GPU vs CPU embedding per sentence (maxAbs/maxRel **plus cosine similarity**, ≥ 0.9999 target);
    (3) GPU vs `compare_minilm_embeddings_py.bin` (row `s·384..(s+1)·384`) when present —
    graceful skip otherwise. Final `GATE: GateVerdict(...)`.
- Update usage/help text (mode list + GPU note) to include minilm.

### 3. Docs & README

- `samples/NivaraInference/README.md`: `--gpu` quick start lines for `minilm`, GPU benchmark table
  row (MiniLM), F32-only note already global.
- `docs/DISTILBERT-GPU.md` (reflection): short addendum noting MiniLM landed as the second
  config-driven model (~1/5 the effort) and confirming the generalization shape; point at M2 for the
  launch-overhead story. No repetition of NivaraInference README content.
- `docs/ROADMAP-SUGGESTION.md`: mark MiniLM option done (option A = config-generalize).

## Verification

- Quick `dotnet build` per commit (sample project build) before each commit.
- Correctness (battery OK): `NivaraInference minilm --gpu` (embedding + L2 norm ≈ 1);
  `NivaraInference minilm --gpu compare` → `GATE: PASS` (all three gates, fixture skipped);
  f32-only reject: `minilm --gpu --precision bf16` → the existing "--gpu is F32-only" error;
  CPU regression: `minilm` (f32) unchanged; DistilBERT GPU regression after runner rename:
  `distilbert --gpu` and `distilbert_sst --gpu` still work.
- Perf (AC only, ask human first): `minilm --gpu benchmark` → record vs CPU 64 ms table row;
  update README numbers.

## Planned commits

1. `docs: plan MiniLM GPU support in TODO.md` (this file)
2. `samples: generalize DistilBertGpuRunner to config-driven BertEncoderGpuRunner` (rename, BertConfig
   ctor + BertGpuNaming key resolver; update 6 distilbert/distilbert_sst call sites)
3. `samples: add minilm --gpu inference/benchmark/compare with CLS+L2-norm pooling and gates`
4. `docs: README + reflection GPU rows for minilm` (and roadmap checkbox)
5. `docs: remove TODO.md — plan executed` (after G2)

## Blast radius

- **`samples/Nivara.Samples/Gpu/DistilBertGpuRunner.cs`** (renamed → `BertEncoderGpuRunner.cs`):
  the only class touched by the generalization. Downstream callers: 6 sites in
  `samples/NivaraInference/Program.cs` (distilbert ×3, distilbert_sst ×3) — all updated in commit 2.
  No other references (checked: runner is not used by NivaraFineTuning or any tests).
- **`samples/NivaraInference/Program.cs`**: minilm case + 3 new functions + help text. No shared
  helper changes (`ParityStats`, `GateVerdict`, `ReportGpuTiming` reused as-is).
- **Kernels** (`Gpu/AttentionKernels.cs`, `ElementwiseKernels.cs`, `GemmKernels.cs`): untouched.
- **Core, Extensions, Nivara.Gpu**: untouched. **Tests**: untouched (sample-scoped feature).
- Docs: README + reflection addendum only.

## GitHub issues log

- [ ] #437 — M2 GPU kernel fusion / lazy-stream to cut per-dispatch launch overhead (deferred from
  the DistilBERT + MiniLM milestones; lifts both models). Created during G1 grounding.

## Reminder

As each task executes, if you find deferred work or a concern, create a GitHub issue immediately
(`gh issue create --repo khurram-uworx/Nivara`) and record its number here — don't rely on memory
or wait until the end of the plan.