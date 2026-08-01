# Plan: DistilBERT SST-2 Inference Showcase in NivaraInference

> **Prerequisite: COMPLETE** (2026-08-01). `docs/PLAN0.md` (DistilBERT base model inference) was completed on `khurram/bert` (PR #84) and is archived/deleted. It delivered:
> - `includeTokenTypeEmbedding` toggle on `BertEncoder<T>` — DistilBERT passes `false`
> - `DistilBertConfig` + `DistilBertLoader` promoted to `Nivara.Samples` (shared, already consumed by `NivaraFineTuning`)
> - `distilbert` inference / benchmark / compare modes in NivaraInference
> - **Exact erf GELU** (`GeluExact`) for BERT-family activations — the base encoder's `last_hidden_state` now matches HuggingFace to `max abs diff 5e-6`, cosine `0.99999988` (was `0.0286` with tanh approx)
>
> `samples/NivaraInference/README.md` already documents the `distilbert` base mode.

## Purpose

Extend the `samples/NivaraInference` showcase to run an **already fine-tuned** HuggingFace model — `distilbert/distilbert-base-uncased-finetuned-sst-2-english` — for binary sentiment classification, entirely in C# with Nivara's zero-dependency tensor engine.

The DistilBERT fine-tuning sample (`samples/NivaraFineTuning`) was a large jump from the existing inference samples. Instead of fine-tuning from scratch, we first reuse an off-the-shelf fine-tuned model to:

- Validate that Nivara has all the neural network layers DistilBERT needs (expectation: yes — `Embedding`, `LayerNorm`, `Linear`, `BertSelfAttention`, GELU).
- Run an A/B comparison between PyTorch and Nivara (same pattern as the MiniLM showcase).
- Benchmark to confirm reasonable performance.
- Only once parity + performance are acceptable do we revisit fine-tuning.

## Background findings

All layers required by `distilbert-base-uncased-finetuned-sst-2-english` already exist. The model code also already exists:

- `BertConfig`, `BertEncoder<T>`, `BertSelfAttention<T>`, `BertLayer<T>`, `MiniLMTokenizer` live in `samples/Nivara.Samples/BertModel.cs` (shared).
- `DistilBertConfig` + `DistilBertLoader.LoadEncoderWeights` live in `samples/Nivara.Samples/DistilBertModel.cs` (shared since PLAN0).
- `DistilBertForSequenceClassification<T>` (with `LoadWeights`) still lives in `samples/NivaraFineTuning/DistilBertModel.cs` — **to be promoted in Task 2 below**.

Verified model facts (from the HF repo):

| Field | Value |
|-------|-------|
| Model id | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` |
| Architecture | `DistilBertForSequenceClassification` |
| Config | `dim=768`, `n_layers=6`, `n_heads=12`, `hidden_dim=3072`, `vocab_size=30522`, `max_position_embeddings=512`, `activation=gelu` |
| Labels | `0=NEGATIVE`, `1=POSITIVE` (2 classes) |
| Pad token id | 0 |
| Weight files | `model.safetensors` (268 MB), `vocab.txt`, `config.json`, `tokenizer_config.json` |
| HF tensor prefixes | `distilbert.embeddings.*`, `distilbert.transformer.layer.{0-5}.*`, `pre_classifier.*`, `classifier.*` |

### Both PLAN0 correctness gaps are now resolved

1. **Token-type embedding contamination — FIXED in PLAN0.** `BertEncoder<T>` has `includeTokenTypeEmbedding: bool = true`; DistilBERT passes `false`. `NivaraFineTuning` already benefits (frozen-encoder parity).
2. **GELU formulation divergence — RESOLVED in PLAN0.** Nivara now has `ReverseGradOperations.GeluExact` (exact erf, matching PyTorch `nn.GELU()` / HF `activation=gelu`). The base encoder matches HF to `max abs diff 5e-6`. The A/B logit comparison (Task 5) should therefore expect float32-level agreement, not a tanh-vs-erf divergence. Tanh `Gelu` is retained for GPT-family `TransformerBlock`.

## Decisions (locked)

1. **GELU:** Exact erf is implemented and validated (`GeluExact`, base encoder matches HF to `5e-6`). A/B logit comparison should target float32-level agreement; no tanh tolerance policy needed.
2. **Performance scope:** A/B + parity first, then benchmark. Fused multi-head attention optimization is a **follow-up** after parity is proven.
3. **Perf test:** Mirror `MiniLm_Inference_Latency` in `tests/Nivara.Tests/AutoDiff/PerfTests.cs` (assert-skip when weights are absent).
4. **Parity gate is already met at the encoder level:** base encoder parity (`max abs diff 5e-6`) is the prerequisite; the SST-2 A/B only adds the `pre_classifier` + `classifier` heads on top.

## Model / code promotion

PLAN0 already moved `DistilBertConfig` and the encoder loader (`DistilBertLoader` + shared `StateDictLoader.LoadEmbed` / `LoadLinear` / `LoadLayerNorm` helpers) into `samples/Nivara.Samples/DistilBertModel.cs` (namespace `Nivara.Samples`). **Remaining:** move `DistilBertForSequenceClassification<T>` (incl. its head `LoadWeights`) from `samples/NivaraFineTuning/DistilBertModel.cs` into the same shared file. Both `NivaraInference` and `NivaraFineTuning` already reference `Nivara.Samples`, so no csproj changes are required.

---

## Task Breakdown

Task list follows `docs/TASKS-TEMPLATE.md`. Each task is sized for one coding agent.

## Task 1: Download the fine-tuned SST-2 model weights

### Priority

High

### Goal

Download `distilbert/distilbert-base-uncased-finetuned-sst-2-english` into `samples/data/distilbert_sst/` so all later tasks can run.

### Why this exists

Every other task depends on having the real weights, config, and vocabulary on disk.

### Scope

- `hf download distilbert/distilbert-base-uncased-finetuned-sst-2-english config.json model.safetensors vocab.txt tokenizer_config.json --local-dir samples/data/distilbert_sst`
- Explicit file list avoids pulling the redundant `pytorch_model.bin` / `tf_model.h5` / `rust_model.ot` duplicates (~1 GB+).

### Acceptance criteria

- `samples/data/distilbert_sst/config.json` exists (verify `dim=768`, `n_layers=6`, `n_heads=12`, `num_classes`/`id2label` present).
- `samples/data/distilbert_sst/model.safetensors` exists (~268 MB).
- `samples/data/distilbert_sst/vocab.txt` exists (30522-line WordPiece vocab).

### Files likely involved

- `samples/data/distilbert_sst/*`

## Task 2: Promote DistilBertForSequenceClassification to Nivara.Samples

### Priority

High

### Goal

Move `DistilBertForSequenceClassification<T>` (the last remaining DistilBERT type) from `NivaraFineTuning` into the shared `Nivara.Samples` project so `NivaraInference` can consume it.

### Why this exists

`NivaraInference` needs the model, but the authoritative implementation currently lives only in the fine-tuning sample. Duplicating would violate the repo's consolidate-logic rule (AGENTS.md).

### Scope

- Move `DistilBertForSequenceClassification<T>` + its head `LoadWeights` (incl. the `pre_classifier` / `classifier` linear loading, which already uses `StateDictLoader.LoadLinear`) from `samples/NivaraFineTuning/DistilBertModel.cs` to `samples/Nivara.Samples/DistilBertModel.cs`, namespace `Nivara.Samples`.
- Delete the local copy from `NivaraFineTuning` (or keep a thin re-export if `NivaraFineTuning/Program.cs` references it).
- `DistilBertConfig` + `DistilBertLoader` are already shared (done in PLAN0).

### Constraints

- No behavior change to the existing fine-tune sample at this step.
- No csproj changes (both projects already reference `Nivara.Samples`).

### Acceptance criteria

- `dotnet build Nivara.slnx` succeeds.
- `samples/NivaraFineTuning` builds with the shared types (no duplicate type definition).

### Files likely involved

- `samples/Nivara.Samples/DistilBertModel.cs` (add classifier)
- `samples/NivaraFineTuning/DistilBertModel.cs` (delete / thin re-export)
- `samples/NivaraFineTuning/Program.cs` (no change expected)

## Task 3: ~~Fix BertEncoder token-type embedding for DistilBERT~~ DONE (PLAN0)

Implemented and verified in `docs/PLAN0.md` (PR #84): `BertEncoder<T>` has `includeTokenTypeEmbedding: bool = true`; DistilBERT passes `false`. No further work. The constructor change for `DistilBertForSequenceClassification<T>` (pass `false`) happens as part of Task 2's promotion.

## Task 4: Add `distilbert_sst` model + CLI modes to NivaraInference

### Priority

High

### Goal

Add a `distilbert_sst` model entry to `samples/NivaraInference` with `predict`, `compare`, and `benchmark` modes.

### Why this exists

This is the showcase deliverable: run the fine-tuned SST-2 model in pure .NET.

### Scope

- New `samples/NivaraInference/DistilBertSst.cs`:
  - Static loader: read `config.json` + `vocab.txt` + `model.safetensors`; build `DistilBertForSequenceClassification<float>` with `numClasses=2`; `LoadWeights`.
  - Predict helper: tokenize via `MiniLMTokenizer` → `Forward(inputIds, attnMask, batch=1, seqLen)` → softmax → `POSITIVE`/`NEGATIVE` + confidence %.
  - `CountParameters` helper for the benchmark header.
- `Program.cs`:
  - Add `distilbert_sst` to the usage/help text and model dispatch.
  - Modes:
    - default: single-sentence demo prediction.
    - `predict`: interactive REPL (type sentences, `quit` to exit).
    - `compare`: forward on shared sentences, save logits + softmax probs to `samples/data/compare_distilbert_sst_cs.bin`.
    - `benchmark`: 3 warmup + 10 timed passes, report avg/min/max, params, weight MB.

### Constraints

- Inference only — no `GradientUtils.Grad()` scope, no graph creation (leaf tensors).
- Reuse shared `MiniLMTokenizer`; do not add a new tokenizer.

### Acceptance criteria

- `dotnet run --project samples/NivaraInference -- distilbert_sst` prints a sentiment prediction with confidence.
- `distilbert_sst benchmark` prints per-pass times + average.
- `distilbert_sst compare` writes `compare_distilbert_sst_cs.bin`.

### Files likely involved

- `samples/NivaraInference/DistilBertSst.cs` (new)
- `samples/NivaraInference/Program.cs`

## Task 5: PyTorch A/B reference + logits comparison

### Priority

High

### Goal

Produce PyTorch reference logits for the same sentences and compare against Nivara output to validate parity.

### Why this exists

A/B parity is the gate before benchmarking and any later optimization.

### Scope

- New `samples/NivaraInference/Python/distilbert_sst_compare.py`:
  - `AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)`, `eval()`.
  - Same sentence list used by the C# `compare` mode.
  - Tokenize with HF tokenizer (`padding=True, truncation=True, max_length=128`), forward with `no_grad`.
  - Save logits + softmax probabilities to `samples/data/compare_distilbert_sst_py.bin`; print first-10 logits + sentiment per sentence.
  - Reuse `hf_loader.MODELS_DIR`.
- (Optional) `distilbert_sst_predict.py` interactive Python CLI for manual A/B.
- Comparison runbook:
  - Run C# `compare`, run Python script, diff logits (max abs diff, argmax agreement).
  - Verify tokenizer parity: token ids from `Microsoft.ML.Tokenizers.BertTokenizer` vs HF `BertTokenizer` match on the demo sentences.

### Constraints

- No new dependencies beyond existing `transformers`, `torch`, `safetensors`, `numpy` (already in `requirements.txt`).

### Acceptance criteria

- Logits match PyTorch at float32-level (expect `max abs diff` ~1e-5, matching the base-encoder compare from PLAN0); sentiment (argmax) agrees on all demo sentences.
- Comparison results (max abs diff, argmax agreement) recorded in `samples/NivaraInference/README.md`.

### Files likely involved

- `samples/NivaraInference/Python/distilbert_sst_compare.py` (new)
- `samples/NivaraInference/Python/distilbert_sst_predict.py` (new, optional)
- `samples/data/compare_distilbert_sst_py.bin`, `samples/data/compare_distilbert_sst_cs.bin`

## Task 6: DistilBERT latency perf test (mirror MiniLM)

### Priority

Medium

### Goal

Add a DistilBERT end-to-end latency test to `tests/Nivara.Tests/AutoDiff/PerfTests.cs` mirroring `MiniLm_Inference_Latency`.

### Why this exists

Regression coverage for inference latency alongside the existing MiniLM test.

### Scope

- Add `DistilBert_Inference_Latency()`:
  - `Assert.Ignore` when `samples/data/distilbert_sst/model.safetensors` (or config/vocab) is absent.
  - Build model, tokenize a sample sentence, 3 warmup + 5 timed passes outside `Grad()`.
  - Print avg/min/max and weight MB.

### Acceptance criteria

- Test compiles and passes (or skips cleanly) in `dotnet test`.
- Mirrors the structure/naming of the MiniLM latency test.

### Files likely involved

- `tests/Nivara.Tests/AutoDiff/PerfTests.cs`

## Task 7: Benchmark + record baseline numbers

### Priority

Medium

### Goal

Run the `distilbert_sst benchmark` mode and record baseline numbers for the README.

### Why this exists

Establishes the baseline to compare against future optimizations and PyTorch.

### Scope

- Run `dotnet run --project samples/NivaraInference -c Release -- distilbert_sst benchmark`.
- Run the equivalent Python timing for the A/B comparison table.
- Record results; note the known hot path (per-head `Slice`/`Transpose`/`MatMul`/`Softmax` loop in `BertSelfAttention.MultiHeadAttention`, batch=1 mask building).

### Acceptance criteria

- Baseline numbers recorded in the README benchmark table.
- Known hot paths noted for the follow-up optimization task.

### Files likely involved

- `samples/NivaraInference/README.md` (benchmark table)

## Task 8: Update documentation

### Priority

Medium

### Goal

Update READMEs to reflect the new showcase model, shared model promotion, GELU divergence policy, and A/B + benchmark results.

### Why this exists

The showcase needs runnable docs and the GELU tolerance decision must be recorded.

### Scope

- `samples/NivaraInference/README.md`:
  - Add `distilbert_sst` to the supported-models table and quick start.
  - Add download command (explicit file list).
  - Document GELU: exact erf (`GeluExact`) is used for BERT-family models (already recorded for the base `distilbert` mode in PLAN0); tanh `Gelu` is retained for GPT-style layers.
  - Add A/B results + benchmark table.
- `samples/NivaraFineTuning/README.md`:
  - Note `DistilBertForSequenceClassification<T>` + `DistilBertConfig` now live in `Nivara.Samples` (shared).
  - Note the token-type embedding fix improves frozen-encoder parity.

### Acceptance criteria

- Both READMEs are accurate and runnable.
- GELU policy and measured A/B agreement are explicitly documented.

### Files likely involved

- `samples/NivaraInference/README.md`
- `samples/NivaraFineTuning/README.md`

## Follow-up (not in this round)

- **Fused multi-head attention**: rewrite `BertSelfAttention.MultiHeadAttention` into a single batched kernel (reshape → row-wise `TensorPrimitives` softmax → attention over V → reshape), plus a batch=1 mask fast path, to close the MiniLM/DistilBERT performance gap.
- **Blocked/SIMD MatMul + vectorized exact GELU**: filed as GitHub issues #81 and #82 during PLAN0.
- **Fine-tuning**: revisit `samples/NivaraFineTuning` once inference parity + performance are acceptable.

## Coordination Notes

- **Decision gates:** none blocking at task start. Task 5 (A/B) is the parity gate: Tasks 6, 7, 8 depend on its results (logit match, benchmark baseline). The encoder-level parity gate (max abs diff 5e-6) is already met from PLAN0.
- **Parallel-safe:** Task 1 (download) is independent and prerequisite for all. Task 2 (promote classifier) touches `samples/Nivara.Samples/DistilBertModel.cs` only. Task 4 depends on Task 2. Task 5 can start as soon as 4's `compare` mode exists; its Python script can be written before 4 lands.
- **Shared-file conflicts:** none expected — Task 2 owns `DistilBertModel.cs`; Task 4 owns `Program.cs` + a new `DistilBertSst.cs`.
- **README benchmark/A/B numbers** (Task 7) must be filled in before Task 8 is finalized.

## Suggested Agent Handout Batches

### Batch A: setup + promotion (sequential)

- Task 1 (download)
- Task 2 (promote `DistilBertForSequenceClassification` to `Nivara.Samples`)

### Batch B: showcase + A/B (depends on A)

- Task 4 (NivaraInference `distilbert_sst` + CLI modes)
- Task 5 (PyTorch A/B reference + comparison)

### Batch C: verification + docs (after B)

- Task 6 (perf test)
- Task 7 (benchmark + record numbers)
- Task 8 (README updates)

## Final Checklist

- Every task has a single-agent-sized scope.
- Every task has acceptance criteria.
- Decision-gate task (Task 5 A/B) is clearly marked and gates 6/7/8.
- Likely files listed to reduce agent search time.
- Execution order reflects real dependencies (download → classifier promotion → showcase → A/B → test/benchmark/docs).
