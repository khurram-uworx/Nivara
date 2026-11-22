# NivaraFineTuning - LoRA SFT for Llama-family SLMs

This guide covers the LoRA (Low-Rank Adaptation) supervised fine-tuning (SFT) track of the
sample: adapting a pretrained Llama-family causal LM (Qwen2.5-0.5B-Instruct headline,
SmolLM-135M generic verification) to a behavioral task (worklog -> structured `LogTime`)
using Nivara's AutoDiff engine. For the DistilBERT/SST-2 track see [BERT-TUNING.md](BERT-TUNING.md).
For the cross-cutting LoRA feature picture see `docs/LORA.md`.

## Concepts and vocabulary

- **Causal LM**: a model that predicts the next token given previous tokens. Used here as
  the base for behavioral adaptation (not classification).
- **LoRA (Low-Rank Adaptation)**: instead of updating all base weights, inject a small
  trainable low-rank pair (A, B) into selected projections. Base weights stay frozen;
  only A and B train. `output = base(x) + (alpha/rank) * B(A(x))`.
- **Rank (r)**: inner dimension of the low-rank pair. Smaller = fewer trainable params and
  faster; larger = more capacity. Typical 4-16.
- **Alpha**: scaling factor for the LoRA contribution. Common rule: alpha = 2 * rank.
- **Targets**: which projections get LoRA adapters (q, k, v, o, ffn). More targets =
  more capacity and params.
- **Frozen base**: base model weights are not updated (no gradients); only adapter params
  train. Keeps base knowledge and reduces memory.
- **Prompt masking**: during SFT, loss is computed only on answer tokens; prompt tokens
  are masked with ignore index -100 so the model learns to produce the answer, not repeat
  the prompt.
- **Adapter bundle** (`nivara-lora-sft-adapters-v1`): on-disk format carrying one entry per
  (layer, target) with base64 A/B matrices plus rank/alpha/targets/template/baseModelId
  metadata. Shared by CPU and Silk trainers; round-trips between backends.
- **CPU reference vs Silk/iGPU**: CPU path is the reference; Silk path targets iGPU.
  CPU/Silk parity is confirmed by controlled synthetic GPU-gated tests; real-checkpoint
  forward+loss smoke tests are scaffolded as `[Explicit, Category("Gpu")]`.
- **Parity**: numerical agreement between CPU and Silk forward/grad/step within documented
  tolerances (forward/loss 1.5e-3/2e-3, grad/weight rel 3e-2/5e-2).

## LoRA SFT (generic Llama-family, CPU reference)

The sample also ships a generic LoRA SFT trainer for Llama-family causal LMs (Qwen2.5-0.5B headline, SmolLM generic verification).

### CLI

```bash
dotnet run -c Release --project samples/NivaraFineTuning -- --mode ft-lora \
  --model-dir samples/data/qwen2.5-0.5b-instruct \
  --train-data samples/data/worklog/train.jsonl \
  --val-data samples/data/worklog/test.jsonl \
  --template qwen --epochs 1 --max-examples 32 --max-len 256 \
  --rank 8 --alpha 16 --targets q,v --seed 42 --out-dir out/worklog-lora
```

Key options: `--template qwen|smollm`, `--rank`, `--alpha`, `--targets q,k,v,o,ffn`, `--patience` (early stopping, 0=disabled), `--min-delta`, `--grad-accum`, `--max-examples`, `--max-len`, `--seed`, `--log-every`, `--max-memory` (advisory memory budget in bytes, reported in the artifact; not a hard cap), `--eval`/`--no-eval` (run base vs adapted eval during ft-lora; default on).

### SmolLM verification path

```bash
dotnet run -c Release --project samples/NivaraFineTuning -- --mode ft-lora \
  --model-dir samples/data/smollm-135m-instruct \
  --train-data samples/data/worklog/train.jsonl \
  --val-data samples/data/worklog/test.jsonl \
  --template smollm --epochs 1 --max-examples 8 --max-len 128 \
  --rank 4 --alpha 8 --targets q,v --seed 42 --out-dir out/worklog-lora-smollm
```

Follow with eval:

```bash
dotnet run -c Release --project samples/NivaraFineTuning -- --mode ft-eval \
  --model-dir samples/data/smollm-135m-instruct \
  --test-data samples/data/worklog/test.jsonl \
  --adapter out/worklog-lora-smollm/lora-adapter.json --template smollm
```

### Report artifacts

Each ft-lora/ft-eval run writes ft-lora-report.json/ft-eval-report.json plus a Markdown companion report (ft-lora-report.md/ft-eval-report.md) with device, params, memory budget, loss curve, eval before/after, adapter hash, and honest parity status. Memory budget is advisory only.

### Evaluation

```bash
dotnet run -c Release --project samples/NivaraFineTuning -- --mode ft-eval \
  --model-dir samples/data/qwen2.5-0.5b-instruct \
  --test-data samples/data/worklog/test.jsonl \
  --adapter out/worklog-lora/lora_adapters.json --template qwen
```

For Silk/OpenCL evaluation, add `--silk` (Qwen/SmolLM causal-LM adapters, no CPU fallback):

```bash
dotnet run -c Release --project samples/NivaraFineTuning -- --mode ft-eval --silk \
  --model-dir samples/data/qwen2.5-0.5b-instruct \
  --test-data samples/data/worklog/test.jsonl \
  --adapter out/worklog-lora/lora-adapter.json --template qwen
```

Reports base vs LoRA-adapted metrics: valid structure, ticket, hours, date, multi-ticket, clarification, convention accuracy.

### Adapter format

Saved as `lora_adapters.json` / `lora-adapter.json` with format `nivara-lora-sft-adapters-v1` (base64-encoded A/B matrices plus rank/alpha/targets/template/baseModelId metadata). Loaded with shape/rank/count validation. Both the CPU and Silk (`--silk`) paths now share one bundle schema (`LoraSftAdapterBundle`) and each `ft-lora`/`ft-eval` run writes a structured JSON report (`ft-lora-report.json` / `ft-eval-report.json`) with device, params, losses, elapsed, adapter path + SHA256, and D2a parity status.

### Defaults / memory guard

Per-example steps, small defaults (`maxLen=256`). `--max-examples` caps data; `--max-len` capped at 2048 and validated against model `MaxPositionEmbeddings`. Gradient accumulation via `--grad-accum`. Early stopping via `--patience`/`--min-delta`.
