# NivaraFineTuning - LoRA SFT for Llama-family SLMs

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

Reports base vs LoRA-adapted metrics: valid structure, ticket, hours, date, multi-ticket, clarification, convention accuracy.

### Adapter format

Saved as `lora_adapters.json` / `lora-adapter.json` with format `nivara-lora-sft-adapters-v1` (base64-encoded A/B matrices plus rank/alpha/targets/template/baseModelId metadata). Loaded with shape/rank/count validation. Both the CPU and Silk (`--silk`) paths now share one bundle schema (`LoraSftAdapterBundle`) and each `ft-lora`/`ft-eval` run writes a structured JSON report (`ft-lora-report.json` / `ft-eval-report.json`) with device, params, losses, elapsed, adapter path + SHA256, and honest D2a parity status. Silk path remains not release-parity-certified.

### Defaults / memory guard

Per-example steps, small defaults (`maxLen=256`). `--max-examples` caps data; `--max-len` capped at 2048 and validated against model `MaxPositionEmbeddings`. Gradient accumulation via `--grad-accum`. Early stopping via `--patience`/`--min-delta`.
