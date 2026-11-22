# LoRA fine-tuning for small language models (SLMs)

> **Role:** Stable public guide for Nivara's LoRA SLM fine-tuning showcase. This is the single LoRA doc; implementation history lives in git history and linked issues/PRs.

This document describes the LoRA SLM fine-tuning showcase: a generic, Llama-family
causal-LM LoRA path that runs on CPU (reference) and Silk/OpenCL (iGPU), with a
behavioral/domain-adaptation scenario (worklog -> structured `LogTime`).

## Status

- **CPU reference path:** implemented and tested (unit tests + trainer tests).
- **Silk/iGPU path:** implemented (`SilkLlamaLoRATrainer`) for training and evaluation.
  CPU/Silk parity is confirmed by controlled synthetic GPU-gated tests (forward,
  prompt-masked CE loss, LoRA grads, 1-step Adam/TrainStep, adapter bundle round-trip/load).
  Real-checkpoint forward+loss smoke tests are scaffolded as `[Explicit, Category("Gpu")]`.
- **Silk `ft-eval`:** `--mode ft-eval --silk` evaluates Qwen/SmolLM causal-LM LoRA adapters on
  Silk/OpenCL with the same base-vs-adapted report shape as CPU. Missing OpenCL/device errors
  clearly; no CPU fallback.
- **Silk `ft-lora` generation eval:** Silk ft-lora reports loss/val-loss and includes base-vs-adapted
  generation eval parity with the CPU path.
- **Honest claim boundary:** do not claim release-certified real-checkpoint parity beyond the
  evidence recorded here and in tests. Smoke evidence is device-specific, not release certification.

## End-to-end runbook

### Prerequisites

- .NET 11 SDK (repo-pinned; see `Directory.Build.props` / csproj files).
- A Llama-family causal-LM checkpoint in `samples/data/...` (Qwen2.5-0.5B-Instruct headline;
  SmolLM-135M-Instruct for genericity verification).
- Worklog seed data: `samples/data/worklog/train.jsonl`, `samples/data/worklog/test.jsonl`.
  Generate deterministic data if needed:

```bat
dotnet run -c Release --project samples/NivaraFineTuning -- --mode ft-generate ^
  --out-dir samples/data/worklog --seed 42 --max-examples 450 --force
```

### Train (CPU reference)

```bat
dotnet run -c Release --project samples/NivaraFineTuning -- --mode ft-lora ^
  --model-dir samples/data/qwen2.5-0.5b-instruct ^
  --train-data samples/data/worklog/train.jsonl ^
  --val-data samples/data/worklog/test.jsonl ^
  --template qwen --epochs 1 --max-examples 32 --max-len 256 ^
  --rank 8 --alpha 16 --targets q,v --seed 42 --out-dir out/worklog-lora
```

### Train (Silk/iGPU)

```bat
dotnet run -c Release --project samples/NivaraFineTuning -- --mode ft-lora --silk ^
  --model-dir samples/data/qwen2.5-0.5b-instruct ^
  --train-data samples/data/worklog/train.jsonl ^
  --val-data samples/data/worklog/test.jsonl ^
  --template qwen --epochs 1 --max-examples 4 --max-len 128 ^
  --rank 4 --alpha 8 --lr 2e-4 --out-dir out/worklog-lora-silk --log-every 1 --seed 42
```

Silk requires an available OpenCL device. The command fails clearly if OpenCL is unavailable; it
does not fall back to CPU.

### Evaluate (CPU or Silk)

```bat
dotnet run -c Release --project samples/NivaraFineTuning -- --mode ft-eval ^
  --model-dir samples/data/qwen2.5-0.5b-instruct ^
  --test-data samples/data/worklog/test.jsonl ^
  --adapter out/worklog-lora/lora_adapters.json --template qwen
```

For Silk evaluation, add `--silk` (no CPU fallback):

```bat
dotnet run -c Release --project samples/NivaraFineTuning -- --mode ft-eval --silk ^
  --model-dir samples/data/qwen2.5-0.5b-instruct ^
  --test-data samples/data/worklog/test.jsonl ^
  --adapter out/worklog-lora/lora-adapter.json --template qwen
```

### Generic SmolLM verification

```bat
dotnet run -c Release --project samples/NivaraFineTuning -- --mode ft-lora ^
  --model-dir samples/data/smollm-135m-instruct ^
  --train-data samples/data/worklog/train.jsonl ^
  --val-data samples/data/worklog/test.jsonl ^
  --template smollm --epochs 1 --max-examples 8 --max-len 128 ^
  --rank 4 --alpha 8 --targets q,v --seed 42 --out-dir out/worklog-lora-smollm
```

### NivaraChat adapter consumption

- CPU path: `--adapter <path>` / `--merge-adapter <path>` load a bundle into Qwen/SmolLM before
  prompting. Rank, alpha, and targets are read from the bundle header.
- Silk inference supports `--adapter <path>` for Qwen/SmolLM bundle consumption. `--merge-adapter`
  with `--silk` errors clearly (no silent fallback).

### Expected artifacts

- Adapter bundle: `lora_adapters.json` / `lora-adapter.json` (`nivara-lora-sft-adapters-v1`).
- Run reports: `ft-lora-report.json` / `ft-eval-report.json` plus Markdown companions, with
  device, params, losses/eval metrics, elapsed, adapter path + SHA256, and parity status.

## Design decisions and scope

- **D1 - FT scenario:** Worklog / company-assistant behavioral SFT (Nivara-native deterministic
  synthetic data). Natural-language worklog -> structured `LogTime` (`ticket`, `hours`, `date`).
  Conventions: standard office day = 8h; "all day"/"full day" = 8h (not 24h); "half day" = 4h;
  explicit durations used as stated; multi-ticket -> separate tool calls/records; ambiguous/non-worklog
  -> clarification. Eval reports before/after base vs LoRA-adapted metrics: valid structure, ticket,
  hours, date, multi-ticket, clarification/abstention, convention accuracy.
- **D2a - Silk/CPU parity, no compromise:** Real Silk causal-LM LoRA training for Qwen2.5-0.5B on
  iGPU with CPU/Silk parity required for release readiness. CPU-first or Silk-first execution is
  allowed; end state requires parity. No false iGPU claim.
- **D3 - Layering:** `LoRALinear`/adapter serializer stay in core; SFT dataset/training CLI in
  `samples/NivaraFineTuning`; shared helpers only when clearly reusable in `samples/Nivara.Samples`.
- **D4 - Data provenance:** Deterministic synthetic generator, no LLM required, seeded, provenance
  metadata/hash, refuse overwrite without `--force`. Maintainer-owned seed data lives under
  `samples/data/worklog/`.
- **D5 - Model scope:** Generic over Llama-family causal LMs (`LlamaForCausalLM<TModel,TWeight>`).
  Headline: Qwen2.5-0.5B-Instruct. SmolLM verifies genericity.

## Adapter formats

Two distinct formats, by design - do not conflate:

- **`nivara-lora-adapter-v1`** - single adapter against one frozen base
  (`Nivara.AutoDiff.Serialization.LoraAdapterSerializer`). Adapter-only A/B tensors + metadata.
- **`nivara-lora-sft-adapters-v1`** - multi-adapter bundle, one entry per (layer, target)
  (`Nivara.Silk.LoraSftAdapterBundle`), shared by CPU and Silk trainers. CPU bundle leaves
  layer/target as defaults; Silk populates them. Loaders validate rank/alpha/count/shapes.

No format bridge is provided; this is intentional.

## Parity, evidence, and guardrails

### Parity matrix

| Case | What | Evidence |
|---|---|---|
| A | Forward logits / last-token argmax | Synthetic GPU-gated parity; real-checkpoint forward+loss smoke scaffolded `[Explicit, Category("Gpu")]` |
| B | Prompt-masked CE loss (`ignoreIndex=-100`) | Synthetic GPU-gated parity; real-checkpoint forward+loss smoke scaffolded `[Explicit, Category("Gpu")]` |
| C | LoRA grad parity per target | Synthetic GPU-gated parity |
| D | 1-step TrainStep/Adam weight parity | Synthetic GPU-gated parity |
| E | Adapter bundle round-trip + CPU<->Silk interop | Synthetic GPU-gated parity + bundle contract |

Tolerances (local Intel iGPU baseline): forward/loss 1.5e-3/2e-3, grad/weight rel 3e-2/5e-2
(SmolLM 5e-2 where noted). Any widening requires documented rationale in evidence.

Real-checkpoint CPU AutoDiff backward/TrainStep on 0.5B-class models is too slow for routine smoke;
real-checkpoint smoke confirms forward+loss parity on real weights, while grad/step/bundle parity
remains covered by synthetic GPU-gated tests.

### Guardrails

- **No false iGPU claim:** docs/CLI/reporting must keep parity/certification status honest.
- **No silent CPU fallback:** `--silk` paths must run on Silk or fail clearly.
- **Memory report contract:** `--max-memory` and report fields `memoryBudgetBytes` /
  `memoryBudgetMode` record a requested advisory budget only (not measured runtime footprint, not a
  hard runtime cap).
- **Data overwrite:** deterministic generator refuses overwrite without `--force`.

## Engineering notes

- **CPU 0.5B smoke:** first backward on 0.5B-class CPU can be slow; use small `--max-examples` /
  `--max-len` and `--log-every 1` for smoke runs.
- **Device-specific smoke:** Silk smoke numbers (e.g. val loss, elapsed) are evidence for the
  recorded device/config, not portable release certification.
- **Shared SFT bundle:** CPU and Silk trainers share `nivara-lora-sft-adapters-v1`
  (`LoraSftAdapterBundle`) so adapters can round-trip between backends.
- **Reports:** ft-lora/ft-eval emit JSON + Markdown reports with device, params, loss/eval metrics,
  adapter path + SHA256, and parity status. Memory budget is advisory only.

## Key files

- `docs/LORA.md` - this stable public guide.
- `samples/data/worklog/` - deterministic seed data (`train.jsonl`, `test.jsonl`, `README.md`).
- `samples/NivaraFineTuning/WorklogSftDataset.cs` - dataset/generator/tokenization.
- `samples/NivaraFineTuning/WorklogSftEvaluator.cs` - eval metrics/parser.
- `samples/NivaraFineTuning/WorklogLoraSftTrainer.cs` - CPU reference trainer.
- `samples/NivaraFineTuning/Program.cs` - `ft-generate`/`ft-lora`/`ft-eval` CLI.
- `samples/NivaraFineTuning/LORA-TUNING.md` - sample-level LoRA SFT guide.
- `src/Nivara.Extensions/Silk/SilkLlamaLoRATrainer.cs` - Silk D2a trainer.
- `src/Nivara.Extensions/Silk/LoraSftAdapterBundle.cs` - shared SFT adapter bundle format.
- `tests/Nivara.Tests/Gpu/SilkLlamaLoRATrainerParityTests.cs` - GPU-gated parity tests.
- `samples/Nivara.Samples/LlamaForCausalLM.cs` - shared Llama-family model/loading.
