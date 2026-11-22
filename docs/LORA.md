# LoRA fine-tuning for small language models (SLM)

This document describes the LoRA SLM fine-tuning showcase: a generic, Llama-family
causal-LM LoRA path that runs on CPU (reference) and Silk/OpenCL (iGPU, in progress),
with a behavioral/domain-adaptation scenario (worklog -> structured `LogTime`).

## Status (D2a)

- **CPU reference path is implemented and tested** (unit tests + trainer tests).
- **Silk/iGPU path exists** (`SilkLlamaLoRATrainer`) and has synthetic + small real smoke
  evidence, but **is not release-parity-certified**. `IsParityReady` remains `false`.
- No false iGPU claim: docs and CLI report parity status honestly as "not
  release-parity-certified" until broader real-checkpoint evidence justifies a flip.
- See `docs/plan/LORA.md` for the full handoff, decisions, and evidence.

## Headline model

- **Qwen2.5-0.5B-Instruct** — primary target. Real-model Silk smoke recorded on Intel
  iGPU: initial val ~14.25, best val ~12.35, adapter saved, ~10.2s (see plan for command).
- **SmolLM-135M-Instruct** — verifies genericity over Llama-family causal LMs (smaller,
  checkpoint-free synthetic weights available for parity/smoke).

## CPU vs Silk parity

- Synthetic-weight parity tests exist in `tests/Nivara.Tests/Gpu/SilkLlamaLoRATrainerParityTests.cs`
  (`[Category("Gpu")]`, skipped when no OpenCL):
  - Qwen-shaped forward parity (LoRA Q/V).
  - Prompt-masked cross-entropy + LoRA grad parity.
  - One-step Adam/TrainStep parity.
  - Adapter bundle round-trip (Silk) and load into CPU model preserving weights.
  - SmolLM-shaped forward + prompt-masked CE + grad parity (all targets).
- Tolerances: forward/loss 1.5e-3/2e-3, grad/weight rel 3e-2/5e-2 on local Intel iGPU.
- These tests are evidence, not release certification. D2a remains not release-ready.

## Adapter formats

Two distinct formats, do not conflate:

- **`nivara-lora-adapter-v1`** — single adapter against one frozen base
  (`Nivara.AutoDiff.Serialization.LoraAdapterSerializer`). Adapter-only A/B tensors + metadata.
- **`nivara-lora-sft-adapters-v1`** — multi-adapter bundle, one entry per (layer, target)
  (`Nivara.Silk.LoraSftAdapterBundle`), shared by CPU and Silk trainers. CPU bundle leaves
  layer/target as defaults; Silk populates them.

## Adapter consumption (NivaraChat)

- CPU path: `--adapter <path>` / `--merge-adapter <path>` load a bundle into Qwen/SmolLM
  before prompting. Rank, alpha, targets read from bundle header.
- Silk inference does not yet consume bundles; passing `--adapter`/`--merge-adapter` with
  `--silk` errors clearly (no silent fallback) — tracked separately.

## Key commands

Train (CPU):

    dotnet run -c Release --project samples/NivaraFineTuning -- --mode ft-lora \
      --model-dir samples/data/qwen2.5-0.5b-instruct \
      --train-data samples/data/worklog/train.jsonl \
      --val-data samples/data/worklog/test.jsonl \
      --template qwen --epochs 1 --max-examples 32 --max-len 256 \
      --rank 8 --alpha 16 --targets q,v --seed 42 --out-dir out/worklog-lora

Train (Silk/iGPU):

    dotnet run -c Release --project samples/NivaraFineTuning -- --mode ft-lora --silk \
      --model-dir samples/data/qwen2.5-0.5b-instruct \
      --train-data samples/data/worklog/train.jsonl \
      --val-data samples/data/worklog/test.jsonl \
      --template qwen --epochs 1 --max-examples 4 --max-len 128 \
      --rank 4 --alpha 8 --lr 2e-4 --out-dir out/worklog-lora-silk --log-every 1 --seed 42

Evaluate:

    dotnet run -c Release --project samples/NivaraFineTuning -- --mode ft-eval \
      --model-dir samples/data/qwen2.5-0.5b-instruct \
      --test-data samples/data/worklog/test.jsonl \
      --adapter out/worklog-lora/lora_adapters.json --template qwen

Generic (SmolLM) verification:

    dotnet run -c Release --project samples/NivaraFineTuning -- --mode ft-lora \
      --model-dir samples/data/smollm-135m-instruct \
      --train-data samples/data/worklog/train.jsonl \
      --val-data samples/data/worklog/test.jsonl \
      --template smollm --epochs 1 --max-examples 8 --max-len 128 \
      --rank 4 --alpha 8 --targets q,v --seed 42 --out-dir out/worklog-lora-smollm

## Limitations

- Silk path is not release-parity-certified; `IsParityReady` remains `false`.
- Adapter format bridge between core single-adapter and SFT bundle is not provided by design.
- Silk inference does not consume adapter bundles yet.
- Memory budget in reports is advisory only.