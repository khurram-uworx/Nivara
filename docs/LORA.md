# LoRA fine-tuning for small language models (SLM)

> **Role:** Stable user-facing guide for the LoRA SLM fine-tuning showcase. For implementation handoff, active gaps, and issue tracking, see `docs/plan/LORA.md`.

This document describes the LoRA SLM fine-tuning showcase: a generic, Llama-family
causal-LM LoRA path that runs on CPU (reference) and Silk/OpenCL (iGPU, in progress),
with a behavioral/domain-adaptation scenario (worklog -> structured `LogTime`).

## Status (D2a)

- **CPU reference path is implemented and tested** (unit tests + trainer tests).
- **Silk/iGPU path exists** (`SilkLlamaLoRATrainer`) and CPU/Silk parity is confirmed by
  controlled synthetic GPU-gated tests (forward, prompt-masked CE loss, LoRA grads, 1-step
  Adam/TrainStep, adapter bundle round-trip/load). Real-checkpoint forward+loss smoke tests
  are scaffolded as `[Explicit, Category("Gpu")]`.
- See `docs/plan/LORA.md` for the handoff, decisions, evidence, and matrix.

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
- Real-checkpoint forward+loss smoke tests are scaffolded as `[Explicit, Category("Gpu")]` and skip cleanly when OpenCL or model dirs are missing.

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
- Silk inference supports `--adapter <path>` for Qwen/SmolLM bundle consumption; `--merge-adapter`
  with `--silk` errors clearly (no silent fallback).

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

## Limitations and open gaps

- **D2a parity:** CPU/Silk LoRA training parity confirmed by controlled synthetic GPU-gated tests; real-checkpoint forward+loss smoke tests are scaffolded as `[Explicit, Category("Gpu")]`. (#1438)
- **Silk `ft-eval` missing:** `--mode ft-eval --silk` currently runs the CPU eval path silently; this is a known gap and must be guarded or implemented. (#1439, #1441)
- **Silk `ft-lora` generation eval missing:** Silk ft-lora reports loss/val-loss but lacks integrated base-vs-adapted generation eval parity with CPU. (#1437)
- **Adapter format bridge by design:** Core single-adapter (`nivara-lora-adapter-v1`) and SFT multi-adapter bundle (`nivara-lora-sft-adapters-v1`) are distinct formats; no bridge provided.
- **Memory report contract:** Memory budget/footprint in reports is advisory/unspecified; contract needs clarification. (#1442)
- **CPU smoke evidence:** Post-merge CPU ft-lora smoke evidence with granular logs is tracked. (#1443)
- **Docs role/handoff drift:** User guide vs handoff roles tracked for cleanup. (#1440, #1444)