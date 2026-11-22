# LoRA SLM iGPU Fine-Tuning Showcase

Status: **active implementation on `khurram/lora`** (D2a in progress; no release-parity claim yet). This document is the handoff for the LoRA plan/PR work and replaces `docs/TODO.md` for this branch.

## 1. Goal

Showcase Nivara's ability to fine-tune small language models (SLMs) on iGPUs with Silk as the headline path, while keeping CPU as the reference path and preserving NivaraChat RAG as a first-class showcase. RAG and FT are both strong capabilities; this plan does not frame them as mutually exclusive.

The FT showcase is **behavioral/domain adaptation**, not doc-QA memorization.

## 2. Decisions (G1 cleared)

- **D1 — FT scenario:** Worklog / company-assistant behavioral SFT (Nivara-native deterministic synthetic data). Natural-language worklog -> structured `LogTime` (`ticket`, `hours`, `date`). Conventions: standard office day = 8h; "all day"/"full day" = 8h (not 24h); "half day" = 4h; explicit durations used as stated (deterministic decimal rule); multi-ticket -> separate tool calls/records; ambiguous/non-worklog -> clarification. Eval: valid structure, ticket, hours, date, multi-ticket, clarification/abstention, convention accuracy; report before/after base vs LoRA-adapted.
- **D2a — Silk/CPU parity, no compromise:** Real Silk causal-LM LoRA training for Qwen2.5-0.5B on iGPU with CPU/Silk parity required for release readiness. CPU-first or Silk-first execution is allowed; end state requires parity. No false iGPU claim.
- **D3 — Layering:** `LoRALinear`/adapter serializer stay in core; SFT dataset/training CLI in `samples/NivaraFineTuning`; shared helpers only when clearly reusable in `samples/Nivara.Samples`.
- **D4 — Data provenance:** Deterministic synthetic generator, no LLM required, seeded, provenance metadata/hash, refuse overwrite without `--force`. Maintainer-owned seed data copied into repo under `samples/data/worklog/`.
- **D5 — Model scope:** Generic over Llama-family causal LMs (`LlamaForCausalLM<TModel,TWeight>`). Headline: Qwen2.5-0.5B-Instruct. SmolLM verifies genericity.

## 3. What exists today (baseline)

- Core LoRA: `LoRALinear<T>` / `LoRALinear<TModel,TWeight>`, merge, frozen base, Kaiming A / zero B, `LoraAdapterSerializer`.
- Llama-family: `LlamaForCausalLM<TModel,TWeight>` with `AttachLora(...)`, merge helpers, Qwen/SmolLM loading paths and chat templates in `samples/Nivara.Samples`.
- AutoDiff/training: `GradientUtils.Grad()`, `AdamW`, `CrossEntropyLoss(ignoreIndex)`, module freeze/param controls, `ModelSerializer`/state dict.
- Silk: inference runners and trainers exist for other model families; `SilkLlamaLoRATrainer` now exists for D2a causal-LM LoRA training path (WIP, not release-certified).
- NivaraFineTuning sample: DistilBERT SST-2 fine-tuning existed; now also has LoRA SFT modes.

## 4. What shipped on this branch

### Data / eval (#1416, partial)
- `samples/data/worklog/` seed data: `train.jsonl` (350), `test.jsonl` (100), `README.md`.
- `samples/NivaraFineTuning/WorklogSftDataset.cs`: JSONL load/save, deterministic split/hash, prompt-masked tokenization for Qwen/SmolLM templates, `GenerateSynthetic(count, seed)`, overwrite guard.
- `samples/NivaraFineTuning/WorklogSftEvaluator.cs`: `LogTime` tool-call parsing, clarification detection, metrics for valid structure/ticket/hours/date/multi-ticket/clarification/convention; before/after report text.
- CLI: `--mode ft-generate --out-dir ... --seed ... --max-examples ... [--force]`.

### CPU reference (#1417, partial)
- `samples/NivaraFineTuning/WorklogLoraSftTrainer.cs`: generic Llama-family LoRA SFT, frozen base, AdamW on adapter params, prompt-masked CE `ignoreIndex:-100`, best-checkpoint restore, adapter bundle save.
- `samples/NivaraFineTuning/Program.cs`: `--mode ft-lora` CPU path; `--mode ft-eval` base + optional adapter before/after report.
- CPU smoke: first-step 0.5B CPU backward can be slow; granular per-step logging added (`forward...`, `backward (loss=...)`, `avgLoss`).

### Silk D2a (#1418, in progress)
- `src/Nivara.Extensions/Silk/SilkKernels.cs` / `SilkLauncher.cs`: `rms_norm_backward`, `rotary_backward`, `swiglu_backward`, `scale`, GQA backward kernels, `SoftmaxCrossEntropyBackward(ignoreIndex)`.
- `src/Nivara.Extensions/Silk/SilkLlamaLoRATrainer.cs`: generic Llama-family LoRA trainer (Qwen-shaped/SmolLM-shaped configs), base weight upload, LoRA A/B/grad/moment buffers, forward with LoRA q/k/v/o/gate/up/down, prompt-masked CE backward, transformer block backward, LoRA grad accumulation, Adam, `TrainStep`, adapter bundle save/load.
- Canonical adapter bundle: `src/Nivara.Extensions/Silk/LoraSftAdapterBundle.cs` (`nivara-lora-sft-adapters-v1`) shared by CPU and Silk; both producers now emit the same schema (index/layer/target/aShape/bShape/a/b) plus trainableParams and alpha; loaders validate rank/alpha/count/shapes. Silk bundle additionally carries layer/target. CPU bundle leaves layer/target as defaults (not tracked) — optional fields.
- CLI: `--mode ft-lora --silk` wired with OpenCL availability check, honest `IsParityReady=false` warning, no silent CPU fallback. Added `MaxLen <= MaxPositionEmbeddings` guard and structured JSON report (`ft-lora-report.json`) with device, params, losses, elapsed, adapter path + SHA256, parity status.
- `ft-eval` now emits `ft-eval-report.json` (base + adapted metrics, adapter hash) and works against Silk-produced bundles.
- Tests: `tests/Nivara.Tests/Gpu/SilkLlamaLoRATrainerParityTests.cs` (`[Category("Gpu")]`) — 6 tests pass on local OpenCL:
  - Qwen-shaped Q/V forward parity;
  - Qwen-shaped prompt-masked CE + LoRA grad parity;
  - Qwen-shaped 1-step Adam/TrainStep parity;
  - adapter bundle round-trip (Silk);
  - adapter bundle loads into CPU model preserving weights (new);
  - SmolLM-shaped Q/K/V/O/FFN forward + prompt-masked CE + grad parity.
- Tolerances: forward/loss 1.5e-3/2e-3, grad/weight rel 3e-2/5e-2, passing on local Intel iGPU.
- Real-model Silk smoke (evidence, not release certification): Qwen2.5-0.5B-Instruct BF16, `Intel(R) Graphics`, `--max-examples 4 --max-len 128 --rank 4 --alpha 8 --epochs 1 --log-every 1`, initial val `14.2516`, best val `12.3500`, adapter saved, `10.2s`.
- `IsParityReady` remains `false`: real-checkpoint parity/eval evidence and broader case matrix still required before flipping.

## 5. Evidence and commands

### Targeted checks run

```bat
dotnet build src/Nivara.Extensions/Nivara.Extensions.csproj -c Release -nologo -v q
dotnet build samples/NivaraFineTuning/NivaraFineTuning.csproj -c Release -nologo -v q
dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkLlamaLoRATrainerParityTests" --nologo -v q
```

Result: builds succeeded; GPU parity 5/5 passed on local OpenCL device.

### Silk smoke (human-runnable / small)

```bat
dotnet run --project samples/NivaraFineTuning/NivaraFineTuning.csproj -c Release -- --mode ft-lora --silk --model-dir samples/data/qwen2.5-0.5b-instruct --train-data samples/data/worklog/train.jsonl --val-data samples/data/worklog/test.jsonl --template qwen --epochs 1 --max-examples 4 --max-len 128 --rank 4 --alpha 8 --lr 2e-4 --out-dir C:\Users\khurram\AppData\Local\Temp\opencode\nivara-ft-silk-smoke --log-every 1 --seed 42
```

Observed (Intel iGPU, smoke only): initial val `14.2516`, best val `12.3500`, adapter saved, `10.2s`.

### CPU reference command

```bat
dotnet run --project samples/NivaraFineTuning/NivaraFineTuning.csproj -c Release -- --mode ft-lora --model-dir samples/data/qwen2.5-0.5b-instruct --train-data samples/data/worklog/train.jsonl --val-data samples/data/worklog/test.jsonl --template qwen --epochs 1 --max-examples 4 --max-len 128 --rank 4 --alpha 8 --lr 2e-4 --out-dir C:\Users\khurram\AppData\Local\Temp\opencode\nivara-ft-cpu-smoke --log-every 1 --seed 42
```

Note: CPU 0.5B first backward can be slow; use small max-examples/max-len for smoke.

## 6. Known gaps / red flags

- **D2a not release-ready:** `SilkLlamaLoRATrainer.IsParityReady == false`; parity evidence is synthetic + small real smoke, not full release certification.
- **Parity scope:** tolerances tightened but still need justification over broader cases; no large real-checkpoint parity matrix.
- **Adapter format:** Converged on shared `LoraSftAdapterBundle` (`nivara-lora-sft-adapters-v1`) between CPU and Silk; loaders validate rank/alpha/count/shapes. Core `LoraAdapterSerializer` (`nivara-lora-adapter-v1`, single-adapter) remains distinct by design; bridge not added (deliberate; multi-adapter bundle is a different contract).
- **CPU/Silk report:** Structured JSON report now emitted (`ft-lora-report.json`, `ft-eval-report.json`) with device, params, losses, elapsed, adapter path + SHA256, parity status, before/after metrics. Memory footprint estimate still approximate.
- **Tests:** generator determinism/split/provenance/evaluator unit tests still needed (#1416/#1421). Full suite not run.
- **Docs:** `docs/plan/LORA.md` is now reshaped; other docs/DocsCoveragePins updates may be needed when #1421 lands.
- **No false iGPU claim:** docs/CLI must keep D2a status honest until parity is release-certified.

## 7. File map (current)

- `docs/plan/LORA.md` — this handoff/plan (authoritative for LoRA branch).
- `samples/data/worklog/README.md`, `train.jsonl`, `test.jsonl` — deterministic seed data.
- `samples/NivaraFineTuning/WorklogSftDataset.cs` — dataset/generator/tokenization.
- `samples/NivaraFineTuning/WorklogSftEvaluator.cs` — eval metrics/parser.
- `samples/NivaraFineTuning/WorklogLoraSftTrainer.cs` — CPU reference trainer.
- `samples/NivaraFineTuning/Program.cs` — `ft-generate`/`ft-lora`/`ft-eval` CLI.
- `src/Nivara.Extensions/Silk/SilkLlamaLoRATrainer.cs` — Silk D2a trainer.
- `src/Nivara.Extensions/Silk/SilkKernels.cs`, `SilkLauncher.cs` — backward kernels/launchers.
- `tests/Nivara.Tests/Gpu/SilkLlamaLoRATrainerParityTests.cs` — GPU-gated parity tests.
- `samples/Nivara.Samples/LlamaForCausalLM.cs` — shared Llama-family model/loading; Qwen tokenizer unk fix.

## 8. Next session pick list

1. **#1418 hardening (recommended first):** run CPU smoke with new granular logs; run `ft-eval` before/after on Silk-produced adapter; collect real-checkpoint evidence; converge adapter format where needed; keep `IsParityReady=false` until evidence justifies flip.
2. **#1420 report polish:** structured JSON/markdown report (base/adapted metrics, param counts, device, elapsed, memory/parity status, adapter path); SmolLM verification command path.
3. **#1416 tests:** deterministic generator tests, split/provenance/hash stability, overwrite guard, evaluator metric unit tests.
4. **#1421 tests/docs/gates:** targeted tests, docs updates, DocsCoveragePins/citation/ADR checks if docs change; doc snippet gate when docs touched.
5. **#1419 done:** NivaraChat adapter consumption demo (`--adapter <path>` on `--qwen`/`--smollm`, CPU path) — additive, no RAG regression. Silk inference does not yet consume bundles (explicit error).
6. **G2 + PR:** branch-vs-plan review, docs gates, push/create PR with confirmation; no `docs/TODO.md` in final PR.

**#1414 done on branch:** `docs/plan/LORA.md` reshaped for D1-D5/D2a; `docs/TODO.md` removed. Keep this file as handoff.

## 9. Guardrails

- No external FineTuning-repo paths/references in code, docs, issues, or commits.
- No false "FT on iGPU" or release-parity claim until D2a evidence supports it.
- Do not run full test suite unprompted; run targeted tests autonomously. Full suite requires confirmation and `Category!=Performance&Category!=ReleaseReadiness&Category!=Gpu`.
- Do not push/create PR without explicit user confirmation.
- Keep `docs/plan/LORA.md` as the handoff; do not reintroduce `docs/TODO.md`.
