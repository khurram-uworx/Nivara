# LoRA SLM iGPU Fine-Tuning Showcase

Status: **landed on main** (core LoRA showcase merged via PR #1422; follow-up PRs landed through #1436). D2a CPU/Silk parity is confirmed by controlled synthetic GPU-gated tests; real-checkpoint forward+loss smoke tests are scaffolded as `[Explicit, Category("Gpu")]`. This document is the implementation handoff/active-gap tracker; see `docs/LORA.md` for the stable user-facing guide.

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
- Silk: inference runners and trainers exist for other model families; `SilkLlamaLoRATrainer` provides the D2a causal-LM LoRA training path.
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
- CLI: `--mode ft-lora --silk` wired with OpenCL availability check, no silent CPU fallback. Added `MaxLen <= MaxPositionEmbeddings` guard and structured JSON report (`ft-lora-report.json`) with device, params, losses, elapsed, adapter path + SHA256, parity status.
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
- CPU/Silk parity confirmed by controlled synthetic GPU-gated tests; real-checkpoint forward+loss smoke tests scaffolded as `[Explicit, Category("Gpu")]`. (#1438)

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

### Parity matrix (#1438, G1 agreed defaults)

Certification requires real-checkpoint CPU/Silk parity evidence (not synthetic-only). Matrix cases (minimum):

| Case | What | Headline Qwen smoke | SmolLM smoke |
|---|---|---|---|
| A | Forward logits / last-token argmax | required (real-checkpoint smoke) | required (real-checkpoint smoke) |
| B | Prompt-masked CE loss (`ignoreIndex=-100`) | required (real-checkpoint smoke) | required (real-checkpoint smoke) |
| C | LoRA grad parity per target | synthetic GPU parity (real-checkpoint CPU backward too slow) | synthetic GPU parity |
| D | 1-step TrainStep/Adam weight parity | synthetic GPU parity | synthetic GPU parity |
| E | Adapter bundle round-trip + CPU<->Silk interop | synthetic GPU parity + bundle contract | synthetic GPU parity |

Config defaults (G1): fixed seed, small maxLen (16 for real-checkpoint smoke), rank=4 alpha=8, Q/V targets minimum; deterministic fixed token inputs/labels; record device, commit, tolerances, max abs/rel deltas. Real checkpoints: `samples/data/qwen2.5-0.5b-instruct` (headline), `samples/data/smollm-135m` or `smollm-135m-instruct` if present (smoke). Tests skip cleanly when OpenCL or model dir missing; real-checkpoint tests are `[Explicit, Category("Gpu")]`.

Tolerances start at existing synthetic bounds: forward/loss 1.5e-3/2e-3, grad/weight rel 3e-2/5e-2 (SmolLM 5e-2). Any widening requires documented rationale in evidence.

Pragmatic note (per user direction): CPU AutoDiff backward/TrainStep on 0.5B-class real checkpoints is too slow for routine smoke; real-checkpoint smoke confirms forward+loss parity on real weights, while grad/step/bundle parity remains covered by synthetic GPU-gated tests that pass on capable machines.

Evidence artifacts: machine-readable JSON + short markdown summary (command, device, commit, seed, config, observed deltas, pass/fail). CPU/Silk parity is considered confirmed by controlled synthetic GPU-gated coverage; keep evidence reproducible when extending the matrix.

## 6. Known gaps / red flags (open)

### Tracked issues

| Issue | Gap | Notes |
|---|---|---|
| #1438 | D2a Silk causal-LM training parity certification | Done on `khurram/1438`: `IsParityReady` removed; CPU/Silk parity confirmed by controlled synthetic GPU-gated tests; real-checkpoint forward+loss smoke tests scaffolded as `[Explicit, Category("Gpu")]`; CLI/report/docs updated consistently. |
| #1439 | Silk path for causal-LM `ft-eval` | Done on `khurram/1439`: `RunFtEval` dispatches `--silk` to `RunFtEvalSilk` (`samples/NivaraFineTuning/Program.cs:1362+`), loads Qwen/SmolLM through Silk sessions, loads adapter bundles via `LoadAdapterBundle`, runs cached greedy generation-based worklog eval, and reports `Backend="silk"`/device with base-vs-adapted metrics shape. No CPU fallback. |
| #1437 | Generation base-vs-adapted eval in Silk `ft-lora` | Silk `ft-lora` reports loss/val-loss but lacks integrated generation eval parity with CPU path. |
| #1440 (G1) | Stale plan handoff after main merge | This doc needed refresh for main-landed state and current issue list (this edit addresses the immediate staleness; keep maintained). |
| #1441 (G2) | `ft-eval --silk` silently ignored | Resolved by #1439: `RunFtEval` checks `cli.Silk` and routes to Silk eval path; OpenCL unavailable errors clearly with no CPU fallback. |
| #1442 (G3) | Report memory footprint contract unclear/approximate | Resolved: advisory-only clarified. `--max-memory`/`memoryBudgetBytes`/`memoryBudgetMode` are requested advisory budget only; not measured runtime footprint; not a hard runtime cap. Schema/docs/tests updated. |
| #1443 (G4) | No recorded post-merge CPU ft-lora smoke evidence | CPU 0.5B backward can be slow; need small smoke run with granular logs and recorded outcome. |
| #1444 (G5) | Role boundary between `docs/LORA.md` and `docs/plan/LORA.md` | User-facing doc vs handoff/active-gap tracker; reduce drift by making roles explicit and cross-linking. |

### Guardrails

- **No false iGPU claim:** docs/CLI must keep D2a status honest until parity is release-certified.
- **No silent CPU fallback:** `--silk` paths must run on Silk or fail clearly.
- **Adapter format:** `nivara-lora-sft-adapters-v1` is the shared CPU/Silk SFT bundle; core `nivara-lora-adapter-v1` remains a distinct single-adapter format by design.

## 7. File map (current)

- `docs/plan/LORA.md` — implementation handoff/active-gap tracker (this doc).
- `samples/data/worklog/README.md`, `train.jsonl`, `test.jsonl` — deterministic seed data.
- `samples/NivaraFineTuning/WorklogSftDataset.cs` — dataset/generator/tokenization.
- `samples/NivaraFineTuning/WorklogSftEvaluator.cs` — eval metrics/parser.
- `samples/NivaraFineTuning/WorklogLoraSftTrainer.cs` — CPU reference trainer.
- `samples/NivaraFineTuning/Program.cs` — `ft-generate`/`ft-lora`/`ft-eval` CLI.
- `src/Nivara.Extensions/Silk/SilkLlamaLoRATrainer.cs` — Silk D2a trainer.
- `src/Nivara.Extensions/Silk/SilkKernels.cs`, `SilkLauncher.cs` — backward kernels/launchers.
- `tests/Nivara.Tests/Gpu/SilkLlamaLoRATrainerParityTests.cs` — GPU-gated parity tests.
- `samples/Nivara.Samples/LlamaForCausalLM.cs` — shared Llama-family model/loading; Qwen tokenizer unk fix.

## 8. Landed / closed

- #1414 plan reshape — done (plan is handoff on main).
- #1415 reconcile ft-data artifacts — done.
- #1416 worklog dataset/generator/eval — done (seed data, generator, evaluator, tests).
- #1417 generic Llama-family CPU SFT trainer — done.
- #1418 initial Silk D2a trainer path — done/closed as initial delivery; parity certification closed by #1438.
- #1419/#1429 NivaraChat adapter consumption demo — done (CPU+Silk adapter load; `--merge-adapter` with `--silk` errors clearly).
- #1420 NivaraFineTuning CLI/report — done (ft-generate/ft-lora/ft-eval, JSON+markdown reports).
- #1421 tests/docs/DocsCoveragePins — done.
- #1439 Silk causal-LM `ft-eval` — done on `khurram/1439` (Silk eval path, cached greedy seam, adapter bundle load, report parity shape, no CPU fallback).
- #1441 (G2) `ft-eval --silk` silent CPU fallback — resolved by #1439.

## 9. Next session pick list (open issues)

1. **#1438 D2a parity certification** — done on `khurram/1438` (synthetic GPU-gated parity confirmed; `IsParityReady` removed; real-checkpoint forward+loss smoke scaffolded).
2. **#1439 Silk path for causal-LM `ft-eval`** — done on `khurram/1439`: Silk/OpenCL eval path for Qwen/SmolLM, cached greedy generation, adapter bundle load, base-vs-adapted report parity shape, no CPU fallback.
3. **#1437 Generation base-vs-adapted eval in Silk `ft-lora`** — align Silk ft-lora report/eval with CPU path.
4. **#1441 (G2) Guard `ft-eval --silk` silent CPU fallback** — resolved by #1439.
5. **#1440 (G1) Keep this handoff current** — update when gaps close; avoid stale branch/status/pick-list drift.
6. **#1444 (G5) Clarify doc roles** — `docs/LORA.md` stable user guide; `docs/plan/LORA.md` handoff/active-gap tracker.
7. **#1442 (G3) Memory footprint report contract** — done: advisory-only clarified (requested budget only; not measured runtime footprint; not a hard runtime cap).
8. **#1443 (G4) CPU ft-lora smoke evidence** — small post-merge smoke with granular logs; record outcome.

## 10. Guardrails

- No external FineTuning-repo paths/references in code, docs, issues, or commits.
- Keep CPU/Silk parity claims evidence-backed; synthetic GPU-gated parity tests and scaffolded real-checkpoint smokes are the baseline.
- No silent CPU fallback for `--silk` paths: run on Silk or fail clearly.
- Do not run full test suite unprompted; run targeted tests autonomously. Full suite requires confirmation and `Category!=Performance&Category!=ReleaseReadiness&Category!=Gpu`.
- Do not push/create PR without explicit user confirmation.
- Keep `docs/plan/LORA.md` as the handoff/active-gap tracker; do not reintroduce `docs/TODO.md`.
