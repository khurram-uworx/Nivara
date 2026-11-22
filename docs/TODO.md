# #1418 — Silk iGPU causal-LM LoRA training with CPU/Silk parity

Branch: `khurram/1418` (off `khurram/1417`). Issue: https://github.com/khurram-uworx/Nivara/issues/1418. Milestone 1.5.4.

## Problem

Deliver real Silk (OpenCL/iGPU) causal-LM LoRA training with CPU/Silk parity (D2a, no compromise) for release readiness. Silk is Nivara's headline; "FT SLMs on iGPUs" is the headline showcase. Release readiness requires CPU/Silk parity when the whole LoRA plan executes. No false iGPU claim; training f32 only unless an existing supported lane applies.

## What exists today (baseline on khurram/1417)

- `SilkLlamaLoRATrainer` (`src/Nivara.Extensions/Silk/SilkLlamaLoRATrainer.cs`): generic Llama-family LoRA trainer (Qwen/SmolLM configs), base weight upload, LoRA A/B/grad/moment buffers, forward with LoRA delta on q/k/v/o/gate/up/down, prompt-masked CE backward, full transformer backward, LoRA grad accumulation, Adam, `TrainStep`, adapter bundle save/load. `IsParityReady => false`.
- Silk kernels/launchers present: `rms_norm_backward`/`RmsNormBackward`, `rotary_backward`/`RotaryBackward`, `swiglu_backward`/`SwiGluBackward`, `scale`, GQA backward prep/dq/dkv, `SoftmaxCrossEntropyBackward(ignoreIndex)`.
- CPU reference: `WorklogLoraSftTrainer` (`samples/NivaraFineTuning/WorklogLoraSftTrainer.cs`): generic Llama-family LoRA SFT, frozen base, AdamW on adapters, prompt-masked CE `ignoreIndex:-100`, best-checkpoint restore, adapter save (`nivara-lora-sft-adapters-v1`).
- CLI: `ft-generate`, `ft-lora` (CPU + `--silk` path, no silent CPU fallback, honest parity warning), `ft-eval` base+adapter before/after report.
- GPU parity tests: 5 `[Category("Gpu")]` tests in `tests/Nivara.Tests/Gpu/SilkLlamaLoRATrainerParityTests.cs` (Qwen Q/V forward/loss/grad, TrainStep/Adam, adapter round-trip, SmolLM all-targets). Skip cleanly when OpenCL unavailable.
- Plan handoff: `docs/plan/LORA.md`.

## Acceptance criteria (issue)

- Silk causal-LM LoRA training runs on iGPU for headline case (Qwen2.5-0.5B).
- CPU/Silk parity tests pass on device-capable machine; skipped cleanly otherwise.
- Generic Llama-family path verified at least by SmolLM smoke where supported.
- No false iGPU/release-parity claim until evidence supports it.

## Proposed changes

### 1. Adapter bundle format convergence (core gap)
- Make CPU `WorklogLoraSftTrainer.SaveAdapters` emit the same rich entries as Silk (`layer`, `target`, `aShape`, `bShape`) in `nivara-lora-sft-adapters-v1`.
- Shared helper for bundle save/load validation (rank/alpha/target count/shapes/layer-target identity) used by both CPU and Silk paths.
- Document and verify that `nivara-lora-sft-adapters-v1` (multi-adapter SFT bundle) is distinct from core `LoraAdapterSerializer` (`nivara-lora-adapter-v1`, single adapter). Add a bridge note; do not force-fit without a deliberate multi-adapter design. Only touch core serializer if an additive, justified bridge is needed.
- Files: `samples/NivaraFineTuning/WorklogLoraSftTrainer.cs`, `src/Nivara.Extensions/Silk/SilkLlamaLoRATrainer.cs`, `samples/NivaraFineTuning/Program.cs` (`LoadCpuAdapterBundle`), possibly `src/Nivara/AutoDiff/Serialization/LoraAdapterSerializer.cs`.

### 2. Memory guard + structured report artifact
- Explicit guards before training/eval: `MaxLen <= model MaxPositionEmbeddings`, sane caps, surface maxExamples/gradAccum; clear errors.
- Structured report (JSON + console summary) for `ft-lora` and `ft-eval`: mode, model/template, device (silk/cpu), rank/alpha/targets, trainable/base params, maxLen/maxExamples/seed, initial/best/train/val loss, elapsed, adapter path + format version + SHA256, parity status, before/after eval metrics when adapter provided. Honest wording (smoke/parity evidence, not release-certified).
- Files: `samples/NivaraFineTuning/Program.cs`, small report DTO/file in `samples/NivaraFineTuning`, `README.md`.

### 3. Silk ft-eval / real-checkpoint evidence path
- Verify `ft-eval` evaluates a Silk-produced adapter against CPU reference model after format convergence.
- Add GPU-gated test: save Silk adapter, reload into CPU model, check logits/loss do not regress on synthetic fixture.
- Prepare human-run evidence commands (do not run autonomously): Qwen2.5-0.5B headline `ft-lora --silk` bounded smoke; `ft-eval` base vs adapted on Silk adapter; optional SmolLM smoke. Capture device, elapsed, initial/best val, before/after metrics, adapter hash.
- Record evidence in `docs/plan/LORA.md` (smoke vs release-cert distinction).

### 4. Parity evidence hardening (keep IsParityReady honest)
- Keep 5 parity tests green; add only justified cases closing 1418 gaps (e.g., parity over another seed/config or real-checkpoint small fixture if deterministic/fast enough for `[Category("Gpu")]`).
- Document tolerance rationale (float accumulation across OpenCL kernels, softmax/attention reductions, Adam moments). Justify any widening with data; prefer tightening only with evidence.
- Flip `IsParityReady` only when parity suite covers headline/SmolLM shapes, real-checkpoint smoke evidence recorded, adapter round-trip/eval path verified, no known correctness gap. Otherwise leave false and record exact blocker.

### 5. Docs/gates (minimum)
- Update `docs/plan/LORA.md` and `samples/NivaraFineTuning/README.md` with status/evidence/honest wording.
- If top-level `docs/*.md` added/removed or citations change, update `DocsCoveragePins` per AGENTS coupling rule. `docs/plan/LORA.md` is excluded from scan.
- Run doc snippet gate only if gated csharp blocks changed.

## Verification steps

- Targeted builds: `dotnet build src/Nivara.Extensions/Nivara.Extensions.csproj -c Release`; `dotnet build samples/NivaraFineTuning/NivaraFineTuning.csproj -c Release`.
- Targeted tests (ask before long runs): Silk parity + fine-tuning tests touched; formatting whitespace gate before PR/push.
- Human-approved/device-dependent: real Qwen2.5-0.5B Silk smoke, `ft-eval` before/after, SmolLM smoke, long CPU reference runs. Record outputs as evidence.
- Do not run full suite unprompted.

## Planned commits

1. `docs: plan #1418 in TODO.md`
2. `feat(ft): converge Silk/CPU adapter bundle schema and shared validation`
3. `feat(ft): add memory guard and structured report artifact for ft-lora/ft-eval`
4. `test(gpu): add Silk adapter -> CPU ft-eval parity smoke test`
5. `docs: update LORA plan and README with #1418 evidence and honest parity status`
6. `docs: remove TODO.md — plan executed` (only after G2 clears)

## Blast radius

- `samples/NivaraFineTuning/WorklogLoraSftTrainer.cs` (SaveAdapters schema)
- `src/Nivara.Extensions/Silk/SilkLlamaLoRATrainer.cs` (SaveAdapterBundle/LoadAdapterBundle, IsParityReady)
- `samples/NivaraFineTuning/Program.cs` (RunFtLoraSilk, RunFtEval, LoadCpuAdapterBundle, guards, report)
- `tests/Nivara.Tests/Gpu/SilkLlamaLoRATrainerParityTests.cs` (new GPU-gated test)
- `docs/plan/LORA.md`, `samples/NivaraFineTuning/README.md`
- Potential: `src/Nivara/AutoDiff/Serialization/LoraAdapterSerializer.cs` (only if additive bridge justified)
- Docs pins (`DocsCoveragePins`) if top-level docs/citations change.

## GitHub issues log

- (none yet; create via `gh issue create --repo khurram-uworx/Nivara` as deferred work is discovered.)

## Guardrails

- No external FineTuning-repo paths/references in code, docs, issues, commits.
- No false "FT on iGPU" or release-parity claim until D2a evidence supports it.
- Keep `docs/plan/LORA.md` as handoff; do not reintroduce `docs/TODO.md` after removal.
- Do not push/create PR without explicit user confirmation (user has now authorized push+PR for this task when sorted).
