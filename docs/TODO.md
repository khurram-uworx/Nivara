# TODO: #1438 - LoRA D2a Silk causal-LM training parity certification

## Problem

Issue: https://github.com/khurram-uworx/Nivara/issues/1438 (milestone 1.5.4, base branch `khurram/lora`, PR target `khurram/lora`)

Silk causal-LM LoRA training (D2a) is functional but not release-parity-certified. `SilkLlamaLoRATrainer.IsParityReady` is hard-coded `false`; CLI/report/docs still report "not release-parity-certified". #1418 delivered initial D2a; parity gate was not fully closed.

## Acceptance criteria (from issue)

- CPU/Silk parity evidence covers the agreed real-checkpoint matrix for headline Qwen and SmolLM smoke.
- Any discovered parity gaps are fixed or tracked with explicit blocking issues.
- `IsParityReady`, CLI messaging, and docs are updated consistently with actual certification state.
- GPU-gated parity tests pass on capable machines and skip cleanly otherwise.

## Grounding / current state

- `src/Nivara.Extensions/Silk/SilkLlamaLoRATrainer.cs:165` — `public bool IsParityReady => false;` (hard gate). Class summary at `:22-25` is stale ("backward/optimizer/parity being completed...").
- Existing GPU-gated synthetic parity tests: `tests/Nivara.Tests/Gpu/SilkLlamaLoRATrainerParityTests.cs` (`[Category("Gpu")]`, skips cleanly on no OpenCL). Covers Qwen-shaped Q/V forward/CE+grads/1-step Adam, bundle round-trip, bundle load into CPU model, SmolLM-shaped all-targets forward+CE+grads. Tolerances: forward/loss 1.5e-3/2e-3, grad/weight rel 3e-2/5e-2 (SmolLM 5e-2).
- CPU reference: `samples/NivaraFineTuning/WorklogLoraSftTrainer.cs` (generic Llama-family, AutoDiff Grad, AdamW, CrossEntropyLoss ignoreIndex -100, prompt-masked, saves `nivara-lora-sft-adapters-v1` with layer/target defaults -1/"").
- Silk trainer: `SilkLlamaLoRATrainer` has forward, prompt-masked CE backward, transformer backward, LoRA grad accumulation, Adam, TrainStep, adapter save/load. Uses Silk kernels/launchers.
- Shared bundle: `src/Nivara.Extensions/Silk/LoraSftAdapterBundle.cs` (`nivara-lora-sft-adapters-v1`).
- CLI/report parity strings: `samples/NivaraFineTuning/Program.cs:1108,1202,1243`; `samples/NivaraFineTuning/LoraSftRunReport.cs:86` default, `:238` markdown footer always says not release-parity-certified.
- Tests asserting not-certified: `tests/Nivara.Tests/FineTuning/LoraSftRunReportTests.cs:133`.
- Docs: `docs/LORA.md:11-16,92`, `docs/plan/LORA.md:3,44,55,87-139`, `samples/NivaraFineTuning/LORA-TUNING.md:30`.
- Related open issues (out of scope for #1438, but adjacent): #1437 Silk ft-lora generation eval, #1439 Silk ft-eval path, #1441 ft-eval --silk silent CPU fallback guard, #1442 memory report contract, #1443 CPU smoke evidence, #1444 doc role/handoff drift.

## G1 decisions / red flags (require human clearance before implementation)

1. **Real-checkpoint parity matrix definition (core decision).** Proposed matrix for "headline Qwen and SmolLM smoke":
   - **Models/checkpoints**: Qwen2.5-0.5B-Instruct at `samples/data/qwen2.5-0.5b-instruct` (headline); SmolLM at `samples/data/smollm-fn-135m` if present locally (actual dir name observed; docs/CLI sometimes say `smollm-135m-instruct` — need to confirm which checkpoint is the intended SmolLM smoke target and whether naming should be normalized). Skip cleanly if model dir missing.
   - **Cases**: (A) forward logits/last-token argmax parity, (B) prompt-masked CE loss parity (ignoreIndex -100), (C) LoRA grad parity per target, (D) 1-step TrainStep/Adam weight parity, (E) adapter bundle save/load round-trip and CPU<->Silk bundle interop.
   - **Config**: rank=4, alpha=8, targets q,v (headline) and all-targets SmolLM smoke; maxLen small (64/128), fixed seed, deterministic synthetic token inputs/labels from local data or fixed sequences. Record device, commit, tolerances, observed max abs/rel deltas.
   - **Tolerances**: start from existing synthetic tolerances (forward/loss 1.5e-3/2e-3, grad/weight rel 3e-2/5e-2, SmolLM 5e-2); any widening requires rationale in evidence.
   - **Evidence**: machine-readable JSON/markdown artifact checked in under an agreed location (proposed: `docs/evidence/lora-d2a-parity/` or noted in PR; avoid large binaries). Human-run on capable OpenCL machine.
2. **Real-checkpoint tests in default test runs?** Real checkpoints are large/local and not in CI. Proposed: add GPU-gated tests that skip cleanly when OpenCL unavailable OR model dir missing; mark real-checkpoint cases `[Explicit]` (or keep `[Category("Gpu")]` with model-dir skip) so normal GPU runs don't fail on missing assets. Need confirmation.
3. **Certification scope boundary.** #1438 acceptance criteria focus on training parity certification. #1437/#1439/#1441 are eval-path gaps; they should not block #1438 unless they reveal a training-parity defect. I will not silently absorb them.
4. **`LoraSftRunReport.ToMarkdown()` footer** currently always emits `_Silk/iGPU path is not release-parity-certified unless explicitly stated._` (`LoraSftRunReport.cs:238`). When flipping certification, this guardrail must be updated consistently — either conditioned on parity status or replaced with an accurate statement. Needs careful wording to avoid false claims for non-certified runs.

## Proposed changes

### A. Define parity matrix + evidence format
- Add parity matrix definition to `docs/plan/LORA.md` (D2a parity certification section) and/or a dedicated evidence note. Keep reproducible command lines.
- Define evidence artifact schema (JSON fields: model, modelDir, device, commit, seed, rank, alpha, targets, maxLen, cases, tolerances, observed deltas, pass/fail, timestamp).

### B. Extend GPU-gated parity tests
- Add real-checkpoint parity tests in `tests/Nivara.Tests/Gpu/SilkLlamaLoRATrainerParityTests.cs` (or sibling file) with:
  - `[Category("Gpu")]`
  - Skip when no OpenCL.
  - Skip when required model dir/files missing.
  - `[Explicit]` for real-checkpoint cases if agreed (to avoid accidental long runs).
- Cover Qwen headline smoke + SmolLM smoke per agreed matrix. Reuse existing comparison helpers where possible.
- Keep existing synthetic tests unchanged.

### C. Fix any parity gaps found
- If matrix reveals CPU/Silk behavioral gaps, fix with minimal scoped changes + regression tests. If a gap is large/non-trivial, create blocking issue immediately and record in GitHub issues log; do not flip `IsParityReady`.

### D. Flip certification state consistently (only when evidence supports it)
- `src/Nivara.Extensions/Silk/SilkLlamaLoRATrainer.cs`: set `IsParityReady => true` (or evidence-backed value), update class summary.
- `samples/NivaraFineTuning/Program.cs`: update Silk D2a parity console/report strings to certified state with evidence reference.
- `samples/NivaraFineTuning/LoraSftRunReport.cs`: update default `ParityStatus` and markdown footer/wording to be accurate and consistent; avoid false claim when not certified.
- `tests/Nivara.Tests/FineTuning/LoraSftRunReportTests.cs`: update assertions to match new certified wording.
- `docs/LORA.md`, `docs/plan/LORA.md`, `samples/NivaraFineTuning/LORA-TUNING.md`: update status consistently.

### E. DocsCoveragePins / docs gate review
- Before deleting TODO, review branch diff for docs/citation changes; revise `DocsCoveragePins` if any gated doc/citation counts changed. Run `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` if docs changed.

## Blast radius

- **Code**: `SilkLlamaLoRATrainer` (flag/summary only expected), `Program.cs` Silk ft-lora reporting, `LoraSftRunReport` default/footer, possibly parity test fixtures/helpers. No kernel/math changes unless parity gap found.
- **Tests**: `LoraSftRunReportTests` (wording), `SilkLlamaLoRATrainerParityTests` (add cases), possibly Docs tests if docs/citations change.
- **Docs**: `docs/LORA.md`, `docs/plan/LORA.md`, `samples/NivaraFineTuning/LORA-TUNING.md`, this TODO (deleted before PR).
- **Risk**: certification claim must be backed by evidence; do not flip on synthetic-only evidence. Real-checkpoint tests must skip cleanly on CI/no-GPU/no-model-dir. Report footer change may affect consumers expecting old string - check tests.

## Verification

- Ask before running any long-running/full test commands.
- Targeted tests (autonomous after asking once per run as needed): `LoraSftRunReportTests`, `SilkLlamaLoRATrainerParityTests` filter (GPU-gated; skip cleanly on no OpenCL/model dir).
- If docs changed: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` and whitespace format check.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` before PR.
- PR test plan: build Release; run targeted tests; run docs tests if docs changed; report GPU skip/pass behavior and real-checkpoint evidence status honestly.

## Planned commits

1. `docs: plan #1438 LoRA D2a parity certification in TODO.md`
2. `test: add real-checkpoint LoRA D2a parity matrix scaffolding` (GPU-gated, skip-cleanly, matrix helpers/evidence format)
3. `docs: define D2a parity matrix and evidence expectations` (plan/LORA or evidence doc)
4. `feat/fix: close D2a parity gaps found by matrix` (only if needed; may be multiple logical commits)
5. `chore: flip D2a parity certification state with evidence` (IsParityReady/CLI/report/docs/tests consistent) — only after evidence supports it
6. `docs: remove TODO.md — plan executed`

## GitHub issues log

- [ ] #1438 — LoRA D2a: close Silk causal-LM training parity certification (this work)
- Related (existing, out of scope): #1437, #1439, #1441, #1442, #1443, #1444

## Execution notes

- Do not push/create PR without explicit user confirmation. (User explicitly requested push/PR in this session.)
- Ask before running tests / long-running commands.
- Create GitHub issues immediately for deferred out-of-scope concerns discovered during execution.
- Do not run full test suite unprompted; full suite requires confirmation and `Category!=Performance&Category!=ReleaseReadiness&Category!=Gpu`.
- Do not sign commits/PRs; no AI attribution trailers.
