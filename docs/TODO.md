# Issue #1222 — Enable native (BF16) Qwen on Silk after #1219

**Status:** Plan (in progress, 2026-11-16)
**Branch:** khurram/1222
**Related:** #1217/#1220 (Qwen native CPU), #1219/#1231 (Silk SmolLM BF16 decode), docs/QWEN.md, docs/SILK.md, docs/plan/MODEL-LOADING-HONESTY.md.

## 1. Problem

After #1217, bare `qwen` defaults to native (BF16 storage / F32 compute) on CPU. On the Silk
(OpenCL) device path, `Program.cs` (qwen Silk block) coerces native/default to the F32 device
path with a diagnostic, because the BF16 cached-decode path was suspected of crashing with
`CL_OUT_OF_RESOURCES` (issue #1219).

## 2. Finding — #1219 did not affect Qwen

`SilkQwenGpuRunner.ForwardCachedArgmax` / `ForwardCached` route the final norm + tied LM head
through the instance `RmsNormGemv` dispatcher (`SilkQwenGpuRunner.cs`), which selects
`RmsNormGemvBf16` when `bf16Weights`. That is exactly the fix #1231 applied to SmolLM. The #1219
bug was SmolLM-only (its runner called `SilkLauncher.RmsNormGemv` (F32 kernel) directly against a
BF16 `tokEmbTransposed` buffer).

Verified 2026-11-16 on Intel Iris Xe:
- `qwen --silk benchmark --precision bf16`: 7 tokens, 3.85/3.98/3.98 s (~567 ms/token).
- `qwen --silk benchmark`: 3.66/3.61/3.62 s (~517 ms/token).
- `dotnet test --filter FullyQualifiedName~SilkQwen`: 5/5 pass.

So #1222 is wiring + coverage + docs — no kernel change. (SmolLM BF16 decode regression test is
separate; #1231 shipped the fix without one — see GitHub issues log.)

## 3. Decision recorded (issue's "decide explicitly ... and record")

On the Silk device path, `native` and `--precision bf16` are the same run: BF16 weights on device,
F32 activations, widen-in-register dots. There is no "BF16 compute" device path. Therefore
`native` maps onto the BF16 device path, mirroring SmolLM (`SilkSmolLMGpuRunner.Bf16.cs`).
The label "Precision: BFloat16" currently printed by `RunToolsSilk` is a misnomer for this; it
becomes "BFloat16 storage / F32 compute".

## 4. Proposed changes

### A. `samples/NivaraInference/Program.cs`
1. qwen Silk block: drop the `#1219` comment, the note and the coercion. Route `bf16 || native` to
   `tensorsBf16` for tools/plain/benchmark; keep `distill` on `F32()` first in branch order.
2. Distill guard: extend `native` to `native || (useSilk && bf16)`, keep the "requires widened F32
   weights" wording.
3. Help text: fix the stale `--precision f32` line ("qwen default: BF16-on-disk, SIMD-widened to
   F32 at load"), the `--precision bf16` parenthetical (claims benchmark-only), and the GPU block
   (name the SmolLM/Qwen BF16-on-Silk default).
4. Enable explicit `--precision bf16` for `qwen --silk` non-distill modes (consistency: the CLI
   would otherwise reject the explicit spelling of the behaviour it just made the default).

### B. `samples/NivaraInference/Qwen.cs`
1. `RunToolsSilk<T>`: print "BFloat16 storage / F32 compute".
2. `RunBenchmarkSilk<T>`: replace the discarded `LoadModel<T,T>(tensors, modelDir)` with
   `LoadConfigAndTokenizer(modelDir)` (avoids building a throwaway CPU model).

### C. `samples/NivaraChat/Qwen/QwenMode.cs`
1. Read `SafeTensorsLoader.Read<BFloat16>` and call `ExecuteSilk<BFloat16>`; delete the note/comment;
   collapse the two identical `typeof(T)` branches.

### D. `samples/NivaraInference/Qwen.cs` / `Program.cs` distill default
Bare `qwen distill` / `qwen --silk distill` currently default to `native` and then hit the distill
guard, so the README's `qwen --silk distill --teacher-examples 12` fails. Decision: keep the guard
(distill stays on explicit `--precision f32`) and fix the README to spell `--precision f32`. Do not
silently default distill to f32 — the issue asks for a named guard, not a silent fallback.

### E. Tests
1. `tests/Nivara.Tests/Qwen/SilkQwenBf16ParityTests.cs`: add a `ForwardCachedArgmax` (and `ForwardCached`)
   BF16-vs-F32-upcast parity test over synthetic weights, `[Category("Gpu")]`, skip without device.
2. `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs` (runs in an empty cwd; no GPU or
   checkpoint needed):
   - `qwen --silk` default → stdout "Loading weights (native)", no "1219", stderr "Model file not found".
   - `qwen --silk --precision bf16 tools` → reaches the model lookup (accepted).
   - `qwen --silk --precision bf16 distill` → rejected, "requires widened F32 weights".
   - `qwen --silk --precision f32 tools` → reaches the model lookup (F32 route untouched).
3. Optional real-checkpoint BF16 tools-loop test mirroring `SilkQwenToolsWeatherLoopTests`
   (reads `Read<BFloat16>`); Gpu category, skips without checkpoint/GPU. Recommended because the
   acceptance criterion is the tool loop.

### F. Docs
- `docs/QWEN.md`: line 7 (chat host no longer coerces), the #1219 sentence at 59, §GPU path 274,
  Silk protocol/results 281/310/344/349 (relate the published 130 ms/token 2026-10-17 row to the
  F32 device path; add the 2026-11-16 same-session datapoint ~517 ms/token F32 / ~567 ms/token BF16
  with a dated caveat that the machine/driver state differs from 2026-10-17).
- `docs/SILK.md`: §10 367-375 ("F32 is the default" is now wrong for SmolLM and Qwen), §13b
  658-670, §14.
- `samples/NivaraInference/README.md`: 182-183, 356-357.
- `samples/NivaraChat/README.md`: add `--qwen` to the `--silk` row.
- `docs/plan/MODEL-LOADING-HONESTY.md`: Phase 1 Silk bullet — mark resolved; drop the #1219
  blocker; record the measurement.
- Prose-only edits keep `DocsCoveragePins` (122 citations / 60 docs / 53 code citations) unchanged.
  Any citation added or removed moves those counts with arithmetic.

## 5. Verification

- `dotnet build Nivara.slnx`
- Targeted: `SilkQwen` filter, `InferenceCliArgumentTests`, `Docs` namespace.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- GPU runs: `qwen --silk tools` (BF16 default) must close the tool loop; `qwen --silk --precision f32 tools` unchanged.
- Human-run: the `data\qwen-bench\` block if we want published figures (see decision 4).

## 6. Blast radius

- `Program.cs` qwen Silk dispatch + guards + help text; `Qwen.cs` labels + `RunBenchmarkSilk` model build.
- `QwenMode.cs` (NivaraChat): load path only.
- Tests: `SilkQwenBf16ParityTests.cs` (new test), `InferenceCliArgumentTests.cs` (new cases).
- Docs: QWEN.md, SILK.md, both READMEs, MODEL-LOADING-HONESTY.md.
- No core library changes; no `DocsCoveragePins` change expected.

## 7. Planned commits

1. docs: plan #1222 in TODO.md
2. test(inference): Qwen BF16 Silk cached-decode parity (ForwardCachedArgmax)
3. test(inference): InferenceCliArgumentTests pin the qwen Silk native default
4. feat(inference): route qwen native/default to BF16 Silk; drop #1219 coercion
5. feat(nivarachat): qwen --silk reads native BF16 weights
6. docs: Qwen/Silk notes for native BF16 default on Silk
7. docs: remove TODO.md — plan executed

## 8. GitHub issues log

- [x] #1232 — Silk SmolLM BF16 cached-decode: add `ForwardCachedArgmax` regression test (#1219 follow-up) (created while working on #1222).
- [x] #1233 — NivaraInference: `--synthetic-weights` is documented but not implemented (created while working on #1222).
- [ ] Published Silk row (130 ms/token, 2026-10-17) is not reproducible today on either dtype (~517 ms/token F32, ~567 ms/token BF16); needs a human-run `data\qwen-bench` capture before quoting new numbers.
- [x] #1234 — NivaraChat `--qwen --silk` access violation in `ReleaseMemObject` during cleanup (pre-existing, not caused by #1222) (created while working on #1222).

## 9. Decisions resolved (G1)

1. Enable explicit `--precision bf16` on `qwen --silk` non-distill modes — yes (consistency).
2. NivaraChat `--qwen --silk` — included.
3. Bare `qwen distill` — keep the guard, fix the README.
4. Superseded Silk measurement — record dated caveat + 2026-11-16 datapoint; defer published numbers to a human-run capture.
5. `--synthetic-weights` — separate issue (not wired here).
6. PR #1231 test — separate issue (not in this branch).
