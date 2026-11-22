# TODO: Silk Qwen/SmolLM LoRA adapter bundle consumption (#1429)

## Problem

GitHub issue [#1429](https://github.com/khurram-uworx/Nivara/issues/1429): `NivaraChat --qwen/--smollm ... --silk --adapter <path>` currently throws `NotSupportedException` because `SilkQwenSession` / `SilkSmolLMSession` have no LoRA adapter loading path. CPU path already supports `--adapter` (and `--merge-adapter`) via `LoraAdapterLoader.Apply` on `LlamaForCausalLM<float,TWeight>`.

Acceptance criteria from issue:
- `NivaraChat --qwen plain --silk --adapter <path>` loads and generates with the adapted model.
- `NivaraChat --smollm plain --silk --adapter <path>` loads and generates with the adapted model.
- Rank/alpha/count/shape/data-length mismatches throw clear errors.
- No RAG regression.

## Grounding (G1)

- **Code-memory navigation**: used `get_architecture_overview`, `find_related_code` for `LoraAdapterLoader`, `LoraSftAdapterBundle`, `SilkQwenSession`, `SilkSmolLMSession`, `QwenMode`, `SmollmMode`; read runners, trainer, tests, and samples.
- **Existing CPU contract** (`samples/NivaraChat/Helpers/LoraAdapterLoader.cs`): loads `LoraSftAdapterBundle` (format `nivara-lora-sft-adapters-v1`, `src/Nivara.Extensions/Silk/LoraSftAdapterBundle.cs`), parses targets, `model.AttachLora(targets, rank, alpha)`, validates rank/alpha/count/index/AShape/BShape/data length, copies base64 float A/B into adapters.
- **Existing Silk trainer pattern** (`src/Nivara.Extensions/Silk/SilkLlamaLoRATrainer.cs`): inference-relevant pieces are `LoraPair` (A/B/grad/moments — inference only needs A/B), `GemmWithLora` (`base Gemm + input@A -> tmp@B -> scale -> add`), target order Q,K,V,O,Gate,Up,Down per layer, `EnumerateLoraPairs`, bundle save/load.
- **Silk runners** (`SilkQwenGpuRunner*.cs`, `SilkSmolLMGpuRunner*.cs`): F32 and BF16 constructors; `LayerWeights` holds base projection buffers only; forward paths include full forward, ForwardArgmax, ForwardPrefill, ForwardPrefillArgmax, ForwardCached, ForwardCachedArgmax. Cached decode uses fused `RmsNormGemv`/`MlpBlock`/`DecodeOproj` (and BF16 variants); these are not LoRA-aware.
- **Bundle target semantics**: `LlamaLoRATarget` (samples) has `FFN` single flag; `SilkLlamaLoRATarget` (Silk) has `Gate|Up|Down|FFN` composite. Bundles may contain `Targets` strings and per-entry `Layer/Target`; CPU-saved test bundles may have `Layer=-1, Target=""`. Loader must be tolerant and order-compatible with `LlamaForCausalLM.AttachLora` / trainer enumeration.
- **microsoft-learn grounding**: not required for a novel .NET tensor API here; work follows established in-repo Silk/OpenCL buffer and kernel patterns. No new .NET preview type/member assumptions.

## Decisions / red flags (G1)

- **`--merge-adapter` on Silk**: issue acceptance only covers `--adapter`. Merging into BF16-stored base weights is possible but higher risk. **Plan**: support `--adapter` on Silk; keep `--merge-adapter` as an explicit `NotSupportedException` on Silk (clear message, no silent fallback) unless implementation proves trivial/safe. Flagged for human confirmation if they want merge parity now.
- **Fused decode paths**: `RmsNormGemv`/`MlpBlock`/`DecodeOproj` have no LoRA inputs. **Plan**: when any relevant LoRA slot is active, decompose fused decode into primitive steps + LoRA delta (RMSNorm/Gemm/AddBias/Scale/Add/SwiGlu/DecodeGqaAttention). Keep fused fast path when no LoRA is active. Correctness first; perf tradeoff acceptable for adapter demo.
- **BF16 base + F32 LoRA**: base weights may be BF16 device buffers; LoRA A/B loaded as F32 device buffers and applied via F32 Gemm on F32 activations. This matches F32 compute model.
- **Validation parity**: mirror `LoraAdapterLoader` checks (rank, alpha, count, index, A/B shape, data length, missing file/format through bundle load). Error messages should be clear and specific.

## Proposed changes

### A. Silk inference LoRA support (`src/Nivara.Extensions/Silk/`)

1. **Inference LoRA slot** - add internal `SilkInferenceLoraPair` (or nested type) with A/B device buffers, in/out/rank, scale (`alpha/rank`). No grad/optimizer state.
2. **Runner layer slots** - extend `LayerWeights` in `SilkQwenGpuRunner` and `SilkSmolLMGpuRunner` with optional Q/K/V/O/Gate/Up/Down LoRA slots; add dispose cleanup.
3. **Runner LoRA state/scratch** - add fields for `hasLora`, `loraRank`, `loraScale`, `loraTmpBuf` (`maxSeqLen*rank`), `loraOutBuf` (`maxSeqLen*maxOutDim`), allocated on load.
4. **GemmWithLora helper** - for rows>=1:
   - base Gemm (existing bf16/f32 dispatch)
   - if slot: `Gemm(input,A,tmp,rows,in,rank)`, `Gemm(tmp,B,out2,rows,rank,out)`, `Scale(out2,out2,scale,rows*out)`, `Add(output,out2,output,rows*out)`
5. **Full/prefill forward paths** - replace projection Gemm calls with GemmWithLora for Q,K,V,O,Gate,Up,Down in all full-sequence forward methods in both runners.
6. **Cached decode paths** - when LoRA active, decompose fused helpers:
   - QKV: RMSNorm then Gemm(rows=1)+bias+LoRA per slot.
   - O proj: DecodeGqaAttention then Gemm(rows=1)+LoRA+residual add.
   - MLP: RMSNorm, gate Gemm+LoRA, up Gemm+LoRA, SwiGlu, down Gemm+LoRA, residual add.
   Keep existing fused fast path when no LoRA is active.
7. **Adapter load API** - add `LoadAdapter(string adapterPath)` (or `ApplyAdapterBundle`) on `SilkQwenSession` and `SilkSmolLMSession`, delegating to runner. Runner validates and uploads.
8. **Validation** - mirror `LoraAdapterLoader`: format/rank/adapters via bundle load; parse targets; expected count from config+targets; rank/alpha/count/index/shape/data-length checks; clear `InvalidOperationException`/`ArgumentException`/`FileNotFoundException` as appropriate.

### B. NivaraChat wiring (`samples/NivaraChat/`)

1. **QwenMode.ExecuteSilk** - remove blanket adapter NotSupportedException; if `AdapterPath` set, load after `SilkQwenSession.Open` and log. Keep `MergeAdapterPath` as explicit Silk NotSupportedException unless implemented.
2. **SmollmMode.RunSilk** - same.
3. **Help text/options** - update `--adapter` help to reflect Silk support (and document merge-adapter boundary if kept unsupported on Silk).

### C. Tests

1. **Validation tests** - add Silk adapter loader validation tests mirroring `LoraAdapterLoaderTests` semantics (rank/alpha/count/shape/data-length/missing file). Prefer CPU-runnable tests where no OpenCL device required by loading/validation logic; if session/runner construction requires device, mark `[Category("Gpu")]`.
2. **GPU smoke tests** - `[Category("Gpu")]` small Silk Qwen/SmolLM adapter load + forward/argmax finite output; verify adapted output differs from base on same input where feasible.
3. **NivaraChat tests** - update any tests expecting Silk+adapter NotSupportedException; add coverage for argument/help surface as needed.

### D. Docs / pins

- Update `docs/plan/LORA.md` status note (#1419 Silk inference does not yet consume bundles).
- Update NivaraChat README/help text as needed.
- Re-run docs coverage/snippet gates if docs touched; adjust `DocsCoveragePins` only if citation/document counts change (extend arithmetic in comments).

## Blast radius

| Area | Files/symbols | Risk | Tests |
|------|---------------|------|-------|
| Silk runner inference | `SilkQwenGpuRunner.cs`, `.Bf16.cs`, `SilkSmolLMGpuRunner.cs`, `.Bf16.cs`, LayerWeights, Forward*/cached helpers | Medium-high: touches hot forward paths; LoRA inactive path must remain unchanged | Existing Silk/GPU tests, new GPU smoke, targeted build |
| Silk session API | `SilkQwenSession.cs`, `SilkSmolLMSession.cs` | Low: additive API | New tests, compile |
| Bundle/loader validation | New loader/helper in `Nivara.Extensions/Silk` or session/runner methods; uses `LoraSftAdapterBundle` | Low-medium: validation parity | New validation tests, existing LoraAdapterLoaderTests unaffected |
| NivaraChat modes | `QwenMode.cs`, `SmollmMode.cs` | Low: removes explicit throw, adds load call | Manual CLI smoke, existing chat tests, arg tests |
| Docs | `docs/plan/LORA.md`, possibly NivaraChat README/help | Low | DocsCoveragePins/snippet gate if affected |

No changes to core Nivara storage/AutoDiff/ML kernels. No RAG behavior changes. No silent CPU fallback.

## Planned commits

1. `docs: plan Silk LoRA adapter consumption in TODO.md` (this file)
2. `feat: add Silk inference LoRA slots and forward delta` (runners + session load/validation + tests)
3. `feat: wire --adapter through Silk Qwen/SmolLM chat modes` (NivaraChat + help/tests)
4. `docs: update LORA plan and adapter help for Silk consumption` (if needed; may fold into 3)
5. `docs: remove TODO.md — plan executed` (after G2 clears)

Commits may be split further if logical units become large (e.g. runner Qwen/SmolLM separately, validation tests separately). One logical change per commit.

## Verification

- Before each commit: `dotnet build Nivara.slnx -c Release` (or targeted project builds), `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` as appropriate.
- Ask before running `dotnet test`. Targeted tests first: new Silk/NivaraChat tests; then `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~LoraAdapterLoader|FullyQualifiedName~SilkLora|FullyQualifiedName~NivaraChat"`. GPU tests only if explicitly confirmed and device available.
- PR test plan: `--qwen plain --silk --adapter <path>` and `--smollm plain --silk --adapter <path>` smoke with a valid bundle; invalid bundle cases (rank/alpha/count/shape/data-length) give clear errors; no RAG regression checks.
- Do not run full suite unprompted.

## GitHub issues log

- [ ] #1429 — Silk Qwen/SmolLM inference: support LoRA adapter bundle consumption (this plan)

## Deferred / follow-ups

- Create GitHub issues immediately if discovered during execution. Candidates to watch: Silk `--merge-adapter` parity, fused LoRA decode kernels for perf, broader GPU-gated CI coverage for adapter consumption.
