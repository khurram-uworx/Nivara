# Plan: Issue #1271 — Silk retrieval training pairwise loss on device

## Problem

`SilkRetrievalTrainer` is listwise-only. In `--rag-learn`, `--silk --loss pairwise` falls back to CPU
(`RagLearnMode.cs` lines ~96-127, ~140-165) and prints "pairwise loss trains on CPU; the Silk fusion
trainer is listwise-only". Goal: implement pairwise loss + gradient on Silk so `--silk --loss pairwise`
runs fully on device, with parity to CPU pairwise within tolerance.

## Proposed changes

### SilkRetrievalTrainer.cs
- Thread `LossKind` through `Train` → `ForwardAndGradient`.
- `ForwardAndGradient` signature: `bool ...(..., LossKind loss, out double trainLoss)` (returns whether a loss existed).
- `ForwardAndGradientGlobal`/`Gated`: after `fusion_forward` produces `f`, branch:
  - Listwise (existing): `g[i] = softmax(f)[i] - target[i]`, `trainLoss = ListwiseLoss`.
  - Pairwise: enumerate grade-ordered pairs; if none → return false (no update). Else accumulate
    `g[better] += (σ(f[b]-f[w]) - 1)/P`, `g[worse] += (1 - σ(...))/P`,
    `trainLoss += stable softplus(f[w]-f[b])`, then `/P`.
  - Upload `g`, run `fusion_wgrad` + softmax reweight + Adam as before.
- Gated path reuses the same `g`/`trainLoss` branch.

### RagLearnMode.cs
- Change both guards from `ctx.UseSilk && options.Loss == LossKind.Listwise` to `ctx.UseSilk`.
- Remove pairwise-CPU message branch; `trainingBackend = "Silk/OpenCL (…)"` on success for both losses.

### Docs
- `docs/SILK.md` §13: state both losses run on device.
- `samples/NivaraChat/README.md`, `samples/NivaraChat/RAG.md`: update pairwise wording.

## Verification steps

- Build: `dotnet build samples/NivaraChat/NivaraChat.csproj` (Release).
- Targeted tests: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkRetrievalTrainerTests"`.
- Runtime parity (GPU available): `--rag-learn --silk --loss pairwise` on a corpus; compare training loss to CPU pairwise; check `Init parity` ~1e-7.
- Snippet gate if docs touched: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~EveryGatedBlock_CompilesWithoutErrors"`.

## Planned commits

1. docs: plan #1271 in TODO.md
2. feat: pairwise loss + gradient on Silk (SilkRetrievalTrainer)
3. feat: allow --silk with pairwise loss in RagLearnMode
4. docs: update Silk/README/RAG pairwise wording
5. test: GPU-gated pairwise training test (optional)
6. docs: remove TODO.md — plan executed

## Blast radius

- `samples/NivaraChat/Helpers/SilkRetrievalTrainer.cs` — training core; ForwardAndGradient signature change. Callers: RagLearnMode (passes options.Loss now), tests.
- `samples/NivaraChat/Modes/RagLearnMode.cs` — mode orchestration.
- Docs: SILK.md, README.md, RAG.md.
- Tests: SilkRetrievalTrainerTests (GPU-gated).

## GitHub issues log

- (none yet)
