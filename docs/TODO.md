# TODO — Issue #1396: LoRA adapter-only persistence (save/load A,B + metadata) and round trip

- Work branch: `khurram/1396`
- Tracking issue: #1396 (open, enhancement, AutoDiff/serialization)
- Parent plan: `docs/plan/LORA.md` §3/§9/§10; prerequisite #1394 (closed)

## Problem

The core LoRA modules exist in two forms:

- `LoRALinear<T>` wraps a frozen `Linear<T>` base and keeps adapters in `T`.
- `LoRALinear<TModel, TWeight>` wraps a frozen mixed-dtype base `Linear<TModel, TWeight>` with adapters in the compute type `TModel`.

However, there is no adapter-only save/load path. `ModelSerializer.Save`/`Load` operate on a full module state dictionary and therefore include base weights, while the LoRA contract requires an adapter-only file containing just the `A`/`B` tensors plus metadata.

## Goal

Add an adapter save/load path that persists only the adapter tensors and metadata, and loads them onto a fresh frozen base.

## Proposed changes

1. Define an adapter file/metadata shape for LoRA.
   - Persist only adapter tensors (`A`, `B`) and metadata.
   - Keep metadata explicit and versioned, including rank, alpha, targets, and base model id where applicable.

2. Add adapter save/load support for the LoRA module(s).
   - Save should reject a merged adapter or encode merge state explicitly, per design decision.
   - Load should validate compatibility (rank, base model, and any agreed shape/target checks).

3. Wire loading onto a fresh frozen base without changing the base weights.
   - Support the existing single-dtype and mixed-dtype LoRA forms where practical.

## Acceptance criteria mapping

- Round trip: save adapter, reload on fresh base, identical logits.
- File contains only adapter tensors + metadata (no base weights).
- Rejects mismatched rank/base model on load.
- Tests added in `tests/Nivara.Tests/AutoDiff/LoRALinearTests.cs` style.

## Verification steps

- Build the solution (`dotnet build Nivara.slnx`).
- Run targeted LoRA tests:
  `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~LoRALinearTests"`
- Run formatting gate before completion:
  `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Review branch changes and update `DocsCoveragePins` if required.

## Planned commits

- `docs: plan LoRA adapter persistence for issue #1396 in TODO.md`
- Implementation commit for adapter-only save/load.
- Test commit for round trip, file-content, and mismatch rejection coverage.
- Docs/coverage follow-up if public API or docs changed.

## Blast radius

- `src/Nivara/AutoDiff/Nn/LoRALinear.cs`
- `src/Nivara/AutoDiff/Serialization/ModelSerializer.cs` (or new adapter serializer)
- `src/Nivara/AutoDiff/Nn/Module.cs` (only if shared state-dict changes are needed)
- `tests/Nivara.Tests/AutoDiff/LoRALinearTests.cs`
- Possibly `samples/Nivara.Samples/LlamaForCausalLM.cs` for adapter consumption.

## GitHub issues log

- [ ] #1395 — LoRA adapter injection into `LlamaForCausalLM` (target selection q/v/k/o/ffn) + merge
- [ ] #1396 — LoRA adapter-only persistence (save/load A,B + metadata) and round trip
