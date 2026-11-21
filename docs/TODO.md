# TODO — Issue #1395: LoRA adapter injection into LlamaForCausalLM (targets q/v/k/o/ffn) + merge

- Work branch: `khurram/1395`
- Tracking issue: #1395 (open, enhancement, sample; depends on #1394)
- Parent plan: #1394 (mixed-dtype LoRALinear<TModel, TWeight>)

## Problem

The core LoRA module exists in two forms:

- `LoRALinear<T>` wraps a frozen `Linear<T>` base and keeps adapters in `T`.
- `LoRALinear<TModel, TWeight>` wraps a frozen mixed-dtype base `Linear<TModel, TWeight>` with adapters in the compute type `TModel`.

The causal LMs used by the LoRA work (`LlamaForCausalLM<TModel, TWeight>`) store projections as mixed-dtype `Linear<TModel, TWeight>` and expose them through read-only properties:

- `LlamaCausalAttention<TModel, TWeight>.QProj/KProj/VProj/OProj`
- `LlamaDecoderBlock<TModel, TWeight>.GateProj/UpProj/DownProj`

There is still no helper that attaches an adapter to selected projections of a loaded causal LM, selects targets (default attention query/value), or folds trained adapters back into the frozen base weights.

## Proposed changes

1. Add an adapter-injection surface for the causal-LM stack.
   - Attach a `LoRALinear<TModel, TWeight>` around a projection such as `QProj`, `VProj`, `KProj`, `OProj`, or the feed-forward projections.
   - Keep the base projection frozen and keep only adapter matrices trainable.
   - Preserve step-0 parity (`adapted == base`) and adapter-only gradients.

2. Define target selection.
   - Default targets: query and value projections.
   - Optional targets: key, output, and feed-forward projections.
   - Represent flags in a stable, user-friendly form consistent with the CLI/docs (`q,v`, `k,o,ffn`).

3. Add merge support for injected adapters.
   - Fold every adapter into its base weight.
   - After merge, inference should use the merged base weights without adapter overhead.
   - Post-merge logits should match unmerged logits within the agreed tolerance (`1e-5`).

4. Add adapter parameter-count reporting for a SmolLM-shaped model at rank 8 with default targets (expected adapter count around 0.3-0.5M for SmolLM-135M).

5. Integrate with the existing mixed-dtype training path where applicable.

## Acceptance criteria mapping

- Default targets `q/v`; flags allow `k`, `o`, `ffn`.
- Adapter parameter count printed for SmolLM-135M at `r=8` on `q/v` (expect roughly 0.3-0.5M).
- `merge()` folds every adapter; post-merge logits match unmerged within `1e-5`.
- Base remains frozen; only adapter params trainable.

## Verification steps

- Build the solution (`dotnet build Nivara.slnx`).
- Run targeted LoRA / mixed-precision / Llama parity tests.
- Add or extend tests for:
  - adapter injection on the selected Llama projections,
  - frozen-base behavior and adapter-only gradients,
  - merge equivalence and post-merge logits parity,
  - parameter-count reporting shape.
- Run formatting gate (`dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`) before completion if code or docs changed.

## Planned commits

- `docs: plan LoRA adapter injection for issue #1395 in TODO.md`
- Implementation commit for adapter injection and merge support.
- Test commit for adapter injection, merge parity, and parameter-count coverage.
- Docs/coverage follow-up if public API or docs changed.

## Blast radius

- `src/Nivara/AutoDiff/Nn/LoRALinear.cs`
- `src/Nivara/AutoDiff/Nn/Module.cs`
- `src/Nivara/AutoDiff/Nn/Linear.cs`
- `src/Nivara/AutoDiff/Nn/LlamaCausalAttention.cs`
- `src/Nivara/AutoDiff/Nn/LlamaDecoderBlock.cs`
- `samples/Nivara.Samples/LlamaForCausalLM.cs`
- `tests/Nivara.Tests/AutoDiff/LoRALinearTests.cs`
- `tests/Nivara.Tests/AutoDiff/LlamaForCausalLMPrefillTests.cs` and related Llama fixtures
- Potentially `samples/NivaraFineTuning` / `samples/NivaraChat` if adapter consumption paths are touched.

## GitHub issues log

- [ ] #1395 — LoRA adapter injection into LlamaForCausalLM (target selection q/v/k/o/ffn) + merge (in scope)
