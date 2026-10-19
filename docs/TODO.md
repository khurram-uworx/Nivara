# Plan: #719 Silk SmolLM runner latent q/k/v bias support

## Problem

`SilkSmolLMGpuRunner` / `SmolLMConfig` assume a bias-free SmolLM (real SmolLM-135M). A biased
variant would load but biases are ignored → wrong attention. Mirror the already-shipped Qwen
runner bias support, gated on config flag or tensor presence.

## Proposed changes

1. `SmolLMConfig`: add `AttentionBias` (default false).
2. `SilkSmolLMGpuRunner`: `qkvBias` field; `LayerWeights.QBias/KBias/VBias`; conditional
   upload; `AddBias` after Q/K/V Gemms in batched paths; `hasBias: qkvBias` in RmsNormGemv
   paths; dispose releases.
3. `samples/NivaraInference/Program.cs` `ToSilkConfig`: propagate `AttentionBias`.

## Verification

- Build Nivara.Extensions + NivaraInference (Release).
- Run synthetic SmolLM Silk benchmark (bias-free) → unchanged results.
- (Optional) biased synthetic variant parity not available; ensure no crash on bias-free.

## Planned commits

1. docs: plan #719 in TODO.md
2. feat(silk): SmolLM runner q/k/v bias support
3. docs: remove TODO.md — plan executed

## GitHub issues log

- (none new)