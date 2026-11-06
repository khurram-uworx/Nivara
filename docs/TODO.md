# Plan: #953 Silk/OpenCL training seam for NivaraFineTuning (DistilBERT)

## Problem

`NivaraFineTuning --mode train --silk` is explicitly rejected today (`Program.cs` throws:
"--silk is not supported for 'train' mode"). Training stays on CPU AutoDiff. The deferred GPU
trainer seam (#953, mirrors #883 for NivaraClassifier) must provide forward + backward + Adam on
device, with weight materialization into a CPU `DistilBertForSequenceClassification` for
`ModelSerializer` compatibility. Acceptance: `--mode train --silk` completes and dev accuracy
matches the CPU run within tolerance.

## Proposed changes

### A. New OpenCL backward kernels (SilkKernels + SilkLauncher)
- `gelu_backward` (elementwise)
- `layer_norm_backward` (fused residual: LayerNormResidual backward)
- `attention_backward` (dQ/dK/dV for batched multi-head attention w/ padding mask)
- `cls_gather_backward` (scatter-add) — may reuse EmbeddingBackwardInput-style scatter
- `embedding_backward_sum` — split into word/pos scatter + token-type row-0 reduction

### B. New `SilkDistilBertTrainer` (src/Nivara.Extensions/Silk/)
Separate class (not mutate inference runner) to keep per-layer activation storage.
- Weights/moments on device in GPU layout (reuse WeightLayout).
- Cap-sized workspace storing per-layer x/qkv/attn/h/f1 + gradients.
- Forward reuses SilkBertEncoderRunner dispatch order.
- Backward chain: head → CLS gather → 6 layers (attn/FFN/LN) → embeddings.
- Adam on all params incl. embeddings.
- ExportDeviceCheckpoint (device-only) + MaterializeCpuModule helper.

### C. CLI wiring (samples/NivaraFineTuning/Program.cs)
- Remove throw; branch on cli.Silk in RunTrain.
- Epoch loop on device; print loss.
- Save via device checkpoint (primary) with optional materialize.

### D. Parity / tests
- SilkDistilBertTrainerTests: forward parity, loss decreases, finite grads, round trip.
- SilkKernelsCatalogTests updated for new names.
- End-to-end smoke `--max-examples 25 --epochs 1`.

## Verification steps
- dotnet build Nivara.slnx
- targeted tests: Nivara.Tests Gpu
- parity vs CPU on small slice

## Planned commits
1. Add gelu_backward + layer_norm_backward kernels + launcher + tests.
2. Add attention_backward kernel(s) + launcher + CPU-replay tests.
3. Add SilkDistilBertTrainer forward + activation storage; forward parity.
4. Add backward chain + Adam; loss-decreasing test.
5. Add device checkpoint save/load.
6. Wire --silk into RunTrain; update help/args.
7. Parity gate + docs.

## Blast radius
- src/Nivara.Extensions/Silk/SilkKernels.cs (Source + constants + KernelNames)
- src/Nivara.Extensions/Silk/SilkLauncher.cs (new methods)
- new SilkDistilBertTrainer.cs
- samples/NivaraFineTuning/Program.cs
- tests/Nivara.Tests/Gpu/*
- docs/SILK.md (remove CPU-only note for FineTuning training)

## GitHub issues log
- [ ] #956 — full device backward for SilkDistilBertTrainer (follow-up to #953)
