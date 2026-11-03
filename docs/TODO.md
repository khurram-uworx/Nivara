# Plan: GPU training foundations for NivaraClassifier via Silk

## Problem
NivaraClassifier trains small text classifiers with reverse-mode autograd on CPU. Silk currently provides inference-only OpenCL kernels. The goal is to lay the foundations for GPU training (f32 first) and expose it through `--silk` on train/predict, while keeping CPU autograd as the reference oracle.

## Scope
- Flag name: `--silk`
- Allowed with `--command train` and `--command predict`
- Predict GPU path only with `--load <nivara-model>` (not `--load-onnx`)
- f32-only first phase
- Linear mode first; conv mode follows after kernel readiness

## Proposed changes
1. Docs/plan (this file).
2. Options/CLI: add `Silk`, parse `--silk`, validate before disk touch, update help/README.
3. New seam interfaces for backend-agnostic classifier GPU training/predict.
4. New `GpuTextClassifier` runtime with OpenCL training primitives and epoch loop.
5. Linear path first; conv second.
6. Materialize CPU Module after training for serialization compatibility.
7. Parity tests and CLI validation tests.

## Verification
- Build Release.
- Targeted tests.
- CPU-vs-GPU parity checks for kernels and small-model training.

## Planned commits
- docs: plan Silk training foundations
- core: CLI plumbing and validation
- foundations: GPU training runtime linear path
- phase 2: conv path
- phase 3: predict routing + parity
- docs: update README/SILK

## GitHub issues log
- (none yet)