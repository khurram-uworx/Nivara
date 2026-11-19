# Plan: ModernBERT head-only fine-tune — Silk (OpenCL) counterpart (#1351)

## Problem

#1280 added a CPU (F32) head-only fine-tune path for ModernBERT. The Silk/OpenCL
counterpart is deferred. SilkPartialTrainingPolicy already defines the freeze
policy and SilkLayaHeadTrainer is the template, but ModernBERT's head is a much
simpler 2-layer MLP on a single [CLS] vector (no sequence attention), so the
trainer can be far smaller than Laya's.

## Proposed changes

### A. Silk weights + trainer (src/Nivara.Extensions/Silk)
- SilkModernBertHeadWeights: holds hidden (hiddenDim->hiddenDim) + classifier
  (hiddenDim->numClasses) weights/biases, gradients, Adam moments, trainable params.
  F32 only this pass.
- SilkModernBertHeadTrainer:
  - Constructor(runtime, hiddenDim, numClasses) allocates buffers.
  - Forward(float[] clsVector) -> float[] logits.
  - Backward(float[] dLogits).
  - AdamUpdate(learningRate, weightDecay).
  - ZeroGradients(), ExportCheckpoint(), Dispose().

### B. Showcase entry point (samples/NivaraInference)
- ModernBert.RunTrainHeadGpu: encode a sentence via SilkModernBertGpuRunner, extract
  [CLS], train head one step, export + reload.
- Wire `modernbert --silk train-head` in Program.cs.

### C. Acceptance
- `modernbert train-head --silk` runs one forward/backward/Adam step on a seeded fixture.
- Encoder stays on device and frozen; only head params update.
- Exported head reloads for inference.
- F32 only.

## Verification
- Build Nivara.Extensions + NivaraInference.
- Run `modernbert train-head --silk` (requires OpenCL device).
- InferenceCliArgumentTests if SilkModes updated.

## Blast radius
- New files in src/Nivara.Extensions/Silk/.
- Program.cs silk branch for modernbert.
- No core library changes.

## GitHub issues log

- (none yet)