# Plan: perf(silk) top-N partial training probe coverage (#1104)

## Problem
Follow-up to #1012 / PR #1101. Current evidence only covers head-only (0 trainable layers)
vs all-layers. Need probe runs for 1, 2, 4 trainable top layers, capture backward ms, and
update SILK.md reduction table and profile command examples.

## Proposed changes
- docs/SILK.md: replace head-only/all-layers reduction table with backward-chain ms by
  trainable top-layer count (0/1/2/4/all) plus reduction % vs all-layers; add profile
  command examples using SILK_TRAINABLE_LAYERS.
- No source code change required: SilkDistilBertTrainingProfileProbe already honours
  SILK_TRAINABLE_LAYERS.

## Measured numbers (Release, Intel(R) Graphics, DistilBERT base 6 layers, L=128, 5 iters)
Backward chain ms:
- all (6): B2 383.2, B4 734.8, B8 1495.0
- 4 layers: B2 258.1, B4 478.9, B8 984.7
- 2 layers: B2 127.4, B4 231.5, B8 461.6
- 1 layer:  B2 67.6,  B4 109.9, B8 214.0
- head-only (0): B2 0.7, B4 0.9, B8 0.9

## Verification steps
- dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore
- Docs snippet gate (docs namespace) if docs changed
- Targeted tests not applicable (docs-only)

## Planned commits
1. docs: plan #1104 in TODO.md
2. docs(silk): record top-N partial training backward reduction and profile commands

## Blast radius
- docs/SILK.md only. No code, tests, or pins affected (no new citations or fenced csharp blocks).

## GitHub issues log
- (none yet)
