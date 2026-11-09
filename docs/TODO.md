# Plan: Issue #1097 — extract removed compare/parity commands into shared helper

## Problem
The visitor-facing `NivaraInference` CLI no longer exposes `compare`/`compare_diag` (issue #1051). Their implementations still live as dead dispatch paths in `samples/NivaraInference/Program.cs` and model files, and the perf harness has no equivalent entry. Docs (`ACCELERATION.md`, `SILK.md`, `MODERNBERT.md`) still reference old sample commands.

## Proposed changes
1. Add shared helper in `Nivara.Samples` (`samples/Nivara.Samples/InferenceCompare.cs`) exposing compare/parity entry points:
   - vision compare/diag (mobilenet_v2, resnet18)
   - ModernBERT compare/diag
   - Laya compare
   - MiniLM/DistilBERT/DistilBERT+SST compare
2. Simplify `samples/NivaraInference/Program.cs` by removing dead compare dispatch arms/variables and calling the helper only where needed.
3. Add perf-harness flags `--inference-compare <model>` and `--inference-compare-diag <model>` invoking the shared helper.
4. Reconcile docs to point to perf-harness commands.
5. Review `DocsCoveragePins` and update pins if needed.

## Verification
- Build solution
- Targeted inference CLI tests
- Docs test namespace if docs changed
- Perf harness command smoke
- Formatting check

## Planned commits
- plan commit
- add shared helper
- simplify Program.cs
- perf harness flags
- docs reconciliation
- pins review

## GitHub issues log
- #1097 — expose removed compare/parity commands in perf harness
