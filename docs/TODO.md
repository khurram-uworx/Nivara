# Plan: AISTACK reassessment + re-scope (docs + follow-up issues)

**Branch:** `khurram/ai` (off `main`) · **Date:** 2026-11-03
**Scope:** document revision + GitHub issue creation only. No code changes. Phase execution (ONNX export, embedding convention, metrics module) is OUT OF SCOPE and will be handled by separate issues/branches.

## Problem

1. The AI-stack lens was never reassessed on `main` even though several roadmap items partially landed: `TensorDataset`/`DataLoader`, `AsTensorView()`, `NivaraEmbeddingGenerator`, dtype-aware `SafeTensorsLoader`, and the model layer (Bert/ModernBert/Laya/Llama/Qwen/SmolLM).
2. The three open design decisions from the last reassessment are now resolved by user steer consistent with `docs/TENSORS.md` non-compete positioning:
   - **Phase 1 ONNX:** re-scope from "export/import" to **export-only** (train in Nivara → deploy via ML.NET/ONNX Runtime). Import/inference explicitly becomes a non-goal (ML.NET/ONNX Runtime own it, first-party).
   - **Phase 2 embeddings:** re-scope from new "embedding column type" to **flat `[rows, dim]` frame convention** (column of `float[]` + existing kernels). No ARROW Phase B dependency.
   - **Phase 4 metrics:** re-scope to a **minimal frame-scoped metrics module** (frame-producing operations: confusion matrix, per-class accuracy frame), explicitly deferring pipeline-level evaluation to ML.NET.
3. No historical/archeological record of the original plan remains once the roadmap is revised — we want the original plan preserved below the revised plan for archeology.

## Proposed changes

### A. `docs/plan/AISTACK-REVIEW.md`
- Add `**Reassessed 2026-11-03:**` status line (supersedes the proposed 2026-10-23 line; keep both for trail) after the existing status line.
- Scorecard evidence updates for pillars 1, 3, 5, 6 to reflect delivered artifacts.
- Refresh "55 PyTorch-validated functional tests" → describe grown NivaraTorch suite.
- §5 gaps: keep three headline gaps; note SafeTensors import is the delivered half of interchange, ONNX export is the remaining half.

### B. `docs/plan/AISTACK-ROADMAP.md` (archeology-style revision)
Keep the original plan text intact for historical perspective. Restructure each revised phase as:

> **Phase X — Revised plan (2026-11-03)** — short rationale + decision provenance.
> *(original plan follows below, retained for archeology)*

Specifically:
- **Phase 1:** re-scope to ONNX export-only. Non-goal: ONNX import/inference (ML.NET/ONNX Runtime first-party). Keep SafeTensors import as the weight-import path.
- **Phase 2:** flat `[rows, dim]` convention instead of new column kind. Keep MEAI `IEmbeddingGenerator` glue note (NivaraEmbeddingGenerator exists).
- **Phase 3:** keep (dataset layer). Add note that `TensorDataset`/`DataLoader` delivered at training-API level; frame-level split/shuffle/batch still missing.
- **Phase 4:** minimal frame-scoped metrics module; pipeline-level evaluation deferred to ML.NET.
- **Phase 5:** already delivered; keep marker.
- Update "Where we are today", vision bullets, and "Definition of done" to match revised scope (remove "ONNX models import and infer over frames" claim).
- Also add the Phase-1/2 de-scope rationale referencing `docs/TENSORS.md` positioning principle.

### C. `docs/TENSORS.md`
- Reword the "Local .NET AI stack" bullet from "ONNX export/import (train in Nivara, exchange with the ecosystem)" to "ONNX export (train in Nivara, deploy via ML.NET/ONNX Runtime)".
- Keep bullet for embedding columns / dataset / metrics but align wording (metrics: frame-scoped only).

### D. Create three follow-up GitHub issues (executable scope is separate from this session)
1. **Re-scope Phase 1 — ONNX export (train→deploy).** Export trained Nivara modules to `.onnx` (module graph traversal via `StateDict()`, common module set: Linear/Embedding/Conv/BatchNorm/LayerNorm/RMSNorm/TransformerBlock/MultiheadAttention/activations/Sequential), weights + inference-mode ops. Import/inference explicitly out of scope (ML.NET/ONNX Runtime first-party). Target `Nivara.Extensions` (`Nivara.Onnx`), no runtime ONNX dep in core. Parity tests vs. Nivara direct.
2. **Phase 2 convention — embeddings as flat `[rows, dim]` frame.** Documented interop pattern + MEAI `IEmbeddingGenerator` glue (export `GeneratedEmbeddings` ↔ frame helpers), leveraging existing `RowCosineSimilarity`/`TopKDescending`. No new column kind. Null semantics preserved.
3. **Phase 4 — minimal frame-scoped metrics module.** Frame-producing metric operations (confusion matrix frame, per-class accuracy frame, regression summary frame), null-excluding per ADR-001, vectorized on existing `TensorPrimitives` kernels. Pipeline-level evaluation (ML.NET metrics) is a non-goal — documented boundary.

Each issue includes motivation, scope, key files, dependencies, acceptance criteria, risks, and explicit non-goals.

### E. DocsCoveragePins / citation check
- No new docs files → `ExpectedDocumentCount` unchanged.
- Edits only touch existing docs; do not add `File.cs:NN` citations (use symbol form `::Symbol` only if needed) to avoid pin churn. Verify no citation-count delta.
- ADR index unchanged.

## Verification steps (to run / report)
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` (docs-only, run to be safe).
- Grep for "55" / "ONNX export/import" / "embedding column" stale phrasings to ensure consistency across docs.
- `gh issue list` to confirm the three new issues exist and are cross-referenced.

## Planned commits (one logical change per commit)
1. `docs: plan AISTACK reassessment + re-scope in TODO.md`
2. `docs: reassess AISTACK-REVIEW scorecard and add 2026-11-03 status line`
3. `docs: re-scope AISTACK-ROADMAP Phases 1/2/4 (export-only, flat embeddings, frame-scoped metrics) with archeology`
4. `docs: align TENSORS.md AI-stack bullet with export-only ONNX scope`
5. `chore: create follow-up GitHub issues for re-scoped AISTACK phases` (issue refs recorded in TODO log)
6. (optional) `docs: remove TODO.md — plan executed` (only after G2 review clears)

## GitHub issues log (created during execution)
- [x] #866 — Phase 1 re-scoped: ONNX export only (train→deploy) (2026-11-03, this branch)
- [x] #867 — Phase 2 re-scoped: embeddings as flat [rows, dim] frame convention (2026-11-03, this branch)
- [x] #868 — Phase 4 re-scoped: minimal frame-scoped metrics module (2026-11-03, this branch)
- [~] #864 — AISTACK reassessment Steps 1-2 (docs-only; partially subsumed by this branch's reassessment work; left open for tracking)
## Blast radius
- Docs only + 3 new GitHub issues. No source, tests, or public APIs touched. DocsCoveragePins should be unaffected if no new citations are added; will verify at G2.
- Risk: stale phrasing elsewhere (e.g., other docs referencing "ONNX export/import"). Mitigation: grep audit in verification.

## Grounding (G1)
To be done before implementation: microsoft-learn for ML.NET/ONNX interop best practices and TENSORS.md positioning already in-repo. Code-memory: confirm `NivaraEmbeddingGenerator`, `TensorDataset`, `AsTensorView` existence (already verified in prior reassessment discussion).

## Status
Branch created; plan committed next. Implementation proceeds phase by phase after G1 clearance.
EOF
echo written