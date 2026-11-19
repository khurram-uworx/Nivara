# Plan: issue #1274 — NivaraChat core Silk text classifiers remain f32

**Branch:** `khurram/1274` (off `main`)

## Problem

`GpuTextLinearClassifier` and `GpuTextConvClassifier` (`src/Nivara.Extensions/Silk/`) are
float-only: no bf16/fp16 partial constructors mirroring the BERT/Laya pattern. Issue #1274
asks whether to add them.

Decision (consistent with the release constraint in `docs/plan/NIVARACHAT-SILK-GAPS.md`
and the #1265 decision): **do not add narrow partials**. Text classifiers in NivaraChat and
NivaraClassifier are trained and serialized as f32; Silk training is f32-only this release;
the CPU reference models (`TextClassifierModel<float>`, `ConvTextClassifierModel<float>`)
are f32. Narrow storage would need new device kernels + weight loaders for no benefit at
classifier scale and would break the existing f32 parity gate. Instead, document the decision
(f32-only with rationale) and keep the f32 CPU↔Silk parity tests as the assurance.

## Blast radius

- `src/Nivara.Extensions/Silk/GpuTextLinearClassifier.cs` — class XML doc only.
- `src/Nivara.Extensions/Silk/GpuTextConvClassifier.cs` — class XML doc only.
- `src/Nivara.Extensions/Silk/GpuTokenClassifier.cs` — align existing "f32 only" note wording.
- `docs/SILK.md` §13c — add precision decision note.
- `docs/plan/NIVARACHAT-SILK-GAPS.md` — mark Gap S4 resolved (decision documented).
- Tests (verification only, no edits): `tests/Nivara.Tests/Gpu/GpuTextLinearClassifierTests.cs`,
  `tests/Nivara.Tests/Gpu/GpuTextConvClassifierTests.cs`.

No public API, kernels, or weight layouts change. No new dependencies.

## Proposed changes

1. Add a `Precision` note to the class XML doc of `GpuTextLinearClassifier` and
   `GpuTextConvClassifier`: f32-only; rationale (training f32-only this release; CPU
   reference is f32; parity gate is f32).
2. Align `GpuTokenClassifier`'s existing "f32 only" note with the same rationale wording.
3. `docs/SILK.md` §13c: add a "Precision: f32 only (decision, #1274)" bullet — no bf16/fp16
   partials; rationale; parity pinned by the two Gpu test fixtures (forward 2e-4, gradients 5e-4).
4. `docs/plan/NIVARACHAT-SILK-GAPS.md` Gap S4: mark resolved — decision documented; point to
   SILK.md note + tests.

## Verification steps

- Build: `dotnet build Nivara.slnx` (Release).
- Targeted tests (requires OpenCL device): `GpuTextLinearClassifierTests`,
  `GpuTextConvClassifierTests`. If no device, record "not measured (no device)".
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
- Docs coverage pins unaffected (no `docs/*.md` count change from these edits? SILK.md edit
  only; document count unchanged).

## Planned commits

1. `docs: plan #1274 in TODO.md`
2. `docs(silk): record f32-only decision for text classifiers (#1274)` (SILK.md + plan doc)
3. `docs(silk): annotate text/token classifiers as f32-only (#1274)` (XML doc comments)

## GitHub issues log

- (none created this plan; #1274 is the issue being resolved)

## Notes

- No push/PR until human confirms.
- f32 parity tests remain the assurance; no test edits.
