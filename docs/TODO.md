# Plan: issue #1163 — breaking changes documented in CHANGELOG

## Problem

Nivara shipped 1.5.0 as a minor release that removed/renamed public types with no
`[Obsolete]` deprecation path and no machine-readable migration notice. The project is
pre-2.0 and early-stage, so breaks are acceptable, but they must be enumerated with
migration snippets. Scope for this branch: documentation only — a "Breaking changes
(migration)" section in CHANGELOG 1.5.0. No shims, no versioning-policy edits, no API
gate (those are deferred).

## Proposed changes

- `CHANGELOG.md`: add `### Breaking changes (migration)` subsection to the 1.5.0 entry,
  listing each removal/rename with before/after snippets:
  - `DataFrameSchemaValidationException` → `QuerySchemaValidationException` / `DataSchemaValidationException`
  - `SchemaValidationException` split → `QuerySchemaValidationException` / `DataSchemaValidationException`
  - `NivaraColumn` (non-generic factory) → `NivaraColumnFactory`
  - Legacy `ExpressionEvaluator` → `FusedExpressionEvaluator`
  - Removed AutoDiff exceptions → `AutoGradException` / standard exceptions
  - Array-based tensor conversions → span-based `TensorInteropExtensions`
  - ILGPU → Silk/OpenCL backend

## Verification

- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- No fenced ```csharp blocks added, so no snippet-gate count changes.

## Planned commits

1. docs: plan #1163 in TODO.md
2. docs: add breaking changes (migration) section to CHANGELOG 1.5.0
3. docs: remove TODO.md — plan executed

## GitHub issues log

- (none created yet)
