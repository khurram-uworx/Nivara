# Plan: reconcile ARROW review/roadmap docs with current Arrow implementation (issue #857)

## Problem

`docs/plan/ARROW-REVIEW.md` and `docs/plan/ARROW-ROADMAP.md` still describe the byte-wise validity
helper as not wired in, even though `ArrowValidityMask` is now wired into `ArrowInterop` for flat
primitive layouts, sliced arrays and the series path (issues #833, #840, #843). ADR-007 has a
matching contradiction in its negative/limits section. Export-only zero-copy is otherwise already
documented correctly.

## Proposed changes

1. `docs/plan/ARROW-REVIEW.md` Pillar 2 ratified paragraph and Pillar 2 scorecard row: helper is
   wired in for flat primitives; packing remains element-wise for variable-binary (`string`/`char`).
2. `docs/adr/007-byte-wise-null-mask-fast-path.md` negative/limits bullet: same correction.
3. `docs/plan/ARROW-ROADMAP.md` Phase C heading and conversion paragraph: boundary conversion
   landed; variable-binary remains element-wise. Keep unimplemented items explicit.

## Verification steps

- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`

## Planned commits

1. docs: plan issue #857 in TODO.md
2. docs: reconcile Arrow validity wiring state in review, ADR, roadmap

## Blast radius

Documentation-only. No public API, tests, or pins touched (no citations added/removed).

## GitHub issues log

- (none created)
