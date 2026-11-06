# Plan: docs polish for NivaraFineTuning Silk story (#949)

## Problem

`#949` is the umbrella issue for the Silk/OpenCL backend story in
`NivaraFineTuning`. Its training/eval/predict wiring landed via `#952`/`#953`/`#956`,
and `#954` updated `docs/SILK.md` and the sample README's `--silk` usage block.
One stale sentence remains in `samples/NivaraFineTuning/README.md` (Expected timing):
"Fine-tuning runs entirely in managed C# on CPU", which contradicts the now-shipped
`--silk` device path. This plan closes that gap so the umbrella's documentation
acceptance criterion is fully met.

## Proposed changes

1. `samples/NivaraFineTuning/README.md` Expected timing: rewrite the opening sentence
   to state that training defaults to managed C# on CPU and that `--silk` runs
   forward/backward/AdamW on the OpenCL device (F32-only, exits 1 if unavailable),
   keeping the CPU steady-state figure and its measurement caveat.
2. No source, test, or citation changes. No new fenced code blocks, so doc coverage
   pins and the snippet gate are unaffected.
3. Close umbrella issue `#949` once the docs polish lands.

## Verification

- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` (human-run)
- `dotnet build samples/NivaraFineTuning/NivaraFineTuning.csproj -c Release`

## Planned commits

1. `docs: plan #949 in TODO.md`
2. `docs: clarify CPU vs Silk training path in NivaraFineTuning README`
3. `docs: remove TODO.md — plan executed`

## Blast radius

Docs only. Single sentence in one sample README. No public API, no tests beyond the
unchanged doc coverage pins.

## GitHub issues log

- [ ] #949 — umbrella: Silk backend story for NivaraFineTuning (closing after this polish)
- [x] #952 — read-only Silk/OpenCL inference (landed)
- [x] #953 — Silk/OpenCL training seam (landed)
- [x] #954 — docs: update Silk story (landed)
