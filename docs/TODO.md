# Issue #1228 — docs/LAYA.md native loading + temperature rounding + README precision row

## Problem

`docs/plan/LAYA-HONEST-WEIGHTS.md` §8 asked for several doc updates that were partially applied in PR #1224:

- `docs/LAYA.md` model-overview line updated and load-path line updated, but no "Native loading" section mirroring the SmolLM/native discussion (storage dtype = snapshot dtype, compute = F32, widening in NarrowFloatKernels, temperature rounding note, Silk asymmetry).
- `docs/LAYA.md` §"Temperature is not in the head" does not record the Half rounding of the 0.1006-style values (~1e-4 relative) for the `temperature [3]` buffer that is widened in the single-dtype map.
- `docs/LAYA.md` performance table has no native vs F32 row (intentionally left for a human-run sweep — keep that).
- `samples/NivaraInference/README.md` precision table row for `laya` not updated to reflect the new native default.

## Proposed changes

1. `docs/LAYA.md` — add a `## Native loading` section (after `## Performance`, before `## GPU path`) covering storage dtype = snapshot dtype, compute = F32, in-register widening, temperature rounding note, Silk asymmetry.
2. `docs/LAYA.md` — append Half-rounding note to the "Temperature is not in the head" bullet.
3. `samples/NivaraInference/README.md` — update the Supported models table row for Laya to reflect native default.
4. Run docs tests; revise `DocsCoveragePins` only if reference counts move (none expected — no new citations or doc links).

## Verification steps

- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` (snippet + coverage gate)

## Planned commits

1. `docs: plan issue #1228 in TODO.md`
2. `docs: add Native loading section + temperature rounding note to LAYA.md`
3. `docs: reflect laya native default in README supported-models row`
4. `docs: remove TODO.md — plan executed`

## Blast radius

- docs/LAYA.md (documentation only)
- samples/NivaraInference/README.md (documentation only)
- No source changes; no test changes expected.

## GitHub issues log

- (none created yet)
