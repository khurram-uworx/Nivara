# Plan: Silk docs — series cross-links, move `SILK.md` into `docs/`, CHANGELOG PR number

Branch: `khurram/silk-docs` (off `khurram/silk-minilm`). Final PR targets `khurram/silk-minilm`.

## Problem

Three documentation follow-ups left by the `--silk` MiniLM backend (PR #617), none of which is a
code change:

1. **CHANGELOG has no PR number.** The `--silk` entry's `(#616)` was removed as fabricated (it named
   an unrelated, merged snippet-gate issue). The real identifier is the PR, **#617**.
2. **The GPU-backend "Series" cross-links omit Silk.NET in six documents.** `tests/Nivara.GpuProbe/SILK.md`
   is the reference — its header lists all seven legs — but the other legs do not link back to it, so
   the series is not navigable both ways. (The same lines are also internally inconsistent: three of
   them omit ComputeSharp. See "Decision to confirm".)
3. **`SILK.md` still lives under `tests/Nivara.GpuProbe/`.** Silk is no longer probe-only: the first
   production model (`--silk` MiniLM) landed, so the doc belongs in `docs/`. This is a **relocation
   only** — the document is not restructured to mirror `docs/ILGPU.md` in this change (see below).

## Deliberate scope decisions

- **`SILK.md` is relocated, not rewritten.** It does **not** need to look like `docs/ILGPU.md` yet.
  The two will converge progressively as more kernels and models are added: the model list is how we
  decide which kernels to implement next, so `ILGPU.md`'s per-model entries accrete over time and
  `SILK.md` follows the same way. This change only moves the file and fixes its links.
- **Only what was asked.** Add Silk.NET to the Series lines, put the PR number in the CHANGELOG, move
  the file. No other document restructuring.

## Changes

### 1. Move `SILK.md` to `docs/`

- `git mv tests/Nivara.GpuProbe/SILK.md docs/SILK.md`.
- Rewrite the two series lines inside it (`:3` top, `:348` bottom): probe siblings become
  `../tests/Nivara.GpuProbe/<X>.md`; `[ILGPU](../../docs/ILGPU.md)` becomes `[ILGPU](ILGPU.md)`; the
  self-link stays `SILK.md`. The `[tests/Nivara.GpuProbe/README.md](../tests/Nivara.GpuProbe/README.md)`
  link on `:34` is already correct at the new depth (`docs/` and `tests/` are both repo-root children).
- Update every inbound reference (verified complete by `git grep -n SILK.md`):
  - `tests/Nivara.GpuProbe/README.md` `:20 :21 :29 :50 :240 :472 :638` → `../../docs/SILK.md`
  - `tests/Nivara.GpuProbe/results/552-arc-140t.md:162` → `../../docs/SILK.md`
  - `docs/ACCELERATION.md` `:190 :191 :585` → `docs/SILK.md`
  - `samples/Nivara.Samples/Gpu/SilkRuntime.cs` and `Nivara.Samples.csproj` already say `docs/SILK.md`;
    no change.

### 2. Add Silk.NET to the six Series lines

Canonical order, from `SILK.md`:
`[SPIR-V / Level Zero] · [SYCL / oneAPI] · [DX12] · [OpenVINO] · [ILGPU] · [ComputeSharp] · [Silk.NET OpenCL]`.

Append `[Silk.NET OpenCL](../../docs/SILK.md)` (from the probe) / `(SILK.md)` (from `docs/`) to:

| file | lines |
|---|---|
| `tests/Nivara.GpuProbe/SPIRV.md` | `:3` (top only — the file has no bottom series line) |
| `tests/Nivara.GpuProbe/SYCL.md` | `:3`, `:499` |
| `tests/Nivara.GpuProbe/DX12.md` | `:3`, `:297` |
| `tests/Nivara.GpuProbe/OPENVINO.md` | `:3`, `:314` |
| `tests/Nivara.GpuProbe/COMPUTESHARP.md` | `:3`, `:294` |
| `docs/ILGPU.md` | `:3`, `:264` |

**Decision to confirm:** three of these lines *also* omit ComputeSharp — `SPIRV.md:3`, and the bottom
lines of `SYCL.md`/`DX12.md`/`OPENVINO.md`. Recommended: add ComputeSharp there too, so all seven
lines become identical and the "Series" list actually means one series. Alternative: add only Silk.NET
and leave the pre-existing ComputeSharp gaps untouched (strictly the requested scope).

### 3. CHANGELOG PR number

- `CHANGELOG.md:9`: `... hand-authored OpenCL C** —` → `... hand-authored OpenCL C (#617)** —`.

## Verification

- `dotnet build Nivara.slnx -c Release` — docs-only; expect 0 warnings / 0 errors.
- Targeted `Nivara.Tests` Docs fixtures in Release (`--filter "FullyQualifiedName~Docs"`), and note the
  only two gates that scan `docs/`:
  - **Snippet gate** (`DocSnippet`/`DocumentationSnippetTests`): scans root + everything under `docs/`.
    `SILK.md` contains **no** ```` ```csharp ```` block, so the classification assertion
    (`Coverage_EverySnippetBearingDocumentIsClassified`) and the exact counts
    (`all.Length == 249`, `ungatedCount == 141`) do not move. `docs/TODO.md` is already in
    `ScanExcludedDocuments`, so this plan file does not move them either.
  - **Citation gate** (`DocCitations`/`DocCitationTests`): `SILK.md` contains no `Foo.cs:NN`
    citation, so `CitationCount == 106` and `CitedDocumentCount == 16` do not move.
- Manual: every inbound reference resolves to `docs/SILK.md`; every rewritten series link resolves on
  disk (relative-path check).

## Blast radius

- Documentation only: `CHANGELOG.md`, `docs/ACCELERATION.md`, `docs/ILGPU.md`, `docs/SILK.md` (moved),
  `tests/Nivara.GpuProbe/*.md`. **No** `src/Nivara`, `Nivara.Extensions`, sample code, or test-project
  change.
- The two `docs/`-scanning gates are analysed above and are provably unmoved (no csharp fences, no
  citations). No markdown link-checker exists in `.github/workflows` or `tests/`. `AdrIndexTests`
  covers `docs/adr/` only.
- Downstream readers: `docs/ACCELERATION.md` §5 and the GpuProbe README both point at the doc; both are
  updated in the same change.

## Planned commits

1. `docs: plan silk-docs in TODO.md` (this file).
2. `docs: move SILK.md from GpuProbe into docs/` (the move + inbound-link updates + internal series paths).
3. `docs: link Silk.NET from the GPU-backend series lines` (the six files; plus ComputeSharp if confirmed).
4. `docs: add the PR number to the --silk CHANGELOG entry`.
5. `docs: delete the silk-docs plan` (after G2).

## GitHub issues log

- (none — no deferred work discovered yet; add here if execution surfaces any)
