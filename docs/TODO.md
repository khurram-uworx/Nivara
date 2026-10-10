# Plan — #478: `--help` still says `--gpu` is distilbert/distilbert_sst/minilm only

Branch: `khurram/478` (off `khurram/536`). PR target: `khurram/536`.

## Problem

The top-level `-h`/`--help` block in `samples/NivaraInference/Program.cs` lists the
`--gpu`-capable models as `distilbert / distilbert_sst / minilm only, this phase`
(`Program.cs:131`). That disagrees with the dispatch switch ~150 lines below it
(`Program.cs:278`), which has a `useGpu` branch for **five** models, and with
`docs/ACCELERATION.md` §1, which publishes GPU tables for `laya` and `modernbert`.

`--help` is the first surface a new user reads, so understating capability on it is a
cheap documentation bug that a reader hits before anything else.

The `F32-only` half of the same help line is still true and stays: the reject path
(`Program.cs:241-245`) is unchanged, and `docs/ACCELERATION.md` §5.2 item 4 records
bf16/fp16 GPU kernels as a deferred promotion-phase decision.

## Evidence (verified)

`switch (modelType)` at `Program.cs:278` has a `useGpu` branch for exactly:

| model | `--gpu` branch | GPU entry points |
|---|---|---|
| `minilm` | `Program.cs:289` | `RunMiniLmGpuInference` / `BenchmarkMiniLmGpu` / `RunMiniLmGpuCompare` |
| `distilbert` | `Program.cs:306` | `RunDistilBertGpuInference` / `BenchmarkDistilBertGpu` / `RunDistilBertGpuCompare` |
| `distilbert_sst` | `Program.cs:321` | `RunDistilBertSstGpuInference` / `BenchmarkDistilBertSstGpu` / `RunDistilBertSstGpuCompare` |
| `laya` | `Program.cs:338` | `Laya.RunGpu` / `RunBenchmarkGpu` / `CompareGpu` |
| `modernbert` | `Program.cs:364` | `ModernBert.RunGpu` / `BenchmarkGpu` / `CompareGpu` |

`mobilenet_v2`, `resnet18`, `smollm`, `qwen` have no `useGpu` branch.

No test asserts the help text (grepped `tests/` for `NivaraInference`, `Run on the OpenCL
GPU`, `GPU (`). The only other `this phase` hits — `Program.cs:243` and `README.md:51` —
are attached to the **F32-only** claim, not to coverage, so they stay.

## Proposed changes

1. `samples/NivaraInference/Program.cs:131` — replace the heading with the five models
   that accept `--gpu` today, drop the stale `this phase` qualifier (the next help line
   already states F32-only):

   ```diff
   -            Console.WriteLine("GPU (distilbert / distilbert_sst / minilm only, this phase):");
   +            Console.WriteLine("GPU (minilm / distilbert / distilbert_sst / modernbert / laya):");
   ```

   Order matches the usage line (`Program.cs:118`) and the README quick-start GPU block
   (`README.md:52-56`). Lines 132-133 (the `--gpu` description + F32-only reject note) are
   untouched.

2. `CHANGELOG.md` — one entry under Unreleased **Fixed** (line 92), matching the style of
   #474's entry (`CHANGELOG.md:185`): bold title with issue number, then prose.

## Decisions (raised at G1, resolved with the human)

- **Drop `this phase`** rather than restate the pending bf16/fp16 kernels in the heading —
  the next line already carries F32-only.
- **Model order** = usage-line / README order.
- **Out of scope (red flag, will be filed):** `--gpu` is silently ignored on
  `mobilenet_v2` / `resnet18` / `smollm` / `qwen` — `useGpu` is parsed (`Program.cs:89`) but
  never checked in those cases, so `smollm --gpu` runs on CPU with no warning. Adjacent to
  this help text but not part of #478.
- **README.md:51 stays** — it scopes F32-only, not coverage.

## Blast radius

- `samples/NivaraInference/Program.cs:131` only — one `Console.WriteLine` string in the
  help path. No symbol, signature, or dispatch change.
- `CHANGELOG.md` Unreleased `Fixed` — additive.
- No tests cover the help text; no `.gated` csharp fence involved, so the snippet gate
  (`docs/adr/005`) is unaffected.

## Verification steps

1. `dotnet build Nivara.slnx` — compile check.
2. `dotnet run --project samples/NivaraInference -- -h` — confirm the new heading prints.
   The help path returns `1` by design (`Program.cs:164`), so assert the exit status
   deliberately rather than treating it as failure.
3. No test changes required.

## Planned commits

1. `docs: plan #478 help-text GPU coverage in TODO.md`
2. `fix(samples): list laya and modernbert in the --gpu help heading (#478)`
   — `Program.cs` + `CHANGELOG.md`
3. Any G2 fix as an additive commit
4. `docs: remove TODO.md — plan executed`

## GitHub issues log

- [ ] Filed during execution: `--gpu` silently ignored for mobilenet_v2/resnet18/smollm/qwen
