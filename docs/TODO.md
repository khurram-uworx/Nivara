# TODO — #474: make the `modernbert` benchmark's sequence lengths and iteration counts selectable

Branch: `khurram/474` (off `main` @ `956facd`, which already contains #447 / PR #475).
Issue: [#474](https://github.com/khurram-uworx/Nivara/issues/474).

## Problem

`modernbert benchmark` hardcodes its sequence lengths, so one perf row cannot be measured
without paying for every row below it:

- `samples/NivaraInference/ModernBert.cs:589` — GPU: `{128, 512, 2048, 4096}`
- `samples/NivaraInference/ModernBert.cs:168` — CPU: `{128, 256}`

The GPU set runs to 4096 on purpose (it is the evidence that the GPU path has no 2048 cap),
which makes it the most useful table in the sample and the most expensive one to reproduce:
the 4096 row is roughly 1000x the attention work of the 128 row. Working on #447 there was
no way to A/B the attention leg at a single sequence length — the options were to edit the
array (changing the published table) or to run the whole sweep and discard most of it.

The `--seq 256` in the original #447 body is not a real flag. Today `Program.cs`'s arg loop
(`samples/NivaraInference/Program.cs:44-94`) has no `--seq` branch, so the token falls through
to `else if (mode.Length == 0) mode = args[i]`:

- `modernbert --seq 256` sets `mode = "--seq"` and silently runs the default inference;
- `modernbert benchmark --seq 256` drops the token on the floor, because `mode` is already set.

### Findings that shape the change

1. **A false claim is already published.** `samples/NivaraInference/README.md:48` documents
   `modernbert benchmark  # 3 warmup + 10 timed; --seq 256 doubles the length`. Neither half
   is true: there is no `--seq` flag, and the CPU path runs **1 untimed + 3 timed**
   (`ModernBert.cs:172-175`). The Python reference is 3 warmup + 3 timed
   (`samples/NivaraInference/Python/modernbert_benchmark.py:25,62`). README:159 and :173
   repeat the "3-pass warmup + 10 timed passes" methodology for the ModernBERT rows.
2. **Dead const.** `ModernBert.cs:49` `const int BenchmarkMaxLength = 128` has zero
   references repo-wide and does not match the CPU table it appears to describe.
3. **The CPU path has a real bound the flag must respect.** `ModernBertMasks.MaxDenseLength`
   = 2048 (`samples/Nivara.Samples/ModernBertModel.cs:225`) and `Build` throws past it
   (`:238-243`). `modernbert benchmark --seq 4096` would therefore throw *after* the 1.5 GB
   weight load, from deep inside the forward.
4. **The GPU path silently skips.** `ModernBert.cs:591-592` `continue`s above
   `config.MaxPositionEmbeddings`. Inert for the default set (ModernBERT-large is 8192), but
   with an explicit list an out-of-range row would vanish with no output at all.
5. **"Not directly comparable" is about the statistic too, not only the counts.**
   `ModernBert`/`Laya` report median + min; the DistilBERT/MiniLM GPU tables report
   Average / Min / Max (`Program.cs:1284-1305`, `1665-1685`).
6. **Laya needs nothing** — its length is config-driven (`agent.MaxLen`, `Laya.cs:196-207`),
   there is no length sweep.

## Proposed change

```
modernbert benchmark        [--seq N[,N...]] [--warmup N] [--iters N]
modernbert --gpu benchmark  [--seq N[,N...]] [--warmup N] [--iters N]
```

| Flag | Default (CPU) | Default (GPU) |
|---|---|---|
| `--seq` | `128,256` | `128,512,2048,4096` |
| `--warmup` | 1 | 1 |
| `--iters` | 3 | 3 |

`--iters` is the issue's own Note ("may be worth pairing with the iteration count"). `--warmup`
ships with it because `--iters 10` alone cannot express the 3 + 10 protocol the
DistilBERT/MiniLM GPU tables use, which is the comparison the note is for.

### `samples/NivaraInference/Program.cs`

- Arg loop: capture `--seq`, `--warmup`, `--iters`, consuming each value with `i++`.
- One validation block beside the existing rejects (`:174-178`), i.e. **before** the
  safetensors load at `:182`, so a bad value costs seconds rather than a 1.5 GB load:
  - any of the three flags on a non-`modernbert` model -> error;
  - any of them outside `benchmark` mode -> error (mirrors `--plain is only valid with the
    default single-shot run or 'benchmark'`, `:335-339`);
  - `--seq` with a missing value, a non-integer token (message names the token — not
    `int.TryParse`'s silent 0), or `n <= 0` -> error;
  - CPU path with `n > ModernBertMasks.MaxDenseLength` -> error naming the cap, the
    dense-mask reason, `modernbert --gpu benchmark`, and #473. Referenced as the const so
    the message cannot drift from the model;
  - `--warmup` / `--iters` `< 1` -> error;
  - `--seq` sorted ascending, deduped.
- Thread the parsed values into `ModernBert.BenchmarkGpu` and `ModernBert.Run`.
- New help block "ModernBERT benchmark options:", deliberately **not** inside the existing
  GPU block (`:111-114`), whose "(distilbert / distilbert_sst / minilm only, this phase)" line
  is already stale now that laya and modernbert have GPU paths. That line is not fixed here.

### `samples/NivaraInference/ModernBert.cs`

- Delete the dead `BenchmarkMaxLength`; replace with `CpuBenchmarkSeqLengths = [128, 256]` and
  `GpuBenchmarkSeqLengths = [128, 512, 2048, 4096]`, each path iterating
  `requested ?? Default…`, so each table is written down in exactly one place.
- `Run` and `BenchmarkGpu` gain `int[]? seqLengths = null, int warmup = 1, int iterations = 3`.
  `Compare` / `CompareDiag` / `RunGpu` / `CompareGpu` are untouched.
- The warmup pass count becomes `warmup`; the timed count becomes `iterations`.
- Median: a true median (mean of the two middle samples) rather than today's
  `timings[iterations / 2]` upper-middle element. No-op at the default 3, so published numbers
  are untouched; it is here because `--iters 10` makes even counts reachable and the printed
  word "median" should mean median.
- The GPU protocol note (`:585-587`) prints the real counts, and its "fewer than the 3 + 10
  used by the DistilBERT/MiniLM GPU benchmarks … Not directly comparable" sentence becomes
  **conditional** on the counts actually differing from 3 + 10, with the median-vs-average
  difference named so the claim stays true.
- The GPU `continue` becomes a visible line naming the skipped row and
  `config.MaxPositionEmbeddings`.
- The CPU path gains the one-line protocol header Laya already prints (`Laya.cs:199`). This is
  the only change to default output: the rows stay byte-identical, and the header makes
  README:48's iteration claim checkable from a run.

### Docs

- `samples/NivaraInference/README.md:48` — the flags plus the *true* counts (1 + 3); `:228`
  "seq 128 + 256" -> default set + flag note; `:159` / `:173` — the ModernBERT methodology
  note in the "3 warmup + 10 timed" wording.
- `docs/ACCELERATION.md:164` — supersede "`modernbert benchmark` has no `--seq` flag
  (**#474**), so re-measuring one row means paying for the whole table". The 4096-row variance
  anomaly itself stands; only the reason it could not be re-measured changes. Also correct the
  mode name in that clause (`modernbert benchmark` -> `modernbert --gpu benchmark`).
- `docs/MODERNBERT.md` — one line in the benchmark section.
- `CHANGELOG.md` `[Unreleased] -> Added`, following the `--simd-widen` precedent (`:205`).

## Blast radius

- `ModernBert.Run<TModel,TWeight>` and `ModernBert.BenchmarkGpu` are the only signatures
  changed, both by **adding optional parameters** — source-compatible. Callers:
  `Program.cs:299` (`BenchmarkGpu`) and `Program.cs:310-312` (`Run`). No other caller in the
  repo; neither is referenced by `tests/` (Nivara.Tests does not reference the
  `NivaraInference` exe).
- `ModernBert.cs` private helpers: `PadTo` is untouched and already truncates over-long input
  with HuggingFace `truncation=True` semantics, so a `--seq` below the sentence length is
  well-defined and shows up in the row's `(valid N)` column.
- No change under `src/` — no library, kernel, or public API surface. `ModernBertMasks` and
  `ModernBertGpuRunner` are read-only inputs (the cap const and the config).
- Tests: none cover `NivaraInference`; the sample CLI is verified by running it. The reject
  paths are chosen to fire before the weight load precisely so they are runnable in a checkout
  with no `samples/data/`.
- Published numbers: unchanged at default. The CPU benchmark gains one header line; no row is
  recomputed or restated.

## Verification

Runnable in this checkout (no weights, no GPU needed) — each must print its message and
**exit 1**; the process exit status is what gets asserted, not a pipeline filter's:

- `modernbert benchmark --seq abc` — non-integer token
- `modernbert benchmark --seq 0` — non-positive
- `modernbert benchmark --seq 4096` — CPU dense-mask cap
- `modernbert benchmark --seq` — missing value
- `modernbert benchmark --iters 0` — non-positive iterations
- `minilm benchmark --seq 128` — wrong model
- `modernbert --seq 128` — wrong mode

Needs the human's OK to run: `dotnet build samples/NivaraInference -c Release`;
`dotnet test` if wanted.

**Not verifiable in this checkout:** `samples/data/` is gitignored and absent and there is no
OpenCL device, so no measured row (CPU or GPU) can be produced. The happy path will not be
claimed as verified; either the human runs
`dotnet run --project samples/NivaraInference -c Release -- modernbert --gpu benchmark --seq 4096`
and pastes the row, or the PR states the gap.

## Commits

1. `docs: plan the #474 selectable modernbert benchmark in TODO.md`
2. `perf(samples): make the modernbert benchmark sequence lengths and iteration counts selectable`
3. `docs: record the modernbert benchmark flags and correct its iteration claims`

## Grounding (G1)

- `int.TryParse` defaults to `NumberStyles.Integer`: digits, surrounding whitespace and a
  leading sign only — **no group separator**
  (https://learn.microsoft.com/dotnet/standard/base-types/parsing-numeric). So `--seq 1,000`
  is a comma-list (`1`, `000`) and `000` is rejected as non-positive. Deterministic and
  visible; a silent 1000 would be the failure mode worth avoiding.
- The parameterless `int.TryParse(string, out int)` reads the **current culture**, which the
  grounding read as a reason to prefer the explicit overload. **A probe falsified that
  justification and replaced it with a narrower true one.** Measured across the invariant culture
  plus fr-FR, de-DE, ar-SA, hi-IN, fa-IR and sv-SE, and across group separators, no-break and
  narrow no-break spaces, a decimal separator, a Unicode minus, surrounding whitespace, an
  exponent and an Arabic-Indic digit, the two overloads disagreed in exactly two cells:
  `"+12"` is accepted by the explicit form and **rejected** by the parameterless one under `ar-SA`
  and `fa-IR`, whose `NumberFormatInfo.PositiveSign` is `؎+`. So the reason for the explicit
  overload is the **leading sign**, not digit spelling — `NumberStyles.Integer` allows a sign and
  the sign character is culture-defined. Every other input, digits included, behaved identically.
  The `TryParseCount` doc comment now says this.
- Splitting uses `StringSplitOptions.TrimEntries | RemoveEmptyEntries`, so `--seq "128, 256"`
  and `--seq 128,,256` both behave.
- `code-memory` `impact_analysis` on `BenchmarkGpu` reports exactly one downstream caller,
  `Main` in `samples/NivaraInference/Program.cs`. A relationship scan plus an exhaustive grep
  finds the ModernBERT entry points called from `Program.Main:299-312` and nowhere else, and
  no test project references the `NivaraInference` exe. The blast radius above holds.

## GitHub issues log

- [x] #477 — `Python/modernbert_benchmark.py` hardcodes `MAX_LENGTHS` with no `argparse`, so the
      PyTorch half of the CPU comparison cannot be steered to one length the way `--seq` now steers
      the C# half.
- [x] #478 — `--help` still heads its GPU block "distilbert / distilbert_sst / minilm only, this
      phase"; `laya` and `modernbert` have working GPU paths and published tables. (Deliberately
      not fixed here — a different reason from #474.)
- [x] #479 — `--seed` and `--teacher-examples` use the parameterless `int.TryParse` and discard
      the failure, so a typo becomes `0` and the run proceeds. `#474` deliberately does not copy
      this shape.
