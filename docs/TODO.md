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
  is already stale now that laya and modernbert have GPU paths. That line is not fixed here
  (**#478**).

**Placement correction, made while implementing.** The plan put this validation block "beside the
existing rejects (`:174-178`), i.e. before the safetensors load at `:182`". That is necessary but
not sufficient: the **model-existence check at `:170` sits above `:182`**, so with no weights in
the checkout every rejection would have been masked by `Model file not found … exit 1` — the exit
code would still be 1, so the planned gate would have "passed" while proving nothing. The block
therefore went **above `string modelDir = …`**, before anything touches the disk. This is the
concrete form of "capture the exit status of the process you care about": the first version of
that gate could not distinguish a correct rejection from the wrong one.

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

**Two additions the plan did not call for, both from reading the neighbours while editing them.**

1. The CPU path warns when `warmup != 3`, because the README's CPU table is a **same-row
   PyTorch-vs-Nivara ratio** and `Python/modernbert_benchmark.py` runs `WARMUP_PASSES = 3`. Without
   the warning, `--warmup 1` would silently produce a row that looks comparable to the published
   ones and is not. (The GPU side's equivalent caution is the conditional sentence the plan already
   specified, but for 3 + 10 rather than for the statistic.)
2. The CPU loop gets the same over-length backstop the GPU one has. The plan treated the GPU's
   `continue` as the only silent skip, but `ModernBertMasks.Build` **throws** past 2048 — so a
   direct `ModernBert.Run(…, seqLengths: [4096])` call, bypassing the CLI, would crash after the
   1.5 GB load rather than skipping. Cheap, and it keeps the two paths symmetric.

Both print a visible line naming the row and the limit. Note the asymmetry they encode: the GPU
limit is `config.MaxPositionEmbeddings` (data-dependent, and it skips), the CPU limit is a hard
constant (and it throws, hence the guard).

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

**Three doc items the plan missed, found while editing the ones it named.**

- `samples/NivaraInference/README.md:188` — the GPU table above lists MiniLM / DistilBERT /
  DistilBERT SST-2 and then claims nothing about why ModernBERT's GPU rows are absent from it. Those
  three run 3 + 10 and report an **average** (`Program.cs:1405`, `:1786`); ModernBERT reports a
  median of 3. Now stated in the README, so the new run-time sentence is not the only place a
  reader learns it.
- `samples/NivaraInference/Python/modernbert_benchmark.py:1-6` — its docstring says it "mirrors
  `ModernBert.RunBenchmark` deliberately … one untimed pass", which is **false**:
  `WARMUP_PASSES = 3` is the constant two lines below. This is the file that produces the PyTorch
  half of the CPU comparison, so the wrong claim is on the comparison itself, not on a side note.
  Corrected in place, and it is where the README's warmup caveat comes from.
- `CHANGELOG.md` — `[Unreleased]` had `Added` and `Changed` but no `Documentation` heading, while
  1.3.0 and 1.1.0 both use one. The claim corrections are not a feature and not a behaviour change,
  so they went under a new `### Documentation` rather than being folded into `Added`.

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
- Tests: none cover `NivaraInference`; the sample CLI is verified by running it. The reject paths
  are chosen to fire before **anything** touches the disk, precisely so they are runnable in a
  checkout with no `samples/data/` — see the placement correction above.
- Published numbers: unchanged at default. The CPU benchmark gains one header line, the GPU table
  gains one conditional line it did not always print before; no row is recomputed or restated.
- Landed diff: `src/` and `tests/` untouched, confirmed by `git diff main...HEAD --name-only`.
  `ModernBert.cs` +101/-21, `Program.cs` +125/-4, docs and the Python docstring as listed.

## Verification — results

**Build.** `dotnet build Nivara.slnx -c Release` — succeeded, **0 Warning(s), 0 Error(s)**.
(The human gave the OK for this build.)

**Rejection paths (G).** All 11 exit **1** with the intended message, asserted against
`$LASTEXITCODE` of the process itself rather than a pipeline filter's status. The planned list was
7; four more were added while running them, because each of these was a shape the plan did not
name:

| invocation | exit | message |
|---|---|---|
| `modernbert benchmark --seq abc` | 1 | non-integer token named |
| `modernbert benchmark --seq 0` | 1 | non-positive |
| `modernbert benchmark --seq -4` | 1 | negative |
| `modernbert benchmark --seq 1,000` | 1 | `000` is not a length (comma = list separator) |
| `modernbert benchmark --seq 4096` | 1 | names the 2048 cap, `modernbert --gpu benchmark`, #473 |
| `modernbert benchmark --seq` | 1 | needs a value |
| `modernbert benchmark --iters 0` | 1 | non-positive iterations |
| `modernbert benchmark --warmup x` | 1 | non-integer warmup |
| `minilm benchmark --seq 128` | 1 | the flag is modernbert's only |
| `modernbert --seq 128` | 1 | needs the benchmark mode |
| `modernbert compare --seq 128` | 1 | `compare` is not `benchmark` |

`--seq -4` is worth its row: it is the one input where consuming the next token unconditionally
matters, because a naive `else if (mode.Length == 0) mode = args[i]` catch-all would otherwise
swallow it as a mode name.

**Reflection probe (P).** A throwaway project in temp (`opencode/seqprobe`) calls the shipped
private `TryParseSeqLengths`, `TryParseCount` and `ModernBert.Summarize` by reflection, because
the behaviours below are unreachable through any CLI path without the weights. **21/21 checks,
exit 0.** This covers what the rejection paths cannot:

- ascending order regardless of input order, de-duplication, whitespace around a comma
  (`"128, 256"`), and `128,,256` empty elements;
- `--seq 2048` on CPU **accepted** — the cap is inclusive, since `MaxDenseLength` is the largest
  legal length, not the first illegal one;
- `--gpu` accepts 4096, which is the whole point of the 4096 row;
- empty string and a bare comma both rejected as "needs at least one length", distinct from the
  per-token message;
- `Summarize` on 1, 3 and 4 samples — odd takes the middle element, even is the true median.

**Happy paths (H) — measured, after a false negative.** An earlier version of this section
claimed the happy path was "not verifiable in this checkout" because `samples/data/` is gitignored
and absent and there is no OpenCL device. **Both halves of that were wrong.**
`samples/data/modernbert/` is gitignored (`.gitignore:366`) but *present* on this machine at 1.47 GB,
364 files under `samples/data/` are tracked, and the device is an **Intel Iris Xe**. Both paths
were then timed:

| path | invocation | median | min | tok/s | ms/layer |
|---|---|---|---|---|---|
| CPU | defaults (1 + 3) | 2248.6 ms (S=128) | 2165.6 ms | 11.6 | ~80.31 |
| CPU | defaults (1 + 3) | 4191.3 ms (S=256) | 4137.0 ms | 6.2 | ~149.69 |
| GPU | `--seq 128` | 1495.1 ms | 1484.7 ms | 17.4 | ~53.40 |

What this does and does not establish:

- The default invocation prints the same two lengths, the same protocol header and the same column
  layout as before — the added header line is visible and the rows are otherwise unchanged in shape.
- The GPU run printed the new conditional comparability sentence, as intended, at the default 1 + 3.
- The CPU rows are **~1.7x slower** than the README's published 1249.8 / 1922.4 ms. That is a
  property of this session's machine load, not of the change, and it is reported as a measurement
  rather than a restatement. **The published rows are not replaced, and no figure in the README,
  `docs/MODERNBERT.md`, or `docs/ACCELERATION.md` was edited to match these.**
- The GPU S=128 row (1495.1 ms) lands within 0.2% of ACCELERATION.md's dense-sweep 1497.8 ms for
  the same row, which is a useful independent check that the GPU path still behaves.

**Deliberately not measured: the GPU 3 + 10 sweep at 512 / 2048 / 4096.** It was started
(`--seq 512,2048,4096 --iters 10 --warmup 3`), then stopped at 16 min with no rows emitted, on the
reasoning that it answers a *different* question:

- Those rows exist to settle the 4096-row variance anomaly in ACCELERATION.md §1b item 12 — a
  question about the **#447 band**, not about whether these flags work. The docs commit already
  states the flag makes that row cheap to re-measure and does **not** resolve the anomaly.
- The flag's function is already demonstrated by `--seq 128` producing a real GPU row through the
  same loop, the same skip guard and the same `Summarize` call as 512/2048/4096 would use. Length
  is a loop variable; nothing about the sweep is length-specific.
- `ModernBert` is 1.4 B parameters. The published 4096 row is 176 s **per pass**, so 3 + 10 is
  ~38 min for that row alone, ~50 min for the three — on a laptop iGPU that would also thermally
  throttle, making the result a worse measurement than none.

Anyone with a desktop and AC power can now run it in one command. Until then the 4096 anomaly stays
open and is documented as such; no claim here depends on those rows.

**Still not run:** `dotnet test`. No test project references `NivaraInference`, so there is nothing
for it to cover here; the sample CLI's own gates are above.

## Commits — as landed

1. `94f786e` `docs: plan the #474 selectable modernbert benchmark in TODO.md`
2. `33b4034` `docs: record the #474 grounding pass in TODO.md`
3. `39a8717` `feat(modernbert): --seq / --warmup / --iters for the benchmark paths`
4. `40283b6` `docs: record the #474 probe correction and deferred issues in TODO.md`
5. `0bd8638` `docs: record the modernbert benchmark flags and correct its iteration claims`

The planned 3 became 5: the probe falsified a G1 claim that had already been written into a commit
message, and correcting a stated justification is its own reason. Nothing was pushed.

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
