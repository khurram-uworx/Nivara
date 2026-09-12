# Nivara.PerformanceTests

Console benchmark harness for the **storage consolidation** (single
`ColumnStorage<T>`) that ran as Task 7 of the storage plan (plan archived in
git history). It doubles as the perf gate for the AutoDiff refactor (span-based
`GradKernels`, inference-default `GradientUtils.Grad()`). It is a plain
stopwatch harness — no BDN dependency — so it runs portably anywhere `dotnet`
is available.

## Scenarios

| Scenario | What it measures |
|---|---|
| `ColumnAdd 1M x float` | `NivaraColumn<float>.Add(NivaraColumn<float>)` — the columnar binary-op path |
| `ColumnSigmoid 1M x float` | Raw kernel — `TensorPrimitives.Sigmoid` over a pre-allocated 1M destination (the `NivaraColumn<float>.Sigmoid()` extension was removed in Task 8 of the refactor) |
| `Span chain 1M x 3 ops (raw)` | `TensorPrimitives.Add`/`Multiply`/`Subtract` into three pre-allocated 1M destinations — zero-allocation control for the wrapper-cost isolation (P3) |
| `Column chain 1M x 3 ops (wrapper)` | Same three ops through `NivaraColumn<float>.Add`/`Multiply`/`Subtract`, which allocate a fresh result column per op — isolates the column+storage wrapper cost (P3) |
| `Fused chain 1M x (Salary*1.1)+1000-Tax` | The fused-evaluator compiled target for `Col("Salary") * 1.1 + 1000 - Col("Tax")` at a vectorized length (gates on the `KernelSelector` heuristic) |
| `Linear forward [32x256] -> [32x256]` | `Linear<float>` inference forward (no `Grad()` scope) |
| `Linear forward+backward [32x256]` | `Linear<float>` forward + `Backward` inside `GradientUtils.Grad()` |
| `TransformerBlock forward [32x64, 4 heads]` | `TransformerBlock<float>` inference forward |
| `Attn per-seq forward [B16 L128 D64 H4]` | `ReverseGradOperations.MultiHeadAttention` looped over 16 sequences — per-head `Slice`/`Transpose` graph nodes, causal mask per sequence |
| `Attn batched forward [B16 L128 D64 H4]` | `ReverseGradOperations.BatchedMultiHeadAttention` — heads packed once, fused QK^T/softmax/PV per-head `TensorPrimitives` row kernels (issue #86) |
| `Attn per-seq fwd+bwd [B16 L128 D64 H4]` | Per-seq `MultiHeadAttention` forward + `Backward` inside `GradientUtils.Grad()` |
| `Attn batched fwd+bwd [B16 L128 D64 H4]` | `BatchedMultiHeadAttention` forward + `Backward` inside `GradientUtils.Grad()` |
| `RowScore per-row copy+dot [10k x 128]` | Status-quo row scoring — per row, copy 128 column values into scratch then `TensorPrimitives.Dot` (10k dots) |
| `Frame RowDot [10k x 128]` | Public `NivaraFrame.RowDot` — row-major materialization + `TensorsHelper.RowDot` (#138, #141) |
| `Frame Slice [10k x 128]` | Public `NivaraFrame.Slice(0, 5000)` — the reflection-free `IColumn.Slice` path (#173) |
| `RowDot kernel raw [10k x 128]` | Raw `TensorsHelper.RowDot` over a pre-built row-major buffer + null mask — the kernel floor (#141) |
| `RowCosineSimilarity kernel raw [10k x 128]` | Raw `TensorsHelper.RowCosineSimilarity` over a pre-built row-major buffer — kernel floor with norm (#141) |
| `Streaming cancel mid-stream 200k rows x 10k chunk` | Phase 4 AC2 probe (#266): `StreamingExecutionStrategy.ExecuteAsync` over a chunk-capable source, cancelled after ~3 chunks. Asserts a clean `OperationCanceledException` with prompt unwind (#280 fixed — the consumer-side catch now uses `TryComplete()`, observes the producer, and disposes in-flight/channel-buffered frames, so the OCE is no longer masked by `ChannelClosedException`). B/op captures the frames the cancelled path disposes |

Each scenario reports **ops/s**, **ns/op**, **bytes/op** (`GC.GetAllocatedBytesForCurrentThread`
delta), and **gen0/op** (`GC.CollectionCount(0)` delta).

## Running

```pwsh
dotnet run --project tests/Nivara.PerformanceTests -c Release
# or, without a restore:
tests/Nivara.PerformanceTests/bin/Release/net10.0/Nivara.PerformanceTests.exe
```

### On-demand helper modes

Two opt-in flags run standalone checks that are not part of the scenario table or the
no-regression gate — reach for them while working on the relevant area instead of building
a throwaway harness:

- `--dataset-test` — DatasetGenerator determinism/row-count/field-range validator
  (IncidentLab data sets).
- `--safetensors-mmap [<path>]` — A/B of the safetensors string-path load (#392):
  memory-mapped `SafeTensorsLoader.Read(path)` vs copy-into-`byte[]`
  `SafeTensorsLoader.Read(File.ReadAllBytes(path))`. Reports per-load ms, sampled
  managed-heap high-water (`GC.GetTotalMemory`), and retained-after-GC over 3 alternating
  rounds. Defaults to `samples/data/qwen2.5-0.5b-instruct/model.safetensors` when no path
  is given.

### No-regression gate (P4)

The harness doubles as an executable perf gate (`ADR-002` P4). Two modes:

- `--json <path>` — emit each scenario's `ops/s`, `ns/op`, `B/op`, `gen0/op`
  as JSON (median across `n` separate child-process runs via `--runs n`,
  default 1).
- `--compare <baseline.json>` — run, compare against a saved `--json` baseline,
  and exit non-zero when any scenario regresses beyond tolerance.

`--runs n` spawns `n` independent child processes (each a single cold pass via
`--runs 1`) and takes the per-scenario median of their JSON reports. This is
deliberate: an in-process repeat loop is skewed by JIT tiering (later passes
run warmed code — TransformerBlock read 1,256 ops/s in-process vs ~130 honest
across processes), so all `--runs > 1` baselines recorded before commit
`e3ac8b7` must be re-verified with the fixed harness.

| Flag | Default | Meaning |
|---|---|---|
| `--json <path>` | — | write results JSON to `<path>` |
| `--compare <baseline.json>` | — | gate against `<baseline.json>`; exit 1 on regression, 2 on unreadable baseline |
| `--runs <n>` | 1 | spawn `n` independent single-pass child processes and take the per-scenario median |
| `--only <substring>` | all | measure only scenarios whose name contains `<substring>` (case-insensitive) — quick targeted gates, e.g. `--only Qwen` |
| `--tolerance <pct>` | 90 | ops/s floor as a percent of baseline for stable rows only (bandwidth-bound rows use a fixed 25% floor, see below) |

Gate criteria (tolerance constants in `GateEvaluator.cs`):
- **Stable rows:** `ops/s` ≥ `--tolerance`% of baseline (default 90%)
- **Bandwidth-bound rows** (name starts with `"Qwen "` — single-row GEMV /
  memory-streaming kernels at the ~30 GB/s DRAM ceiling): `ops/s` ≥ 25% of
  baseline (fixed floor, issue #420). These rows read ~2.5–3× *slower* under
  machine load while B/op stays byte-stable, which swamped the hard 90% floor
  and failed every Qwen row spuriously for real perf changes ~10% in size.
- `B/op` ≤ baseline × 1.01 (all rows — allocation slack absorbs run-to-run jitter)
- `gen0/op` ≤ baseline + 0.05 (all rows — GC scheduling is not allocation-proportional)

Bandwidth-bound classification is by name prefix (`"Qwen "`), matching the 16
committed Qwen gate rows exactly. It is the stable key between baseline and
compare. Only the `ops/s` leg is relaxed for these rows — `B/op` and `gen0/op`
stay strict for every row, so the gate still catches allocation regressions.

Per-phase workflow (on an idle machine — see the load caveat below):
1. **Baseline** before the phase: `--json baseline.json --runs 3`
2. **Measure** after the phase: `--compare baseline.json --runs 3`
3. `--compare` exits 0 on pass; on FAIL, bisect to the offending change before
   proceeding (ADR-002 no-regression gate).

**Targeted gates:** when a phase touches only one surface (Qwen inference rows,
the AutoDiff kernels, etc.), scope both runs with `--only <substring>` — e.g.
`--only Qwen` measures just the Qwen rows (~1–2 min vs the full cold harness),
and `--compare` ignores every other scenario. Baseline and compare must use the
same filter on the same machine (the committed `qwen-*.json` pairs in this
folder are qwen-only gates).

## Methodology

- **No forced GC** in measurements; steady-state warmup (5 iterations) before
  timing so JIT/type-init effects settle before the baseline is taken.
- **Allocation accounting** starts after warmup, so setup allocations (module
  and column construction) are excluded.
- Compare **on the same machine/config**; use `--runs 3` (three independent
  child processes, per-scenario median) rather than re-running in-process —
  in-process repeats are JIT-tiering-skewed (see the `--runs` note above) and
  run-to-run variance is ~±10% for these scenarios under load.

### Baseline policy (release-by-release rolling history)

- The **Results** table is a release-by-release rolling track on one machine:
  the **Prev** column holds the most recent prior release's reading, the
  **Current** column holds this release's fresh measurement, and **Δ%** is
  the this-vs-last delta: `((Current − Prev) / Prev) × 100`.
- When measuring a new release: shift the existing Current to Prev, place the
  new numbers in Current, and recompute Δ%. The previous Prev is discarded —
  it is superseded by the new Prev. History before that lives in git.
- New scenarios with no prior reading: leave Prev and Δ% blank (e.g.
  `Row.Where nullable-element GetValue 100k`).
- If the Previous reading was on a **different machine**, note the machine
  difference in the Prev column — the delta is not meaningful across machines.
- **B/op** and **gen0/op** are stability indicators (not throughput metrics)
  and are copied alongside, unchanged, as the allocation-driven regression
  signal.

## Results

*Recorded 2026-08-30 — Intel Core Ultra 7 255H, 16 logical processors, .NET 11.0.0 (Release). Medians of 3 child processes (`--runs 3`).*

Machine: Intel Core Ultra 7 255H, 16 logical processors, x64, .NET 11.0.0 (Release). Medians of 3 child processes (`--runs 3`).

| Scenario | Prev | Current | Δ% | B/op | gen0/op |
|---|---|---|---|---|---|
| ColumnAdd 1M x float | 1,515 | 1,684 | +11.2% | 4,000,192 | 0.24 |
| ColumnSigmoid 1M x float | 625 | 993 | +58.9% | 0 | 0.00 |
| Span chain 1M x 3 ops (raw) | 934 | 994 | +6.4% | 0 | 0.00 |
| Column chain 1M x 3 ops (wrapper) | 324 | 323 | −0.3% | 12,000,416 | 0.34 |
| Fused chain 1M x (Salary\*1.1)+1000-Tax | 284 | 278 | −2.1% | 16,005,408 | 0.34 |
| Fused chain chunked 1M x 64k rows | 240 | 240 | 0.0% | 16,005,408 | 0.34 |
| Fused single-op TP 1M x (Salary\*1.1) | 479 | 674 | +40.7% | 8,002,986 | 0.24 |
| Column mul-scalar 1M (wrapper) | 555 | 642 | +15.7% | 8,000,272 | 0.22 |
| Linear forward [32x256] -> [32x256] | 960 | 1,363 | +42.0% | 69,122 | 0.00 |
| Linear forward+backward [32x256] | 124 | 227 | +83.1% | 668,974 | 0.10 |
| TransformerBlock forward [32x64, 4 heads] | 118 | 284 | +140.7% | 186,457 | 0.00 |
| Attn per-seq forward [B16 L128 D64 H4] | 91 | 68 | −25.3% | 2,126,467 | 0.17 |
| Attn batched forward [B16 L128 D64 H4] | 338 | 410 | +21.3% | 528,637 | 0.00 |
| Attn per-seq fwd+bwd [B16 L128 D64 H4] | 25 | 29 | +16.0% | 7,935,987 | 0.42 |
| Attn batched fwd+bwd [B16 L128 D64 H4] | 118 | 110 | −6.8% | 7,875,807 | 0.42 |
| RowScore per-row copy+dot [10k x 128] | 114 | 140 | +22.8% | 2 | 0.00 |
| Frame RowDot [10k x 128] | 357 | 496 | +38.9% | 51,706 | 0.00 |
| Frame Slice [10k x 128] | 14,996 | 7,091 | −52.7% | 89,942 | 0.02 |
| RowDot kernel raw [10k x 128] | 1,161 | 1,226 | +5.6% | 1 | 0.00 |
| RowCosineSimilarity kernel raw [10k x 128] | 246 | 429 | +74.4% | 1 | 0.00 |
| RollingSum null-free 1M x int (w10) | 520 | 526 | +1.2% | 5,000,137 | 0.10 |
| RollingSum nulls 1M x int (w10) | 82 | 93 | +13.4% | 22,000,233 | 0.26 |
| RankKernel RowNumber 100k x int | 35 | 44 | +25.7% | 1,700,313 | 0.00 |
| GroupBy 1M rows x 1000 keys (typed) | 30 | 29 | −3.3% | 8,906,940 | 0.75 |
| GroupBy 1M rows x 100 string keys (typed) | 17 | 23 | +35.3% | 13,188,494 | 1.05 |
| PartitionedWindow RollingSum 1M x 100 parts | 11 | 18 | +63.6% | 36,216,494 | 2.15 |
| Row.Where nullable-element GetValue 100k | — | 117 | — | 7,316,162 | 0.35 |
| Streaming cancel mid-stream 200k x 10k chunk | 2,871 | 6,152 | +114.3% | 5,587 | 0.07 |
| AutoDiff Pow(2.5) fwd+bwd 1M x float | 68 | 92 | +35.3% | 8,001,874 | 0.10 |
| AutoDiff Pow(2.5) scalar baseline 1M x float | 24 | 38 | +58.3% | 2 | 0.00 |
| AutoDiff RMSNorm fwd+bwd 1M x float | 360 | 405 | +12.5% | 8,002,194 | 0.15 |
| AutoDiff RMSNorm scalar baseline 1M x float | 466 | 764 | +63.9% | 2 | 0.00 |
| Qwen LM head matmul [1x896 @ 151936x896] | 5 | 36 | +620.0% | 5 | 0.00 |
| Qwen FFN gate/up/down x3 | 61 | 529 | +767.2% | 1 | 0.00 |
| Qwen attn Q/K/V/O proj | 640 | 6,658 | +940.3% | 1 | 0.00 |
| Qwen Linear fwd [1x896 -> 2688] | 362 | 5,088 | +1305.5% | 22,769 | 0.00 |
| Qwen LM head fwd [1x896 -> 151936] | 5 | 42 | +740.0% | 608,469 | 0.00 |
| Qwen decode-attn fwd [1x896 @ kvLen=64] | 1,226 | 1,992 | +62.5% | 26,313 | 0.00 |
| Qwen decode-attn fwd [1x896 @ kvLen=128] | 827 | 1,967 | +138.0% | 26,313 | 0.00 |
| Qwen decode-attn fwd [1x896 @ kvLen=256] | 560 | 1,552 | +177.0% | 26,314 | 0.00 |
| Qwen prefill seed [8 tok] | 1.3 | 2.4 | +79.5% | 25,364,843 | 0.83 |
| Qwen prefill seed [16 tok] | 0.6 | 2.1 | +270.6% | 49,691,035 | 1.67 |
| Qwen prefill seed [64 tok] | — | 1.3 | — | 195,780,248 | 0.33 |
| Qwen prefill seed [256 tok] | — | 0.4 | — | 785,702,840 | 0.33 |
| Qwen full fwd [64 tok] | 0.6 | 0.6 | +7.8% | 234,069,039 | 0.50 |
| Qwen full fwd [256 tok] | 0.2 | 0.2 | +0.4% | 1,120,578,724 | 0.67 |

### Notes

- **Qwen decode rows added 2026-09-07 with P0-2 (single-row GEMV fast path).** Prev holds
  the pre-fix readings from `qwen-fast-baseline.json` (same machine, same-day medians);
  Current holds the post-fix medians after `TensorsHelper.MultiplyCore` stopped renting/
  copying/clearing the workspace for `bTransposed && aRows == 1`. LM-head matmul +620%
  (B/op 53 → 5, gen0 0.00 both) and LM-head forward +740%; FFN, attn projections, and the
  `Linear` forward all gain the same fast path. The table's other rows are untouched by
  this change; a full-table refresh was deferred because today's machine was not idle
  (non-Qwen rows read ~40% below the 2026-08-30 baseline under load).
- **Qwen decode-attention rows added 2026-09-08 with P0 (fused GQA single-query
  decode-attention).** Prev holds the pre-fix medians from `qwen-gqa-baseline.json`; Current
  holds the post-fix medians after `LlamaCausalAttention.ForwardCached` gained a zero-copy
  fused decode kernel (`AttentionKernels.DecodeAttention` — no BlockCopy of the cached KV
  prefix, no `GqaRepeatKV` expansion, no `PackHeads` re-layout, no mask alloc; inference-only
  behind `GradientUtils`. B/op collapses to a **flat ~26 KB regardless of kvLen** (the residual
  op-boxing allocs tracked by the P1 fused-decoder-block item) — 21×/41×/81× reduction at
  kvLen 64/128/256 respectively, and the context-proportional growth slope is gone. gen0 stays
  0.00 on all three rows.
- **Qwen prefill rows added 2026-09-09 with the Qwen-fast P0 (batched prompt prefill).**
  Prev holds the loop-based `SeedCache` readings from `qwen-prefill-baseline.json` (one
  `ForwardCached` per prompt token); Current holds the post-fix medians after
  `LlamaForCausalLM<T>.ForwardPrefill` seeds the whole prompt in one `[L, hidden]` forward
  (K/V captured at the pre-repeat rows, last-row logits via the single-row LM head). The
  seed rows drop from L full-model walks (761 ms/8 tok, 1,739 ms/16 tok) to a single walk
  plus attention (424 ms/8 tok, 469 ms/16 tok) — +79.5%/+270.6% ops/s with B/op and
  gen0/op down. The 64/256-token seed rows are NEW (the baseline carried loop rows only to
  L=16 because loop L=64+ ≈ 25+ s/op); their before/after is carried by the E2E split
  timing (prefill 5,566 → 747 ms, docs/QWEN.md → *Making Qwen fast*). The `full fwd` siblings are
  unchanged-code `model.Forward` rows and are flat (B/op byte-identical; the [64] row's
  gen0 0.33 → 0.50 is GC-scheduling noise — ops/s and B/op are flat).
- **This table is the current-machine rolling history.** The Prev column
  carries the numbers recorded 2026-08-21 on .NET 10.0.11; the Current column
  carries the re-measured numbers recorded 2026-08-30 on .NET 11.0.0. B/op
  values are stable across runs (allocation-driven), confirming no regressions.
- **This refresh spans a runtime change (net10.0.11 → net11.0).** The Δ%
  compares across runtimes and is **indicative only** — same policy as
  cross-machine comparisons. Notable shifts (Frame Slice −52.7%,
  TransformerBlock +140.7%, Attn per-seq forward −25.3%) reflect the runtime
  retarget, not a code regression; this measurement re-baselines the
  `--compare` gate on the current build.
- **Frame Slice and AutoDiff RMSNorm fwd+bwd tripped the ops/s floor (90%) on
  an immediate follow-up `--compare` run** (5,366 vs 7,091 and 356 vs 405
  ops/s) with **byte-identical B/op and gen0** — throughput-only noise on the
  preview runtime, not an allocation regression. Issue #354 tracks the
  flakiness; ops/s on these rows is order-of-magnitude per the guidance above.
- **Row.Where nullable-element GetValue 100k** is a new baseline row (no prior
  reading): 116.7 ops/s, **7,316,162 B/op** (~73 B/row) — the FilterByMask
  result-frame construction after the #349 fix removed per-element boxing. It
  now gates in `--compare` instead of printing NEW.
- **B/op and gen0/op are allocation-driven and stable** across runs — they
  are the reliable regression signals for the `--compare` gate.
  ColumnSigmoid and the raw span chain are 0 B/op by construction (destination
  pre-allocated).
- **ops/s are load-sensitive.** Treat ops/s as order-of-magnitude; B/op and
  gen0/op are the reliable signals.

## Release Benchmark

Run this during release prep (step 5 of `RELEASING.md`). No external dependencies
beyond the .NET SDK.

```powershell
dotnet run --project tests/Nivara.PerformanceTests -c Release -- --json <path> --runs 3
```

Save the JSON output (e.g., `baseline-vX.Y.Z.json`) and reference it in the PR.

**Update the Results table:**
1. Shift existing **Current** ops/s values to the **Prev** column (this is the
   prior release's reading — the previous Prev is superseded).
2. Place fresh measurements in the **Current** column.
3. Compute **Δ%** (`((Current − Prev) / Prev) × 100`) — the this-vs-last delta.
4. Keep **B/op** and **gen0/op** as-is (stability indicators).
5. New scenarios with no prior reading: leave Prev/Δ% blank.
6. Update the machine line and recording date at the top of the table.
