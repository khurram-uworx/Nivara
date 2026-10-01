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
- `--gemm` — tiled-GEMM regression gate (#435): the lasting promotion of the deleted
  `%TEMP%\opencode\gemm-measure\` harness. Loads all ten ILGPU GEMM kernels
  (`samples/Nivara.Samples/Gpu/GemmKernels.cs` — `OneToOne`, `Row4`, the M2 fused
  siblings `Row4Bias`/`Row4Gelu`/`Row4Relu`/`Row4Qkv`, and the four #440 tile geometries
  `Reg2x2K16`/`Reg2x2K32`/`Reg4x2K32`/`Reg1x8K16`) through the public `IlgpuRuntime` and
  runs them over the model GEMM shapes (DistilBERT 768/3072, MiniLM 384/1536, seq-len 128),
  the Laya / ModernBERT-large shapes (d=1024, fused input|gate Wi=5248, ffn=2624, seq-len
  512, plus the 2-layer decision head and the act/scorer tail), and the padded-grid edge
  shapes (non-multiple-of-16 rows/K/cols, plus non-multiple-of-32 cases for the #440 tiles).
  Each (kernel, shape)
  cell is gated `maxAbs(gpu − double-precision truth) ≤ 1e-3` and timed best-of-25
  synchronized launches, reporting GMAC/s; exit code 0 = pass, N = failed cells,
  10 = UNBUILT (no OpenCL GPU). Runs on AC power only — the harness warns on battery
  because the iGPU throttles flat (the correctness leg still runs). Re-run after any
  driver/IGC bump and compare the GMAC/s column (docs/BERT-GPU.md). The baseline below was
  recorded on this machine's first run.

  **The four #440 geometries additionally assert byte-identity to Row4**, and are kept as a
  labelled negative baseline: they were measured 2026-09-29 and every one of them is *slower*
  than Row4 on every shape, because on this iGPU shared-memory capacity per group dominates
  shared-memory traffic per MAC. Nothing routes at them; they stay so the result stays
  reproducible, and so nobody re-attempts the same geometry without reading the baseline
  below — but see the supersession note on that table, because Row4's denominator has since
  moved. The byte-identity assertion is not decorative — it is what caught an ILGPU lowering
  bug (#468) that the `maxAbs` tolerance alone would have argued with.

  **Every cell also asserts its f32 bits against a committed fingerprint**
  (`gemm-f32-baseline.json`, FNV-1a 64 over `BitConverter.SingleToInt32Bits`, keyed
  `variant|shape`, recorded pre-migration so the #468 migration could be proved a no-op).
  Tolerance cannot catch a kernel that is *reliably* wrong inside the band, and these kernels
  promise exactness — same f32 values, same ascending-K order — so the gate asserts the bits:

  ```
  Fingerprint PASS - 171 of 171 baseline cells bit-identical
  ```

  A normal run only ever compares. Re-recording the reference takes an explicit
  `--gemm --write-gemm-baseline`, because a gate that can rewrite its own reference is a gate
  whose green means nothing. A missing baseline file is a **failure**, not a pass: with nothing
  to be exact against, the run verified nothing. The file is keyed by device *and* toolchain
  (`OpenCL <version>, driver <CL_DRIVER_VERSION>`, read from the ICD — `CLDevice.DeviceVersion`
  is the OpenCL version, not the driver), so a driver bump reports one `Fingerprint NOT VERIFIED`
  line naming both toolchains instead of 171 numeric failures that read like a kernel
  regression. Coverage is always reported as "N of M baseline cells", never as a bare count of
  what the run happened to measure.

#### GEMM gate baseline (2026-09-19)

`dotnet run --project tests/Nivara.PerformanceTests -c Release -- --gemm` — Intel Core
Ultra 7 255H (Arc iGPU via OpenCL/ILGPU 1.5.3), AC line. 36 cells, all PASS
(`maxAbs ≤ 1e-3`). Per-shape worst maxAbs and the best kernel's GMAC/s:

| shape | maxAbs (worst kernel) | GMAC/s (best kernel) |
|---|---|---|
| distilbert qkv/o [128·768·768] | 4.06e-5 | 180 (Row4Bias) |
| distilbert fc1 [128·768·3072] | 5.27e-5 | 183 (Row4Bias) |
| distilbert fc2 [128·3072·768] | 1.70e-4 | 177 (Row4Bias) |
| distilbert head [128·768·2] | 2.19e-5 | 3 (OneToOne; launch-overhead bound) |
| minilm qkv/o [128·384·384] | 1.86e-5 | 139 (Row4) |
| minilm fc1 [128·384·1536] | 2.42e-5 | 164 (Row4Gelu) |
| minilm fc2 [128·1536·384] | 8.18e-5 | 164 (Row4) |
| minilm head [128·384·2] | 9.81e-6 | 3 (OneToOne; launch-overhead bound) |
| edge padded rows [100·770·70] | 3.47e-5 | 55 (Row4) |
| edge padded K [64·1032·130] | 4.64e-5 | 68 (OneToOne) |

The maxAbs values reproduce the documented f32-vs-DP summation-order floor exactly
(K=768 → 4.0e-5, K=3072 → 1.7e-4), leaving ~25–250× margin to the 1e-3 gate and ~6 orders
to the Row4 `colBase`-class bug signal (~40). GMAC/s is load-sensitive (this run read the
Arc iGPU at ~half the idle-machine scenario benchmark 303–379 GMAC/s, docs/BERT-GPU.md) —
the correctness gate is the primary contract; compare GMAC/s moves, not absolutes.

#### GEMM gate baseline (2026-09-27, Laya shapes extended)

70 cells, all PASS. The nine pre-existing shapes reproduce the 2026-09-19 rows above; these
are the ten Laya shapes, which is what the Laya backend decision (#449) turns on. Gate
runtime 94 s on AC. *(A tenth pre-existing shape, `distilbert qkv/o [128x768x768]`, was
accidentally dropped by commit `05fc6bb` on 2026-09-29 and restored later the same day,
so this row count reads nine where it read ten at the time. Its 2026-09-27 figures were
Row4 182 GMAC/s, maxAbs 4.00e-5.)*

| shape | maxAbs (worst kernel) | GMAC/s (best kernel) |
|---|---|---|
| laya qkv [512x1024x3072] | 7.18e-5 | 202 (Row4 / Row4Bias / Row4Qkv, tied) |
| laya attn out [512x1024x1024] | 6.46e-5 | 202 (Row4) |
| laya fc1 (Wi) [512x1024x5248] | 7.18e-5 | 205 (Row4) |
| laya fc2 (Wo) [512x2624x1024] | 1.68e-4 | 197 (Row4 / Row4Bias, tied) |
| laya head ff1 [512x1024x4096] | 7.18e-5 | 204 (Row4Bias) |
| laya head ff2 [512x4096x1024] | 2.40e-4 | 155 (Row4Bias / Row4Relu, tied) |
| laya act 1 [512x1028x256] | 5.79e-5 | 188 (Row4 / Row4Bias, tied) |
| laya scorer 1 [8x1024x1024] | 4.41e-5 | 59 (Row4Bias) |
| laya scorer 2 [8x1024x1] | 1.34e-5 | under 1 (launch-overhead bound) |
| laya qkv@128 [128x1024x3072] | 6.54e-5 | 193 (Row4Bias / Row4Qkv, tied) |

The pre-existing rows also drifted up to 182-189 GMAC/s from 177-183 in the same session,
which is the load-sensitivity the note above warns about rather than a kernel change - the
ten shapes added here cannot affect ten other shapes' timings, since `DpCore` is computed
once per shape and the variants loop is per shape.

`laya scorer 1`/`2` are M=8 and M=8/N=1: too few rows and columns to fill the device, so
they are launch- and bandwidth-bound at 59 and under 1 GMAC/s. They are in the gate because
they are Laya's real decision-tail shapes, and a gate that only measures fat shapes would
miss the tail. `laya head ff2` is the one Laya shape that does not reach 190 (155 GMAC/s):
K=4096 with N=1024 is a worse aspect ratio for the 16x16 tile than the other head shapes.

#### GEMM gate baseline (2026-09-29, #440 tile geometries — a measured negative)

> **Superseded as a comparison on 2026-09-30, not re-recorded — read this before using the ratios.**
> The denominator moved. `Row4` and its fused siblings were migrated off two `SharedMemory.Allocate2D`
> calls onto a single wide `Allocate2D` (#468) and came out **~20% faster** on the same shapes
> (Row4 +19.9% median over 22 shapes, per-cell run-to-run noise median 1.2%). The four `#440`
> geometries were *not* touched and still stage their tiles through the 1D
> `SharedMemory.Allocate<float>` form, so re-recording this table today would compare 1D addressing
> against 2D and book the ~20% form difference as if it were a tile-geometry effect — which is the
> one thing this table exists to isolate. The table below is therefore left exactly as measured on
> 2026-09-29 and is kept only as the record of that experiment. Re-measuring the family with all
> five kernels on one addressing form is outstanding work; until then, treat "no geometry wins" as
> a statement about the 1D form the family was built in, not about the geometries.

171 cells, all PASS, plus **byte-identity PASS** against Row4 on all #440 cells. Same machine,
AC line. The four new geometries are the *plain* kernels only — the fused-epilogue siblings
were not built, because the plain ones did not win and a sibling cannot beat its base.

Ratio to Row4 on the same shape (GMAC/s, `1.00x` = parity, below 1 = slower):

| shape | Row4 GMAC/s | 2x2@KT16 | 2x2@KT32 | 4x2@KT32 | 1x8@KT16 |
|---|---|---|---|---|---|
| laya qkv [512x1024x3072] | 201 | 0.74x | 0.60x | 0.76x | 0.82x |
| laya fc1 (Wi) [512x1024x5248] | 202 | 0.71x | 0.58x | 0.73x | 0.79x |
| laya fc2 (Wo) [512x2624x1024] | 197 | 0.75x | 0.60x | 0.77x | 0.81x |
| laya attn out [512x1024x1024] | 203 | 0.74x | 0.59x | 0.74x | 0.84x |
| laya head ff1 [512x1024x4096] | 204 | 0.76x | 0.58x | 0.76x | 0.82x |
| laya head ff2 [512x4096x1024] | 155 | 0.89x | 0.67x | 0.98x | **1.01x** |
| laya act 1 [512x1028x256] | 189 | 0.76x | 0.60x | 0.74x | 0.84x |
| laya qkv@128 [128x1024x3072] | 193 | 0.73x | 0.58x | 0.77x | 0.78x |
| laya scorer 1 [8x1024x1024] | 59 | 0.41x | 0.42x | 0.25x | 0.37x |
| distilbert qkv/o [128x768x768] | 182 | 0.69x | 0.60x | 0.74x | 0.73x |
| distilbert fc1 [128x768x3072] | 189 | 0.72x | 0.59x | 0.76x | 0.78x |
| distilbert fc2 [128x3072x768] | 186 | 0.68x | 0.60x | 0.76x | 0.72x |
| minilm qkv/o [128x384x384] | 135 | 0.68x | 0.62x | 0.64x | 0.64x |
| edge padded rows [100x770x70] | 61 | 0.64x | 0.74x | 0.43x | 0.43x |
| edge padded K [64x1032x130] | 69 | 0.61x | 0.70x | 0.42x | 0.35x |

The three `#440`-only edge shapes (`edge 1x8 N=100`, `edge 4x2 M=20 N=40`, `edge 2x2 K=17`)
are gated for correctness — byte-identity and halo handling — but are left out of this table
because their absolute rates are so low that the ratio is mostly launch overhead in both
numerator and denominator. Read their rows from the run output when you need them.

**No geometry wins.** The best cell is 1.01x on `laya head ff2` — the one shape already known
to be weak at 155 GMAC/s — and that is a wash inside run-to-run noise. Everything else is a
regression. The launch-bound narrow shapes (`head`, `scorer`) regress hardest, as they should:
a wider tile means fewer groups, so fewer items to hide launch latency behind.

**Why, since the family was built on a theory that did not hold.** The premise was
*shared-memory traffic per MAC* — 2x2 needs 4 shared reads per 4 MACs against Row4's 5, a 20%
cut. The binding constraint on this device is *shared-memory capacity per group*. The pair that
isolates it: `2x2@KT16` and `2x2@KT32` have identical blocking factors, so identical reads per
MAC, and differ only in footprint (4 KB vs 8 KB per group).

**Scope that claim honestly, because the unscoped version is false on 6 of 15 shapes.** On the
nine largest shapes — the ones that carry the throughput — 2x2@KT32 is **18–25% slower than
2x2@KT16, every one of them**. It weakens to 8–13% on mid shapes, and it *inverts* on the three
launch-bound ones, where 2x2@KT32 comes out 2–16% **faster** (`laya scorer 1`, `edge padded
rows`, `edge padded K`). Footprint dominates where the device is saturated with work and is not
a predictor at all where there are too few work groups to saturate it — at which point group
count dominates instead.

`4x2@KT32` is consistent with that mechanism rather than with the traffic metric: it has the
best traffic ratio in the family (0.75 reads/MAC) and the worst footprint (12 KB), and it lands
mid-pack. So going from Row4's 5 KB to 8–12 KB costs more co-resident work groups than the
20–40% traffic saving returns. See [ACCELERATION.md](../../docs/ACCELERATION.md) §5.2.

**A second finding, about the gate rather than the kernel.** All four geometries were *wrong*
on the first run (maxAbs 20–119) and `1x8@KT16` failed to compile at all, from an ILGPU OpenCL
lowering bug (#468): two `SharedMemory.Allocate2D` calls in one kernel whose extents disagree
get the second tile mis-placed. `2x2@KT32` passed throughout, and the tell is that its two tiles
happen to share a 32x32 extent, so the lowering has nothing to disagree about. Fixed with a
single `SharedMemory.Allocate<float>` and hand-computed 2D offsets. Nothing about the failure
was visible in the source — every extent, stride and element index was in bounds and the load
destination equalled the read source. **The byte-identity assertion is what caught it**; the
`maxAbs` tolerance would have been argued with.

#### `--gemm-legs` - where the wall-clock actually goes

`dotnet run --project tests/Nivara.PerformanceTests -c Release -- --gemm-legs`

Attributes a whole encoder layer's GPU time to individual kernel *legs* rather than to the GEMM
as one bucket: `BatchedAttention`, all GEMM shapes, `LayerNorm1D`, `SplitColumns`, `GeGlu`,
`Add`, `Rotary`. Profiling, not a gate — it exits 0 and asserts nothing. It exists because the
GEMM-throughput work started from a claim that GEMM is ~99.9% of the cost, and that claim is
about a share of *arithmetic*, not of time.

The claim does not survive contact with the clock. Laya at 512 rows / d=1024 / 28 layers,
2221.12 ms total: **BatchedAttention 1266.87 ms (57.0%)**, GEMM across four shapes 870.84 ms
(39.2%), `LayerNorm1D` 28.15 (1.3%), `SplitColumns` 27.86 (1.3%), `GeGlu` 11.77, `Add` 9.06,
`Rotary` 6.49. ModernBERT base, 1536.53 ms total: **BatchedAttention 939.22 ms (61.1%)**, GEMM
35.3%. So attention — not GEMM — is the single largest leg, and the two are the same order of
magnitude rather than one dwarfing the other. `BatchedAttention` carries the RoPE band
internally but has three unconditional `j < seqLen` passes, so `band` currently reduces no work
at all. Tracked as **#447**.

An earlier run of this probe on **battery** gave attention 51.3% / GEMM 45.4% on Laya. Battery
compressed the gap rather than inventing it — the AC run shifts attention *up* to 57.0% — so
the conclusion strengthened. Worth knowing that the ratio moved ~6 points between power states,
if you ever have to compare a battery figure against an AC one.

**On the `band` parameter, because it is easy to misread 57% as optimistic or pessimistic when
it is neither.** The probe passes `GlobalAttentionBand` for every layer, and ModernBERT-large
is a 3:1 global:sliding mix with `SlidingWindow = 256`, so the natural worry is that this
overstates the real cost. It does not: `BatchedAttention` sweeps `for (int j = 0; j < seqLen;
j++)` three times unconditionally and uses `band` only to decide whether to overwrite a score
with `-inf`. Its cost is therefore **identical for a sliding band**, so no choice of band in
this probe could have changed the number. 57% is what the kernel actually costs today.

The useful corollary is the opposite of the worry: 57% is an upper bound on what attention
*would* cost once the band is made to reduce work rather than only mask it, which is exactly
what #447 would do. **The prize is larger than 57%, not smaller** — and that gap between the
current cost and the banded cost is the size of the opportunity.

Run this before optimising anything on the GPU path. It is the difference between a real 51%
and an assumed 99.9%.

#### `--gpu-alloc` - device-memory ceiling

`dotnet run --project tests/Nivara.PerformanceTests -c Release -- --gpu-alloc`

Answers one binary question: can the iGPU back Laya's working set? Laya's checkpoint is
421,293,830 F16 parameters and the GPU path is F32-only, so F32 needs
`421,293,830 * 4 B = 1.685 GB` (1.569 GiB) of device buffers. Every published GPU row in
`samples/NivaraInference/README.md` is 66-110M parameters (0.26-0.44 GB), so this is a 4-6x
jump beyond anything known to work on this device.

Allocation is chunked at 64 MiB rather than taken as one contiguous block, because Laya's
working set is an embedding table plus ~200 per-layer weight tensors - the real question is
whether the device can back many medium buffers totalling 1.569 GiB. Every chunk is
host-filled and a sample read back, since a reservation that succeeds but cannot be written
is not a working set. Targets step 256 MB to 3 GB so a failure localises to a size.

Exit 0 = Laya's working set allocated, filled and verified; 1 = it did not; 10 = UNBUILT.

Measured 2026-09-27, AC, Intel Arc iGPU (7.559 GiB reported):

| target | buffers | alloc ms | fill GB/s | result |
|---|---|---|---|---|
| 256 MB | 4 | 26.7 | 6.79 | ok |
| 512 MB | 8 | 40.4 | 7.68 | ok |
| 1.000 GB | 16 | 87.3 | 8.04 | ok |
| 1.500 GB | 24 | 104.1 | 8.33 | ok |
| **1.569 GiB (Laya F32)** | 26 | 109.1 | 8.31 | **ok** |
| 2.000 GB | 32 | 245.7 | 8.26 | ok |
| 3.000 GB | 48 | 788.5 | 7.58 | ok |

Allocation cost is superlinear - 109 ms at 1.569 GiB but 789 ms at 3 GB for 1.9x the
memory, which is driver paging/eviction rather than anything in Nivara. It does not affect
a one-time load, but it means the headroom above Laya's size is narrower than the raw
7.559 GiB figure suggests.

**The `alloc ms` and `fill GB/s` columns are not a contract; the verdict is.** Four
consecutive runs on the same machine and AC line gave Laya's row 109.1 / 147.7 / 133.1 /
100.6 ms and 8.31 / 7.32 / 7.18 / 7.77 GB/s, and the 1.500 GB row ranged 104-259 ms. Every
run returned exit 0 with the same verdict, so the answer is stable and the timings are not.
Read this mode as a binary gate plus a rough fill-rate estimate; do not threshold on the
milliseconds. (Same class of instability as the single-row variance noted under
`--cpu-gemm`.)

#### `--cpu-gemm` - CPU GEMM ceiling

`dotnet run --project tests/Nivara.PerformanceTests -c Release -- --cpu-gemm`

Measures what the CPU can do at the same shapes the GPU gate uses, so the two backends are
compared on identical inputs rather than across harnesses. Three legs, all gated
`maxAbs(leg - double-precision truth) <= 1e-3` against the same truth the GPU gate uses:

| leg | what it is |
|---|---|
| `AutoDiff` | `GradKernels.MatMulTransposedB`, B as `[N x K]`, no per-call transpose - what the AutoDiff path calls |
| `BTranspose` | `GradKernels.MatMul`, B as `[K x N]`, transposes B on every call |
| `Blocked` | probe-local `Parallel.For` + `Vector<float>` register accumulators held across the whole K loop, A and B read in place with no `RentCopy` and no staging copy |

`Blocked` exists because there is no BCL matrix multiply to compare against
(`TensorPrimitives.Dot` is the *vector* dot product; `Tensor.MatrixMultiply` has not
shipped, dotnet/runtime#95863, noted at `TensorsHelper.cs:30-36`). It is the shape a
register-blocked CPU GEMM would take, written in the harness so it measures a number
instead of proposing a change to `src/Nivara`.

Note the two in-tree legs take **different B layouts** - `MatMulTransposedB` means "B is
already transposed" and expects `[N x K]`, while `MatMul` takes `[K x N]` and transposes.
Both compute `A[M x K] · B[K x N]`.

Projected Laya forward, GEMM only (28 encoder layers of qkv + attn out + Wi + Wo at S=512,
plus the 2-layer decision head and the act/scorer tail; attention, norms and dispatch
excluded, so it is a lower bound on wall-clock for every leg equally):

| leg | enc layer ms | x28 layers ms | head ms | total ms | GMAC/s |
|---|---|---|---|---|---|
| AutoDiff | 81.5 | 2282 | 132.3 | **2414** | 76 |
| BTranspose | 93.0 | 2603 | 188.4 | 2791 | 66 |
| Blocked | 86.1 | 2412 | 117.3 | 2529 | 73 |

`Blocked` does **not** beat the in-tree kernel (2529 vs 2414 ms), so the hypothesis that
`MultiplyRowFloat`'s one-`Dot`-per-output-element structure is the CPU bottleneck is
falsified - BCL's `Dot` is already well tuned. The `AutoDiff` vs `BTranspose` gap (1.16x
overall, up to 1.51x at head ff2) is the real cost of the per-call B transpose, avoidable
by hoisting the transpose to weight-load time as the GPU path already does.

Caveat: individual rows vary up to 6.5x run to run (`distilbert fc1` `AutoDiff` read 6.3 ms
and 41.3 ms on consecutive runs, most likely `ArrayPool` `clearArray` interacting with GC).
The in-tree legs' projected roll-up is stable to ~2% across three runs (2414 / 2462 /
2453 ms for `AutoDiff`), so read conclusions off the roll-up, not single rows. The
probe-local `Blocked` leg is looser (~9%: 2529 / 2646 / 2772 ms) because it carries its own
`Parallel.For` scheduling behaviour, which widens the interval on the thin `scorer` rows
where task overhead is a large share of the row. It was behind `AutoDiff` in all three
runs, so the conclusion does not depend on the spread. Runs in ~7 s.

**Steady-state vs per-call.** The two in-tree legs are exactly the two cases the probe plan
asked to distinguish. `AutoDiff` receives B as `[N x K]` ready-made, so it is the
steady-state case a real inference loop reaches after one transpose at load; `BTranspose`
pays the transpose on every call. Their 2414 vs 2791 ms gap (1.16x overall, 1.51x at
`laya head ff2`) is therefore the measured cost of not hoisting it, and hoisting it is what
the GPU path already does.

**Not delivered.** The plan also asked for the pooled `bT` staging buffer and
`RentCopy(a)` to be reported separately, on the grounds that they are copies rather than
GEMM work. The `bT` half of that is what the `AutoDiff`/`BTranspose` pair above measures, so
it is covered. The `RentCopy` half was not measured: it sits inside the parallel path that
both in-tree legs take, and separating it would have needed a fourth leg. It was dropped
once the blocked reference failed to beat the in-tree kernel, since the question it was
meant to inform - "is there a CPU GEMM fix worth making" - had been answered no.

### `--mask` — `ApplyMask` vs the `TensorPrimitives.Add` it replaced (#480)

```bash
dotnet run -c Release --project tests/Nivara.PerformanceTests -- --mask
```

#448 replaced a vectorized `TensorPrimitives.Add` over the `[qLen, kvLen]` score
buffer with `AttentionKernels<T>.ApplyMask`'s compare-and-select loop, because BCL
has no select/blend primitive. The justification left on record was an inference
from a MAC count, never a stopwatch reading, so this mode supplies the reading.
It is a **probe, not a gate**: it exits 0 on a healthy run and asserts only the
parity contract. It is deliberately not a scenario row, because the scenario table
measures rows sequentially and cannot produce a ratio within one run — a ratio
there means a recorded baseline JSON, which is cross-day and cross-machine drift.

**Design** (from `tests/Nivara.SimdProbe/TransposeKernelProbe.cs`; the superseded
transpose gate #482 failed 3 of 5 runs on a clean tree on a 0.2% margin because it
took best-of-N per route, measured the routes sequentially with one always first,
and warmed neither up):

| aspect | choice | why |
|---|---|---|
| statistics | interleaved A/B, 30 rounds/cell, first-measured route alternates every round | cancels first-measured systematic drift |
| warmup | 5 untimed passes **per route per `T`** | `ApplyMask<T>` is generic-specialized; tiered JIT settles per instantiation |
| estimate | median ratio + win count + range | a distribution, not a point estimate |
| noise band | ±3% | reused from `TransposeKernelProbe.Classify`, so two probes do not invent two thresholds |
| build | `IsDebugBuild()` guard | Debug numbers are void, not a result |

**Grid**: shapes `[512,512]` / `[2048,2048]` / `[1,512]` × `float` / `double` /
`Half` / `BFloat16` (ADR-001's `IFloatingPointIeee754` domain) × mask regimes
`None` / `Causal` (~50% `-inf`, the unpredictable-branch case) / `All` (`-inf`)
× `rowFlags` empty vs populated (inference vs training). 72 cells.

**Parity contract.** The two routes *cannot* be required to agree everywhere:
assigning `-inf` rather than summing into it **is** the #448 fix, because
`NaN + (-inf) = NaN` lets a diverged score escape suppression. So parity is
asserted exactly (by raw bytes, no tolerance) only where the routes are required
to agree — all-finite scores with a mask in `{0, -inf}` — and the two intentional
divergences (a `NaN` score; a `+inf` score under a `-inf` cell) are documented
rather than papered over. A shortfall is reported as a coverage failure on its own
exit path, never folded into a numeric verdict.

**Recorded baseline** (Release, .NET 11.0.0, x64, AVX2 — `Vector512` *not*
accelerated; parity 72 of 72 bit-identical; exit 0):

Aggregate over measurable cells: **median ratio 1.130**, classified as MASK SLOWER
in 29–32, add faster in 15, within noise in 4 (of 48, second run). Coverage:
**48 of 72 measurable**, 24 excluded as below the clock's resolution.

The aggregate is close to meaningless on its own — it mixes types that disagree in
*direction*. Scoped per type:

| type | shape | flags | `None` | `Causal` | `All` |
|---|---|---|---|---|---|
| `float` | wide prefill | empty | 1.96× | 1.78× | 1.91× |
| `float` | wide prefill | tracked | 3.27× | 2.97× | 2.95× |
| `float` | S=512 | empty | **0.75×** | 1.17× | 1.31× |
| `float` | S=512 | tracked | 2.11× | 2.18× | 1.60× |
| `double` | wide prefill | empty | 1.70× | 2.46× | **0.86×** |
| `double` | wide prefill | tracked | 1.92× | 1.61× | 1.26× |
| `Half` | wide prefill | empty | **0.74×** | **0.56×** | **0.36×** |
| `Half` | wide prefill | tracked | 1.13× | 1.03× | 1.10× |
| `BFloat16` | wide prefill | empty | 1.12× | **0.81×** | **0.48×** |
| `BFloat16` | wide prefill | tracked | 1.36× | 1.06× | **0.65×** |

Three findings, and the third is the one that constrains any fix:

1. **On `float32` at prefill widths the scalar loop is genuinely slower.** All six
   wide-prefill `float` cells are outside the ±3% band with **0 of 30 rounds won**
   — the strongest evidence in the grid, because the win count is independent of
   the ratio. `float` is the shipping path and the only type Laya uses.
2. **It is not uniform, and "add a fast path" is not the whole story.** `float` at
   S=512 with an all-zero mask and no flag tracking *wins* (0.75×, 28 of 30 rounds).
3. **For `Half`, `ApplyMask` is usually FASTER than the BCL add** (0.36–0.74× on
   wide prefill, 30 of 30 rounds in 5 of 6 `Half` cells). GB/s explains it: the
   BCL `Add<Half>` path runs at ~1.6 GB/s while `ApplyMask` reaches 4.5 GB/s —
   neither is vectorized (no `Vector128<Half>`), and the hand-written loop simply
   avoids the BCL's conversion overhead. **So restoring `TensorPrimitives.Add`
   for `Half` would be a regression, not a fix.** This is the concrete form of the
   "no hand-rolled SIMD" convention problem: a `Vector256.ConditionalSelect` fast
   path cannot even be spelled for `Half`/`BFloat16` (`ConditionalSelect<T>`
   throws `NotSupportedException` for them), so the current loop is the *only*
   path for those types and it is currently winning.

**Laya leg profile** (B=1, S=512, d=1024, H=16, headDim=64, 28 layers; legs mirror
`ReverseGradOperations.MultiHeadAttention`). Two consecutive runs of the
per-round-chain design:

| leg | ms/call A | ms/call B | ms/call C | spread |
|---|---|---|---|---|
| PackHeads (×3) | 1.908 | — | — | — |
| QK^T matmul | 7.743 | 8.292 | — | 7% |
| scale multiply | 0.282 | — | — | — |
| **ApplyMask** | **0.388** | **0.429** | **0.316** | **36%** |
| *add (counterfactual)* | *0.138* | *0.132* | *0.101* | *37%* |
| softmax rows | 17.053 | 7.355 | 6.791 | **2.5× (A is the outlier)** |
| PV matmul | 4.310 | 4.586 | 4.394 | 6% |
| ScatterHead | 0.064 | — | 0.060 | — |
| **attention total (ms/layer)** | **479.4** | **338.1** | **323.7** | **48%** |

The mask's absolute cost is **96–133 ms per 28-layer forward** across the three runs.
The *share* is **1.3–2.0%** of attention, which is unstable for the reason below.

**Two earlier defects in this profile, both corrected here, and both of which had
produced false claims.** The original design timed each leg 30× *in isolation*, and
that is wrong in two separate ways:

1. It never re-ran the chain, so each leg saw the buffer state its predecessor left
   after 30 applications. `SoftmaxRows` is **not idempotent** — iterating it
   converges toward uniform — so softmax was timed on already-softmaxed rows rather
   than on masked scores. Its real cost was understated by **2.3×**.
2. The `add` counterfactual was taken *after* the chain finished, so it was measured
   on softmax output while the mask had been measured on post-scale output. The
   headline delta was therefore comparing two routes on **two different inputs**.

The replacement runs the whole chain in dependency order once per round, timing each
leg in place, so `QK^T` rewrites `scores` from scratch every round and every leg sees
the data it sees in production. The `add` counterfactual now sits immediately beside
`ApplyMask` on the same buffer; both are idempotent and agree bit-for-bit here
(finite scores, mask in `{0,-inf}`: `x + 0 == x`, `-inf + -inf == -inf`), so the
delta is a real A/B on identical input rather than two independent measurements.

**Consequence: the previously published 62% `QK^T` share and its 11× layout gap
were mostly measurement artifact, and are retracted.** Under the corrected design
`QK^T` is **25.8–39.2%** and the `QK^T`/`PV` gap is **~1.8×** (7.74 vs 4.31; 8.29 vs
4.59), not 11×. The old gap was produced by `PV` reading the degenerate converged
softmax output (measured 1.395 ms, vs 4.31 ms on real data) and by `QK^T` being timed
back-to-back against a hot cache instead of interleaved with the other legs. A
materiality argument built on that gap does not survive; it must not be cited from
earlier revisions of this file.

**The correction did not fix the reconciliation, and it did not fix stability.**
Attention alone is 324–479 ms/layer = **9 100–13 400 ms per forward**, still far
from `docs/LAYA.md`'s recorded ~4.1 s whole forward, and the total swings 48% across
three runs while its neighbour legs sit at 6–7%. The spread is dominated by run A's
`softmax rows` at 17.05 ms against 7.36/6.79 in runs B and C — so B and C agree
within 8% and A is the outlier, rather than the leg being reliably bimodal. Three
runs cannot distinguish "one contaminated run" from "a genuine low-probability
state", so it is recorded as unexplained rather than characterized. Either way
`SoftmaxSingle` — a scalar max loop plus `TensorPrimitives.Subtract`/`Exp`/`Divide`
— costs 6.8–17 ms for 262 144 elements, i.e. 26–65 ns/element, far off a vectorized
`Exp` rate, and at 33–57% it is the dominant attention leg. **No percentage-of-forward
is published.** The mode prints the reconciliation conflict instead.

**Two caveats carried in the output.** (a) The `[1,512]` row measures the mask
API *at decode shape*; `DecodeAttention` and `BatchedAttention` use a `Keep`
predicate and never call `ApplyMask`, so it is not the decode path's cost. (b) All
24 `[1,512]` cells are below timer resolution (~10 Stopwatch ticks) and print `n/a`;
before that guard they reported 5–10× "regressions" that were pure clock
quantization — the tell being a `61.44 GB/s` reading that is just a rounded tick
count over 6 KB of L1-resident data. Publishing that as "6× slower at decode" is
the kind of false claim this mode exists to prevent.

**Ratio stability is limited — read this before quoting any single cell.** Two
consecutive runs:

| quantity | run A | run B | run C | verdict |
|---|---|---|---|---|
| parity | 72 of 72 | 72 of 72 | 72 of 72 | **exact, stable** |
| measurable cells | 48 of 72 | 48 of 72 | 48 of 72 | **exact, stable** |
| absolute delta (ms / 28-layer forward) | 112.3 | 133.0 | 96.3 | **39% spread** |
| `float` wide prefill, 6 cells | 0 of 30 rounds won | 0–1 of 30 | 0 of 30 | **stable** |
| aggregate median ratio | 1.130 | 1.168 | — | ±3% wobble |
| `float` `[512,512]` causal/empty | 1.17× (SLOWER) | 1.05× (SLOWER) | — | **straddles the band** |
| `ApplyMask` leg (ms/call) | 0.388 | 0.429 | 0.316 | 36% |
| `add` counterfactual (ms/call) | 0.138 | 0.132 | 0.101 | 37% |
| `softmax rows` leg (ms/call) | 17.05 | 7.36 | 6.79 | **2.5×, A the outlier** |
| attention total (ms/layer) | 479.4 | 338.1 | 323.7 | 48% |

Five consequences:

1. **The `float` `[512,512]` causal/empty cell straddles the ±3% band between
   runs** (1.05–1.17×). It is *not* a reliable "slower" verdict, and the table
   above records it as a ratio, not a conclusion.
2. **The earlier "delta stable to 0.02 ms" is retracted.** That stability came
   from the isolated-leg design, where both routes timed the same static buffer
   every round. Under the corrected chain the delta spans 96–133 ms across three
   runs, and both the mask leg (0.316–0.429) and the counterfactual (0.101–0.138)
   move with it in the same direction. The earlier figure was stable *because* it
   was measuring something other than the production data path, not because the
   quantity is stable.
3. **The attention total's 48% spread is dominated by one leg and one run.** Run
   A's `softmax rows` reads 17.05 ms against 7.36/6.79 in B and C, and softmax is
   33–57% of attention, so softmax alone accounts for the total's spread. The other
   legs hold 6–7%. Three runs cannot separate "run A was contaminated" from "the
   leg has a real low-probability slow state", so it stays unexplained. Until it is
   settled the attention *share* is a 1.3–2.0% band, and the mask's *absolute*
   96–133 ms/forward band is the citable quantity.
4. The instability is **not** inherited from the kernel A/B grid: parity (72/72),
   measurable-cell count (48/72) and the win counts are identical in all three
   runs. Only the wall-clock legs move.
5. Within one run, the same nominal configuration also read differently by
   measurement path: `float` `[512,512]` causal/empty was 1.17× in the A/B and
   2.96× in the leg profile.

**What survives:** the `float` wide-prefill regression (all six cells outside the
band, essentially 0 of 30 rounds won, across all three runs), the exact parity and
coverage counts, and the mask's absolute cost as the 96–133 ms/forward band.
**What does not:** any single-cell magnitude near the band, any second digit on the
attention share, the prior "stable to 0.02 ms" delta, and the retracted 62%/11×
`QK^T` figures.

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
