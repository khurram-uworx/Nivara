# PERF-66 — NivaraFineTuning training speed on CPU

**Status:** in progress · **Scope:** AutoDiff training hot paths + sample usage · **Tracks:** GitHub issue #66

This is the plan of record for the performance investigation into
`samples/NivaraFineTuning` (67M-parameter DistilBERT, 6 layers, 12 heads,
768 hidden, B=2/L=128) — the ">10 minutes per epoch" report. The plan
was derived from chat history (issue #66 root-cause analysis) and is
recorded here so the P1/P2/P3 work items and their status survive outside
the conversation.

---

## Root-cause analysis (B=2, L=128, D=768, 6 layers, 12 heads)

Per-batch profile (measured/derived from code):

- ~30G MACs forward+backward, dominated by 38 Linear matmuls per batch.
- ~1–1.5 GB heap churn per batch — co-primary bottleneck alongside FLOPs.
- 50% of attention FLOPs wasted at B=2 (75% at B=4) via the flattened
  single-op attention + block-diagonal mask.

### Five concrete bottlenecks

| # | Bottleneck | Where | Impact |
|---|---|---|---|
| 1 | Repro runs in Debug — `dotnet run` defaults to Debug (no JIT optimization); issue command + README used it; only `benchmark_timing.cmd` used `-c Release` | sample usage docs | several-× slowdown, essentially free |
| 2 | AdamW allocates `new T[n]` per param per step = 268 MB/batch for 67M params, then replaces the tensor | `src/Nivara/AutoDiff/Optimizer/AdamW.cs` | dominant single allocation |
| 3 | Flattened `MultiHeadAttention` on `[B·L, D]` + block-diagonal mask computes all B·L cross-sequence scores then masks them; `BatchedMultiHeadAttention` exists and is ~4.5× faster per the perf harness, but the sample never used it | `samples/Nivara.Samples/BertModel.cs` | halves attention FLOPs + kills ~3 MB/layer mask churn |
| 4 | Training Linear double-transposes every weight: cached transpose is undone/redone by `MultiplyCore`'s per-call `bTransposed:false` transpose | `src/Nivara/AutoDiff/Nn/Linear.cs`, `src/Nivara/Tensors/TensorsHelper.cs` | ~589K-element naive transpose per matmul |
| 5 | Backward/forward allocation churn: every op allocates fresh grad arrays; `AccumulateGradient` copies via `NivaraColumn.Create`; no ServerGC | `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs`, sample csproj | GC + memcpy overhead under 1.5 GB/batch |

---

## The plan

### P1 — Sample/usage quick wins (low risk, high visible impact) ✅ DONE

1. **Add `-c Release` + ServerGC/TieredPGO to the sample** — commit `0c29079`:
   - `samples/NivaraFineTuning/NivaraFineTuning.csproj`: `<ServerGarbageCollection>true</ServerGarbageCollection>` + `<TieredPGO>true</TieredPGO>`.
   - README usage lines + issue repro command document `-c Release`.
2. **Switch `BertSelfAttention<T>.ForwardBatched` to `BatchedMultiHeadAttention`** — commit `325f741`:
   - Q/K/V reshaped to `[B, L, D]`, per-batch padding mask `[B, L, L]`, fused op — removes the block-diagonal mask builder and wasted cross-sequence compute.
   - Parity covered by `BatchedMultiHeadAttentionTests` + new regression tests (`ForwardBatched_BatchedEqualsIndependentSingleSequenceRuns`, `ForwardBatched_Backward_ProducesParameterGradients`).

### P2 — Core AutoDiff (medium risk, biggest structural wins)

3. **In-place optimizers** (AdamW + Adam + SGD) — write into the parameter's
   existing backing array + `param.Touch()` instead of allocating a new
   `T[n]` and replacing the tensor. Needs a write-through internal
   writable-span accessor on `NivaraColumn<T>`. Eliminates the 268 MB/batch
   allocation.
   - Files: `src/Nivara/Storage/ColumnStorage.cs`, `src/Nivara/AutoDiff/Optimizer/{AdamW,Adam,SGD}.cs`, `src/Nivara/AutoDiff/Optimizer/Optimizer.cs` (if shared).
   - Status: **pending**.
4. **Grad-tracking matmul with `bTransposed: true`** — a `MatMulTransposedB`
   that records a VJP, so training Linear stops double-transposing weights.
   - Files: `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs`, `src/Nivara/AutoDiff/Nn/Linear.cs`.
   - Status: **pending**.

### P3 — Kernel micro-optimizations (measure first)

5. **Kernel micro-optimizations**:
   - Skip redundant `aCopy` in `MultiplyCore` when the source is already contiguous.
   - Blocked/cache-friendly `Transpose`.
   - Fused single-pass softmax rows with row-level parallelism.
   - Files: `src/Nivara/Tensors/TensorsHelper.cs`.
   - Status: **pending**.

---

## Verification protocol

- `dotnet build Nivara.slnx` (Release).
- Targeted tests: `OptimizerTests`, `LinearTransposedWeightCacheTests` (rewritten in P2 item 4), NivaraTorch parity, ForwardParityTests, `BatchedMultiHeadAttentionTests`, `DistilBertSequenceClassificationTests`, `PerfTests`.
- Full test suite run with human confirmation before starting (AGENTS.md).
- **Timed benchmark:** `samples/NivaraFineTuning/benchmark_timing.cmd 25 2 1`
  (`-c Release --epochs 1 --batch-size 2 --max-examples 25`). 25 examples is
  a deliberate quick signal (vs. the original 100) so we can move fast.
  Run baseline before P2, then after each phase; median of 3 runs
  (run-to-run variance ~±10% per `tests/Nivara.PerformanceTests/README.md`).
  Record in this file + `benchmark_results.txt`.
- Kernel measurements via `tests/Nivara.PerformanceTests` (point A baseline
  recorded 2026-08-05 in `tests/Nivara.PerformanceTests/README.md`; rolls
  forward per the baseline policy there).

---

## Execution log

| Step | Result |
|---|---|
| P1 item 1 | ✅ `0c29079` |
| P1 item 2 | ✅ `325f741` |
| P2 item 4 | |
| P2 item 3 | |
| P3 5a/5b/5c | |
| Benchmark 25 | |

---

## Follow-ups (out of scope for P1/P2/P3)

- **Bottleneck #5 grad-array churn**: every op allocates fresh grad arrays;
  `AccumulateGradient` copies via `NivaraColumn.Create`. Pooled grad buffers
  are a candidate next step. ServerGC (P1) partially mitigates.
- Backlog: `docs/AISTACK-ROADMAP.md` phases (ONNX export/import is the
  highest-value bridge).
