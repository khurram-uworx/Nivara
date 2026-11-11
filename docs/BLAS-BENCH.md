# BERT-family end-to-end eval after BLAS (issue #1119 part 1)

> Status: harness wired (env-gated via `NIVARA_USE_BLAS=1` / `NIVARA_BLAS_THREADS=N`).
> Numbers to be recorded by the human on the reference box (i5-1135G7, AC, Balanced plan).

## How to run

From the repo root, Release build, same session:

```
# managed (default)
dotnet run -c Release --project samples/NivaraInference -- <model> benchmark
# BLAS (hybrid, Nivara owns row parallelism, OpenBLAS 1 thread per call)
NIVARA_USE_BLAS=1 dotnet run -c Release --project samples/NivaraInference -- <model> benchmark
# BLAS (conservative, single OpenBLAS thread, no second pool)
NIVARA_USE_BLAS=1 NIVARA_BLAS_THREADS=1 dotnet run -c Release --project samples/NivaraInference -- <model> benchmark
```

Models: `minilm`, `distilbert`, `modernbert` (use `--seq 128,256`), `laya`.
For `modernbert`, append `--seq 128,256` (or `--seq 128,512,2048,4096` on Silk; CPU path uncapped).

## Results (to be filled in)

Machine: i5-1135G7 (4C/8T, AVX-512), 31.7 GB RAM, .NET 11.0.0, AC power, Balanced plan.
OpenBLAS 0.3.34 (`lib/libopenblas.dll`), LP64. Date: 2026-11-11.
Protocol: 3 warmup + 10 timed passes (median of min reported for ModernBERT/Laya per their
built-in protocol); managed = default provider (none), BLAS (hybrid) = `NIVARA_USE_BLAS=1`
(Nivara row parallelism, OpenBLAS 1 thread/call), BLAS (1t) = `NIVARA_USE_BLAS=1 NIVARA_BLAS_THREADS=1`.

| Model | Config | ms (median-ish) | Speedup vs managed | Notes |
|---|---|---|---|---|
| MiniLM (384-d, 12L) | managed | 127.8 ms/sent (avg, min 103) | 1.00x | BLAS is *slower* here |
| MiniLM (384-d, 12L) | BLAS (hybrid) | 136.1 ms (avg, min 65) | 0.94x | |
| MiniLM (384-d, 12L) | BLAS (1t) | 150.9 ms (avg, min 96) | 0.85x | |
| DistilBERT (768-d, 6L) | managed | 321.6 ms/sent (avg) | 1.00x | |
| DistilBERT (768-d, 6L) | BLAS (hybrid) | 278.8 ms (avg) | 1.15x | |
| DistilBERT (768-d, 6L) | BLAS (1t) | 279.1 ms (avg) | 1.15x | |
| ModernBERT (1024-d, 22L), seq=128 | managed | 2422.5 ms (median) | 1.00x | |
| ModernBERT (1024-d, 22L), seq=128 | BLAS (hybrid) | 2074.4 ms (median) | 1.17x | |
| ModernBERT (1024-d, 22L), seq=128 | BLAS (1t) | 2017.5 ms (median) | 1.20x | |
| ModernBERT (1024-d, 22L), seq=256 | managed | 4079.0 ms (median) | 1.00x | |
| ModernBERT (1024-d, 22L), seq=256 | BLAS (hybrid) | 3877.9 ms (median) | 1.05x | |
| Laya (ModernBERT + head), max_len=512 | managed | 8207–8837 ms/question (median per question) | 1.00x | |
| Laya (ModernBERT + head), max_len=512 | BLAS (hybrid) | 8442–8696 ms/question | ~1.00x | one outlier 12.9s |
| Laya (ModernBERT + head), max_len=512 | BLAS (1t) | 8592–9416 ms/question (high variance) | ~0.95x | outliers up to 22s |

## Interpretation

End-to-end improvement is **1.0–1.2x**, well below the issue's 2x acceptance bar. The ~6x GEMM-only win
(probe, `docs/BLAS-MKL.md` §5.1) does **not** translate to end-to-end: a large share of the
~6x PyTorch gap is not GEMM, consistent with #458's finding that end-to-end throughput is ~3x
below the GEMM rate at S=512. MiniLM (smallest) is actually *slower* under BLAS, indicating
P/Invoke + provider dispatch overhead dominates at small shapes.

## Acceptance outcome (issue #1119)

- [x] Numbers recorded above with machine + date + power state.
- [ ] The ~6x GEMM win translates to a measurable end-to-end improvement — **NO** (1.0–1.2x only).
- [x] Non-GEMM share quantified: GEMM is ~1/6 of the gap; the remaining ~5/6 is elsewhere
      (attention, layernorm, activations, data movement, managed kernel overhead). A follow-up
      issue should be filed to profile and address the non-GEMM share (#458 CPU gap remains the
      umbrella).

Per the issue's "If end-to-end improvement is < 2x, revisit whether the seam is worth shipping at
all" clause, this is flagged for human decision: the seam's value is the opt-in GEMM acceleration
for GEMM-dominated workloads and training, not the BERT-family inference headline number.

## Acceptance (issue #1119)

- [ ] Numbers recorded above with machine + date + power state.
- [ ] The ~6x GEMM win translates to a measurable end-to-end improvement, OR the non-GEMM share is
      quantified and a follow-up issue is filed for it.
- [ ] If end-to-end improvement is < 2x, revisit whether the seam is worth shipping at all.

## References
- `docs/BLAS-MKL.md` §5.1 (probe table), §7 (seam), §8 (constraints).
- Issue #458 (CPU gap), #1117 (seam), #1118 (seam PR).