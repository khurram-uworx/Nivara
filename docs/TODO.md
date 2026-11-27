# #1457 — Chunked linear + cross-entropy for SmolLM/Qwen training (Silk)

## Problem

Issue: `khurram-uworx/Nivara#1457` (OPEN, milestone 1.5.5). Deferred from #1453.

`SilkLlamaLoRATrainer` materializes full logits tensors for causal-LM training:

- `AllocateCommonBuffers` allocates `logitsBuf = cap * vocab` and `dLogitsBuf = cap * vocab` (`src/Nivara.Extensions/Silk/SilkLlamaLoRATrainer.cs:642-643`).
- `ForwardDevice` computes full logits with `Gemm(normedBuf, tokEmbTransposed, logitsBuf, seq, hidden, vocab)` (`:620`).
- `BackwardPromptMaskedCrossEntropy` downloads full logits to host, computes masked CE on host, launches `SoftmaxCrossEntropyBackward` over full `[seq,vocab]`, then computes `dNormed = dLogits @ tokEmb` (`:340-372`).

For Qwen vocab ~152k, `MaxSeqLen * vocab * 4 bytes` per buffer is too large (two buffers = full logits + dLogits). The issue asks for chunked linear + cross-entropy: compute loss/grad in vocab chunks without materializing the full logits tensor.

## Grounding / constraints (G1)

- **microsoft-learn MCP grounding:** queried for OpenCL sub-buffer/buffer-offset and numerically-stable chunked softmax. Results were not directly relevant to Silk/OpenCL.NET internals (mostly C runtime buffer APIs, DirectML/MPS/TensorPrimitives softmax docs). No conflicting Microsoft guidance found; the implementation is in Silk/OpenCL kernels, not BCL tensor APIs. `TensorPrimitives.SoftMax` exists in .NET but is not applicable to device-resident logits.
- **Code-memory navigation:** mapped `SilkLlamaLoRATrainer`, `SilkLoss`, `SilkKernels`, `SilkLauncher`, call sites, and relationships. Confirmed `BackwardPromptMaskedCrossEntropy` callers: trainer `TrainStep`, `samples/NivaraFineTuning/Program.cs:1359`, and `SilkLlamaLoRATrainerParityTests`.
- **Runtime facts verified in source:**
  - `SilkRuntime.CreateSubview(parent, elementOffset, elementCount, readOnly)` exists (`SilkRuntime.cs:329`) and is used by `SilkBertEncoderRunner`, `SilkDistilBertTrainer`, `GpuTextConvClassifier`, etc.; sub-buffers are separate CL objects and must be released.
  - `GemmNt`: `C[M,N] = A[M,K]·Bᵀ`, B row-major `[N,K]` (`SilkLauncher.cs:1302`). Forward chunk fits as `normed[seq,hidden] @ tokEmb[vStart:vStart+chunk, hidden]^T` via contiguous row-slice sub-buffer of `tokEmb[vocab,hidden]`.
  - `Gemm` overwrites C; `GemmAdd` computes `C=A*B+residual` (`SilkLauncher.cs:698`) and can accumulate dNormed across chunks.
  - Existing `softmax_cross_entropy_backward` kernel does 3 sequential passes over full vocab per row (max, sumexp, grad) with global max (`SilkKernels.cs:2898-2932`).
- **Blast radius:** see section below. No public signature removal planned; `Forward()` keeps returning full logits (materialized on host by chunk download, not as a persistent full device logits buffer). `BackwardPromptMaskedCrossEntropy` keeps signature/semantics.

## G1 decisions / red flags (need human clearance before implementation)

1. **Pass count / FP numeric order (recommended: 2-pass, accept tolerance):** Chunked CE needs global max and sumexp before dLogits. A 2-pass design (pass 1 max+sumexp with online merge across chunks; pass 2 recompute logits chunk, emit dLogits + loss, accumulate dNormed) means 2x LM-head GEMM per training step. It is numerically stable but floating-point accumulation order differs slightly from the current single-kernel 3-pass-full-vocab scan. Existing parity tests allow tolerance (loss 1.5e-3 rel, grads 3e-2 rel); I expect to stay within tolerance, but it is a numeric-order change. Alternative: 3-pass (max, sumexp, grad) gives closer-to-identical numerics but 3x LM-head GEMM. **Recommendation:** 2-pass, verify parity within existing tolerances; escalate if any parity test fails beyond tolerance.
2. **`Forward()` host materialization:** `Forward()` currently downloads full logits from a persistent device buffer. Plan keeps API but builds the host `float[seq*vocab]` by chunked download; no persistent full device logits buffer. Host still allocates full array - acceptable given issue states "without materializing the full logits tensor" (device). Confirm.
3. **`Options.ChunkVocab` default:** Add `int ChunkVocab = 8192` to `SilkLlamaLoRATrainer.Options` record (source-compatible default). Used for chunk buffer sizing and chunk loop; `effective = Min(ChunkVocab, VocabSize)`. Exposed for tests/tuning. Confirm.
4. **Sub-buffer lifecycle:** Create/release tokEmb sub-buffer views per chunk per pass. Small overhead vs GEMM; simple and safe. Alternative is caching sub-buffers for fixed chunk layout - can defer. Confirm.

## Proposed changes

### 1) Chunked CE kernels (SilkKernels + SilkLauncher)

Add two OpenCL kernels and launcher methods:

- `chunked_cross_entropy_max_sumexp` — one work item per row. Inputs: logitsChunk `[n,chunkSize]`, rowMax `[n]`, rowSumExp `[n]`, n, chunkStart, chunkSize, vocabSize, ignoreIndex. For non-ignored rows, merge chunk local max/sumexp into global accumulators using online log-sum-exp merge: `newMax=max(oldMax,localMax); sum = oldSum*exp(oldMax-newMax) + localSum*exp(localMax-newMax)`. First chunk initializes; subsequent chunks merge. Ignored rows leave accumulators unchanged (initialized to -inf/0).
- `chunked_cross_entropy_backward_chunk` — one work item per row. Inputs: logitsChunk, labels, dLogitsChunk, rowMax, rowSumExp, rowLoss `[n]`, n, chunkStart, chunkSize, vocabSize, scale, ignoreIndex. For ignored rows: write zero dLogitsChunk, rowLoss 0. For active rows: write `dZ = (exp(logit-max)/sum - onehot_in_chunk)*scale`; if label falls inside `[chunkStart, chunkStart+chunk)`, set `rowLoss[i]=-(logit_label-max-log(sum))` (only the containing chunk writes it; others leave it). `rowLoss` initialized to 0 before pass 2.

Add `SilkLauncher.ChunkedCrossEntropyMaxSumExp(...)` and `SilkLauncher.ChunkedCrossEntropyBackwardChunk(...)`.

Files: `src/Nivara.Extensions/Silk/SilkKernels.cs`, `src/Nivara.Extensions/Silk/SilkLauncher.cs`.

### 2) Trainer buffer/layout changes

In `SilkLlamaLoRATrainer`:

- Add `ChunkVocab` option? Or internal constant. Prefer `Options` field `ChunkVocab = 8192` (default) with validation (`>0`, `<=VocabSize`) for testability; adding record parameter with default is source-compatible.
- Replace full `logitsBuf`/`dLogitsBuf` (`cap*vocab`) with chunk buffers: `logitsChunkBuf = Alloc(cap*chunkVocab)`, `dLogitsChunkBuf = Alloc(cap*chunkVocab)`. Add small per-row accumulators: `ceRowMaxBuf`, `ceRowSumBuf`, `ceRowLossBuf` (cap floats each), and `ceIgnoreBuf`/reuse labels.
- Keep `normedBuf`, `finalNormInputBuf`, `tokEmb`, `tokEmbTransposed` as-is.
- `AllocateCommonBuffers`: allocate chunk buffers based on `effectiveChunk = Math.Min(ChunkVocab, VocabSize)`. Remove full logits/dLogits allocation.

### 3) Chunked forward/logits APIs

- `ForwardDevice`: stop at final RMSNorm producing `normedBuf` (already does). Remove the full LM-head GEMM into `logitsBuf` from the hot training path. For public `Forward()`, compute full logits on host by iterating vocab chunks: for each chunk create sub-buffer view of tokEmb rows `[vStart*hidden, chunk*hidden]` (read-only), `GemmNt(normedBuf, tokEmbChunkView, logitsChunkBuf, seq, hidden, chunk)`, download chunk into the right slice of a host `float[seq*vocab]`. Release sub-buffer views. This preserves API/behavior without persistent full device logits buffer.
- `ForwardLastLogits`: compute only last-token logits by chunked GemmNt with rows=1? Or run normed for seq and project row - simpler: compute full normed as today, then for each chunk GemmNt using a subview of normedBuf at last row offset? `normedBuf` is `[cap*hidden]` contiguous; last row at `offset=(seq-1)*hidden`, count=hidden - CreateSubview gives that. Then GemmNt with aRows=1. Download chunk into host array slice.

### 4) Chunked BackwardPromptMaskedCrossEntropy

Restructure to two-pass chunked CE:

1. Validate args, count valid labels (`valid`), compute `scale=1/valid`, upload labels.
2. Allocate/zero row accumulators: `rowMax = -INF`, `rowSum=0`, `rowLoss=0` on device (small Fill kernels).
3. **Pass 1 (max/sumexp):** for each vocab chunk `[vStart, vEnd)`, create tokEmb chunk subview `[vStart*hidden, chunk*hidden]` read-only; `GemmNt(normedBuf, tokEmbChunkView, logitsChunkBuf, seq, hidden, chunkSize)`; launch `ChunkedCrossEntropyMaxSumExp`; release subview.
4. **Pass 2 (loss + dLogits + dNormed):** zero `dTmpBuf` (dNormed acc). For each chunk, GemmNt to recompute logitsChunk; launch `ChunkedCrossEntropyBackwardChunk` to produce dLogitsChunk and rowLoss contribution; accumulate `dNormed`: first chunk `Gemm(dLogitsChunkBuf, tokEmbChunkView, dTmpBuf, seq, chunk, hidden)`, subsequent `GemmAdd(dLogitsChunkBuf, tokEmbChunkView, dTmpBuf, dTmpBuf, seq, chunk, hidden)`. Release subview.
5. Download `rowLossBuf[seq]` (tiny) and sum valid rows on host for loss scalar (or download and sum; seq is small). Alternatively compute loss in pass 2 on host from downloaded chunks - but we want to avoid full download. Device rowLoss buffer is cheap.
6. Final RMSNorm backward as today: `RmsNormBackward(dTmpBuf=dNormed, finalNormInputBuf, finalNormW, dCurBuf, dLnAccBuf, ...)`.
7. Return `(loss, valid)`.

Notes:
- Two-pass means 2x LM-head GemmNt over vocab per training step - accepted tradeoff per issue (memory/bandwidth win).
- Sub-buffer create/release per chunk is small overhead vs GEMM; acceptable. Reuse if profiling shows issue (defer).
- `CreateSubview` for tokEmb chunk: offset `vStart*hidden`, count `chunk*hidden`. Same for backward normal Gemm B arg.
- For last partial chunk, `chunkSize = vEnd-vStart`.

### 5) Keep parity / API compatibility

- `Forward(int[])` still returns full `float[]` logits (host-materialized by chunked download). Parity tests comparing full logits remain valid.
- `ForwardLastLogits(int[])` returns last row logits.
- `BackwardPromptMaskedCrossEntropy(int[], int)` signature unchanged.
- `TrainStep` unchanged externally.

### 6) Tests

- Keep existing `SilkLlamaLoRATrainerParityTests` green (synthetic + real-weight smoke tests as currently present). They already cover forward argmax/maxAbs parity and backward loss+grad parity.
- Add/extend a synthetic test that exercises chunking boundary: set small `ChunkVocab` (e.g. 7 or 16, not dividing vocab evenly) and assert same loss/grads/logits parity as default/un-chunked-equivalent path. This pins the no-full-logits chunking logic.
- If needed, add a focused test asserting device buffer sizing does not allocate `cap*vocab` for logits/dLogits (white-box via reflection/field or indirect memory-sensitive check) - only if reliable; prefer behavioral parity tests.
- Run targeted GPU/parity tests first (ask before running).

### 7) Docs

- Update `docs/SILK.md` if it documents trainer logits buffers / CE behavior (check during implementation).
- Update `CHANGELOG.md` with issue #1457 entry.
- Update `docs/TODO.md` issues log and remove on completion per skill.

## Blast radius

| Area | Files | Risk | Notes |
|---|---|---|---|
| Silk trainer CE path | `src/Nivara.Extensions/Silk/SilkLlamaLoRATrainer.cs` | High | Core training path; logits buffer allocation, ForwardDevice tail, BackwardPromptMaskedCrossEntropy, Forward/ForwardLastLogits. |
| Silk kernels/launcher | `src/Nivara.Extensions/Silk/SilkKernels.cs`, `src/Nivara.Extensions/Silk/SilkLauncher.cs` | Medium | Additive new kernel+launcher; existing `SoftmaxCrossEntropyBackward` remains for other callers (DistilBert, TinyShakespeare). |
| Silk runtime | `src/Nivara.Extensions/Silk/SilkRuntime.cs` | Low/None | Existing `CreateSubview` reused; no runtime API change expected. |
| Samples | `samples/NivaraFineTuning/Program.cs` | Low | Calls `BackwardPromptMaskedCrossEntropy` - signature unchanged. |
| Tests | `tests/Nivara.Tests/Gpu/SilkLlamaLoRATrainerParityTests.cs` | Medium | Must remain green; may add chunk-boundary test. |
| Other CE callers | `SilkDistilBertTrainer`, `SilkTinyShakespeareTrainer`, `GpuText*Classifier`, `SilkLoss` | None/Low | Not in scope; share no full-logits buffer with LlamaLoRATrainer. |
| Docs | `docs/SILK.md`, `CHANGELOG.md` | Low | Document behavior/memory improvement. |

## Verification

- [ ] `dotnet build Nivara.slnx -c Release` (ask before long-running build/test as needed; build is usually fine).
- [ ] Targeted GPU tests (ask before running): `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkLlamaLoRATrainerParityTests"`
- [ ] Broader Silk/GPU smoke as reasonable (ask before running): `--filter "Category=Gpu&FullyQualifiedName~Llama|FullyQualifiedName~SilkLoss|FullyQualifiedName~SoftmaxCrossEntropy"` or targeted set.
- [ ] If docs touched: docs gate `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`.
- [ ] `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` before push.
- [ ] Review DocsCoveragePins if any gated docs/citations/snippets changed.
- [ ] Final full suite not run without explicit confirmation (AGENTS.md).

## Planned commits

1. `docs: plan #1457 chunked linear+cross-entropy in TODO.md`
2. `feat(silk): add chunked cross-entropy kernels and launcher` (kernels + launcher + any small runtime helper if needed)
3. `feat(silk): chunk Llama LoRA logits/CE without full logits tensor` (trainer refactor + options + tests)
4. `docs: record #1457 chunked CE in SILK/CHANGELOG` (if needed)
5. `docs: remove TODO.md — plan executed` (only after G2 clears)

## GitHub issues log

- [ ] #1457 — Silk: chunked linear+cross-entropy for SmolLM/Qwen training (no full logits) (this task)

## Execution notes / decisions

- 2026-05-29: Plan drafted from issue + code-memory mapping. G1 grounding pending after plan commit. User explicitly requested branch creation, implementation, push/PR, PR test/fix, and PR update/comment.
