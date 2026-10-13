# Silk support for ModernBERT and Laya

Branch: `khurram/silk-laya`. No tracker issue. This file is the plan; `docs/plan/silk-modernbert-laya.md` was folded in and deleted.

As each task executes, if deferred work or a concern appears outside this plan, create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record it in the log below. Do not hold it in memory.

## Problem

`--silk` (`Silk.NET.OpenCL` 2.23.0, hand-authored OpenCL C) runs MiniLM, DistilBERT, and DistilBERT+SST-2. `--ilgpu` also runs ModernBERT-large and the Laya decision head. The CLI allow-lists are the contract:

- `GpuModes` — `minilm`, `distilbert`, `distilbert_sst`, `modernbert`, `laya` (`samples/NivaraInference/Program.cs:38-44`)
- `SilkModes` — the first three only (`Program.cs:51-56`)

Laya's GPU path is not a separate architecture. `Laya.LoadGpuModels` constructs `ModernBertGpuRunner` with prefix `"encoder"` and hangs `LayaHeadGpuRunner` off `HiddenOnDevice` (`samples/NivaraInference/Laya.cs:239-275`). A Silk Laya without a Silk ModernBERT encoder cannot exist. The two ship as one plan and two implementation units: encoder first, head second.

SmolLM (#628), Qwen (#629), and vision (#630) are out of scope. They have no ILGPU runner to port.

## Requirements

Explicit:

- `modernbert --silk` and `laya --silk` in the modes ILGPU already accepts: default, `benchmark`, `compare`. Not `compare_diag` — ILGPU rejects that too (`GpuModeCases` in `InferenceCliArgumentTests`).
- F32 only. Mutual exclusion with `--ilgpu` already exists; do not alias `--gpu`.
- Operation-for-operation with the ILGPU runners, so the existing GPU-vs-CPU gates transfer: `|gpu−cpu| ≤ 1e-3·(1+|cpu|)` for ModernBERT hidden state, and the Laya head compared field-for-field against the in-process CPU head.
- `modernbert --silk benchmark` must accept the same lengths as `modernbert --ilgpu benchmark`, including 4096. The 2048 CPU cap exists because the CPU materialises a dense `[L, L]` mask; both GPU backends carry the band inside the kernel.

Implicit:

- Sample-scoped. No `src/Nivara` change. The ILGPU path is the reference, not a thing to rewrite.
- New kernels are ports of the ILGPU kernels of the same shape, sharing `GEMM_ROW4_ACCUMULATE`, with `#pragma OPENCL FP_CONTRACT OFF` left in force.
- RoPE cos/sin come from the CPU `RotaryEmbedding<float>`, not from a device math library. The ILGPU runner states why: a libm-vs-`XMath` ulp would be the largest single error term in the gate (`ModernBertGpuRunner.cs:26-29`).
- Laya's four act-head features stay on `LayaHeadScoring` on the host. Do not add a device softmax, top-k, or entropy kernel.
- A checkpoint that is not a Laya model is rejected at head construction (`type_emb.weight` missing), and the encoder is disposed on that path. Same trap the ILGPU loader already handles (`Laya.cs:251-257`).

Acceptance:

- `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --silk-kernels` covers the new kernels, including one banded attention cell. Byte-identity where the kernel is pure multiply-and-add; the model's numeric bound where it calls `exp` or `sqrt`.
- `modernbert --silk compare` and `laya --silk compare` pass on a machine that has the checkpoints and an OpenCL GPU. These are manual gates, same as the ILGPU compares. Do not put them in the filtered NUnit suite.
- `InferenceCliArgumentTests` pins the widened allow-list and the rejects that stay rejects.

## Design

Seven kernels, added to the one translation unit in `SilkKernels.cs`. Each is a port of the ILGPU kernel `ModernBertGpuRunner` or `LayaHeadGpuRunner` actually loads. Do not invent a fusion the gate cannot compare.

| kernel | port of | who launches it | gate |
|---|---|---|---|
| `layer_norm` | `ElementwiseKernels.LayerNorm1D` | both | numeric bound (`sqrt`) |
| `add` | `ElementwiseKernels.Add` | both | byte-identity |
| `add_bias` | `ElementwiseKernels.AddBias` | Laya type-embedding, in place | byte-identity |
| `gemm_row4` | `GemmKernels.TiledGemmKernelRow4` | ModernBERT bias-free projections | byte-identity |
| `split_columns` | `ElementwiseKernels.SplitColumns` | ModernBERT QKV only | byte-identity |
| `rotary` | `ElementwiseKernels.Rotary` | ModernBERT q and k, in place; v is not rotated | byte-identity (tables are inputs) |
| `geglu` | `ElementwiseKernels.GeGlu` | ModernBERT FFN; **first** half is the GELU | numeric bound (`exp`) |

Reuse, do not re-port:

- `gather_rows`, `gemm_row4_bias`, `gemm_row4_relu`, `gemm_row4_gelu`, `gemm_row4_qkv` — Laya's head already matches these signatures, including the real biases.
- `batched_attention` — already takes `band` and already narrows the three passes to `[qPos-band, qPos+band]` (`SilkKernels.cs:345-400`). ModernBERT `HeadDim` is 1024/16 = 64 (`ModernBertModel.cs:21-22, 63`). Laya's head is `hidden/64` = 64 at hidden 1024 (`LayaHeadGpuRunner.cs:134-142`). Both sit on `MaxHeadDim` 64, the bound DistilBERT already drives. Do not raise it.

Rejected:

- **Zero-buffer reuse instead of `layer_norm` and `gemm_row4`.** `layer_norm_residual` with a zero partner reduces `a[c]+b[c]`, not `x[c]`. `gemm_row4_bias` with a zero bias stores `acc+0.0f`, which canonicalizes `-0`. The gate compares kernels of the same shape (`docs/SILK.md` §15). A different instruction stream fails that comparison for a reason that is not a rounding story. One macro, a plain store beside the epilogues, is the smaller honest diff.
- **Route ModernBERT QKV through `gemm_row4_qkv` and skip `split_columns`.** ModernBERT's fused `Wqkv` is written row-interleaved (`[q|k|v]` inside each row) and then split. Laya's `in_proj` is the other layout: three contiguous blocks, which `gemm_row4_qkv` already writes. Collapsing the two would make the ModernBERT gate compare a layout the ILGPU runner does not produce. The comment on `SplitColumns` (`ElementwiseKernels.cs:35-48`) is the bug this avoids: a contiguous sub-view of a row-interleaved GEMM is right for row 0 and wrong after it.
- **A Silk-side RoPE.** Rejected for the reason the ILGPU runner already recorded.
- **Duplicated `RunSilk` methods.** The BERT path already learned this: `OpenSilkEncoder` / `OpenIlgpuEncoder` share one set of mode bodies so a fix cannot be forgotten in one backend (`docs/SILK.md` §14). ModernBERT and Laya do not have that seam — `ModernBert.RunGpu` constructs `ModernBertGpuRunner` directly (`ModernBert.cs:576-579`). Add a session, not a second copy of the benchmark table.

Session seam, internal to the sample, not a public API. `HiddenOnDevice` stays backend-private (`MemoryBuffer1D` vs `nint`). A shared interface that exposed either would leak the runtime. The Laya session owns the pair and passes the buffer across inside one backend, which is what `ForwardGpu` does today (`Laya.cs:265-275`).

```csharp
interface IModernBertGpuSession : IDisposable
{
    string DeviceName { get; }
    float[] Forward(int[] tokenIds, int validLength);
}

interface ILayaGpuSession : IDisposable
{
    string DeviceName { get; }
    LayaHeadOutput Forward(int[] ids, int validLength, LayaQuestionType type, IReadOnlyList<int> markers);
}
```

`ModernBert.RunGpu` / `BenchmarkGpu` / `CompareGpu` and the Laya GPU entry points take a session factory. Add `modernbert` and `laya` to `SilkModes` with `["", "benchmark", "compare"]`.

Workspace. The BERT Silk runner hard-caps at batch ≤ 8, seqLen ≤ 128 (`SilkBertEncoderRunner.cs:19-29`). That cap is the BERT scenario's maximum, not a property of the runtime. ModernBERT must grow the way `GpuBuffers.Ensure` does (`GpuBuffers.cs:206-221`): allocate on first use, release and reallocate when the next forward is longer, never shrink. `SilkRuntime` has `Allocate` and no grow helper — add one next to it, same null-then-replace shape. Cap is `config.MaxPositionEmbeddings` (8192), the same cap `ModernBertGpuRunner.seqCap` enforces. Laya's `max_len` 512 fits inside that.

Rough workspace at S=4096, hidden 1024, intermediate 2624: five hidden-sized buffers, two packed-QKV buffers, one `2×intermediate` buffer, one gated buffer, two RoPE tables. About 300 MB, on top of the same F32 weight upload ILGPU already performs. Laya's checkpoint is 421,293,830 F16 parameters, 1.685 GB as F32 (`docs/LAYA.md`). Device memory is not a new question; if ILGPU holds it, Silk holds the same tensors. Counted from `ModernBertGpuRunner.EnsureWorkspace`, not measured on a device.

Per-layer band. Pass `config.Band(index)` into the existing attention kernel. Global layers pass `-1`. Do not leave the runner on `GlobalAttentionBand` — that would compute the right answer (`Keep` already suppresses out-of-band keys) and miss the reason the 4096 row exists.

Silent wrong answers the ports must not "simplify":

- GeGLU activates the first half. Reversing the halves still looks plausible (`ElementwiseKernels.cs:241-248`).
- Layer 0 has no `attn_norm`. The CPU makes it an identity. A zero gamma would not (`ModernBertGpuRunner.cs:242-248`).
- Laya norms are biased. The encoder's shared zero beta is a silent wrong answer. Eps is `1e-5`, not `config.NormEps` (`LayaHeadGpuRunner.cs:27-34, 134-145`).
- The residual is added by the caller, not inside attention.
- RoPE rotates q and k only.

### CLI

- `TryParseSeqLengths` treats `useGpu` as "may exceed 2048" (`Program.cs:669-687`). Silk must take the same branch, and the error text must name both backends. A `--silk` user sent to `--ilgpu` is a wrong instruction.
- Help text at `Program.cs:251-264` currently describes Silk as the three BERT names. Update it in the same change as the allow-list.
- `InferenceCliArgumentTests.Silk_OnModelWithoutSilkKernels_Rejected` pins the allow-list string. Update that string. Add the counterpart of `QwenAcceptedModes_ReachTheModelFileLookup`: `modernbert --silk` and `laya --silk` must reach the model-file check, not the model-scope reject. Add `modernbert --silk compare_diag` to the mode-reject cases, matching the ILGPU case.
- F32 reject and mutual exclusion already exist. Extend the precision reject test to one of the new models so a future allow-list edit cannot drop it silently. One case is enough.

### Gate

Extend `SilkKernelsGate` rather than adding a second program.

- New shape: ModernBERT at seq 128, hidden 1024, 16 heads, headDim 64, intermediate 2624. Seq 128 is enough for the kernel gate; the long sequence is an end-to-end benchmark, not a kernel-identity question. headDim 64 is already the compiled local-memory bound.
- Byte-identity: `gemm_row4`, `split_columns`, `add`, `add_bias`, `rotary`.
- Numeric bound: `layer_norm`, `geglu`.
- One `batched_attention` cell with `band = 64` (ModernBERT's `SlidingWindow` at the default `local_attention` 128). The current gate launches only `GlobalAttentionBand` (`SilkKernelsGate.cs:388, 403`), so the loop-bound path has never been compared. That cell is part of this work, not a follow-up: without it the 4096 benchmark can be fast and wrong in a way the global cell will not catch.
- Timing column stays not-a-verdict.

End-to-end compares stay manual, Release, AC power, same protocol as the ILGPU rows. Do not quote a Silk-vs-ILGPU millisecond from a different session; the ratio is the comparable quantity (`docs/ACCELERATION.md` §1).

## Docs to update

These land in the docs commit, after the code, so they describe what shipped. Do not invent a timing table. Do not add a gated `csharp` fence (ADR-005); `SILK.md` already has one ungated illustrative fence for `KernelNames` — match that or stay in prose and tables.

### `docs/SILK.md` (Part 2 is the source map; edit the sections, do not add a third part)

- **§10.** Scope sentence today names MiniLM, DistilBERT, DistilBERT+SST-2. Widen it to ModernBERT and Laya, still F32-only, same modes, still mutually exclusive with `--ilgpu`. Add rows to the file table for `SilkModernBertGpuRunner`, `SilkLayaHeadGpuRunner`, and the two session interfaces. Do not rewrite the BERT runner row.
- **§11.1.** `KernelNames` list and the kernel table gain the seven kernels, with abridged signatures in the same columns (`kernel | signature | purpose`). Say which model launches each. Note that `gemm_row4` stores the accumulator and takes no bias, and that `geglu` activates the first half.
- **§11.3.** The macro paragraph says "the three Row4 GEMM epilogues". `gemm_row4` shares that macro and only the store differs. Say so. Do not weaken the `FP_CONTRACT OFF` / mul-add claim.
- **§12.** Document the grow helper next to `Allocate`: release and replace when the next forward is longer, never shrink. Same shape as `GpuBuffers.Ensure`.
- **After §13.** New sections in the §13 style (uploads, workspace, forward, the traps) for the ModernBERT runner and the Laya head. Do not fold them into `SilkBertEncoderRunner`. Name the prefix (`"model"` vs `"encoder"`), layer-0 identity, per-layer band, CPU-uploaded RoPE tables, biased Laya norms, eps `1e-5`, and host-side `LayaHeadScoring`.
- **§14.** `SilkModes` gains `modernbert` and `laya`. Gate paragraph gains the new byte-identity cells, the numeric-bound cells, the ModernBERT shape, and the `band = 64` attention cell. Compare paragraph gains `modernbert --silk compare` and `laya --silk compare` at the same 1e-3 bound.
- **§15.** Step 1 currently says only `SilkKernels.Source` and `SilkBertEncoderRunner` change. A new model that is not BERT-shaped adds a runner file; say that, using ModernBERT/Laya as the worked example. The rest of the checklist still holds.

### `docs/ACCELERATION.md` (index and story; measurements only when measured)

- **Status paragraph** under "GPU Acceleration — ILGPU on Intel Arc", and the **§1 "Second OpenCL backend" scope bullet**: silk covers MiniLM, DistilBERT, DistilBERT+SST-2, ModernBERT, and Laya. Do not claim a speed win.
- **Kernel-set bullet:** eight kernels becomes the original eight plus the seven ports. Point at `SILK.md` §11 for the table. Do not paste the kernel bodies.
- **Gates bullet:** add the new `--silk-kernels` cells (including `band = 64`) and the two end-to-end compares. Timing column stays not-a-verdict.
- **Do not add a Silk-vs-ILGPU millisecond row** for ModernBERT or Laya unless a same-session A/B was actually run on AC power. The doc's own rule: the ratio is the comparable quantity; do not mix sessions; battery numbers are not quotable. If the compares ran and no A/B was timed, say the compares passed and the timing was not measured.
- **§2 index.** The silk row already points at `SILK.md` Part 2. No new document, no new row, unless the section title stops covering the new runners — it does not.
- **"Not done / planned".** One line each, home document, issue number, matching the list's format: #628 SmolLM `--silk`, #629 Qwen `--silk`, #630 vision `--silk`. Home is the issue plus this plan's out-of-scope note. Do not retitle #457; it is the CPU conv gap and is already listed.

### Model pages (same commit, shorter)

- `docs/MODERNBERT.md` — `--silk` is a second backend with the same contract as `--ilgpu` (modes, 4096, F32, band inside the kernel). Detail stays in `SILK.md`.
- `docs/LAYA.md` — `--silk` runs the encoder and the head; the gate is still GPU-vs-CPU against the in-process CPU head. Name the biased-norm and eps traps only if the page's GPU section does not already.

## Decision gates

None that block execution. Two offers, not gates:

- File a tracker issue for this plan. Not filed. The human did not ask for one.
- Cut the 4096 benchmark row out of the Silk contract. Not recommended: that row is why the GPU path has no dense-mask cap.

## ADR and roadmap

- No ADR in `docs/adr/` constrains the kernels. ADR-005 applies only if a gated `csharp` fence is added. This plan does not add one.
- `docs/plan/` holds Polars, Arrow, and AI-stack roadmaps. None own the GPU sample. The live index is `docs/ACCELERATION.md` §1 and `docs/SILK.md` §15. This does not reopen #450 (fused pre-norm) or #447 (the band already landed; this plan only passes it through).

## Blast radius

- Files: `SilkKernels.cs`, `SilkRuntime.cs`, new `SilkModernBertGpuRunner.cs`, new `SilkLayaHeadGpuRunner.cs`, `ModernBert.cs`, `Laya.cs`, `Program.cs`, `SilkKernelsGate.cs`, `InferenceCliArgumentTests.cs`, `docs/SILK.md`, `docs/ACCELERATION.md`, `docs/MODERNBERT.md`, `docs/LAYA.md`.
- Symbols: `SilkKernels.KernelNames`, `SilkModes`, `ModernBertGpuRunner.ForwardOnDevice`, `LayaHeadGpuRunner.Forward`, `Laya.LoadGpuModels`, `TryParseSeqLengths`.
- Downstream: CLI dispatch for `modernbert` and `laya` only. No `src/Nivara` caller. `IBertEncoderRunner` is not involved.
- Tests: `InferenceCliArgumentTests` (no device). `SilkKernelsGate` (device, Release, not NUnit). Manual compares.

## Implementation tasks

### Task 1: Kernels

**Goal.** The seven kernels exist in `SilkKernels.Source`, `KernelNames` matches the translation unit, and geometry helpers for the new launches are pure functions.

**Constraints.** Share `GEMM_ROW4_ACCUMULATE`. Plain `gemm_row4` stores the accumulator, it does not add a bias. `geglu` uses the existing `gelu_exact` and activates the first half. `rotary` is half-split and writes both halves of the pair from one work item. In-place aliasing matches the ILGPU kernels: `add` may write back to `a`, `rotary` may write back to `x`, `layer_norm` may write back to `x` only where the ILGPU runner does (embedding norm). Do not change the eight kernels the BERT path already gates. OpenCL parameter names must not shadow builtins (`cos` / `sin`); the tables are inputs under other names.

**Acceptance.** A device-free check can see every new name in `KernelNames` and in `Source`. `GemmGlobalCols` is not re-derived.

**Depends on.** Nothing.

### Task 2: Gate cells

**Goal.** `--silk-kernels` compares each new kernel to the ILGPU original, plus one banded attention cell (`band = 64`, numeric bound).

**Acceptance.** Byte-identity cells fail the process on any bit difference. An unavailable device still exits `UNBUILT` (10). Timing is printed and is not a failure.

**Depends on.** Task 1. Ask before running it; it needs a GPU and Release.

### Task 3: ModernBERT runner and CLI

**Goal.** `SilkModernBertGpuRunner` mirrors `ModernBertGpuRunner`, including the `prefix` parameter, `Forward` / `ForwardOnDevice`, grow-on-demand workspace, CPU-uploaded RoPE tables for both thetas, layer-0 identity, and per-layer band. The ModernBERT GPU entry points take a session factory. `modernbert --silk` selects the Silk session.

**Constraints.** Weight keys are the ones `LayerWeights` already resolves (`{prefix}.embeddings.tok_embeddings.weight`, `{prefix}.layers.{i}.attn.Wqkv.weight`, and the rest at `ModernBertGpuRunner.cs:128-143, 393-410`). Transpose with the same `GpuBuffers` helper the ILGPU runner uses. Dispose is unconditional for every buffer the constructor allocated.

**Acceptance.** `modernbert --silk compare` passes the existing GPU-vs-CPU bound, when the checkpoint and a GPU are present. `modernbert --silk benchmark --seq 4096` is accepted by the CLI (running it is a manual measurement, not a unit test). `modernbert --silk compare_diag` is rejected. `modernbert --silk --precision bf16` is rejected.

**Depends on.** Task 1. Task 2 should be green before this is called done.

### Task 4: Laya head and CLI

**Goal.** `SilkLayaHeadGpuRunner` mirrors `LayaHeadGpuRunner`. The Silk Laya session owns the encoder (prefix `"encoder"`) and the head, and disposes the encoder if head construction throws.

**Constraints.** Biased norms, eps `1e-5`, no RoPE, no band (`GlobalAttentionBand`), QKV via the existing `gemm_row4_qkv` with the real bias, features via `LayaHeadScoring`. Do not add kernels in this task.

**Acceptance.** `laya --silk compare` passes the existing GPU-vs-CPU head gate, when the checkpoint and a GPU are present. A stock encoder checkpoint (no `type_emb.weight`) throws the existing missing-tensor message and does not leak the encoder. `laya --silk compare_diag` is rejected.

**Depends on.** Task 3.

### Task 5: Docs

**Goal.** The edits under "Docs to update" above. `SILK.md` and `ACCELERATION.md` are required, in the shapes that section names. `MODERNBERT.md` and `LAYA.md` get the short pointers.

**Depends on.** Tasks 3 and 4, so the docs describe what landed. The kernel table in `SILK.md` §11 can be drafted with Task 1 and finished here.

## Verification

- `dotnet build Nivara.slnx` after each task.
- `dotnet test -c Release --filter "FullyQualifiedName~InferenceCliArgumentTests"` after Task 3 and again after Task 4. One fixture. Ask before any wider suite.
- Ask before `--silk-kernels`, and before either `--silk compare`. Both need a GPU, Release, and AC power. Battery numbers are not quotable.

**Run so far (2026-10-13).** `--silk-kernels` ran on the GPU: **43/43 bit-identical**, but on battery, so the gate's timing column is not quotable. CLI arg tests pass (39). `modernbert --ilgpu benchmark`, `modernbert --silk benchmark`, `laya --ilgpu benchmark`, `laya --silk benchmark` all ran on AC power, back-to-back per model in one session, and the Silk-vs-ILGPU ratios are recorded in `docs/ACCELERATION.md` §1, `docs/MODERNBERT.md`, and `docs/LAYA.md` (ModernBERT: parity at 128, ~0.75× at 4096; Laya: 0.81–0.93× per question). **Compares (AC, same day):** `modernbert --silk compare` passes at maxAbs 2.861E-005 / maxRel 5.577E-004 / cosine 1.0000001 — identical to the ILGPU gate — and `laya --silk compare` passes at 1.110E-005 = 1.1% of the 1e-3 bound, mask clamp PASS on both. Every planned verification has now been run; nothing remains before G2.

## Planned commits

1. `docs: plan Silk support for ModernBERT and Laya` — this file only.
2. `feat: port ModernBERT/Laya elementwise and bias-free GEMM kernels to Silk` — Task 1, and the gate cells (Task 2) if a device run is available. Split the gate if it is not.
3. `feat: modernbert --silk` — Task 3.
4. `feat: laya --silk` — Task 4.
5. `docs: record --silk ModernBERT and Laya` — Task 5 (`SILK.md`, `ACCELERATION.md`, `MODERNBERT.md`, `LAYA.md`).

No trailers. No `git commit -S`. Do not push.

## Grounding

Code claims are from the symbols and lines cited above. No microsoft-learn claim is load-bearing: this is a port of in-repo ILGPU kernels to OpenCL C. Microsoft Learn's OpenCL page points at the Khronos registry, which `docs/SILK.md` already cites. The HLSL `GroupMemoryBarrier` pages are the wrong backend; the existing GEMM macro already uses OpenCL `barrier(CLK_LOCAL_MEM_FENCE)`.

Assumptions:

- ModernBERT-large and the Laya checkpoint are both present on this machine, so Tasks 3 and 4 ran end to end, including both `--silk compare` gates (passing, see Verification). The contingency — stop at the kernel gate plus CLI argument tests and report the compare as not run — did not apply.
- `a + 0.0f` is not bit-identical to `a` for `-0`. That is why `gemm_row4` and `layer_norm` are dedicated ports. IEEE rule, not measured this session.
- Workspace at S=4096 is about 300 MB. Counted, not measured.

## GitHub issues log

- [x] #628 — SmolLM `--silk` (filed before execution; out of scope)
- [x] #629 — Qwen `--silk` (filed before execution; out of scope)
- [x] #630 — vision `--silk` (filed before execution; out of scope)
