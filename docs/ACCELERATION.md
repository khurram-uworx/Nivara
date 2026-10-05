# Acceleration

This is the **acceleration index** — all performance and resource-budget topics in one place. For model-specific architecture and usage, see the model pages:

- [DISTELBERT.md](DISTELBERT.md) — DistilBERT, MiniLM, SST-2
- [MODERNBERT.md](MODERNBERT.md) — ModernBERT-large
- [SMOLLM.md](SMOLLM.md) — SmolLM-135M
- [QWEN.md](QWEN.md) — Qwen2.5-0.5B-Instruct
- [LAYA.md](LAYA.md) — Laya decision head
- [VISION.md](VISION.md) — MobileNetV2, ResNet-18

Related: probe verdicts in [docs/ILGPU.md](ILGPU.md), backend case studies in `tests/Nivara.GpuProbe/`, streaming docs in [docs/STREAMING.md](STREAMING.md).

---

# GPU Acceleration — ILGPU on Intel Arc

Status: **Implemented, gated, and measured** (2026-09-19; DistilBERT via PR #436, MiniLM via `khurram/minilm-gpu`, ModernBERT via issue #449 on `khurram/449`).

## 1. Measured results

All F32-only, OpenCL iGPU via ILGPU 1.5.3. Sample-scoped (no `src/Nivara`, `Nivara.Extensions`, or `src/Nivara.Gpu` changes). 128 tokens, 3-pass warmup + 10 timed, AC power (GPU↔CPU-Nivara same-session; PyTorch = recorded CPU baseline):

| scenario | Nivara GPU (iGPU) | Nivara CPU | PyTorch CPU | vs Nivara CPU | vs PyTorch |
|---|---|---|---|---|---|
| distilbert | **65.3 ms** (62–73) | 194.7 ms (152–232) | 35 ms | **~3.0× faster** | ~1.9× slower |
| distilbert_sst | **64.0 ms** (61–71) | 166.4 ms (134–208) | 35 ms | **~2.6× faster** | ~1.8× slower |
| minilm | **26.8 ms** (25–29) | 76.3 ms (49–102) | 11 ms | **~2.9× faster** | ~2.4× slower |

The "vs PyTorch" column is **shape-specific, not a standing property of the iGPU.** At 128 tokens
and 66 M parameters a 128-EU iGPU is latency- and bandwidth-bound while 16 CPU threads have work to
fill. The same accelerator **ties** PyTorch at Laya's shape — 512 tokens, 1.4 B parameters, iGPU
3.00 s against PyTorch 2.97 s — where quadrupling `M` in every GEMM finally gives the device enough
parallel work per weight load. One shape is one data point and the §1 rows are unchanged, but read
the deficit as "at these shapes" rather than as a property to assume anywhere else; measure at the
shape you intend to ship. Detail in [docs/LAYA.md](LAYA.md).

**M2 — kernel fusion (issue #437) landed 2026-09-19** on `khurram/lazystream-gpu` (PR targets `khurram/minilm-gpu`). Same session, AC power:

| scenario | pre-M2 (M1) | post-M2 | Δ |
|---|---|---|---|
| minilm | 26.8 ms | **24.3 ms** (23–25) | **1.10×** |
| distilbert | 65.3 ms | **63.1 ms** (61–67) | **1.04×** |
| distilbert_sst | 64.0 ms | **63.7 ms** (62–67) | ~1.00× |

What fused (all sample-scoped; parity gates byte-identical at every step — MiniLM maxAbs 1.87e-5 / DistilBERT 1.53e-5 / SST 3.8e-6 + argmax 8/8): GEMM epilogue bias (+ exact-GELU for fc1, ReLU for the head pre-classifier) via lean per-activation sibling kernels; `q/k/v` merged into **one** GEMM writing a block-separable destination (`[q-block | k-block | v-block]`, dense SubViews for the unchanged attention kernel); residual-add folded into LayerNorm; word/position/token-type embedding sums fused into one gather; seqLen-deterministic posIds cached. Dispatches went **~113–119 → 44–48** per forward.

**Honest outcome:** the pre-M2 launch model was wrong. Removing ~70 dispatches moved MiniLM 26.8 → 24.3 ms, i.e. **~30 µs per dependent kernel** (not the ~0.18–0.2 ms first estimated — the model was off ~6× because GEMM + non-GEMM compute dominate at these shapes). The #437 ">2× for MiniLM" acceptance (~13.4 ms) is **not reachable via fusion on this iGPU** — it would need ~420 more dispatches removed, and only ~50 exist. The real lever is GEMM *throughput* (§5 item 2); even an optimistic 2–3× GEMM speedup lands MiniLM near ~22 ms at batch 1. The remaining micro-fusions were completed because they were already designed; the numbers above are the honest result.

**Battery caveat confirmed again:** an intermediate session on battery produced contaminated 25.6–33 ms MiniLM numbers drifting downward as the charge drained — only AC numbers are valid. Do not benchmark this path on battery.

Correctness gates — **all PASS**:
- `distilbert --gpu compare`: final hidden `[128,768]` vs same-process CPU — maxAbs 1.53e-5, maxRel 3.24e-6, **0/98304 violations** (bound `|gpu−cpu| ≤ 1e-3·(1+|cpu|)`).
- `distilbert_sst --gpu compare`: logits vs CPU maxRel 6.7e-7, vs PyTorch fixture maxRel 5.8e-7, **argmax 8/8** both (CPU's own doc'd bound vs HF is 9.5e-7 — same class).
- `minilm --gpu compare`: hidden `maxRel 1.04e-5`, pooled-embedding `maxRel 1.5e-7`, **0/245760 violations**, minimum pooled-embedding cosine **1.000000**.
- `--precision bf16|fp16` + `--gpu` → clear F32-only rejection (exit 1).

### ModernBERT GPU (issue #449)

`ModernBertGpuRunner` + `modernbert --gpu`. Same device (`Intel(R) Graphics`), same ILGPU 1.5.3 OpenCL path, still sample-scoped. This is the first runner whose architecture shares **no** structure with DistilBERT or MiniLM: rotary instead of learned positions, a gated feed-forward instead of a plain FFN, pre-norm instead of post-norm, and **per-layer attention geometry**.

The gate is GPU-vs-CPU in-process, so it needs no PyTorch fixture (Phase 1 had already pinned the CPU encoder to HuggingFace, so this pins the GPU runner transitively). Fixture seqLen 128, 26 valid positions:

| region | maxAbs | maxRel | cosine | violations |
|---|---|---|---|---|
| valid (26 x 1024 = 26,624 values) | **2.861E-005** | 5.577E-004 | 1.0000001 | 0 |
| padding (104,448 values) | 6.866E-004 | 4.867E-003 | — | 0 |

Forward: **415–449 ms GPU vs a 7.3–7.5 s CPU reference** (the CPU figure is a 28-layer float forward with no batching, so it is not a throughput claim — see "correctness, not speed" below). Benchmark rows scale with sequence length, as they must: 128 → 400 ms, 512 → 2.30 s, 2048 → 19.6 s, 4096 → 63.0 s.

Those rows use **1 warmup + 3 timed passes, median reported**, not the 3 + 10 of the §1 table — at S=4096 a single pass is ~60 s, so the full protocol would cost minutes per row. They are therefore not comparable to the §1 numbers, which is why they are not in that table. The mode prints this caveat itself.

**Three kernel changes and one shared-kernel fix:**
- `BatchedAttention` grew a **`band` parameter** (negative = global, the CPU's own `ModernBertMasks.Build` convention) folded into one `Keep` predicate at the three sites the padding mask was already applied at.
- **`max == -inf → zeros` clamp**, mirroring `GradKernels.cs:487`. Not theoretical: the band intersects padding, so query rows past `valid_length + band` have *no* visible key, and without the clamp `s - max` is `NaN` and the whole residual stream below that row is poisoned. This is the hazard **#448** documented and `docs/LAYA.md` already flagged as live; #448 has since closed the CPU half of it structurally (§5.2 item 3) and this GPU clamp was the prerequisite half. `BertEncoderGpuRunner` passes `GlobalAttentionBand = -1`, so DistilBERT / SST / MiniLM are untouched and their gates came back **identical to the digit**.
- **The `[B, H, S, S]` score spill is gone.** `headDim` probability-weighted V rows accumulate in a shared-memory tile and `attnOut` is written once. Same three score recomputes and the same ascending per-`d` `j` loop, so it is **bit-identical** while removing 16.8 MB at S=512, 268 MB at S=2048 and 4.3 GB at S=8192. That last number is the one that matters: it is what makes the 4096 benchmark row above possible at all.
- New: `ElementwiseKernels.Rotary`, `ElementwiseKernels.GeGlu`, `ElementwiseKernels.SplitColumns`; the duplicated A–S 7.1.26 GELU polynomial promoted to one scalar `ElementwiseKernels.GeluExact` (AGENTS.md rule 8).

**This issue buys correctness, not speed — and the issue's premise was stale.** #449 expected attention to become the bottleneck. `docs/LAYA.md` (merged in PR #459, four hours after the issue was filed) had already measured GEMM at **~99.9% of the arithmetic** with attention at 0.13% at S=512, and named **#440** (tile-32 / 2x2 GEMM) the leading open item. *(That nomination was later measured and did not hold — see §5.2. And the arithmetic-share framing was itself the deeper error: a leg profile measures `BatchedAttention` at 57% of the Laya forward, not 0.13%.)* ModernBERT has 1.4 B parameters against DistilBERT's 66 M, so it is a GEMM-throughput problem exactly as DistilBERT was. The value delivered here is that the architecture is *portable at all* — a banded RoPE'd gated pre-norm encoder now runs on the accelerator and is pinned to the CPU encoder — not that it is faster.

**The Laya head was out of scope for #449.** It has since landed on CPU as issue **#460** — `LayaDecisionHead<T>`, gated against the `laya==0.3.20` wheel, reflection in [docs/LAYA.md](LAYA.md). It reached the GPU as issue **#462** (2026-09-29, `khurram/462`): the "zero new kernels" claim held, and it is gated GPU-vs-CPU against the CPU head rather than against PyTorch directly — `laya compare` already pinned the CPU path to the wheel, so pinning the GPU runner to it pins the GPU runner transitively, at no second wheel run. The seam onto this runner is additive: `ForwardOnDevice` plus `HiddenOnDevice`/`HiddenRows`, leaving the three existing `modernbert` call sites untouched.

## 2. What shipped — architecture and decisions

All GPU code lives in `samples/Nivara.Samples/Gpu/`:

| file | contents |
|---|---|
| `IlgpuRuntime.cs` | context/accelerator/stream lifecycle; device select (`CL_DEVICE_TYPE_GPU` + Intel vendor, **asserted — no CPU fallback**); persistent buffer upload/download helpers |
| `GemmKernels.cs` | **Row4** register-blocked 1×4 tiled GEMM (local-memory staging, `TiledGemmKernelRow4`) — the keystone; used for every matmul (q/k/v/o, lin1, lin2, head). M2 added the fused-epilogue siblings (`Row4Bias`/`Row4Gelu`/`Row4Relu` — bias added in the register accumulators, GELU/ReLU folded into fc1 / head pre-classifier) and `Row4Qkv` — one packed `q/k/v` GEMM with a block-separable destination |
| `AttentionKernels.cs` | fused 12-head score+scale+mask+row-softmax+weighted-V (`XMath.Exp`) — unchanged by M2 (consumes the q/k/v SubViews as dense row-major views) |
| `ElementwiseKernels.cs` | LayerNorm row-reduce, GELU (direct A–S 7.1.26 erf poly port of `GradKernels.Erf`, `XMath.Exp`), bias/residual adds, embedding gather. M2 added `LayerNormResidual1D` (residual folded into the LN row-reduction) and `EmbeddingSum` (word/position/token-type gathers + sum in one launch) |
| `BertEncoderGpuRunner.cs` (was `DistilBertGpuRunner.cs`) | uploads weights (transposed once at ctor) by the loader key set — naming/role-resolved via `BertGpuNaming` (`DistilBert` | `Bert`), config-driven from `BertConfig`; runs the full BERT-family encoder forward + optional SST-2 head; returns per-stage readbacks for gating. M2: every projection through a fused epilogue launch, `q/k/v` one packed GEMM (`Wqkv`/`Bqkv` uploads), posIds cached per seqLen, `Add`/`AddBias` launches removed |

Design decisions (deliberate, and worth keeping for the next model):
- **Correctness-first runner**: one kernel launch per operation, everything on one stream, one device sync before readbacks. Fusion/stream tricks were deliberately deferred — the first milestone had to be trivially verifiable.
- **Weights transposed once at upload** so the GEMM consumes plain row-major operands (first-cut correctness over layout cleverness, as the assessment said).
- **Persistent cap-sized workspace** with `Ensure`-only growth; payload copies (mask/ids/posIds/clsIds) go through explicit-length `View.SubView(0, n)` — see the `CopyFromCPU` lesson below.
- **Attention mask semantics copied exactly from CPU**: per-batch column-j `−inf` when `mask[b][j] < 0.5`, identical across query rows.
- **Gate discipline**: hidden states + logits vs same-process CPU forward (identical tokenization) with the per-element bound above; SST-2 argmax 8/8; explicit GATE PASS/FAIL verdicts. Weights were **not** pre-provisioned — provisioned via `hf download` (`samples/data/distilbert{,_sst}`, ~511 MB, now part of the README quick start).

## 3. What we learned using ILGPU (for the next model)

These are the durable, non-obvious lessons from implementing the first model:

1. **`ArrayView.CopyFromCPU<T>(T[])` fills the *whole* view** — its guard is `data.Length ≥ view.Length` and it throws `ArgumentOutOfRangeException('data')` if your payload is shorter than the buffer. Copying a 128-element payload into a 1024-cap persistent buffer crashed the first gate run. Fix: copy into an explicit-length `View.SubView(0, n)`. There is no "copy prefix" overload.
2. **Stream ordering is your responsibility.** Buffer-level `CopyFromCPU` uses the default stream; kernels run on your own stream. Route *all* payload copies through the kernel stream, and add a device sync after bulk ctor uploads, or the uploads may not be visible to kernel launches. (This bit us; it also silently would have produced flaky results.)
3. **Launch/dispatch overhead dominates at inference shapes — but the per-kernel cost is ~30 µs, not the ~0.4–0.7 ms first estimated.** ~100 dispatches per forward at 128-row shapes. M2 (#437) removed ~70 of them and moved MiniLM 26.8 → 24.3 ms / DistilBERT 65.3 → 63.1 ms — consistent with a **~30 µs dependent-kernel latency**. The corrected model: launch overhead was ~3 ms of the baseline, not ~21 ms. Fusion is nearly exhausted; the next lever is kernel *throughput* (GEMM headroom, §5 item 2), not launch count.
4. **A tiled GEMM is a requirement, not an optimization**: the probe's naive gemv rate (6.6 GMAC/s) would take ~825 ms for DistilBERT's matmuls. Row4 (1 thread owns 4 accumulators, one shared A-tile load) hit **303–379 GMAC/s** on all DistilBERT shapes, keeping the per-output-column accumulation order unchanged → bit-identical results to the 1×1 kernel, which kept parity simple.
5. **Double-precision-truth GEMM gates catch real bugs.** A Row4 `colBase` bug (global thread-Y instead of group index) produced maxAbs ~40–77 and was caught by the bounds check. This is why issue #435 (a lasting GEMM correctness+perf harness) is worth doing.
6. **f32 parity between different summation orders has a floor.** Two valid reduction orders disagree at ~4.4e-5…1.6e-4 (K=768…3072), so a `1e-6`-class gate is impossible — the `1e-3·(1+|x|)` bound is right, and real bugs still produce ~40-class outliers that trip it. Measured hidden maxAbs 1.5e-5 is the summation-order floor, not a kernel defect.
7. **`XMath` has no `erf`** — GELU needs an in-kernel polynomial. The direct A–S 7.1.26 port with `XMath.Exp` matched CPU `GeluExact` inside hidden-state parity (maxRel 3.2e-6); no `XMath.Tanh`-based approximation needed.
8. **Battery throttles the iGPU flat** — every shape drops to ~150–160 GMAC/s regardless of size (this finally explained the probe's flat gemv class). **All GPU perf claims require AC power.**
9. **ILGPU deployment is the whole story**: NuGet package = entire install, in-box Windows `OpenCL.dll` + Intel ICD; kernels are C# static methods. Assert `CL_DEVICE_TYPE_GPU` (no silent CPU fallback) and re-run the gates after every driver bump (two compilers in the path: ILGPU JIT + IGC).
10. **iGPU shares DRAM** — the 255.5 MB weight upload is memcpy-class (not a PCIe bottleneck); transposing once at upload costs nothing and buys GEMM coalescing.
11. **Workspace hygiene**: allocate cap buffers once and reuse (`Ensure` only grows); integer payloads need an explicit integer-buffer alloc. Readbacks via `GetAsArray()` return the full cap — always slice by payload length.
12. **Same-process CPU reference beats fixture-only gating**: the compare mode (identical tokenization, `BertEncoder.Forward` vs the GPU runner in one process) caught everything the fixture couldn't (fixtures may be absent).
13. **A pre-fused projection is interleaved per row, not block-concatenated.** ModernBERT ships one `Wqkv` of `[3072, 1024]`, and a row-major GEMM lays it down as row `r` = `[q(r) | k(r) | v(r)]` — *not* all of q, then all of k, then all of v. Contiguous `SubView` offsets therefore return the correct values for row 0 and wrong values for every row after it, which is the worst possible failure shape: the first row looks perfect. `SplitColumns` (`dst[r, c] = src[r, part*blockCols + c]`, one launch per block) walks rows. Note the same buffer needs **no** split on the gate/up side, because `GeGlu` indexes the fused buffer directly — check which of the two layouts the consumer actually assumes before reaching for a split.
14. **The RoPE tables should come from the CPU, not from device `XMath`.** `RotaryEmbedding<T>.GetPositionTables` is now `public` and the runner uploads *its* cos/sin. Recomputing them on the device would be one `XMath.Cos` vs one host-libm ulp per entry, and that would become the largest single error term in the gate — measured against DistilBERT's `maxRel 3.2e-6` that is a three-order-of-magnitude regression, for a difference that has nothing to do with whether the port is correct. Promote the authoritative definition to an accessor rather than duplicating the formula; duplicating it guarantees drift.
15. **ILGPU's `SharedMemory` is per work *group*, not per work item**, and its `Allocate` size must be **statically known**. The first shared-tile attempt gave each thread its own `Allocate<float>(64)` and every thread in the group raced on the same 64 floats — caught immediately as 97809 violations at `maxRel 3.42`, i.e. structurally wrong rather than rounding-level. The tile is 2-D (`[MaxHeadDim rows x AttentionGroupSize columns]`, column `Group.IdxX`, `DenseX(MaxHeadDim)` stride) with the size over-allocated to a compile-time constant because `headDim` is a runtime argument. Query `Accelerator.MaxSharedMemoryPerGroup` at construction — note the property is `MaxSharedMemoryPerGroup`, **not** `MaxLocalMemorySize`. Read the *total* the group stages, not one tile: a kernel that needs both an A tile and a B tile must size the single allocation for both, which is lesson 19's rule.
16. **Verify a suspected kernel defect with an isolated probe before blaming the runner.** `TiledGemmKernelRow4` at 128x1024x3072 was confirmed exact (maxAbs 1.5e-5, 0 bad) on the same accelerator before *and* after the runner constructor, and the runner's own uploaded `Wqkv` buffer was confirmed bit-exact against the host transpose (0/3,145,728 mismatches). Those two probes cost one run and ruled out the two most expensive hypotheses. Build the reference with the *exact* layout the device buffer has, every time.
17. **Diagnostic references are code and they rot like code.** Of five "kernel bugs" chased while implementing #449, **three were faults in the throwaway stage diagnostic**: one concatenated block-major against a row-major device buffer, one interleaved per row against a block-contiguous readback (the same mistake, opposite direction, in the diagnostic written to check the fix for the first), and one omitted the `GeluExact` on a gated activation — which shows up as a suspiciously perfect **100% relative error**, i.e. "the GPU produced ~0 where a value was expected".
18. **The ILGPU stream is in order; do not go looking for a race.** A split→rotary dependency was suspected unordered for a while. `AcceleratorStreamFlags` does not exist in ILGPU 1.5.3, `Accelerator` exposes only `CreateStream()` and `DefaultStream`, and the parameterless `CreateStream()` is what `BertEncoderGpuRunner` and the GpuProbe ILGPU leg have always used. Once the diagnostic reference was fixed the dependency was never a problem. Lesson 2 above is the real ordering rule (route every copy through the kernel stream); "the default stream is unordered" is not one of its failure modes.

19. **At most one `SharedMemory` allocation per kernel body — and note the correction to the narrower rule this lesson used to give.** Building the #440 tile family, every kernel staging a KT-wide A tile against a TileCols-wide B tile returned *plausible* wrong values — `maxAbs` 20–119, the right order of magnitude for a real dot product taken against the wrong elements, so it does not look like an out-of-bounds smear either. ILGPU's OpenCL lowering mis-places the second allocation when the two extents disagree. It is invisible in review: every extent, stride and element index is in bounds and the load destination equals the read source, so the kernel is self-consistent. One variant never got that far — it failed at `LoadKernel`, recorded at the time as a bare `CLException` with no error code, which is part of why it took a bisect across four geometries to attribute. That code is now measured, because the `--gemm` gate reports it (`Describe` in `GemmBenchmark.cs`): **`CL_OUT_OF_RESOURCES`**, raised at kernel creation on a device with shared memory to spare. A lowering bug surfacing as an allocation failure is the kind of report that gets closed as "out of resources, retry later" upstream, so it is worth quoting verbatim in a bug report. **The rule first written here — "the two allocations need matching extents" — was wrong, and the way it was wrong matters more than the bug did.** It rested on `Row4` having escaped because its A tile is 16×16, i.e. square, which happens to match. The issue's own `1x8@KT16` row refutes it: that kernel had a square 16×16 A tile as well, and it did not merely return wrong values, it would not build. So `Row4`'s survival was a property of this ILGPU build and driver, not of its source — an observation about one toolchain cannot be written down as a rule about kernels, and any rule that needs an exemption list rots the moment someone adds the next geometry nobody checked. **Fix: stage every tile through a single allocation.** Two forms work and they are not equivalent in speed. The 1-D `Allocate<float>(n)` with hand-computed 2-D offsets is what #440 was migrated to; re-doing the same migration as one wide `Allocate2D` (`Index2D(16, 80)`, A in columns [0,16) and B in [16,80), the same 1280 floats and the same 5120 B) measured **~20% faster** on the Row4 family and 45% on the 1×1 kernel, and the 1-D form's 10–20% cost sits far outside a run-to-run band of median 1.2% (p90 8.6%). So the six live kernels are on the 2-D form, and the removal of the #468 exposure made the production GEMM *faster*, not slower. No mechanism is claimed for either direction: ILGPU 1.5.3 has no kernel-dump facility on `Context.Builder`, so what the lowering does differently is not observable from here. `SharedMemoryAllocationTests` asserts the count on the compiled IL rather than on source text; stated limit, in the test's own comment — it sees the kernel body, so a kernel that moved its allocation into a helper and called it twice would not be caught, and no such shape exists in this tree. Filed as **#468**; an upstream bug, not a Nivara one, but any future kernel that widens a shared tile will hit it. **The generalisable part is about the gate, not the kernel: the byte-identity check caught this, and the `maxAbs` tolerance would have been argued with.** "Correct to 5.7e-7 on the first shape, wrong by 38 on the next" is a far cheaper thing to act on than "3 of 171 cells exceed 0.001". When a new kernel must reproduce a reference bit-for-bit, assert that — a tolerance band is not a substitute, and a gate that can only be cleared by argument is a gate that will be cleared by accident. That is now enforced rather than advised: every `--gemm` cell asserts its f32 bits against a committed fingerprint (`gemm-f32-baseline.json`, FNV-1a 64 over `BitConverter.SingleToInt32Bits`), which is also what let the migration above be published as a bit-exact no-op — 171 of 171 cells identical — instead of "within tolerance, probably fine". Generalized in [GUIDELINES.md](../GUIDELINES.md) under *Assert Exactness When Exactness Is Available*.

## 4. Where reality diverged from the pre-implementation estimate

Honest delta between the assessment's expectations and the measured outcome:

- **E2E landed at ~65 ms vs the estimated ~15–25 ms.** The GEMM extrapolation was right (Row4 rates × 5.44 GMAC ≈ 20–30 ms); the miss was the *unmeasured* portion — attention + elementwise legs **plus per-dispatch overhead × ~100 dispatches**, first over-estimated at ~0.4–0.7 ms/launch. M2 measured the truth: **~30 µs/dependent kernel**, so the ~70 dispatches fusion removed (MiniLM 26.8 → 24.3 ms) matched the corrected model, and the remaining gap is kernel throughput — not a launch-count problem fusion can fix.
- **M2's >2× acceptance was not reachable.** The #437 ">2× MiniLM" target (~13.4 ms) assumed ~0.19 ms/launch; at the measured ~30 µs it would need ~420 fewer dispatches than exist. Re-scoped to honest ~1.1× with the GEMM-throughput item promoted to the leading GPU follow-up.
- **The "clear win vs PyTorch CPU 35 ms" did not materialize.** The iGPU is ~1.9× *behind* PyTorch's MKL-backed CPU path here (PyTorch starts ~5.6× ahead of Nivara-CPU). We still won ~3.0× over our own CPU, which was the scenario's actual goal — but the assessment's PyTorch-vs-GPU framing was optimistic for a 128-EU iGPU.
- **The keystone GEMM decision gate held exactly as designed**: Row4 measured 303–379 GMAC/s (inside the 0.6–1.6 T MAC/s *good* window) and the OpenVINO fallback was never needed.

## 5. GPU suggestions & follow-ups

Prioritized for the next iterations of the GPU journey (see also [ROADMAP-SUGGESTION.md](ROADMAP-SUGGESTION.md) for the CPU-side picture):

1. ~~**Kernel fusion + lazy stream (M2)**~~ — **DONE 2026-09-19** (`khurram/lazystream-gpu` → PR targets `khurram/minilm-gpu`): every planned fusion landed and the parity gates stayed byte-identical. The lazy-stream half was already true (one stream, one sync before readback). Honest result: MiniLM 26.8 → **24.3 ms**, DistilBERT 65.3 → **63.1 ms** — the per-launch model was corrected by measurement to ~30 µs, so launch-count fusion is nearly exhausted; the remaining gap is kernel throughput.
2. ~~**GEMM headroom is the leading GPU item**~~ — **MEASURED, NULL RESULT 2026-09-29** (`khurram/440`): the tile-32 / 2×2 register-block family was built, gated and measured. **Every geometry lost on every shape.** Best cell 1.01× (`1x8@KT16` on `laya head ff2`, the one shape already known to be weak, and a wash inside run-to-run noise); everything else 0.6–0.85×, with the launch-bound narrow shapes (`head`, `scorer`) regressing hardest as wider tiles mean fewer groups. No winner, so no runner was repointed. **The reason is the useful part.** The family was premised on *shared-memory traffic per MAC* (2×2 cuts it 20% against Row4's 5 reads per 4 MACs), but the binding constraint on this device is *shared-memory capacity per group*. The isolating pair: `2x2@KT16` and `2x2@KT32` have identical blocking factors — identical reads per MAC — and differ only in footprint (4 KB vs 8 KB). **On the nine largest shapes, the ones that carry the throughput, the 8 KB one is 18–25% slower, every one of them.** It weakens to 8–13% on mid shapes and *inverts* on the three launch-bound ones, where 2×2@KT32 is 2–16% faster. So footprint dominates where the device is saturated with work and is not a predictor at all where there are too few work groups to saturate it. `4x2@KT32` agrees with the mechanism rather than with the traffic metric: best traffic ratio in the family (0.75 reads/MAC), worst footprint (12 KB), mid-pack result. The scoped claim is "footprint dominates at saturation", not "footprint predicted the ordering" — the unscoped version is false on 6 of 15 shapes. So the issue's "occupancy headroom exists" was not merely unmeasured but **pointing the wrong way**: a 5 KB → 8–12 KB tile costs more co-resident groups than the 20–40% traffic saving returns. **Consequences:** Row4 stands; #440 closes as a measured negative. The four geometries stay in `GemmKernels.cs` as a labelled negative baseline so the result is reproducible, and the byte-identity assertion stays in the `--gemm` gate (it is what caught #468 — see lesson 19). Row4's ~155–204 GMAC/s is close to what this iGPU offers at these shapes; **the remaining GPU headroom is in `BatchedAttention` (57% of Laya wall-clock), tracked as #447 — not in the GEMM.** Keep the double-truth bounds check - and promote it into issue **#435** (lasting GEMM regression harness). **Amendment 2026-09-30 (#468): this result is now conditioned and has not been re-measured.** Migrating `Row4` off two `SharedMemory.Allocate2D` calls onto one wide `Allocate2D` made it **~20% faster** on these same shapes, while the four #440 geometries stayed on the 1-D `Allocate<float>` form the family was built in — and lesson 19 records that form as costing 10–20% against it. So the ratios above now compare 1-D addressing against 2-D addressing, and the table is left as measured rather than re-recorded, because what this item exists to isolate is *tile geometry*, and re-recording would book the addressing difference as a geometry effect. The isolating pair (identical reads per MAC, differing footprint) is unaffected in kind — both members are 1-D — so the footprint-dominates-at-saturation mechanism is not in question; only the absolute "every geometry loses to Row4" verdict is, and it is now under ~20% of headroom rather than all of it. Re-measuring the family with all five kernels on one addressing form is outstanding work, and the four kernels exist to make it a five-line change.
3. **Transpose-free in-kernel GEMM** — dropped for first-cut correctness; worth revisiting if upload time ever matters (it doesn't here — shared DRAM).
4. **bf16/fp16 GPU** — later decision; neither precision gets **native** support in ILGPU 1.5.3, so F32 stays the GPU path until a promotion-phase decision (possibly re-evaluating the backend for that model class):
   - **BF16: no native *arithmetic* in ILGPU.** "No native type or kernels" is a statement about **ILGPU 1.5.3**, not about the hardware — the distinction matters for the promotion decision, so it is worth being exact. The only proven *compute* route is the "unmerged packed-widen" trick — pack two BF16 values per 32-bit slot, widen to FP32 on load, compute in FP32, pack back. That pays FP32-class compute plus pack/unpack overhead, negating most of BF16's memory/bandwidth win.
     - **Measured on the device (Arc 140T, `CL_DEVICE_EXTENSIONS`): the driver does advertise `cl_intel_bfloat16_conversions`**, giving a native BF16↔F32 conversion (`intel_convert_as_bfloat16_float` and vector forms, plus the reverse). Per the Khronos registry spec it is backed by `OpConvertBF16ToFINTEL` / `OpConvertFToBF16INTEL` and SPIR-V capability `Bfloat16ConversionINTEL` — the same instruction `SpvKernels.Bf16Native` hand-authors on the Level Zero path — and BF16→F32 is lossless. **But the extension is conversion-only:** every function it defines is `float ↔ bfloat16`, with no BF16 multiply, FMA or dot product.
     - **So the raw-OpenCL escape hatch does not open a BF16 door.** A Silk.NET/hand-written-OpenCL route would get a free native widen and still have to compute in FP32, exactly as the packed-widen trick does. There is no reachable BF16 *arithmetic* on this device through OpenCL, only a cheaper way to reach the F32 one. Verified in `tests/Nivara.GpuProbe/SILK.md`; note the extension is spelled `bfloat16`, so a substring search for `bf16` reports this device as having no BF16 support at all.
   - **FP16 (`Half`): software-emulated in ILGPU, native through raw OpenCL.** ILGPU ships a `Half` kernel type but its arithmetic is emulated scalar math — there are **no native vectorized FP16 kernels**, and ILGPU's OpenCL path does not drive `cl_khr_fp16` for you. That is an ILGPU limitation, not a device one, and it is **not** the same gap as BF16. `cl_khr_fp16` is advertised on this Arc 140T and, measured through the Silk.NET leg, native `half` arithmetic builds and runs: `gemv_f16` matched a CPU `Half` accumulator at 0 half-ULP on all 1536 rows and missed the f32-then-round reference by 32427 half-ULP, so the compiler did not promote it. It was also not faster — two same-session runs put the f16/f32 ratio at 0.97–1.07× on both silu and the naive gemv, inside the run-to-run band. Reachable, and not a reason to leave f32 at these shapes. Detail in `tests/Nivara.GpuProbe/SILK.md` §"Precision on this iGPU (BF16 and FP16)".
   - For contrast: OpenVINO's bf16 gemm probe row (86.8 µs, ~5.9 T MAC/s) is genuinely native — it sits on Xe2 **DPAS BF16** hardware instructions that ILGPU 1.5.3 cannot reach. **That advantage is specific to GEMM throughput and is not a blanket win:** the same OpenVINO bf16 leg fails the `silu` correctness gate 402/576 (a real bf16 IR precision defect, pre-existing and still unfiled), so a native-BF16 GEMM row should not be read as a clean bill of health for the leg.
   The F32-only reject (`--gpu` + `--precision bf16|fp16` → clear error) keeps the door clean until that decision.
5. ~~**Second model**: MiniLM~~ — **DONE on `khurram/minilm-gpu`** (PR follows; on the same "one config-driven runner, N encoder models" shape at ~20% of the original effort): the runner is now `BertEncoderGpuRunner`, naming- and config-driven (`BertGpuNaming.DistilBert | Bert`; `BertConfig` ctor), so MiniLM reused the kernel set unchanged. One genuine generalization lesson: MiniLM (BERT-style keys) **does** feed token-type embeddings — the CPU `BertEncoder` defaults `includeTokenTypeEmbedding: true` and PyTorch adds `token_type_embeddings[0]` (all-zero segment ids) — where DistilBERT does not; the runner broadcasts that row 0 when the key is present (key-presence driven, so the DistilBERT path is untouched). Gates: hidden `maxRel 1.0e-5`, 0/245760 violations, pooled-embedding cosine 1.000000; benchmark **26.8 ms** iGPU vs 76.3 ms Nivara CPU (~2.9×), launch-overhead-bound at ≈1.36 GMAC — tracked as **#437**; post-M2 (fused epilogues + q/k/v merge, same runner): **24.3 ms** (see the M2 block in §1) — still latency-bound at batch 1, which is exactly why the GEMM-throughput item (§5.2) is now the leading follow-up. SmolLM once KV-cached decode exists (see [SMOLLM.md](SMOLLM.md)) remains the next, much larger candidate.
6. **Promotion decision**: with real measured numbers in hand, decide whether GPU support moves into `src/Nivara.Gpu` (which backend, which project, bf16, which models). Nothing in core changes until that decision.
7. **Regenerate the `last_hidden_state_py.bin` fixture** (absent in this checkout) to restore the full 3-way GPU-vs-CPU-vs-PyTorch gate on hidden states, matching the SST-2 fixture coverage.
8. **Driver-bump hygiene**: re-run the probe `kernels` gate (seconds) and the two model compares after any Intel driver update (two compilers in the path).
9. **CPU-side performance (independent thread)** — a managed tiled GEMM and/or an opt-in native BLAS bridge would close the ~5.6× Nivara-CPU deficit (possibly beating the GPU at these shapes); options, targets, and the M1/M2/M3 decision gate are captured in ROADMAP-SUGGESTION.md.
10. ~~**Laya decision head (issues #460, #462)**~~ — **done 2026-09-27 CPU / 2026-09-29 GPU** (`khurram/laya`, `khurram/462`): `LayaDecisionHead<T>`, `LayaPromptBuilder`, a `laya` mode, and a wheel gate (prompt ids byte-exact, logits max |diff| 1.3e-5); then `LayaHeadGpuRunner` reusing this trunk through an additive `ForwardOnDevice`/`HiddenOnDevice` seam, needing **zero** new kernels — the head's ops are pre-norm, *biased* LayerNorm, ReLU, fused *biased* QKV, all already present. The GPU half is gated GPU-vs-CPU against the CPU head, which `laya compare` has already pinned to the wheel. Reflection in [docs/LAYA.md](LAYA.md). Two notes for whoever picks up the next model: the head's norms are **biased** where every ModernBERT norm is not, so the encoder's shared zero beta would be a silent wrong answer; and the head's eps is `LayaDecisionHead`'s `1e-5` default, not `config.NormEps`. **On the gate, the generalisable lesson:** it runs at the same 1e-3 bound and passes at **1.1%** of it, while the encoder gate spends 56% — even though the head sits *on top of* that encoder. A gate's tightness is a property of the quantity being compared, not of the depth behind it. The encoder gate compares the raw `[L, 1024]` hidden state, where 28 layers of reduction-order drift land undiluted; this one compares a `1024 → 1` projection taken after a LayerNorm has re-normalized each row, so per-position errors average down by `sqrt(1024) = 32` and the bias component is removed outright. Do not carry a hidden-state budget over to a projected output. Every row prints its measured figures and the percentage of bound consumed, so this stays checkable rather than asserted. **Perf:** `laya --gpu benchmark` vs `laya benchmark`, whole model at `max_len` 512, 1 warmup + 3 timed median — CPU ≈8.48 s vs iGPU ≈3.00 s, **≈2.83×** (2.62–2.95× per question), in the same 2.6–3.0× band as the §1 BERT-family rows despite 421 M parameters and 4× the sequence length. **And at this shape the iGPU catches PyTorch:** the same fixture timed through the `laya==0.3.20` wheel (`Python/laya_benchmark.py`, all 16 cores) gives ≈2.97 s, a 1.01× tie with a sign flip per row, against §1's 1.8–2.4× deficit at 128 tokens. The deficit is shape-dependent, not fixed — plausibly arithmetic intensity, since 512 tokens quadruples the `M` in every GEMM. Cold start also favours the device: 5013 ms to build on the iGPU vs 11231 ms to bind on the Nivara CPU path and 30017 ms for PyTorch. Full table, thread-count confound, and over-reading caveats in [docs/LAYA.md](LAYA.md); the protocol differs from §1's 128-token / 10-timed and the PyTorch column is a separate run, so the row is deliberately not merged into that table.
11. ~~**Mask-as-select (issue #448)**~~ — **done 2026-09-30** (`khurram/448`). The local `max == -inf → zeros` clamp in `BatchedAttention` was the *prerequisite* #449 landed; #448 is the structural version, and it landed on the CPU side. `AttentionKernels<T>.ApplyMask` now suppresses by **assigning** `-inf` to a suppressed score cell instead of computing `score + (-inf)`, because summing cannot suppress a non-finite score (`NaN + (-inf) = NaN`, `(+inf) + (-inf) = NaN`) and a diverged score would otherwise escape the mask and poison every query row in the frame on the next layer. Only the `-inf` cell selects: a `NaN` or `+inf` **mask** cell stays additive (PyTorch parity) and a finite `finfo.min` fill keeps its magnitude (a fully-masked row still collapses to one constant), so the behavioural delta is confined to the two cells that were the bug and no existing fixture changed. The GPU side needed no change — `BatchedAttention` was already an assign, so it was immune to this class; issue #448's comment reporting the GPU path as unguarded predated #447. Residual follow-ups: #480 (measure the scalar `ApplyMask` cost against the vectorized add it replaced) and #481 (both attention ops list the mask as an `OpNode` input but never accumulate a gradient for it).
12. **Banded/sparse attention (issue #447)** — **done 2026-09-30** (`khurram/447`): the GPU kernel's band now restricts the loop bounds of all three passes instead of only overwriting an already-computed score with `-inf`, so a sliding layer's cost tracks its 129-wide window rather than `seqLen`. Carrying the band is not the same as using it — before this the band was plumbed per layer (`ModernBertGpuRunner.cs:408`) and had no effect on iteration count, so the GPU leg paid full price for a window. Measured end to end on an **Intel Iris Xe** iGPU, AC power, F32, 1 warmup + 3 timed median, ModernBERT-large (28 layers, 10 full + 18 sliding, band 64), against the same binary with the dense sweep restored:

    | seq | dense (pre-#447) | banded | speedup |
    |---|---|---|---|
    | 128 | 1497.8 ms | 1397.7 ms | 1.07× |
    | 512 | 9224.3 ms | 7264.8 ms | 1.27× |
    | 2048 | 83342.7 ms | 57117.7 ms | 1.46× |
    | 4096 | 279704.1 ms | 176001.2 ms | 1.59× |

    Two cautions on that table. The 4096 row is the weakest number in it: the banded run's min and median differ by 11.8% (157450.2 vs 176001.2 ms) where every other row's differ by under 1%, and the saving there is ~21× the saving at 2048 where `S²` scaling of the attention work predicts ~4×. It is reported because it was measured, not because it is understood; `modernbert --gpu benchmark` now takes `--seq` (**#474**, `khurram/474`), so that one row can be re-measured on its own with `--seq 4096 --iters 10 --warmup 3` instead of paying for the whole table. **That re-measurement has not been run**, so the table stands as-is and the 4096 row remains the one unexplained number — the flag removes the cost of investigating it, not the anomaly. And the change is provably output-invariant — `Keep` already suppressed every out-of-band key, and such a key contributed `exp(-inf - max) = 0` and then `0 * v = ±0` to an accumulator starting at `+0` — confirmed by reverting the kernel to the dense sweep and re-running `GpuAttentionBandTests` (all green) and `modernbert --gpu` (0 of 10 sentences differ in any hidden-state statistic). `modernbert --gpu compare` passes on this machine at max abs 3.052E-005, max rel 4.807E-004 (48% of the 1e-3 bound), cosine 1.0000000000, 0 non-finite on either side, with the CPU reference regenerated from `modernbert_compare.py` because `samples/data/` is gitignored and the fixture is not in the checkout. The CPU side still needs the equivalent to lift `ModernBertMasks.MaxDenseLength` = 2048, and that is a **feasibility** fix rather than a speed one: attention is 2.1%/4.1% of CPU MACs at L=128/256, so **#473**. `BertEncoderGpuRunner` keeps its `batch<=8, seqLen<=128` limit, and the Laya head keeps the same cap on the CPU, for the same reason.
13. **The ModernBERT `maxRel` gap is depth, not a GEMM defect — and #440 is now measured dead.** `maxRel 5.577E-004` against the CPU where DistilBERT sits at `3.238E-006` is 28 layers of F32 reduction-order drift accumulating, not a kernel fault; ModernBERT's weights are `1.4 B` and the encoder is 28 layers, so the comparison is not like-for-like with a 6-layer model. A second device, the Iris Xe, lands in the same place at `maxRel 4.807E-004` / max abs 3.052E-005 on the same gate (the issue text quotes 1.62e-5 max abs, so device and driver move the absolute figure by ~2x while the ratio and the conclusion do not change). This item originally read "a 2x2 or tile-32 register block is the only realistic path" and pointed at **#440**. That is now measured: the family loses to Row4 on every shape, because on this device shared-memory *capacity per group* dominates shared-memory *traffic per MAC* (§5.2). ModernBERT's remaining GPU lever was its attention leg, and **#447** has now taken it.

---

# Streaming Memory-Budget Enforcement

> **Status: design proposal — not implemented.** Nothing in this section ships. The
> `BudgetEnforcement` enum, the `MemoryBudget` class, and an `enforced:` parameter on
> `QueryFrame.AsStream` do not exist anywhere in `src/`; the whole signature is
> `AsStream(int chunkSize = 10000, CancellationToken ct = default)` (`QueryFrame.cs:461`).
> Every code block below is the *proposed* shape, not something a reader can call. For the
> streaming contract that does ship, see [`docs/STREAMING.md`](STREAMING.md).
>
> It also no longer matches the issue it names. #325 proposes **spill-to-disk** for boundary
> operators (`SpillDirectory` on the context, spill then reconstruct); what is described here
> is an in-memory byte-budget gate that pauses the producer. Both are open; neither is
> planned. Treat the divergence as an open question for whoever picks this up, not as a
> settled design.

Phase 1 of [issue #325](https://github.com/khurram-uworx/Nivara/issues/325) — read the status
note above before treating this as that issue's design.

## Problem

The streaming execution strategy tracks accumulated chunk memory via `StreamingBudgetTracker` but never enforces the limit. Source readers produce chunks regardless of downstream capacity, and boundary operations (Sort, GroupBy, Join) accumulate all chunks before processing. Peak memory is unbounded relative to the configured budget.

## Goal

> The reader cannot allocate another chunk unless the memory budget has capacity for it.

Backpressure at the source — not reactive spill after the fact.

## Two modes

The budget behavior is **configurable** and defaults to the current advisory mode for backward compatibility.

| Mode | Behavior | Use when |
|------|----------|----------|
| **Advisory** (default) | Budget is tracked; `StreamingBudgetTracker` emits a `PerformanceWarning` when accumulated memory exceeds the threshold. No reads are blocked. | Existing code, diagnostic profiling, scenarios where over-budget is acceptable. |
| **Enforced** | Budget is a hard cap on in-flight data. Source reads are gated — the producer cannot advance until the consumer has released enough budget. | Memory-constrained environments, containerized deployments, large-file pipelines where OOM is a real risk. |

### Configuration

**Today there is no knob for any of this.** `QueryFrame.AsStream` builds its own
`NivaraExecutionContext(ExecutionStrategy.Streaming)` internally (`QueryFrame.cs:461-478`)
and sets only `CancellationToken`, `ChunkSize`, and `ExecutionDiagnostics`;
`MemoryBudget` is left at its 1 GB constructor default (`NivaraExecutionContext.cs:17`) and
no overload accepts a context. The one sizing knob a caller can reach is `AsStream(chunkSize:)`.
`StreamingExecutionStrategy` is internal, so its
`StreamChunksAsync(QueryPlan, NivaraExecutionContext, CancellationToken)` cannot be called
from outside the assembly either; the only public route to a custom context is
`ExecutionEngine.Execute(plan, context)`, which materializes one whole frame and yields no
chunks at all. So the old sentence here — "mode is set via `NivaraExecutionContext` or at
strategy construction time" — described an object that never reaches `AsStream`, and a
strategy constructor that takes no arguments.

What actually bounds streaming memory today is the bounded producer/consumer channel, with
`StreamingBudgetTracker` warning on top — not a byte budget. See `docs/STREAMING.md`
§"Memory budget → chunk size" and §"AC3 resolution (memory budget enforcement)", and
`StreamingBackpressureTests`.

```csharp
// What is settable today: the chunk size, on AsStream.
await using var telemetry = Csv.ScanAsQueryFrame("data.csv")
    .Filter(ColumnExpressions.Col("status") == "OK");

await foreach (var chunk in telemetry.AsStream(chunkSize: 50_000))
{
    // The consumer owns each yielded chunk.
    using (chunk)
    {
        Process(chunk);
    }
}
```

**Proposed**, per this section's design and not compilable today — `BudgetEnforcement` does
not exist, and neither the property nor an `enforced:` shorthand is reachable from the
streaming surface for the reasons above:

```csharp
// Proposed only. Does not compile: no BudgetEnforcement, no AsStream(enforced:).
var context = new NivaraExecutionContext(ExecutionStrategy.Streaming)
{
    MemoryBudget = 256 * 1024 * 1024,   // 256 MB
    BudgetEnforcement = BudgetEnforcement.Enforced
};
```

Implementing it would need a public route for the context to reach the strategy — tracked
as #514.

Two properties of the proposal, neither of which holds today:

- `AsStream` takes `(int chunkSize = 10000, CancellationToken ct = default)` and no context.
  Enforcement would have to travel on the execution context, and nothing public currently
  carries a caller's context into the streaming strategy.
- Both `MemoryBudget` and the `BudgetEnforcement` flag would be additive to
  `NivaraExecutionContext`, which today has neither.

When `BudgetEnforcement` is `Advisory` (the proposed default), the pipeline behaves
identically to today: `StreamingBudgetTracker` records and warns, but reads are never
blocked.

When `BudgetEnforcement` is `Enforced`, the proposed `MemoryBudget` primitive gates every
chunk allocation at the source boundary.

## Architecture

```
                  MemoryBudget (enforced mode)
                       │
                       ▼
┌────────┐      ┌─────────────┐      ┌──────────┐
│ Source │─────►│ Chunk Buffer│─────►│ Consumer │
│ Reader │      │  (Channel)  │      │          │
└────────┘      └─────────────┘      └──────────┘
     ▲                 │
     │                 │
     └─── pause ───────┘
        (budget full)
```

### Invariant

**In enforced mode**, at any point in time:

```
Σ (estimated bytes of all in-flight chunks) ≤ MemoryBudget
```

The source cannot produce a new chunk until the consumer has released enough budget. No chunk is ever created "then checked" — the check happens **before** the read.

### Why bytes, not item count

A bounded `Channel<T>` with item-count capacity is insufficient:

```
Channel capacity = 10

Chunk 1 = 2 MB
Chunk 2 = 3 GB    ← exceeds budget, but channel accepted it
Chunk 3 = 1 MB
```

Chunk sizes vary dramatically (different row widths, column types, compression). The budget must be **byte-aware** — it tracks estimated memory, not item count.

## MemoryBudget primitive

A byte-level async resource governor. Not a tracker — a **control point**.

```csharp
public sealed class MemoryBudget : IDisposable
{
    public MemoryBudget(long capacityBytes);

    /// <summary>
    /// Waits until the budget has room for <paramref name="requestedBytes"/>,
    /// then reserves that amount. Blocks (async) when the budget is exhausted.
    /// </summary>
    public ValueTask AcquireAsync(long requestedBytes, CancellationToken ct = default);

    /// <summary>
    /// Returns previously acquired bytes to the budget, unblocking any
    /// waiting acquirer.
    /// </summary>
    public void Release(long releasedBytes);

    /// <summary>
    /// Current estimated in-flight bytes (for diagnostics).
    /// </summary>
    public long CurrentUsage { get; }

    /// <summary>
    /// Maximum capacity.
    /// </summary>
    public long Capacity { get; }

    public void Dispose();
}
```

Internally backed by `SemaphoreSlim` for async wait, with `Interlocked` for thread-safe usage tracking. Zero-allocation on the happy path (no `Task` allocation when the budget has room).

### Acquire/Release contract

The producer and consumer follow a strict acquire/release protocol:

```csharp
// Producer side (inside StreamingExecutionStrategy or source adapter)
await memoryBudget.AcquireAsync(chunk.EstimatedMemoryBytes, cancellationToken);
try
{
    await channel.Writer.WriteAsync(chunk, cancellationToken);
}
catch
{
    memoryBudget.Release(chunk.EstimatedMemoryBytes);
    throw;
}

// Consumer side
var chunk = await channel.Reader.ReadAsync(cancellationToken);
try
{
    await ProcessAsync(chunk);
}
finally
{
    memoryBudget.Release(chunk.EstimatedMemoryBytes);
}
```

The producer **cannot advance past budget capacity**. The consumer **must release** when done. Ownership is unambiguous.

### Chunk memory estimation

Chunk memory is estimated via `NivaraFrame.estimateFrameMemoryUsage()`, the same method already used by `StreamingBudgetTracker`. This returns the estimated byte count of the frame's column data (arrays, null masks, string references).

## Integration with StreamingExecutionStrategy

### Enforced mode: executeCoreInternalAsync

```
1. Create MemoryBudget(context.MemoryBudget)
2. Create bounded Channel<NivaraFrame>
3. Producer Task.Run:
   for each chunk from source.ToAsyncEnumerable():
       estimate = chunk.EstimatedMemoryBytes
       budget.AcquireAsync(estimate)          // ← blocks here if budget full
       channel.Writer.WriteAsync(chunk)
4. Consumer (main thread):
   foreach chunk from channel.Reader.ReadAllAsync():
       budget.Release(previousChunk.MemorySize)  // release previous
       chunkFrames.Add(chunk)
5. After channel drains: budget is fully released
```

The key difference from today: step 3 **blocks the producer** when the consumer hasn't kept up. Today, the producer reads unconditionally.

### Enforced mode: StreamChunksAsync (boundary operators)

For plans with Sort/GroupBy/Join, chunks must be accumulated before the boundary operator runs. In enforced mode:

```
1. Create MemoryBudget(context.MemoryBudget)
2. For each chunk from source:
       budget.AcquireAsync(chunkMemory)
       chunkFrames.Add(chunk)
3. ConcatenateVertical(chunkFrames)          // boundary operator needs all data
4. Dispose individual chunks
5. budget.Release(all)                       // free budget after concat
6. Run boundary operator on concatenated result
```

This is inherently **not** constant-memory — the boundary operator requires the full dataset. But the budget still provides value: it **pauses the source** during step 2, so peak memory is bounded to `MemoryBudget + concatenation overhead` rather than being completely uncontrolled.

The guarantee is:

> Peak memory ≤ MemoryBudget × (1 + accounting tolerance)

where accounting tolerance covers GC overhead, temporary buffers, and the concatenation itself (documented, typically ≤ 1.5×).

### Enforced mode: StreamChunksAsync (pure streamable)

For fully streamable plans (no boundary ops), the `IAsyncEnumerable` pull semantics already provide natural backpressure. Enforced mode adds the `MemoryBudget` gate as an additional safeguard:

```
foreach chunk from source.ToAsyncEnumerable():
    budget.AcquireAsync(chunkMemory)
    process streamable ops
    yield return chunkFrame
    // consumer disposes → implicit release via ownership transfer
```

The yield/await enumeration means the consumer controls the pace. The budget is a secondary guard against unexpected chunk sizes.

### Advisory mode (current behavior, unchanged)

```
1. Create StreamingBudgetTracker(context.MemoryBudget)
2. Producer reads chunks unconditionally
3. Consumer records each chunk in the tracker
4. After all chunks: tracker.RecordWarningIfExceeded(diag)
```

No `MemoryBudget` primitive is created. No blocking. Identical to today.

## What stays the same

| Component | Unchanged? | Notes |
|-----------|-----------|-------|
| `StreamingBudgetTracker` | ✓ | Remains the diagnostic tracker. Emits `PerformanceWarning` in both modes. |
| Source readers (`CsvLazySource`, `ParquetLazySource`, `JsonLazySource`) | ✓ | They read `chunkSize` rows as requested. Budget gating happens at the *caller*, not inside the reader. |
| `NivaraExecutionContext.MemoryBudget` | ✓ | Still a `long` in bytes. Default 1 GB. |
| `StreamingExecutionStrategy.CalculateChannelCapacity` | ✓ | Channel capacity formula is unchanged. |
| `QueryFrame.AsStream()` | ✓ | Public API unchanged by this proposal — it would *gain* an `enforced:` parameter, since it has none today. Reaching the strategy from that parameter is the open part: `StreamingExecutionStrategy` is internal and `AsStream` takes no context. |
| Streamix bridge (`NivaraFlux`) | ✓ | Uses `AsStream` under the hood, so Enforced mode would flow through — conditional on the same missing route as the row above. |

## Migration path

1. **Default is Advisory** — existing code is unaffected. No behavior change unless `BudgetEnforcement.Enforced` is explicitly set.
2. **Opt-in enforcement** — users who want hard guarantees set the flag on their execution context.
3. **Future: Enforced may become the default** — once the enforcement is battle-tested, the default can flip to `Enforced` with a deprecation period for `Advisory`.

## What Phase 1 does NOT cover

| Concern | Phase | Why deferred |
|---------|-------|-------------|
| Spill-to-disk for boundary operators | Phase 2 | Only makes sense after sources obey backpressure. Operators that require materialization (Sort, Join) need their own external-memory algorithms — that's a separate architectural layer. |
| External sort / hash join | Phase 2 | Requires operator-level spill abstraction. |
| Hard process-level memory limits | Future | .NET GC, runtime allocations, native decoders all contribute memory outside Nivara's control. Hard process limits need OS/container enforcement. |
| Budget-adaptive chunk sizing | Future | Today chunk size is derived once at strategy start. Could adapt at runtime based on observed chunk sizes. |

## Acceptance criteria

- [ ] `MemoryBudget` class with `AcquireAsync`/`Release`/`CurrentUsage`/`Capacity`
- [ ] `BudgetEnforcement` enum (`Advisory`, `Enforced`) on `NivaraExecutionContext`
- [ ] `StreamingExecutionStrategy` uses `MemoryBudget` when `Enforced`, `StreamingBudgetTracker` when `Advisory`
- [ ] Producer blocks when budget is exhausted (enforced mode)
- [ ] Consumer releases budget after processing each chunk
- [ ] `QueryFrame.AsStream(enforced: true)` flows through to strategy
- [ ] Existing `StreamingBackpressureTests` pass unchanged (advisory mode)
- [ ] New tests: `MemoryBudget` unit tests (acquire/release/blocking)
- [ ] New tests: enforced-mode integration tests proving budget is respected
- [ ] Benchmark: `bench-stream` shows reduced peak memory in enforced mode for boundary-op plans

## Related

- **Issue:** [#325 — Spill-to-disk for streaming execution memory budget enforcement](https://github.com/khurram-uworx/Nivara/issues/325)
- **Existing streaming docs:** [`docs/STREAMING.md`](STREAMING.md)
- **Budget tracker:** `src/Nivara/Execution/StreamingBudgetTracker.cs`
- **Strategy:** `src/Nivara/Execution/StreamingExecutionStrategy.cs`
- **Context:** `src/Nivara/Execution/NivaraExecutionContext.cs`
- **Backpressure tests:** `tests/Nivara.Tests/Execution/StreamingBackpressureTests.cs`

---

# Combined next steps

## GPU

1. ~~**#440 — tile-32 / 2×2 GEMM.**~~ **Measured, null 2026-09-29** (`khurram/440`): all four geometries lost on every shape; best cell 1.01×, typical 0.6–0.85×. Shared-memory capacity per group dominates shared-memory traffic per MAC on this iGPU. Kept as a labelled negative baseline in `GemmKernels.cs`. See §5.2.
2. **#435 — lasting GEMM regression harness.** Promote the double-truth bounds check into a permanent gate. Now partly delivered: the gate asserts **byte-identity**, not just a tolerance, which is what caught #468.
3. ~~**#448 — mask-as-select.**~~ **Done** 2026-09-30 (`khurram/448`) — see item 11 above. The fully-masked-row `NaN` hazard is closed on both halves: the CPU masks by assign, and `BatchedAttention` was already an assign.
4. ~~**#447 — banded attention.**~~ **Done** 2026-09-30 (`khurram/447`) — see item 12 above for the measured end-to-end result (1.07× / 1.27× / 1.46× / 1.59× at seq 128 / 512 / 2048 / 4096 on an Iris Xe). The original case for it, kept here because the leg profile is still the tool that produced it: a 2026-09-29 leg profile (`--gemm-legs`, AC) measured `BatchedAttention` at **57.0%** of Laya wall-clock (1266.87 of 2221.12 ms) against 39.2% for all GEMM together, and **61.1%** on ModernBERT (939.22 of 1536.53 ms). The second data point, an **Intel Iris Xe** (edge class, ~4.4× slower in absolute terms than the Arc figures above): `modernbert base` 61.9% and `laya large` 57.8%, GEMM 36.5%/40.3%. **Treat the agreement between those two columns as weaker than it looks.** Re-running `--gemm-legs` on this machine against two builds that do *identical* attention work — `band = -1`, where #447 cannot change a single iteration — moved the attention share by **3.7 pp and 4.5 pp**, with the sign flipping between the two models. A 0.8 pp cross-device agreement is therefore inside this machine's own build-to-build noise, and is not by itself evidence that the share is a device-independent structural property. The end-to-end A/B in item 12, which compares two builds in the same session on the same machine, is the measurement that carries the claim.
5. ~~**#462 — wire the Laya head onto `ModernBertGpuRunner`.**~~ **Done** 2026-09-29: zero new kernels, gated GPU-vs-CPU against the CPU head at 1e-3. See item 10 above and [docs/LAYA.md](LAYA.md).
6. **Promotion decision**: decide whether GPU support moves into `src/Nivara.Gpu`.

## Streaming memory budget

1. **Phase 2 — spill-to-disk for boundary operators.** Only makes sense after sources obey backpressure.
2. **External sort / hash join.** Requires operator-level spill abstraction.
3. **Hard process-level memory limits.** Needs OS/container enforcement.
4. **Budget-adaptive chunk sizing.** Adapt at runtime based on observed chunk sizes.

## CPU-side (independent thread)

- A managed tiled GEMM and/or an opt-in native BLAS bridge would close the ~5.6× Nivara-CPU deficit (possibly beating the GPU at these shapes); options, targets, and the M1/M2/M3 decision gate are captured in [ROADMAP-SUGGESTION.md](ROADMAP-SUGGESTION.md).

## References

- [ILGPU.md](ILGPU.md) — probe verdicts: `Index1D`, padded-grid bounds checks, `CL_DEVICE_TYPE_GPU`, `AsContiguous().GetAsArray()`, `MathMode.Fast` off
- [SMOLLM.md](SMOLLM.md) — SmolLM GPU investigation
- [STREAMING.md](STREAMING.md) — streaming execution strategy
- [ROADMAP-SUGGESTION.md](ROADMAP-SUGGESTION.md) — CPU/GPU-next roadmap, options, decision gate
- [`samples/NivaraInference/README.md`](../samples/NivaraInference/README.md) — usage, `--gpu` quick start, benchmark tables, CPU baselines
- [`tests/Nivara.GpuProbe/README.md`](../tests/Nivara.GpuProbe/README.md) — eight-way backend comparison, `kernels` correctness gate. **The probe legs are not a backend-selection argument:** each runs a *naive* reference kernel (one work-item per row, serial reduction, no tiling), so their timings are not comparable to the production `Row4` GEMM above. `tests/Nivara.GpuProbe/SILK.md` records the Silk.NET OpenCL leg specifically, including the measured answer to whether a raw-OpenCL path could unlock BF16 (§5 item 4) — it cannot.
- Issues: **#325** (streaming memory budget) · **#435** (tiled-GEMM regression gate, open) · **PR #436** (DistilBERT GPU) · **#437** (kernel fusion, done) · **#440** (tile-32/2x2 GEMM, **measured null** — closed) · **#447** (banded/sparse attention, **done** 2026-09-30 — was the leading GPU item) · **#448** (mask-as-select, **done** 2026-09-30) · **#449** (ModernBERT GPU path) · **#460** + **#462** (Laya decision head, CPU + GPU) · **#467** (LayerNorm one-work-item-per-row, minor) · **#468** (ILGPU OpenCL `Allocate2D` lowering, upstream)
