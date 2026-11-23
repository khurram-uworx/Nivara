# TODO: #1471 — Stage normed-x in local memory for decode GEMV

## Problem

Issue: https://github.com/khurram-uworx/Nivara/issues/1471 (follow-up to #1466 / PR #1474, closed).

After #1466 landed vectorized weight loads for decode GEMV kernels, normed-`x` staging in local memory was deferred. In `rms_norm_gemv` and `rms_norm_gemv_bf16`, every lane in the 32-thread group recomputes `av = x[k] * normScale * gamma[k]` in the hot dot loop, reading `x[k]` and `gamma[k]` from global memory redundantly across lanes. Staging the normalized vector in `__local` memory lets all lanes read from local memory after a single cooperative fill.

Scope per issue body: `rms_norm_gemv` / `rms_norm_gemv_bf16`. `gemm_gemv_add` / `gemm_gemv_add_bf16` take already-normalized `a[k]` and have no RMSNorm; they are out of scope.

## Proposed changes

### 1) Kernel staging with compile-time max + fallback (`SilkKernels.cs`)

Add a compile-time staged-normed capacity and stage `normed[k]` in `__local` memory when `aCols <= STAGED_NORM_MAX`. Fallback to the current non-staged path otherwise.

Sketch (applies to both `rms_norm_gemv` and `rms_norm_gemv_bf16`; bf16 variant differs only in weight type/widen):

```c
#define STAGED_NORM_MAX 8192u  // covers current decode hidden sizes; 32KB local

__kernel void rms_norm_gemv(...)
{
    ...
    uint col = colBase + get_local_id(0) * BLOCK;
    uint lid = get_local_id(0);

    float ssq = 0.0f;
    for (uint k = 0; k < aCols; ++k) { float v = x[k]; ssq += v * v; }
    float normScale = 1.0f / sqrt(ssq / (float)aCols + eps);

    if (aCols <= STAGED_NORM_MAX)
    {
        __local float normed[STAGED_NORM_MAX];
        for (uint k = lid; k < aCols; k += GEMVGROUP)
            normed[k] = x[k] * normScale * gamma[k];
        barrier(CLK_LOCAL_MEM_FENCE);

        if (col >= bCols) return; // after barrier - safe
        uint avail = bCols - col;
        float a0=0,a1=0,a2=0,a3=0;
        for (uint k = 0; k < aCols; ++k)
        {
            float av = normed[k];
            uint bBase = k * bCols + col;
            a0 += av * b[bBase];
            if (avail > 1u) a1 += av * b[bBase + 1u];
            if (avail > 2u) a2 += av * b[bBase + 2u];
            if (avail > 3u) a3 += av * b[bBase + 3u];
        }
        // store (same as today)
    }
    else
    {
        if (col >= bCols) return; // no barrier in fallback
        // current non-staged dot loop
    }
}
```

Key invariants:
- `normScale` computed identically (same formula, same loop order, every thread).
- `normed[k] = x[k] * normScale * gamma[k]` written once per k, same formula as today - bit-identical value.
- Out-of-range threads (`col >= bCols`) participate in staging + barrier, then return before dot/store. No deadlock.
- Fallback keeps current no-barrier early-return behavior.
- Update kernel comment: barrier is present in staged path; fallback has no barrier.

Design choice: compile-time `__local float normed[STAGED_NORM_MAX]` (no SilkRuntime API change). `STAGED_NORM_MAX=8192` (32KB) covers current model hidden sizes (SmolLM 576/960, Qwen 896/1536, up to 8192); fallback guards larger shapes. Dynamic local arg would require `SilkRuntime.SetLocalArg` and kernel signature changes - deferred unless needed.

### 2) Host-side: no signature change expected

`SilkLauncher.RmsNormGemv` / `RmsNormGemvBf16` argument lists unchanged. No `SilkRuntime` local-arg API needed for compile-time static local. If dynamic sizing is later chosen, add `SetLocalArg` - not in this plan.

### 3) Tests

- Existing parity tests in `tests/Nivara.Tests/Gpu/SilkRmsNormGemvTests.cs` must remain green (bit-identical). They cover biased/bias-free, 1/2/3 slots, straddling widths, SmolLM cached head.
- Add a case exercising `aCols > STAGED_NORM_MAX` fallback? Only if we can construct it without huge allocations; alternatively add a smaller compile-time guard in test via a dedicated kernel constant is not possible. Better: add a test that exercises a large aCols within STAGED_NORM_MAX and confirm bit-identical; fallback is the existing code path (covered implicitly by being unchanged). Consider adding an explicit fallback-path test by temporarily lowering the conceptual max is not feasible at runtime. We'll document fallback as code-path-preserved and rely on existing tests for the staged path.
- Gpu tests are `[Category("Gpu")]` and ignored when no OpenCL device.

### 4) Docs / pins

- Check `DocsCoveragePins` if any `docs/*.md` or citation changed. This change is kernel-internal; no doc changes expected unless we update SILK.md. Check SILK.md for GEMV description - update only if it makes a specific claim that becomes stale.

## Verification

- `dotnet build Nivara.slnx -c Release` (ask before running tests)
- Targeted GPU tests if OpenCL available: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkRmsNormGemvTests|FullyQualifiedName~SilkMlpBlockDecompositionTests"`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Docs snippet gate / DocsCoverageTests if docs touched: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`
- Benchmark before/after: human-run per AGENTS (SmolLM/Qwen synthetic Silk profile); commands write to `data\qwen-bench\`. Record evidence in PR.

## Blast radius

- **Code**: `src/Nivara.Extensions/Silk/SilkKernels.cs` — two OpenCL kernel bodies (`rms_norm_gemv`, `rms_norm_gemv_bf16`) and possibly a new `#define STAGED_NORM_MAX`. No C# API signature changes.
- **Host**: `SilkLauncher.RmsNormGemv` / `RmsNormGemvBf16` unchanged. `SilkRuntime` unchanged.
- **Callers**: all decode paths that launch these kernels through `SilkLauncher` (Qwen/SmolLM/Laya decode, fused projections). Behavior must remain bit-identical.
- **Tests**: `tests/Nivara.Tests/Gpu/SilkRmsNormGemvTests.cs` (bit-identical parity, Gpu category). `SilkMlpBlockDecompositionTests.cs` also exercises MLP block (different kernel, but nearby) - not directly affected.
- **Risk**: barrier deadlock if early-return discipline wrong; wrong STAGED_NORM_MAX causing local-memory overuse on small devices; non-bit-identical normed value (would fail parity tests). Mitigated by bit-identical tests and fallback path.

## Planned commits

1. `docs: plan #1471 normed-x staging in TODO.md`
2. `perf(silk): stage normed-x in local memory for decode GEMV (#1471)`
3. `test/docs: cover #1471 staging path and update docs if needed` (only if needed)
4. `docs: remove TODO.md — plan executed`

## GitHub issues log

- [ ] #1471 — Silk: stage normed-x in local memory for decode GEMV (this work)

## Reminder

As each task executes, if you find deferred work or a concern, create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its number in the log — don't rely on memory or wait until the end of the plan, as compaction during execution can lose important items.
