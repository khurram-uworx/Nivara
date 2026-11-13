# Plan: fix NivaraBlas registration — explicit path, per-call threads, global static

**Issue:** #1171 (review M8). **Branch:** `khurram/1171`.

## Problem

`NivaraBlas.UseBlas(string? path, ...)` accepts `path`, documents "bound to path", and never uses it —
`OpenBlasInterop.ExplicitPath` is declared and never assigned, so its resolver branch is dead.
The provider is registered in process-global statics with no atomicity (`GemmProviders.Current` and
`GemvProviders.Current` are assigned separately, so a reader can see a half-registration), and
`openblas_set_num_threads` is called inside every `MatMul`/`Gemv`, mutating process-wide OpenBLAS
state on the hot path and clobbering settings any other library in the process may have chosen.
`BlasOptions.DllPath` forwards into the unused parameter; `UseManaged()` leaves the pool size set.

## Proposed changes

1. **Atomic registration slot (core).** Replace the two independent statics in `GemmProvider.cs` /
   `GemvProvider.cs` with one `internal sealed class ProviderRegistration(IGemmProvider? gemm, IGemvProvider? gemv)`
   held in a single `internal static volatile ProviderRegistration? Current` (`ProviderRegistry`).
   `GemmProviders.Current` / `GemvProviders.Current` become properties reading through it. No call-site
   changes; `MultiplyCore` (TensorsHelper.cs:126,164) and `BlasSmokeTests` keep working.
2. **Apply threads once (Extensions).** Move `SetNumThreads` out of `MatMul`/`Gemv` into the provider
   constructors; document that it mutates a **process-wide** OpenBLAS setting. Capture the previous
   pool size at registration and restore it in `UseManaged()` and in the new `EnterBlas` scope.
3. **Honour `path`.** `NivaraBlas` assigns `OpenBlasInterop.ExplicitPath = path` before any P/Invoke.
   `TryInitialize` validates it first (missing file -> `false` with the path in the message, before any
   native call, so it is order-independent), reports `LoadedPath` (the file the resolver actually won),
   and rejects a *different* explicit path after a load with both paths named (the resolver is only
   consulted on the first load in a process — the ALC caches the handle).
4. **Diagnostics.** `LastDiagnostic` property: the `TryInitialize` detail on failure (path included),
   or the resolved config on success. Today the constructor's `InvalidOperationException` is swallowed,
   so `samples/NivaraInference/Program.cs:430` prints "not found" for an ILP64 build.
5. **Disposable scope.** `EnterBlas(...)` returns `IDisposable` that restores the previous registration
   and thread count; nesting composes. `UseBlas` is the non-scoped form.
6. **Honest docs.** Fix `OpenBlasInterop` remarks ("single canonical location … no search path" vs
   three candidates) and `NivaraBlas` XML ("PATH / OPENBLAS_DLL" — neither exists). `BlasOptions`
   docs: `DllPath` does something; `OpenBlasThreads` is process-wide, applied once, restored on
   unregister.
7. **Docs + changelog.** `docs/BLAS-MKL.md` §7/§8 notes; M8 "(Resolved in #1171.)"; CHANGELOG entry.

## Blast radius

- Core: `ProviderRegistration`/`ProviderRegistry` new; `GemmProvider.cs`, `GemvProvider.cs` accessors
  become properties. Downstream: `TensorsHelper.MultiplyCore` (readers only, no change); all AutoDiff
  model code (`GradKernels`, `ForwardGradOperations`, `Linear`, `LlamaFusedKernels`, `AttentionKernels`,
  `ModernBertModel`) reaches the provider only through `MultiplyCore`, so no edits there.
- Extensions: `NivaraBlas.cs`, `OpenBlasInterop.cs`, `OpenBlasGemmProvider.cs`,
  `OpenBlasGemvProvider.cs`, `BlasOptions`.
- Tests: `tests/Nivara.Tests/BlasSmokeTests.cs` (new tests, drop manual `SetNumThreads(1)` cleanups).
- Sample: `samples/NivaraInference/Program.cs` prints `LastDiagnostic`.
- Docs: `docs/BLAS-MKL.md`, `docs/REVIEW-2026-11-12.md`, `CHANGELOG.md`.
- Context plumbing is **not** in scope: `MultiplyCore` has no `NivaraExecutionContext` and its callers
  (AutoDiff/Nn) have none either — that is the #133/#444 seam decision, not this issue.

## Verification

- `dotnet build Nivara.slnx -c Release`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.BlasSmoke"`
- `dotnet test ... --filter "FullyQualifiedName~Nivara.Tests.Docs"` (docs changed)
- Behavioural proofs: MatMul/Gemv do not set threads; `GetNumThreads() == 4` right after registration;
  nonexistent path -> false + path named; explicit path -> LoadedPath equals it; scope dispose restores.

## Planned commits

1. `refactor(core): single atomic provider registration slot (#1171)`
2. `fix(blas): apply the OpenBLAS thread count at registration, not per call (#1171)`
3. `fix(blas): honour UseBlas(path) and surface the registration diagnostic (#1171)`
4. `feat(blas): disposable registration scope with restore semantics (#1171)`
5. `docs(blas): record the seam's process-wide limits (#1171)`

## GitHub issues log

(none yet)