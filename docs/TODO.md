# Plan: issue #629 — `--silk` for Qwen2.5-0.5B (causal LM, KV cache, tool loop)

Branch: khurram/629
Issue: https://github.com/khurram-uworx/Nivara/issues/629

## Problem

`--silk` does not run Qwen2.5-0.5B-Instruct. The Qwen sample is a Llama-family
causal LM with 24 layers, `hidden=896`, `intermediate=4864`, 14 query / 2 KV
heads (GQA), `headDim=64`, SiLU gated FFN, RMSNorm (`eps=1e-6`), RoPE
`theta=1_000_000`, `max_position_embeddings=32768`, vocab `151936`, tied
embeddings, and **biased Q/K/V projections**. The existing Silk decoder runner
targets SmolLM only (canonical Llama: no QKV bias). Qwen also carries a KV-cached
decode loop and the `--qwen tools` functional-calling loop.

The Silk kernel set is already sufficient (no new kernels should be added for
this issue): `RmsNorm`, `Silu`, `SwiGlu`, `CausalGqaAttention`, `DecodeGqaAttention`,
`Rotary`, `AddBias`, `Gemm`. The work is a Qwen-specific runner + session plus
sample wiring through an adapter so the tool loop is not reimplemented.

## Proposed changes

### Slice 1 — kernels (no kernel changes expected)

The existing Silk decoder kernels cover Qwen. Verify launcher support for bias:
`AddBias(SilkRuntime, x, bias, y, rows, cols)` is already present. No kernel
addition, no `KernelNames` addition.

### Slice 2 — `SilkQwenGpuRunner` + session (F32, KV-cached)

New files mirroring `SilkSmolLMGpuRunner` / `SilkSmolLMSession`, but with
Qwen-specific configuration and QKV bias:

- `src/Nivara.Extensions/Silk/SilkQwenGpuRunner.cs`
- `src/Nivara.Extensions/Silk/SilkQwenSession.cs`
- `SilkQwenConfig` record

Differences from the SmolLM runner:
- Qwen dimensions/head count/eps/rope theta/vocab/max positions.
- Layer storage adds `QBias`, `KBias`, `VBias` buffers.
- After each Q/K/V projection GEMM, call `AddBias`.
- RoPE uses `RotaryEmbedding<float>(headDim, maxPos, ropeTheta)`.
- Full `maxPositionEmbeddings = 32768` KV cache.
- Exposes `Forward`, `ForwardLastLogits`, `ForwardPrefill`, `ForwardCached`,
  `ResetCache`, `CachedLength`.

### Slice 3 — sample adapter (`IQwenDecoder`)

Add a sample-only abstraction so CPU and Silk share the tool loop:

- `IQwenDecoder`
- `CpuQwenDecoder`
- `SilkQwenDecoder`

Refactor only the model interaction points:
- `Generate(...)`
- `RunPlain(...)`
- benchmark paths

`RunTools(...)` continues to call `Generate(...)`, so tool use works without
reimplementing chat logic.

### Slice 4 — CLI wiring

- add `qwen` to `SilkModes`
- support default / `benchmark` / `plain` / `tools`
- keep `distill` CPU-only
- reject bf16/fp16 as `--silk is F32-only`

### Slice 5 — tests and CLI arg pins

- update `InferenceCliArgumentTests` expectations
- synthetic parity smoke
- real-weight tool loop gated on checkpoint availability
- KV-cache equivalence (`useKvCache` on/off)
- stop on either `151645` or `151643`

### Slice 6 — large-context / attention check

- benchmark cached decode at `kvLen` 512, 1024, 4096
- benchmark tied LM head
- document #399 behavior if attention becomes the bottleneck

### Slice 7 — docs

- update `docs/SILK.md`
- update widened-scan/doc catalog counts as needed

## Verification steps

- `dotnet build Nivara.slnx`
- targeted `SilkKernelsCatalogTests` / widened scan tests
- targeted inference CLI tests
- synthetic and real-weight parity smoke on an OpenCL GPU
- KV-cache ID equivalence on a short prompt

## Planned commits

1. docs: plan #629 in TODO.md
2. feat(silk): add SilkQwenGpuRunner + session (F32, biased Q/K/V, KV cache)
3. feat(silk): apply QKV bias via AddBias; pin Qwen config values
4. test(perf): add Silk vs CPU parity smoke for synthetic Qwen weights
5. refactor(sample): introduce IQwenDecoder adapters (CPU/Silk)
6. feat(sample): wire --silk for Qwen (tools, plain, benchmark)
7. test: update CLI arg pins for qwen in SilkModes
8. test: model-gated Qwen weather tool loop on Silk
9. docs: update SILK.md and doc scan counts

## Blast radius

- `src/Nivara.Extensions/Silk/SilkKernels.cs`, `SilkLauncher.cs` — no new kernels expected, but Qwen bias path exercises `AddBias`.
- New runner/session: consumed by sample adapter.
- `samples/NivaraInference/Program.cs` — dispatch, mode allow-list, SilkModes; changes affect CLI arg tests.
- `samples/NivaraInference/Qwen.cs` — refactor model interaction points only; tool/chat logic unchanged.
- `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs` — pins exact reject messages.
- No `src/Nivara` changes.

## GitHub issues log

- [ ] #384 — Qwen QKV bias loader gap (relevant cross-check; ensure runner consumes bias tensors)
- [ ] #399 — DecodeAttention per-dot loop at large kvLen (benchmark before completion)
- [ ] #445 — ILGPU decode for SmolLM (backend parity reference)

## Non-goals

- no ILGPU Qwen path
- no GGUF / INT8
- no `src/Nivara` change
- no second decoder kernel family
- no reimplementation of chat template, tokenizer, or tool loop
- no BF16 translation unit
- no premature promotion of CPU models from samples to Extensions
