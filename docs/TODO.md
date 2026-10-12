# Plan: extend `--silk` from MiniLM to the DistilBERT family

Branch: `khurram/silk-distilbert`

> **Process reminder.** As each task executes, if deferred work or a concern surfaces (a limitation,
> follow-up, refactor, or red flag), create a GitHub issue immediately with
> `gh issue create --repo khurram-uworx/Nivara` and record its number in the **GitHub issues log**
> at the bottom of this file. Do not rely on memory or wait until the plan finishes — compaction
> during execution can lose it.

## Problem

`--silk` (the hand-authored OpenCL C backend on `Silk.NET.OpenCL`, `samples/Nivara.Samples/Gpu/`)
is MiniLM-only. The encoder runner `SilkBertEncoderRunner` hardcodes `BertGpuNaming.Bert` in two
places, and it rejects any checkpoint that ships classification-head weights
(`pre_classifier`/`classifier`), because the six-kernel set carries no gather and no ReLU GEMM
epilogue.

The ILGPU backend (`BertEncoderGpuRunner`) is already naming-driven and runs both DistilBERT and
DistilBERT SST-2 through `BertGpuNaming.DistilBert`. The Silk runner is the only piece that does not
generalize. DistilBERT is a genuine extension of the frontier:

| Property | MiniLM | DistilBERT |
|---|---|---|
| Weight keys | BERT-style (`embeddings.*`) | `distilbert.*` |
| Hidden / intermediate | 384 / 1536 | 768 / 3072 |
| headDim | 32 | **64 == `SilkKernels.MaxHeadDim`** (the boundary) |
| Token-type embedding | present | absent |
| Head | none (CLS pooling on host) | `pre_classifier → ReLU → classifier` |

## Decisions (confirmed with the human before implementation)

1. **Scope: both stages.** Ship `distilbert --silk` (encoder) and `distilbert_sst --silk` (head).
2. **Head implementation: device kernels.** Add `gather_rows` (CLS extraction, exact copy) and
   `gemm_row4_relu` (third epilogue over the shared `GEMM_ROW4_ACCUMULATE` macro). This keeps the
   runner device-pure and the byte-identity gate meaningful. The host-round-trip alternative was
   rejected (adds a sync, breaks the "same code, different `Forward`" seam).

## Resolved during execution (2026-10-12)

- **Stage 4 placement — restructure now, on this branch, as its own commit.** Not deferred to a
  separate branch/PR. Stage 3 therefore follows Stage 4 (it writes into the final homes).
- **`--silk-kernels` timing red flags — leave as a note for now.** The six out-of-band cells listed
  under Status stay recorded in the plan/report as informal-timing observations; the gate's timing
  column is explicitly *not* a verdict, and no GitHub issue is opened for them.

## Resolved during execution (2026-10-13)

- **The streaming design proposal got its own document: `docs/STREAMING-PROPOSAL.md`.** The 2026-10-12
  directive placed it in `docs/STREAMING.md`, but `STREAMING.md` is **gated** — the snippet gate
  compiles every fenced `csharp` block a gated document contains (the extractor has no per-block
  marker filter), and the proposal's four `csharp` blocks deliberately do not compile. `ACCELERATION.md`
  was the ungated home; it now has to be gate-free to serve as story + index. So the proposal moves
  **ungated → ungated** into its own clearly-named document, and `STREAMING.md` keeps the shipped
  contract and links to it. The gate bookkeeping moves with it: `docs/STREAMING-PROPOSAL.md` replaces
  `docs/ACCELERATION.md` in `UngatedDocuments`, the `ProposalDocuments_AreExcludedByNameNotByGlob`
  assertions retarget the new name, and `docs/SILK.md` is added to the ungated list (Part 2 introduced
  one `csharp` block). The streaming move itself is net-neutral; the pinned repository-wide totals
  moved by the SILK block plus the new docs' citations, and were updated deliberately (snippet
  249→250 / 141→142; citation 106→113 / 49→50 / 16→18).

## Documentation directives (added 2026-10-12, mid-Stage-4)

New human instructions that refine Stage 4 and add work to it:

1. **Split rule (confirmed 2026-10-12) — hold this line:**
   - **`ACCELERATION.md` = the acceleration *story* + index + roadmap.** The *why*, the *strategy*,
     the background/context from before the work ("the things before what"), the cross-backend
     results, the narrative of what shipped, and **what's coming next**. It is the one place to
     review/discuss the landscape (the 360 view), and its index **points into `ILGPU.md` / `SILK.md`**
     (and `STREAMING.md`) for the code-level parts.
   - **`ILGPU.md` / `SILK.md` = *what's in the code and why it is like that*.** Architecture, files,
     classes, kernel designs, launch geometry, buffer layout, design decisions, implementation
     lessons — the detailed code-level "what and why".
   - **`STREAMING.md`** = the streaming topic doc: the shipped-behavior reference **and** the
     memory-budget-enforcement design proposal; `ACCELERATION.md` carries only its one-line roadmap
     entry + link. *(Revised 2026-10-13: the proposal got its own ungated document — see "Resolved
     during execution".)*
2. **They are currently scattered — consolidate them.** Outstanding work is today spread across the
   `§5` follow-ups (live items mixed with struck-through "done" ones), the `# Combined next steps`
   GPU/Streaming/CPU lists, the `# Streaming Memory-Budget Enforcement` proposal, and inline
   "not implemented / later decision / outstanding work" sentences. Gather **every** outstanding item
   into **one concise `# Not done / planned` section** (one line each: status + home-doc link +
   issue number) so it can be reviewed and discussed. Completed items reduce to a terse "done"
   marker or drop to their home doc's history.
3. **`ILGPU.md` and `SILK.md` each get two clearly identifiable sections.**
   - **Section 1 — the Spike work:** exactly what each document contains today (the GpuProbe case
     study). Preserved unchanged.
   - **Section 2 — the respective implementation:** how the production implementation actually works
     (`BertEncoderGpuRunner`, `ModernBertGpuRunner`, `LayaHeadGpuRunner`, `GemmKernels`,
     `AttentionKernels`, `ElementwiseKernels`, `IlgpuRuntime`, `GpuBuffers` for ILGPU; `SilkRuntime`,
     `SilkKernels`, `SilkBertEncoderRunner` and the CLI/gate for Silk). **Written using the code as
     the reference** (verified against source, not reconstructed from memory). Additive — the spike
     content is neither removed nor rewritten.

## Status (progress log)

- **Stages 1–2 complete and verified** — commits `f7ad4d7b` (naming), `2ad7228b` (kernels +
  gate), `f9183b8b` (head + routing).
  - `dotnet build Nivara.slnx` — clean, 0 warnings.
  - `dotnet test -c Release --filter "FullyQualifiedName~InferenceCliArgumentTests"` — 36/36 pass.
  - `--silk-kernels` gate — **PASS**: 32/32 cells bit-identical to ILGPU, including `gather_rows`,
    `gemm_row4_relu`, and the DistilBERT headDim-64 attention cell (the `MaxHeadDim` boundary).
    6 cells fall outside the x0.8–x1.2 informal timing band — red flags to report, not failures:
    minilm fc1 gelu x1.24, minilm attention x1.26, distilbert qkv bias x1.43, distilbert qkv relu
    x1.25, distilbert fc1 bias x1.26, distilbert fc1 relu x1.33.
  - `distilbert_sst --silk compare` — **PASS**: 0/16 violations, argmax 8/8 vs CPU and vs PyTorch.
  - `distilbert --silk compare` — **PASS**: 0/98304 violations (PyTorch fixture absent).
  - `distilbert_sst --ilgpu compare` — **PASS** (the ILGPU path is unaffected by the refactor).
- **Stage 4 complete and verified** — commit `dd55a0b8`.
  - `ACCELERATION.md` is now story + index + one `# Not done / planned` list (zero fenced blocks);
    `ILGPU.md` and `SILK.md` each carry Part 1 (spike) + Part 2 (production implementation, written
    from the code); `STREAMING-PROPOSAL.md` is the new ungated home of the streaming proposal, linked
    from `STREAMING.md`.
  - `dotnet test -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` — **62/62 pass**
    (snippet gate compiles every gated block; citation gate resolves every `File.cs:NN`). Pinned
    snippet and citation counts updated deliberately.
  - Every inbound section anchor repaired (`ACCELERATION.md` §1b/§5.2/§5.5 and "lesson 15/19" →
    `ILGPU.md`/`SILK.md`); the citation gate also surfaced and fixed two wrong line numbers and one
    ambiguous basename.
- **Stage 3 (docs for `--silk` DistilBERT) next.** `docs/SILK.md` Part 2 (scope, kernel set, head
  path, gate), `docs/DISTELBERT.md` GPU path, `samples/NivaraInference/README.md`, `CHANGELOG.md`,
  and the `Program.cs` help/scope strings.

## Proposed changes

### Stage 1 — `distilbert --silk` (encoder)

`samples/Nivara.Samples/Gpu/SilkBertEncoderRunner.cs`
- Add a `BertGpuNaming naming` ctor parameter; store it in a field; use it at both sites that
  currently hardcode `BertGpuNaming.Bert` (ctor line ~88 and `UploadLayer` line ~273).
- Keep the head rejection for Stage 1 (the `distilbert` base checkpoint has no head, so it never
  fires on the intended input); it is removed in Stage 2.
- Update the doc comment's MiniLM-only / six-kernel wording once Stage 2 lands.

`samples/NivaraInference/Program.cs`
- `OpenEncoder` delegate (line ~1409) gains a `BertGpuNaming naming` parameter. `OpenIlgpuEncoder`
  passes it through (currently hardcodes `Bert`); `OpenSilkEncoder` passes it to the new ctor. The
  three MiniLM bodies pass `BertGpuNaming.Bert`.
- Refactor `RunDistilBertGpuInference` / `BenchmarkDistilBertGpu` / `RunDistilBertGpuCompare`
  (lines ~1922, ~1969, ~2127) into backend-agnostic bodies parameterized by
  `(string backend, OpenEncoder open, BertGpuNaming naming, tensors)`, mirroring the MiniLM trio,
  with `*Gpu` and `*Silk` wrappers. They use `DistilBertConfig.ToBertConfig()` and `DistilBert`.
- `SilkModes` (line ~50): add `["distilbert"] = ["", "benchmark", "compare"]`; rewrite the
  MiniLM-only comment above it.
- Dispatch `case "distilbert"` (line ~561): add `if (useSilk) { … RunDistilBertSilk* }` before the
  `useGpu` branch.
- Guards/help: model-scope message (line ~358) → `(Silk models: minilm, distilbert, distilbert_sst)`;
  help text (lines ~252–255); F32 message (line ~462) "six-kernel Silk stack" → "hand-authored Silk
  kernel stack".

`tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs`
- Update `Silk_OnModelWithoutSilkKernels_Rejected` expected string (line ~263) and its doc comment
  (lines ~254–259).

`tests/Nivara.PerformanceTests/SilkKernelsGate.cs`
- Add a DistilBERT-shaped attention cell at headDim 64 (numeric bound), exercising the
  `MaxHeadDim` boundary the MiniLM dims do not.

End-to-end: `distilbert --silk compare` — GPU vs same-process CPU
(`|gpu−cpu| ≤ 1e-3·(1+|cpu|)`). `last_hidden_state_py.bin` is absent, so GPU-vs-CPU only (the
situation ModernBERT was in).

### Stage 2 — `distilbert_sst --silk` (head)

`samples/Nivara.Samples/Gpu/SilkKernels.cs`
- Add `gather_rows` — exact port of `ElementwiseKernels.Gather` (pure copy → byte-identity):
  `__kernel void gather_rows(__global const float* table, __global const int* ids, __global float* out, const uint hidden, const uint total)`.
- Add `gemm_row4_relu` — third epilogue over the existing `GEMM_ROW4_ACCUMULATE` macro
  (`c = max(acc + bias, 0)`; pure arithmetic → byte-identity, like `gemm_row4_bias`).
- Add both to `KernelNames` (line ~68) and update the "six kernels" doc comment. `SilkRuntime`
  compiles the whole translation unit and looks up each name in `KernelNames`, so these two edits
  are all that is required host-side.

`SilkBertEncoderRunner.cs`
- Replace the head rejection with the ILGPU pattern
  `hasHead = tensors.ContainsKey(PreWKey) && tensors.ContainsKey(ClsWKey)`; upload
  `preW`/`preB`/`clsW`/`clsB` (GEMM weights transposed) and allocate `clsOut` (`8*hidden`),
  `logits` (`8*2`), `clsIds`.
- In `Forward`, after the layer stack: upload `clsIds[b] = b*seqLen`;
  `Gather → gemm_relu(clsOut→h) → gemm_bias(h→logits, bCols=2)`; read back and return
  `new BertEncoderGpuResult(x, logits)`. Mirrors `BertEncoderGpuRunner.Forward` lines ~305–320.
- Add `LaunchGather` and a `LaunchGemmRelu` path (the existing `LaunchGemm` helper already takes a
  kernel name, so `gemm_row4_relu` needs no new code beyond passing the name).

`Program.cs`
- Add `["distilbert_sst"]` to `SilkModes`; refactor `RunDistilBertSstGpuInference` /
  `BenchmarkDistilBertSstGpu` / `RunDistilBertSstGpuCompare` (lines ~2000, ~2043, ~2184) into
  backend-agnostic bodies with `*Silk` wrappers; add the `distilbert_sst` `if (useSilk)` dispatch
  branch; update the model-scope string.

`SilkKernelsGate.cs`
- Add byte-identity cells for `gather_rows` and `gemm_row4_relu`.

End-to-end: `distilbert_sst --silk compare` — logits vs CPU + argmax 8/8, and the PyTorch 3-way
(`compare_distilbert_sst_py.bin` is present).

### Stage 3 — `--silk` DistilBERT docs (feature content, into the *final* homes)

Lands **after** the Stage 4 restructure, so the new content is written directly into its home doc
instead of into `ACCELERATION.md` and then moved.

- `docs/SILK.md` — extend its model-path section (created by Stage 4): scope MiniLM + DistilBERT +
  SST-2; kernel set six → eight (`gather_rows`, `gemm_row4_relu`); the device-pure head path; gate
  coverage (`--silk-kernels` cells + the two new E2E compares). Add the DistilBERT informal A/B only
  if measured on AC (else "not measured / non-quotable", as the MiniLM battery note does).
- `docs/DISTELBERT.md` — "GPU path": add a `--silk` paragraph/row + verification-table rows.
- Consistency (same branch): `samples/NivaraInference/README.md` quick start, `CHANGELOG.md`
  Unreleased bullet, and the `Program.cs` help/scope strings.

### Stage 4 — Documentation restructure (`ACCELERATION.md` → index + home docs + one plan list)

**Refined by the 2026-10-12 directives (see above):** (a) the outstanding-work list is a
*consolidation into* `ACCELERATION.md`, not a scattering out of it — the plan lives here, only
supporting detail moves; (b) `ILGPU.md` and `SILK.md` each gain an **additive** production-model-path
section written from the code (directive 3), keeping their spike content intact.

Motivation (raised by the human): `docs/ACCELERATION.md` has grown into the single home for every
acceleration topic — ILGPU production lessons, the Silk.NET `--silk` backend, the GpuProbe verdicts,
and a streaming memory-budget proposal — so it is both **too long** and **internally mixed**.
Restructure it into an **overview/index** that points at cohesive per-topic documents, and make the
specific documents self-contained:

| content in `ACCELERATION.md` | destination |
|---|---|
| title/intro index, `§1` cross-backend results table + the shape-specificity paragraph, `§5` promotion decision, CPU-side pointer, References | **stays** — `ACCELERATION.md` is the overview/index |
| `§1` ModernBERT GPU (#449) + Laya head GPU (#462) blocks, M2-fusion block, correctness-gate bullets, `§2` ILGPU file rows + design decisions, `§3` the 19 ILGPU lessons, `§4` reality-vs-estimate, `§5` ILGPU follow-ups | **`docs/ILGPU.md`** — merge with the existing GpuProbe ILGPU case study so it is the one cohesive ILGPU document (probe verdicts **and** the production model path), **plus a new production-model-path section written from the code** (directive 3) |
| `§1` "Second OpenCL backend: `--silk`" subsection, the Silk row of `§2`, Silk gates and follow-ups | **`docs/SILK.md`** — merge with the existing Silk.NET OpenCL probe doc so it is the one cohesive Silk document (spike **and** model path), **plus a new production-model-path section written from the code** (directive 3) |
| the whole `# Streaming Memory-Budget Enforcement` section (Phase 1 / #325 design proposal, "not implemented" banner) and the `## Streaming memory budget` combined-next-steps block | **`docs/STREAMING.md`** — as a clearly-bannered "Design proposal: memory-budget enforcement (#325)" section beside the existing AC3 section; it is a streaming topic, not a GPU/CPU acceleration topic |
| cross-references into moved anchors (GpuProbe series headers, `tests/Nivara.GpuProbe/README.md`, `docs/*.md` links) | fix in the same commit |

- **Consolidate "not done / planned / future" into one list at the bottom of `ACCELERATION.md`**
  (directive 2 — the human's primary ask: one concise, reviewable section).
  Today it is scattered across struck-through done items in `§5`, the `# Combined next steps`
  GPU/Streaming/CPU subsections, and inline "not implemented / later decision / outstanding work"
  sentences. Move **every outstanding item** into a single **`# Not done / planned`** section at the
  bottom (of the index doc), one line each with a link to its home doc and its issue number;
  completed items become a terse "Done" marker or drop to the home doc's history. No future-work
  item should remain only inline in a results section.
- **Ordering note.** Because the ILGPU/Silk/streaming specifics move *out*, this restructure is a
  near-pure move plus link repair — no rewriting of the technical content, so the diff stays
  reviewable. Stage 3 then adds the DistilBERT `--silk` content to the new homes.
- **Deferred decision — RESOLVED 2026-10-12: the restructure lands on *this* branch as its own
  commit.** Stage 3 follows it, so the DistilBERT `--silk` content is written straight into the final
  homes instead of into `ACCELERATION.md` and then moved.

## Verification steps

- `dotnet build Nivara.slnx`
- `dotnet test -c Release --filter "FullyQualifiedName~InferenceCliArgumentTests"` (targeted;
  ask before the full suite)
- `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --silk-kernels`
- `dotnet run -c Release --project samples/NivaraInference -- distilbert --silk compare`
- `dotnet run -c Release --project samples/NivaraInference -- distilbert_sst --silk compare`
- Benchmarks (`… distilbert --silk benchmark`, `… distilbert_sst --silk benchmark`) — **Release,
  AC power only** for numbers that go in the docs.
- Docs restructure (Stage 4): `grep -rn "ACCELERATION.md" docs tests samples` to find links/anchors
  into moved sections and repair them; confirm the moved fenced blocks (if any) still satisfy the
  documentation snippet gate — `dotnet test -c Release --filter "FullyQualifiedName~SnippetGate"`
  or the `EveryGatedBlock_CompilesWithoutErrors` test.

## Blast radius

- **Sample + gate only.** No `src/Nivara` changes; the core library, its public API and its tests
  are untouched.
- `samples/Nivara.Samples/Gpu/SilkKernels.cs` — shared by `--silk` (sample) and `--silk-kernels`
  (gate); adding kernels is additive.
- `samples/Nivara.Samples/Gpu/SilkBertEncoderRunner.cs` — implements `IBertEncoderRunner`; the
  MiniLM path is exercised by `minilm --silk {,benchmark,compare}` and must stay green.
- `samples/NivaraInference/Program.cs` — CLI surface; the `(model, mode)` allow-lists and guard
  messages are pinned by `InferenceCliArgumentTests`, which must move in lockstep.
- `tests/Nivara.PerformanceTests/SilkKernelsGate.cs` — standalone harness (`--silk-kernels`), not
  part of the CI unit suite.
- Downstream callers: none outside the sample (the `IBertEncoderRunner` seam is sample-internal).
- **Docs (Stage 4):** `docs/ACCELERATION.md`, `docs/ILGPU.md`, `docs/SILK.md`, `docs/STREAMING.md`
  are restructured (content moved, not rewritten); inbound links from `docs/*.md`,
  `tests/Nivara.GpuProbe/README.md`, `samples/NivaraInference/README.md`, `AGENTS.md`, `CHANGELOG.md`
  and README files must be re-pointed. No code or test behavior changes.

## Planned commits

1. `Add bert-family weight naming to the Silk encoder runner` — Stage 1 code + `distilbert`
   routing/scope + CLI test update.
2. `Add gather_rows and gemm_row4_relu Silk kernels` — Stage 2 kernels + `KernelNames` + gate cells.
3. `Port the DistilBERT classification head to the Silk runner` — head support + `distilbert_sst`
   routing/scope.
4. `restructure acceleration docs into an index plus backend home docs` — Stage 4: move the
   ILGPU/Silk/streaming specifics into `ILGPU.md` / `SILK.md` / `STREAMING.md`, repair links, and
   consolidate outstanding work into one `# Not done / planned` section at the bottom of
   `ACCELERATION.md`. (Placement on this branch vs a separate PR is the deferred decision above.)
5. `document --silk support for the DistilBERT family` — Stage 3: `SILK.md`, `DISTELBERT.md`,
   `README.md`, `CHANGELOG.md`.

Order note: 4 lands before 5 so the new DistilBERT `--silk` docs are written directly into their
final home rather than into `ACCELERATION.md` and then moved.

## GitHub issues log

- [ ] (none yet)
