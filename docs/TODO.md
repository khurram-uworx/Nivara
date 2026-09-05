# TODO: Qwen2.5 showcase in NivaraInference — tool loop + teacher distillation

Branch `khurram/qwen-inference` (created off `khurram/qwen` tip `3761ecb`).
The NivaraInference sample gains a `qwen` model mode with two sub-shows:
native function calling (`tools`) and **teacher-annotated-data distillation** into a
tiny sentiment classifier (`distill`) — the "cool" ML-heavy showcase. The README
also documents, per model, the **library enhancements / gap fixes** this branch
(and #382) contributed.

Two user-mandated test conventions apply to everything this work touches:
(1) every new neural-network construct is pinned against PyTorch the way the existing
building blocks are, and (2) every fixture/model-gated test silently `Assert.Ignore`s
when its files are absent — the `Tokenizer_EncodeFinalPrompt_MatchesTorchIds` pattern. See §4–§5.

## GitHub issues log

- [ ] #NNN — record issue numbers here as they are created (create via
      `gh issue create --repo khurram-uworx/Nivara` the moment a concern is
      found, never hold in memory)

## Problem

NivaraInference showcases how to load HF checkpoints into Nivara's zero-dependency
engine and run them with PyTorch CPU parity + benchmarks, documenting the library
gaps each new model exposed. Qwen2.5-0.5B-Instruct is fully loadable today (branch
`khurram/qwen` shipped the tokenizer/loader/qkvBias groundwork) but the sample has
no `qwen` mode, and the README does not yet record the library work that made this
6th model load (the one additive `src/Nivara` change + the sample-scoped fixes).

## Proposed changes

### 1. `samples/NivaraInference/Qwen.cs` (new) — `qwen` mode

**Shared core** (both sub-modes):
- Load: `LlamaConfig.FromJson(config.json)` + `LlamaLoader.Load<float,float>(config, tensors)`
  (qkvBias auto-detected from the checkpoint — branch `fd92dc6`).
- Precision `--precision ushort|f32` (exact user scope; other values → clear error):
  - `f32` (default): existing `SafeTensorsLoader.Read(modelPath)` (BF16→F32 widen).
  - `ushort`: `SafeTensorsLoader.ReadUInt16(modelPath)` → `WidenToF32(...)` — raw-BF16 read,
    half the byte payload, SIMD widen, identical F32 weights (~2.5× faster load; branch `a301cd1`).
- Tokenizer: `new Gpt2BpeTokenizer(vocab.json, merges.txt, tokenizerJsonPath: tokenizer.json)`
  (Qwen Split-regex path + added tokens; branch `48c789a`/`0acf104`).
- Generation: greedy, KV-cached (`LlamaKVCache<float>(numLayers, kvWidth)` +
  `model.ForwardCached`) after a per-turn `SeedCache(ids, cache)` prefill; stop ids
  hardcoded `[151645, 151643]` (Qwen `eos_token_id` is an array). `model.Eval()`,
  decode outside `GradientUtils.Grad()` (ADR-001/002 inference-default).
- Minimal Qwen ChatML/tool renderer copied into the sample (format-faithful, no MEAI —
  NivaraInference stays dependency-free): system+tools turn via the same
  type-first/json-literal JSON layout as `QwenChatTemplate.BuildToolsSystemMessage`,
  `<tool_call>` parsing (strict-then-tolerant regex), `<tool_response>` user turn.

**`qwen` / `qwen tools` — native function calling with KV cache + PyTorch diff:**
- Fixed prompt "What's the weather in Paris?" + baked GetWeather tools system turn.
- Turn 1: render → encode → KV prefill → decode → parse `<tool_call>` (19 ids) →
  `GetWeather("Paris")` → feed `<tool_response>` back in a `user` turn.
- Turn 2: render the full final prompt → decode final answer (semantic).
- Fixture diff (when `samples/data/qwen_tool_*.bin` present; skip gracefully otherwise):
  rendered prompt-1 ids == `qwen_tool_prompt_ids.bin` (byte-exact); tool-call turn
  == first 19 ids of `qwen_tool_ids_py.bin` (byte-exact); final answer text contains
  "partly cloudy"; last-position logits vs `qwen_tool_logits_py.bin` within the SF
  tolerance (3% rel + 0.5 abs floor — same envelope as `QwenInstructParityTests`).
- Flags: `--no-kv-cache` (naive re-forward A/B timing), `benchmark` (median-of-3
  decode timing).

**`qwen distill` — the showpiece: teacher-annotated-data distillation via tool calls:**
- Embedded corpus in `Qwen.cs`: 10 train sentences + the 8
  `DistilBertSst.CompareSentences` eval rows; **gold labels stored ONLY for eval**
  (used to score teacher/student; never shown to the teacher or student at train time).
- Teacher pass: one fresh KV-cached chat per row with a `classify_sentiment`
  tool (`label: "positive"|"negative"`); parse the tool-call arguments → label
  (structured, machine-verifiable). Malformed/refused rows logged and excluded.
  Results appended incrementally to `samples/data/qwen_distill_labels.json`
  (**resumable** — a long first run can be split across invocations); `--force`
  regenerates; `--teacher-examples N` caps total rows. ETA printed per row
  (honest ~1–2 min/row at ~0.3–1 tok/s; cache makes reruns instant).
- Student: hashed word+bigram bag-of-words features (~4096) →
  `SentimentMLP<float>` = `Linear<float>(4096→64)` + `ReLU` + `Linear<float>(64→2)`
  (a small `Module<float>` subclass, ~0.3 M params). Trained with public AutoDiff
  APIs: `Adam<float>(lr 1e-3)` + `optimizer.AddParameterGroup(model.GetParameters().Values)`,
  `CrossEntropyLoss<float>().Forward(logits, int[] targets)`, loss `Backward()` +
  `optimizer.Step()`, all inside `using (GradientUtils.Grad())` (training is the
  opt-in path). Fixed seed; ~200 epochs full-batch; linear-only baseline printed.
- Report: per-row table on the 8 eval rows with columns
  `Qwen(teacher) | Student | DistilBERT SST-2 | Gold` (DistilBERT column via the
  sample's existing `DistilBertSst.PredictLogits`, skipped if its model dir is
  absent), teacher-vs-gold / student-vs-gold / student-vs-teacher accuracies,
  params, epochs, timings.

### 2. `samples/NivaraInference/Program.cs` — wiring
- Usage line + help text: add `qwen` to the model list, document sub-modes
  (`tools`/`distill`/`benchmark`) and `--precision ushort|f32`.
- Precision normalization: add `"ushort"`.
- Top-level tensor load: `precision == "ushort"` → `WidenToF32(ReadUInt16(path))`
  with a distinct timing label (avoid double-reading the file).
- `case "qwen"` dispatch → `RunQwenTools` / `RunQwenDistill` / benchmark.
- qwen-specific flags parsed like the existing `--simd-widen` style:
  `--teacher-examples`, `--force`, `--no-kv-cache`, `--seed`, `--text`.

### 3. `samples/NivaraInference/README.md` — documentation (incl. library story)
- Quick start: Qwen download command + `-- qwen` and `-- qwen distill` runs.
- Supported-models table row; Usage block; architecture section
  "Qwen2.5-0.5B-Instruct (Qwen/Qwen2.5-0.5B-Instruct)".
- **"Core library improvements (gaps found & filled by the 6th model)"** —
  per the user's explicit direction, itemize everything this branch contributed
  (documented along the model, as the README does for SmolLM):
  - Core `src/Nivara` — ONE additive change: optional `qkvBias` on
    `LlamaCausalAttention<T>` / `LlamaDecoderBlock<T>` (default `false`, canonical
    Llama unchanged; **#384**, Torch-verified, tool turn 19/19 byte-exact).
  - Sample-scoped `samples/Nivara.Samples` (shared with NivaraChat): Qwen
    Split-regex pretokenizer path + added-token merge (`48c789a`, `0acf104`);
    `SafeTensorsLoader.ReadUInt16` + SIMD `WidenBf16ToF32` (`a301cd1`,
    ~2.5× load); `LlamaLoader` qkvBias auto-detect (`fd92dc6`).
  - Gaps that did NOT need filling (generalized already): RoPE/GQA/RMSNorm/SiLU/
    KV cache/tied head — the SmolLM-era stack.
  - Cross-refs: `docs/research/QWEN-TOOL-CALLING.md`, #384.
- Narrow-precision (`ushort` vs `f32` loader bench), Results rows, Sample-data
  rows (incl. `qwen_distill_labels.json` cache), "capabilities exercised"
  subsection for Qwen (tools loop, KV-cached decode, distillation, structured
  output), and honest notes (first teacher pass ~20–30 min once, cached after;
  `fp16` untested).

### 4. Torch-parity for new neural-network constructs (user-mandated, step 1)
- **General rule:** any new NN construct this work adds must be pinned against PyTorch
  the same way existing building blocks are (NivaraTorch block-parity suite + committed
  `samples/data/torch-comparison/` fixtures).
- **Already covered on the branch:** the only additive `src/Nivara` change — `qkvBias` on
  `LlamaCausalAttention<T>` / `LlamaDecoderBlock<T>` — is Torch-verified by
  `QwenInstructParityTests` (byte-exact prompt ids, tool turn 19/19) plus the tool-loop
  logits diff (`qwen_tool_logits_py.bin`, 3% rel + 0.5 abs). The student's building blocks
  (`Linear`, `ReLU`, `CrossEntropyLoss`, `LogSoftmax`) each have committed parity fixtures
  (`linear_128_64_*`, `relu_1d_*`, `cross_entropy_*`).
- **New for the composed `SentimentMLP` (Linear→ReLU→Linear→CrossEntropyLoss):** pin the
  COMPOSITION end-to-end against a fresh Torch fixture, not just the kernels:
  - New committed reference script `samples/NivaraInference/Python/qwen_distill_reference.py`
    (**user-approved relaxation** of the earlier "no new Python files" scope — the branch's
    `qwen_tool_reference.py` cannot generate this). It writes a small **model-independent**
    fixture set: fixed seeded `Linear(4096→64)` + `Linear(64→2)` weights, input features
    `[B×4096]`, int targets, gold forward logits, gold mean-CE loss, and gold first-layer
    weight gradient after one `Backward` — to committed `samples/data/qwen-distill/*.bin`
    (same layout as the committed `torch-comparison/` fixtures; CI-runnable without the
    989 MB Qwen checkpoint).
  - New NUnit test `tests/Nivara.Tests/Qwen/QwenDistillStudentParityTests.cs`: builds the
    identical stack inline from `Nivara.AutoDiff.Nn` (the tests project references Nivara +
    Nivara.Samples, not the NivaraInference exe), injects the fixture weights, asserts forward
    logits + the one-backward weight gradient match Torch (tolerance ~1e-4); per §5 it
    `Assert.Ignore`s when the fixture files are absent.

### 5. Fixture/file-gated tests — silent-skip convention (user-mandated, step 2)
- Every test written in this work must silently `Assert.Ignore` when the model files / fixtures
  it needs are absent, mirroring the existing pattern the user cited:
  `QwenInstructParityTests.Tokenizer_EncodeFinalPrompt_MatchesTorchIds` (gates in the
  `Tokenizer`/`Model` property getters + `ReadInt32`/`ReadFloat32` helpers, lines 25–118),
  `Gpt2BpeTokenizerTests` (lines 30–31: "SmolLM tokenizer files absent; skipping…"), and
  `QwenChatTemplateTests` / `QwenToolsWeatherLoopTests` (lines 33–51).
- Applies to: the new student-parity test (fixture dir absent → Ignore) and the sample's
  `qwen tools` fixture diff (fixtures absent → skip the diff, keep the end-to-end run).
- Existing Qwen tests already follow this convention; no changes needed there.

### 6. CI — run build + tests on every PR (user-mandated)
- **Why:** `.github/workflows/ci.yml` currently triggers `pull_request` only when the *base*
  branch is `main`. This branch's stacked PR targets `khurram/qwen` — a non-main base — so
  today CI would never run on it.
- **Change:** drop the `branches: [ main ]` filter on the `pull_request` trigger so *every*
  PR runs the build + test job (any base branch). Keep `push` on `main` as-is. GitHub
  evaluates the workflow from the PR head, so committing the new `ci.yml` on this branch is
  what makes the stacked PR light up.
- **Stays green without model data:** the test job already filters `Category!=Performance`;
  the new parity test silently `Assert.Ignore`s when its committed fixtures are absent, and
  the 989 MB Qwen checkpoint is not a CI dependency. No runner upgrade needed.

## Verification

- `dotnet build Nivara.slnx` (0 warnings), then **ask** before `dotnet test`.
- Acceptance (local model/fixtures present, from the verified #382 data):
  - `qwen tools` → transcript `[assistant → getWeather(city: Paris)]` →
    tool result → "partly cloudy" final answer; fixture diffs pass (prompt byte-exact,
    tool turn 19/19, logits ≤ 3% rel + 0.5 abs).
  - `qwen distill --teacher-examples 12` → teacher labels cached, student trains,
    eval table printed with real numbers for the README.
  - `--precision ushort` vs `f32` → record load timings for the README bench row.
- Torch-parity (student): run `python samples/NivaraInference/Python/qwen_distill_reference.py`
  (Torch env) to generate `samples/data/qwen-distill/*.bin`, then the new
  `QwenDistillStudentParityTests` pass (forward logits + backward grad ≈ Torch, ~1e-4). When the
  fixture files are absent the test silently `Assert.Ignore`s (CI/clean).
- README numbers come from the actual runs on this machine.
- CI evidence: after the stacked PR (base `khurram/qwen`) is created, the `CI` workflow
  must run build + tests on the PR head — proof the `pull_request` trigger now fires for
  non-main bases. Check the PR "checks" tab; the new parity test shows as skipped (Ignored)
  only when its fixtures are absent, otherwise passes.

## Planned commits

1. `docs: plan Qwen inference showcase + distillation in TODO.md` (this file)
2. `samples: add qwen tools + distill mode to NivaraInference (tool loop, KV cache, teacher distillation)`
3. `tests: pin composed student MLP against Torch (qwen_distill_reference.py + parity fixtures + test)`
4. `ci: run build+test on every PR (drop the pull_request main-only base filter)`
5. `docs(samples): document Qwen2.5 showcase and library gaps in NivaraInference README`
6. `docs: remove completed plan (iterative-work G2 cleared)` — after the two-gate review
7. Offer push + stacked PR (base `khurram/qwen`), human-confirmed.

Then `gh pr create` a stacked PR; do not push without confirmation.

## Blast radius

- **Files touched:** `samples/NivaraInference/Qwen.cs` (new), `samples/NivaraInference/Program.cs`,
  `samples/NivaraInference/README.md`,
  `samples/NivaraInference/Python/qwen_distill_reference.py` (new),
  `tests/Nivara.Tests/Qwen/QwenDistillStudentParityTests.cs` (new),
  `.github/workflows/ci.yml` (pull_request trigger: drop `branches: [ main ]`),
  `samples/data/qwen-distill/*.bin` (new committed small fixtures),
  `samples/data/qwen_distill_labels.json` (gitignored cache artifact).
- **No `src/Nivara` and no `samples/Nivara.Samples` changes** — everything needed already exists
  on the branch (tokenizer ctor, loader, qkvBias, KV cache). No new project references
  (Nivara + Nivara.Samples already referenced).
- **Downstream callers:** none — NivaraInference is a leaf sample. Existing modes
  (`smollm`, `distilbert_sst`, …) are untouched; the shared tensor-load block is only
  widened for the `ushort` precision value.
- **Tests:** ONE new test class (`QwenDistillStudentParityTests`) + one committed reference
  script, both isolated: the parity test is model-independent (synthetic fixtures) and
  silently `Assert.Ignore`s when its fixtures are absent. The existing Qwen tests already
  exercise the underlying behavior (template, parser, loader, tool loop) and are untouched;
  the sample re-diffs the same fixtures.
- **Runtime risk to flag:** the distill teacher pass is compute-heavy on CPU
  (~1–2 min/row). Mitigated by the resumable incremental label cache and
  `--teacher-examples`; documented in the README.