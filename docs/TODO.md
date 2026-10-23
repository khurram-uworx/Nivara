# Plan: issue #823 — drop ILGPU MiniLM GPU path; keep Silk

Problem: MiniLM still reaches its GPU path through the ILGPU-backed
`OpenIlgpuEncoder` / `BertEncoderGpuRunner` via `RunMiniLmGpu*` wrappers, even though Silk
(`OpenSilkEncoder` / `SilkBertEncoderRunner`) is already wired behind `--silk`. The issue asks
that MiniLM expose only the Silk GPU backend and that `minilm --silk` default/benchmark/compare
still pass parity against CPU.

## Blast radius

- `samples/NivaraInference/Program.cs`: `GpuModes` table, ILGPU guard message text, usage text,
  `case "minilm"` dispatch, three `RunMiniLmGpu*` / `BenchmarkMiniLmGpu` wrappers.
- `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs`: the CLI-surface pins that name
  `minilm --ilgpu` (`GpuModeCases`, `GpuPrecisionCombination_Rejected`).
- `samples/NivaraInference/README.md`: one `minilm --ilgpu` example command.
- `OpenIlgpuEncoder` itself stays (DistilBERT / DistilBERT-SST still use it); `GpuModes`
  removal makes `minilm --ilgpu …` reach the existing model-scope guard.

## Proposed changes

1. Program.cs:
   - Remove `"minilm"` from `GpuModes`.
   - Drop `(GPU models: minilm, …)` → `(GPU models: distilbert, …)`.
   - Drop `minilm` from the usage help `GPU (minilm / distilbert / …)` → `GPU (distilbert / …)`.
   - In `case "minilm":`, delete the `if (useGpu) { … }` block; keep `useSilk` and CPU paths.
   - Delete `RunMiniLmGpuInference`, `BenchmarkMiniLmGpu`, `RunMiniLmGpuCompare`.
   - Keep Silk wrappers and backend-generic bodies (OpenSilkEncoder-only now); update stale
     comment that mentions both GPU backends.
2. InferenceCliArgumentTests.cs:
   - Remove the `minilm` entry from `GpuModeCases`.
   - Change `GpuPrecisionCombination_Rejected` to use `distilbert --ilgpu --precision bf16`
     (still an ILGPU model, F32-only) so the guard remains exercised.
3. README.md: replace `minilm --ilgpu` with `minilm --silk`.

## Verification

- `dotnet build Nivara.slnx`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~InferenceCliArgumentTests"`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- `minilm --ilgpu` prints `--ilgpu is not supported for 'minilm'`.
- Manual parity: `minilm --silk` (default), `minilm --silk benchmark`, `minilm --silk compare`
  each show the CPU/Silk parity gate PASS, pooled-embedding cosine min >= 0.9999.

## Planned commits

1. `docs: plan #823 in TODO.md`
2. `chore(inference): drop ILGPU MiniLM GPU entry points, keep Silk`
3. `test(inference): update CLI argument surface for minilm --ilgpu removal`
4. `docs(inference): point README at minilm --silk`
5. `docs: remove TODO.md — plan executed` (after G2 reviews)

## GitHub issues log

- (none yet)
