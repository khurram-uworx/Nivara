# Plan: Silk attention Torch parity — Phase C (#775)

## Problem

Issue #775 asks to extend Silk (OpenCL) Torch-parity coverage to the attention kernels (`batched_attention`, `causal_gqa_attention`), as a new test class `SilkAttentionTorchParityTests` under `tests/Nivara.Tests/Gpu/`. Coverage must include the banded bidirectional + padding case where a fully-masked row occurs, with `Assert.Ignore` on no-GPU (CI-safe).

## Proposed changes

- Add `tests/Nivara.Tests/Gpu/SilkAttentionTorchParityTests.cs` (test-only; no production code changes).
- `batched_attention` coverage:
  - banded bidirectional + padding against `attn_band_padding_*` fixtures (band=1, validLength=6), including explicit assertion that fully-masked row 7 is exactly zeros and no NaN.
  - global band against `attn_self_*` fixtures (all keys valid).
- `causal_gqa_attention` coverage:
  - self-causal MHA against `attn_self_causal_*` fixtures (numHeads=numKvHeads=4).
- Use `TestHelpers.AssertTensorEqual` (default tolerance) and `SilkLauncher` APIs exactly as the existing parity suites do.

## Grounding / blast radius

- Blast radius: new test file only. No `src/` changes. Risk limited to test compile/runtime. No ADR/DocsCoveragePins impact expected (no docs changes, no citation counts).

## Verification

- `dotnet build Nivara.slnx`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkAttentionTorchParityTests"` (ask before run)

## Planned commits

1. `docs: plan #775 silk attention torch parity in TODO.md`
2. `test: add SilkAttentionTorchParityTests for issue #775`
3. `docs: remove TODO.md — plan executed`

## GitHub issues log
(none)
EOF