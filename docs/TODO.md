# Plan: Silk subgroup follow-ups (#1484)

## Problem

Follow-up to #1472/#1477. The subgroup capability gating and parity tests landed, but several items remain: measure benefit, exercise both capability paths on one device, audit local-memory cost of the hoisted `__local normed` array, explore a true K-split reduction, and document.

## Proposed changes (scoped)

1. **Force-fallback test hook** — add an opt-in disable to `SilkRuntime` (constructor parameter `enableSubgroups = true` plus `NIVARA_SILK_SUBGROUPS=0` env override) so the no-subgroup path can be exercised on a subgroup-capable device. `SupportsSubgroups` reflects the override.
2. **Tests** — add a Gpu-category test constructing `SilkRuntime(enableSubgroups: false)` and verifying subgroup launchers fall back (bit-identical to base for gemm; identical for rms since fallback = base).
3. **Local-memory audit note** — document the 16 KB/group `normed` allocation and occupancy consideration in `docs/SILK.md`; no code change (hoist is required by OpenCL).
4. **Docs** — `docs/SILK.md` section on subgroup capability flag, variant naming, parity policy, and force-fallback hook.

## Deferred (tracked as separate concerns)

- **Measurement (tok/s + GB/s before/after)** — benchmarks are human-run per AGENTS.md; methodology documented, numbers to be captured by human on SmolLM/Qwen profiles.
- **True K-split GEMV subgroup reduction** — requires accumulation-order/geometry redesign and a parity-policy decision; separate issue.

## Blast radius

- `src/Nivara.Extensions/Silk/SilkRuntime.cs` — constructor signature (additive optional param), `SupportsSubgroups` logic.
- `tests/Nivara.Tests/Gpu/` — new force-fallback parity test.
- `docs/SILK.md` — new section.

## Verification

- `dotnet build Nivara.slnx -c Release`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- New force-fallback test on GPU device.
- Docs namespace tests.

## GitHub issues log

- (none yet)
