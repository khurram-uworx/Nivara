# Plan: #1407 — Mixed-dtype LoRALinear adapter save/load round trip

## Problem

Issue [#1407](https://github.com/khurram-uworx/Nivara/issues/1407) (OPEN, milestone 1.5.4): "Mixed-dtype LoRALinear adapter save/load round trip is uncovered". Follow-up from #1394 (mixed-dtype `LoRALinear<TModel, TWeight>` landed).

**Acceptance:** "adapter save/load round trip preserves A/B (F32 adapters) on a fresh frozen mixed base."

**User clarification:** "this is not ModelSerializer" — scope is the adapter-only serializer path (`LoraAdapterSerializer`), not `ModelSerializer`.

## Current state (grounded)

- `Nivara.AutoDiff.Nn.LoRALinear<TModel,TWeight>`: `src/Nivara/AutoDiff/Nn/LoRALinear.cs:157-294`.
- `LoraAdapterSerializer` mixed Save/Load: `src/Nivara/AutoDiff/Serialization/LoraAdapterSerializer.cs:67-92,129-140`. Saves A/B (TModel/F32) + metadata; no base weights. Load validates format/rank/alpha/baseModelId/targets/shapes and restores A/B.
- Existing mixed adapter round-trip test already matches acceptance: `LoRALinear_AdapterSerializer_Mixed_RoundTrip_IdenticalLogitsOnFreshBase` (`tests/Nivara.Tests/AutoDiff/LoRALinearTests.cs:198-236`) — saves A/B, loads into fresh frozen mixed base with same BF16 weights, asserts identical logits within `1e-6f`.
- Single-dtype ModelSerializer round-trip exists (`LoRALinear_Serialization_RoundTrip`, L142-165); mixed-dtype ModelSerializer round-trip is absent but explicitly out of scope per user direction.

## Blast radius

- Expected code change: **none or minimal** (likely docs/issue-evidence only) unless verification reveals a real gap in LoraAdapterSerializer mixed path.
- Files inspected: `src/Nivara/AutoDiff/Nn/LoRALinear.cs`, `src/Nivara/AutoDiff/Serialization/LoraAdapterSerializer.cs`, `tests/Nivara.Tests/AutoDiff/LoRALinearTests.cs`, `src/Nivara/AutoDiff/Serialization/ModelSerializer.cs`.
- Tests covering area: `Nivara.Tests.AutoDiff.LoRALinearTests` (LoRA adapter serializer + mixed LoRA behavior).

## Proposed changes

1. **Verify acceptance is covered** — run targeted LoRA tests; confirm `LoRALinear_AdapterSerializer_Mixed_RoundTrip_IdenticalLogitsOnFreshBase` passes and satisfies acceptance as written.
2. **If gap found** — add/fix the minimal LoraAdapterSerializer mixed-dtype adapter round-trip coverage (not ModelSerializer). No format bump without concrete need.
3. **If already covered** — record evidence in PR/issue; no production code churn. Only touch DocsCoveragePins if doc/citation counts change (not expected).
4. **Branch/PR hygiene** — review branch, push, create PR, test PR plan, update PR description with results.

## Verification steps

- [ ] Ask before running tests.
- [ ] `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~LoRALinearTests"` (targeted).
- [ ] If changes touch docs, run Docs namespace gate as required by AGENTS.
- [ ] `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` before push.
- [ ] Review branch diff against TODO (G2) before deleting TODO.

## Planned commits

1. `docs: plan #1407 LoRA adapter round trip in TODO.md`
2. (only if needed) `test: cover mixed-dtype LoRA adapter round trip` / fix commit
3. `docs: remove TODO.md — plan executed`

## GitHub issues log

- [ ] #1407 — Mixed-dtype LoRALinear adapter save/load round trip is uncovered (target issue)

## Grounding (G1) — to complete before implementation

- Confirm LoraAdapterSerializer mixed-dtype Save/Load semantics and acceptance mapping.
- Confirm no ModelSerializer scope per user direction.
- Confirm test covers "A/B (F32 adapters) on a fresh frozen mixed base" literally.
- Escalate any red flag/decision to human.
