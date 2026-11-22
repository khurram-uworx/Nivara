# LoRA SLM iGPU FT showcase — tests, docs, release-readiness (#1421)

Status: active implementation on `khurram/1421` (branched from `khurram/1420`).

## 1. Goal

Add tests, docs, and release-readiness checks for the LoRA SLM iGPU FT showcase so D2a
(CPU/Silk parity, no compromise) is honestly represented and docs describe the showcase
accurately. See `docs/plan/LORA.md` (authoritative handoff) and issue #1421.

## 2. Scope

- Unit tests: LoRA parity/merge/freeze/gradcheck (extend existing), prompt masking, deterministic data gen, overfit sanity small model, adapter save/load round-trip, generic SmolLM smoke.
- GPU tests: Silk causal-LM LoRA parity, skip when no OpenCL.
- Docs: update `docs/plan/LORA.md`, add `docs/LORA.md` (or appropriate doc), `samples/NivaraFineTuning/README.md`, `samples/NivaraChat/README.md` only if touched.
- Check DocsCoveragePins/ADR index/citation counts if docs/citations change; update in same commit.

## 3. Acceptance criteria

- Targeted tests pass; docs gates pass when docs touched.
- No false iGPU/parity claims in docs.
- DocsCoveragePins updated if required.

## 4. Planned commits

1. `docs: plan #1421 in TODO.md`
2. `docs: update plan/LORA.md #1421 status and evidence`
3. `docs: add LORA.md feature doc (honest D2a status)`
4. `docs: touch NivaraFineTuning README if needed`
5. `docs: touch NivaraChat README if needed`
6. `test: add generic SmolLM smoke (CPU) if needed`
7. `docs: update DocsCoveragePins and run docs gates`
8. `docs: remove TODO.md — plan executed`

## 5. Blast radius

- `docs/plan/LORA.md`, new `docs/LORA.md` → DocsCoveragePins `ExpectedDocumentCount` moves (+1).
- `docs/LORA.md` citations (symbol-anchored) → CitationCount/CitedDocumentCount may move.
- No core source changes expected; tests only if gaps found.
- ADR index: no new ADR required by #1421.

## GitHub issues log

- (none yet)
```

Then commit.<|end|><|start|>Assistant to=bash<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name="command">
git add docs/TODO.md && git commit -m "docs: plan #1421 in TODO.md" 2>&1