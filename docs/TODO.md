# Plan: Polish typesafe nivara verify deltas and help text (#703)

## Problem

`typesafe nivara verify` only prints both answers (`in=1.54 (46%) http=1.54 (28%)`) without
explicit deltas. The verification section expects per-item/per-question deltas on per-option
probability, `AnswerConfidence`, and score against 1e-3, with a hard gate on typed decisions
and reported deltas otherwise. CLI help also does not mention the `nivara` engine or `verify`
mode, making the expanded surface undiscoverable.

## Proposed changes

1. `samples/NivaraInference/TypeSafe.cs`
   - Add `VerifyDeltaTolerance = 1e-3`.
   - In `Verify`, after the AGREE/DISAGREE summary line, print a `deltas:` block:
     * Choice: per-option probability delta (`in - http`), worst |Δ|, `AnswerConfidence` delta.
     * Score: score delta, per-level probability deltas, `AnswerConfidence` delta.
     * Noul: probability delta; AnswerConfidence from `AdditionalData` when available else `n/a`.
   - Add `AnswerConfidence(Answer)` and delta formatting helpers.
   - Hard gate stays `DecisionsAgree`; deltas are reported only.

2. `samples/NivaraInference/Program.cs`
   - Update top-level usage block: mode list add `verify`; TypeSafe engines add `nivara`;
     modes `default | benchmark | verify`; examples add `typesafe nivara`,
     `typesafe nivara benchmark`, `typesafe nivara verify`.

3. Docs
   - `docs/DECISION-MODELS.md` and `samples/NivaraInference/README.md`: update
     engine/mode wording and examples; note observed delta bound after run.

## Verification

- Build `NivaraInference` Release.
- Targeted: `InferenceCliArgumentTests`.
- Run `typesafe nivara verify` (Ollaya on :11435) and record deltas; confirm AGREE.
- If docs change: full `Nivara.Tests.Docs` namespace.

## Planned commits

1. docs: plan #703 polish in TODO.md
2. feat(inference): report explicit deltas in typesafe verify
3. docs/help: mention nivara engine and verify mode
4. docs: record observed verify deltas

## GitHub issues log

- (none created)

## Blast radius

- `TypeSafe.Verify`, `DecisionsAgree`, `Summarize` in `samples/NivaraInference/TypeSafe.cs`.
- Help-text tables/printing in `samples/NivaraInference/Program.cs`.
- CLI argument tests may assert help substrings; no guard messages changed.
- Docs cited by snippet/doc gates; keep structure unchanged.
