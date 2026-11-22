# Worklog SFT seed data

Synthetic worklog fine-tuning seed data for the Nivara LoRA SLM showcase.

## Files

- `train.jsonl` — 351 training examples
- `test.jsonl` — 99 evaluation examples
- `provenance.json` — deterministic generation metadata and hash

## Format

Each line is a JSON object with `messages`: `system`, `user`, `assistant`.

The task is behavioral/domain adaptation: convert natural-language worklog entries into
structured `LogTime` tool calls with fields `ticket`, `hours`, `date` (ISO `YYYY-MM-DD`).

## Domain conventions

- Standard office day = 8 hours.
- "all day" / "full day" = 8 hours (not 24h).
- "half day" = 4 hours.
- Explicit durations use the stated value; minute durations use a deterministic fraction rule
  (`minutes / 60`, rounded to 2 decimals, `MidpointRounding.AwayFromZero`).
- Relative dates are resolved deterministically from a fixed base date/seed policy
  (default base date `2026-07-19` in `provenance.json`).
- Multiple tickets produce separate tool calls.
- Ambiguous/non-worklog input asks for clarification.

## Provenance

Synthetic data generated deterministically by `NivaraFineTuning.WorklogSftDataset.GenerateSynthetic`
(seed/count/baseDate/ratios recorded in `provenance.json`). No external path dependency required.
