# Worklog SFT seed data

Synthetic worklog fine-tuning seed data for the Nivara LoRA SLM showcase.

## Files

- `train.jsonl` — 350 training examples
- `test.jsonl` — 100 evaluation examples

## Format

Each line is a JSON object with `messages`: `system`, `user`, `assistant`.

The task is behavioral/domain adaptation: convert natural-language worklog entries into
structured `LogTime` tool calls with fields `ticket`, `hours`, `date`.

## Domain conventions

- Standard office day = 8 hours.
- "all day" / "full day" = 8 hours (not 24h).
- "half day" = 4 hours.
- Explicit durations use the stated value.
- Multiple tickets produce separate tool calls.
- Ambiguous/non-worklog input asks for clarification.

## Provenance

Synthetic data owned by the project maintainer and included in-repo for deterministic
fine-tuning/eval. No external path dependency required.
