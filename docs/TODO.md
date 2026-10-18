# Plan: Promote NFC normalization to shared byte-level BPE tokenizer path (issue #451)

## Problem

NFC normalization is implemented on one byte-level BPE entry point
(`LoadFromTokenizerJson`) and not on the shared legacy path, so a tokenizer that
needs normalization and arrives through the older route silently gets
unnormalized text. The flag is a per-call-site decision (`normalizeNfc = true`),
and `tokenizer.json` already carries a `normalizer` block that is ignored.

## Proposed changes

1. Add `ReadNormalizer(string json)` helper in `Gpt2BpeTokenizer` that parses
   the top-level `normalizer` block:
   - missing / null / `{"type":"None"|"Identity"}` (and empty Sequence) → false
   - `{"type":"NFC"}` or a Sequence/array containing only NFC → true
   - any other transform (`NFKC`, `NFKD`, `NFD`, `Lowercase`, `nmt_nfkc`,
     `Replace`, unknown) → throw `InvalidOperationException` naming the transform.
2. Change private constructor param from `bool normalizeNfc` to
   `bool? normalizeNfcOverride = null`; derive from file when
   `tokenizerJsonPath != null`, validating via `ReadNormalizer`; override wins if
   supplied.
3. Public vocab constructor forwards the new optional override (default null)
   instead of `false`.
4. `LoadFromTokenizerJson` signature becomes
   `LoadFromTokenizerJson(path, unkToken, bool? normalizeNfc = null)` (derive by
   default).
5. Make the Qwen split-regex branch conditional on `normalizeNfc` instead of
   unconditionally calling `.Normalize(FormC)`.

## Verification steps

- Build `Nivara.slnx`.
- Targeted tests: `Gpt2BpeTokenizerTests`, `Gpt2BpeTokenizerMergeFormatTests`,
  `QwenInstructParityTests`, `QwenToolsWeatherLoopTests`.
- New tests: file-derived NFC; unsupported transform throws naming it.
- SmolLM/Qwen fixtures stay byte-identical (including Qwen 206/258-id
  tool-prompt reference).

## Planned commits

1. `docs: plan issue #451 in TODO.md`
2. `feat: derive NFC normalization from tokenizer.json normalizer block`
3. `test: cover file-derived NFC and rejection of unsupported transforms`
4. `docs: remove TODO.md — plan executed`

## Blast radius

- src/Nivara.Extensions/AI/Gpt2BpeTokenizer.cs (field derivation, constructor,
  factory, Encode paths).
- Call sites in samples/tests retain optional override; no breaking change for
  existing positional callers.
- Qwen legacy path now derives NFC from file (file declares NFC); previously
  normalized only in split branch, so ASCII prompts unchanged.

## GitHub issues log

- (none yet)
