# Plan — add `--seq` to `modernbert_benchmark.py` (issue #477)

## Problem

`modernbert --gpu benchmark` / `modernbert benchmark` gained `--seq` in #474
(`Program.TryParseSeqLengths`), so a single row of the published GPU/CPU table can be
re-measured. The PyTorch half of the CPU comparison,
`samples/NivaraInference/Python/modernbert_benchmark.py`, hardcodes

```python
MAX_LENGTHS = (128, 256)
```

and takes no arguments at all — no `argparse`, no `sys.argv` (the file only uses `sys`
for `sys.path`). So a like-for-like CPU A/B at one length still means editing the file,
and the two sides can be measured at different lengths by accident — the exact failure
`--seq` was added to remove. The C# and Python tables are read side by side in
`samples/NivaraInference/README.md`, so if one side can be steered from the command line
and the other cannot, the pair drifts.

Parity is with `ModernBert.RunBenchmark`, which the script's own docstring says it
"mirrors deliberately".

## Proposed changes

### `samples/NivaraInference/Python/modernbert_benchmark.py`

- Add `import argparse`.
- Keep `MAX_LENGTHS = (128, 256)` as the default (do not delete it); `--seq` defaults to
  `None`, and the resolved list falls back to `MAX_LENGTHS`, so the constant stays the
  single source of truth and current output is unchanged.
- Add a module-level `_parse_seq_lengths(raw)` used as the argparse `type=`, mirroring
  `Program.TryParseSeqLengths`:
  - split on `,`, `.strip()` each token, skip empty tokens;
  - `int()` each token; reject non-integers and values `<= 0` with
    `argparse.ArgumentTypeError` (same wording family as the C# CLI);
  - error if nothing parsed;
  - return `sorted(set(...))` (ascending, de-duplicated).
  - The `--seq 1,000` trap resolves identically to C#: tokens `1` and `000`, and `000`
    parses to `0`, which is rejected.
- In `main()`: `ap.add_argument("--seq", type=_parse_seq_lengths, default=None,
  metavar="N[,N...]", help=...)`; `lengths = args.seq if args.seq is not None else
  list(MAX_LENGTHS)`; loop over `lengths`.
- Print the lengths actually run (an explicit line in addition to the per-row `seq=N`).
- Update the module docstring: the default is `(128, 256)`; `--seq` selects a subset.
  Note `WARMUP_PASSES = 3` is unchanged — the warmup-count divergence is parked in the
  #474 docs commit, not here.
- **No length cap.** The C# CPU path rejects `> ModernBertMasks.MaxDenseLength` (2048)
  because it allocates a dense `[L, L]` mask; the PyTorch reference has no such
  allocation constraint, so imposing a C#-CPU limitation on it would be artificial. The
  docstring notes that a length above 2048 has no C#-CPU partner (it pairs with
  `modernbert --ilgpu benchmark --seq`).

### `samples/NivaraInference/README.md`

- In the `--seq` paragraph (~line 208), state the Python script takes the same `--seq`
  (default `128,256`); add a `python modernbert_benchmark.py --seq 256` example.
- Note Python also takes `--seq` where the PyTorch commands are listed (~175, ~279).

### `docs/MODERNBERT.md`

- Line ~65 says `--seq` picks lengths "on both benchmark paths"; clarify it covers the
  C# CPU/GPU benchmark paths **and** the PyTorch script.

## Blast radius

- `samples/NivaraInference/Python/modernbert_benchmark.py` — sample-only, not compiled
  into any library or test. No C# code depends on it.
- `samples/NivaraInference/README.md`, `docs/MODERNBERT.md` — prose only.
- **No effect** on `src/Nivara`, `Nivara.Extensions`, the CLI (`Program.cs`,
  `ModernBert.cs`), or `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs`
  (that gate covers the C# `--seq` surface and is untouched).
- Default behavior (no args) is byte-for-byte the same output as before.

## Verification

- `python -m py_compile modernbert_benchmark.py` — syntax check.
- `python modernbert_benchmark.py --help` — **if** PyTorch/transformers and the model
  weights are installed; the module imports `torch`/`transformers` at top level, so
  `--help` needs the deps. Otherwise verification is the compile check plus a read of the
  parser against `TryParseSeqLengths`.
- No new test harness; the change is sample-only Python with no test project. No
  `dotnet test` run (nothing C#-side changed).

## Planned commits

1. `docs: plan #477 — add --seq to modernbert_benchmark.py in TODO.md`
2. `Add --seq to modernbert_benchmark.py so the PyTorch side matches the C# benchmark`
3. `docs: note the Python benchmark's --seq in README and MODERNBERT.md`
4. `docs: remove TODO.md — plan executed`

## GitHub issues log

- [ ] (none yet)
