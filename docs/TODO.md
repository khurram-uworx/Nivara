# Plan — #479: `--seed` / `--teacher-examples` silently swallow an unparseable integer

Branch: `khurram/479` (off `khurram/478`). PR target: `khurram/478`.

## Problem

In `samples/NivaraInference/Program.cs` the arg loop parses two integer options with the
parameterless BCL overload and throws the failure away:

```csharp
else if (args[i] == "--teacher-examples" && i + 1 < args.Length)
{
    int.TryParse(args[i + 1], out teacherExamples);   // 'abc' -> 0 (== the "all" default)
    i++;
}
else if (args[i] == "--seed" && i + 1 < args.Length)
{
    int.TryParse(args[i + 1], out seed);              // 'xyz' -> 0, not the 42 default
    i++;
}
```

`--teacher-examples abc` silently becomes "all"; `--seed xyz` silently uses seed 0 instead of the
42 default. `RunDistill` (`Qwen.cs:707`) treats `teacherExamples <= 0` as "all" and only prints
`seed` (`Qwen.cs:844`, unused today). Either way a typo changes behaviour or is undetectable.

The issue's `## Notes` also flags `--precision`: an unrecognised value falls through
`var other => other` and is then used as a precision label (downstream, so louder, but still not
rejected). **User decision: include `--precision` in scope.**

`#474` (commit `58f06068`) introduced the model to follow: `TryParseCount` (invariant culture,
`NumberStyles.Integer`) and the `--seq/--warmup/--iters` branch that consumes its value
unconditionally so it can never fall through to the positional-mode branch.

## Blast radius

- Sole edit site: `samples/NivaraInference/Program.cs`, arg loop (`for` at line 48) — branches at
  lines 50–60 (`--precision`), 75–79 (`--teacher-examples`), 80–84 (`--seed`).
- `TryParseCount(string raw, out int value)` (`Program.cs:435`) — reused, not modified.
- Downstream consumers unchanged: `RunDistill` (`Qwen.cs:707`), `precision` compared only against
  `"fp16"`/`"bf16"` (`Program.cs:232-233`) and printed.
- No `src/Nivara` code touched. No public API change.
- Blast radius is confined to the sample CLI's argument surface. Nothing depends on these symbols
  outside the sample.
- `grep` for parameterless `TryParse(`/`.Parse(` in `samples/NivaraInference/` returns only these
  two integer calls (plus `JsonDocument.Parse`, unrelated). Sibling samples' `int.Parse` throw on
  garbage, so they are not silent and are out of scope.

## Proposed changes

### 1. `--precision` (replace lines 50–60)

Drop the `&& i + 1 < args.Length` guard (so a trailing `--precision` is "needs a value", not a
mode-name fall-through). Keep the normalize-switch, then reject non-canonical results.

```csharp
if (args[i] == "--precision")
{
    if (i + 1 >= args.Length)
    {
        Console.Error.WriteLine("--precision needs a value (f32|bf16|fp16).");
        return 1;
    }

    precision = args[i + 1].ToLowerInvariant() switch
    {
        "f32" or "float" => "f32",
        "bf16" or "bfloat16" => "bf16",
        "fp16" or "f16" or "half" => "fp16",
        var other => other
    };

    if (precision is not ("f32" or "bf16" or "fp16"))
    {
        Console.Error.WriteLine($"--precision expects f32, bf16 or fp16; got '{args[i + 1]}'.");
        return 1;
    }
    i++;
}
```

Bare-token shortcuts (`bf16`, `fp16`, `f16`, `half`, lines 61–68) unchanged.

### 2 & 3. `--teacher-examples` and `--seed` (replace lines 75–84)

One uniform branch, shaped like `--seq`; value always consumed; reuses `TryParseCount`.

```csharp
else if (args[i] is "--teacher-examples" or "--seed")
{
    // The value is consumed even when it is nonsense, so the flag's value can never
    // fall through to the positional-mode branch below and be read as a mode name.
    if (i + 1 >= args.Length)
    {
        Console.Error.WriteLine($"{args[i]} needs a value.");
        return 1;
    }

    if (!TryParseCount(args[i + 1], out int parsed))
    {
        Console.Error.WriteLine($"{args[i]} expects an integer; got '{args[i + 1]}'.");
        return 1;
    }

    if (args[i] == "--teacher-examples") teacherExamples = parsed;
    else seed = parsed;
    i++;
}
```

No range change: `0`/negative still mean "all" for `teacher-examples`; any int is a valid seed.
Using `TryParseCount` also makes a leading `+` (`+42`) culture-independent.

### 4. CHANGELOG

New `### Fixed` bullet under `[Unreleased]` (near the `#474` entry), ending `(#479)`.

## Non-goals

- Not restricting these flags to the models/modes that consume them (unlike `#474`'s `--seq`).
- Not trimming whitespace in `--precision` values.
- No automated test — no harness exists for the sample CLI (`Program` is internal,
  `Main`/`TryParseCount` private, no test project references it) and `#474` set no precedent.
  Verified manually (below).

## Verification (all reject paths fail before the model load)

| Command | Expected |
|---|---|
| `qwen distill --seed xyz` | `--seed expects an integer; got 'xyz'.`, exit 1 |
| `qwen distill --teacher-examples abc` | `--teacher-examples expects an integer; got 'abc'.`, exit 1 |
| `qwen --seed` | `--seed needs a value.`, exit 1 |
| `minilm --precision f64` | `--precision expects f32, bf16 or fp16; got 'f64'.`, exit 1 |
| `minilm --precision` | `--precision needs a value (f32\|bf16\|fp16).`, exit 1 |
| `qwen --seed 7 --teacher-examples 12` | validation passes -> reaches `Model file not found` when model absent |
| `qwen --seed +42`, `minilm --precision BF16` | accepted |
| `qwen --precision fp16`, `minilm --gpu --precision bf16` | existing downstream rejections unchanged |

Build: `dotnet build Nivara.slnx`.

## Planned commits

1. `docs: plan #479 integer-option rejection in TODO.md`
2. `fix(samples): reject unparseable --seed/--teacher-examples/--precision values (#479)`
3. `docs: remove TODO.md — plan executed`

## GitHub issues log

- (none yet)
