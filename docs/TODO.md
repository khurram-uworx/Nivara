# Plan: #571 — reject `--teacher-examples` / `--seed` / `--force` outside `qwen distill`

- **Branch:** `khurram/571` (off `khurram/570`)
- **PR target:** `khurram/570`
- **Issue:** https://github.com/khurram-uworx/Nivara/issues/571

## Problem

`NivaraInference` parses `--teacher-examples`, `--seed`, and `--force` for every model
(`samples/NivaraInference/Program.cs:85-106`) but only `qwen distill` consumes them: the single call
site is `Program.cs:444` → `Qwen.RunDistill` (`Qwen.cs:707`). `minilm --seed 7` is accepted and does
nothing. This is the silent-acceptance failure mode #474 (`--seq`/`--warmup`/`--iters`) and #568
(`--gpu`) already closed: the operator cannot tell from the run that the option had no effect.

- #474 gave the ModernBERT timing flags the opposite treatment — rejected unless the model is
  `modernbert` and the mode is `benchmark` (`Program.cs:196-208`).
- #568 rejected `--gpu` for the five models that do not dispatch on it (`Program.cs:229-233`).
- #570 (this branch's base) wired `--seed` into the student/baseline init *inside* `RunDistill`, so
  the flag is live within `qwen distill` — which makes the silent no-op elsewhere more misleading,
  not less.

`--force` is in exactly the same position as the other two: parsed unconditionally, consumed only by
`Qwen.RunDistill` at the same call site, and silently ignored for any other model.

`--plain` / `--no-kv-cache` / `--text` (Qwen only) and `--simd-widen` (SmolLM) are further inert
flags on this surface; they are **out of scope** for #571 and captured as a follow-up (see the
issues log).

## Proposed changes

All code changes are in `samples/NivaraInference/Program.cs`; plus a `CHANGELOG.md` entry.

### 1. Track presence of the three flags

Declarations (near line 42):

```csharp
int teacherExamples = 0;
int seed = 42;
bool teacherExamplesSet = false;
bool seedSet = false;
bool forceSet = false;
```

Arg loop — `--force` body grows from one statement to two, so it gains braces (style guide):
the `--force` branch (line 85-86) becomes

```csharp
else if (args[i] == "--force")
{
    force = true;
    forceSet = true;
}
```

and the `--teacher-examples`/`--seed` branch (line 103-104) becomes

```csharp
if (args[i] == "--teacher-examples") { teacherExamples = parsed; teacherExamplesSet = true; }
else { seed = parsed; seedSet = true; }
```

Presence is tracked with dedicated booleans because the default values are indistinguishable from
explicit ones: `teacherExamples` defaults to `0` (same as an explicit `--teacher-examples 0`) and
`seed` defaults to `42`. This is the same reason #474 tracked `seqArg`/`warmupArg`/`itersArg` as
nullable strings.

### 2. Scope guard

Inserted as a pure addition between the `--gpu` guard (ends `Program.cs:233`) and
`string modelDir = ...` (line 235), so it runs before any disk touch — mirroring #474's
"a typo should cost a second rather than a 1.5 GB safetensors read":

```csharp
if ((teacherExamplesSet || seedSet || forceSet) && modelType != "qwen")
{
    Console.Error.WriteLine("--teacher-examples / --seed / --force are options of 'qwen distill' only; no other model takes them.");
    return 1;
}

if ((teacherExamplesSet || seedSet || forceSet) && mode != "distill")
{
    Console.Error.WriteLine($"--teacher-examples / --seed / --force need the 'distill' mode; got '{modelType}{(mode.Length > 0 ? " " + mode : "")}'. Drop the flag or add 'distill'.");
    return 1;
}
```

This mirrors #474's two-branch shape (model first, then mode) and names the whole group of scoped
flags, as #474 names `--seq / --warmup / --iters`.

### 3. Help text

Add one line to the `Qwen options:` block (after line 175), mirroring #568's `--gpu` "Any other
model rejects..." line:

```csharp
Console.WriteLine("  (--teacher-examples / --seed / --force are 'qwen distill' only; any other model or mode rejects them.)");
```

### 4. CHANGELOG

New `### Fixed` bullet above the #568 entry in the `[Unreleased]` section, in the established shape:
names the three flags, the `qwen distill` requirement, that the guard precedes the weight load, and
"Sample CLI only — no `src/Nivara` change".

## Blast radius

- **Sample CLI only:** `samples/NivaraInference/Program.cs` (arg loop, one new guard, help text) and
  `CHANGELOG.md`. No `src/Nivara` change, no library public surface, no ADR.
- **Downstream callers:** none — `Program.Main` is the sample process entry point.
- **Behavior delta:** only previously-inert invocations now exit `1` (`minilm --seed 7`,
  `minilm --force`, `modernbert benchmark --seed 7`, `qwen benchmark --force`, …). Valid
  `qwen distill --seed/--teacher-examples/--force` is unchanged.
- **Tests:** `Nivara.Tests` does not reference the sample project, so no unit test can reach
  `Main`; matches the #474/#479/#568 precedent. Verification is direct invocation with exit-status
  capture.

## Verification

- **Build:** `dotnet build samples/NivaraInference -c Release` (expect 0 warnings).
- **Model-scope reject** (capture `$LASTEXITCODE`; no weights needed — the guard precedes the disk
  read): `minilm --seed 7`, `minilm --force`, `modernbert benchmark --seed 7`, `smollm --seed 7`
  → "qwen distill only" message, exit 1.
- **Mode-scope reject:** `qwen --seed 7` (bare / no mode) and `qwen benchmark --force`
  → "need the 'distill' mode" message, exit 1.
- **No misfire:** `qwen distill --seed 7 --teacher-examples 3 --force` → guard passes, then
  "Model file not found" (Qwen weights are not in the checkout).
- **Precedence unchanged:** `qwen --seed abc` → still the #479 "expects an integer" error.
- **`-h`** unchanged apart from the added line.

## Planned commits

1. `docs: plan #571 qwen-distill option scoping in TODO.md`
2. `fix(samples): reject --teacher-examples / --seed / --force outside qwen distill (#571)`
3. `docs: remove TODO.md — plan executed`

## GitHub issues log

- [ ] TBD — capture the remaining inert flags (`--plain` / `--no-kv-cache` / `--text` /
      `--simd-widen`) as a follow-up to #571 (created during execution).

> As each task executes, any deferred work or concern is filed immediately via
> `gh issue create --repo khurram-uworx/Nivara` and recorded here — not held in memory, because
> compaction during execution can lose it.
