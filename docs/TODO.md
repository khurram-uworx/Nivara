# Plan: reject `--gpu` for models with no GPU path (#568)

Branch: `khurram/568` (off `main` @ `23ceef7e`).

> As each task executes, if you find deferred work or a concern (known limitations, follow-ups,
> refactors) that is outside this plan, create a GitHub issue immediately via
> `gh issue create --repo khurram-uworx/Nivara` and record its number in the **GitHub issues log**
> below. Do not wait until the plan finishes — compaction can lose it.

## Problem

`samples/NivaraInference/Program.cs:111-112` parses `--gpu` unconditionally
(`else if (args[i] == "--gpu") useGpu = true;`). Only five `switch (modelType)` cases read
`useGpu`:

- `minilm` (`Program.cs:311`)
- `distilbert` (`Program.cs:328`)
- `distilbert_sst` (`Program.cs:343`)
- `laya` (`Program.cs:360`)
- `modernbert` (`Program.cs:386`)

`mobilenet_v2`, `resnet18`, `smollm`, and `qwen` have no `useGpu` branch, so `--gpu` is silently
accepted and ignored — the run proceeds on the CPU with no warning and no error:

```
dotnet run --project samples/NivaraInference -c Release -- smollm --gpu
# runs the normal CPU path; --gpu has no effect and is never reported
```

Every other rejected option on this surface fails loudly: the five GPU-model cases enumerate their
supported modes in their error paths, and `--gpu` + `--precision bf16|fp16` is a hard reject
(`Program.cs:263-267`). The top-level help (fixed in #478) lists the models that accept `--gpu`, so
the help is a promise the parser does not keep. #568 is the enforcement half of the same question
#478 fixed the text half of.

## Proposed change

1. **Guard the flag in the parser** — after the `--seq / --warmup / --iters` validation block (after
   the closing `}` at `Program.cs:226`) and *before* `modelDir`/`File.Exists` (line 228), add:

   ```csharp
   if (useGpu && modelType is not ("minilm" or "distilbert" or "distilbert_sst" or "modernbert" or "laya"))
   {
       Console.Error.WriteLine($"--gpu is not supported for '{modelType}' (GPU models: minilm, distilbert, distilbert_sst, modernbert, laya).");
       return 1;
   }
   ```

   - Positive allow-list, so it stays in step with the five `useGpu` dispatch branches (and with the
     help heading fixed in #478). Unknown model + `--gpu` now reports the `--gpu` message rather than
     `Unknown model type` — accepted (see decisions).
   - Placed before the disk touch, matching #479's "a typo should cost a second rather than a 1.5 GB
     read". This also makes the reject verifiable for `qwen`, whose weights are absent from this
     checkout (`samples/data/qwen2.5-0.5b-instruct/` does not exist), instead of it being masked by
     `Model file not found`.

2. **Help line** — add one line to the GPU block (`Program.cs:153-155`) so the help also states the
   negative:

   ```csharp
   Console.WriteLine("                     Any other model rejects --gpu (no GPU path).");
   ```

3. **CHANGELOG** — a `### Fixed` entry under `[Unreleased]`, in the #478/#479 voice, noting the
   guard is an allow-list and exits `1` before the weight file is touched.

## Decisions (confirmed with the human before implementation)

- **Hard reject with exit 1**, not a permissive one-line notice (matches #479's fail-loudly
  precedent and the issue's primary proposal).
- **Positive allow-list**; the change of `Unknown model type` precedence for an unknown model +
  `--gpu` is accepted.
- **Add a help note** for the rejection.

## Blast radius

- **Files touched:** `samples/NivaraInference/Program.cs` (parser guard + one help line),
  `CHANGELOG.md`. No `src/Nivara`, no library API, no public contract.
- **Symbols:** guarded inside `NivaraInference.Program.Main(string[] args)`. `TryParseSeqLengths`
  still receives `useGpu` unchanged. The five `useGpu` dispatch branches are untouched and cannot be
  pre-empted (the guard passes only those five model names).
- **Downstream callers:** none — `NivaraInference` is an `Exe` sample. No project references it
  (`tests/` references `Nivara.Samples` and `NivaraChat` only), so nothing consuming the assembly is
  affected.
- **Existing tests:** none cover this CLI surface; no test project references `NivaraInference`, and
  no test drives `Main`. The change is sample-only and does not touch any tested code path, so no
  test-suite change is planned (consistent with #478/#479).
- **Behavioral risk:** anyone currently invoking `--gpu` on the four non-GPU models gets a hard
  failure (exit `1`) where they previously got a silent CPU run. This is the intent of the issue.

## Verification steps

Build: `dotnet build samples/NivaraInference -c Release`. Capture `$LASTEXITCODE` for each run.

| Command | Expected |
|---|---|
| `smollm --gpu` | exit `1`, new `--gpu is not supported for 'smollm' …` message |
| `mobilenet_v2 --gpu` | exit `1`, new message |
| `resnet18 --gpu` | exit `1`, new message |
| `qwen --gpu` | exit `1`, new message (not `Model file not found`) |
| `smollm --gpu --precision bf16` | exit `1`, **new** message (guard wins over the F32-only reject) |
| `minilm\|distilbert\|distilbert_sst\|modernbert\|laya --gpu --precision bf16` | exit `1` via the **existing** F32-only reject → proves the guard did not misfire, with no GPU device needed |
| `-- -h` | unchanged, exit `1` |

`src/Nivara` is untouched, so no `dotnet test` run is warranted (and the suite is not launched
without explicit confirmation, per AGENTS.md).

## Planned commits

1. `docs: plan #568 --gpu rejection for non-GPU models in TODO.md`
2. `fix(samples): reject --gpu for models with no GPU path (#568)` — `Program.cs` + `CHANGELOG.md`
3. `docs: remove TODO.md — plan executed` (after both G2 reviews clear)

## GitHub issues log

- [ ] #568 — `--gpu` silently ignored for mobilenet_v2/resnet18/smollm/qwen (the issue this plan executes)
