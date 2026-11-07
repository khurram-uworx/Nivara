# Release benchmark evidence

This directory holds the **permanent, release-wise benchmark evidence** for Nivara
releases. It is committed alongside release prep so reviewers and future maintainers
can verify that a table was refreshed on the recorded machine rather than carried
forward.

## Layout

One folder per release version. Never overwrite or repurpose a prior release folder:

```
release/
  v1.5.0/
    nivara-perf-v1.5.0.json
    nivara-incident-v1.5.0.txt
    nivarainference-v1.5.0.txt
  v1.6.0/
    ...
```

## Naming

- Core perf harness JSON: `nivara-perf-vX.Y.Z.json` (from
  `tests/Nivara.PerformanceTests --json ...`).
- Sample/console capture: `<sample>-vX.Y.Z.txt` (and optional `.csv`), one file per
  benchmark command.
- Keep the version in every file name so a path can be cited unambiguously.

## Recording line

Every document table keeps the consolidated machine/recording line format from
`RELEASING.md` Step 4:

```
*Recorded YYYY-MM-DD — <CPU model>, <logical cores> logical processors, .NET <version>, <extra context>.*
```

Machine, CPU/core count, .NET SDK/runtime version, OS, power state (AC/battery), and
relevant driver versions must be stated. Do not recompute across machines.

## Retention and authority

- Committed files under `release/<version>/` are the canonical, permanent artefacts.
- `%TEMP%` files are staging during active prep. They may remain in place while the
  release is in progress and are cleaned up manually afterwards.
- The PR may also attach the same files for review. PR attachments are snapshots and
  can expire; cite the committed `release/<version>/` path as the authoritative
  location.
- Cross-machine comparisons: set Prev to `—` where machines differ (see
  `RELEASING.md` Step 5).

## Regeneration

Regenerate tables from captured JSON with the per-sample helpers, pointing `--json`
at the committed file:

```
python samples/NivaraInference/Python/regen_perf_table.py \
    --json release/vX.Y.Z/nivara-perf-vX.Y.Z.json \
    --readme tests/Nivara.PerformanceTests/README.md
```
