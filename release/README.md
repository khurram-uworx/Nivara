# Release tooling and artefacts

This folder holds two kinds of content:

1. **Canonical benchmark runner** — `run-benchmarks.cmd <version>`.
2. **Helper scripts** — `wrap-table.py` (hidden-tag wrapping) and the per-sample
   `regen_*_perf_table.py` scripts under `samples/*/Python/`.
3. **Committed evidence** — `release/<version>/` subfolders with captured benchmark
   output (see `RELEASING.md` "Benchmark artefact policy").

## Running all benchmarks

From the repository root:

```powershell
cmd /c release\run-benchmarks.cmd 1.5.1
```

Launch it **in the background with no timeout** (it takes tens of minutes). It writes
captures to `release\<version>\`:

- `nivara-perf-v<version>.json` — core perf harness (canonical evidence).
- `nivaraincident-v<version>.txt`
- `nivarainference-cpu/silk/blas-v<version>.txt`
- `pytorch-*-v<version>.txt`
- `nivarafinetuning-v<version>.txt`
- `status.txt` — written last; its presence means the run completed.

Never run it as a blocking foreground command (tool timeouts will kill it).

## Regenerating README tables

1. Wrap the target table in markers:

```powershell
python release\wrap-table.py --readme tests\Nivara.PerformanceTests\README.md --header "Scenario | Prev"
```

2. Regenerate rows from the captured JSON:

```powershell
python samples\Nivara.PerformanceTests\..\..\samples\NivaraInference\Python\regen_perf_table.py --json release\vX.Y.Z\nivara-perf-vX.Y.Z.json --readme tests\Nivara.PerformanceTests\README.md
```

(Use the matching regen script for NivaraIncident / NivaraFineTuning.)

3. Review the `.bak` diff, update recording/machine lines by hand, then remove the
   `<!--hidden-start-->` / `<!--hidden-end-->` markers before committing.

## Rules

- Never overwrite or repurpose a prior `release/<version>/` folder.
- `release.cmd` at the repo root is commit noise and must never be committed; the
  committed runner here (`run-benchmarks.cmd`) is the canonical entry point.
- Update this README when benchmark commands or artefact names change.