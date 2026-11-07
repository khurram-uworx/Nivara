# Plan: release benchmarking Silk (OpenCL) A/B (issue #984)

## Problem
Release benchmarking currently records CPU numbers but not Silk/OpenCL backend numbers
for inference samples. Release docs must require Silk A/B runs in the same session,
same units, AC power, and OpenCL device/driver reporting.

## Proposed changes
1. `RELEASING.md`: make Silk a required, non-skippable release benchmark line; add row
   for NivaraInference Silk and prerequisites.
2. `samples/NivaraInference/README.md`: add required `--silk benchmark` commands for
   minilm, distilbert, distilbert_sst, modernbert, laya, qwen generation/decode paths;
   replace "left as —" note with fill-from-capture instructions.
3. `samples/NivaraChat/README.md` and `samples/NivaraClassifier/README.md`: pin Silk
   benchmark paths as release-prep items.
4. `src/Nivara.Extensions/Silk/SilkRuntime.cs`: expose `DriverVersion` and include
   it in `Describe()` for release recording.
5. Review `DocsCoveragePins` only if citation/reference counts change.

## Blast radius
Docs only except SilkRuntime property; no behavior test impact except optional tests.
Tests: DocsCoverageTests, InferenceCliArgumentTests if commands changed (they are
not).

## Planned commits
1. docs: plan issue 984 in TODO.md
2. docs: make Silk A/B runs a required release benchmark line
3. feat: expose OpenCL driver version in SilkRuntime
4. tests: no code tests expected; doc pins if needed

## GitHub issues log
- (none yet)
