## Release PR checklist

- [ ] Changelog drafted and version bumps applied (RELEASING.md Steps 1-3)
- [ ] Machine configuration recorded (RELEASING.md Step 4)
- [ ] Benchmarks actually run, listed below with outcome:
  - [ ] Core perf harness
  - [ ] NivaraIncident
  - [ ] NivaraInference
  - [ ] NivaraFineTuning
- [ ] Benchmarks that could not be run, with reason
- [ ] Committed benchmark evidence paths under `release/<version>/`:
  - [ ] `release/<version>/nivara-perf-<version>.json`
  - [ ] per-benchmark `release/<version>/*-<version>.txt` (or `.csv`)
- [ ] Optional PR review attachments noted (may expire)
- [ ] Cross-machine `Prev = —` notes recorded where applicable
- [ ] Silk-backend numbers recorded where applicable (RELEASING.md Step 5)
- [ ] CD test results (3200+ tests) green
- [ ] `dotnet build Nivara.slnx` and targeted tests green locally
