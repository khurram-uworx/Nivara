# v1.4.0 Release Retrospective

Observations and improvement ideas from the v1.4.0 release process. Not
action items yet — internal discussion first.

## What went well

- RELEASING.md hardening (step 5 warning gate, prerequisites, sub-steps)
  caught the root cause of the "skipped benchmarks" failure mode and
  the consistent `## Release Benchmark` sections in each README worked as
  designed — each had exact commands, we ran them, updated the tables.
- CI caught the flaky backpressure test before we merged, confirming the
  gate works.
- The CD workflow published both packages cleanly in ~2 minutes.

## Bugs / Tech Debt

### 1. Flaky `Backpressure_FailMode_BridgePath_PropagatesException`

The test relies on `Task.Delay(1)` producer vs `Task.Delay(10)` consumer with
channel capacity 1. On faster hardware the producer finishes before the channel
overflows, so the `BackpressureException` never fires. Fails consistently on
the 16-core Ultra 7 255H but passes on CI (ubuntu-latest, likely fewer cores
or different scheduler behavior).

**Fix options:**
- Use a `ManualResetEventSlim` gate on the consumer side so the producer
  definitely overflows the channel.
- Produce enough items (e.g., 10,000) with zero consumer delay and channel
  capacity 1 — the bounded channel will reject writes regardless of timing.
- Increase the item count and remove `Task.Delay` from the consumer entirely.

### 2. NivaraIncident `--benchmark` timeout on 10M records

The CLI runs 5 iterations per analysis. On 10M records the full benchmark
takes >5 minutes. The default shell timeout is 2 minutes; anyone following
the Release Benchmark instructions will hit a timeout without knowing why.

**Options:**
- Add a `--quick` flag (1 iteration instead of 5) for release prep.
- Document the expected runtime in the `## Release Benchmark` section.
- Increase the default shell timeout in the release instructions.

### 3. Python vision benchmarks don't pin thread count

`mobilenet.py` and `resnet18.py` don't call `torch.set_num_threads()`, so
MKL uses all available cores. The Nivara vs PyTorch ratio is meaningless
across machines with different core counts (4P/8T laptop vs 16-core desktop
gives ~2.5× PyTorch speedup just from threading).

**Options:**
- Pin `torch.set_num_threads(Environment.ProcessorCount)` in each script
  and document the dependency.
- Or pin to a fixed value (e.g., 4) and document that the ratio is only
  meaningful on machines with ≥4 cores.

## Release Process

### 4. No explicit build+test gate in step 6

RELEASING.md step 6 (Commit, PR, merge) requires benchmark evidence in the
PR description but doesn't say "run `dotnet test` and confirm green before
merging." We got lucky that CI caught the flaky test. If someone merges a
docs-only PR without CI, they wouldn't know.

**Suggestion:** Add a line to step 6: "The PR must show a green CI check
before merging. Do not merge with `--admin` until CI passes."

### 5. Cross-machine Prev/Current is misleading

We carried forward i5-1135G7 numbers as "Prev" on an Ultra 7 255H. The
Ratio/Δ% columns are meaningless across machines (Nivara improved but
PyTorch improved more due to more cores). This makes the table look like a
regression when it's actually an improvement on both sides.

**Suggestion:** RELEASING.md should say: "If the Prev reading was on a
different machine, set Prev to `—` and note the machine difference. Do not
compute ratios across machines."

---

*Recorded during the v1.4.0 release on 2026-08-21.*
