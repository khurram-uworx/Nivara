# Decision models — the TypeSafe ecosystem and the `typesafe` demo

A **decision model** is not a text-completion model. The caller declares its questions up front —
a choice, a yes/no, a score — and the server returns a typed answer with a confidence for each,
rather than free-form text. The wire protocol for that is the **TypeSafe "System One"** API: one
`POST /v1/systemone` contract that a decision model sits behind the way OpenAI Chat Completions
sits behind text models.

[docs/LAYA.md](LAYA.md) describes the decision head itself — a typed head on ModernBERT-large. This
page is the ecosystem around it: which engines serve decision models, how the `typesafe` demo in
[`samples/NivaraInference`](../samples/NivaraInference/README.md) talks to them, and how to compare
one engine against another.

## Engines

Two local engines speak the same route today:

| Engine | Default URL | Default model | What else it holds |
|--------|-------------|---------------|--------------------|
| **Ollaya** | `http://localhost:11435` | `laya:en` | one model |
| **Ollama 0.35+** | `http://localhost:11434` | `tev1:0.8b` | `nimble` (9B), `tev1` (4B) — see the [announcement](https://ollama.com/blog/ollama-now-supports-jev-style-decision-models) |
| **Nivara (in-process)** | in-process (Silk F32) | `convaiinnovations/laya` | no HTTP round trip; one engine forward per question; forwards serialised per request |

Because the route is identical, the two are interchangeable from a client's point of view: only
the base URL and the model name change. That is the shape OpenAI's Chat Completions API gave text
models — one request/response contract that many servers implement, so a decision model becomes a
drop-in behind a stable endpoint. It is roughly what "OpenAI Chat Completions" is becoming for
decision models.

**Not every model an engine holds is a decision model.** Ollama in particular serves many models
that answer no `/v1/systemone` question set; asking one of them is not a client error but an
engine-side `400`, e.g. `beam is not a decision model`. The demo prints that response rather than
swallowing it, and exits non-zero when every assessment failed. The engine is therefore the
authority on what is a decision model — which is why `--model` below is a general override rather
than a checked list on the client.

## The demo

`typesafe` loads no weights; it is the HTTP client. The command line is:

```
NivaraInference typesafe [engine] [mode] [--url <url>] [--model <name>]
```

`engine` may now also be `nivara` (in-process Silk F32, default model `convaiinnovations/laya`); `mode` may now also be `verify`.

```bash
dotnet run --project samples/NivaraInference -c Release -- typesafe                            # Ollaya, laya:en
dotnet run --project samples/NivaraInference -c Release -- typesafe ollama                     # Ollama, tev1:0.8b
dotnet run --project samples/NivaraInference -c Release -- typesafe ollama --model tev1:0.8b   # model made explicit
dotnet run --project samples/NivaraInference -c Release -- typesafe --url http://host:11434 --model tev1:0.8b
dotnet run --project samples/NivaraInference -c Release -- typesafe nivara                     # in-process Silk F32
dotnet run --project samples/NivaraInference -c Release -- typesafe nivara benchmark           # timed in-process assessments
dotnet run --project samples/NivaraInference -c Release -- typesafe nivara verify              # compare in-process vs Ollaya
```

`verify` keeps the typed decision as the hard gate and reports per-item/per-question deltas on
per-option probability, `AnswerConfidence`, and score. Observed maxima on the six news items:
per-option probability 0.0000, score 0.0000, `AnswerConfidence` 0.2507; the probability and
score deltas are the quantities expected to stay within 1e-3.

- **`engine`** is `ollaya` (the default) or `ollama`. It is a positional because engines are a
  table, not a set of flags: a third engine later is one row, not another boolean to type.
- **`mode`** is empty (the news assessment) or `benchmark`. The two positional token sets are
  disjoint, which is what lets one slot hold both.
- **`--url`** takes a *required* absolute http(s) URL. Required rather than optional is a parsing
  decision: an optional value is a non-flag token that competes with the mode, and a mode read as a
  URL rejects a well-formed command.
- **`--model`** names the decision model to load on that engine. Both flags are `typesafe`-only and
  any other model rejects them.

The demo sends six AI/ML news headlines through one typed question set — a category choice, three
yes/no questions and two score questions — and prints the answer and confidence for each.

## Comparing engines: `typesafe benchmark`

```bash
dotnet run --project samples/NivaraInference -c Release -- typesafe benchmark
dotnet run --project samples/NivaraInference -c Release -- typesafe ollama benchmark
```

The protocol is the repo's existing one, the same as `laya benchmark`: **1 warmup + 3 timed passes
per news item, median reported** — the defaults, kept so an invocation that does not ask for more
samples sees no change. `--warmup N` and `--iters N` raise the counts (this scope was opened by
[#676](https://github.com/khurram-uworx/Nivara/issues/676)); `--seq` remains `modernbert
benchmark`-only, since it names a sequence shape rather than a sample count.

**A sample is a whole HTTP round trip** — client serialization + transport + engine inference — so
these numbers are *not* comparable with `laya benchmark`, which times an in-process forward pass
over weights. The CLI prints that caveat in its own output so a reader does not line the tables up.

First numbers, both engines on one machine (Windows, .NET 11, Release build), sequential, one
request at a time — two runs of the same command, back to back:

| Engine | Model | Overall median | items/s |
|--------|-------|----------------|---------|
| Ollaya | `laya:en` | 653.5–862 ms | 1.2–1.5 |
| Ollama | `tev1:0.8b` | 2457.5–2743.5 ms | 0.4 |

The overall median is the median of the six per-item medians — for an even count that is the mean
of the two middle values, which is why the figures carry a `.5`. The ranges are two back-to-back
runs: within a run each item's maximum sits within ~3% of its median, but absolute latency drifts
run to run (Ollaya's overall moved +32% between the two runs, Ollama's +12%), so use these to
compare engines only by re-running both, not as fixed figures. What held is the ordering and the
ratio — Ollama was 3.8× Ollaya in run one and 3.2× in run two.

Per item, run one — Ollaya:

```
  item       median        min        max
     1     668 ms     667 ms     683 ms
     2     613 ms     604 ms     615 ms
     3     719 ms     712 ms     719 ms
     4     630 ms     621 ms     631 ms
     5     655 ms     651 ms     655 ms
     6     652 ms     650 ms     656 ms

Overall: median 653.5 ms per assessment across 6 items, 1.5 items/s sequential at those medians.
```

Per item, run one — Ollama:

```
  item       median        min        max
     1    2267 ms    2267 ms    2278 ms
     2    2361 ms    2272 ms    2365 ms
     3    2483 ms    2476 ms    2501 ms
     4    2432 ms    2408 ms    2451 ms
     5    2910 ms    2482 ms    2937 ms
     6    2730 ms    2722 ms    2789 ms

Overall: median 2457.5 ms per assessment across 6 items, 0.4 items/s sequential at those medians.
```

Run two, same commands with nothing changed in between: Ollaya overall median **862 ms**
(1.2 items/s), Ollama **2743.5 ms** (0.4 items/s). Both numbers moved; the ordering and the
ratio band did not. `items/s` is derived from the per-item medians, so it describes sequential
latency at those medians, not batched capacity — and it is the ratio, not either absolute, that a
reader should carry away.

Every figure above comes from the CLI's own output, so re-running the two commands replaces them
with your machine's numbers in the same shape.


The in-process `nivara` row is not comparable with the HTTP rows above (no transport) nor with
`laya benchmark` (one forward): its sample is a whole `SystemOneAsync` call, six encoder forwards on
the device.

## In-process (Nivara) benchmark

`typesafe nivara benchmark` runs the same 1-warmup + 3-timed-pass protocol, with each sample a whole
`SystemOneAsync` call (six encoder forwards on the device, no transport). It is not comparable with
the HTTP rows above or with `laya benchmark` (one forward); it is here for order of magnitude only.

First numbers (Windows, .NET 11, Release, Intel Graphics OpenCL, sequential):

| Engine | Model | Device | Overall median | items/s |
|--------|-------|--------|----------------|---------|
| Nivara (in-process) | `convaiinnovations/laya` | Intel Graphics (Silk F32) | 1417 ms | 0.7 |

Per item:

```text
  item       median        min        max
     1    1456 ms    1455 ms    1632 ms
     2    1386 ms    1381 ms    1400 ms
     3    1597 ms    1596 ms    1605 ms
     4    1387 ms    1386 ms    1389 ms
     5    1416 ms    1416 ms    1421 ms
     6    1418 ms    1418 ms    1419 ms

Overall: median 1417 ms per assessment across 6 items, 0.7 items/s sequential at those medians.
```

Model load 4881 ms; the first (warmup) request includes kernel compilation.

## See also

- [docs/LAYA.md](LAYA.md) — the decision head, its calibration, and the wheel-pinned comparison.
- [`samples/NivaraInference/README.md`](../samples/NivaraInference/README.md) — the sample's
  full usage, including the other models.
