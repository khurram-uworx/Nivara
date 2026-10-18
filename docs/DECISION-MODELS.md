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

Three local engines speak the same route today:

| Engine | Default URL | Default model | What else it holds |
|--------|-------------|---------------|--------------------|
| **Ollaya** | `http://localhost:11435` | `laya:en` | one model |
| **Ollama 0.35+** | `http://localhost:11434` | `tev1:0.8b` | `nimble` (9B), `tev1` (4B) — see the [announcement](https://ollama.com/blog/ollama-now-supports-jev-style-decision-models) |
| **Nivara (in-process)** | in-process (Silk F32) | `convaiinnovations/laya` | no HTTP round trip; one engine forward per question; forwards serialised per request |

Because the route is identical, the engines are interchangeable from a client's point of view: only
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

```text
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

- **`engine`** is `ollaya` (the default), `ollama`, or `nivara`. It is a positional because engines
  are a table, not a set of flags: a third engine later is one row, not another boolean to type.
- **`mode`** is empty (the news assessment), `benchmark`, or `verify`. The two positional token
  sets are disjoint, which is what lets one slot hold both.
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
dotnet run --project samples/NivaraInference -c Release -- typesafe nivara benchmark
```

The protocol is the repo's existing one: **1 warmup + 3 timed passes per news item, median
reported**, with a whole `/v1/systemone` round trip as the unit of work. The Nivara row has no
transport but is still a `SystemOneAsync` call. These numbers are **not** comparable with
`laya benchmark`, which times one in-process forward pass over weights.

Run one, one machine (Windows, .NET 11, Release build, sequential, one request at a time):

| Engine | Median | Min | Max | items/s | Model load |
|--------|--------|-----|-----|---------|------------|
| Ollaya | 653.5 ms | 667 ms | 683 ms | 1.5 | — |
| Nivara | **1216 ms** | 1216 ms | 1224 ms | 0.8 | 4914 ms |
| Ollama | 2457.5 ms | 2278 ms | 2451 ms | 0.4 | — |

Per news item (median ms, min–max range):

| Item | Ollaya | Nivara | Ollama |
|------|--------|--------|--------|
| 1 | 668 (667–683) | 1219 (1216–1224) | 2267 (2267–2278) |
| 2 | 613 (604–615) | 1184 (1182–1186) | 2361 (2272–2365) |
| 3 | 719 (712–719) | 1368 (1360–1384) | 2483 (2476–2501) |
| 4 | 630 (621–631) | 1273 (1190–1275) | 2432 (2408–2451) |
| 5 | 655 (651–655) | 1211 (1209–1225) | 2910 (2482–2937) |
| 6 | 652 (650–656) | 1213 (1208–1216) | 2730 (2722–2789) |
| **Overall median** | **653.5** | **1216** | **2457.5** |

Nivara is consistently faster than Ollama and slower than Ollaya in this machine's measured run.
Absolute latency drifts run to run (Ollaya's overall moved +32% between two back-to-back runs,
Ollama's +12%), so use these to compare engines only by re-running both. Re-running the three
commands replaces every figure with your machine's numbers in the same shape.

## Correctness check: `typesafe nivara verify`

`typesafe nivara verify` keeps the typed decision as the hard gate and reports per-item/per-question
deltas on per-option probability, `AnswerConfidence`, and score. Current run on the six news items:
per-option probability 0.0000, score 0.0000, `AnswerConfidence` 0.2507; the probability and score
deltas are the quantities expected to stay within 1e-3. The command exits non-zero only on a typed
decision disagreement.

## See also

- [docs/LAYA.md](LAYA.md) — the decision head, its calibration, and the wheel-pinned comparison.
- [`samples/NivaraInference/README.md`](../samples/NivaraInference/README.md) — the sample's
  full usage, including the other models.
