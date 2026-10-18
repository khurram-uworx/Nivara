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

```bash
dotnet run --project samples/NivaraInference -c Release -- typesafe                            # Ollaya, laya:en
dotnet run --project samples/NivaraInference -c Release -- typesafe ollama                     # Ollama, tev1:0.8b
dotnet run --project samples/NivaraInference -c Release -- typesafe ollama --model tev1:0.8b   # model made explicit
dotnet run --project samples/NivaraInference -c Release -- typesafe --url http://host:11434 --model tev1:0.8b
```

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
per news item, median reported**. It deliberately does not take `--warmup` / `--iters` — those are
`modernbert benchmark` options, pinned as such by the CLI's scope guard; extending that scope is
tracked in [#676](https://github.com/khurram-uworx/Nivara/issues/676).

**A sample is a whole HTTP round trip** — client serialization + transport + engine inference — so
these numbers are *not* comparable with `laya benchmark`, which times an in-process forward pass
over weights. The CLI prints that caveat in its own output so a reader does not line the tables up.

First numbers, both engines on one machine (Windows, .NET 11, Release build), sequential, one
request at a time:

| Engine | Model | Overall median | items/s |
|--------|-------|----------------|---------|
| Ollaya | `laya:en` | 688–905 ms | 1.1–1.5 |
| Ollama | `tev1:0.8b` | 2325–2514 ms | 0.4–0.5 |

The ranges are two runs of the same command. Within a run the per-item spread is tight, but
absolute latency drifts run to run — so use these to compare engines only by re-running both, not
as fixed figures. The ordering held in both runs.

Per item, first run — Ollaya:

```
  item       median        min        max
     1     696 ms     676 ms     731 ms
     2     633 ms     628 ms     637 ms
     3     750 ms     742 ms     753 ms
     4     656 ms     652 ms     664 ms
     5     683 ms     682 ms     692 ms
     6     688 ms     687 ms     694 ms

Overall: median 688 ms per assessment across 6 items, 1.5 items/s sequential at those medians.
```

Per item, first run — Ollama:

```
  item       median        min        max
     1    1944 ms    1937 ms    1952 ms
     2    1938 ms    1918 ms    1958 ms
     3    2052 ms    2050 ms    2136 ms
     4    2365 ms    2356 ms    2387 ms
     5    2357 ms    2355 ms    2377 ms
     6    2325 ms    2322 ms    2356 ms

Overall: median 2325 ms per assessment across 6 items, 0.5 items/s sequential at those medians.
```

Second run, same commands, after the command line was reworked (the measurement protocol did not
change): Ollaya overall median **905 ms** (1.1 items/s), Ollama overall median **2514 ms**
(0.4 items/s) — Ollama 2.8× Ollaya, against 3.4× in the first run. That ratio is what to read off
these tables, not the absolutes.

Raw output is also recorded in [PR #675](https://github.com/khurram-uworx/Nivara/pull/675).

## See also

- [docs/LAYA.md](LAYA.md) — the decision head, its calibration, and the wheel-pinned comparison.
- [`samples/NivaraInference/README.md`](../samples/NivaraInference/README.md) — the sample's
  full usage, including the other models.
