# ADR-005: record the snippet-gate authoring contract (#538)

## Problem

The contract for writing a fenced ```csharp block that passes the documentation
compile gate lives only in `tests/Nivara.Tests/Docs/*.cs`, and the reasoning
behind it lived only in a `docs/TODO.md` that the completing PR deletes on
purpose. After that, the next person editing a fenced block rediscovers all of
it by experiment — and several rules look arbitrary or actively wrong until you
know why, so the experiment is the only way to learn them.

`CONTRIBUTING.md` does not mention the gate at all, and it cannot: its own
header (lines 5-11) declares the file human-owned, tells AI tools not to modify
it, and routes "AI coding guidance" to `AGENTS.md`. The gate's authoring contract
*is* AI coding guidance, so `AGENTS.md` is the correct home for the pointer.

Three of the rules are not preferences. They are consequences of
`DocSnippetCompiler`'s actual behaviour, and a how-to page would present them as
choices to be made rather than facts to be worked around. A reader who
"improves" one will not discover it was load-bearing until a snippet silently
stops being checked.

The sharpest case: **`preamble:` does nothing.** `DocSnippetCompiler` never
reads `PreambleName`. Row types resolve because `BaseUsings` imports
`Nivara.Tests.Docs.Preambles` into *every* block unconditionally
(`DocSnippetCompiler.cs:65`). The directive is bookkeeping; the namespace is the
mechanism. Someone who tidies up the apparent redundancy by making the import
conditional breaks every annotated snippet at once.

## Grounding (G1)

Grounded against the harness, not from memory. Every claim below was read out of
the source in this session.

| Claim | Evidence |
| --- | --- |
| One preamble file, one namespace | `tests/Nivara.Tests/Docs/Preambles/SnippetStubs.cs` (188 lines, sole file in `Preambles/`) |
| `PreambleName` is never read by the compiler | `DocSnippetCompiler.Compile` reads only `block.Locals` and `block.Mode`; `PreambleName` appears only in the `DocBlock` record and `DocumentationSnippetTests.cs:28` |
| The mechanism is an unconditional import | `DocSnippetCompiler.cs:65` `using Nivara.Tests.Docs.Preambles;`, and `:67` `using static Nivara.Tests.Docs.Preambles.SnippetReport;` |
| 11 preamble names over one flat namespace | `DocSnippet.cs:91-104` (`KnownPreambles`) |
| Names constrain nothing | one namespace, imported wholesale — a `streaming-flux-window` block may use `Employee` and still compile |
| Two names supply no types at all | `agent-null-tensor`, `agent-tensor-kernel`; those blocks take all context from `locals:` |
| `locals:` + `mode: File` is CS8805 | locals emitted by the `else` branch at `:119`, which covers `TopLevel` **and** `File`; `File` compiles as `DynamicallyLinkedLibrary` (`:137-139`) |
| `locals:` + `mode: MemberDecl` is silently dropped | the `MemberDecl` branch (`:107-115`) never reads `block.Locals` |
| Splitting a fence moves the pins | `DocumentationSnippetTests.cs:72-80` |
| Sweep landed | #524-#532 and #541 all CLOSED |

### Two additions the issue's outline omits

The issue lists 12 rules. Reading `DocSnippetCompiler.Compile` surfaced two more
interactions that no current document exercises and no test covers. Both are
recorded in the ADR because they are the same class of trap as rule 6 — they
look fine until they are used:

- **`locals:` + `mode: File` is CS8805**, not a working combination.
- **`locals:` + `mode: MemberDecl` is silently dropped.** No diagnostic, so the
  block compiles as if the locals were never written.

Corollary to rules 5 and 6: **`locals:` is load-bearing only in `TopLevel` and
`GenericLocal`.**

### Pin arithmetic (why 46 becomes 47)

`DocCitationTests.cs:23` pins `ExpectedDocumentCount = 46`, asserted at `:43`
against `DocSnippetExtractor.MarkdownFiles()`. Counted this session: 9 root
`.md` + 37 under `docs/` = 46, ADRs included. ADR-005 makes it 47.

The bump is scope growth, not a workaround: ADR-005 is now scanned. It must land
in the same commit or `EveryCitationInTheDocumentation` goes red for what looks
like a citation defect.

The other two citation pins do **not** move, verified by grep:

- `ExpectedCitationCount = 106` / `ExpectedCitedDocumentCount = 16` — the four
  existing ADRs contain zero `File.cs:NN` citations. ADR-005 must too.
- Snippet pins (106 / 2 / 141 / 249) — the existing ADRs contain zero ```csharp
  fences. ADR-005 must stay prose-only, or it silently enters the ungated count
  and breaks `Coverage_IsFullyAccountedForAcrossTheWholeRepository`.

That constraint is itself part of the contract, so it is stated in the ADR:
**an ADR is not a snippet-bearing document.**

### Blast radius

| Change | Affects | Risk |
| --- | --- | --- |
| `docs/adr/005-*.md` (new) | doc scan scope only | none; prose-only, no citations |
| `AGENTS.md` +1 line | agent context only | none; one line, agent-facing |
| `tests/Nivara.Tests/Docs/AdrIndexTests.cs` (new) | new test fixture | reads repo files at test time, like the existing doc gates |
| `DocCitationTests.cs:23` 46 -> 47 | citation gate only | low; scope genuinely grew |
| `CHANGELOG.md` | release notes | none |

No public API, no `src/` change, no behaviour change. The harness is untouched.

## Proposed changes

### 1. `docs/adr/005-snippet-gate-authoring-contract.md` (new)

Existing ADR shape: `Status` / `Date` / `Context` / `Decision` /
`Consequences` / `Amendment process`.

- **Context** — gated set (4 documents); `EXAMPLES.md` and `GETTING-STARTED.md`
  at zero diagnostics; 141 of 249 blocks ungated and therefore *unverified, not
  known-good*. Then the Preambles reality above, since that is the sharpest
  thing here and the ADR is where it must live.
- **Decision** — the issue's 12 rules, each with the reason it exists, plus the
  two additions and the `locals:` corollary above. Rule 8 keeps the
  masked-defect corollary (read members in source before annotating — it bit
  twice on this branch: `schema.Columns` behind `Csv.InferSchema`,
  `GetColumn(0)` behind `frame`).
- **Consequences** — two-step edit (snippet + pin when a fence splits); the stub
  namespace is global, so a stub addition can clear unrelated sections and that
  coupling is invisible at the call site; `using static SnippetReport` makes
  `Report(...)` resolve in every block.
- **Amendment process** — matching ADR-002/003/004.
- **Answers in-record**: Q1 current behaviour, Rule 6 marked load-bearing with
  the CS-cascade reason; Q2 accepted steady state, pointing at
  `Coverage_IsFullyAccountedForAcrossTheWholeRepository` as what keeps it
  honest; Q3 by the DRY strategy below.

### 2. `AGENTS.md` — one line after ADR-004

What it constrains + **when it applies**, so relevance is decidable from the
line alone. A routing predicate, not a summary.

### 3. `tests/Nivara.Tests/Docs/AdrIndexTests.cs` (new) — the drift gate

Follows `TypeNameUniquenessTests`' idiom, including the part that matters: the
scan lives in its own method so negative controls can drive it, because a gate
that only ever runs over real input can pass because it silently found nothing.

- Scan: basenames in `docs/adr/*.md` vs the `docs/adr/NNN-...` references in
  `AGENTS.md`, both directions.
- `NoAdrIsMissingFromTheAgentsIndex`
- `NegativeControl_TheScanReportsAnAdrMissingFromTheIndex`
- `NegativeControl_TheScanAcceptsAFullyListedSet`
- Remarks state coverage: `docs/adr/` only, and that `AGENTS.md`'s
  `See docs/adr/` pointer line is not a per-ADR reference.

Reuses `DocSnippetExtractor.RepoRoot` / `MarkdownFiles()` rather than re-walking
the tree.

**Why this gate exists.** ADR-004 landed in `660fc229` (2026-08-15) and its
`AGENTS.md` index line only appeared in `5563c412` (2026-09-27) — a six-week lag
with nothing to catch it. The gate makes the omission impossible rather than
merely documented.

### 4. `DocCitationTests.cs:23` — 46 -> 47

Same commit as the ADR. Message meaning unchanged: scope grew, deliberately.

### 5. `CHANGELOG.md` `[Unreleased]`

ADR-004 house style: rationale plus the `docs/adr/005-...` path, noting the gate
that keeps the index honest.

## DRY strategy for ADR references (Q3)

One list, in the file every agent already loads. Enforced, not duplicated.

- `docs/adr/005-*.md` is the source of truth.
- `AGENTS.md` §"Architectural Decisions (ADRs)" is the single index.
- **No** `docs/adr/README.md`. A second index is the duplication this avoids,
  and it would drift for the same reason ADR-004's line did.
- **No** edit to `CONTRIBUTING.md` (human-owned by its own header) or
  `ARCHITECTURE.md` (architecture, not test-harness contract).
- `RELEASING.md:43` already treats `docs/adr/` as living — no change.
- `.agents/skills/plan-issue/SKILL.md:25,90` already tells agents to list
  `docs/adr/` each session — that is the discovery mechanism, no change.

## Verification steps

1. `dotnet build --configuration Release` — clean.
2. `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --nologo
   --filter "FullyQualifiedName~Docs"` — the three doc gates plus the new one.
   Release only: Debug compares unoptimized `Nivara.dll` against ReadyToRun
   tensors and reports false regressions.
3. Confirm `EveryGatedBlock_CompilesWithoutErrors` is green, so the ADR states
   rules the sweep actually exercised.
4. Confirm `AdrIndexTests` fails when the AGENTS.md line is removed (its own
   negative controls cover the scan; the live assertion covers the real pair).

## Planned commits

1. `docs: plan ADR-005 snippet-gate authoring contract in TODO.md`
2. `docs: record the snippet-gate authoring contract as ADR-005`
3. `test: gate the AGENTS.md ADR index against docs/adr/`
4. `docs: bump the doc-citation document count for ADR-005`
5. `docs: record ADR-005 and the ADR-index gate in the changelog`
6. (G2 only, once both reviews clear) `docs: remove TODO.md - plan executed`

> Reminder: as each task executes, if you find deferred work or a concern,
> create a GitHub issue immediately (`gh issue create --repo
> khurram-uworx/Nivara`) and record its number below. Do not rely on memory or
> wait until the plan finishes — compaction during execution can lose it.

## GitHub issues log

- [ ] #NNN — `preamble:` should gate the namespace import (created while
  planning ADR-005; costs a harness change and re-verification of all 108 gated
  blocks)
- [ ] #NNN — `docs/TODO.md` is listed in `ScanExcludedDocuments` but was deleted
  in `da3f3c66`; the gate still passes because the name is simply never matched,
  so the exclusion and its stated rationale are dead (created while planning
  ADR-005)
