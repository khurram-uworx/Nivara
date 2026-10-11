# Plan — #546: make `preamble:` resolve to its own namespace

## Problem

`DocSnippetCompiler.Compile` never reads `DocBlock.PreambleName`. Two lines in
`BaseUsings` (`tests/Nivara.Tests/Docs/DocSnippetCompiler.cs:65,67`) emit

```
using Nivara.Tests.Docs.Preambles;
using static Nivara.Tests.Docs.Preambles.SnippetReport;
```

into the prologue of **every** block, unconditionally. Every stub lives in the one
flat namespace `Nivara.Tests.Docs.Preambles`
(`tests/Nivara.Tests/Docs/Preambles/SnippetStubs.cs`), so the eleven `KnownPreambles`
names are labels that constrain nothing. The contract is a lie, and the mechanism is
one "tidy-up" away from breaking every annotated snippet at once. Recorded in
ADR-005 rule 6.

## Proposed changes

### 1. One source of truth for preamble -> namespace (`DocSnippet.cs`)

Replace the `KnownPreambles` string array with an ordered mapping and derive
`KnownPreambles` from its keys. Name -> namespace + optional static imports, so the
name and the resolved namespace can never disagree.

```csharp
internal sealed record PreambleImport(string Namespace, string[] StaticImports);

internal static readonly IReadOnlyDictionary<string, PreambleImport> Preambles = new Dictionary<string, PreambleImport>(StringComparer.Ordinal)
{
    ["nn-models"]              = new("Nivara.Tests.Docs.Preambles.NnModels", []),
    ["row-types"]              = new("Nivara.Tests.Docs.Preambles.RowTypes", []),
    ["streaming-as-stream"]    = new("Nivara.Tests.Docs.Preambles.StreamingAsStream",
                                     ["Nivara.Tests.Docs.Preambles.StreamingAsStream.SnippetReport"]),
    ["streaming-flux-basic"]   = new("Nivara.Tests.Docs.Preambles.StreamingFluxBasic", []),
    ["streaming-flux-publish"] = new("Nivara.Tests.Docs.Preambles.StreamingFluxPublish", []),
    ["streaming-flux-window"]  = new("Nivara.Tests.Docs.Preambles.StreamingFluxWindow", []),
};

internal static string[] KnownPreambles => [.. Preambles.Keys.Order(StringComparer.Ordinal)];
```

### 2. `DocSnippetCompiler` reads `PreambleName`

- Delete the two unconditional lines from `BaseUsings`.
- After seeding `generated` with `BaseUsings`, emit the block's preamble imports and
  throw on an unknown/empty-resolving name (so a typo fails loudly, not silently).

```csharp
if (block.PreambleName is { Length: > 0 } name)
{
    if (!DocSnippetExtractor.Preambles.TryGetValue(name, out var import))
        throw new InvalidDataException($"{block.Location} unknown preamble '{name}'");

    generated.Add($"using {import.Namespace};");
    generated.AddRange(import.StaticImports.Select(s => $"using static {s};"));
}
```

### 3. Split `SnippetStubs.cs` per preamble

Delete `tests/Nivara.Tests/Docs/Preambles/SnippetStubs.cs`; add:

| file | namespace | types |
|---|---|---|
| `Preambles/RowTypes.cs` | `…Preambles.RowTypes` | `Employee`, `Customer`, `Person`, `Player`, `Contestant`, `User` |
| `Preambles/NnModels.cs` | `…Preambles.NnModels` | `LinearModel`, `FraudNet` |
| `Preambles/StreamingAsStream.cs` | `…Preambles.StreamingAsStream` | `SnippetReport` |
| `Preambles/StreamingFluxBasic.cs` | `…Preambles.StreamingFluxBasic` | `ISink`, `RecordingSink` |
| `Preambles/StreamingFluxWindow.cs` | `…Preambles.StreamingFluxWindow` | `IPager`, `RecordingPager` |
| `Preambles/StreamingFluxPublish.cs` | `…Preambles.StreamingFluxPublish` | `Dashboard`, `Archival`, `LiveUi` |

Types stay `public` (the generated `Nivara.DocSnippet` assembly holds no
`InternalsVisibleTo` grant). Keep the union-remark XML docs on `Employee`/`Person`
and the `LinearModel`/`FraudNet` notes verbatim.

### 4. Drop the five type-less preambles (decision confirmed with human)

`agent-null-tensor`, `agent-tensor-kernel`, `streaming-flux-training`,
`streaming-flux-reverse`, `streaming-flux-frame` supply no types — their blocks take
all context from `locals:`. Remove each name and its `preamble:` annotation:

- `docs/AGENT-CODE-EXAMPLES.md:8,23`
- `docs/STREAMING.md:361,396,409` (396's comment becomes empty -> delete the comment)

`KnownPreambles` shrinks to six names.

### 5. Annotate the blocks that relied on the global import

These six use `row-types` stubs but carry no `preamble:`; they compile today only
because the import is unconditional. Add `preamble: row-types`:

- `GETTING-STARTED.md:508, 1077, 1095` — edit the existing gate comment
- `GETTING-STARTED.md:545, 1016, 1582` — add a new `<!-- gate -->` comment

### 6. Tests

`DocSnippetCompilerTests.cs`:
- Extend the `Block(...)` helper with `string? preamble = null`.
- `Compile_TypeFromADifferentPreamble_Fails` — `preamble: nn-models` body using
  `Employee` must report `CS0246`/`CS0103`.
- `Compile_TypeFromItsOwnPreamble_Compiles` — the same body under `row-types` must be
  clean (paired control: proves the failure mode is caught, not vacuous).
- `Compile_UnknownPreambleName_Fails` — `preamble: not-a-preamble` errors.

`DocumentationSnippetTests.cs`:
- `PreambleNames_ResolveInBothDirections` unchanged (now checks the six kept names).
- Add `PreambleNamespaces_AreDeclaredAndNonEmpty` via reflection: every mapped
  namespace exists, has >= 1 type, is referenced by exactly one preamble name, and the
  old flat `Nivara.Tests.Docs.Preambles` namespace holds no types.
- `NegativeControl_TheGateStillRejectsABrokenSnippet` stays untouched.

### 7. Docs

- `docs/adr/005-snippet-gate-authoring-contract.md`: rule 6, the "Preambles are one
  namespace" section, the related Consequences bullets, the Amendment-process caveat,
  and the title (which currently asserts the opposite).
- `AGENTS.md:170`: replace "`preamble:` is a label, not a resolver".
- `CHANGELOG.md`: entry describing the change and why the import is now load-bearing.

## Blast radius

- **Harness (tests only):** `DocSnippet.cs`, `DocSnippetCompiler.cs`,
  `DocSnippetCompilerTests.cs`, `DocumentationSnippetTests.cs`, and
  `Preambles/*.cs`. No `src/` change, no public API change.
- **Docs:** the four gated documents' gate comments, ADR-005, `AGENTS.md`, `CHANGELOG.md`.
- **Coverage pins unchanged** — no fence is split or added, so 106 gated / 2 excluded /
  141 ungated / 249 total and `GatedStreamixBlocks == 9` stay put.
- **Downstream:** nothing else references `Nivara.Tests.Docs.Preambles` (verified by
  grep) and nothing reads `KnownPreambles` except `PreambleNames_ResolveInBothDirections`.

## Verification steps

1. `dotnet build Nivara.slnx` after each commit.
2. `dotnet test -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` — must show
   `EveryGatedBlock_CompilesWithoutErrors` green across all 106 blocks, the new
   negative controls failing as designed, and the reflection gate passing.
3. Confirm the ADR index gate (`AdrIndexTests`) still passes after the `AGENTS.md` edit.

## Planned commits

1. `docs: plan #546 in TODO.md`
2. `test: resolve preamble names to per-preamble namespaces in the snippet gate (#546)`
   — items 1, 2, 3, 4, 5, 6 (atomic: the gate cannot be green between them)
3. `docs: record preamble resolution in ADR-005, AGENTS.md and CHANGELOG (#546)` — item 7

## GitHub issues log

- (none yet)
