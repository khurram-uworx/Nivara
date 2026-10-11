# ADR-005: The Snippet Gate Compiles Each Fence Alone, and Each Preamble Resolves to Its Own Namespace

**Status:** Accepted
**Date:** 2026-10-04

## Context

`DocumentationSnippetTests` compiles every fenced `csharp` block in four
documents — `docs/AGENT-CODE-EXAMPLES.md`, `docs/STREAMING.md`, `EXAMPLES.md`
and `GETTING-STARTED.md` — and fails on any Roslyn error. Since it landed
(#515), the sweep behind #524-#532 has driven `EXAMPLES.md` and
`GETTING-STARTED.md` to zero diagnostics.

The gate works. The **contract for authoring a block that passes it** does not
exist as a document. It lives in `tests/Nivara.Tests/Docs/`, and the reasoning
behind it lived only in a `docs/TODO.md` that the completing PR deletes on
purpose. The next person editing a fenced block rediscovers all of it by
experiment.

That experiment is expensive here because most rules are not preferences. They
are consequences of `DocSnippetCompiler`'s actual behaviour, and several
contradict what the gate's own code appears to say. A reader who "improves" one
will not find out it was load-bearing until a snippet silently stops being
checked.

### Coverage, stated honestly

Of 249 fenced `csharp` blocks repository-wide, the gate compiles 106. Two more
are explicitly excluded (they need `Streamix.AspNetCore`, which the gate's
references do not include). **The remaining 141 are ungated and therefore
unverified — not known-good.** They are enumerated in `UngatedDocuments` so
none can drift into the skip bucket unnoticed, and
`Coverage_EverySnippetBearingDocumentIsClassified` fails if a snippet-bearing
document is neither gated nor listed. That is the accepted steady state: the
gate makes coverage legible rather than pretending to be total.

### Each preamble resolves to its own namespace

This is the sharpest part of the harness, and the one most likely to be
"tidied up" by someone who reads the code. The section records both the mechanism
as it stands after the #546 amendment and what it was before, because most of the
rules below were written against the "before".

A gate comment may annotate a block with `preamble: <name>`. The name resolves
through one mapping, `DocSnippetExtractor.Preambles`, to exactly **one** namespace
per name, plus any `using static` imports that name needs. A block annotated
`preamble: streaming-flux-window` imports
`Nivara.Tests.Docs.Preambles.StreamingFluxWindow` and sees only that bundle's
types: `Employee` is a `row-types` type and does **not** resolve there. A name that
is not in the mapping is a harness defect, so the compiler throws
`InvalidDataException` instead of emitting nothing and letting the snippet fail as a
cascade of unresolved names.

A preamble that would supply no types is not listed at all: a name that imports an
empty namespace is decoration, not a bundle. Five such names existed
(`agent-null-tensor`, `agent-tensor-kernel`, `streaming-flux-training`,
`streaming-flux-reverse`, `streaming-flux-frame`) and their `preamble:` annotations
were removed — those blocks take all of their context from `locals:`.

The stub types live under `tests/Nivara.Tests/Docs/Preambles/`, **one file per
namespace**, so a type cannot sit in the flat `Nivara.Tests.Docs.Preambles`
namespace where the enclosing-namespace lookup would make it reachable from every
block.

**Before #546** there was exactly one preamble namespace. `DocSnippetCompiler`
emitted its import plus `using static ...SnippetReport` into every block
unconditionally and never read `PreambleName`, so the names were labels over subsets
of one flat namespace and constrained nothing. Making the import conditional was
known to break every annotated snippet in the same commit, which is why it was left
alone until #546 did it deliberately.

## Decision

**Each fence is compiled as an independent compilation unit against the full
runtime closure, wrapped by a per-block mode, and annotated only by an adjacent
`<!-- gate -->` comment. A preamble name resolves to exactly one namespace.**

1. **Each fence compiles alone.** Prose continuity is invisible to the compiler,
   so every name a block uses must be declared in that block or supplied by a
   directive. A reader who copies one block must get something that builds.

2. **Directive syntax.** An HTML comment immediately above the opening fence,
   closing on its own line:

   ```text
   <!-- gate
   mode: File
   locals: NivaraFrame f = NivaraFrame.Create(("A", NivaraColumn<int>.Create([1])));
   -->
   ```

   Blank lines between the comment and the fence are tolerated. One `key: value`
   per line; keys are `mode`, `preamble`, `locals`, `exclude`, `reason`. Any
   other key is read and discarded.

3. **Wrap modes.** `TopLevel` (default) — top-level statements, supports `await`,
   compiles as an executable. `File` — a library: type declarations only, **no
   top-level statements**. `MemberDecl` — members wrapped in a class, modelling a
   snippet meant to be pasted inside a type. `GenericLocal` — statements wrapped
   in an unconstrained-by-usage generic local function, for snippets whose prose
   uses a `T`.

   Only `TopLevel` and `GenericLocal` produce top-level statements, and only those
   two compile as executables. `File` and `MemberDecl` wrap in a type and are
   genuine libraries — required, because top-level statements in a library are
   **CS8805**.

4. **A block cannot both declare a type and use it.** Split it: the declaration in
   a `mode: File` fence, the usage in a `TopLevel` fence. This is #531's remedy
   and the only one. A type declaration before a statement is **CS8803**
   (*top-level statements must precede namespace and type declarations*), and the
   rule applies to *any* type starting before *any* statement — not just the first
   of each, which is the check that let two blocks survive a classifier pass.

5. **`locals:` supplies variables, and it is one line.** It is spliced in before
   the body, so it can only declare locals. A type there is **CS8803**.

6. **`preamble:` resolves to exactly one namespace.** The name is looked up in
   `DocSnippetExtractor.Preambles`, which maps it to a namespace and any
   `using static` imports, and those are emitted for that block alone. A block
   annotated with a preamble sees that preamble's types and none from another; a
   name not in the mapping is a harness defect and `DocSnippetCompiler.Compile`
   throws `InvalidDataException` for it, as does a name whose namespace declares no
   types: `DocSnippetExtractor.TypeNamespaces` is the derived set of namespaces that
   declare at least one type, and the compile path and the reflection gate share it.
   Stubs for a name live in
   `Preambles/<Name>.cs` under `Nivara.Tests.Docs.Preambles.<Name>` and must stay
   `public`, because the generated `Nivara.DocSnippet` assembly holds no
   `InternalsVisibleTo` grant. `KnownPreambles` is the mapping's keys; it is
   asserted against actual usage in both directions, so a typo and a stale registry
   entry both fail, and `PreambleNamespaces_AreDeclaredAndNonEmpty` proves each
   mapped namespace exists, is non-empty, is used by exactly one name, and is not
   the old flat namespace.

7. **Row-type stubs are unions where a name is reused with different shapes
   across sections.** A preamble's one namespace cannot be overloaded by document,
   so the stub is the union of the shapes, and the conflict must be written into the
   stub rather than resolved silently. `Employee` is the worked example: two
   sections disagree on `Salary`, so the union carries `double`, and the
   disagreement is recorded in the stub's remarks.

8. **The masked-defect rule.** A receiver with an error type suppresses member
   lookup, so fixing one error can reveal others behind it. A section is clean
   only when the gate says so — never by arithmetic, and never by a hand-rolled
   classifier. **Corollary: read the members in source before annotating**, because
   a broken receiver hides them from the compiler too. This bit twice on the
   branch that produced this ADR.

9. **Splitting a fence moves the coverage pins.** Update
   `Coverage_IsFullyAccountedForAcrossTheWholeRepository` in the same commit and
   state the arithmetic. A drop in the ungated count is reduced coverage, not a
   neutral event.

10. **Assembly reach.** Snippets see the whole runtime closure — `Nivara`,
    `Nivara.Extensions`, Apache.Arrow, Parquet.Net, NUnit — so a snippet may use
    Extensions APIs. They cannot see `internal`: the generated assembly is
    deliberately named `Nivara.DocSnippet` and given no `InternalsVisibleTo`
    grant, so an internal type fails the gate instead of quietly validating.

11. **`using` hoisting.** Genuine using directives in a block are lifted into the
    generated prologue, because a directive is only legal ahead of every other
    element. `using var` and `using (...)` are **statements** and are not lifted;
    they stay where they stand, which is what keeps their variables in scope. A
    directive that appears after a statement is a document defect and is reported
    as **CS1529** rather than silently fixed.

12. **Adding a gated document.** Add it to `GatedDocuments` or
    `UngatedDocuments`, then update the pins.

### Further consequences of the wrapper, recorded because they are traps

13. **`locals:` + `mode: File` is CS8805, not a working combination.** Locals are
    emitted ahead of the body in the same branch that serves `TopLevel` *and*
    `File`, but `File` compiles as a library, and entry points are valid only in
    executables.

14. **`locals:` + `mode: MemberDecl` is silently dropped.** That branch never
    reads `Locals`. No diagnostic is produced, so the block compiles as if the
    locals had never been written — the one interaction here that fails quietly.

15. **Therefore `locals:` is load-bearing only in `TopLevel` and `GenericLocal`.**

16. **A `TopLevel` block needs at least one executable statement** (**CS8937**).
    A block containing only comments or blank lines does not qualify as an entry
    point.

17. **CS8802 is unreachable** — only one compilation unit may carry top-level
    statements, and the gate gives every block its own unit. Recorded so nobody
    spends time looking for a collision between blocks.

18. **An ADR is not a snippet-bearing document.** A fence in a file under
    `docs/adr/` would enter the ungated count and break the coverage assertion; a
    `File.cs:NN` citation would move the citation pins. Both are cheap to avoid and
    expensive to discover later, so this ADR obeys the rule it states.

### Where this record is linked, and why only there

`AGENTS.md` carries the single ADR index, and this ADR is listed in it. That is the
only place. Three alternatives were considered and rejected:

- **A `docs/adr/README.md` index** — a second list of the same records, kept in sync by
  remembering. That is the failure this record was written after: ADR-004 landed and its
  index line appeared six weeks later.
- **`CONTRIBUTING.md`** — the obvious home by convention, but its own header declares
  the file human-owned, forbids AI tools from modifying it, and routes AI coding
  guidance to `AGENTS.md`. This contract *is* AI coding guidance.
- **`ARCHITECTURE.md`** — that file describes the architecture. A test harness's
  authoring contract is not part of it.

Because a list kept in sync by remembering is not a list, `AdrIndexTests` reads the
`AGENTS.md` section by name and compares it against this folder in both directions,
so an ADR that lands unlisted fails the build instead of being invisible. An index
entry must also carry a sentence saying what the decision constrains: a bare link
does not let a reader decide relevance without opening the file, which is the only
reason the index is worth having.

## Consequences

**Positive:**

- Editing a fenced block becomes reviewable against a stated contract instead of
  rediscovered by experiment.
- The trap rules are recorded with their reasons, so "improving" one is a visible
  contradiction rather than a silent regression.
- The coverage claim stays honest: gated, excluded and ungated counts are pinned
  and sum to a scanned total.
- A preamble name now *is* what it reads like — a bundle of exactly one namespace —
  so the reason to reach for the mapping when a type does not resolve is the first
  thing a reader sees rather than the last.

**Negative:**

- Editing a snippet is a **two-step** change whenever a fence is split: the
  snippet, and the pin. This is deliberate friction, not overhead.
- Adding a preamble name is now a two-part change: the mapping entry and the
  `Preambles/<Name>.cs` file. Forgetting either fails the gate rather than
  silently widening what every block can see, which is the point.
- A stub is reachable only from blocks annotated with its preamble, so a snippet
  that uses a stub type without the annotation fails as **CS0246**. That is the
  intended failure (a missing using is a real authoring mistake), but it reads like
  a documentation defect on first encounter.

## Amendment process

To change the wrapping contract, the directive set, or the preamble mechanism,
file an issue referencing ADR-005 and update the wrapper tests in
`tests/Nivara.Tests/Docs/DocSnippetCompilerTests.cs` in the same change. Any
amendment must keep the negative control failing
(`NegativeControl_TheGateStillRejectsABrokenSnippet`) — a green run that accepts
a broken snippet means nothing. An amendment that changes the preamble mechanism
must keep the mapping in `DocSnippetExtractor.Preambles` the single source of
truth, give each name its own non-empty namespace, and re-verify every gated block
in the same change — `EveryGatedBlock_CompilesWithoutErrors` is that verification.
#546 made the import conditional in exactly this way, and it is the reason rules 6
and 7 and the consequences above read as they do.
