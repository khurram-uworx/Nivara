# ADR-005: The Snippet Gate Compiles Each Fence Alone, and Preambles Are Labels Over One Namespace

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

### The Preambles are one namespace, and `preamble:` does nothing

This is the sharpest trap in the harness, and the one most likely to be
"tidied up" by someone who reads the code.

There is exactly **one** preamble file and **one** preamble namespace,
`Nivara.Tests.Docs.Preambles`. `DocSnippetCompiler` emits
`using Nivara.Tests.Docs.Preambles;` into the prologue of **every** block,
unconditionally. It **never reads `PreambleName`.**

So the eleven names in `KnownPreambles` are **labels over subsets of one flat
namespace**, not bundles. A block annotated `preamble: streaming-flux-window`
may use `Employee` and still compile; the annotation constrains nothing.
Two names (`agent-null-tensor`, `agent-tensor-kernel`) supply no types at all —
those blocks take all of their context from `locals:`. And because
`using static ...SnippetReport` is likewise unconditional, the `Report(...)`
call in the streaming example resolves in *every* block.

Making the import conditional would restore the obvious semantics and, in the
same commit, break every annotated snippet in the repository. The directive is
bookkeeping; the namespace is the mechanism.

## Decision

**Each fence is compiled as an independent compilation unit against the full
runtime closure, wrapped by a per-block mode, and annotated only by an adjacent
`<!-- gate -->` comment. Preamble names are documentation, never resolution.**

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

6. **`preamble:` is bookkeeping only.** It is not read by the compiler and never
   resolves anything; the unconditional namespace import is what supplies types.
   Stubs live in `SnippetStubs.cs` and must stay `public`, because the generated
   `Nivara.DocSnippet` assembly holds no `InternalsVisibleTo` grant.
   `KnownPreambles` exists so the *name* is not decorative: it is asserted against
   actual usage in both directions, so a typo and a stale registry entry both
   fail. Treat the name as a comment about intent, and put the type where it
   belongs — in the stub file.

7. **Row-type stubs are unions where a name is reused with different shapes
   across sections.** One namespace cannot be overloaded by document, so the stub
   is the union of the shapes, and the conflict must be written into the stub
   rather than resolved silently. `Employee` is the worked example: two sections
   disagree on `Salary`, so the union carries `double`, and the disagreement is
   recorded in the stub's remarks.

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

**Negative:**

- Editing a snippet is a **two-step** change whenever a fence is split: the
  snippet, and the pin. This is deliberate friction, not overhead.
- Preamble names that read like bundles invite the reader to expect resolution
  that does not exist. Rule 6 exists to contain that expectation; it does not
  remove it.
- The stub namespace is **global**, so adding a stub can clear diagnostics in
  unrelated sections. That coupling is invisible at the call site and is the main
  reason the stub file's remarks are load-bearing.
- `using static ...SnippetReport` means `Report(...)` resolves in every block,
  which is one more piece of global coupling the harness does not intend.

## Amendment process

To change the wrapping contract, the directive set, or the preamble mechanism,
file an issue referencing ADR-005 and update the wrapper tests in
`tests/Nivara.Tests/Docs/DocSnippetCompilerTests.cs` in the same change. Any
amendment must keep the negative control failing
(`NegativeControl_TheGateStillRejectsABrokenSnippet`) — a green run that accepts
a broken snippet means nothing. If an amendment makes the preamble import
conditional, it must re-verify every gated block in the same change, because rule
6's consequence is that no test currently covers the coupling.
