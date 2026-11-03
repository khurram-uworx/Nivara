# ADR-008: Symbol-Anchored Documentation Citations

**Status:** Accepted
**Date:** 2026-11-03

## Context

Documentation cites source locations as `File.cs:NN` (optionally a range). A gate
(`tests/Nivara.Tests/Docs/DocCitations.cs`) verifies each citation resolves to an
existing file and a line in range. That check is necessary but not sufficient: it
cannot tell that a cited line is the *right* line, and line numbers drift silently
when code is edited. Issues #832, #841, and #850 are all instances of the same
failure mode — citations that resolve yet describe the wrong symbol.

Line numbers are also poor anchors for coding agents: they cannot be grepped, and
an agent that lands on a stale number reads the wrong code with no warning.

## Decision

**Citations anchor to a source symbol, not a line number.**

1. **New form: `File.cs::MemberName`.** The member name is a class, struct,
   interface, record, enum, method, constructor, property, field, or event declared
   in that file. A signature is appended only when needed to disambiguate
   overloads: `File.cs::Slice(int, int)`.
2. **The gate resolves symbols with Roslyn.** It parses each cited `.cs` file once
   (cached) into a member index and checks that the named symbol exists, with the
   given signature when supplied. No regex is used against source.
3. **The `:` line-number form is retired** once all citations are migrated. During
   migration both forms are accepted.
4. **No line numbers are carried**, even as a hint. Tools (editor, agent, LSP)
   resolve the symbol to its current span, which is the durability benefit. The
   gate never verified exact lines anyway.

## Consequences

**Positive:**
- Citations survive edits that add or remove lines, because the anchor is the
  symbol, not a position.
- A citation names the symbol it claims, so a wrong-symbol mismatch is visible in
  the text rather than hidden behind a plausible line number.
- Symbols are greppable and LSP-navigable by coding agents.
- Resolution reuses Roslyn (`Microsoft.CodeAnalysis.CSharp`), already a test
  dependency, following the same syntax-walk pattern as CodeMemory's
  `RoslynSymbolExtractor`.

**Negative / limits:**
- Sub-line precision is lost: a citation that pointed at three lines inside a
  method now names the whole method. Navigation lands in the right neighbourhood,
  which is what agents and readers need; exact lines were never verified anyway.
- Migration touches every citation in the repository (see follow-up issues).
- Signature matching is string-based on the normalised parameter-type list, so
  exotic formatting differences must be normalised.

**Revisit when:**
- A citation needs to point at a statement that is not a member (rare); consider
  a line offset relative to the enclosing symbol only then.
