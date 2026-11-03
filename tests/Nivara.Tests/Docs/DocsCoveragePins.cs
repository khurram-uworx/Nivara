namespace Nivara.Tests.Docs;

/// <summary>
/// The six pinned documentation coverage counts, consolidated so a single test can assert
/// all of them in one <c>Assert.Multiple</c>. See <see cref="DocsCoverageTests"/>.
/// </summary>
internal static class DocsCoveragePins
{
    /// <summary>
    /// Count of citations found repository-wide. Pinned deliberately: see the coverage test.
    /// The zero-copy Arrow decision gate (#836) took this from 107 to 108. Four citations were
    /// added: three for the import-flattening range <c>ArrowInterop.cs:1096-1121</c> (roadmap,
    /// review scorecard, review §4.1) and one for the export entry point
    /// <c>ArrowInterop.cs:951</c>. Three were removed: two for the stale
    /// <c>ArrowInterop.cs:973-1057</c> range and one for <c>ArrowInterop.cs:999-1015</c>, both
    /// superseded by the ranges above. The <c>ColumnStorage.cs</c> citations in the scorecard row
    /// were rewritten in place, so they cancel. 107 + 4 - 3 = 108.
    /// ADR-008 (#850) added 3 symbol citations in ADR-008/AGENTS.md and converted 2 shorthand
    /// `:NN` citations in ARROW-REVIEW.md to full symbol form. 108 + 3 + 2 = 113.
    /// </summary>
    internal const int ExpectedCitationCount = 113;

    /// <summary>Count of documents carrying at least one citation. Pinned deliberately.</summary>
    internal const int ExpectedCitedDocumentCount = 19;

    /// <summary>
    /// Documents in scope: the 9 <c>*.md</c> files at the repository root plus the 43 under
    /// <c>docs/</c>. Pinned so a shrinking scope cannot look like a pass. A transient plan document at
    /// <c>docs/TODO.md</c> does <em>not</em> increase this number - excluding it is the point, and it is
    /// what keeps the count stable across a workflow that commits one. The move of <c>SILK.md</c>
    /// from <c>tests/Nivara.GpuProbe/</c> into <c>docs/</c> is what took <c>docs/</c> from 39 to 40:
    /// the gate scans the repository root plus <c>docs/</c>, never <c>tests/</c>, so the probe tree
    /// was not counted before and the document entered scope on arrival. The acceleration-doc
    /// restructure took it from 40 to 41 by extracting the ungated streaming design proposal into
    /// its own <c>docs/STREAMING-PROPOSAL.md</c>. The #492 reconciliation added
    /// <c>docs/BLAS-MKL.md</c>, taking <c>docs/</c> from 41 to 42. The decision-model work added
    /// <c>docs/DECISION-MODELS.md</c>, taking <c>docs/</c> from 42 to 43 and this count from 51 to 52.
    /// The validity-mask ratification added <c>docs/adr/007-byte-wise-null-mask-fast-path.md</c>,
    /// taking this count from 53 to 54: <c>docs/adr/</c> is scanned recursively, so a new ADR file
    /// enters scope on arrival even though it changes no code.
    /// ADR-008 added `docs/adr/008-symbol-anchored-citations.md`, taking this count from 54 to 55.
    /// </summary>
    internal const int ExpectedDocumentCount = 55;

    /// <summary><c>File.cs:NN</c> citations discovered across the narrowed markdown scope. Pinned deliberately.</summary>
    internal const int ExpectedCodeCitationCount = 27;

    /// <summary>
    /// <c>docs/…md</c> references discovered across the widened scope. Pinned deliberately.
    /// Includes tolerated historical mentions, which are still discovered and counted here.
    /// The validity-mask ratification took this from 226 to 229, in three parts: one for the ADR-007
    /// line in the <c>AGENTS.md</c> index (<c>AGENTS.md</c> is in the widened scope), and two for the
    /// plan documents the new ADR names — the roadmap phase it ratifies and the review whose
    /// scorecard it corrects. 226 + 1 + 2 = 229.
    /// ADR-008 added one reference to the new ADR in the AGENTS.md index. 229 + 1 = 230.
    /// </summary>
    internal const int ExpectedDocumentReferenceCount = 230;
}
