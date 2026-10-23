namespace Nivara.Tests.Docs;

/// <summary>
/// The six pinned documentation coverage counts, consolidated so a single test can assert
/// all of them in one <c>Assert.Multiple</c>. See <see cref="DocsCoverageTests"/>.
/// </summary>
internal static class DocsCoveragePins
{
    /// <summary>Count of citations found repository-wide. Pinned deliberately: see the coverage test.</summary>
    internal const int ExpectedCitationCount = 114;

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
    /// </summary>
    internal const int ExpectedDocumentCount = 53;

    /// <summary><c>File.cs:NN</c> citations discovered across the narrowed markdown scope. Pinned deliberately.</summary>
    internal const int ExpectedCodeCitationCount = 31;

    /// <summary>
    /// <c>docs/…md</c> references discovered across the widened scope. Pinned deliberately.
    /// Includes tolerated historical mentions, which are still discovered and counted here.
    /// </summary>
    internal const int ExpectedDocumentReferenceCount = 226;
}
