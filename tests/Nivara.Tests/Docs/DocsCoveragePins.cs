namespace Nivara.Tests.Docs;

/// <summary>
/// The six pinned documentation coverage counts, consolidated so a single test can assert
/// all of them in one <c>Assert.Multiple</c>. See <see cref="DocsCoverageTests"/>.
/// </summary>
internal static class DocsCoveragePins
{
    /// <summary>
    /// Count of citations found repository-wide. Pinned deliberately: see the coverage test.
    /// </summary>
    /// <remarks>
    /// The scan scope excludes <c>docs/plan/</c> and <c>docs/plans/</c>: plan documents are working
    /// notes, not reader-facing documentation.
    /// </remarks>
    internal const int ExpectedCitationCount = 122;

    /// <summary>
    /// Count of documents carrying at least one citation. Pinned deliberately.
    /// </summary>
    internal const int ExpectedCitedDocumentCount = 20;

    /// <summary>
    /// Documents in scope: the <c>*.md</c> files at the repository root plus the reader-facing
    /// documents under <c>docs/</c>.
    /// </summary>
    internal const int ExpectedDocumentCount = 60;

    /// <summary>
    /// Source citations discovered across the narrowed markdown scope, in both forms: the
    /// </summary>
    internal const int ExpectedCodeCitationCount = 53;

    /// <summary>
    /// <c>docs/…md</c> references discovered across the widened scope. Pinned deliberately.
    /// Includes tolerated historical mentions, which are still discovered and counted here.
    /// </summary>
    internal const int ExpectedDocumentReferenceCount = 493;
}
