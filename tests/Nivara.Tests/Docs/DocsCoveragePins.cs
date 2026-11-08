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
    /// notes, not reader-facing documentation. Those documents carried 35 citations, which are now
    /// unchecked. That is a deliberate scope decision, not a neutral edit; restoring plan documents
    /// to the scope means restoring the count with them.
    /// </remarks>
    internal const int ExpectedCitationCount = 113;

    /// <summary>
    /// Count of documents carrying at least one citation. Pinned deliberately.
    /// </summary>
    /// <remarks>The five <c>docs/plan/</c> documents that carried citations left the scope with it.</remarks>
    internal const int ExpectedCitedDocumentCount = 17;

    /// <summary>
    /// Documents in scope: the <c>*.md</c> files at the repository root plus the reader-facing
    /// documents under <c>docs/</c>.
    /// </summary>
    /// <remarks>
    /// <c>docs/plan/</c> and <c>docs/plans/</c> are excluded by <c>DocSnippetExtractor.IsScanned</c>.
    /// Eight plan documents are out of scope as a result (six AISTACK/ARROW/POLARS review/roadmap
    /// documents, <c>RAG-LLM-EVAL.md</c> and <c>CRAZY-ROADMAP.md</c> moved into <c>docs/plan/</c>,
    /// plus the new <c>NIVARACHAT-EVALS.md</c>).
    /// </remarks>
    internal const int ExpectedDocumentCount = 55;

    /// <summary>
    /// Source citations discovered across the narrowed markdown scope, in both forms: the
    /// </summary>
    internal const int ExpectedCodeCitationCount = 50;

    /// <summary>
    /// <c>docs/…md</c> references discovered across the widened scope. Pinned deliberately.
    /// Includes tolerated historical mentions, which are still discovered and counted here.
    /// </summary>
    internal const int ExpectedDocumentReferenceCount = 238;
}
