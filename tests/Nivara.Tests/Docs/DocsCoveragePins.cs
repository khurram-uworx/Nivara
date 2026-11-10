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
    /// to the scope means restoring the count with them. #868 adds two: one in
    /// <c>docs/METRICS.md</c> (<c>TrainingLoop.cs::OnEpochEnd</c>) and one in
    /// <c>docs/TENSORS.md</c> (<c>NivaraMetrics.cs::ConfusionMatrixFrame</c>) — 113 + 2 = 115.
    /// Issue #1066 adds one in <c>docs/METRICS.md</c>
    /// (<c>MLNetMetricsCrossCheckTests.cs::ConfusionMatrix_MatchesMLNetEvaluation_Exactly</c>):
    /// 115 + 1 = 116.
    /// </remarks>
    internal const int ExpectedCitationCount = 116;

    /// <summary>
    /// Count of documents carrying at least one citation. Pinned deliberately.
    /// </summary>
    /// <remarks>
    /// The five <c>docs/plan/</c> documents that carried citations left the scope with it. #868 adds
    /// two: <c>docs/METRICS.md</c>, which carries its own citations, and <c>docs/TENSORS.md</c>,
    /// which had none until its new boundary paragraph cited
    /// <c>NivaraMetrics.cs::ConfusionMatrixFrame</c> — 17 + 2 = 19.
    /// </remarks>
    internal const int ExpectedCitedDocumentCount = 19;

    /// <summary>
    /// Documents in scope: the <c>*.md</c> files at the repository root plus the reader-facing
    /// documents under <c>docs/</c>.
    /// </summary>
    /// <remarks>
    /// <c>docs/plan/</c> and <c>docs/plans/</c> are excluded by <c>DocSnippetExtractor.IsScanned</c>.
    /// Eight plan documents are out of scope as a result (six AISTACK/ARROW/POLARS review/roadmap
    /// documents, <c>RAG-LLM-EVAL.md</c> and <c>CRAZY-ROADMAP.md</c> moved into <c>docs/plan/</c>,
    /// plus the new <c>NIVARACHAT-EVALS.md</c>). #868 adds <c>docs/METRICS.md</c>: 56 + 1 = 57.
    /// #1064 adds <c>docs/adr/009-metrics-mergeable-accumulator.md</c>: 57 + 1 = 58.
    /// #1077 adds <c>docs/adr/010-chunk-count-growth-policy-on-concat.md</c>: 58 + 1 = 59.
    /// </remarks>
    internal const int ExpectedDocumentCount = 59;

    /// <summary>
    /// Source citations discovered across the narrowed markdown scope, in both forms: the
    /// </summary>
    /// <remarks>
    /// The widened scope adds every <c>.cs</c> file; #868's two new citations (see
    /// <see cref="ExpectedCitationCount"/>) both land here too: 50 + 2 = 52.
    /// Issue #1066's test citation also lands here: 52 + 1 = 53.
    /// </remarks>
    internal const int ExpectedCodeCitationCount = 53;

    /// <summary>
    /// <c>docs/…md</c> references discovered across the widened scope. Pinned deliberately.
    /// Includes tolerated historical mentions, which are still discovered and counted here.
    /// </summary>
    /// <remarks>
    /// #868 adds twelve markdown references - five in <c>docs/METRICS.md</c>, two in
    /// <c>docs/TENSORS.md</c>, two in <c>README.md</c> (the capability bullet and the documentation
    /// index), two in <c>AGENTS.md</c> (link text and target) - plus one in a source comment
    /// (<c>NivaraMetrics.cs</c> names <c>docs/METRICS.md</c>): 241 + 12 = 253.
    /// Issue #1060 adds one more in <c>tests/Nivara.PerformanceTests/README.md</c> (the new
    /// <c>--metrics</c> mode documents the O(k²) warning in <c>docs/METRICS.md</c>): 253 + 1 = 254.
    /// Issue #1064 adds one from the ADR-009 index line in <c>AGENTS.md</c>: 254 + 1 = 255.
    /// Issue #1068 adds one from the ranked-retrieval boundary note in
    /// <c>samples/NivaraChat/README.md</c> (<c>docs/METRICS.md</c>): 255 + 1 = 256.
    /// Issue #1077 adds one from the ADR-010 index line in <c>AGENTS.md</c>: 256 + 1 = 257.
    /// The OpenBLAS probe adds three: one backticked README reference in
    /// <c>docs/BLAS-MKL.md</c> (the section 6 deviations note), one markdown link to
    /// <c>docs/BLAS-MKL.md</c> and one backticked README reference in
    /// <c>tests/Nivara.PerformanceTests/README.md</c>: 257 + 3 = 260.
    /// </remarks>
    internal const int ExpectedDocumentReferenceCount = 260;
}
