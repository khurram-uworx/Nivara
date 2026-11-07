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
    // +5: EMBEDDINGS.md "Learning the fusion weights" section cites five new sample symbols
    // (RetrievalTrainer.cs::FusionModel, RetrievalTrainer.cs::RetrievalTrainer.Train(),
    // RetrievalTrainingData.cs::CandidatePool.Select(), RetrievalTrainingData.cs::QuerySplit.Create(),
    // RetrievalEvaluation.cs::RetrievalEvaluation.Compare()).
    internal const int ExpectedCitationCount = 148;

    /// <summary>
    /// Count of documents carrying at least one citation. Pinned deliberately.
    /// </summary>
    internal const int ExpectedCitedDocumentCount = 22;

    /// <summary>
    /// Documents in scope: the 9 <c>*.md</c> files at the repository root plus other under
    /// </summary>
    internal const int ExpectedDocumentCount = 61;

    /// <summary>
    /// Source citations discovered across the narrowed markdown scope, in both forms: the
    /// </summary>
    // +5: same five citations as ExpectedCitationCount, all in code-citation form.
    internal const int ExpectedCodeCitationCount = 50;

    /// <summary>
    /// <c>docs/…md</c> references discovered across the widened scope. Pinned deliberately.
    /// Includes tolerated historical mentions, which are still discovered and counted here.
    /// </summary>
// +2: #991 inference release note now cites docs/SILK.md alongside docs/ACCELERATION.md.
    // +14: top-level doc refresh (README, GETTING-STARTED, EXAMPLES, ARCHITECTURE, AGENTS) added
    // markdown links to the promoted docs (EMBEDDINGS, SAFETENSORS, SILK, ONNX, etc.).
    internal const int ExpectedDocumentReferenceCount = 235;
}
