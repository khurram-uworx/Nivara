namespace Nivara.Tests.Docs;

/// <summary>
/// The six pinned documentation coverage counts, consolidated so a single test can assert
/// all of them in one <c>Assert.Multiple</c>. See <see cref="DocsCoverageTests"/>.
/// </summary>
internal static class DocsCoveragePins
{
    /// <summary>
    /// Count of citations found repository-wide. Pinned deliberately: see the coverage test.
    /// +3: <c>docs/EMBEDDINGS.md</c> cites <c>NivaraFrame.RowCosineSimilarity()</c> twice and
    /// <c>NivaraFrame.RowDot()</c> once, so 113 -> 116. Deliberate addition; not a drop.
    /// +5: the new worked-contrast section cites <c>EmbeddingMode.cs::Run()</c>,
    /// <c>RagMode.cs::Run(ModeContext)</c>, <c>RagIndexCache.cs::Save()</c>,
    /// <c>RagIndexCache.cs::ToEmbeddings()</c> and
    /// <c>NivaraEmbeddingExtensionsTests.cs::EndToEnd_GenerateFilterScoreRetrieve()</c>,
    /// so 116 -> 121. Deliberate addition; not a drop.
    /// </summary>
    internal const int ExpectedCitationCount = 121;

    /// <summary>
    /// Count of documents carrying at least one citation. Pinned deliberately.
    /// +1: <c>docs/EMBEDDINGS.md</c> is new and cites three symbols, so 19 -> 20.
    /// </summary>
    internal const int ExpectedCitedDocumentCount = 20;

    /// <summary>
    /// Documents in scope: the 9 <c>*.md</c> files at the repository root plus other under
    /// <c>docs/</c>. Pinned so a shrinking scope cannot look like a pass. A transient plan document at
    /// <c>docs/TODO.md</c> does <em>not</em> increase this number - excluding it is the point, and it is
    /// what keeps the count stable across a workflow that commits one.
    /// +1: <c>docs/EMBEDDINGS.md</c> joins the 9 root files and the 47 already under <c>docs/</c>,
    /// so 56 -> 57.
    /// </summary>
    internal const int ExpectedDocumentCount = 57;

    /// <summary>
    /// Source citations discovered across the narrowed markdown scope, in both forms: the
    /// <c>File.cs:NN</c> line form and the <c>File.cs::Symbol</c> form ADR-008 prefers.
    /// Pinned deliberately.
    /// +3: <c>docs/EMBEDDINGS.md</c>'s three symbol-form citations, so 27 -> 30. They count here
    /// because this pin counts every citation that names a source file, not only the retired
    /// line-number form.
    /// +5: the new worked-contrast section cites five source files, so 30 -> 35.
    /// </summary>
    internal const int ExpectedCodeCitationCount = 35;

    /// <summary>
    /// <c>docs/…md</c> references discovered across the widened scope. Pinned deliberately.
    /// Includes tolerated historical mentions, which are still discovered and counted here.
    /// Re-scoped 2026-11-03 (branch khurram/ai): TENSORS.md's "ONNX export" AI-stack bullet now
    /// points at `docs/plan/AISTACK-ROADMAP.md` (Phase 1 re-scope to export-only) — one new
    /// document-reference occurrence (TENSORS.md did not previously reference that path),
    /// so 230 -> 231. Deliberate addition; not a drop.
    /// 232: +1 for the README Silk requirement citing `docs/SILK.md`.
    /// +3 (#867): `docs/TENSORS.md` now points at `docs/EMBEDDINGS.md`, and that document points
    /// back at `docs/plan/AISTACK-ROADMAP.md` and `docs/TENSORS.md`, so 232 -> 235.
    /// +1 (#867): the CHANGELOG entry for the convention points at `docs/EMBEDDINGS.md`, so 235 -> 236.
    /// +2 (#867): the AGENTS.md implementation-map line names that document as both link text and
    /// link target, matching the Silk bullet's style, so 236 -> 238. Two occurrences, not one.
    /// +2: the NivaraChat README links to docs/EMBEDDINGS.md from the --embed and --rag sections,
    /// so 238 -> 240.
    /// </summary>
    internal const int ExpectedDocumentReferenceCount = 240;
}
