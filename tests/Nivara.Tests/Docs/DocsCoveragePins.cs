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
    internal const int ExpectedCitationCount = 113;

    /// <summary>Count of documents carrying at least one citation. Pinned deliberately.</summary>
    internal const int ExpectedCitedDocumentCount = 19;

    /// <summary>
    /// Documents in scope: the 9 <c>*.md</c> files at the repository root plus other under
    /// <c>docs/</c>. Pinned so a shrinking scope cannot look like a pass. A transient plan document at
    /// <c>docs/TODO.md</c> does <em>not</em> increase this number - excluding it is the point, and it is
    /// what keeps the count stable across a workflow that commits one.
    /// </summary>
    internal const int ExpectedDocumentCount = 56;

    /// <summary><c>File.cs:NN</c> citations discovered across the narrowed markdown scope. Pinned deliberately.</summary>
    internal const int ExpectedCodeCitationCount = 27;

    /// <summary>
    /// <c>docs/…md</c> references discovered across the widened scope. Pinned deliberately.
    /// Includes tolerated historical mentions, which are still discovered and counted here.
    /// Re-scoped 2026-11-03 (branch khurram/ai): TENSORS.md's "ONNX export" AI-stack bullet now
    /// points at `docs/plan/AISTACK-ROADMAP.md` (Phase 1 re-scope to export-only) — one new
    /// document-reference occurrence (TENSORS.md did not previously reference that path),
    /// so 230 -> 231. Deliberate addition; not a drop.
    /// </summary>
    internal const int ExpectedDocumentReferenceCount = 232;
}
