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
    internal const int ExpectedDocumentCount = 61;

    /// <summary>
    /// Source citations discovered across the narrowed markdown scope, in both forms: the
    /// </summary>
    internal const int ExpectedCodeCitationCount = 53;

    /// <summary>
    /// <c>docs/…md</c> references discovered across the widened scope. Pinned deliberately.
    /// Includes tolerated historical mentions, which are still discovered and counted here.
    /// </summary>
    /// <remarks>
    /// 2026-11-16 (native SmolLM default): 493 + 6 = 499. Three markdown links were added and each is
    /// discovered twice by <c>DocumentPattern</c> — once for the backticked name or link label and once
    /// for the link target: two <c>SMOLLM.md</c> pointers from <c>docs/BFLOAT16.md</c> (the loader-default
    /// paragraph and the A/B paragraph) and one <c>docs/SMOLLM.md</c> pointer from the SmolLM footnote in
    /// <c>samples/NivaraInference/README.md</c>. The rewritten overview line in <c>docs/SMOLLM.md</c> kept
    /// its existing <c>docs/BFLOAT16.md</c> pair, so it contributes no net change.
    /// 2026-11-16 (#1218): +1 = 500. The new native-loading note in <c>docs/MODERNBERT.md</c> references
    /// <c>docs/plan/LAYA-HONEST-WEIGHTS.md</c> in backticks; that path is excluded from the document
    /// scan but the backticked mention is still discovered by <c>DocumentPattern</c>.
    /// 2026-11-17 (#1229): +2 = 501. New subsection in `docs/LAYA.md` references two test file paths
    /// in backticks, both discovered by `DocumentPattern`.
    /// 2026-11-17 (honest-weights cleanup): +5 = 506. New "Model loading honesty (cross-model)" section in
    /// `docs/BFLOAT16.md` references `docs/SMOLLM.md`, `docs/QWEN.md`, `docs/LAYA.md`,
    /// `docs/MODERNBERT.md` and `docs/LAYA.md` (temperature note), each discovered once by
    /// `DocumentPattern`.
    /// 2026-11-17 (#1293): +1 = 507. New "Verification" section in `docs/ONNX.md` references one test
    /// file path in backticks discovered by `DocumentPattern`.
    /// 2026-11-18 (v1.5.2 review doc): +9 = 516. `docs/RELEASE1.5.2-REVIEW.md` references nine
    /// backticked `.md` paths in its doc-refresh confirmation list.
    /// </remarks>
    internal const int ExpectedDocumentReferenceCount = 516;
}
