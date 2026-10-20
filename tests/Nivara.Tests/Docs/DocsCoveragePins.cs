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

    /// <summary>
    /// Files read by the widened scan. Pinned deliberately: a drop is reduced coverage.
    /// 696 = 691 + the six new Silk/model files (<c>WeightLayout</c>, <c>BertEncoderGpuTypes</c>,
    /// <c>GpuSeam</c>, <c>SilkLauncher</c>, <c>SilkModernBertSession</c>, <c>SilkLayaSession</c>)
    /// minus the deleted samples <c>IBertEncoderRunner.cs</c>. 698 = 696 +
    /// <c>docs/BLAS-MKL.md</c> and <c>tests/Nivara.PerformanceTests/LayaForwardProbe.cs</c>
    /// added by the #492 reconciliation. 700 = 698 + the two SmolLM Silk files
    /// (<c>SilkSmolLMGpuRunner.cs</c>, <c>SilkSmolLMSession.cs</c>) added by #628. 710 = 709 +
    /// <c>docs/DECISION-MODELS.md</c>, the decision-model ecosystem doc split out of the
    /// inference sample's README and <c>docs/LAYA.md</c>. 711 = 710 + the #670 fused-decode
    /// fixture, <c>tests/Nivara.Tests/Gpu/SilkRmsNormGemvTests.cs</c>. 712 = 711 + the #670
    /// device-free geometry fixture, <c>tests/Nivara.Tests/Gpu/SilkGemvGeometryTests.cs</c>, which
    /// was split out of that one so the bit-identity fixture stays byte-identical while the
    /// geometry it relies on is reshaped. 715 = 714 + the #681 bit-identity fixture,
    /// <c>tests/Nivara.Tests/Gpu/SilkGemmGemvAddTests.cs</c>. 722 = 721 + <c>docs/443-PLAN.md</c>.
    /// The merge of PR #726 added three counted files without moving the pin -
    /// <c>samples/Nivara.Samples/Incident/GpuOffload.cs</c>,
    /// <c>src/Nivara.Extensions/Silk/SilkAnalytics.cs</c>, and
    /// <c>tests/Nivara.Tests/Qwen/SynthesizeTensorsBiasGatingTests.cs</c> - so the merged tree was
    /// actually 725 and this gate was red on <c>main</c>. 729 = 725 + the four files the #443
    /// Phase 2/3 work adds: <c>samples/Nivara.Samples/Incident/GpuProfitability.cs</c>,
    /// <c>tests/Nivara.Tests/Gpu/SilkGroupedHashReduceTests.cs</c>,
    /// <c>tests/Nivara.Tests/Incident/GpuIncidentTests.cs</c>, and
    /// <c>tests/Nivara.Tests/Incident/GpuProfitabilityTests.cs</c>.
    /// 735 = 734 + <c>tests/Nivara.Tests/NivaraTorch/LlamaForCausalLMTests.cs</c>, added by the
    /// LlamaForCausalLM torch parity work on khurram/372 without moving the pin.
    /// 738 = 735 + the three files added by #383: <c>samples/Nivara.Samples/NivaraAIFunctions.cs</c>,
    /// <c>tests/Nivara.Tests/Qwen/QwenToolsNivaraLoopTests.cs</c>, and
    /// <c>tests/Nivara.Tests/Samples/NivaraAIFunctionsTests.cs</c>.
    /// 741 = 738 + the three files added by #325 phase 1:
    /// <c>src/Nivara/Execution/BudgetEnforcement.cs</c>,
    /// <c>src/Nivara/Execution/MemoryBudget.cs</c>, and
    /// <c>tests/Nivara.Tests/Execution/MemoryBudgetTests.cs</c>.
    /// 742 = 741 + <c>tests/Nivara.Tests/AutoDiff/MixedPrecisionMatmulTests.cs</c>, added by #387.
    /// 743 = 742 + <c>tests/Nivara.Tests/AutoDiff/MixedPrecisionTrainingTests.cs</c>, added by #746.
    /// 744 = 743 + <c>tests/Nivara.Tests/AutoDiff/MixedPrecisionJvpTests.cs</c>, added by #747.
    /// </summary>
    internal const int ExpectedFileCount = 744;

    /// <summary><c>File.cs:NN</c> citations discovered across the widened scope. Pinned deliberately.</summary>
    internal const int ExpectedCodeCitationCount = 127;

    /// <summary>
    /// <c>docs/…md</c> references discovered across the widened scope. Pinned deliberately.
    /// Includes tolerated historical mentions, which are still discovered and counted here.
    /// </summary>
    internal const int ExpectedDocumentReferenceCount = 362;
}
