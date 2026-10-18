using Nivara.AI;
using Nivara.Diagnostics;
using NUnit.Framework;

namespace Nivara.Tests.AI;

[TestFixture]
public class InferenceDiagnosticsTests
{
    [Test]
    public void InferenceDiagnostics_WhenDisabled_DoNotRecordOperations()
    {
        InferenceDiagnostics.Enabled = false;
        DiagnosticsTracker.IsEnabled = false;
        DiagnosticsTracker.ClearRecordedOperations();

        var config = new ModernBertConfig
        {
            HiddenSize = 4,
            IntermediateSize = 8,
            NumHiddenLayers = 1,
            NumAttentionHeads = 1,
        };
        var encoder = new ModernBertEncoder<float>(config);
        var ids = new[] { 1, 2 };

        _ = encoder.Forward(ids, ids.Length);

        Assert.That(DiagnosticsTracker.GetRecordedOperations(), Is.Empty);
    }
}