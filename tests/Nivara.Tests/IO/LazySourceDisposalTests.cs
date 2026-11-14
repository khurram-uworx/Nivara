using Nivara.IO;
using NUnit.Framework;

namespace Nivara.Tests.IO;

[TestFixture]
public class LazySourceDisposalTests
{
    [Test]
    public void DisposeReader_Success_DoesNotThrowAndReleasesGate()
    {
        var gateReleased = false;

        Assert.DoesNotThrow(() => LazySourceDisposal.DisposeReader(
            () => ValueTask.CompletedTask,
            () => gateReleased = true));

        Assert.That(gateReleased, Is.True);
    }

    [Test]
    public void DisposeReader_CloseThrows_PropagatesOriginalException()
    {
        var gateReleased = false;

        var ex = Assert.Throws<InvalidOperationException>(() => LazySourceDisposal.DisposeReader(
            () => ValueTask.FromException(new InvalidOperationException("close failed")),
            () => gateReleased = true));

        Assert.That(ex!.Message, Is.EqualTo("close failed"));
        Assert.That(gateReleased, Is.True,
            "the gate must be released even when the close fails");
    }

    [Test]
    public void DisposeReader_GateReleaseThrows_Propagates()
    {
        Assert.Throws<InvalidOperationException>(() => LazySourceDisposal.DisposeReader(
            () => ValueTask.CompletedTask,
            () => throw new InvalidOperationException("gate failed")));
    }

    [Test]
    public void DisposeReader_CloseAndGateBothThrow_PropagatesReleaseGateFailure()
    {
        var ex = Assert.Throws<InvalidOperationException>(() => LazySourceDisposal.DisposeReader(
            () => ValueTask.FromException(new InvalidOperationException("close failed")),
            () => throw new InvalidOperationException("gate failed")));

        Assert.That(ex!.Message, Is.EqualTo("gate failed"));
    }
}
