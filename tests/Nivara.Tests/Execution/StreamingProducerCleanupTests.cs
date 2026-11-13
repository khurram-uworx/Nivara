using Nivara.Execution;
using NUnit.Framework;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace Nivara.Tests.Execution;

[TestFixture]
public class StreamingProducerCleanupTests
{
    [Test]
    public async Task Observe_FaultedProducer_ReturnsItsException()
    {
        var producer = Task.FromException(new InvalidOperationException("producer failed"));
        var primary = new Exception("consumer failed");

        var result = await StreamingProducerCleanup.ObserveAsync(producer, primary, CancellationToken.None);

        Assert.That(result, Is.Not.Null);
        Assert.That(result!.Message, Is.EqualTo("producer failed"));
    }

    [Test]
    public async Task Observe_SuccessfulProducer_ReturnsNull()
    {
        var producer = Task.CompletedTask;
        var primary = new Exception("consumer failed");

        var result = await StreamingProducerCleanup.ObserveAsync(producer, primary, CancellationToken.None);

        Assert.That(result, Is.Null);
    }

    [Test]
    public async Task Observe_SameInstance_ReturnsNull()
    {
        var shared = new InvalidOperationException("shared");
        var producer = Task.FromException(shared);

        var result = await StreamingProducerCleanup.ObserveAsync(producer, shared, CancellationToken.None);

        Assert.That(result, Is.Null);
    }

    [Test]
    public async Task Observe_CancelledProducerWithCancelledToken_ReturnsNull()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        var producer = Task.FromCanceled(cts.Token);
        var primary = new OperationCanceledException(cts.Token);

        var result = await StreamingProducerCleanup.ObserveAsync(producer, primary, cts.Token);

        Assert.That(result, Is.Null);
    }

    [Test]
    public async Task Observe_CancelledProducerWithoutCancellation_ReturnsIt()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        var producer = Task.FromCanceled(cts.Token);
        var primary = new Exception("consumer failed");

        var result = await StreamingProducerCleanup.ObserveAsync(producer, primary, CancellationToken.None);

        Assert.That(result, Is.Not.Null);
    }
}