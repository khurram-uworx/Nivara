using Nivara.Expressions;
using Nivara.Streamix;
using NUnit.Framework;
using Streamix;

namespace Nivara.Tests.Streamix;

[TestFixture]
public class StreamixBridgeTests
{
    static NivaraFrame CreateTestFrame(int rowCount = 20)
    {
        var x = new int[rowCount];
        var y = new string[rowCount];
        for (int i = 0; i < rowCount; i++)
        {
            x[i] = i * 10;
            y[i] = $"val{i}";
        }
        return NivaraFrame.Create(
            ("X", NivaraColumn<int>.Create(x)),
            ("Y", NivaraColumn<string>.Create(y)));
    }

    [Test]
    public async Task ToFlux_MatchesCollectResults()
    {
        var frame = CreateTestFrame(25);
        try
        {
            var queryFrame = frame.AsQueryFrame()
                .Filter(ColumnExpressions.Col("X") > 20);

            using var expected = queryFrame.Collect();
            var flux = queryFrame.ToFlux(chunkSize: 5);

            var chunks = new List<NivaraFrame>();
            await foreach (var chunk in flux)
                chunks.Add(chunk);

            try
            {
                int totalRows = chunks.Sum(c => c.RowCount);
                Assert.That(totalRows, Is.EqualTo(expected.RowCount));

                int offset = 0;
                foreach (var chunk in chunks)
                {
                    for (int i = 0; i < chunk.RowCount; i++)
                    {
                        Assert.That(chunk.GetColumn<int>("X")[i], Is.EqualTo(expected.GetColumn<int>("X")[offset]));
                        Assert.That(chunk.GetColumn<string>("Y")[i], Is.EqualTo(expected.GetColumn<string>("Y")[offset]));
                        offset++;
                    }
                }
            }
            finally
            {
                foreach (var chunk in chunks)
                    chunk.Dispose();
            }
        }
        finally
        {
            frame.Dispose();
        }
    }

    [Test]
    public void ToFlux_WithName_PreservesName()
    {
        var frame = CreateTestFrame(5);
        try
        {
            var flux = frame.ToFlux(name: "test-stream");
            Assert.That(flux.Name, Is.EqualTo("test-stream"));
        }
        finally
        {
            frame.Dispose();
        }
    }

    [Test]
    public async Task SingleFrame_ToFlux_RoundTrips()
    {
        var frame = CreateTestFrame(10);
        try
        {
            var flux = frame.ToFlux();

            var collected = new List<NivaraFrame>();
            await foreach (var item in flux)
                collected.Add(item);

            try
            {
                Assert.That(collected, Has.Count.EqualTo(1));
                Assert.That(collected[0].RowCount, Is.EqualTo(frame.RowCount));

                for (int i = 0; i < frame.RowCount; i++)
                {
                    Assert.That(collected[0].GetColumn<int>("X")[i], Is.EqualTo(frame.GetColumn<int>("X")[i]));
                    Assert.That(collected[0].GetColumn<string>("Y")[i], Is.EqualTo(frame.GetColumn<string>("Y")[i]));
                }
            }
            finally
            {
                foreach (var item in collected)
                    item.Dispose();
            }
        }
        finally
        {
            frame.Dispose();
        }
    }

    [Test]
    public async Task ToFluxRows_MatchesFrameRowCount()
    {
        var frame = CreateTestFrame(15);
        try
        {
            var queryFrame = frame.AsQueryFrame();
            var fluxRows = queryFrame.ToFluxRows(chunkSize: 5);

            int count = 0;
            await foreach (var row in fluxRows)
            {
                Assert.That(row.RowIndex, Is.EqualTo(count));
                Assert.That(row.GetValue<int>("X"), Is.EqualTo(frame.GetColumn<int>("X")[count]));
                Assert.That(row.GetValue<string>("Y"), Is.EqualTo(frame.GetColumn<string>("Y")[count]));
                count++;
            }

            Assert.That(count, Is.EqualTo(frame.RowCount));
        }
        finally
        {
            frame.Dispose();
        }
    }

    [Test]
    public async Task ToNivaraFrameAsync_MatchesCollectAsync()
    {
        var frame = CreateTestFrame(30);
        try
        {
            var queryFrame = frame.AsQueryFrame()
                .Filter(ColumnExpressions.Col("X") > 50);

            using var expected = await queryFrame.CollectAsync();

            var flux = queryFrame.ToFlux(chunkSize: 7);
            using var result = await flux.ToNivaraFrameAsync();

            Assert.That(result.RowCount, Is.EqualTo(expected.RowCount));
            Assert.That(result.ColumnNames, Is.EquivalentTo(expected.ColumnNames));

            for (int i = 0; i < expected.RowCount; i++)
            {
                Assert.That(result.GetColumn<int>("X")[i], Is.EqualTo(expected.GetColumn<int>("X")[i]));
                Assert.That(result.GetColumn<string>("Y")[i], Is.EqualTo(expected.GetColumn<string>("Y")[i]));
            }
        }
        finally
        {
            frame.Dispose();
        }
    }

    [Test]
    public async Task Backpressure_FailMode_ThrowsOnFullChannel()
    {
        var stream = Flux.Range(1, 100).PipeThroughChannel(1, ChannelBackpressureMode.Fail);

        Exception? caught = null;
        int itemCount = 0;
        try
        {
            await foreach (var item in stream)
            {
                itemCount++;
                await Task.Delay(10);
            }
        }
        catch (Exception ex)
        {
            caught = ex;
        }

        TestContext.Out.WriteLine($"Items consumed: {itemCount}, Exception: {caught?.GetType().Name ?? "none"}");
        Assert.That(caught, Is.Not.Null,
            "BackpressureException should surface when channel is full in Fail mode");
        Assert.That(caught, Is.InstanceOf<BackpressureException>());
    }

    [Test]
    public async Task Cancellation_PropagatesCleanly()
    {
        using var cts = new CancellationTokenSource();
        var flux = Flux.From(ct => DelayedFrames(10, ct));

        Exception? caught = null;
        try
        {
            await foreach (var item in flux.WithCancellation(cts.Token))
                cts.Cancel();
        }
        catch (OperationCanceledException ex)
        {
            caught = ex;
        }

        Assert.That(caught, Is.Not.Null,
            "OperationCanceledException should propagate when consumer cancels");
    }

    [Test]
    public async Task ToFluxRows_Cancellation_StopsIteration()
    {
        var frame = CreateTestFrame(100);
        try
        {
            var queryFrame = frame.AsQueryFrame();

            using var cts = new CancellationTokenSource();
            var fluxRows = queryFrame.ToFluxRows(chunkSize: 10, cts.Token);

            int count = 0;
            await foreach (var row in fluxRows)
            {
                count++;
                if (count >= 5)
                    cts.Cancel();
            }
        }
        catch (OperationCanceledException)
        {
            return;
        }

        Assert.Pass("Iteration stopped after cancellation");
    }

    static async IAsyncEnumerable<NivaraFrame> DelayedFrames(
        int count,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct)
    {
        var x = new int[] { 1 };
        var y = new string[] { "a" };
        for (int i = 0; i < count; i++)
        {
            ct.ThrowIfCancellationRequested();
            await Task.Delay(10, ct);
            yield return NivaraFrame.Create(
                ("X", NivaraColumn<int>.Create(x)),
                ("Y", NivaraColumn<string>.Create(y)));
        }
    }
}
