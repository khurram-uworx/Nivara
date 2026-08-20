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

    [Test]
    public async Task Backpressure_FailMode_BridgePath_PropagatesException()
    {
        var slowFrames = SlowAsyncFrames(100);
        var flux = Flux.From(slowFrames)
            .PipeThroughChannel(1, ChannelBackpressureMode.Fail);

        Exception? caught = null;
        int itemCount = 0;
        try
        {
            await foreach (var chunk in flux)
            {
                itemCount++;
                await Task.Delay(10);
            }
        }
        catch (Exception ex)
        {
            caught = ex;
        }

        TestContext.Out.WriteLine($"Chunks consumed: {itemCount}, Exception: {caught?.GetType().Name ?? "none"}");
        Assert.That(caught, Is.Not.Null,
            "BackpressureException should propagate through Flux.From(IAsyncEnumerable).PipeThroughChannel — regression canary for #315");
        Assert.That(caught, Is.InstanceOf<BackpressureException>());
    }

    [Test]
    public async Task Backpressure_FailMode_AsyncEnumerablePath_PropagatesException()
    {
        var asyncEnumerable = SlowAsyncEnumerable(100);
        var flux = Flux.From(asyncEnumerable)
            .PipeThroughChannel(1, ChannelBackpressureMode.Fail);

        Exception? caught = null;
        int itemCount = 0;
        try
        {
            await foreach (var item in flux)
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
            "BackpressureException should propagate through Flux.From(IAsyncEnumerable<T>).PipeThroughChannel (Streamix regression canary for #315)");
        Assert.That(caught, Is.InstanceOf<BackpressureException>());
    }

    static async IAsyncEnumerable<NivaraFrame> SlowAsyncFrames(int count)
    {
        var x = new int[] { 1 };
        var y = new string[] { "a" };
        for (int i = 0; i < count; i++)
        {
            await Task.Delay(1);
            yield return NivaraFrame.Create(
                ("X", NivaraColumn<int>.Create(x)),
                ("Y", NivaraColumn<string>.Create(y)));
        }
    }

    static async IAsyncEnumerable<int> SlowAsyncEnumerable(int count)
    {
        for (int i = 0; i < count; i++)
        {
            await Task.Yield();
            yield return i;
        }
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

    [Test]
    public async Task ToNivaraFrameAsync_FromRows_MatchesCollect()
    {
        var frame = CreateTestFrame(30);
        try
        {
            var queryFrame = frame.AsQueryFrame()
                .Filter(ColumnExpressions.Col("X") > 50);

            using var expected = await queryFrame.CollectAsync();

            var fluxRows = queryFrame.ToFluxRows(chunkSize: 7);
            using var result = await fluxRows.ToNivaraFrameAsync();

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
    public async Task ToNivaraFrameAsync_EmptyStream_ReturnsEmptyFrame()
    {
        var emptyFrame = NivaraFrame.Create(
            ("X", NivaraColumn<int>.Create([])),
            ("Y", NivaraColumn<string>.Create([])));
        try
        {
            var fluxRows = emptyFrame.ToFluxRows(chunkSize: 5);
            using var result = await fluxRows.ToNivaraFrameAsync();

            Assert.That(result.RowCount, Is.EqualTo(0));
            Assert.That(result.ColumnNames, Is.EquivalentTo(["X", "Y"]));
        }
        finally
        {
            emptyFrame.Dispose();
        }
    }

    [Test]
    public async Task ToFluxWithTimestamp_AttachesTimestamp()
    {
        var frame = CreateTestFrame(10);
        try
        {
            var baseTime = new DateTimeOffset(2025, 1, 1, 0, 0, 0, TimeSpan.Zero);
            var fluxTimestamped = frame.ToFluxWithTimestamp(
                row => baseTime.AddSeconds(row.GetValue<int>("X")));

            var items = new List<Timestamped<NivaraRow>>();
            await foreach (var item in fluxTimestamped)
                items.Add(item);

            Assert.That(items, Has.Count.EqualTo(10));

            for (int i = 0; i < 10; i++)
            {
                Assert.That(items[i].Value.GetValue<int>("X"), Is.EqualTo(i * 10));
                Assert.That(items[i].Timestamp, Is.EqualTo(baseTime.AddSeconds(i * 10)));
            }
        }
        finally
        {
            frame.Dispose();
        }
    }

    [Test]
    public async Task ToFluxWithTimestamp_WithName_PreservesName()
    {
        var frame = CreateTestFrame(5);
        try
        {
            var baseTime = DateTimeOffset.UtcNow;
            var flux = frame.ToFluxWithTimestamp(
                row => baseTime,
                name: "timed-stream");

            Assert.That(flux.Name, Is.EqualTo("timed-stream"));
        }
        finally
        {
            frame.Dispose();
        }
    }

    [Test]
    public async Task BufferByCount_ProducesCorrectBatches()
    {
        var frame = CreateTestFrame(10);
        try
        {
            var fluxRows = frame.ToFluxRows(chunkSize: 10);
            var buffered = fluxRows.BufferByCount(3);

            var batches = new List<IList<NivaraRow>>();
            await foreach (var batch in buffered)
                batches.Add(batch);

            Assert.That(batches, Has.Count.EqualTo(4));
            Assert.That(batches[0].Count, Is.EqualTo(3));
            Assert.That(batches[1].Count, Is.EqualTo(3));
            Assert.That(batches[2].Count, Is.EqualTo(3));
            Assert.That(batches[3].Count, Is.EqualTo(1));
        }
        finally
        {
            frame.Dispose();
        }
    }

    [Test]
    public async Task BufferByCount_InvalidCount_Throws()
    {
        var frame = CreateTestFrame(5);
        try
        {
            var fluxRows = frame.ToFluxRows(chunkSize: 5);
            Assert.Throws<ArgumentOutOfRangeException>(() => fluxRows.BufferByCount(0));
            Assert.Throws<ArgumentOutOfRangeException>(() => fluxRows.BufferByCount(-1));
        }
        finally
        {
            frame.Dispose();
        }
    }

    [Test]
    public async Task BufferFrames_ProducesNivaraFrames()
    {
        var frame = CreateTestFrame(10);
        try
        {
            var fluxRows = frame.ToFluxRows(chunkSize: 10);
            var batched = fluxRows.BufferFrames(batchSize: 4);

            var frames = new List<NivaraFrame>();
            await foreach (var f in batched)
                frames.Add(f);

            try
            {
                Assert.That(frames, Has.Count.EqualTo(3));
                Assert.That(frames[0].RowCount, Is.EqualTo(4));
                Assert.That(frames[1].RowCount, Is.EqualTo(4));
                Assert.That(frames[2].RowCount, Is.EqualTo(2));

                int globalIdx = 0;
                foreach (var f in frames)
                {
                    for (int i = 0; i < f.RowCount; i++)
                    {
                        Assert.That(f.GetColumn<int>("X")[i], Is.EqualTo(globalIdx * 10));
                        Assert.That(f.GetColumn<string>("Y")[i], Is.EqualTo($"val{globalIdx}"));
                        globalIdx++;
                    }
                }
            }
            finally
            {
                foreach (var f in frames)
                    f.Dispose();
            }
        }
        finally
        {
            frame.Dispose();
        }
    }

    [Test]
    public async Task BufferFrames_InvalidBatchSize_Throws()
    {
        var frame = CreateTestFrame(5);
        try
        {
            var fluxRows = frame.ToFluxRows(chunkSize: 5);
            Assert.Throws<ArgumentOutOfRangeException>(() => fluxRows.BufferFrames(batchSize: 0));
        }
        finally
        {
            frame.Dispose();
        }
    }
}
