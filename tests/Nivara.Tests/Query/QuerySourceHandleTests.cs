using Nivara.Expressions;
using Nivara.Query;
using NUnit.Framework;

namespace Nivara.Tests.Query;

/// <summary>
/// Contract of <c>QuerySourceHandle</c>: one source, one explicit release. There is no
/// abandoned-resource cleanup (see #1168): explicit Dispose/Release is the only release path.
/// </summary>
[TestFixture]
public class QuerySourceHandleTests
{
    [Test]
    public void Release_CalledTwice_DisposesSourceOnceAndDoesNotThrow()
    {
        var source = new RecordingLazySource();
        var handle = new QuerySourceHandle(source);

        handle.Release();
        handle.Release();

        Assert.Multiple(() =>
        {
            Assert.That(handle.Released, Is.True);
            Assert.That(source.DisposeCount, Is.EqualTo(1));
        });
    }

    [Test]
    public void DerivedFrame_Abandoned_DoesNotReleaseSourceWhileSiblingLives()
    {
        // The regression test. Losing an intermediate frame used to dispose the shared source out
        // from under live siblings, because every derived frame was tracked with a cleanup action
        // that disposed that source.
        var source = new RecordingLazySource();
        var leaf = BuildTwoStepLeaf(new QueryFrame(source));

        // There is no abandoned-resource cleanup any more (#1168): dropping the intermediate
        // frame cannot release the shared source, because only the handle owns release.
        Assert.Multiple(() =>
        {
            Assert.That(source.DisposeCount, Is.Zero,
                "an abandoned intermediate frame must not release a source a live sibling still uses");
            Assert.That(leaf.Collect().RowCount, Is.EqualTo(3),
                "the surviving frame must still be readable");
        });
    }

    [Test]
    public void Frame_Disposed_SiblingFrame_ThrowsObjectDisposedException()
    {
        var root = new QueryFrame(new RecordingLazySource());
        var sibling = root.Where(ColumnExpressions.Col("A") > 0);

        root.Dispose();

        Assert.Multiple(() =>
        {
            Assert.Throws<ObjectDisposedException>(() => sibling.Collect());
            Assert.Throws<ObjectDisposedException>(() => sibling.Where(ColumnExpressions.Col("A") > 1));
            Assert.Throws<ObjectDisposedException>(() => _ = sibling.IsLazy);
            Assert.Throws<ObjectDisposedException>(() => _ = sibling.Schema);
        });
    }

    [Test]
    public void Frame_Disposed_SiblingFrame_ReportsSameDisposedState()
    {
        var root = new QueryFrame(new RecordingLazySource());
        var sibling = root.Where(ColumnExpressions.Col("A") > 0);

        Assert.That(sibling.ToString(), Does.Not.Contain("QueryFrame [Disposed]"),
            "precondition: the sibling is not disposed yet");

        root.Dispose();

        Assert.Multiple(() =>
        {
            Assert.That(root.ToString(), Is.EqualTo("QueryFrame [Disposed]"));
            Assert.That(sibling.ToString(), Is.EqualTo("QueryFrame [Disposed]"),
                "ToString reports the released state rather than throwing, and must agree across a chain");
        });
    }

    // ── Disposal-error contract (#1170) ──

    [Test]
    public void Release_SourceDisposeThrows_Propagates()
    {
        var source = new ThrowingSource();
        var handle = new QuerySourceHandle(source);

        Assert.Throws<InvalidOperationException>(() => handle.Release(),
            "Dispose must propagate the source-disposal failure");
        Assert.Multiple(() =>
        {
            Assert.That(handle.Released, Is.True);
            Assert.That(source.DisposeCount, Is.EqualTo(1));
        });
    }

    [Test]
    public async Task ReleaseAsync_SourceDisposeThrows_Propagates()
    {
        // A source implementing only IDisposable takes ReleaseAsync's synchronous fallback branch;
        // the fallback must propagate just as Release does.
        var source = new ThrowingSource();
        var handle = new QuerySourceHandle(source);

        await Assert.ThrowsAsync<InvalidOperationException>(async () => await handle.ReleaseAsync(),
            "DisposeAsync must propagate when the synchronous fallback faults");
        Assert.Multiple(() =>
        {
            Assert.That(handle.Released, Is.True);
            Assert.That(source.DisposeCount, Is.EqualTo(1));
        });
    }

    [Test]
    public async Task ReleaseAsync_AsyncDisposableThrows_Propagates()
    {
        // Exercises the awaited IAsyncDisposable branch. No production IQuerySource implements
        // IAsyncDisposable, so this is the only place that branch is executed.
        var source = new RecordingAsyncSource(throwOnDispose: true);
        var handle = new QuerySourceHandle(source);

        await Assert.ThrowsAsync<InvalidOperationException>(async () => await handle.ReleaseAsync(),
            "DisposeAsync must propagate when the awaited source disposal faults");
        Assert.Multiple(() =>
        {
            Assert.That(handle.Released, Is.True);
            Assert.That(source.DisposeAsyncCount, Is.EqualTo(1));
        });
    }

    [Test]
    public async Task ReleaseAsync_AsyncDisposable_DisposesViaAsyncPath()
    {
        var source = new RecordingAsyncSource(throwOnDispose: false);
        var handle = new QuerySourceHandle(source);

        await handle.ReleaseAsync();

        Assert.Multiple(() =>
        {
            Assert.That(source.DisposeAsyncCount, Is.EqualTo(1));
            Assert.That(source.DisposeCount, Is.Zero,
                "the synchronous fallback must not run when the source is IAsyncDisposable");
        });
    }

    [Test]
    public async Task ReleaseAsync_CalledTwice_DisposesSourceOnceAndDoesNotThrow()
    {
        var source = new RecordingAsyncSource(throwOnDispose: false);
        var handle = new QuerySourceHandle(source);

        await handle.ReleaseAsync();
        await handle.ReleaseAsync();

        Assert.Multiple(() =>
        {
            Assert.That(handle.Released, Is.True);
            Assert.That(source.DisposeAsyncCount, Is.EqualTo(1));
        });
    }

    [Test]
    public void Release_AfterFailedDispose_DoesNotThrowAndDoesNotRetry()
    {
        var source = new ThrowingSource();
        var handle = new QuerySourceHandle(source);

        Assert.Throws<InvalidOperationException>(() => handle.Release());

        Assert.DoesNotThrow(() => handle.Release(),
            "a second release after a failed one is a no-op and must not throw");
        Assert.Multiple(() =>
        {
            Assert.That(handle.Released, Is.True);
            Assert.That(source.DisposeCount, Is.EqualTo(1),
                "the source must be disposed at most once");
        });
    }

    [Test]
    public async Task ReleaseAsync_AfterFailedDispose_DoesNotThrowAndDoesNotRetry()
    {
        var source = new RecordingAsyncSource(throwOnDispose: true);
        var handle = new QuerySourceHandle(source);

        await Assert.ThrowsAsync<InvalidOperationException>(async () => await handle.ReleaseAsync());

        await Assert.DoesNotThrowAsync(async () => await handle.ReleaseAsync(),
            "a second release after a failed one is a no-op and must not throw");
        Assert.Multiple(() =>
        {
            Assert.That(handle.Released, Is.True);
            Assert.That(source.DisposeAsyncCount, Is.EqualTo(1));
        });
    }
    /// Kept in its own method so the intermediate is provably unreachable on return.
    /// </summary>
    static QueryFrame BuildTwoStepLeaf(QueryFrame root)
        => root.Where(ColumnExpressions.Col("A") > 0).Where(ColumnExpressions.Col("A") > -1);

    sealed class RecordingLazySource : IQuerySource
    {
        public Schema Schema { get; } = new(new[] { ("A", typeof(int)) });
        public bool IsLazy => true;
        public int DisposeCount { get; private set; }

        public IReadOnlyDictionary<string, IColumn> Execute()
        {
            ObjectDisposedException.ThrowIf(DisposeCount > 0, this);

            return new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(new[] { 1, 2, 3 }) };
        }

        public void Dispose() => DisposeCount++;
    }

    sealed class ThrowingSource : IQuerySource
    {
        public Schema Schema { get; } = new(new[] { ("A", typeof(int)) });
        public bool IsLazy => true;
        public int DisposeCount { get; private set; }

        public IReadOnlyDictionary<string, IColumn> Execute() =>
            new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(new[] { 1, 2, 3 }) };

        public void Dispose()
        {
            DisposeCount++;
            throw new InvalidOperationException("disposal failed");
        }
    }

    sealed class RecordingAsyncSource : IQuerySource, IAsyncDisposable
    {
        readonly bool throwOnDispose;

        public RecordingAsyncSource(bool throwOnDispose) => this.throwOnDispose = throwOnDispose;

        public Schema Schema { get; } = new(new[] { ("A", typeof(int)) });
        public bool IsLazy => true;
        public int DisposeCount { get; private set; }
        public int DisposeAsyncCount { get; private set; }

        public IReadOnlyDictionary<string, IColumn> Execute() =>
            new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(new[] { 1, 2, 3 }) };

        public void Dispose() => DisposeCount++;

        public ValueTask DisposeAsync()
        {
            DisposeAsyncCount++;
            return throwOnDispose
                ? ValueTask.FromException(new InvalidOperationException("async disposal failed"))
                : ValueTask.CompletedTask;
        }
    }
}
