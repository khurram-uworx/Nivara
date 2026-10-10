using Nivara.Expressions;
using Nivara.Helpers;
using Nivara.Query;
using NUnit.Framework;

namespace Nivara.Tests.Query;

/// <summary>
/// Contract of <c>QuerySourceHandle</c>: one source, one explicit release, and abandoned-resource
/// cleanup that fires only when nothing can still reach the source.
/// </summary>
[TestFixture]
public class QuerySourceHandleTests
{
    [SetUp]
    public void SetUp() => NivaraResourceManager.Enable();

    [TearDown]
    public void TearDown() => NivaraResourceManager.Disable();

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
    public void Handle_Abandoned_ReleasesSourceWhenNothingReferencesIt()
    {
        // Guards the closure-capture constraint: NivaraResourceManager holds cleanup actions in a
        // static dictionary, so an action capturing the handle would keep it permanently reachable
        // and this test - and abandoned-resource cleanup generally - would silently never fire.
        var source = CreateAbandonedChain();

        CollectAbandoned();

        Assert.That(source.DisposeCount, Is.GreaterThanOrEqualTo(1),
            "an abandoned lazy source must be released once nothing references it");
    }

    [Test]
    public void DerivedFrame_Added_RegistersNoAdditionalTrackedResource()
    {
        // The deterministic half of the ownership contract: deriving a frame must not register
        // another tracked resource, because the source is owned by the handle and not per frame.
        var source = new RecordingLazySource();
        var root = new QueryFrame(source);

        var trackedAfterRoot = NivaraResourceManager.GetResourceStatistics().TotalTrackedResources;

        var leaf = root.Filter(ColumnExpressions.Col("A") > 0).Filter(ColumnExpressions.Col("A") > 1);

        Assert.Multiple(() =>
        {
            Assert.That(NivaraResourceManager.GetResourceStatistics().TotalTrackedResources,
                Is.EqualTo(trackedAfterRoot),
                "deriving frames must not add tracked resources - only the source handle is tracked");
            Assert.That(source.DisposeCount, Is.Zero);
            Assert.That(leaf, Is.Not.Null);
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

        CollectAbandoned();

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
        var sibling = root.Filter(ColumnExpressions.Col("A") > 0);

        root.Dispose();

        Assert.Multiple(() =>
        {
            Assert.Throws<ObjectDisposedException>(() => sibling.Collect());
            Assert.Throws<ObjectDisposedException>(() => sibling.Filter(ColumnExpressions.Col("A") > 1));
            Assert.Throws<ObjectDisposedException>(() => _ = sibling.IsLazy);
            Assert.Throws<ObjectDisposedException>(() => _ = sibling.Schema);
        });
    }

    [Test]
    public void Frame_Disposed_SiblingFrame_ReportsSameDisposedState()
    {
        var root = new QueryFrame(new RecordingLazySource());
        var sibling = root.Filter(ColumnExpressions.Col("A") > 0);

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

    // ── Disposal-error contract (#508) ──

    [Test]
    public void Release_SourceDisposeThrows_DoesNotPropagate()
    {
        var source = new ThrowingSource();
        var handle = new QuerySourceHandle(source);

        Assert.DoesNotThrow(() => handle.Release(),
            "Dispose must not throw; the source-disposal error is swallowed");
        Assert.Multiple(() =>
        {
            Assert.That(handle.Released, Is.True);
            Assert.That(source.DisposeCount, Is.EqualTo(1));
        });
    }

    [Test]
    public async Task ReleaseAsync_SourceDisposeThrows_DoesNotPropagate()
    {
        // A source implementing only IDisposable takes ReleaseAsync's synchronous fallback branch;
        // the fallback must swallow just as Release does.
        var source = new ThrowingSource();
        var handle = new QuerySourceHandle(source);

        await Assert.DoesNotThrowAsync(async () => await handle.ReleaseAsync(),
            "DisposeAsync must not throw; the synchronous fallback swallows like Release");
        Assert.Multiple(() =>
        {
            Assert.That(handle.Released, Is.True);
            Assert.That(source.DisposeCount, Is.EqualTo(1));
        });
    }

    [Test]
    public async Task ReleaseAsync_AsyncDisposableThrows_DoesNotPropagate()
    {
        // Exercises the awaited IAsyncDisposable branch. No production IQuerySource implements
        // IAsyncDisposable, so this is the only place that branch is executed.
        var source = new RecordingAsyncSource(throwOnDispose: true);
        var handle = new QuerySourceHandle(source);

        await Assert.DoesNotThrowAsync(async () => await handle.ReleaseAsync(),
            "DisposeAsync must not throw when the awaited source disposal faults");
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

    /// <summary>
    /// Builds a two-step chain, discarding the intermediate frame, and returns the leaf.
    /// Kept in its own method so the intermediate is provably unreachable on return.
    /// </summary>
    static QueryFrame BuildTwoStepLeaf(QueryFrame root)
        => root.Filter(ColumnExpressions.Col("A") > 0).Filter(ColumnExpressions.Col("A") > -1);

    static RecordingLazySource CreateAbandonedChain()
    {
        var source = new RecordingLazySource();
        var chain = BuildTwoStepLeaf(new QueryFrame(source));

        Assert.That(chain, Is.Not.Null, "precondition: the chain was built");

        return source;
    }

    static void CollectAbandoned()
    {
        for (var attempt = 0; attempt < 3; attempt++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        GC.Collect();
        NivaraResourceManager.ForceCleanup();
    }

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