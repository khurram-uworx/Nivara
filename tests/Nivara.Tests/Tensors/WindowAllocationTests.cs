using Nivara.Operations;
using Nivara.Tensors;
using NUnit.Framework;

namespace Nivara.Tests.Tensors;

/// <summary>
/// Allocation regression guards for the #251 window/rank/group-by allocation reduction:
/// the pooled-scratch fast paths and the typed (non-boxing) kernels must stay well below the
/// boxed/slow-path baselines. Bounds are calibrated with wide margins from the
/// Nivara.PerformanceTests harness measurements so they fail only on a real regression
/// (e.g. reintroducing per-row boxing or dropping the pooled prefix path).
/// </summary>
[TestFixture]
public class WindowAllocationTests
{
    /// <summary>
    /// Runs <paramref name="op"/> once after a warmup + GC settle and returns the bytes
    /// allocated on the current thread for a single steady-state call.
    /// </summary>
    static long MeasureOnce(Action op)
    {
        for (int i = 0; i < 3; i++)
            op();
        GC.Collect();
        GC.WaitForPendingFinalizers();
        long pre = GC.GetAllocatedBytesForCurrentThread();
        op();
        return GC.GetAllocatedBytesForCurrentThread() - pre;
    }

    /// <summary>
    /// Best-of-N steady-state allocation for <paramref name="op"/> (min shields against GC noise).
    /// </summary>
    static long MeasureBestOf(Action op, int samples = 5)
    {
        long best = long.MaxValue;
        for (int i = 0; i < samples; i++)
            best = Math.Min(best, MeasureOnce(op));
        return best;
    }

    static NivaraColumn<int> CreateInt(int[] values) => NivaraColumn<int>.Create(values);

    static NivaraColumn<int> FillInt(int[] values)
    {
        for (int i = 0; i < values.Length; i++)
            values[i] = i;
        return CreateInt(values);
    }

    [Test]
    public void RollingSum_NullFreeFastPath_AllocatesLessThanNullMaskedSlowPath()
    {
        var data = FillInt(new int[1_000_000]);
        var nulls = NivaraColumn<int>.CreateFromSpans(
            new int[1_000_000],
            BuildMask(1_000_000, i => i % 7 == 0));

        long fastAlloc = MeasureBestOf(() => { _ = data.RollingSum(10); });
        long slowAlloc = MeasureBestOf(() => { _ = nulls.RollingSum(10, nullHandler: () => 0); });

        TestContext.Out.WriteLine($"null-free: {fastAlloc / (1024.0 * 1024.0):F1}MB, nulls: {slowAlloc / (1024.0 * 1024.0):F1}MB");
        Assert.That(fastAlloc, Is.LessThan(slowAlloc),
            "null-free fast path must allocate less than the null-bearing handler path");
    }

    [Test]
    public void RollingSum_NullFreeFastPath_AllocationBound()
    {
        var data = FillInt(new int[1_000_000]);

        long alloc = MeasureBestOf(() => { _ = data.RollingSum(10); });

        TestContext.Out.WriteLine($"null-free 1M int RollingSum: {alloc / (1024.0 * 1024.0):F1}MB");
        Assert.That(alloc, Is.LessThan(16L * 1024 * 1024),
            "null-free RollingSum must stay near the ~10MB result-column cost, not regress to per-element allocation");
    }

    [Test]
    public void RankKernel_RowNumber_NoPerCompareBoxing()
    {
        var columns = new Dictionary<string, IColumn>
        {
            ["v"] = FillInt(new int[100_000]),
        };
        var orderBy = new[] { new SortKey("v", SortDirection.Ascending) };

        long alloc = MeasureBestOf(() =>
            _ = RankKernel.Compute(columns, [], orderBy, RankKind.RowNumber));

        TestContext.Out.WriteLine($"RankKernel RowNumber 100k: {alloc / (1024.0 * 1024.0):F1}MB");
        Assert.That(alloc, Is.LessThan(8L * 1024 * 1024),
            "rank sort must not box per comparison (~47MB baseline); expected ~2.5MB output-column cost");
    }

    [Test]
    public void GroupBy_TypedKeys_AllocationBound()
    {
        var keys = new int[1_000_000];
        for (int i = 0; i < keys.Length; i++)
            keys[i] = i % 1000;
        var columns = new Dictionary<string, IColumn> { ["k"] = CreateInt(keys) };

        long alloc = MeasureBestOf(() =>
            _ = GroupByOperation.CreateGroupsInternal(columns, new[] { "k" }));

        TestContext.Out.WriteLine($"GroupBy 1M rows x 1000 typed keys: {alloc / (1024.0 * 1024.0):F1}MB");
        Assert.That(alloc, Is.LessThan(25L * 1024 * 1024),
            "typed grouping must stay near the ~9MB reps/bucket cost, not box a key per row");
    }

    [Test]
    public void PartitionedWindow_ScatterEngine_AllocationBound()
    {
        var data = new int[1_000_000];
        var groups = new string[1_000_000];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = i;
            groups[i] = (i % 100).ToString();
        }

        var columns = new Dictionary<string, IColumn>
        {
            ["g"] = NivaraColumn<string>.CreateForReferenceType(groups),
            ["v"] = CreateInt(data),
        };
        var spec = new WindowSpec().PartitionBy("g");

        long alloc = MeasureBestOf(() =>
            _ = PartitionedWindowEngine.Compute(
                columns, columns["v"], spec,
                col => ((NivaraColumn<int>)col).RollingSum(10, 1)));

        TestContext.Out.WriteLine($"PartitionedWindow 1M x 100 parts: {alloc / (1024.0 * 1024.0):F1}MB");
        Assert.That(alloc, Is.LessThan(70L * 1024 * 1024),
            "partitioned window must not box during reorder/scatter (~96MB baseline); expected ~40MB");
    }

    static bool[] BuildMask(int length, Func<int, bool> predicate)
    {
        var mask = new bool[length];
        for (int i = 0; i < length; i++)
            mask[i] = predicate(i);
        return mask;
    }
}
