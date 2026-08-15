using Nivara.Operations;
using Nivara.Tensors;
using NUnit.Framework;

namespace Nivara.Tests.Tensors;

/// <summary>
/// Tests for <see cref="PartitionedWindowEngine"/>'s partition → sort → compute → scatter pipeline,
/// focused on the index-map scatter refactor (issue #251): results must land on the original rows
/// after per-partition stable sorting, including the pooled scratch path for large partitions.
/// </summary>
[TestFixture]
public class PartitionedWindowEngineTests
{
    static IReadOnlyDictionary<string, IColumn> Columns(params (string Name, IColumn Column)[] columns)
        => columns.ToDictionary(c => c.Name, c => c.Column, StringComparer.OrdinalIgnoreCase);

    static Func<IColumn, IColumn> RollingSum(int windowSize, int? minPeriods = null)
        => col => ((NivaraColumn<int>)col).RollingSum(windowSize, minPeriods);

    [Test]
    public void Compute_MultiPartitionOrdered_ScattersToOriginalRows()
    {
        var columns = Columns(
            ("g", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "A", "B", "A", "B", "B" })),
            ("t", NivaraColumn<int>.Create(new[] { 3, 1, 2, 2, 1, 3 })),
            ("v", NivaraColumn<int>.Create(new[] { 30, 10, 20, 20, 10, 30 })));
        var spec = new WindowSpec().PartitionBy("g").OrderBy("t");

        var result = (NivaraColumn<int>)PartitionedWindowEngine.Compute(columns, columns["v"], spec, RollingSum(2, 1));

        // Partition A sorted by t: rows 1, 3, 0 → sums 10, 30, 50 → scatter {0:50, 1:10, 3:30}
        // Partition B sorted by t: rows 4, 2, 5 → sums 10, 30, 50 → scatter {2:30, 4:10, 5:50}
        Assert.That(result[0], Is.EqualTo(50));
        Assert.That(result[1], Is.EqualTo(10));
        Assert.That(result[2], Is.EqualTo(30));
        Assert.That(result[3], Is.EqualTo(30));
        Assert.That(result[4], Is.EqualTo(10));
        Assert.That(result[5], Is.EqualTo(50));
    }

    [Test]
    public void Compute_SinglePartitionOrdered_ScattersToOriginalRows()
    {
        var columns = Columns(
            ("t", NivaraColumn<int>.Create(new[] { 5, 4, 3, 2, 1 })),
            ("v", NivaraColumn<int>.Create(new[] { 5, 4, 3, 2, 1 })));
        var spec = new WindowSpec().OrderBy("t");

        var result = (NivaraColumn<int>)PartitionedWindowEngine.Compute(columns, columns["v"], spec, RollingSum(2, 1));

        // Sorted by t ascending: rows 4, 3, 2, 1, 0 → v 1,2,3,4,5 → sums 1,3,5,7,9
        Assert.That(result[0], Is.EqualTo(9));
        Assert.That(result[1], Is.EqualTo(7));
        Assert.That(result[2], Is.EqualTo(5));
        Assert.That(result[3], Is.EqualTo(3));
        Assert.That(result[4], Is.EqualTo(1));
    }

    [Test]
    public void Compute_LargeSinglePartition_AbovePoolThreshold()
    {
        const int n = 2048;
        var values = new int[n];
        for (int i = 0; i < n; i++)
            values[i] = i * 3 % 101;

        var columns = Columns(("v", NivaraColumn<int>.Create(values)));
        var spec = new WindowSpec();

        var result = (NivaraColumn<int>)PartitionedWindowEngine.Compute(columns, columns["v"], spec, RollingSum(1, 1));

        for (int i = 0; i < n; i++)
            Assert.That(result[i], Is.EqualTo(values[i]), $"mismatch at {i}");
    }

    [Test]
    public void Compute_TiedOrderKeys_PreserveRowOrder()
    {
        var columns = Columns(
            ("t", NivaraColumn<int>.Create(new[] { 5, 5, 5, 5 })),
            ("v", NivaraColumn<int>.Create(new[] { 1, 2, 1, 2 })));
        var spec = new WindowSpec().OrderBy("t");

        var result = (NivaraColumn<int>)PartitionedWindowEngine.Compute(columns, columns["v"], spec, RollingSum(2, 1));

        Assert.That(result[0], Is.EqualTo(1));
        Assert.That(result[1], Is.EqualTo(3));
        Assert.That(result[2], Is.EqualTo(3));
        Assert.That(result[3], Is.EqualTo(3));
    }

    [Test]
    public void Compute_SourceNulls_NullHandlerAppliedAndScattered()
    {
        var source = NivaraColumn<int>.CreateFromSpans(new[] { 1, 0, 3, 4 }, new[] { false, true, false, false });
        var columns = Columns(
            ("t", NivaraColumn<int>.Create(new[] { 1, 2, 3, 4 })),
            ("v", source));
        var spec = new WindowSpec().OrderBy("t");

        var result = (NivaraColumn<int>)PartitionedWindowEngine.Compute(
            columns,
            columns["v"],
            spec,
            col => ((NivaraColumn<int>)col).RollingSum(2, 1, () => 0));

        // Null replaced by 0: sums 1, 1, 3, 7 in original (sorted) order.
        Assert.That(result.HasNulls, Is.False);
        Assert.That(result[0], Is.EqualTo(1));
        Assert.That(result[1], Is.EqualTo(1));
        Assert.That(result[2], Is.EqualTo(3));
        Assert.That(result[3], Is.EqualTo(7));
    }

    [Test]
    public void Compute_EmptySpec_ShortCircuitsToDelegate()
    {
        var columns = Columns(("v", NivaraColumn<int>.Create(new[] { 1, 2, 3 })));
        var spec = new WindowSpec();
        bool delegateCalled = false;

        var result = (NivaraColumn<int>)PartitionedWindowEngine.Compute(columns, columns["v"], spec, col =>
        {
            delegateCalled = true;
            return ((NivaraColumn<int>)col).RollingSum(2, 1);
        });

        Assert.That(delegateCalled, Is.True);
        Assert.That(result[2], Is.EqualTo(5));
    }
}
