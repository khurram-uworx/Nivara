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

    // ── Property tests: each partition behaves as if computed independently (#252) ──

    static (int[] Values, bool[] Mask) SumOracle(int[] group, int[] order, int[] value, bool[] mask, int windowSize, int minPeriods)
    {
        int n = group.Length;
        var expected = new int[n];
        var expectedMask = new bool[n];

        foreach (var g in group.Distinct())
        {
            var rows = Enumerable.Range(0, n).Where(i => group[i] == g).ToArray();
            var sorted = rows.OrderBy(i => order[i]).ToArray();

            int validInWindow = 0;
            long sum = 0;
            for (int k = 0; k < sorted.Length; k++)
            {
                int i = sorted[k];
                if (k - windowSize >= 0 && !mask[sorted[k - windowSize]])
                {
                    validInWindow--;
                    sum -= value[sorted[k - windowSize]];
                }

                if (!mask[i])
                {
                    validInWindow++;
                    sum += value[i];
                }

                expectedMask[i] = validInWindow < minPeriods;
                expected[i] = validInWindow >= minPeriods ? (int)sum : 0;
            }
        }

        return (expected, expectedMask);
    }

    [Test]
    public void Compute_RandomMultiPartition_RollingSum_MatchesIndependentOracle()
    {
        var random = new Random(51);
        int n = 120;
        int[] group = new int[n];
        int[] order = new int[n];
        int[] value = new int[n];
        bool[] mask = new bool[n];
        for (int i = 0; i < n; i++)
        {
            group[i] = random.Next(4);
            order[i] = random.Next(10);
            value[i] = random.Next(-50, 51);
            mask[i] = random.Next(4) == 0;
        }

        var columns = Columns(
            ("g", NivaraColumn<int>.Create(group)),
            ("t", NivaraColumn<int>.Create(order)),
            ("v", NivaraColumn<int>.CreateFromSpans(value, mask)));
        var spec = new WindowSpec().PartitionBy("g").OrderBy("t");

        for (int window = 1; window <= 4; window++)
        {
            for (int minPeriods = 1; minPeriods <= window; minPeriods++)
            {
                var result = (NivaraColumn<int>)PartitionedWindowEngine.Compute(columns, columns["v"], spec, RollingSum(window, minPeriods));
                var (expectedValues, expectedMask) = SumOracle(group, order, value, mask, window, minPeriods);

                for (int i = 0; i < n; i++)
                {
                    Assert.That(result.IsNull(i), Is.EqualTo(expectedMask[i]),
                        $"mask mismatch at {i} window={window} minPeriods={minPeriods}");
                    if (!expectedMask[i])
                        Assert.That(result[i], Is.EqualTo(expectedValues[i]),
                            $"value mismatch at {i} window={window} minPeriods={minPeriods}");
                }
            }
        }
    }

    [Test]
    public void Compute_RandomMultiPartition_RollingMeanMinMax_MatchesIndependentOracle()
    {
        var random = new Random(52);
        int n = 120;
        int[] group = new int[n];
        int[] order = new int[n];
        int[] value = new int[n];
        bool[] mask = new bool[n];
        for (int i = 0; i < n; i++)
        {
            group[i] = random.Next(4);
            order[i] = random.Next(10);
            value[i] = random.Next(-50, 51);
            mask[i] = random.Next(4) == 0;
        }

        var columns = Columns(
            ("g", NivaraColumn<int>.Create(group)),
            ("t", NivaraColumn<int>.Create(order)),
            ("v", NivaraColumn<int>.CreateFromSpans(value, mask)));
        var spec = new WindowSpec().PartitionBy("g").OrderBy("t");

        const int windowSize = 3;
        const int minPeriods = 2;

        var mean = (NivaraColumn<double>)PartitionedWindowEngine.Compute(
            columns, columns["v"], spec, col => ((NivaraColumn<int>)col).RollingMean(windowSize, minPeriods));
        var min = (NivaraColumn<int>)PartitionedWindowEngine.Compute(
            columns, columns["v"], spec, col => ((NivaraColumn<int>)col).RollingMin(windowSize, minPeriods));
        var max = (NivaraColumn<int>)PartitionedWindowEngine.Compute(
            columns, columns["v"], spec, col => ((NivaraColumn<int>)col).RollingMax(windowSize, minPeriods));

        foreach (var g in group.Distinct())
        {
            var rows = Enumerable.Range(0, n).Where(i => group[i] == g).ToArray();
            var sorted = rows.OrderBy(i => order[i]).ToArray();

            for (int k = 0; k < sorted.Length; k++)
            {
                int lo = Math.Max(0, k - windowSize + 1);
                int validInWindow = 0;
                int windowMin = int.MaxValue;
                int windowMax = int.MinValue;
                long sum = 0;
                for (int j = lo; j <= k; j++)
                {
                    int src = sorted[j];
                    if (!mask[src])
                    {
                        validInWindow++;
                        sum += value[src];
                        windowMin = Math.Min(windowMin, value[src]);
                        windowMax = Math.Max(windowMax, value[src]);
                    }
                }

                int originalRow = sorted[k];
                bool isNull = validInWindow < minPeriods;
                Assert.That(mean.IsNull(originalRow), Is.EqualTo(isNull), $"mean mask mismatch at {originalRow}");
                Assert.That(min.IsNull(originalRow), Is.EqualTo(isNull), $"min mask mismatch at {originalRow}");
                Assert.That(max.IsNull(originalRow), Is.EqualTo(isNull), $"max mask mismatch at {originalRow}");

                if (!isNull)
                {
                    Assert.That(mean[originalRow], Is.EqualTo((double)sum / validInWindow).Within(1e-9), $"mean mismatch at {originalRow}");
                    Assert.That(min[originalRow], Is.EqualTo(windowMin), $"min mismatch at {originalRow}");
                    Assert.That(max[originalRow], Is.EqualTo(windowMax), $"max mismatch at {originalRow}");
                }
            }
        }
    }
}
