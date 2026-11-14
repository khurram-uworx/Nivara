using Nivara.Expressions;
using Nivara.Linq;
using Nivara.Operations;
using Nivara.Query;
using NUnit.Framework;

namespace Nivara.Tests.Query;

[TestFixture]
public class CancellationMidOperationTests
{
    sealed class Row
    {
        public int X { get; set; }
        public int Key { get; set; }
    }

    sealed class FilterRow
    {
        public int X { get; set; }
    }

    [Test]
    public void Filter_CancelledMidFlight_ThrowsOperationCanceledException()
    {
        const int n = 5_000_000;
        var data = new int[n];
        for (int i = 0; i < n; i++) data[i] = i;
        var frame = Nivara.NivaraFrame.Create(
            ("X", Nivara.NivaraColumn<int>.Create(data)));
        var query = frame.Query<FilterRow>().Where(r => r.X >= 0);

        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(10));
        Assert.Throws<OperationCanceledException>(() =>
            query.CollectAsync(cts.Token).GetAwaiter().GetResult());
    }

    [Test]
    public void GroupBy_CancelledMidFlight_ThrowsOperationCanceledException()
    {
        const int n = 1_000_000;
        var data = new int[n];
        for (int i = 0; i < n; i++) data[i] = i % 1000;
        var columns = new Dictionary<string, Nivara.IColumn>(StringComparer.OrdinalIgnoreCase)
        {
            ["Key"] = Nivara.NivaraColumn<int>.Create(data)
        };

        var groupByOp = new GroupByOperation(new[] { Nivara.Expressions.ColumnExpressions.Col("Key") });

        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(10));
        Assert.Throws<OperationCanceledException>(() =>
            groupByOp.ExecuteAsync(columns, cts.Token).GetAwaiter().GetResult());
    }

    [Test]
    public void Join_CancelledMidFlight_ThrowsOperationCanceledException()
    {
        const int n = 500_000;
        var keys = new int[n];
        for (int i = 0; i < n; i++) keys[i] = i % 1000;
        var left = new Dictionary<string, Nivara.IColumn>(StringComparer.OrdinalIgnoreCase)
        {
            ["Key"] = Nivara.NivaraColumn<int>.Create(keys)
        };
        var right = new Dictionary<string, Nivara.IColumn>(StringComparer.OrdinalIgnoreCase)
        {
            ["Key"] = Nivara.NivaraColumn<int>.Create(keys)
        };

        var joinOp = new JoinOperation(left, right, JoinType.Inner, new[] { new JoinKey("Key") });

        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(10));
        Assert.Throws<OperationCanceledException>(() =>
            joinOp.ExecuteAsync(new Dictionary<string, Nivara.IColumn>(), cts.Token).GetAwaiter().GetResult());
    }
}