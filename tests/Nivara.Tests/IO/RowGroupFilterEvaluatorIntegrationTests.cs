using Nivara.Expressions;
using Nivara.IO;
using NUnit.Framework;

namespace Nivara.Tests.IO;

[TestFixture]
public class RowGroupFilterEvaluatorIntegrationTests
{
    [Test]
    public void ApplyFilterPredicate_SingleColumn_GreaterThan_SkipsNonMatchingRowGroups()
    {
        var tempFile = Path.GetTempFileName();
        try
        {
            var values = Enumerable.Range(0, 3000).Select(i => i).ToArray();
            var frame = NivaraFrame.Create(("Value", NivaraColumn<int>.Create(values)));
            NivaraParquetWriter.WriteParquet(frame, tempFile,
                ParquetWriteOptions.Default.With(rowGroupSize: 1000));
            frame.Dispose();

            using var source = new ParquetLazySource(tempFile);
            Assert.That(source.RowGroupCount, Is.EqualTo(3));

            var schema = new Schema(new[] { ("Value", typeof(int)) });
            var filter = new ComparisonExpression(
                ComparisonOperator.GreaterThan,
                new ColumnReference("Value"),
                new LiteralExpression(2500));

            source.ApplyFilterPredicate(filter, schema);

            // Pushdown only eliminates row groups, not individual rows.
            // Row group 2 [2000-2999] is the only group that could contain values > 2500.
            var result = source.Execute();
            var column = result["Value"];
            Assert.That(column.Length, Is.EqualTo(1000),
                "Pushdown should keep the entire row group [2000-2999] that may contain matching rows");
            Assert.That(column.GetValue(0), Is.EqualTo(2000));
            source.Dispose();
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public void ApplyFilterPredicate_SingleColumn_LessThan_SkipsNonMatchingRowGroups()
    {
        var tempFile = Path.GetTempFileName();
        try
        {
            var values = Enumerable.Range(0, 3000).Select(i => i).ToArray();
            var frame = NivaraFrame.Create(("Value", NivaraColumn<int>.Create(values)));
            NivaraParquetWriter.WriteParquet(frame, tempFile,
                ParquetWriteOptions.Default.With(rowGroupSize: 1000));
            frame.Dispose();

            using var source = new ParquetLazySource(tempFile);

            var schema = new Schema(new[] { ("Value", typeof(int)) });
            var filter = new ComparisonExpression(
                ComparisonOperator.LessThan,
                new ColumnReference("Value"),
                new LiteralExpression(1000));

            source.ApplyFilterPredicate(filter, schema);

            var result = source.Execute();
            var column = result["Value"];
            Assert.That(column.Length, Is.EqualTo(1000));
            Assert.That(column.GetValue(999), Is.EqualTo(999));
            source.Dispose();
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public void ApplyFilterPredicate_SingleColumn_Equal_SkipsNonMatchingRowGroups()
    {
        var tempFile = Path.GetTempFileName();
        try
        {
            var values = Enumerable.Range(0, 3000).Select(i => i).ToArray();
            var frame = NivaraFrame.Create(("Value", NivaraColumn<int>.Create(values)));
            NivaraParquetWriter.WriteParquet(frame, tempFile,
                ParquetWriteOptions.Default.With(rowGroupSize: 1000));
            frame.Dispose();

            using var source = new ParquetLazySource(tempFile);

            var schema = new Schema(new[] { ("Value", typeof(int)) });
            var filter = new ComparisonExpression(
                ComparisonOperator.Equal,
                new ColumnReference("Value"),
                new LiteralExpression(500));

            source.ApplyFilterPredicate(filter, schema);

            // Pushdown only eliminates row groups, not individual rows.
            // Row group 0 [0-999] is the only group that could contain value 500.
            var result = source.Execute();
            var column = result["Value"];
            Assert.That(column.Length, Is.EqualTo(1000),
                "Pushdown should keep the entire row group [0-999] that may contain the matching value");
            source.Dispose();
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public void ApplyFilterPredicate_AndChain_EliminatesMultipleRowGroups()
    {
        var tempFile = Path.GetTempFileName();
        try
        {
            var values = Enumerable.Range(0, 3000).Select(i => i).ToArray();
            var frame = NivaraFrame.Create(("Value", NivaraColumn<int>.Create(values)));
            NivaraParquetWriter.WriteParquet(frame, tempFile,
                ParquetWriteOptions.Default.With(rowGroupSize: 1000));
            frame.Dispose();

            using var source = new ParquetLazySource(tempFile);

            var schema = new Schema(new[] { ("Value", typeof(int)) });
            var left = new ComparisonExpression(
                ComparisonOperator.GreaterThanOrEqual,
                new ColumnReference("Value"),
                new LiteralExpression(1200));
            var right = new ComparisonExpression(
                ComparisonOperator.LessThan,
                new ColumnReference("Value"),
                new LiteralExpression(1800));
            var filter = new BinaryExpression(BinaryOperator.And, left, right);

            source.ApplyFilterPredicate(filter, schema);

            // Pushdown only eliminates row groups, not individual rows.
            // AND(>= 1200, < 1800): only row group 1 [1000-1999] satisfies both conditions.
            var result = source.Execute();
            var column = result["Value"];
            Assert.That(column.Length, Is.EqualTo(1000),
                "Pushdown should keep the entire row group [1000-1999] that may contain matching rows");
            source.Dispose();
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public void ApplyFilterPredicate_CanPushdownFilter_ReportsEligibility()
    {
        var tempFile = Path.GetTempFileName();
        try
        {
            var frame = NivaraFrame.Create(("Value", NivaraColumn<int>.Create(new[] { 1, 2, 3 })));
            NivaraParquetWriter.WriteParquet(frame, tempFile);
            frame.Dispose();

            using var source = new ParquetLazySource(tempFile);
            var schema = new Schema(new[] { ("Value", typeof(int)) });

            var eligibleFilter = new ComparisonExpression(
                ComparisonOperator.GreaterThan,
                new ColumnReference("Value"),
                new LiteralExpression(10));
            Assert.That(source.CanPushdownFilter(eligibleFilter, schema), Is.True);

            var ineligibleFilter = new ComparisonExpression(
                ComparisonOperator.GreaterThan,
                new ColumnReference("Nonexistent"),
                new LiteralExpression(10));
            Assert.That(source.CanPushdownFilter(ineligibleFilter, schema), Is.False);

            source.Dispose();
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }
}
