using Nivara.Expressions;
using NUnit.Framework;
using System.Text.Json;

namespace Nivara.Tests.Expressions;

/// <summary>
/// Tests for the broadcast quantile/median aggregate expression nodes
/// (<see cref="ColumnExpressions.Quantile"/> / <see cref="ColumnExpressions.Median"/>),
/// mirroring the typed series path and the polars reference fixtures.
/// </summary>
[TestFixture]
public class BroadcastAggregateExpressionTests
{
    const double Tolerance = 1e-9;

    static IColumn Evaluate(ColumnExpression expression, IReadOnlyDictionary<string, IColumn> input)
        => new FusedExpressionEvaluator().Evaluate(expression, input);

    [Test]
    public void Evaluate_QuantileExpression_BroadcastsSeriesQuantile()
    {
        var column = NivaraColumn<double>.Create(new[] { 1.0, 2.0, 3.0, 4.0 });
        var input = new Dictionary<string, IColumn> { ["A"] = column };

        using var series = new NivaraSeries<double>(column);
        var expected = series.Quantile(0.25);

        var result = Evaluate(ColumnExpressions.Quantile(ColumnExpressions.Col("A"), 0.25), input);

        Assert.That(result.ElementType, Is.EqualTo(typeof(double)));
        Assert.That(result.HasNulls, Is.False);
        for (int i = 0; i < column.Length; i++)
            Assert.That((double)result.GetValue(i)!, Is.EqualTo(expected).Within(Tolerance), $"row {i}");
    }

    [Test]
    public void Evaluate_MedianExpression_BroadcastsSeriesMedian()
    {
        var column = NivaraColumn<double>.Create(new[] { 1.0, 2.0, 3.0, 4.0 });
        var input = new Dictionary<string, IColumn> { ["A"] = column };

        using var series = new NivaraSeries<double>(column);
        var expected = series.Median();

        var result = Evaluate(ColumnExpressions.Median(ColumnExpressions.Col("A")), input);

        Assert.That(result.ElementType, Is.EqualTo(typeof(double)));
        for (int i = 0; i < column.Length; i++)
            Assert.That((double)result.GetValue(i)!, Is.EqualTo(expected).Within(Tolerance), $"row {i}");
    }

    [Test]
    public void Evaluate_QuantileExpression_WithNulls_IgnoresNulls()
    {
        var column = NivaraColumn.CreateFromNullable(new double?[] { 1.0, null, 3.0, 4.0 });
        var input = new Dictionary<string, IColumn> { ["A"] = column };

        using var series = new NivaraSeries<double>(column);
        var expected = series.Quantile(0.5);

        var result = Evaluate(ColumnExpressions.Quantile(ColumnExpressions.Col("A"), 0.5), input);

        Assert.That(result.HasNulls, Is.False, "nulls in the source must not leak into the broadcast result");
        for (int i = 0; i < column.Length; i++)
            Assert.That((double)result.GetValue(i)!, Is.EqualTo(expected).Within(Tolerance), $"row {i}");
    }

    [Test]
    public void Evaluate_QuantileExpression_ComposedWithArithmetic_FusesSurroundingExpression()
    {
        var column = NivaraColumn<double>.Create(new[] { 1.0, 2.0, 3.0, 4.0 });
        var input = new Dictionary<string, IColumn> { ["A"] = column };

        using var series = new NivaraSeries<double>(column);
        var expected = series.Quantile(0.25) * 2;

        var fused = new FusedExpressionEvaluator();
        var result = fused.Evaluate(ColumnExpressions.Quantile(ColumnExpressions.Col("A"), 0.25) * 2, input);

        Assert.That(fused.FusedPathEvaluationCount, Is.EqualTo(2), "window source passthrough + one fused outer pass");
        for (int i = 0; i < column.Length; i++)
            Assert.That((double)result.GetValue(i)!, Is.EqualTo(expected).Within(Tolerance), $"row {i}");
    }

    [Test]
    public void Evaluate_QuantileExpression_AllNullSource_ProducesAllNullColumn()
    {
        var column = NivaraColumn.CreateFromNullable(new double?[] { null, null, null });
        var input = new Dictionary<string, IColumn> { ["A"] = column };

        var result = Evaluate(ColumnExpressions.Quantile(ColumnExpressions.Col("A"), 0.5), input);

        Assert.That(result.HasNulls, Is.True);
        for (int i = 0; i < column.Length; i++)
            Assert.That(result.IsNull(i), Is.True, $"row {i}");
    }

    [Test]
    public void Factory_Quantile_OutOfRangeQ_ThrowsArgumentOutOfRangeException()
    {
        foreach (var q in new[] { double.NaN, -0.5, 1.5 })
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                ColumnExpressions.Quantile(ColumnExpressions.Col("A"), q));
        }
    }

    [Test]
    public void Factory_QuantileAndMedian_NamesReflectArguments()
    {
        Assert.That(ColumnExpressions.Quantile(ColumnExpressions.Col("A"), 0.25).Name, Is.EqualTo("Quantile(A, 0.25)"));
        Assert.That(ColumnExpressions.Median(ColumnExpressions.Col("A")).Name, Is.EqualTo("Median(A)"));
    }

    [Test]
    public void Validate_QuantileExpression_ResolvesDoubleResultType()
    {
        var schema = new Schema(new[] { ("A", typeof(int)) });

        var quantile = ColumnExpressions.Quantile(ColumnExpressions.Col("A"), 0.25);
        quantile.Validate(schema);
        Assert.That(quantile.ResultType, Is.EqualTo(typeof(double)));

        var median = ColumnExpressions.Median(ColumnExpressions.Col("A"));
        median.Validate(schema);
        Assert.That(median.ResultType, Is.EqualTo(typeof(double)));
    }

    [Test]
    public void ContainsWindowExpression_QuantileOrMedian_ReturnsTrue()
    {
        Assert.That(FusedExpressionEvaluator.ContainsWindowExpression(
            ColumnExpressions.Quantile(ColumnExpressions.Col("A"), 0.25)), Is.True);
        Assert.That(FusedExpressionEvaluator.ContainsWindowExpression(
            ColumnExpressions.Median(ColumnExpressions.Col("A"))), Is.True);
    }

    [Test]
    public void QueryFrame_Select_QuantileMedian_MatchesPolarsFixtures()
    {
        var manifestPath = Path.Combine(
            TestContext.CurrentContext.TestDirectory,
            "..", "..", "..", "..", "..", "samples", "data", "polars-quantile", "manifest.json");
        Assert.That(File.Exists(manifestPath), Is.True, $"Missing reference file: {manifestPath}");

        using var doc = JsonDocument.Parse(File.ReadAllBytes(manifestPath));
        var cases = doc.RootElement.EnumerateArray().ToList();
        Assert.That(cases, Is.Not.Empty, "manifest has no cases");

        foreach (var caseEl in cases)
        {
            var name = caseEl.GetProperty("name").GetString()!;
            var kind = caseEl.GetProperty("kind").GetString()!;
            var values = caseEl.GetProperty("v").EnumerateArray()
                .Select(e => e.ValueKind == JsonValueKind.Null ? (double?)null : e.GetDouble())
                .ToArray();

            using var frame = NivaraFrame.Create(("v", NivaraColumn.CreateFromNullable(values)));

            if (kind == "quantile")
            {
                var q = caseEl.GetProperty("q").GetDouble();
                var expected = caseEl.GetProperty("quantile").GetDouble();

                using var result = frame.AsQueryFrame()
                    .Select(ColumnExpressions.Quantile(ColumnExpressions.Col("v"), q))
                    .Collect();
                var column = result.GetColumn(result.ColumnNames[0]);
                for (int i = 0; i < column.Length; i++)
                    Assert.That((double)column.GetValue(i)!, Is.EqualTo(expected).Within(Tolerance),
                        $"{name}: Quantile({q}) row {i} mismatch");
            }
            else
            {
                Assert.That(kind, Is.EqualTo("median"), $"{name}: unknown kind");

                var expected = caseEl.GetProperty("median").GetDouble();

                using var result = frame.AsQueryFrame()
                    .Select(ColumnExpressions.Median(ColumnExpressions.Col("v")))
                    .Collect();
                var column = result.GetColumn(result.ColumnNames[0]);
                for (int i = 0; i < column.Length; i++)
                    Assert.That((double)column.GetValue(i)!, Is.EqualTo(expected).Within(Tolerance),
                        $"{name}: Median() row {i} mismatch");
            }
        }
    }
}
