using Nivara.Exceptions;
using Nivara.Expressions;
using Nivara.Operations;
using NUnit.Framework;

namespace Nivara.Tests.Expressions;

[TestFixture]
public class WindowExpressionTests
{
    static Schema CreateSchema()
    {
        return new Schema(new[]
        {
            ("ID", typeof(int)),
            ("Salary", typeof(double)),
            ("Count", typeof(int)),
            ("Category", typeof(string)),
            ("Score", typeof(int)),
            ("Dept", typeof(string)),
        });
    }

    [Test]
    public void RollingAggregates_ResultType_MapsPerKind()
    {
        var salary = ColumnExpressions.Col<double>("Salary");

        Assert.That(ColumnExpressions.RollingSum(salary, 2).ResultType, Is.EqualTo(typeof(double)));
        Assert.That(ColumnExpressions.RollingMean(salary, 2).ResultType, Is.EqualTo(typeof(double)));
        Assert.That(ColumnExpressions.RollingMin(salary, 2).ResultType, Is.EqualTo(typeof(double)));
        Assert.That(ColumnExpressions.RollingMax(salary, 2).ResultType, Is.EqualTo(typeof(double)));

        var count = ColumnExpressions.Col<int>("Count");
        Assert.That(ColumnExpressions.RollingSum(count, 2).ResultType, Is.EqualTo(typeof(int)));
        Assert.That(ColumnExpressions.RollingMean(count, 2).ResultType, Is.EqualTo(typeof(double)), "rolling mean promotes to double");
    }

    [Test]
    public void CumulativeAggregates_ResultType_MapsPerKind()
    {
        var salary = ColumnExpressions.Col<double>("Salary");

        Assert.That(ColumnExpressions.CumulativeSum(salary).ResultType, Is.EqualTo(typeof(double)));
        Assert.That(ColumnExpressions.CumulativeMax(salary).ResultType, Is.EqualTo(typeof(double)));
        Assert.That(ColumnExpressions.CumulativeMin(salary).ResultType, Is.EqualTo(typeof(double)));
        Assert.That(ColumnExpressions.CumulativeProduct(salary).ResultType, Is.EqualTo(typeof(double)));
        Assert.That(ColumnExpressions.CumulativeCount(salary).ResultType, Is.EqualTo(typeof(long)), "running count is long");
        Assert.That(ColumnExpressions.CumulativeCount(ColumnExpressions.Col<string>("Category")).ResultType, Is.EqualTo(typeof(long)));
    }

    [Test]
    public void ShiftAndLead_ResultType_PreservesSourceType()
    {
        Assert.That(ColumnExpressions.Shift(ColumnExpressions.Col<double>("Salary"), 1).ResultType, Is.EqualTo(typeof(double)));
        Assert.That(ColumnExpressions.Lead(ColumnExpressions.Col<int>("ID"), 2).ResultType, Is.EqualTo(typeof(int)));
    }

    [Test]
    public void RankFamily_ResultType_IsLongOrDouble()
    {
        var order = new[] { new SortExpressionKey(ColumnExpressions.Col("Score")) };

        Assert.That(ColumnExpressions.RowNumber().ResultType, Is.EqualTo(typeof(long)));
        Assert.That(ColumnExpressions.Rank(order).ResultType, Is.EqualTo(typeof(long)));
        Assert.That(ColumnExpressions.DenseRank(order).ResultType, Is.EqualTo(typeof(long)));
        Assert.That(ColumnExpressions.PercentRank(order).ResultType, Is.EqualTo(typeof(double)));
    }

    [Test]
    public void Validate_ResolvesUntypedSource_AndRecomputesResultType()
    {
        var schema = CreateSchema();
        var expression = (WindowExpression)ColumnExpressions.RollingMean(ColumnExpressions.Col("Salary"), 3);

        expression.Validate(schema);

        Assert.That(expression.ResultType, Is.EqualTo(typeof(double)));
    }

    [Test]
    public void Validate_MissingSourceColumn_Throws()
    {
        var schema = CreateSchema();
        var expression = ColumnExpressions.RollingSum(ColumnExpressions.Col("Missing"), 2);

        Assert.That(() => expression.Validate(schema), Throws.TypeOf<SchemaValidationException>());
    }

    [Test]
    public void Validate_RankFamily_ValidatesKeysAndPartitions()
    {
        var schema = CreateSchema();
        var expression = (WindowExpression)ColumnExpressions.Rank(
            new[] { new SortExpressionKey(ColumnExpressions.Col("Score")) },
            new[] { ColumnExpressions.Col("Dept") });

        expression.Validate(schema);

        Assert.That(expression.ResultType, Is.EqualTo(typeof(long)));
    }

    [Test]
    public void Rank_WithoutOrderKeys_Throws()
    {
        Assert.That(
            () => ColumnExpressions.Rank(Array.Empty<SortExpressionKey>()),
            Throws.TypeOf<ArgumentException>());
    }

    [Test]
    public void Rolling_WithNonPositiveWindow_Throws()
    {
        Assert.That(
            () => ColumnExpressions.RollingSum(ColumnExpressions.Col("Salary"), 0),
            Throws.TypeOf<ArgumentOutOfRangeException>());
    }

    [Test]
    public void Name_FormatsKindSourceAndWindow()
    {
        Assert.That(ColumnExpressions.RollingSum(ColumnExpressions.Col("Salary"), 2).Name, Is.EqualTo("RollingSum(Salary, 2)"));
        Assert.That(ColumnExpressions.CumulativeCount(ColumnExpressions.Col("Salary")).Name, Is.EqualTo("CumulativeCount(Salary)"));
        Assert.That(ColumnExpressions.Shift(ColumnExpressions.Col("Salary"), 1).Name, Is.EqualTo("Shift(Salary, 1)"));
        Assert.That(ColumnExpressions.Lead(ColumnExpressions.Col("Salary"), 2).Name, Is.EqualTo("Lead(Salary, 2)"));
    }

    [Test]
    public void Name_RankFamily_IncludesOrderAndPartition()
    {
        var expression = ColumnExpressions.Rank(
            new[] { new SortExpressionKey(ColumnExpressions.Col("Score"), SortDirection.Descending) },
            new[] { ColumnExpressions.Col("Dept") });

        Assert.That(expression.Name, Is.EqualTo("Rank(Score) OVER (PARTITION BY Dept)"));
    }
}
