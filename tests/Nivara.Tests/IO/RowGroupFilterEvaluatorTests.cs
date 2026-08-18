using Nivara.Expressions;
using Nivara.IO;
using NUnit.Framework;

namespace Nivara.Tests.IO;

[TestFixture]
public class RowGroupFilterEvaluatorTests
{
    [Test]
    public void CanEvaluate_SingleComparison_ReturnsTrue()
    {
        var schema = new Schema(new[] { ("Age", typeof(int)) });
        var expr = new ComparisonExpression(
            ComparisonOperator.GreaterThan,
            new ColumnReference("Age"),
            new LiteralExpression(10));

        Assert.That(RowGroupFilterEvaluator.CanEvaluate(expr, schema), Is.True);
    }

    [Test]
    public void CanEvaluate_AndChain_ReturnsTrue()
    {
        var schema = new Schema(new[] { ("Age", typeof(int)), ("Score", typeof(double)) });
        var left = new ComparisonExpression(
            ComparisonOperator.GreaterThan,
            new ColumnReference("Age"),
            new LiteralExpression(10));
        var right = new ComparisonExpression(
            ComparisonOperator.LessThan,
            new ColumnReference("Score"),
            new LiteralExpression(90.0));
        var expr = new BinaryExpression(BinaryOperator.And, left, right);

        Assert.That(RowGroupFilterEvaluator.CanEvaluate(expr, schema), Is.True);
    }

    [Test]
    public void CanEvaluate_OrExpression_ReturnsFalse()
    {
        var schema = new Schema(new[] { ("Age", typeof(int)) });
        var left = new ComparisonExpression(
            ComparisonOperator.GreaterThan,
            new ColumnReference("Age"),
            new LiteralExpression(10));
        var right = new ComparisonExpression(
            ComparisonOperator.LessThan,
            new ColumnReference("Age"),
            new LiteralExpression(5));
        var expr = new BinaryExpression(BinaryOperator.Or, left, right);

        Assert.That(RowGroupFilterEvaluator.CanEvaluate(expr, schema), Is.False);
    }

    [Test]
    public void CanEvaluate_ColumnNotInSchema_ReturnsFalse()
    {
        var schema = new Schema(new[] { ("Age", typeof(int)) });
        var expr = new ComparisonExpression(
            ComparisonOperator.GreaterThan,
            new ColumnReference("Nonexistent"),
            new LiteralExpression(10));

        Assert.That(RowGroupFilterEvaluator.CanEvaluate(expr, schema), Is.False);
    }

    [Test]
    public void CanEvaluate_LiteralOnLeft_ReturnsFalse()
    {
        var schema = new Schema(new[] { ("Age", typeof(int)) });
        var expr = new ComparisonExpression(
            ComparisonOperator.GreaterThan,
            new LiteralExpression(10),
            new ColumnReference("Age"));

        Assert.That(RowGroupFilterEvaluator.CanEvaluate(expr, schema), Is.False);
    }

    [Test]
    public void EvaluateRowGroup_GreaterThan_MinMatches_KeepsRowGroup()
    {
        var schema = new Schema(new[] { ("Age", typeof(int)) });
        var expr = new ComparisonExpression(
            ComparisonOperator.GreaterThan,
            new ColumnReference("Age"),
            new LiteralExpression(5));

        var stats = new RowGroupFilterEvaluator.RowGroupColumnStats { MinValue = 10, MaxValue = 100 };

        bool result = RowGroupFilterEvaluator.EvaluateRowGroup(
            expr,
            _ => stats,
            schema);

        Assert.That(result, Is.True);
    }

    [Test]
    public void EvaluateRowGroup_GreaterThan_MinExceedsLiteral_SkipsRowGroup()
    {
        var schema = new Schema(new[] { ("Age", typeof(int)) });
        var expr = new ComparisonExpression(
            ComparisonOperator.GreaterThan,
            new ColumnReference("Age"),
            new LiteralExpression(50));

        var stats = new RowGroupFilterEvaluator.RowGroupColumnStats { MinValue = 10, MaxValue = 20 };

        bool result = RowGroupFilterEvaluator.EvaluateRowGroup(
            expr,
            _ => stats,
            schema);

        Assert.That(result, Is.False);
    }

    [Test]
    public void EvaluateRowGroup_LessThan_MaxMatches_KeepsRowGroup()
    {
        var schema = new Schema(new[] { ("Age", typeof(int)) });
        var expr = new ComparisonExpression(
            ComparisonOperator.LessThan,
            new ColumnReference("Age"),
            new LiteralExpression(50));

        var stats = new RowGroupFilterEvaluator.RowGroupColumnStats { MinValue = 10, MaxValue = 20 };

        bool result = RowGroupFilterEvaluator.EvaluateRowGroup(
            expr,
            _ => stats,
            schema);

        Assert.That(result, Is.True);
    }

    [Test]
    public void EvaluateRowGroup_LessThan_MaxBelowLiteral_SkipsRowGroup()
    {
        var schema = new Schema(new[] { ("Age", typeof(int)) });
        var expr = new ComparisonExpression(
            ComparisonOperator.LessThan,
            new ColumnReference("Age"),
            new LiteralExpression(5));

        var stats = new RowGroupFilterEvaluator.RowGroupColumnStats { MinValue = 10, MaxValue = 20 };

        bool result = RowGroupFilterEvaluator.EvaluateRowGroup(
            expr,
            _ => stats,
            schema);

        Assert.That(result, Is.False);
    }

    [Test]
    public void EvaluateRowGroup_Equal_LiteralInRange_KeepsRowGroup()
    {
        var schema = new Schema(new[] { ("Id", typeof(int)) });
        var expr = new ComparisonExpression(
            ComparisonOperator.Equal,
            new ColumnReference("Id"),
            new LiteralExpression(15));

        var stats = new RowGroupFilterEvaluator.RowGroupColumnStats { MinValue = 10, MaxValue = 20 };

        bool result = RowGroupFilterEvaluator.EvaluateRowGroup(
            expr,
            _ => stats,
            schema);

        Assert.That(result, Is.True);
    }

    [Test]
    public void EvaluateRowGroup_Equal_LiteralOutOfRange_SkipsRowGroup()
    {
        var schema = new Schema(new[] { ("Id", typeof(int)) });
        var expr = new ComparisonExpression(
            ComparisonOperator.Equal,
            new ColumnReference("Id"),
            new LiteralExpression(25));

        var stats = new RowGroupFilterEvaluator.RowGroupColumnStats { MinValue = 10, MaxValue = 20 };

        bool result = RowGroupFilterEvaluator.EvaluateRowGroup(
            expr,
            _ => stats,
            schema);

        Assert.That(result, Is.False);
    }

    [Test]
    public void EvaluateRowGroup_AndChain_BothMatch_KeepsRowGroup()
    {
        var schema = new Schema(new[] { ("Age", typeof(int)), ("Score", typeof(double)) });
        var left = new ComparisonExpression(
            ComparisonOperator.GreaterThan,
            new ColumnReference("Age"),
            new LiteralExpression(5));
        var right = new ComparisonExpression(
            ComparisonOperator.LessThan,
            new ColumnReference("Score"),
            new LiteralExpression(90.0));
        var expr = new BinaryExpression(BinaryOperator.And, left, right);

        var statsDict = new Dictionary<string, RowGroupFilterEvaluator.RowGroupColumnStats>
        {
            ["Age"] = new() { MinValue = 10, MaxValue = 50 },
            ["Score"] = new() { MinValue = 20.0, MaxValue = 80.0 }
        };

        bool result = RowGroupFilterEvaluator.EvaluateRowGroup(
            expr,
            name => statsDict.TryGetValue(name, out var s) ? s : null,
            schema);

        Assert.That(result, Is.True);
    }

    [Test]
    public void EvaluateRowGroup_AndChain_OneFails_SkipsRowGroup()
    {
        var schema = new Schema(new[] { ("Age", typeof(int)), ("Score", typeof(double)) });
        var left = new ComparisonExpression(
            ComparisonOperator.GreaterThan,
            new ColumnReference("Age"),
            new LiteralExpression(5));
        var right = new ComparisonExpression(
            ComparisonOperator.LessThan,
            new ColumnReference("Score"),
            new LiteralExpression(10.0));
        var expr = new BinaryExpression(BinaryOperator.And, left, right);

        var statsDict = new Dictionary<string, RowGroupFilterEvaluator.RowGroupColumnStats>
        {
            ["Age"] = new() { MinValue = 10, MaxValue = 50 },
            ["Score"] = new() { MinValue = 20.0, MaxValue = 80.0 }
        };

        bool result = RowGroupFilterEvaluator.EvaluateRowGroup(
            expr,
            name => statsDict.TryGetValue(name, out var s) ? s : null,
            schema);

        Assert.That(result, Is.False);
    }

    [Test]
    public void EvaluateRowGroup_NullStats_KeepsRowGroup()
    {
        var schema = new Schema(new[] { ("Age", typeof(int)) });
        var expr = new ComparisonExpression(
            ComparisonOperator.GreaterThan,
            new ColumnReference("Age"),
            new LiteralExpression(50));

        bool result = RowGroupFilterEvaluator.EvaluateRowGroup(
            expr,
            _ => null,
            schema);

        Assert.That(result, Is.True);
    }

    [Test]
    public void EvaluateRowGroup_NullMinMax_KeepsRowGroup()
    {
        var schema = new Schema(new[] { ("Age", typeof(int)) });
        var expr = new ComparisonExpression(
            ComparisonOperator.GreaterThan,
            new ColumnReference("Age"),
            new LiteralExpression(50));

        var stats = new RowGroupFilterEvaluator.RowGroupColumnStats { MinValue = null, MaxValue = null };

        bool result = RowGroupFilterEvaluator.EvaluateRowGroup(
            expr,
            _ => stats,
            schema);

        Assert.That(result, Is.True);
    }

    [Test]
    public void EvaluateRowGroup_NotEqual_AlwaysKeepsRowGroup()
    {
        var schema = new Schema(new[] { ("Age", typeof(int)) });
        var expr = new ComparisonExpression(
            ComparisonOperator.NotEqual,
            new ColumnReference("Age"),
            new LiteralExpression(15));

        var stats = new RowGroupFilterEvaluator.RowGroupColumnStats { MinValue = 15, MaxValue = 15 };

        bool result = RowGroupFilterEvaluator.EvaluateRowGroup(
            expr,
            _ => stats,
            schema);

        Assert.That(result, Is.True);
    }

    [Test]
    public void EvaluateRowGroup_Equal_ColumnValuesSame_SkipsRowGroup()
    {
        var schema = new Schema(new[] { ("Id", typeof(int)) });
        var expr = new ComparisonExpression(
            ComparisonOperator.Equal,
            new ColumnReference("Id"),
            new LiteralExpression(5));

        var stats = new RowGroupFilterEvaluator.RowGroupColumnStats { MinValue = 10, MaxValue = 10 };

        bool result = RowGroupFilterEvaluator.EvaluateRowGroup(
            expr,
            _ => stats,
            schema);

        Assert.That(result, Is.False);
    }

    [Test]
    public void EvaluateRowGroup_GreaterThanOrEqual_MinEqualsLiteral_KeepsRowGroup()
    {
        var schema = new Schema(new[] { ("Age", typeof(int)) });
        var expr = new ComparisonExpression(
            ComparisonOperator.GreaterThanOrEqual,
            new ColumnReference("Age"),
            new LiteralExpression(10));

        var stats = new RowGroupFilterEvaluator.RowGroupColumnStats { MinValue = 10, MaxValue = 50 };

        bool result = RowGroupFilterEvaluator.EvaluateRowGroup(
            expr,
            _ => stats,
            schema);

        Assert.That(result, Is.True);
    }

    [Test]
    public void EvaluateRowGroup_LessThanOrEqual_MaxEqualsLiteral_KeepsRowGroup()
    {
        var schema = new Schema(new[] { ("Age", typeof(int)) });
        var expr = new ComparisonExpression(
            ComparisonOperator.LessThanOrEqual,
            new ColumnReference("Age"),
            new LiteralExpression(50));

        var stats = new RowGroupFilterEvaluator.RowGroupColumnStats { MinValue = 10, MaxValue = 50 };

        bool result = RowGroupFilterEvaluator.EvaluateRowGroup(
            expr,
            _ => stats,
            schema);

        Assert.That(result, Is.True);
    }

    [Test]
    public void EvaluateRowGroup_Equal_StringValuesInRange_KeepsRowGroup()
    {
        var schema = new Schema(new[] { ("Name", typeof(string)) });
        var expr = new ComparisonExpression(
            ComparisonOperator.Equal,
            new ColumnReference("Name"),
            new LiteralExpression("Charlie"));

        var stats = new RowGroupFilterEvaluator.RowGroupColumnStats { MinValue = "Alice", MaxValue = "Eve" };

        bool result = RowGroupFilterEvaluator.EvaluateRowGroup(
            expr,
            _ => stats,
            schema);

        Assert.That(result, Is.True);
    }

    [Test]
    public void EvaluateRowGroup_Equal_StringValuesOutOfRange_SkipsRowGroup()
    {
        var schema = new Schema(new[] { ("Name", typeof(string)) });
        var expr = new ComparisonExpression(
            ComparisonOperator.Equal,
            new ColumnReference("Name"),
            new LiteralExpression("Zach"));

        var stats = new RowGroupFilterEvaluator.RowGroupColumnStats { MinValue = "Alice", MaxValue = "Eve" };

        bool result = RowGroupFilterEvaluator.EvaluateRowGroup(
            expr,
            _ => stats,
            schema);

        Assert.That(result, Is.False);
    }

    [Test]
    public void EvaluateRowGroup_GreaterThan_LiteralBetweenMinMax_KeepsRowGroup()
    {
        var schema = new Schema(new[] { ("Age", typeof(int)) });
        var expr = new ComparisonExpression(
            ComparisonOperator.GreaterThan,
            new ColumnReference("Age"),
            new LiteralExpression(15));

        var stats = new RowGroupFilterEvaluator.RowGroupColumnStats { MinValue = 10, MaxValue = 20 };

        bool result = RowGroupFilterEvaluator.EvaluateRowGroup(
            expr,
            _ => stats,
            schema);

        Assert.That(result, Is.True,
            "Row group [10,20] contains values > 15, so it must be kept");
    }

    [Test]
    public void EvaluateRowGroup_GreaterThanOrEqual_LiteralBetweenMinMax_KeepsRowGroup()
    {
        var schema = new Schema(new[] { ("Age", typeof(int)) });
        var expr = new ComparisonExpression(
            ComparisonOperator.GreaterThanOrEqual,
            new ColumnReference("Age"),
            new LiteralExpression(15));

        var stats = new RowGroupFilterEvaluator.RowGroupColumnStats { MinValue = 10, MaxValue = 20 };

        bool result = RowGroupFilterEvaluator.EvaluateRowGroup(
            expr,
            _ => stats,
            schema);

        Assert.That(result, Is.True,
            "Row group [10,20] contains values >= 15, so it must be kept");
    }

    [Test]
    public void EvaluateRowGroup_LessThan_LiteralBetweenMinMax_KeepsRowGroup()
    {
        var schema = new Schema(new[] { ("Age", typeof(int)) });
        var expr = new ComparisonExpression(
            ComparisonOperator.LessThan,
            new ColumnReference("Age"),
            new LiteralExpression(15));

        var stats = new RowGroupFilterEvaluator.RowGroupColumnStats { MinValue = 10, MaxValue = 20 };

        bool result = RowGroupFilterEvaluator.EvaluateRowGroup(
            expr,
            _ => stats,
            schema);

        Assert.That(result, Is.True,
            "Row group [10,20] contains values < 15, so it must be kept");
    }

    [Test]
    public void EvaluateRowGroup_LessThanOrEqual_LiteralBetweenMinMax_KeepsRowGroup()
    {
        var schema = new Schema(new[] { ("Age", typeof(int)) });
        var expr = new ComparisonExpression(
            ComparisonOperator.LessThanOrEqual,
            new ColumnReference("Age"),
            new LiteralExpression(15));

        var stats = new RowGroupFilterEvaluator.RowGroupColumnStats { MinValue = 10, MaxValue = 20 };

        bool result = RowGroupFilterEvaluator.EvaluateRowGroup(
            expr,
            _ => stats,
            schema);

        Assert.That(result, Is.True,
            "Row group [10,20] contains values <= 15, so it must be kept");
    }
}
