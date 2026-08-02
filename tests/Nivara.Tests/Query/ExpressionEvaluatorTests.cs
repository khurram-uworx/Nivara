using Nivara.Expressions;
using Nivara.Helpers;
using Nivara.Linq;
using NUnit.Framework;

namespace Nivara.Tests.Query;

/// <summary>
/// Regression guardrails for claim honesty (claims-integrity triage).
/// These tests fail if a corresponding honesty fix is reverted:
/// - the typed expression-evaluator fast path is actually selected for same-type
///   numeric columns and its output matches boxed semantics, including nulls;
/// - OrderBy on a computed key does not throw.
/// </summary>
[TestFixture]
public class ExpressionEvaluatorTests
{
    [Test]
    public void Evaluate_SameTypeNumericBinary_UsesTypedPath_AndMatchesBoxedReference()
    {
        // Arrange
        var left = NivaraColumn<double>.CreateFromNullable(new double?[] { 1.5, 2.5, null, 4.0 });
        var right = NivaraColumn<double>.CreateFromNullable(new double?[] { 10.0, null, 30.0, 40.0 });
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = left,
            ["B"] = right
        };
        var evaluator = new ExpressionEvaluator();
        var expression = ColumnExpressions.Col("A") + ColumnExpressions.Col("B");

        // Act
        var result = evaluator.Evaluate(expression, input);

        // Assert
        Assert.That(evaluator.TypedPathEvaluationCount, Is.EqualTo(1), "same-type double operands must use the typed fast path");
        Assert.That(evaluator.BoxedPathEvaluationCount, Is.EqualTo(0), "typed path must not fall back to boxed for same-type operands");

        for (int i = 0; i < result.Length; i++)
        {
            var expectedNull = left.IsNull(i) || right.IsNull(i);
            Assert.That(result.IsNull(i), Is.EqualTo(expectedNull), $"null mask at {i} must be left-OR-right");
            if (!expectedNull)
                Assert.That(result.GetValue(i), Is.EqualTo((double)left.GetValue(i)! + (double)right.GetValue(i)!),
                    $"value at {i} must match boxed addition");
        }
    }

    [Test]
    public void Evaluate_SameTypeNumericComparison_UsesTypedPath_AndMatchesBoxedReference()
    {
        // Arrange
        var left = NivaraColumn<double>.CreateFromNullable(new double?[] { 1.5, 2.5, null, 4.0 });
        var right = NivaraColumn<double>.CreateFromNullable(new double?[] { 10.0, null, 30.0, 40.0 });
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = left,
            ["B"] = right
        };
        var evaluator = new ExpressionEvaluator();
        var expression = ColumnExpressions.Col("A") > ColumnExpressions.Col("B");

        // Act
        var result = evaluator.Evaluate(expression, input);

        // Assert
        Assert.That(evaluator.TypedPathEvaluationCount, Is.EqualTo(1), "same-type comparison must use the typed fast path");

        for (int i = 0; i < result.Length; i++)
        {
            var expectedNull = left.IsNull(i) || right.IsNull(i);
            Assert.That(result.IsNull(i), Is.EqualTo(expectedNull), $"null mask at {i} must be left-OR-right");
            if (!expectedNull)
                Assert.That(result.GetValue(i), Is.EqualTo((double)left.GetValue(i)! > (double)right.GetValue(i)!),
                    $"comparison value at {i} must match boxed semantics");
        }
    }

    [Test]
    public void Evaluate_MixedStorageSameTypeComparison_UsesTypedPath_WithNullMask()
    {
        // Memory-backed nullable column vs Tensor-backed constant column (issue #96)
        var left = NivaraColumn<double>.CreateFromNullable(new double?[] { 1.5, 2.5, null, 4.0 });
        var right = NivaraColumn<double>.Create(new[] { 2.0, 2.0, 2.0, 2.0 });
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = left,
            ["B"] = right
        };
        var evaluator = new ExpressionEvaluator();
        var expression = ColumnExpressions.Col("A") > ColumnExpressions.Col("B");

        var result = evaluator.Evaluate(expression, input);

        Assert.That(evaluator.TypedPathEvaluationCount, Is.EqualTo(1), "mixed-storage same-type comparison must use the typed fast path");
        Assert.That(evaluator.BoxedPathEvaluationCount, Is.EqualTo(0), "typed path must not fall back to boxed for mixed storage");

        for (int i = 0; i < result.Length; i++)
        {
            var expectedNull = left.IsNull(i) || right.IsNull(i);
            Assert.That(result.IsNull(i), Is.EqualTo(expectedNull), $"null mask at {i} must be left-OR-right");
            if (!expectedNull)
                Assert.That(result.GetValue(i), Is.EqualTo((double)left.GetValue(i)! > (double)right.GetValue(i)!),
                    $"comparison value at {i} must match boxed semantics");
        }
    }

    [Test]
    public void Evaluate_MixedStorageSameTypeBinary_UsesTypedPath_WithNullMask()
    {
        // Memory-backed nullable column vs Tensor-backed constant column (issue #96)
        var left = NivaraColumn<double>.CreateFromNullable(new double?[] { 1.5, 2.5, null, 4.0 });
        var right = NivaraColumn<double>.Create(new[] { 10.0, 10.0, 10.0, 10.0 });
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = left,
            ["B"] = right
        };
        var evaluator = new ExpressionEvaluator();
        var expression = ColumnExpressions.Col("A") + ColumnExpressions.Col("B");

        var result = evaluator.Evaluate(expression, input);

        Assert.That(evaluator.TypedPathEvaluationCount, Is.EqualTo(1), "mixed-storage same-type binary must use the typed fast path");
        Assert.That(evaluator.BoxedPathEvaluationCount, Is.EqualTo(0), "typed path must not fall back to boxed for mixed storage");

        for (int i = 0; i < result.Length; i++)
        {
            var expectedNull = left.IsNull(i) || right.IsNull(i);
            Assert.That(result.IsNull(i), Is.EqualTo(expectedNull), $"null mask at {i} must be left-OR-right");
            if (!expectedNull)
                Assert.That(result.GetValue(i), Is.EqualTo((double)left.GetValue(i)! + (double)right.GetValue(i)!),
                    $"value at {i} must match boxed addition");
        }
    }

    [Test]
    public void Evaluate_MixedTypeNumeric_FallsBackToBoxedPath()
    {
        // Arrange
        var doubles = NivaraColumn<double>.CreateFromNullable(new double?[] { 1.5, 2.5, null, 4.0 });
        var ints = NivaraColumn<int>.CreateFromNullable(new int?[] { 10, null, 30, 40 });
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = doubles,
            ["B"] = ints
        };
        var evaluator = new ExpressionEvaluator();
        var expression = ColumnExpressions.Col("A") + ColumnExpressions.Col("B");

        // Act
        var result = evaluator.Evaluate(expression, input);

        // Assert
        Assert.That(evaluator.TypedPathEvaluationCount, Is.EqualTo(0), "mixed element types must skip the typed path");
        Assert.That(evaluator.BoxedPathEvaluationCount, Is.EqualTo(1), "mixed element types must use the boxed fallback");

        for (int i = 0; i < result.Length; i++)
        {
            var expectedNull = doubles.IsNull(i) || ints.IsNull(i);
            Assert.That(result.IsNull(i), Is.EqualTo(expectedNull), $"null mask at {i} must be left-OR-right");
            if (!expectedNull)
                Assert.That(result.GetValue(i), Is.EqualTo(Convert.ToDouble(doubles.GetValue(i)) + Convert.ToDouble(ints.GetValue(i))),
                    $"value at {i} must match boxed Convert.ToDouble addition");
        }
    }

    [Test]
    public void OrderBy_OnComputedKey_DoesNotThrow()
    {
        // Arrange
        var ids = NivaraColumn<int>.Create(new[] { 3, 1, 2 });
        var vals = NivaraColumn<int>.Create(new[] { 30, 10, 20 });
        using var frame = NivaraFrame.Create(("ID", ids), ("Val", vals));

        // Act & Assert
        using var result = frame.AsQueryFrame()
            .OrderBy(x => x["Val"] + x["ID"])
            .Collect();

        Assert.That(result.RowCount, Is.EqualTo(3));
        Assert.That(result.GetColumn<int>("ID")[0], Is.EqualTo(1), "computed key 11 sorts first");
        Assert.That(result.GetColumn<int>("ID")[1], Is.EqualTo(2), "computed key 22 sorts second");
        Assert.That(result.GetColumn<int>("ID")[2], Is.EqualTo(3), "computed key 33 sorts last");
    }

    [Test]
    public void Evaluate_BoxedNullableGuidComparison_PropagatesNullMask()
    {
        // Guid is not a typed-fast-path element type, so this comparison must use the boxed fallback
        var target = Guid.NewGuid();
        var left = NivaraColumn<Guid>.CreateFromNullable(new Guid?[] { target, null, Guid.NewGuid(), target });
        var right = NivaraColumn<Guid>.CreateFromNullable(new Guid?[] { null, Guid.NewGuid(), Guid.NewGuid(), target });
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = left,
            ["B"] = right
        };
        var evaluator = new ExpressionEvaluator();
        var expression = ColumnExpressions.Col("A") > ColumnExpressions.Col("B");

        var result = evaluator.Evaluate(expression, input);

        Assert.That(evaluator.TypedPathEvaluationCount, Is.EqualTo(0), "Guid is not a typed-fast-path element type");
        Assert.That(evaluator.BoxedPathEvaluationCount, Is.EqualTo(1), "Guid comparison must use the boxed fallback");
        Assert.That(result.HasNulls, Is.True, "boxed comparison must propagate a null mask");

        for (int i = 0; i < result.Length; i++)
        {
            var expectedNull = left.IsNull(i) || right.IsNull(i);
            Assert.That(result.IsNull(i), Is.EqualTo(expectedNull), $"null mask at {i} must be left-OR-right");
            if (!expectedNull)
            {
                var l = (Guid)left.GetValue(i)!;
                var r = (Guid)right.GetValue(i)!;
                Assert.That(result.GetValue(i), Is.EqualTo(l.CompareTo(r) > 0), $"value at {i} must match boxed comparison");
            }
        }
    }

    [Test]
    public void Evaluate_BoxedNullableGuidEqualToLiteral_PropagatesNullMask()
    {
        // issue #103 regression: boxed comparison against a literal must produce SQL null semantics
        var target = Guid.NewGuid();
        var column = NivaraColumn<Guid>.CreateFromNullable(new Guid?[] { target, null, Guid.NewGuid(), target });
        var input = new Dictionary<string, IColumn>
        {
            ["ID"] = column
        };
        var evaluator = new ExpressionEvaluator();
        var expression = ColumnExpressions.Col("ID") == (object)target;

        var result = evaluator.Evaluate(expression, input);

        Assert.That(evaluator.TypedPathEvaluationCount, Is.EqualTo(0), "Guid is not a typed-fast-path element type");
        Assert.That(evaluator.BoxedPathEvaluationCount, Is.EqualTo(1), "Guid == literal must use the boxed fallback");
        Assert.That(result.HasNulls, Is.True, "boxed comparison must propagate a null mask");
        Assert.That(result.IsNull(1), Is.True, "null row must be masked");
        Assert.That(result.GetValue(0), Is.EqualTo(true), "matching value must compare equal");
        Assert.That(result.GetValue(2), Is.EqualTo(false), "non-matching value must not compare equal");
        Assert.That(result.GetValue(3), Is.EqualTo(true), "trailing matching value must compare equal");
    }

    [Test]
    public void Filter_BoxedNullableGuidNotEqual_ExcludesNullRows()
    {
        // issue #103 regression: a boxed NotEqual filter excludes null rows (SQL null semantics)
        var target = Guid.NewGuid();
        var ids = NivaraColumn<Guid>.CreateFromNullable(new Guid?[] { target, null, Guid.NewGuid() });
        using var frame = NivaraFrame.Create(("ID", ids));

        using var result = frame.AsQueryFrame()
            .Where(x => x["ID"] != (object)target)
            .Collect();

        Assert.That(result.RowCount, Is.EqualTo(1), "null row must be excluded, keeping only the non-equal row");
        Assert.That(result.GetColumn("ID").GetValue(0), Is.Not.EqualTo(target));
    }
}
