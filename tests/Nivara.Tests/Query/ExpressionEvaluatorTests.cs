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
    public void Evaluate_MixedTypeNumericBinary_UsesTypedPromotedPath()
    {
        // double + int promotes to double and uses the typed promoted kernel (C# binary numeric promotion)
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
        Assert.That(evaluator.TypedPathEvaluationCount, Is.EqualTo(1), "mixed promotable numerics must use the typed promoted path");
        Assert.That(evaluator.BoxedPathEvaluationCount, Is.EqualTo(0), "typed promoted path must not fall back to boxed");
        Assert.That(result.ElementType, Is.EqualTo(typeof(double)), "double + int must promote to double");

        for (int i = 0; i < result.Length; i++)
        {
            var expectedNull = doubles.IsNull(i) || ints.IsNull(i);
            Assert.That(result.IsNull(i), Is.EqualTo(expectedNull), $"null mask at {i} must be left-OR-right");
            if (!expectedNull)
                Assert.That(result.GetValue(i), Is.EqualTo(doubles[i] + ints[i]),
                    $"value at {i} must match C# promoted addition");
        }
    }

    [Test]
    public void Evaluate_MixedTypeNumericBinary_PromotesToLong()
    {
        // int + long promotes to long
        var ints = NivaraColumn<int>.CreateFromNullable(new int?[] { 1, 2, null, 4 });
        var longs = NivaraColumn<long>.CreateFromNullable(new long?[] { 100, null, 300, 400 });
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = ints,
            ["B"] = longs
        };
        var evaluator = new ExpressionEvaluator();
        var expression = ColumnExpressions.Col("A") + ColumnExpressions.Col("B");

        var result = evaluator.Evaluate(expression, input);

        Assert.That(evaluator.TypedPathEvaluationCount, Is.EqualTo(1));
        Assert.That(evaluator.BoxedPathEvaluationCount, Is.EqualTo(0));
        Assert.That(result.ElementType, Is.EqualTo(typeof(long)), "int + long must promote to long");

        for (int i = 0; i < result.Length; i++)
        {
            var expectedNull = ints.IsNull(i) || longs.IsNull(i);
            Assert.That(result.IsNull(i), Is.EqualTo(expectedNull), $"null mask at {i} must be left-OR-right");
            if (!expectedNull)
                Assert.That(result.GetValue(i), Is.EqualTo(ints[i] + longs[i]),
                    $"value at {i} must match C# promoted addition");
        }
    }

    [Test]
    public void Evaluate_ScalarMixedType_UsesTypedPromotedPath()
    {
        // double column + int literal promotes to double
        var doubles = NivaraColumn<double>.CreateFromNullable(new double?[] { 1.5, 2.5, null });
        var input = new Dictionary<string, IColumn> { ["A"] = doubles };
        var evaluator = new ExpressionEvaluator();
        var expression = ColumnExpressions.Col("A") + 1;

        var result = evaluator.Evaluate(expression, input);

        Assert.That(evaluator.TypedPathEvaluationCount, Is.EqualTo(1), "column + int literal must use the typed promoted path");
        Assert.That(evaluator.BoxedPathEvaluationCount, Is.EqualTo(0));
        Assert.That(result.ElementType, Is.EqualTo(typeof(double)), "double column + int literal must promote to double");

        for (int i = 0; i < result.Length; i++)
        {
            var expectedNull = doubles.IsNull(i);
            Assert.That(result.IsNull(i), Is.EqualTo(expectedNull), $"null mask at {i} must propagate from the column");
            if (!expectedNull)
                Assert.That(result.GetValue(i), Is.EqualTo((double)doubles.GetValue(i)! + 1),
                    $"value at {i} must match C# promoted addition");
        }
    }

    [Test]
    public void Evaluate_MixedTypeNumericComparison_UsesTypedPromotedPath()
    {
        // double column vs int literal promotes to double and compares via the typed kernel
        var doubles = NivaraColumn<double>.CreateFromNullable(new double?[] { 1.5, 2.5, null, 4.0 });
        var input = new Dictionary<string, IColumn> { ["A"] = doubles };
        var evaluator = new ExpressionEvaluator();
        var expression = ColumnExpressions.Col("A") > 1;

        var result = evaluator.Evaluate(expression, input);

        Assert.That(evaluator.TypedPathEvaluationCount, Is.EqualTo(1), "double column vs int literal must use the typed promoted comparison path");
        Assert.That(evaluator.BoxedPathEvaluationCount, Is.EqualTo(0));
        Assert.That(result.ElementType, Is.EqualTo(typeof(bool)), "comparison result must be bool");

        for (int i = 0; i < result.Length; i++)
        {
            var expectedNull = doubles.IsNull(i);
            Assert.That(result.IsNull(i), Is.EqualTo(expectedNull), $"null mask at {i} must propagate from the column");
            if (!expectedNull)
                Assert.That(result.GetValue(i), Is.EqualTo((double)doubles.GetValue(i)! > 1),
                    $"value at {i} must match C# promoted comparison");
        }
    }

    [Test]
    public void Evaluate_DecimalInt_PromotesToDecimal()
    {
        // decimal + int promotes to decimal
        var decimals = NivaraColumn<decimal>.CreateFromNullable(new decimal?[] { 1.5m, null, 3.5m });
        var ints = NivaraColumn<int>.Create(new[] { 10, 20, 30 });
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = decimals,
            ["B"] = ints
        };
        var evaluator = new ExpressionEvaluator();
        var expression = ColumnExpressions.Col("A") + ColumnExpressions.Col("B");

        var result = evaluator.Evaluate(expression, input);

        Assert.That(evaluator.TypedPathEvaluationCount, Is.EqualTo(1));
        Assert.That(evaluator.BoxedPathEvaluationCount, Is.EqualTo(0));
        Assert.That(result.ElementType, Is.EqualTo(typeof(decimal)), "decimal + int must promote to decimal");

        for (int i = 0; i < result.Length; i++)
        {
            var expectedNull = decimals.IsNull(i) || ints.IsNull(i);
            Assert.That(result.IsNull(i), Is.EqualTo(expectedNull), $"null mask at {i} must be left-OR-right");
            if (!expectedNull)
                Assert.That(result.GetValue(i), Is.EqualTo(decimals[i] + ints[i]),
                    $"value at {i} must match C# promoted addition");
        }
    }

    [Test]
    public void Evaluate_ByteInt_PromotesToInt()
    {
        // byte + int promotes to int (C# integral promotion)
        var bytes = NivaraColumn<byte>.Create(new byte[] { 1, 2, 3 });
        var ints = NivaraColumn<int>.Create(new[] { 100, 200, 300 });
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = bytes,
            ["B"] = ints
        };
        var evaluator = new ExpressionEvaluator();
        var expression = ColumnExpressions.Col("A") + ColumnExpressions.Col("B");

        var result = evaluator.Evaluate(expression, input);

        Assert.That(evaluator.TypedPathEvaluationCount, Is.EqualTo(1));
        Assert.That(evaluator.BoxedPathEvaluationCount, Is.EqualTo(0));
        Assert.That(result.ElementType, Is.EqualTo(typeof(int)), "byte + int must promote to int");

        for (int i = 0; i < result.Length; i++)
        {
            Assert.That(result.IsNull(i), Is.False);
            Assert.That(result.GetValue(i), Is.EqualTo(bytes[i] + ints[i]), $"value at {i} must match C# promoted addition");
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

    [Test]
    public void Filter_GuidColumn_TypedAccessStillWorks()
    {
        // issue #104 regression: filtering must preserve the Guid element type so
        // typed GetColumn<Guid> still works after a Where
        var target = Guid.NewGuid();
        var ids = NivaraColumn<Guid>.CreateFromNullable(new Guid?[] { target, null, Guid.NewGuid() });
        using var frame = NivaraFrame.Create(("ID", ids));

        using var result = frame.AsQueryFrame()
            .Where(x => x["ID"] == (object)target)
            .Collect();

        Assert.That(result.RowCount, Is.EqualTo(1));
        var filtered = result.GetColumn<Guid>("ID");
        Assert.That(filtered[0], Is.EqualTo(target), "typed access must still work after filtering");
    }
}
