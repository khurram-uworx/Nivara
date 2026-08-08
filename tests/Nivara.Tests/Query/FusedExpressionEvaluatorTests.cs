using Nivara.Expressions;
using Nivara.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Query;

/// <summary>
/// Guardrails for the fused expression evaluator (POLARS-ROADMAP Phase 2).
/// These tests fail if the fused path is reverted to per-operator materialization or boxing:
/// - the fused evaluator is actually selected for fusable expressions;
/// - fused output is bit-equivalent to the legacy evaluator, including null masks;
/// - unsupported/opaque expressions throw instead of degrading to boxed evaluation.
/// </summary>
[TestFixture]
public class FusedExpressionEvaluatorTests
{
    [Test]
    public void Infer_ChainedArithmetic_UnifiesToDouble_WithLeafBindings()
    {
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<double>.Create(new[] { 1.0, 2.0, 3.0 }),
            ["B"] = NivaraColumn<double>.Create(new[] { 10.0, 20.0, 30.0 })
        };
        var expression = ColumnExpressions.Col("A") * 1.1 + 1000 - ColumnExpressions.Col("B");

        var plan = ExpressionTypeInferer.TryInfer(expression, input);

        Assert.That(plan, Is.Not.Null);
        Assert.That(plan.ResultType, Is.EqualTo(typeof(double)));
        Assert.That(plan.IsGenericMath, Is.True);
        Assert.That(plan.HasNulls, Is.False);
        Assert.That(plan.Columns.Count, Is.EqualTo(2));
    }

    [Test]
    public void Infer_ObjectTypedColumn_IsNotFusable()
    {
        var input = new Dictionary<string, IColumn>
        {
            ["O"] = NivaraColumn<object?>.Create(new object?[] { "a", 1, null })
        };
        var expression = ColumnExpressions.Col("O") + ColumnExpressions.Col("O");

        var plan = ExpressionTypeInferer.TryInfer(expression, input);

        Assert.That(plan, Is.Null);
    }

    [Test]
    public void Evaluate_ChainedArithmetic_MatchesLegacyReference_WithNullMasks()
    {
        var left = NivaraColumn<double>.CreateFromNullable(new double?[] { 1.5, 2.5, null, 4.0 });
        var right = NivaraColumn<double>.CreateFromNullable(new double?[] { null, 10.0, 30.0, 40.0 });
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = left,
            ["B"] = right
        };
        var expression = ColumnExpressions.Col("A") * 1.1 + 1000 - ColumnExpressions.Col("B");

        var legacy = new ExpressionEvaluator().Evaluate(expression, input);
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(expression, input);

        Assert.That(fused.FusedPathEvaluationCount, Is.EqualTo(1), "chained arithmetic must run through the fused evaluator");
        Assert.That(fused.CompiledPathEvaluationCount, Is.EqualTo(1), "chained arithmetic must use the compiled target");
        AssertEquivalent(legacy, result);
    }

    [Test]
    public void Evaluate_Comparison_MatchesLegacyReference_WithNullsMaskedFalse()
    {
        var column = NivaraColumn<double>.CreateFromNullable(new double?[] { 50.0, 250.0, null, 400.0 });
        var input = new Dictionary<string, IColumn> { ["A"] = column };
        var expression = ColumnExpressions.Col("A") > 100;

        var legacy = new ExpressionEvaluator().Evaluate(expression, input);
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(expression, input);

        Assert.That(fused.FusedPathEvaluationCount, Is.EqualTo(1));
        Assert.That(fused.CompiledPathEvaluationCount, Is.EqualTo(1));
        AssertEquivalent(legacy, result);
        Assert.That(result.ElementType, Is.EqualTo(typeof(bool)));
        Assert.That(result.IsNull(2), Is.True, "null operand must yield a masked null result");
        Assert.That(((NivaraColumn<bool>)result)[2], Is.False, "masked comparison position must hold false (SQL semantics)");
    }

    [Test]
    public void Evaluate_MixedIntDouble_PromotesToDouble()
    {
        var ints = NivaraColumn<int>.CreateFromNullable(new int?[] { 1, null, 3, 4 });
        var doubles = NivaraColumn<double>.CreateFromNullable(new double?[] { 10.5, 20.5, null, 40.5 });
        var input = new Dictionary<string, IColumn>
        {
            ["I"] = ints,
            ["D"] = doubles
        };
        var expression = ColumnExpressions.Col("I") + ColumnExpressions.Col("D");

        var legacy = new ExpressionEvaluator().Evaluate(expression, input);
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(expression, input);

        Assert.That(result.ElementType, Is.EqualTo(typeof(double)));
        AssertEquivalent(legacy, result);
    }

    [Test]
    public void Evaluate_AndOrNot_MatchesLegacyReference()
    {
        var a = NivaraColumn<bool>.CreateFromNullable(new bool?[] { true, false, null, true });
        var b = NivaraColumn<bool>.CreateFromNullable(new bool?[] { false, false, true, null });
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = a,
            ["B"] = b
        };
        var and = new BinaryExpression(BinaryOperator.And, ColumnExpressions.Col("A"), ColumnExpressions.Col("B"));
        var or = new BinaryExpression(BinaryOperator.Or, ColumnExpressions.Col("A"), ColumnExpressions.Col("B"));
        var not = new NotExpression(ColumnExpressions.Col("A"));
        var fused = new FusedExpressionEvaluator();

        AssertEquivalent(new ExpressionEvaluator().Evaluate(and, input), fused.Evaluate(and, input));
        AssertEquivalent(new ExpressionEvaluator().Evaluate(or, input), fused.Evaluate(or, input));
        AssertEquivalent(new ExpressionEvaluator().Evaluate(not, input), fused.Evaluate(not, input));
    }

    [Test]
    public void Evaluate_DirectColumnReference_ReturnsTheColumnItself()
    {
        var column = NivaraColumn<double>.Create(new[] { 1.0, 2.0 });
        var input = new Dictionary<string, IColumn> { ["A"] = column };
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.Col("A"), input);

        Assert.That(fused.CompiledPathEvaluationCount, Is.EqualTo(0), "passthrough must not build a kernel");
        Assert.That(ReferenceEquals(result, column), Is.True);
    }

    [Test]
    public void Evaluate_DecimalColumn_RunsThroughFusedPath()
    {
        var column = NivaraColumn<decimal>.CreateFromNullable(new decimal?[] { 1.5m, null, 3.5m });
        var input = new Dictionary<string, IColumn> { ["A"] = column };
        var expression = ColumnExpressions.Col("A") * 2;
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(expression, input);

        Assert.That(result.ElementType, Is.EqualTo(typeof(decimal)));
        AssertEquivalent(new ExpressionEvaluator().Evaluate(expression, input), result);
    }

    [Test]
    public void Evaluate_HalfSameType_RunsThroughFusedPath()
    {
        var column = NivaraColumn<Half>.Create(new[] { (Half)1.5f, (Half)2.5f, (Half)3.5f });
        var input = new Dictionary<string, IColumn> { ["A"] = column };
        var expression = ColumnExpressions.Col("A") + ColumnExpressions.Col("A");
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(expression, input);

        Assert.That(fused.FusedPathEvaluationCount, Is.EqualTo(1));
        Assert.That(result.ElementType, Is.EqualTo(typeof(Half)));
        Assert.That((float)(Half)result.GetValue(0)!, Is.EqualTo(3.0f));
    }

    [Test]
    public void Evaluate_UnsupportedMixedComparison_Throws()
    {
        var strings = NivaraColumn<string>.Create(new[] { "a", "b" });
        var dates = NivaraColumn<DateTime>.Create(new[] { DateTime.UtcNow, DateTime.UtcNow.AddDays(1) });
        var input = new Dictionary<string, IColumn>
        {
            ["S"] = strings,
            ["D"] = dates
        };
        var expression = ColumnExpressions.Col("S") == ColumnExpressions.Col("D");

        Assert.Throws<Nivara.Exceptions.QueryExecutionException>(() =>
            new FusedExpressionEvaluator().Evaluate(expression, input));
    }

    [Test]
    public void Evaluate_ObjectTypedArithmetic_Throws()
    {
        var column = NivaraColumn<object?>.Create(new object?[] { "a", 1, null });
        var input = new Dictionary<string, IColumn> { ["O"] = column };
        var expression = ColumnExpressions.Col("O") + ColumnExpressions.Col("O");

        Assert.Throws<Nivara.Exceptions.QueryExecutionException>(() =>
            new FusedExpressionEvaluator().Evaluate(expression, input));
    }

    [Test]
    public void NodeTreeKernel_UniformGenericMath_MatchesReference_WithMask()
    {
        var column = NivaraColumn<double>.CreateFromNullable(new double?[] { 2.0, null, 6.0 });
        var columnRef = new ColumnReference("A");
        var binding = new FusedColumnBinding(columnRef, column);
        var expression = columnRef * 2 + 3;

        var result = (NivaraColumn<double>)FusedKernel.Evaluate<double>(expression, new[] { binding }, new[] { false, true, false });

        Assert.That(result.IsNull(1), Is.True);
        Assert.That(result[0], Is.EqualTo(7.0));
        Assert.That(result[2], Is.EqualTo(15.0));
    }

    static void AssertEquivalent(IColumn expected, IColumn actual)
    {
        Assert.That(actual.Length, Is.EqualTo(expected.Length));
        Assert.That(actual.ElementType, Is.EqualTo(expected.ElementType));
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.That(actual.IsNull(i), Is.EqualTo(expected.IsNull(i)), $"null mask at {i} must match legacy evaluator");
            if (!expected.IsNull(i))
                Assert.That(actual.GetValue(i), Is.EqualTo(expected.GetValue(i)), $"value at {i} must be bit-equivalent to legacy evaluator");
        }
    }
}
