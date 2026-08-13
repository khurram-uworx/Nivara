using Nivara.Expressions;
using NUnit.Framework;

namespace Nivara.Tests.Query;

/// <summary>
/// Guardrails for the fused expression evaluator (POLARS-ROADMAP Phase 2).
/// These tests fail if the fused path is reverted to per-operator materialization or boxing:
/// - the fused evaluator is actually selected for fusable expressions;
/// - fused output carries the correct values and null masks;
/// - unsupported/opaque expressions throw instead of degrading to boxed evaluation.
/// The legacy per-operator <c>ExpressionEvaluator</c> was removed; expectations are asserted
/// against hand-computed values (see issue #152).
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
    public void Evaluate_ChainedArithmetic_ComputesCorrectValues_WithNullMasks()
    {
        var left = NivaraColumn.CreateFromNullable(new double?[] { 1.5, 2.5, null, 4.0 });
        var right = NivaraColumn.CreateFromNullable(new double?[] { null, 10.0, 30.0, 40.0 });
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = left,
            ["B"] = right
        };
        var expression = ColumnExpressions.Col("A") * 1.1 + 1000 - ColumnExpressions.Col("B");

        var fused = new FusedExpressionEvaluator();
        var result = fused.Evaluate(expression, input);

        Assert.That(fused.FusedPathEvaluationCount, Is.EqualTo(1), "chained arithmetic must run through the fused evaluator");
        Assert.That(fused.NodeTreePathEvaluationCount, Is.EqualTo(1), "null-bearing uniform arithmetic must run through the span kernel");
        AssertColumn(result, new double?[] { null, 992.75, null, 964.4 });
    }

    [Test]
    public void Evaluate_Comparison_WithNullsMaskedFalse()
    {
        var column = NivaraColumn.CreateFromNullable(new double?[] { 50.0, 250.0, null, 400.0 });
        var input = new Dictionary<string, IColumn> { ["A"] = column };
        var expression = ColumnExpressions.Col("A") > 100;

        var fused = new FusedExpressionEvaluator();
        var result = fused.Evaluate(expression, input);

        Assert.That(fused.FusedPathEvaluationCount, Is.EqualTo(1));
        Assert.That(fused.CompiledPathEvaluationCount, Is.EqualTo(1));
        Assert.That(result.ElementType, Is.EqualTo(typeof(bool)));
        Assert.That(result.IsNull(2), Is.True, "null operand must yield a masked null result");
        Assert.That(((NivaraColumn<bool>)result)[2], Is.False, "masked comparison position must hold false (SQL semantics)");
        Assert.That(((NivaraColumn<bool>)result)[0], Is.False, "50 > 100 is false");
        Assert.That(((NivaraColumn<bool>)result)[1], Is.True, "250 > 100 is true");
        Assert.That(((NivaraColumn<bool>)result)[3], Is.True, "400 > 100 is true");
    }

    [Test]
    public void Evaluate_MixedIntDouble_PromotesToDouble()
    {
        var ints = NivaraColumn.CreateFromNullable(new int?[] { 1, null, 3, 4 });
        var doubles = NivaraColumn.CreateFromNullable(new double?[] { 10.5, 20.5, null, 40.5 });
        var input = new Dictionary<string, IColumn>
        {
            ["I"] = ints,
            ["D"] = doubles
        };
        var expression = ColumnExpressions.Col("I") + ColumnExpressions.Col("D");

        var fused = new FusedExpressionEvaluator();
        var result = fused.Evaluate(expression, input);

        Assert.That(result.ElementType, Is.EqualTo(typeof(double)));
        AssertColumn(result, new double?[] { 11.5, null, null, 44.5 });
    }

    [Test]
    public void Evaluate_AndOrNot_PropagatesNullMasks()
    {
        var a = NivaraColumn.CreateFromNullable(new bool?[] { true, false, null, true });
        var b = NivaraColumn.CreateFromNullable(new bool?[] { false, false, true, null });
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = a,
            ["B"] = b
        };
        var and = new BinaryExpression(BinaryOperator.And, ColumnExpressions.Col("A"), ColumnExpressions.Col("B"));
        var or = new BinaryExpression(BinaryOperator.Or, ColumnExpressions.Col("A"), ColumnExpressions.Col("B"));
        var not = new NotExpression(ColumnExpressions.Col("A"));
        var fused = new FusedExpressionEvaluator();

        AssertColumn(fused.Evaluate(and, input), new bool?[] { false, false, null, null });
        AssertColumn(fused.Evaluate(or, input), new bool?[] { true, false, null, null });
        AssertColumn(fused.Evaluate(not, input), new bool?[] { false, true, null, false });
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
        var column = NivaraColumn.CreateFromNullable(new decimal?[] { 1.5m, null, 3.5m });
        var input = new Dictionary<string, IColumn> { ["A"] = column };
        var expression = ColumnExpressions.Col("A") * 2;
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(expression, input);

        Assert.That(result.ElementType, Is.EqualTo(typeof(decimal)));
        AssertColumn(result, new decimal?[] { 3.0m, null, 7.0m });
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
    public void Infer_ByteSameType_UnifiesToInt()
    {
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<byte>.Create(new byte[] { 1, 2, 3 }),
            ["B"] = NivaraColumn<byte>.Create(new byte[] { 4, 5, 6 })
        };
        var plan = ExpressionTypeInferer.TryInfer(ColumnExpressions.Col("A") + ColumnExpressions.Col("B"), input);

        Assert.That(plan, Is.Not.Null);
        Assert.That(plan.ResultType, Is.EqualTo(typeof(int)), "byte + byte must unify to int (C# rule 1)");
    }

    [Test]
    public void Evaluate_ByteSameType_PromotesToInt()
    {
        var left = NivaraColumn<byte>.Create(new byte[] { 10, 20, 30 });
        var right = NivaraColumn<byte>.Create(new byte[] { 1, 2, 3 });
        var input = new Dictionary<string, IColumn> { ["A"] = left, ["B"] = right };
        var expression = ColumnExpressions.Col("A") + ColumnExpressions.Col("B");

        var fused = new FusedExpressionEvaluator();
        var result = fused.Evaluate(expression, input);

        Assert.That(fused.FusedPathEvaluationCount, Is.EqualTo(1));
        Assert.That(result.ElementType, Is.EqualTo(typeof(int)), "byte + byte must produce a NivaraColumn<int>");
        AssertColumn(result, new int?[] { 11, 22, 33 });
    }

    [Test]
    public void Evaluate_ByteScalar_PromotesToInt()
    {
        var column = NivaraColumn<byte>.Create(new byte[] { 10, 20, 30 });
        var input = new Dictionary<string, IColumn> { ["A"] = column };
        var expression = ColumnExpressions.Col("A") * 2;

        var fused = new FusedExpressionEvaluator();
        var result = fused.Evaluate(expression, input);

        Assert.That(result.ElementType, Is.EqualTo(typeof(int)), "byte column * int scalar must produce int");
        AssertColumn(result, new int?[] { 20, 40, 60 });
    }

    [Test]
    public void Infer_CharSameType_UnifiesToInt()
    {
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<char>.Create(new char[] { (char)1, (char)2, (char)3 }),
            ["B"] = NivaraColumn<char>.Create(new char[] { (char)4, (char)5, (char)6 })
        };
        var plan = ExpressionTypeInferer.TryInfer(ColumnExpressions.Col("A") + ColumnExpressions.Col("B"), input);

        Assert.That(plan, Is.Not.Null);
        Assert.That(plan.ResultType, Is.EqualTo(typeof(int)), "char + char must unify to int (C# rule 1)");
    }

    [Test]
    public void Evaluate_CharSameType_PromotesToInt()
    {
        var left = NivaraColumn<char>.Create(new char[] { (char)10, (char)20, (char)30 });
        var right = NivaraColumn<char>.Create(new char[] { (char)1, (char)2, (char)3 });
        var input = new Dictionary<string, IColumn> { ["A"] = left, ["B"] = right };
        var expression = ColumnExpressions.Col("A") + ColumnExpressions.Col("B");

        var fused = new FusedExpressionEvaluator();
        var result = fused.Evaluate(expression, input);

        Assert.That(fused.FusedPathEvaluationCount, Is.EqualTo(1));
        Assert.That(result.ElementType, Is.EqualTo(typeof(int)), "char + char must produce a NivaraColumn<int>");
        AssertColumn(result, new int?[] { 11, 22, 33 });
    }

    [Test]
    public void Evaluate_CharScalar_PromotesToInt()
    {
        var column = NivaraColumn<char>.Create(new char[] { (char)10, (char)20, (char)30 });
        var input = new Dictionary<string, IColumn> { ["A"] = column };
        var expression = ColumnExpressions.Col("A") * 2;

        var fused = new FusedExpressionEvaluator();
        var result = fused.Evaluate(expression, input);

        Assert.That(result.ElementType, Is.EqualTo(typeof(int)), "char column * int scalar must produce int");
        AssertColumn(result, new int?[] { 20, 40, 60 });
    }

    [Test]
    public void Evaluate_UintByte_PromotesToUint()
    {
        var left = NivaraColumn<uint>.Create(new uint[] { 1u, 2u, 3u });
        var right = NivaraColumn<byte>.Create(new byte[] { 10, 20, 30 });
        var input = new Dictionary<string, IColumn> { ["A"] = left, ["B"] = right };
        var expression = ColumnExpressions.Col("A") + ColumnExpressions.Col("B");

        var fused = new FusedExpressionEvaluator();
        var result = fused.Evaluate(expression, input);

        Assert.That(result.ElementType, Is.EqualTo(typeof(uint)), "uint + byte must promote to uint (C# rule 7)");
        AssertColumn(result, new uint?[] { 11u, 22u, 33u });
    }

    [Test]
    public void Infer_UintSameType_UnifiesToUint()
    {
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<uint>.Create(new uint[] { 1u, 2u }),
            ["B"] = NivaraColumn<uint>.Create(new uint[] { 3u, 4u })
        };
        var plan = ExpressionTypeInferer.TryInfer(ColumnExpressions.Col("A") + ColumnExpressions.Col("B"), input);

        Assert.That(plan, Is.Not.Null);
        Assert.That(plan.ResultType, Is.EqualTo(typeof(uint)));
    }

    [Test]
    public void Evaluate_UintSameType_StaysUint()
    {
        var left = NivaraColumn<uint>.Create(new uint[] { 1u, 2u, 3u });
        var right = NivaraColumn<uint>.Create(new uint[] { 10u, 20u, 30u });
        var input = new Dictionary<string, IColumn> { ["A"] = left, ["B"] = right };
        var expression = ColumnExpressions.Col("A") + ColumnExpressions.Col("B");

        var fused = new FusedExpressionEvaluator();
        var result = fused.Evaluate(expression, input);

        Assert.That(result.ElementType, Is.EqualTo(typeof(uint)));
        AssertColumn(result, new uint?[] { 11u, 22u, 33u });
    }

    [Test]
    public void Evaluate_DecimalBinary_StaysDecimal()
    {
        var left = NivaraColumn<decimal>.Create(new decimal[] { 1.5m, 2.5m });
        var right = NivaraColumn<decimal>.Create(new decimal[] { 0.5m, 0.25m });
        var input = new Dictionary<string, IColumn> { ["A"] = left, ["B"] = right };
        var expression = ColumnExpressions.Col("A") + ColumnExpressions.Col("B");

        var fused = new FusedExpressionEvaluator();
        var result = fused.Evaluate(expression, input);

        Assert.That(result.ElementType, Is.EqualTo(typeof(decimal)));
        AssertColumn(result, new decimal?[] { 2.0m, 2.75m });
    }

    [Test]
    public void Evaluate_ModuloBinary_ComputesRemainder_WithNullMasks()
    {
        var left = NivaraColumn.CreateFromNullable(new int?[] { 10, null, 25, 30 });
        var right = NivaraColumn.CreateFromNullable(new int?[] { 3, 4, 7, 9 });
        var input = new Dictionary<string, IColumn> { ["A"] = left, ["B"] = right };
        var expression = ColumnExpressions.Col("A") % ColumnExpressions.Col("B");

        var fused = new FusedExpressionEvaluator();
        var result = fused.Evaluate(expression, input);

        Assert.That(fused.FusedPathEvaluationCount, Is.EqualTo(1));
        Assert.That(fused.NodeTreePathEvaluationCount, Is.EqualTo(1), "null-bearing uniform modulo must run through the span kernel");
        Assert.That(result.ElementType, Is.EqualTo(typeof(int)));
        AssertColumn(result, new int?[] { 1, null, 4, 3 });
    }

    [Test]
    public void Evaluate_ModuloScalar_ComputesRemainder()
    {
        var column = NivaraColumn<double>.Create(new[] { 5.5, 10.0, 3.25 });
        var input = new Dictionary<string, IColumn> { ["A"] = column };
        var expression = ColumnExpressions.Col("A") % 2.0;

        var fused = new FusedExpressionEvaluator();
        var result = fused.Evaluate(expression, input);

        Assert.That(result.ElementType, Is.EqualTo(typeof(double)));
        AssertColumn(result, new double?[] { 1.5, 0.0, 1.25 });
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
        var column = NivaraColumn.CreateFromNullable(new double?[] { 2.0, null, 6.0 });
        var columnRef = new ColumnReference("A");
        var binding = new FusedColumnBinding(columnRef, column);
        var expression = columnRef * 2 + 3;

        var result = (NivaraColumn<double>)FusedKernel.Evaluate<double>(expression, new[] { binding });

        Assert.That(result.IsNull(1), Is.True);
        Assert.That(result[0], Is.EqualTo(7.0));
        Assert.That(result[2], Is.EqualTo(15.0));
    }

    [Test]
    public void SpanKernel_TwoLeafNullMasks_OrPropagatesToOutput()
    {
        var leftRef = new ColumnReference("A");
        var rightRef = new ColumnReference("B");
        var left = NivaraColumn.CreateFromNullable(new double?[] { 2.0, null, 6.0 });
        var right = NivaraColumn.CreateFromNullable(new double?[] { null, 5.0, 7.0 });
        var expression = leftRef + rightRef;
        var leaves = new[]
        {
            new FusedColumnBinding(leftRef, left),
            new FusedColumnBinding(rightRef, right)
        };

        var output = new double[3];
        var outputMask = new bool[3];
        FusedKernel.Execute<double>(expression, leaves,
            new[] { left.Storage.Data, right.Storage.Data },
            new[]
            {
                left.Storage.NullMaskMemory ?? default,
                right.Storage.NullMaskMemory ?? default
            },
            output, outputMask);

        Assert.That(outputMask[0], Is.True, "right leaf null at 0 must propagate");
        Assert.That(outputMask[1], Is.True, "left leaf null at 1 must propagate");
        Assert.That(outputMask[2], Is.False, "position 2 has no nulls");
        Assert.That(output[0], Is.EqualTo(0.0), "masked positions write default(T)");
        Assert.That(output[2], Is.EqualTo(13.0));
    }

    [Test]
    public void Evaluate_NullBearingUniformPlan_RoutesToSpanKernel()
    {
        var column = NivaraColumn.CreateFromNullable(new double?[] { 1.0, null, 3.0 });
        var input = new Dictionary<string, IColumn> { ["A"] = column };
        var expression = ColumnExpressions.Col("A") * 2.0;

        var fused = new FusedExpressionEvaluator();
        var result = fused.Evaluate(expression, input);

        Assert.That(fused.NodeTreePathEvaluationCount, Is.EqualTo(1), "null-bearing uniform arithmetic must use the span kernel");
        Assert.That(fused.CompiledPathEvaluationCount, Is.EqualTo(0), "null-bearing uniform arithmetic must not use the compiled target");
        AssertColumn(result, new double?[] { 2.0, null, 6.0 });
    }

    [Test]
    public void Evaluate_NullFreeUniformPlan_StaysOnCompiledPath()
    {
        var column = NivaraColumn<double>.Create(new[] { 1.0, 2.0, 3.0 });
        var input = new Dictionary<string, IColumn> { ["A"] = column };
        var expression = ColumnExpressions.Col("A") * 2.0;

        var fused = new FusedExpressionEvaluator();
        var result = fused.Evaluate(expression, input);

        Assert.That(fused.CompiledPathEvaluationCount, Is.EqualTo(1), "null-free uniform arithmetic must stay on the compiled path");
        Assert.That(fused.NodeTreePathEvaluationCount, Is.EqualTo(0), "null-free uniform arithmetic must not route to the span kernel");
        AssertColumn(result, new double?[] { 2.0, 4.0, 6.0 });
    }

    static void AssertColumn<T>(IColumn actual, IReadOnlyList<T?> expected)
        where T : struct
    {
        Assert.That(actual.Length, Is.EqualTo(expected.Count));
        Assert.That(actual.ElementType, Is.EqualTo(typeof(T)));
        for (int i = 0; i < expected.Count; i++)
        {
            Assert.That(actual.IsNull(i), Is.EqualTo(!expected[i].HasValue), $"null mask at {i}");
            if (expected[i].HasValue)
                Assert.That(actual.GetValue(i), Is.EqualTo(expected[i]!.Value), $"value at {i}");
        }
    }

    [Test]
    public void Evaluate_LitHalfConstant_ProducesTypedColumn()
    {
        var input = new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(new[] { 1, 2, 3 }) };
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.Lit((Half)1.5f), input);

        Assert.That(result, Is.InstanceOf<NivaraColumn<Half>>(), "literal column must stay typed");
        AssertColumn(result, new Half?[] { (Half)1.5f, (Half)1.5f, (Half)1.5f });
    }

    [Test]
    public void Evaluate_LitNIntConstant_ProducesTypedColumn()
    {
        var input = new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(new[] { 1, 2 }) };
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.Lit((nint)7), input);

        Assert.That(result, Is.InstanceOf<NivaraColumn<nint>>(), "literal column must stay typed");
        AssertColumn(result, new nint?[] { 7, 7 });
    }

    [Test]
    public void Evaluate_LitUIntConstant_ProducesTypedColumn()
    {
        var input = new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(new[] { 1, 2, 3 }) };
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.Lit(7u), input);

        Assert.That(result, Is.InstanceOf<NivaraColumn<uint>>(), "literal column must stay typed");
        AssertColumn(result, new uint?[] { 7u, 7u, 7u });
    }

    [Test]
    public void Evaluate_LitDateOnlyConstant_ProducesTypedColumn()
    {
        var date = new DateOnly(2024, 5, 1);
        var input = new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(new[] { 1, 2 }) };
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.Lit(date), input);

        Assert.That(result, Is.InstanceOf<NivaraColumn<DateOnly>>(), "literal column must stay typed");
        Assert.That(result.Length, Is.EqualTo(2));
        Assert.That(result.GetValue(0), Is.EqualTo(date));
        Assert.That(result.GetValue(1), Is.EqualTo(date));
    }
}
