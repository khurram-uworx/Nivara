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
        Assert.That(fused.SpanKernelPathEvaluationCount, Is.EqualTo(1), "null-bearing uniform arithmetic must run through the span kernel");
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
        Assert.That(fused.SpanKernelPathEvaluationCount, Is.EqualTo(1), "null-bearing uniform modulo must run through the span kernel");
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

        Assert.That(fused.SpanKernelPathEvaluationCount, Is.EqualTo(1), "null-bearing uniform arithmetic must use the span kernel");
        Assert.That(fused.CompiledPathEvaluationCount, Is.EqualTo(0), "null-bearing uniform arithmetic must not use the compiled target");
        AssertColumn(result, new double?[] { 2.0, null, 6.0 });
    }

    [Test]
    public void Evaluate_NullFreeSingleOp_DispatchesToTensorPrimitives()
    {
        var column = NivaraColumn<double>.Create(new[] { 1.0, 2.0, 3.0 });
        var input = new Dictionary<string, IColumn> { ["A"] = column };
        var expression = ColumnExpressions.Col("A") * 2.0;

        var fused = new FusedExpressionEvaluator();
        var result = fused.Evaluate(expression, input);

        Assert.That(fused.TensorPrimitivesPathEvaluationCount, Is.EqualTo(1), "null-free single-op arithmetic must dispatch to TensorPrimitives");
        Assert.That(fused.CompiledPathEvaluationCount, Is.EqualTo(0), "null-free single-op arithmetic must not build the compiled delegate");
        Assert.That(fused.SpanKernelPathEvaluationCount, Is.EqualTo(0), "null-free single-op arithmetic must not route to the span kernel");
        AssertColumn(result, new double?[] { 2.0, 4.0, 6.0 });
    }

    [Test]
    public void Evaluate_NullFreeSingleOpModulo_StaysOnCompiledPath()
    {
        var column = NivaraColumn<double>.Create(new[] { 5.5, 10.0, 3.25 });
        var input = new Dictionary<string, IColumn> { ["A"] = column };
        var expression = ColumnExpressions.Col("A") % 2.0;

        var fused = new FusedExpressionEvaluator();
        var result = fused.Evaluate(expression, input);

        Assert.That(fused.CompiledPathEvaluationCount, Is.EqualTo(1), "null-free modulo has no TensorPrimitives dispatch and must stay on the compiled path");
        Assert.That(fused.TensorPrimitivesPathEvaluationCount, Is.EqualTo(0), "modulo is not a TensorPrimitives candidate");
        AssertColumn(result, new double?[] { 1.5, 0.0, 1.25 });
    }

    [Test]
    public void EvaluateChunked_NullFreeChain_MatchesWholeColumn()
    {
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<double>.Create(Enumerable.Range(0, 2000).Select(i => (double)i).ToArray()),
            ["B"] = NivaraColumn<double>.Create(Enumerable.Range(0, 2000).Select(i => (double)(i * 3)).ToArray())
        };
        var expression = ColumnExpressions.Col("A") * 1.1 + 1000 - ColumnExpressions.Col("B");

        var fused = new FusedExpressionEvaluator();
        AssertChunkedMatchesWhole(fused, expression, input, new[] { 1, 2, 3, 511, 512, 1024, 1999, 2000 });

        Assert.That(fused.FusedPathEvaluationCount, Is.EqualTo(9), "whole + 8 chunk sizes must all run through the fused evaluator");
        Assert.That(fused.CompiledPathEvaluationCount, Is.EqualTo(9), "null-free chains must run through the compiled target");
    }

    [Test]
    public void EvaluateChunked_SingleOp_MatchesWholeColumn()
    {
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<double>.Create(Enumerable.Range(0, 2000).Select(i => (double)i).ToArray())
        };
        var expression = ColumnExpressions.Col("A") * 2.0;

        var fused = new FusedExpressionEvaluator();
        AssertChunkedMatchesWhole(fused, expression, input, new[] { 1, 2, 3, 511, 512, 1024, 1999, 2000 });

        Assert.That(fused.TensorPrimitivesPathEvaluationCount, Is.EqualTo(9), "single-op plans must keep dispatching to TensorPrimitives when chunked");
        Assert.That(fused.CompiledPathEvaluationCount, Is.EqualTo(0));
    }

    [Test]
    public void EvaluateChunked_NullBearingChain_MatchesWholeColumn()
    {
        var values = new double?[2000];
        for (int i = 0; i < values.Length; i++)
            values[i] = i % 97 == 0 ? null : (double)i;
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn.CreateFromNullable(values),
            ["B"] = NivaraColumn<double>.Create(Enumerable.Range(0, 2000).Select(i => (double)(i * 3)).ToArray())
        };
        var expression = ColumnExpressions.Col("A") * 1.1 + 1000 - ColumnExpressions.Col("B");

        var fused = new FusedExpressionEvaluator();
        AssertChunkedMatchesWhole(fused, expression, input, new[] { 1, 2, 3, 511, 512, 1024, 1999, 2000 });

        Assert.That(fused.SpanKernelPathEvaluationCount, Is.EqualTo(9), "null-bearing chains must keep routing to the span kernel when chunked");
        Assert.That(fused.CompiledPathEvaluationCount, Is.EqualTo(0));
    }

    /// <summary>
    /// Asserts chunked evaluation of the expression is bit-identical to whole-column evaluation
    /// (values and null masks) for every requested chunk size.
    /// </summary>
    static void AssertChunkedMatchesWhole(FusedExpressionEvaluator fused, ColumnExpression expression, IReadOnlyDictionary<string, IColumn> input, IReadOnlyList<int> chunkSizes)
    {
        var whole = fused.Evaluate(expression, input);
        foreach (var chunkSize in chunkSizes)
        {
            var chunked = fused.EvaluateChunked(expression, input, chunkSize);
            Assert.That(chunked.Length, Is.EqualTo(whole.Length), $"chunkSize {chunkSize}: length must match");
            for (int i = 0; i < whole.Length; i++)
            {
                Assert.That(chunked.IsNull(i), Is.EqualTo(whole.IsNull(i)), $"chunkSize {chunkSize}: null mask at {i}");
                if (!whole.IsNull(i))
                    Assert.That(chunked.GetValue(i), Is.EqualTo(whole.GetValue(i)), $"chunkSize {chunkSize}: value at {i}");
            }
        }
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

    // ── #250: native-size and 128-bit integer promotions in fused expressions ──

    [Test]
    public void Evaluate_NIntColumnPlusIntScalar_ProducesNIntColumn()
    {
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<nint>.Create(new nint[] { 10, 15, 20 })
        };
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.Col("A") + 5, input);

        Assert.That(result, Is.InstanceOf<NivaraColumn<nint>>(), "nint + int must stay nint");
        AssertColumn(result, new nint?[] { 15, 20, 25 });
    }

    [Test]
    public void Evaluate_NIntColumnPlusUIntScalar_PromotesToLong()
    {
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<nint>.Create(new nint[] { 10, 15, 20 })
        };
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.Col("A") + 5u, input);

        Assert.That(result, Is.InstanceOf<NivaraColumn<long>>(), "nint + uint must promote to long");
        AssertColumn(result, new long?[] { 15, 20, 25 });
    }

    [Test]
    public void Evaluate_NIntColumnPlusULongScalar_PromotesToDouble()
    {
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<nint>.Create(new nint[] { 10, 15, 20 })
        };
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.Col("A") + (ulong)5, input);

        Assert.That(result, Is.InstanceOf<NivaraColumn<double>>(), "nint + ulong is a C# error pair -> safe superset double");
        AssertColumn(result, new double?[] { 15, 20, 25 });
    }

    [Test]
    public void Evaluate_NIntColumnPlusNIntScalar_ProducesNIntColumn()
    {
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<nint>.Create(new nint[] { 10, 15, 20 })
        };
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.Col("A") + ColumnExpressions.Lit((nint)5), input);

        Assert.That(result, Is.InstanceOf<NivaraColumn<nint>>());
        AssertColumn(result, new nint?[] { 15, 20, 25 });
    }

    [Test]
    public void Evaluate_Int128ColumnPlusIntScalar_ProducesInt128Column()
    {
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<Int128>.Create(new Int128[] { 10, 15, 20 })
        };
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.Col("A") + 5, input);

        Assert.That(result, Is.InstanceOf<NivaraColumn<Int128>>(), "Int128 + int must stay Int128");
        AssertColumn(result, new Int128?[] { 15, 20, 25 });
    }

    [Test]
    public void Evaluate_UInt128ColumnPlusIntScalar_PromotesToDouble()
    {
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<UInt128>.Create(new UInt128[] { 10, 15, 20 })
        };
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.Col("A") + 5, input);

        Assert.That(result, Is.InstanceOf<NivaraColumn<double>>(), "UInt128 + int is a C# error pair -> safe superset double");
        AssertColumn(result, new double?[] { 15, 20, 25 });
    }

    [Test]
    public void Evaluate_HalfColumnPlusIntScalar_PromotesToDouble()
    {
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn<Half>.Create(new Half[] { (Half)1, (Half)2, (Half)3 })
        };
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.Col("A") + 1, input);

        Assert.That(result, Is.InstanceOf<NivaraColumn<double>>(), "Half + int is a C# error pair -> safe superset double");
        AssertColumn(result, new double?[] { 2, 3, 4 });
    }

    [Test]
    public void Evaluate_NIntColumnWithIntScalar_NullBearing_RunsThroughFusedPath()
    {
        var input = new Dictionary<string, IColumn>
        {
            ["A"] = NivaraColumn.CreateFromNullable(new nint?[] { 10, null, 20 })
        };
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.Col("A") + 5, input);

        Assert.That(result, Is.InstanceOf<NivaraColumn<nint>>());
        AssertColumn(result, new nint?[] { 15, null, 25 });
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

    // ── #249: literal-only plans must constant-fold instead of throwing ──

    [Test]
    public void Evaluate_LiteralOnlyArithmetic_ProducesConstantColumn()
    {
        var input = new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(new[] { 1, 2, 3 }) };
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.Lit(2) * 2, input);

        Assert.That(fused.FusedPathEvaluationCount, Is.EqualTo(1), "literal-only plan must run through the fused evaluator");
        Assert.That(fused.CompiledPathEvaluationCount, Is.EqualTo(1), "literal-only plan must run through the compiled target");
        Assert.That(result.ElementType, Is.EqualTo(typeof(int)));
        AssertColumn(result, new int?[] { 4, 4, 4 });
    }

    [Test]
    public void Evaluate_LiteralOnlyMixedNumeric_PromotesCorrectly()
    {
        var input = new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(new[] { 1, 2 }) };
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.Lit(2.5) * 2, input);

        Assert.That(result.ElementType, Is.EqualTo(typeof(double)), "2.5 * 2 must promote to double");
        AssertColumn(result, new double?[] { 5.0, 5.0 });
    }

    [Test]
    public void Evaluate_LiteralOnlyComparison_ProducesConstantBoolColumn()
    {
        var input = new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(new[] { 1, 2 }) };
        var fused = new FusedExpressionEvaluator();

        var strings = fused.Evaluate(ColumnExpressions.Lit("a") == ColumnExpressions.Lit("b"), input);
        var numbers = fused.Evaluate(ColumnExpressions.Lit(1) > ColumnExpressions.Lit(2), input);

        AssertColumn(strings, new bool?[] { false, false });
        AssertColumn(numbers, new bool?[] { false, false });
    }

    [Test]
    public void Evaluate_LiteralOnlyPlan_WithEmptyInput_ProducesLengthOneColumn()
    {
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.Lit(2) * 2, new Dictionary<string, IColumn>());

        Assert.That(result.Length, Is.EqualTo(1), "no input columns means a single-element constant column");
        AssertColumn(result, new int?[] { 4 });
    }

    // ── #246: literal runtime type must be part of the plan signature ──

    [Test]
    public void Infer_LiteralTypesWithSameText_HaveDistinctSignatures()
    {
        var input = new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(new[] { 1, 2 }) };

        var floatSig = ExpressionTypeInferer.TryInfer(ColumnExpressions.Col("A") + 0.1f, input)!.Signature;
        var doubleSig = ExpressionTypeInferer.TryInfer(ColumnExpressions.Col("A") + 0.1, input)!.Signature;
        var decimalSig = ExpressionTypeInferer.TryInfer(ColumnExpressions.Col("A") + 1.1m, input)!.Signature;
        var doubleSig2 = ExpressionTypeInferer.TryInfer(ColumnExpressions.Col("A") + 1.1, input)!.Signature;
        var intSig = ExpressionTypeInferer.TryInfer(ColumnExpressions.Col("A") + 7, input)!.Signature;
        var nintSig = ExpressionTypeInferer.TryInfer(ColumnExpressions.Col("A") + (nint)7, input)!.Signature;

        Assert.That(floatSig, Is.Not.EqualTo(doubleSig), "0.1f and 0.1 must not share a signature");
        Assert.That(decimalSig, Is.Not.EqualTo(doubleSig2), "1.1m and 1.1 must not share a signature");
        Assert.That(decimalSig, Is.Not.EqualTo(floatSig));
        Assert.That(intSig, Is.Not.EqualTo(nintSig), "int and nint literals must not share a signature");
    }

    [Test]
    public void Evaluate_FloatAndDoubleLiterals_ProduceTypedResults()
    {
        var input = new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(new[] { 1, 2, 3 }) };
        var fused = new FusedExpressionEvaluator();

        var floatResult = fused.Evaluate(ColumnExpressions.Col("A") + 0.1f, input);
        var doubleResult = fused.Evaluate(ColumnExpressions.Col("A") + 0.1, input);

        Assert.That(floatResult.ElementType, Is.EqualTo(typeof(float)));
        Assert.That(doubleResult.ElementType, Is.EqualTo(typeof(double)));
        AssertColumn(floatResult, new float?[] { 1.1f, 2.1f, 3.1f });
        AssertColumn(doubleResult, new double?[] { 1.1, 2.1, 3.1 });
    }

    [Test]
    public void Evaluate_DecimalAndDoubleLiterals_ProduceTypedResults()
    {
        var input = new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(new[] { 1, 2 }) };
        var fused = new FusedExpressionEvaluator();

        var decimalResult = fused.Evaluate(ColumnExpressions.Col("A") + 1.1m, input);
        var doubleResult = fused.Evaluate(ColumnExpressions.Col("A") + 1.1, input);

        Assert.That(decimalResult.ElementType, Is.EqualTo(typeof(decimal)));
        Assert.That(doubleResult.ElementType, Is.EqualTo(typeof(double)));
        AssertColumn(decimalResult, new decimal?[] { 2.1m, 3.1m });
        AssertColumn(doubleResult, new double?[] { 2.1, 3.1 });
    }

    // ── #247: compiled path must short-circuit masked positions ──

    [Test]
    public void Evaluate_CompiledMaskedDivide_DoesNotThrow_AndMasksToDefault()
    {
        // decimal is not generic math, so this plan routes to the compiled delegate. The right
        // leaf carries a null, whose backing storage is default(int) = 0 — dividing by it must
        // not throw, and masked positions must hold default(decimal), not a computed value.
        var left = NivaraColumn<decimal>.Create(new decimal[] { 10m, 20m, 30m, 40m });
        var right = NivaraColumn.CreateFromNullable(new int?[] { 2, null, 4, null });
        var input = new Dictionary<string, IColumn> { ["A"] = left, ["B"] = right };
        var expression = ColumnExpressions.Col("A") / ColumnExpressions.Col("B");

        var fused = new FusedExpressionEvaluator();
        var result = fused.Evaluate(expression, input);

        Assert.That(fused.CompiledPathEvaluationCount, Is.EqualTo(1), "decimal plans must run through the compiled target");
        AssertColumn(result, new decimal?[] { 5m, null, 7.5m, null });
        Assert.That(((NivaraColumn<decimal>)result)[1], Is.EqualTo(0m), "masked position must hold default(decimal)");
        Assert.That(((NivaraColumn<decimal>)result)[3], Is.EqualTo(0m), "masked position must hold default(decimal)");
    }

    [Test]
    public void EvaluateChunked_CompiledMaskedDivide_MatchesWholeColumn()
    {
        var left = NivaraColumn<decimal>.Create(new decimal[] { 10m, 20m, 30m, 40m, 50m, 60m, 70m, 80m });
        var right = NivaraColumn.CreateFromNullable(new int?[] { 2, null, 4, null, 5, null, 7, 8 });
        var input = new Dictionary<string, IColumn> { ["A"] = left, ["B"] = right };
        var expression = ColumnExpressions.Col("A") / ColumnExpressions.Col("B");

        var fused = new FusedExpressionEvaluator();
        AssertChunkedMatchesWhole(fused, expression, input, new[] { 2, 3, 5 });

        Assert.That(fused.CompiledPathEvaluationCount, Is.EqualTo(4), "whole + 3 chunk sizes must all route to the compiled target");
    }

    [Test]
    public void Evaluate_CompiledAndSpanBackends_MaskedDivergence_IsClosed()
    {
        // The span kernel (uniform generic math, null-bearing) already short-circuits masked
        // positions to default(T). The compiled path (decimal, not generic math) must now agree
        // on both the null mask and the raw masked backing values.
        var spanLeft = NivaraColumn.CreateFromNullable(new double?[] { 10.0, 20.0, 30.0, 40.0 });
        var spanRight = NivaraColumn.CreateFromNullable(new double?[] { 2.0, null, 4.0, null });
        var spanInput = new Dictionary<string, IColumn> { ["A"] = spanLeft, ["B"] = spanRight };
        var spanFused = new FusedExpressionEvaluator();
        var spanResult = spanFused.Evaluate(ColumnExpressions.Col("A") / ColumnExpressions.Col("B"), spanInput);
        Assert.That(spanFused.SpanKernelPathEvaluationCount, Is.EqualTo(1), "double plans with nulls must route to the span kernel");

        var compiledLeft = NivaraColumn<decimal>.Create(new decimal[] { 10m, 20m, 30m, 40m });
        var compiledRight = NivaraColumn.CreateFromNullable(new int?[] { 2, null, 4, null });
        var compiledInput = new Dictionary<string, IColumn> { ["A"] = compiledLeft, ["B"] = compiledRight };
        var compiledFused = new FusedExpressionEvaluator();
        var compiledResult = compiledFused.Evaluate(ColumnExpressions.Col("A") / ColumnExpressions.Col("B"), compiledInput);
        Assert.That(compiledFused.CompiledPathEvaluationCount, Is.EqualTo(1), "decimal plans must route to the compiled target");

        for (int i = 0; i < 4; i++)
        {
            Assert.That(compiledResult.IsNull(i), Is.EqualTo(spanResult.IsNull(i)), $"null mask at {i}");
            if (!spanResult.IsNull(i))
                Assert.That(compiledResult.GetValue(i), Is.EqualTo(spanResult.GetValue(i)), $"value at {i}");
        }
    }
}
