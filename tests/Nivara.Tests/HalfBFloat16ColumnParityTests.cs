using Nivara;
using Nivara.Expressions;
using Nivara.Operations;
using Nivara.Query;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests;

/// <summary>
/// Consolidated targeted parity coverage for <see cref="Half"/> and <see cref="BFloat16"/>
/// columns (issue #1021).
/// </summary>
[TestFixture]
public class HalfBFloat16ColumnParityTests
{
    static NivaraColumn<Half> HalfColumn(params Half[] values) => NivaraColumn<Half>.Create(values);

    static NivaraColumn<BFloat16> BFloat16Column(params BFloat16[] values) => NivaraColumn<BFloat16>.Create(values);

    static NivaraFrame FrameWith(params (string Name, IColumn Column)[] columns) => new NivaraFrame(columns);

    [Test]
    public void Arithmetic_NullMaskPropagation_Half()
    {
        var values = new Half?[] { (Half)1.5f, null, (Half)3.5f };
        var column = NivaraColumnFactory.CreateFromNullable(values);
        var result = column.Multiply((Half)2.0f);

        for (int i = 0; i < values.Length; i++)
            Assert.That(result.IsNull(i), Is.EqualTo(values[i] == null), $"null propagation mismatch at {i}");
        Assert.That((float)(Half)result.GetValue(0)!, Is.EqualTo(3.0f));
        Assert.That((float)(Half)result.GetValue(2)!, Is.EqualTo(7.0f));
    }

    [Test]
    public void Arithmetic_NullMaskPropagation_BFloat16()
    {
        var values = new BFloat16?[] { (BFloat16)1.5f, null, (BFloat16)3.5f };
        var column = NivaraColumnFactory.CreateFromNullable(values);
        var result = column.Multiply((BFloat16)2.0f);

        for (int i = 0; i < values.Length; i++)
            Assert.That(result.IsNull(i), Is.EqualTo(values[i] == null), $"null propagation mismatch at {i}");
        Assert.That((float)(BFloat16)result.GetValue(0)!, Is.EqualTo(3.0f));
        Assert.That((float)(BFloat16)result.GetValue(2)!, Is.EqualTo(7.0f));
    }

    [Test]
    public void Comparison_SqlLikeNullMask_Half()
    {
        var left = NivaraColumnFactory.CreateFromNullable(new Half?[] { (Half)1f, null, (Half)3f });
        var right = NivaraColumnFactory.CreateFromNullable(new Half?[] { (Half)2f, (Half)2f, null });

        var greater = left.GreaterThan(right);
        Assert.That(greater.IsNull(0), Is.False);
        Assert.That(greater[0], Is.False);
        Assert.That(greater.IsNull(1), Is.True);
        Assert.That(greater.IsNull(2), Is.True);
    }

    [Test]
    public void Comparison_SqlLikeNullMask_BFloat16()
    {
        var left = NivaraColumnFactory.CreateFromNullable(new BFloat16?[] { (BFloat16)1f, null, (BFloat16)3f });
        var right = NivaraColumnFactory.CreateFromNullable(new BFloat16?[] { (BFloat16)2f, (BFloat16)2f, null });

        var greater = left.GreaterThan(right);
        Assert.That(greater.IsNull(0), Is.False);
        Assert.That(greater[0], Is.False);
        Assert.That(greater.IsNull(1), Is.True);
        Assert.That(greater.IsNull(2), Is.True);
    }

    [Test]
    public void Window_RollingSum_Half()
    {
        var frame = FrameWith(("v", HalfColumn((Half)1, (Half)2, (Half)3, (Half)4)));
        var result = frame.RollingSum("v", "sum", 2);
        var sum = result.GetColumn<Half>("sum");
        Assert.That(sum.IsNull(0), Is.True);
        Assert.That((float)(Half)sum.GetValue(1)!, Is.EqualTo(3.0f));
        Assert.That((float)(Half)sum.GetValue(3)!, Is.EqualTo(7.0f));
    }

    [Test]
    public void Window_RollingSum_BFloat16()
    {
        var frame = FrameWith(("v", BFloat16Column((BFloat16)1, (BFloat16)2, (BFloat16)3, (BFloat16)4)));
        var result = frame.RollingSum("v", "sum", 2);
        var sum = result.GetColumn<BFloat16>("sum");
        Assert.That(sum.IsNull(0), Is.True);
        Assert.That((float)(BFloat16)sum.GetValue(1)!, Is.EqualTo(3.0f));
        Assert.That((float)(BFloat16)sum.GetValue(3)!, Is.EqualTo(7.0f));
    }

    [Test]
    public void Sorting_OrderBy_Half()
    {
        var frame = FrameWith(("v", HalfColumn((Half)3, (Half)1, (Half)2)));
        var sorted = frame.OrderBy("v", ascending: true);
        var v = sorted.GetColumn<Half>("v");
        Assert.That((float)(Half)v.GetValue(0)!, Is.EqualTo(1.0f));
        Assert.That((float)(Half)v.GetValue(1)!, Is.EqualTo(2.0f));
        Assert.That((float)(Half)v.GetValue(2)!, Is.EqualTo(3.0f));
    }

    [Test]
    public void Sorting_OrderBy_BFloat16()
    {
        var frame = FrameWith(("v", BFloat16Column((BFloat16)3, (BFloat16)1, (BFloat16)2)));
        var sorted = frame.OrderBy("v", ascending: true);
        var v = sorted.GetColumn<BFloat16>("v");
        Assert.That((float)(BFloat16)v.GetValue(0)!, Is.EqualTo(1.0f));
        Assert.That((float)(BFloat16)v.GetValue(1)!, Is.EqualTo(2.0f));
        Assert.That((float)(BFloat16)v.GetValue(2)!, Is.EqualTo(3.0f));
    }

    [Test]
    public void FusedExpressions_HalfSameType()
    {
        var column = NivaraColumn<Half>.Create(new[] { (Half)1.5f, (Half)2.5f, (Half)3.5f });
        var input = new Dictionary<string, IColumn> { ["A"] = column };
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.Col("A") + ColumnExpressions.Col("A"), input);
        Assert.That(result.ElementType, Is.EqualTo(typeof(Half)));
        Assert.That((float)(Half)result.GetValue(0)!, Is.EqualTo(3.0f));
    }

    [Test]
    public void FusedExpressions_BFloat16SameType()
    {
        var column = NivaraColumn<BFloat16>.Create(new[] { (BFloat16)1.5f, (BFloat16)2.5f, (BFloat16)3.5f });
        var input = new Dictionary<string, IColumn> { ["A"] = column };
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.Col("A") + ColumnExpressions.Col("A"), input);
        Assert.That(result.ElementType, Is.EqualTo(typeof(BFloat16)));
        Assert.That((float)(BFloat16)result.GetValue(0)!, Is.EqualTo(3.0f));
    }

    [Test]
    public void FusedExpressions_HalfPlusInt_PromotesToDouble()
    {
        var frame = FrameWith(("A", HalfColumn((Half)1, (Half)2, (Half)3)));
        var input = new Dictionary<string, IColumn> { ["A"] = frame.GetColumn<Half>("A") };
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.Col("A") + ColumnExpressions.Lit(1), input);
        Assert.That(result.ElementType, Is.EqualTo(typeof(double)));
    }

    [Test]
    public void FusedExpressions_BFloat16PlusInt_PromotesToDouble()
    {
        var frame = FrameWith(("A", BFloat16Column((BFloat16)1, (BFloat16)2, (BFloat16)3)));
        var input = new Dictionary<string, IColumn> { ["A"] = frame.GetColumn<BFloat16>("A") };
        var fused = new FusedExpressionEvaluator();
        var result = fused.Evaluate(ColumnExpressions.Col("A") + ColumnExpressions.Lit(1), input);
        Assert.That(result.ElementType, Is.EqualTo(typeof(double)));
    }
}
