using Nivara.Expressions;
using Nivara.Operations;
using NUnit.Framework;

namespace Nivara.Tests.Query;

/// <summary>
/// Guardrails for window-expression hydration in the fused evaluator (#159).
/// Window expressions embedded in Select-style expressions are materialized through their
/// kernels and the surrounding elementwise expression fuses over the materialized column;
/// nested windows compose; a bare window short-circuits to its materialized column.
/// </summary>
[TestFixture]
public class WindowExpressionEvaluationTests
{
    static Dictionary<string, IColumn> Input(params (string Name, IColumn Column)[] columns)
        => columns.ToDictionary(c => c.Name, c => c.Column, StringComparer.OrdinalIgnoreCase);

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
    public void Evaluate_StandaloneRollingSum_ReturnsMaterializedWindow()
    {
        var input = Input(("A", NivaraColumn<int>.Create(new[] { 1, 2, 3, 4 })));
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.RollingSum(ColumnExpressions.Col("A"), 2), input);

        Assert.That(fused.FusedPathEvaluationCount, Is.EqualTo(2), "source passthrough + bare window passthrough, no outer kernel");
        AssertColumn(result, new int?[] { null, 3, 5, 7 });
    }

    [Test]
    public void Evaluate_WindowFusedIntoElementwise_SingleOuterPass()
    {
        var input = Input(("A", NivaraColumn<int>.Create(new[] { 1, 2, 3, 4 })));
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.RollingSum(ColumnExpressions.Col("A"), 2) * 2, input);

        Assert.That(fused.FusedPathEvaluationCount, Is.EqualTo(2), "window source + one fused outer pass");
        Assert.That(fused.SpanKernelPathEvaluationCount, Is.EqualTo(1), "null-bearing window result fused through the span kernel");
        AssertColumn(result, new int?[] { null, 6, 10, 14 });
    }

    [Test]
    public void Evaluate_ComposedWindow_MixedOperandTypes_Promote()
    {
        var input = Input(
            ("A", NivaraColumn<int>.Create(new[] { 1, 2, 3, 4 })),
            ("B", NivaraColumn<double>.Create(new[] { 10.0, 20.0, 30.0, 40.0 })));
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(
            ColumnExpressions.RollingSum(ColumnExpressions.Col("A"), 2) + ColumnExpressions.Col("B"),
            input);

        Assert.That(result.ElementType, Is.EqualTo(typeof(double)));
        AssertColumn(result, new double?[] { null, 23.0, 35.0, 47.0 });
    }

    [Test]
    public void Evaluate_NestedWindows_Compose()
    {
        var input = Input(("A", NivaraColumn<int>.Create(new[] { 1, 2, 3, 4 })));
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(
            ColumnExpressions.RollingSum(ColumnExpressions.CumulativeSum(ColumnExpressions.Col("A")), 2),
            input);

        Assert.That(fused.FusedPathEvaluationCount, Is.EqualTo(3), "leaf passthrough + inner window passthrough + outer window passthrough");
        AssertColumn(result, new int?[] { null, 4, 9, 16 });
    }

    [Test]
    public void Evaluate_StandaloneRank_OverExpressionKeys()
    {
        var input = Input(
            ("Dept", NivaraColumn<string>.CreateForReferenceType(new[] { "X", "Y", "X", "Z" })),
            ("Score", NivaraColumn<int>.Create(new[] { 1, 2, 3, 4 })));
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(
            ColumnExpressions.Rank(
                new[] { new SortExpressionKey(ColumnExpressions.Col("Score")) },
                new[] { ColumnExpressions.Col("Dept") }),
            input);

        Assert.That(result.ElementType, Is.EqualTo(typeof(long)));
        AssertColumn(result, new long?[] { 1, 1, 2, 1 });
    }

    [Test]
    public void Evaluate_Shift_WithFillValue_AppliesAtBoundary()
    {
        var input = Input(("A", NivaraColumn<int>.Create(new[] { 1, 2, 3, 4 })));
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.Shift(ColumnExpressions.Col("A"), 1, -1), input);

        AssertColumn(result, new int?[] { -1, 1, 2, 3 });
    }

    [Test]
    public void Evaluate_Lead_NegatesPeriods()
    {
        var input = Input(("A", NivaraColumn<int>.Create(new[] { 1, 2, 3, 4 })));
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.Lead(ColumnExpressions.Col("A"), 2, 0), input);

        AssertColumn(result, new int?[] { 3, 4, 0, 0 });
    }

    [Test]
    public void Evaluate_RollingSum_OverNulls_PropagatesWindowSemantics()
    {
        var input = Input(("A", NivaraColumn.CreateFromNullable(new int?[] { 1, null, 3, 4 })));
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.RollingSum(ColumnExpressions.Col("A"), 2), input);

        AssertColumn(result, new int?[] { null, null, null, 7 });
    }

    [Test]
    public void Evaluate_CumulativeCount_OverSourceWithNulls_CountsValidOnly()
    {
        var input = Input(("A", NivaraColumn.CreateFromNullable(new int?[] { 1, null, 3, 4 })));
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.CumulativeCount(ColumnExpressions.Col("A")), input);

        AssertColumn(result, new long?[] { 1, null, 2, 3 });
    }

    [Test]
    public void Evaluate_WindowInComparison_FiltersThroughFusedPath()
    {
        var input = Input(("A", NivaraColumn<int>.Create(new[] { 1, 2, 3, 4 })));
        var fused = new FusedExpressionEvaluator();

        var result = fused.EvaluateBoolean(ColumnExpressions.RollingSum(ColumnExpressions.Col("A"), 2) > 4, input);

        Assert.That(result.ElementType, Is.EqualTo(typeof(bool)));
        AssertColumn(result, new bool?[] { null, false, true, true });
    }

    // ── #255: synthetic window names must not collide with user columns ──

    [Test]
    public void Evaluate_WindowNameCollidesWithUserColumn_DoesNotShadowUserColumn()
    {
        var input = Input(
            ("A", NivaraColumn<int>.Create(new[] { 1, 2, 3, 4 })),
            ("__window_0", NivaraColumn<int>.Create(new[] { 100, 200, 300, 400 })));
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(
            ColumnExpressions.RollingSum(ColumnExpressions.Col("A"), 2) + ColumnExpressions.Col("__window_0"),
            input);

        AssertColumn(result, new int?[] { null, 203, 305, 407 });
    }

    [Test]
    public void Evaluate_StandaloneWindow_WithCollidingUserColumn_UsesNextName()
    {
        var input = Input(
            ("A", NivaraColumn<int>.Create(new[] { 1, 2, 3, 4 })),
            ("__window_0", NivaraColumn<int>.Create(new[] { 100, 200, 300, 400 })));
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(ColumnExpressions.RollingSum(ColumnExpressions.Col("A"), 2), input);

        AssertColumn(result, new int?[] { null, 3, 5, 7 });
    }

    [Test]
    public void Evaluate_MultipleWindows_SkipAllCollidingSyntheticNames()
    {
        var input = Input(
            ("A", NivaraColumn<int>.Create(new[] { 1, 2, 3, 4 })),
            ("B", NivaraColumn<int>.Create(new[] { 10, 20, 30, 40 })),
            ("__window_0", NivaraColumn<int>.Create(new[] { 1, 1, 1, 1 })),
            ("__window_1", NivaraColumn<int>.Create(new[] { 2, 2, 2, 2 })));
        var fused = new FusedExpressionEvaluator();

        var result = fused.Evaluate(
            ColumnExpressions.RollingSum(ColumnExpressions.Col("A"), 2) + ColumnExpressions.RollingSum(ColumnExpressions.Col("B"), 2),
            input);

        AssertColumn(result, new int?[] { null, 33, 55, 77 });
    }
}
