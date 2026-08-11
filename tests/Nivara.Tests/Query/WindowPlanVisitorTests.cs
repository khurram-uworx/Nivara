using Nivara.Expressions;
using Nivara.Operations;
using Nivara.Query;
using Nivara.Tensors;
using Nivara.Tests.Execution;
using NUnit.Framework;

namespace Nivara.Tests.Query;

/// <summary>
/// Tests for window operation support in the query plan layer (#159): visitors,
/// transformers, and plan analysis must route window operations to dedicated hooks
/// instead of treating them as unknown operations.
/// </summary>
[TestFixture]
public class WindowPlanVisitorTests
{
    static QueryPlan PlanWith(params IQueryOperation[] operations)
        => new(new StubQuerySource(), operations);

    // ── StatisticsVisitor ──

    [Test]
    public void StatisticsVisitor_CountsWindowOperationTypes()
    {
        var plan = PlanWith(
            new RollingOperation("A", "ma", 3, null, null, NivaraFrameExtensions.RollingKind.Sum),
            new RankOperation("rn", RankKind.Rank, new[] { new SortKey("A") }),
            new SelectOperation(new[] { ColumnExpressions.Col("A") }));

        var visitor = new QueryPlanStatisticsVisitor();
        visitor.Visit(plan);

        Assert.That(visitor.TotalOperations, Is.EqualTo(3));
        Assert.That(visitor.OperationCounts[OperationType.Rolling], Is.EqualTo(1));
        Assert.That(visitor.OperationCounts[OperationType.Rank], Is.EqualTo(1));
        Assert.That(visitor.OperationCounts[OperationType.Select], Is.EqualTo(1));
    }

    // ── VisitorBase dispatch ──

    [Test]
    public void VisitorBase_DispatchesWindowOperations_ToVisitWindow()
    {
        var plan = PlanWith(
            new RollingOperation("A", "ma", 3, null, null, NivaraFrameExtensions.RollingKind.Sum),
            new CumulativeOperation("A", "cum", null, NivaraFrameExtensions.CumulativeKind.Sum),
            new ShiftOperation("A", "lag", 1),
            new RankOperation("rn", RankKind.Rank, new[] { new SortKey("A") }),
            new SelectOperation(new[] { ColumnExpressions.Col("A") }));

        var visitor = new RecordingWindowVisitor();
        visitor.Visit(plan);

        Assert.That(visitor.WindowVisits, Is.EqualTo(4), "all four window operations must reach VisitWindow");
    }

    [Test]
    public void VisitorBase_NonWindowOperation_IsNotRoutedToVisitWindow()
    {
        var plan = PlanWith(
            new SelectOperation(new[] { ColumnExpressions.Col("A") }),
            new RollingOperation("A", "ma", 3, null, null, NivaraFrameExtensions.RollingKind.Sum));

        var visitor = new RecordingWindowVisitor();
        visitor.Visit(plan);

        Assert.That(visitor.WindowVisits, Is.EqualTo(1));
    }

    // ── TransformerBase pass-through ──

    [Test]
    public void Transformer_DispatchesWindowOperations_ToVisitWindow()
    {
        var rolling = new RollingOperation("A", "ma", 3, null, null, NivaraFrameExtensions.RollingKind.Sum);
        var rank = new RankOperation("rn", RankKind.Rank, new[] { new SortKey("A") });

        var transformer = new RecordingWindowTransformer();
        var rollingResult = transformer.Visit(rolling);
        var rankResult = transformer.Visit(rank);

        Assert.That(transformer.WindowVisits, Is.EqualTo(2));
        Assert.That(rollingResult, Is.SameAs(rolling), "default transformer must pass window ops through unchanged");
        Assert.That(rankResult, Is.SameAs(rank));
    }

    // ── GenerateDiagnosticInfo / plan analysis ──

    [Test]
    public void GenerateDiagnosticInfo_IncludesWindowOperationDetails()
    {
        var plan = PlanWith(
            new RollingOperation("A", "ma", 3, null, null, NivaraFrameExtensions.RollingKind.Sum),
            new CumulativeOperation("A", "cum", null, NivaraFrameExtensions.CumulativeKind.Sum),
            new ShiftOperation("A", "lag", 1),
            new RankOperation("rn", RankKind.Rank, new[] { new SortKey("A") }));

        var diagnostics = QueryPlanAnalyzer.GenerateDiagnosticInfo(plan);

        Assert.That(diagnostics, Does.Contain("Rolling: A (3)"));
        Assert.That(diagnostics, Does.Contain("Cumulative: A"));
        Assert.That(diagnostics, Does.Contain("Shift: A (1)"));
        Assert.That(diagnostics, Does.Contain("Rank: rn"));
    }

    sealed class RecordingWindowVisitor : QueryPlanVisitorBase
    {
        public int WindowVisits { get; private set; }

        protected override void VisitWindow(IQueryOperation operation)
        {
            WindowVisits++;
        }
    }

    sealed class RecordingWindowTransformer : QueryPlanTransformerBase<IQueryOperation>
    {
        public int WindowVisits { get; private set; }

        protected override IQueryOperation VisitWindow(IQueryOperation operation)
        {
            WindowVisits++;
            return operation;
        }
    }
}
