using Nivara.Query;
using Nivara.Tests.Execution;
using NUnit.Framework;

namespace Nivara.Tests.Query;

[TestFixture]
public class QueryPlanAnalyzerTests
{
    // ── Explain ──

    [Test]
    public void Explain_NullPlan_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            QueryPlanAnalyzer.Explain(null!));
    }

    [Test]
    public void Explain_ReturnsTreeOutputWithSource()
    {
        var plan = new QueryPlan(new StubQuerySource(), Array.Empty<IQueryOperation>());

        var result = QueryPlanAnalyzer.Explain(plan);

        Assert.That(result, Does.Contain("Query Execution Plan"));
        Assert.That(result, Does.Contain("StubQuerySource"));
        Assert.That(result, Does.Contain("Result Schema"));
    }

    [Test]
    public void Explain_SingleOperation_IncludesOperation()
    {
        var plan = new QueryPlan(new StubQuerySource(), new IQueryOperation[]
        {
            new StubQueryOperation("Filter"),
        });

        var result = QueryPlanAnalyzer.Explain(plan);

        Assert.That(result, Does.Contain("Filter"));
        Assert.That(result, Does.Contain("1."));
    }

    [Test]
    public void Explain_MultipleOperations_AllIncluded()
    {
        var plan = new QueryPlan(new StubQuerySource(), new IQueryOperation[]
        {
            new StubQueryOperation("Filter"),
            new StubQueryOperation("Select"),
        });

        var result = QueryPlanAnalyzer.Explain(plan);

        Assert.That(result, Does.Contain("1. Filter"));
        Assert.That(result, Does.Contain("2. Select"));
    }

    [Test]
    public void Explain_ZeroOperations_StillValid()
    {
        var plan = new QueryPlan(new StubQuerySource(), Array.Empty<IQueryOperation>());

        var result = QueryPlanAnalyzer.Explain(plan);

        Assert.That(result, Does.Contain("Result Schema"));
    }

    [Test]
    public void Explain_IncludesSchemaChanges()
    {
        var schema = new Schema(new[] { ("A", typeof(int)), ("B", typeof(string)) });
        var source = new StubQuerySource(schema: schema);
        var op = new StubQueryOperation("Select");
        op.TransformSchemaFn = input => input.WithoutColumn("B");
        var plan = new QueryPlan(source, new[] { op });

        var result = QueryPlanAnalyzer.Explain(plan);

        Assert.That(result, Does.Contain("Schema:"));
    }

    // ── AnalyzeOptimizations ──

    [Test]
    public void AnalyzeOptimizations_NullPlan_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            QueryPlanAnalyzer.AnalyzeOptimizations(null!));
    }

    [Test]
    public void AnalyzeOptimizations_MultipleFilters_Detected()
    {
        var plan = new QueryPlan(new StubQuerySource(), new IQueryOperation[]
        {
            new StubQueryOperation("Filter"),
            new StubQueryOperation("Filter"),
        });

        var suggestions = QueryPlanAnalyzer.AnalyzeOptimizations(plan);

        Assert.That(suggestions, Has.Some.Contains("combining"));
    }

    [Test]
    public void AnalyzeOptimizations_MultipleSelects_Detected()
    {
        var plan = new QueryPlan(new StubQuerySource(), new IQueryOperation[]
        {
            new StubQueryOperation("Select"),
            new StubQueryOperation("Select"),
        });

        var suggestions = QueryPlanAnalyzer.AnalyzeOptimizations(plan);

        Assert.That(suggestions, Has.Some.Contains("projections"));
    }

    [Test]
    public void AnalyzeOptimizations_PredicatePushdown_Suggested()
    {
        var plan = new QueryPlan(
            new StubQuerySource(isLazy: true),
            new IQueryOperation[] { new StubQueryOperation("Filter") });

        var suggestions = QueryPlanAnalyzer.AnalyzeOptimizations(plan);

        Assert.That(suggestions, Has.Some.Contains("predicate pushdown"));
    }

    [Test]
    public void AnalyzeOptimizations_ColumnSelection_Suggested()
    {
        var schema = new Schema(new[] { ("A", typeof(int)), ("B", typeof(string)) });
        var source = new StubQuerySource(schema: schema);
        var op = new StubQueryOperation("Filter");
        op.TransformSchemaFn = input => input.WithoutColumn("B");
        var plan = new QueryPlan(source, new[] { op });

        var suggestions = QueryPlanAnalyzer.AnalyzeOptimizations(plan);

        Assert.That(suggestions, Has.Some.Contains("column selection"));
    }

    [Test]
    public void AnalyzeOptimizations_OptimalPlan_ReturnsEmpty()
    {
        var plan = new QueryPlan(new StubQuerySource(), Array.Empty<IQueryOperation>());

        var suggestions = QueryPlanAnalyzer.AnalyzeOptimizations(plan);

        Assert.That(suggestions, Is.Empty);
    }

    // ── GenerateDiagnosticInfo ──

    [Test]
    public void GenerateDiagnosticInfo_NullPlan_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            QueryPlanAnalyzer.GenerateDiagnosticInfo(null!));
    }

    [Test]
    public void GenerateDiagnosticInfo_IncludesSourceAndOperationCount()
    {
        var plan = ExecutionTestHelpers.CreateTestPlan();

        var info = QueryPlanAnalyzer.GenerateDiagnosticInfo(plan);

        Assert.That(info, Does.Contain("StubQuerySource"));
        Assert.That(info, Does.Contain("Operations Count"));
    }

    [Test]
    public void GenerateDiagnosticInfo_IncludesInputOutputColumns()
    {
        var plan = ExecutionTestHelpers.CreateTestPlan();

        var info = QueryPlanAnalyzer.GenerateDiagnosticInfo(plan);

        Assert.That(info, Does.Contain("Input Columns"));
        Assert.That(info, Does.Contain("Output Columns"));
    }

    [Test]
    public void GenerateDiagnosticInfo_IncludesSchemaDetails()
    {
        var plan = ExecutionTestHelpers.CreateTestPlan();

        var info = QueryPlanAnalyzer.GenerateDiagnosticInfo(plan);

        Assert.That(info, Does.Contain("Input Schema"));
        Assert.That(info, Does.Contain("Output Schema"));
        Assert.That(info, Does.Contain("A:"));
        Assert.That(info, Does.Contain("B:"));
    }

    [Test]
    public void GenerateDiagnosticInfo_IncludesOperationDetails()
    {
        var plan = ExecutionTestHelpers.CreateTestPlan();

        var info = QueryPlanAnalyzer.GenerateDiagnosticInfo(plan);

        Assert.That(info, Does.Contain("Operation Details"));
        Assert.That(info, Does.Contain("Filter"));
    }

    [Test]
    public void GenerateDiagnosticInfo_IncludesErrorSection_WhenProvided()
    {
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var error = new InvalidOperationException("test error");

        var info = QueryPlanAnalyzer.GenerateDiagnosticInfo(plan, error);

        Assert.That(info, Does.Contain("Execution Error"));
        Assert.That(info, Does.Contain("InvalidOperationException"));
        Assert.That(info, Does.Contain("test error"));
    }

    [Test]
    public void GenerateDiagnosticInfo_ExcludesErrorSection_WhenNull()
    {
        var plan = ExecutionTestHelpers.CreateTestPlan();

        var info = QueryPlanAnalyzer.GenerateDiagnosticInfo(plan, null);

        Assert.That(info, Does.Not.Contain("Execution Error"));
    }

    [Test]
    public void GenerateDiagnosticInfo_IncludesInnerExceptionDetails()
    {
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var inner = new ArgumentException("inner error");
        var outer = new InvalidOperationException("outer error", inner);

        var info = QueryPlanAnalyzer.GenerateDiagnosticInfo(plan, outer);

        Assert.That(info, Does.Contain("Inner Exception"));
        Assert.That(info, Does.Contain("ArgumentException"));
        Assert.That(info, Does.Contain("inner error"));
    }
}
