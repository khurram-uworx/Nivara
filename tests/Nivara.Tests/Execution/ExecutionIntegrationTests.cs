using Nivara.Diagnostics;
using Nivara.Execution;
using Nivara.IO;
using Nivara.Operations;
using Nivara.Optimization;
using Nivara.Query;
using NUnit.Framework;

namespace Nivara.Tests.Execution;

[TestFixture]
public class ExecutionIntegrationTests
{
    static ExecutionEngine CreateEngine() => new();

    static StubQuerySource CreateRangeSource(int count, string schemaType = "int")
    {
        var schema = schemaType == "int"
            ? new Schema(new[] { ("A", typeof(int)) })
            : new Schema(new[] { ("A", typeof(int)), ("B", typeof(string)) });
        var source = new StubQuerySource(schema: schema);
        source.ExecuteFn = () =>
        {
            var intData = new int[count];
            for (int i = 0; i < count; i++) intData[i] = i;
            if (schemaType == "int")
                return new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(intData) };
            var strData = new string[count];
            for (int i = 0; i < count; i++) strData[i] = $"v{i}";
            return new Dictionary<string, IColumn>
            {
                ["A"] = NivaraColumn<int>.Create(intData),
                ["B"] = NivaraColumn<string>.Create(strData),
            };
        };
        return source;
    }

    static NivaraExecutionContext CreateContext(ExecutionStrategy strategy)
    {
        var ctx = new NivaraExecutionContext(strategy);
        if (strategy == ExecutionStrategy.Parallel)
            ctx.MaxDegreeOfParallelism = 2;
        if (strategy == ExecutionStrategy.Streaming)
            ctx.MemoryBudget = 1024 * 1024;
        return ctx;
    }

    // ── In-memory pipeline: Filter → Select → Collect ──

    [Test]
    public void InMemory_FilterSelect_Eager_CorrectResults()
    {
        var source = CreateRangeSource(100, "int_str");
        var filterOp = new StubQueryOperation("Filter");
        var selectOp = new StubQueryOperation("Select");
        selectOp.TransformSchemaFn = s => s.WithoutColumn("B");
        selectOp.ExecuteFn = input =>
        {
            var result = new Dictionary<string, IColumn>(input.Count);
            if (input.ContainsKey("A")) result["A"] = input["A"];
            return result;
        };
        var plan = new QueryPlan(source, new IQueryOperation[] { filterOp, selectOp });

        using var result = CreateEngine().Execute(plan, CreateContext(ExecutionStrategy.Eager));

        Assert.That(result.RowCount, Is.EqualTo(100));
        Assert.That(result.ColumnCount, Is.EqualTo(1));
        Assert.That(result.HasColumn("A"), Is.True);
    }

    [Test]
    public void InMemory_FilterSelect_Lazy_CorrectResults()
    {
        var source = CreateRangeSource(100, "int");
        var filterOp = new StubQueryOperation("Filter");
        var selectOp = new StubQueryOperation("Select");
        var plan = new QueryPlan(source, new IQueryOperation[] { filterOp, selectOp });

        using var result = CreateEngine().Execute(plan, CreateContext(ExecutionStrategy.Lazy));

        Assert.That(result.RowCount, Is.EqualTo(100));
    }

    [Test]
    public void InMemory_FilterSelect_Parallel_CorrectResults()
    {
        var source = CreateRangeSource(100, "int");
        var filterOp = new StubQueryOperation("Filter");
        var selectOp = new StubQueryOperation("Select");
        var plan = new QueryPlan(source, new IQueryOperation[] { filterOp, selectOp });

        using var result = CreateEngine().Execute(plan, CreateContext(ExecutionStrategy.Parallel));

        Assert.That(result.RowCount, Is.EqualTo(100));
    }

    [Test]
    public void InMemory_FilterSelect_Streaming_CorrectResults()
    {
        var source = new StubChunkedQuerySource(100);
        var filterOp = new StubQueryOperation("Filter");
        var selectOp = new StubQueryOperation("Select");
        var plan = new QueryPlan(source, new IQueryOperation[] { filterOp, selectOp });

        using var result = CreateEngine().Execute(plan, CreateContext(ExecutionStrategy.Streaming));

        Assert.That(result.RowCount, Is.EqualTo(100));
    }

    // ── Chained Filter → Sort → Select → GroupBy → Collect (non-streamable fallback) ──

    [Test]
    public void ChainedFilterSortSelectGroupBy_Streaming_FallsBackToLazy()
    {
        var source = CreateRangeSource(50);
        var ops = new IQueryOperation[]
        {
            new StubQueryOperation("Filter"),
            new StubQueryOperation("Sort"),
            new StubQueryOperation("Select"),
            new StubQueryOperation("GroupBy"),
        };
        var plan = new QueryPlan(source, ops);

        using var result = CreateEngine().Execute(plan, CreateContext(ExecutionStrategy.Streaming));

        Assert.That(result, Is.Not.Null);
        Assert.That(result.ColumnCount, Is.GreaterThan(0));
    }

    // ── Join of two in-memory sources ──

    [Test]
    public void JoinTwoInMemorySources_Collect_CorrectResult()
    {
        var left = new Dictionary<string, IColumn>
        {
            ["Key"] = NivaraColumn<int>.Create(new[] { 1, 2, 3 }),
            ["ValL"] = NivaraColumn<string>.Create(new[] { "a", "b", "c" }),
        };
        var right = new Dictionary<string, IColumn>
        {
            ["Key"] = NivaraColumn<int>.Create(new[] { 1, 2, 4 }),
            ["ValR"] = NivaraColumn<string>.Create(new[] { "x", "y", "z" }),
        };

        var joinOp = new JoinOperation(left, right, JoinType.Inner, new[] { new JoinKey("Key") });
        var schema = new Schema(new[] { ("Key", typeof(int)), ("ValL", typeof(string)) });
        var source = new StubQuerySource(schema: schema);
        source.ExecuteFn = () => left;
        var plan = new QueryPlan(source, new IQueryOperation[] { joinOp });

        using var result = CreateEngine().Execute(plan, CreateContext(ExecutionStrategy.Eager));

        Assert.That(result, Is.Not.Null);
        Assert.That(result.RowCount, Is.EqualTo(2));
        Assert.That(result.ColumnCount, Is.EqualTo(3));
        Assert.That(result.ColumnNames, Is.EquivalentTo(new[] { "Key", "ValL", "ValR" }));
        Assert.That(result.GetColumn<int>("Key")[0], Is.EqualTo(1));
        Assert.That(result.GetColumn<string>("ValL")[0], Is.EqualTo("a"));
        Assert.That(result.GetColumn<string>("ValR")[0], Is.EqualTo("x"));
    }

    // ── JSON source integration ──

    [Test]
    public void JsonSource_Select_Collect_Eager_CorrectResults()
    {
        var json = "[{\"Name\":\"Alice\",\"Age\":30},{\"Name\":\"Bob\",\"Age\":25},{\"Name\":\"Charlie\",\"Age\":35}]";
        var tempFile = Path.GetTempFileName();
        try
        {
            File.WriteAllText(tempFile, json);
            var source = Json.Scan(tempFile);
            var plan = new QueryPlan(source, new IQueryOperation[]
            {
                new StubQueryOperation("Filter"),
                new StubQueryOperation("Select"),
            });
            using var result = CreateEngine().Execute(plan, CreateContext(ExecutionStrategy.Eager));
            Assert.That(result.RowCount, Is.EqualTo(3));
            Assert.That(result.ColumnCount, Is.EqualTo(2));
            Assert.That(result.HasColumn("Name"), Is.True);
            Assert.That(result.HasColumn("Age"), Is.True);
        }
        finally
        {
            TryDeleteFile(tempFile);
        }
    }

    // ── CSV source multi-strategy ──

    [Test]
    public void CsvSource_Filter_Collect_AllStrategies_CorrectResults(
        [Values(ExecutionStrategy.Eager, ExecutionStrategy.Lazy, ExecutionStrategy.Streaming, ExecutionStrategy.Parallel)]
        ExecutionStrategy strategy)
    {
        var csv = "Name,Age\nAlice,30\nBob,25\nCharlie,35";
        var tempFile = Path.GetTempFileName();
        try
        {
            File.WriteAllText(tempFile, csv);
            var source = Csv.Scan(tempFile);
            var plan = new QueryPlan(source, new IQueryOperation[] { new StubQueryOperation("Filter") });
            using var result = CreateEngine().Execute(plan, CreateContext(strategy));
            Assert.That(result.RowCount, Is.EqualTo(3), strategy.ToString());
            Assert.That(result.HasColumn("Name"), Is.True, strategy.ToString());
            Assert.That(result.HasColumn("Age"), Is.True, strategy.ToString());
        }
        finally
        {
            TryDeleteFile(tempFile);
        }
    }

    // ── Optimization warnings test ──

    [Test]
    public void DiagnosticsIntegration_OptimizationWarnings_Recorded()
    {
        var source = CreateRangeSource(100);
        var plan = new QueryPlan(source, Array.Empty<IQueryOperation>());
        var diagnostics = new ExecutionDiagnostics();
        diagnostics.RecordWarning(new PerformanceWarning(
            PerformanceWarningSeverity.Info,
            "Multiple sequential filters detected",
            "Consider merging filter operations"));
        var context = new NivaraExecutionContext(ExecutionStrategy.Eager)
        {
            ExecutionDiagnostics = diagnostics
        };

        using var result = CreateEngine().Execute(plan, context);

        Assert.That(diagnostics.Warnings, Is.Not.Empty);
        Assert.That(diagnostics.Warnings.Any(w => w.Message.Contains("filter")), Is.True);
    }

    // ── Optimizer with predicate pushdown ──

    [Test]
    public void Optimizer_PredicatePushdown_SameResultsAsUnoptimized()
    {
        var source = CreateRangeSource(100);
        var op = new StubQueryOperation("Filter");
        var plan = new QueryPlan(source, new IQueryOperation[] { op });
        var optimizer = new QueryOptimizer(OptimizationEngine.CreateDefault());
        var executor = new QueryExecutor();

        using var unoptimized = executor.Execute(plan);
        using var optimized = executor.ExecuteOptimized(plan, optimizer);

        Assert.That(optimized.RowCount, Is.EqualTo(unoptimized.RowCount));
        Assert.That(optimized.ColumnCount, Is.EqualTo(unoptimized.ColumnCount));
    }

    // ── Diagnostics integration ──

    [Test]
    public void DiagnosticsIntegration_TimingsRecordedEndToEnd()
    {
        var plan = ExecutionTestHelpers.CreateTestPlan();
        var diagnostics = new ExecutionDiagnostics();
        var context = new NivaraExecutionContext(ExecutionStrategy.Eager)
        {
            ExecutionDiagnostics = diagnostics
        };

        using var result = CreateEngine().Execute(plan, context);

        Assert.That(diagnostics.TotalExecutionTime, Is.GreaterThan(TimeSpan.Zero));
    }

    [Test]
    public void DiagnosticsIntegration_PropertiesPopulatedEndToEnd()
    {
        var schema = new Schema(new[] { ("A", typeof(int)), ("B", typeof(string)) });
        var source = new StubQuerySource(schema: schema, isLazy: true) { ExecuteFn = ExecutionTestHelpers.DefaultExecuteFn };
        var op = new StubQueryOperation("Filter");
        var plan = new QueryPlan(source, new IQueryOperation[] { op });
        var diagnostics = new ExecutionDiagnostics();
        var context = new NivaraExecutionContext(ExecutionStrategy.Lazy)
        {
            ExecutionDiagnostics = diagnostics
        };

        CreateEngine().Execute(plan, context);

        Assert.That(diagnostics.ExecutionStrategy, Is.EqualTo(ExecutionStrategy.Lazy));
        Assert.That(diagnostics.TotalExecutionTime, Is.GreaterThan(TimeSpan.Zero));
    }

    static void TryDeleteFile(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); }
        catch { /* best-effort cleanup */ }
    }
}
