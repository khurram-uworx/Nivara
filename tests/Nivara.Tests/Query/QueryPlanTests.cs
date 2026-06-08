using Nivara.Exceptions;
using Nivara.Query;
using Nivara.Tests.Execution;
using NUnit.Framework;
using System.Text.Json;

namespace Nivara.Tests.Query;

[TestFixture]
public class QueryPlanTests
{
    [Test]
    public void Constructor_NullSource_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new QueryPlan(null!, Array.Empty<IQueryOperation>()));
    }

    [Test]
    public void Constructor_NullOperations_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new QueryPlan(new StubQuerySource(), null!));
    }

    [Test]
    public void Constructor_ComputesResultSchemaFromSource()
    {
        var schema = new Schema(new[] { ("A", typeof(int)), ("B", typeof(string)) });
        var source = new StubQuerySource(schema: schema);
        var plan = new QueryPlan(source, Array.Empty<IQueryOperation>());

        Assert.That(plan.ResultSchema, Is.Not.Null);
        Assert.That(plan.ResultSchema.ColumnNames, Is.EquivalentTo(new[] { "A", "B" }));
    }

    [Test]
    public void Constructor_OperationsIsReadOnlyList()
    {
        var ops = new IQueryOperation[] { new StubQueryOperation() };
        var plan = new QueryPlan(new StubQuerySource(), ops);

        Assert.That(plan.Operations, Is.InstanceOf<IReadOnlyList<IQueryOperation>>());
        Assert.That(plan.Operations.Count, Is.EqualTo(1));
    }

    [Test]
    public void WithOperation_AddsOperationToPipeline()
    {
        var plan = new QueryPlan(new StubQuerySource(), Array.Empty<IQueryOperation>());

        var newPlan = plan.WithOperation(new StubQueryOperation("Filter"));

        Assert.That(newPlan, Is.Not.SameAs(plan));
        Assert.That(newPlan.Operations.Count, Is.EqualTo(1));
        Assert.That(newPlan.Operations[0].OperationType, Is.EqualTo("Filter"));
    }

    [Test]
    public void WithOperation_NullOperation_Throws()
    {
        var plan = new QueryPlan(new StubQuerySource(), Array.Empty<IQueryOperation>());

        Assert.Throws<ArgumentNullException>(() => plan.WithOperation(null!));
    }

    [Test]
    public void WithOperations_AddsMultipleOperations()
    {
        var plan = new QueryPlan(new StubQuerySource(), Array.Empty<IQueryOperation>());

        var newPlan = plan.WithOperations(new IQueryOperation[]
        {
            new StubQueryOperation("Filter"),
            new StubQueryOperation("Select"),
        });

        Assert.That(newPlan.Operations.Count, Is.EqualTo(2));
    }

    [Test]
    public void WithOperations_Null_Throws()
    {
        var plan = new QueryPlan(new StubQuerySource(), Array.Empty<IQueryOperation>());

        Assert.Throws<ArgumentNullException>(() => plan.WithOperations(null!));
    }

    [Test]
    public void ToString_ContainsSourcePipelineAndSchema()
    {
        var plan = new QueryPlan(new StubQuerySource(), new IQueryOperation[]
        {
            new StubQueryOperation("Filter"),
        });

        var str = plan.ToString();

        Assert.That(str, Does.Contain("QueryPlan"));
        Assert.That(str, Does.Contain("StubQuerySource"));
        Assert.That(str, Does.Contain("Filter"));
        Assert.That(str, Does.Contain("Schema"));
    }

    [Test]
    public void Serialize_ReturnsValidJson()
    {
        var plan = ExecutionTestHelpers.CreateTestPlan();

        var json = plan.Serialize();

        Assert.That(json, Is.Not.Null);
        Assert.That(json, Is.Not.Empty);
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        Assert.That(root.TryGetProperty("Source", out _), Is.True);
        Assert.That(root.TryGetProperty("Operations", out _), Is.True);
        Assert.That(root.TryGetProperty("ResultSchema", out _), Is.True);
    }

    [Test]
    public void Serialize_ContainsSourceAndOperationsInfo()
    {
        var plan = new QueryPlan(new StubQuerySource(), new IQueryOperation[]
        {
            new StubQueryOperation("Filter"),
            new StubQueryOperation("Select"),
        });

        var json = plan.Serialize();
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        Assert.That(root.GetProperty("Source").GetString(), Does.StartWith("StubQuerySource"));
        var ops = root.GetProperty("Operations");
        Assert.That(ops.GetArrayLength(), Is.EqualTo(2));
        Assert.That(ops[0].GetProperty("Type").GetString(), Is.EqualTo("Filter"));
        Assert.That(ops[1].GetProperty("Type").GetString(), Is.EqualTo("Select"));
    }

    [Test]
    public void ToDebugString_IncludesSourceOperationsAndResult()
    {
        var plan = new QueryPlan(new StubQuerySource(), new IQueryOperation[]
        {
            new StubQueryOperation("Filter"),
        });

        var debug = plan.ToDebugString();

        Assert.That(debug, Does.Contain("Source:"));
        Assert.That(debug, Does.Contain("Operations:"));
        Assert.That(debug, Does.Contain("[Filter]"));
        Assert.That(debug, Does.Contain("Result:"));
    }

    [Test]
    public void SchemaComputationFailure_ThrowsSchemaValidationException()
    {
        Assert.Throws<SchemaValidationException>(() =>
            new QueryPlan(new StubQuerySource(), new IQueryOperation[]
            {
                new FailingTransformSchemaOperation(),
            }));
    }

    [Test]
    public void ZeroOperations_PassesThroughSchema()
    {
        var schema = new Schema(new[] { ("A", typeof(int)) });
        var source = new StubQuerySource(schema: schema);
        var plan = new QueryPlan(source, Array.Empty<IQueryOperation>());

        Assert.That(plan.ResultSchema.ColumnNames, Is.EquivalentTo(new[] { "A" }));
    }

    [Test]
    public void Constructor_ComputedResultSchemaFromTransformChain()
    {
        var schema = new Schema(new[] { ("A", typeof(int)), ("B", typeof(string)) });
        var source = new StubQuerySource(schema: schema);
        var op = new StubQueryOperation("Select");
        op.TransformSchemaFn = input => input.WithoutColumn("B");
        var plan = new QueryPlan(source, new[] { op });

        Assert.That(plan.ResultSchema.ColumnNames, Is.EquivalentTo(new[] { "A" }));
    }
}
