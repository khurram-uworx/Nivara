using NUnit.Framework;

namespace Nivara.Tests;

[TestFixture]
public class QueryNodeTests
{
    Schema testSchema;

    [SetUp]
    public void SetUp()
    {
        testSchema = new Schema(new[] { ("Id", typeof(int)), ("Name", typeof(string)) });
    }

    [Test]
    public void SourceNode_Constructor_SetsProperties()
    {
        // Arrange
        var mockSource = new TestQuerySource(testSchema);

        // Act
        var node = new SourceNode(mockSource);

        // Assert
        Assert.That(node.Source, Is.SameAs(mockSource));
        Assert.That(node.OutputSchema, Is.EqualTo(testSchema));
        Assert.That(node.NodeType, Is.EqualTo("Source"));
        Assert.That(node.Children, Is.Empty);
    }

    [Test]
    public void SourceNode_NullSource_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new SourceNode(null!));
    }

    [Test]
    public void FilterNode_Constructor_SetsProperties()
    {
        // Act
        var node = new FilterNode(testSchema, "Age > 30", 0.3);

        // Assert
        Assert.That(node.Predicate, Is.EqualTo("Age > 30"));
        Assert.That(node.Selectivity, Is.EqualTo(0.3));
        Assert.That(node.OutputSchema, Is.EqualTo(testSchema));
        Assert.That(node.NodeType, Is.EqualTo("Filter"));
    }

    [Test]
    public void FilterNode_SelectivityClamping_ClampsToValidRange()
    {
        // Act
        var node1 = new FilterNode(testSchema, "test", -0.5); // Below 0
        var node2 = new FilterNode(testSchema, "test", 1.5);  // Above 1

        // Assert
        Assert.That(node1.Selectivity, Is.EqualTo(0.0));
        Assert.That(node2.Selectivity, Is.EqualTo(1.0));
    }

    [Test]
    public void ProjectionNode_Constructor_SetsProperties()
    {
        // Arrange
        var projections = new[] { "Id", "Name" };
        var outputSchema = new Schema(new[] { ("Id", typeof(int)), ("Name", typeof(string)) });

        // Act
        var node = new ProjectionNode(outputSchema, projections);

        // Assert
        Assert.That(node.Projections, Is.EqualTo(projections));
        Assert.That(node.OutputSchema, Is.EqualTo(outputSchema));
        Assert.That(node.NodeType, Is.EqualTo("Projection"));
    }

    [Test]
    public void ProjectionNode_EliminatesColumns_DetectsColumnElimination()
    {
        // Arrange
        var inputSchema = new Schema(new[] { ("Id", typeof(int)), ("Name", typeof(string)), ("Age", typeof(int)) });
        var outputSchema = new Schema(new[] { ("Id", typeof(int)), ("Name", typeof(string)) });
        var projections = new[] { "Id", "Name" };

        var inputNode = new SourceNode(new TestQuerySource(inputSchema));
        var projectionNode = new ProjectionNode(outputSchema, projections);
        projectionNode.Children.Add(inputNode);

        // Act
        var eliminatesColumns = projectionNode.EliminatesColumns;

        // Assert
        Assert.That(eliminatesColumns, Is.True);
    }

    [Test]
    public void GroupByNode_Constructor_SetsProperties()
    {
        // Arrange
        var groupByColumns = new[] { "Department" };
        var aggregations = new[] { "COUNT(*)", "AVG(Salary)" };
        var outputSchema = new Schema(new[] { ("Department", typeof(string)), ("Count", typeof(int)), ("AvgSalary", typeof(double)) });

        // Act
        var node = new GroupByNode(outputSchema, groupByColumns, aggregations);

        // Assert
        Assert.That(node.GroupByColumns, Is.EqualTo(groupByColumns));
        Assert.That(node.Aggregations, Is.EqualTo(aggregations));
        Assert.That(node.OutputSchema, Is.EqualTo(outputSchema));
        Assert.That(node.NodeType, Is.EqualTo("GroupBy"));
    }

    [Test]
    public void QueryNode_EstimatedProperties_CanBeSetAndRetrieved()
    {
        // Arrange
        var node = new FilterNode(testSchema, "test", 0.5);
        var expectedRowCount = 1000L;
        var expectedExecutionTime = TimeSpan.FromMilliseconds(500);

        // Act
        node.EstimatedRowCount = expectedRowCount;
        node.EstimatedExecutionTime = expectedExecutionTime;

        // Assert
        Assert.That(node.EstimatedRowCount, Is.EqualTo(expectedRowCount));
        Assert.That(node.EstimatedExecutionTime, Is.EqualTo(expectedExecutionTime));
    }

    [Test]
    public void QueryNode_ToString_ReturnsFormattedString()
    {
        // Arrange
        var node = new FilterNode(testSchema, "Age > 30", 0.3)
        {
            EstimatedRowCount = 500
        };

        // Act
        var result = node.ToString();

        // Assert
        Assert.That(result, Does.Contain("Filter"));
        Assert.That(result, Does.Contain("500"));
        Assert.That(result, Does.Contain("2 columns"));
    }

    [Test]
    public void QueryNode_ToString_UnknownRowCount_ShowsUnknown()
    {
        // Arrange
        var node = new FilterNode(testSchema, "test", 0.5); // EstimatedRowCount defaults to -1

        // Act
        var result = node.ToString();

        // Assert
        Assert.That(result, Does.Contain("Unknown"));
    }

    [Test]
    public void WithChildren_CreatesNewNodeWithChildren()
    {
        // Arrange
        var originalNode = new FilterNode(testSchema, "test", 0.5);
        var childNode = new SourceNode(new TestQuerySource(testSchema));
        var newChildren = new[] { childNode };

        // Act
        var newNode = originalNode.WithChildren(newChildren);

        // Assert
        Assert.That(newNode, Is.Not.SameAs(originalNode)); // Different instance
        Assert.That(newNode.Children, Contains.Item(childNode));
        Assert.That(originalNode.Children, Is.Empty); // Original unchanged
    }

    // Test helper class
    class TestQuerySource : IQuerySource
    {
        public TestQuerySource(Schema schema)
        {
            Schema = schema;
        }

        public Schema Schema { get; }
        public bool IsLazy => false;

        public IReadOnlyDictionary<string, IColumn> Execute()
        {
            return new Dictionary<string, IColumn>();
        }

        public void Dispose()
        {
            // Test implementation - no resources to dispose
        }
    }

    // Test visitor implementation
    class TestVisitor : IQueryNodeVisitor
    {
        public List<string> VisitedNodes { get; } = new();

        public void Visit(SourceNode node) => VisitedNodes.Add("Source");
        public void Visit(FilterNode node) => VisitedNodes.Add("Filter");
        public void Visit(ProjectionNode node) => VisitedNodes.Add("Projection");
        public void Visit(GroupByNode node) => VisitedNodes.Add("GroupBy");
    }

    [Test]
    public void Accept_Visitor_CallsCorrectVisitMethod()
    {
        // Arrange
        var visitor = new TestVisitor();
        var sourceNode = new SourceNode(new TestQuerySource(testSchema));
        var filterNode = new FilterNode(testSchema, "test", 0.5);
        var projectionNode = new ProjectionNode(testSchema, new[] { "Id" });
        var groupByNode = new GroupByNode(testSchema, new[] { "Name" }, new[] { "COUNT(*)" });

        // Act
        sourceNode.Accept(visitor);
        filterNode.Accept(visitor);
        projectionNode.Accept(visitor);
        groupByNode.Accept(visitor);

        // Assert
        Assert.That(visitor.VisitedNodes, Is.EqualTo(new[] { "Source", "Filter", "Projection", "GroupBy" }));
    }
}