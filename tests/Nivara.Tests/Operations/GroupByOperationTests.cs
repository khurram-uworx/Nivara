using Nivara.Expressions;
using NUnit.Framework;

namespace Nivara.Tests.Operations;

[TestFixture]
public class GroupByOperationTests
{
    [Test]
    public void Constructor_WithValidColumns_CreatesOperation()
    {
        // Arrange
        var columns = new[] { ColumnExpressions.Col("Name") };

        // Act
        var operation = new GroupByOperation(columns);

        // Assert
        Assert.That(operation.GroupByColumns, Has.Count.EqualTo(1));
        Assert.That(operation.GroupByColumns[0].Name, Is.EqualTo("Name"));
        Assert.That(operation.OperationType, Is.EqualTo("GroupBy"));
    }

    [Test]
    public void Constructor_WithNullColumns_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new GroupByOperation(null!));
    }

    [Test]
    public void Constructor_WithEmptyColumns_ThrowsArgumentException()
    {
        // Arrange
        var columns = Array.Empty<ColumnExpression>();

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => new GroupByOperation(columns));
        Assert.That(ex.Message, Contains.Substring("Must specify at least one column expression"));
    }

    [Test]
    public void Execute_WithSingleGroupingColumn_ReturnsDistinctValues()
    {
        // Arrange
        var names = new[] { "Alice", "Bob", "Alice", "Charlie", "Bob" };
        var ages = new[] { 25, 30, 25, 35, 30 };

        var nameColumn = NivaraColumn<string>.Create(names);
        var ageColumn = NivaraColumn<int>.Create(ages);

        var input = new Dictionary<string, IColumn>
        {
            ["Name"] = nameColumn,
            ["Age"] = ageColumn
        };

        var operation = new GroupByOperation(new[] { ColumnExpressions.Col("Name") });

        // Act
        var result = operation.Execute(input);

        // Assert
        Assert.That(result, Has.Count.EqualTo(1));
        Assert.That(result.ContainsKey("Name"), Is.True);

        var resultColumn = result["Name"];
        Assert.That(resultColumn.Length, Is.EqualTo(3)); // Alice, Bob, Charlie

        var resultValues = new List<string>();
        for (int i = 0; i < resultColumn.Length; i++)
        {
            resultValues.Add((string)resultColumn.GetValue(i)!);
        }

        Assert.That(resultValues, Contains.Item("Alice"));
        Assert.That(resultValues, Contains.Item("Bob"));
        Assert.That(resultValues, Contains.Item("Charlie"));
    }

    [Test]
    public void Execute_WithMultipleGroupingColumns_ReturnsDistinctCombinations()
    {
        // Arrange
        var names = new[] { "Alice", "Bob", "Alice", "Alice", "Bob" };
        var departments = new[] { "IT", "HR", "IT", "Finance", "HR" };

        var nameColumn = NivaraColumn<string>.Create(names);
        var deptColumn = NivaraColumn<string>.Create(departments);

        var input = new Dictionary<string, IColumn>
        {
            ["Name"] = nameColumn,
            ["Department"] = deptColumn
        };

        var operation = new GroupByOperation(new[] {
            ColumnExpressions.Col("Name"),
            ColumnExpressions.Col("Department")
        });

        // Act
        var result = operation.Execute(input);

        // Assert
        Assert.That(result, Has.Count.EqualTo(2));
        Assert.That(result.ContainsKey("Name"), Is.True);
        Assert.That(result.ContainsKey("Department"), Is.True);

        var nameResultColumn = result["Name"];
        var deptResultColumn = result["Department"];

        // Distinct combinations: (Alice,IT), (Bob,HR), (Alice,Finance) = 3 combinations
        Assert.That(nameResultColumn.Length, Is.EqualTo(3));
        Assert.That(deptResultColumn.Length, Is.EqualTo(3));
    }

    [Test]
    public void Execute_WithNullValues_HandlesNullsCorrectly()
    {
        // Arrange
        var names = new string?[] { "Alice", null, "Alice", "Bob", null };
        var nameColumn = NivaraColumn<string?>.Create(names);

        var input = new Dictionary<string, IColumn>
        {
            ["Name"] = nameColumn
        };

        var operation = new GroupByOperation(new[] { ColumnExpressions.Col("Name") });

        // Act
        var result = operation.Execute(input);

        // Assert
        Assert.That(result, Has.Count.EqualTo(1));
        var resultColumn = result["Name"];

        // Should have distinct values including null
        Assert.That(resultColumn.Length, Is.EqualTo(3)); // Alice, null, Bob
    }

    [Test]
    public void Execute_WithEmptyInput_ReturnsEmptyResult()
    {
        // Arrange
        var emptyColumn = NivaraColumn<string>.Create(Array.Empty<string>());
        var input = new Dictionary<string, IColumn>
        {
            ["Name"] = emptyColumn
        };

        var operation = new GroupByOperation(new[] { ColumnExpressions.Col("Name") });

        // Act
        var result = operation.Execute(input);

        // Assert
        Assert.That(result, Has.Count.EqualTo(1));
        Assert.That(result["Name"].Length, Is.EqualTo(0));
    }

    [Test]
    public void TransformSchema_WithValidSchema_ReturnsCorrectSchema()
    {
        // Arrange
        var inputSchema = new Schema(new[]
        {
            ("Name", typeof(string)),
            ("Age", typeof(int)),
            ("Salary", typeof(double))
        });

        var operation = new GroupByOperation(new[] {
            ColumnExpressions.Col("Name"),
            ColumnExpressions.Col("Age")
        });

        // Act
        var resultSchema = operation.TransformSchema(inputSchema);

        // Assert
        Assert.That(resultSchema.ColumnNames, Has.Count.EqualTo(2));
        Assert.That(resultSchema.ColumnNames, Contains.Item("Name"));
        Assert.That(resultSchema.ColumnNames, Contains.Item("Age"));
        Assert.That(resultSchema.GetColumnType("Name"), Is.EqualTo(typeof(string)));
        Assert.That(resultSchema.GetColumnType("Age"), Is.EqualTo(typeof(int)));
    }

    [Test]
    public void ToString_ReturnsFormattedString()
    {
        // Arrange
        var operation = new GroupByOperation(new[] {
            ColumnExpressions.Col("Name"),
            ColumnExpressions.Col("Department")
        });

        // Act
        var result = operation.ToString();

        // Assert
        Assert.That(result, Is.EqualTo("GroupBy(Name, Department)"));
    }
}