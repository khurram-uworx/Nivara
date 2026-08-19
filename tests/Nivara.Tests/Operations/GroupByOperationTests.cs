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
    public void Execute_WithHalfKeyColumn_ReturnsTypedDistinctColumn()
    {
        var column = NivaraColumn<Half>.Create(new Half[] { (Half)1.5, (Half)2.5, (Half)1.5 });
        var input = new Dictionary<string, IColumn> { ["Rate"] = column };
        var operation = new GroupByOperation(new[] { ColumnExpressions.Col("Rate") });

        var result = operation.Execute(input);

        var resultColumn = result["Rate"];
        Assert.That(resultColumn, Is.InstanceOf<NivaraColumn<Half>>());
        Assert.That(resultColumn.Length, Is.EqualTo(2));
        Assert.That(resultColumn.GetValue(0), Is.EqualTo((Half)1.5).Or.EqualTo((Half)2.5));
        Assert.That(resultColumn.GetValue(1), Is.EqualTo((Half)1.5).Or.EqualTo((Half)2.5));
        Assert.That(resultColumn.GetValue(0), Is.Not.EqualTo(resultColumn.GetValue(1)));
    }

    [Test]
    public void Execute_WithNIntKeyColumn_ReturnsTypedDistinctColumn()
    {
        var column = NivaraColumn<nint>.Create(new nint[] { 10, 20, 10, 30 });
        var input = new Dictionary<string, IColumn> { ["Id"] = column };
        var operation = new GroupByOperation(new[] { ColumnExpressions.Col("Id") });

        var result = operation.Execute(input);

        var resultColumn = result["Id"];
        Assert.That(resultColumn, Is.InstanceOf<NivaraColumn<nint>>());
        Assert.That(resultColumn.Length, Is.EqualTo(3));
    }

    [Test]
    public void Execute_WithUIntKeyColumn_ReturnsTypedDistinctColumn()
    {
        var column = NivaraColumn<uint>.Create(new uint[] { 1u, 2u, 1u });
        var input = new Dictionary<string, IColumn> { ["Id"] = column };
        var operation = new GroupByOperation(new[] { ColumnExpressions.Col("Id") });

        var result = operation.Execute(input);

        var resultColumn = result["Id"];
        Assert.That(resultColumn, Is.InstanceOf<NivaraColumn<uint>>());
        Assert.That(resultColumn.Length, Is.EqualTo(2));
    }

    [Test]
    public void Execute_WithCharKeyColumn_ReturnsTypedDistinctColumn()
    {
        var column = NivaraColumn<char>.Create(new char[] { 'a', 'b', 'a' });
        var input = new Dictionary<string, IColumn> { ["Code"] = column };
        var operation = new GroupByOperation(new[] { ColumnExpressions.Col("Code") });

        var result = operation.Execute(input);

        var resultColumn = result["Code"];
        Assert.That(resultColumn, Is.InstanceOf<NivaraColumn<char>>());
        Assert.That(resultColumn.Length, Is.EqualTo(2));
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

    [Test]
    public void Execute_WithAggregation_ReturnsKeyAndAggregatedColumns()
    {
        var names = new[] { "Alice", "Bob", "Alice", "Charlie", "Bob" };
        var salaries = new[] { 100.0, 200.0, 150.0, 300.0, 250.0 };

        var nameColumn = NivaraColumn<string>.Create(names);
        var salaryColumn = NivaraColumn<double>.Create(salaries);

        var input = new Dictionary<string, IColumn>
        {
            ["Name"] = nameColumn,
            ["Salary"] = salaryColumn
        };

        var aggregations = new List<GroupedAggregation>
        {
            new("TotalSalary", ColumnExpressions.Col("Salary"), AggregationFunctions.Sum())
        };

        var operation = new GroupByOperation(
            new[] { ColumnExpressions.Col("Name") },
            null,
            aggregations);

        var result = operation.Execute(input);

        Assert.That(result, Has.Count.EqualTo(2));
        Assert.That(result.ContainsKey("Name"), Is.True);
        Assert.That(result.ContainsKey("TotalSalary"), Is.True);

        var nameResult = result["Name"];
        var salaryResult = result["TotalSalary"];
        Assert.That(nameResult.Length, Is.EqualTo(3));
        Assert.That(salaryResult.Length, Is.EqualTo(3));
    }

    [Test]
    public void Execute_WithMultipleAggregations_ComputesAll()
    {
        var groups = new[] { "A", "B", "A", "B", "A" };
        var values = new[] { 10, 20, 30, 40, 50 };

        var groupColumn = NivaraColumn<string>.Create(groups);
        var valueColumn = NivaraColumn<int>.Create(values);

        var input = new Dictionary<string, IColumn>
        {
            ["Group"] = groupColumn,
            ["Value"] = valueColumn
        };

        var aggregations = new List<GroupedAggregation>
        {
            new("SumVal", ColumnExpressions.Col("Value"), AggregationFunctions.Sum()),
            new("MaxVal", ColumnExpressions.Col("Value"), AggregationFunctions.Max()),
            new("MeanVal", ColumnExpressions.Col("Value"), AggregationFunctions.Mean())
        };

        var operation = new GroupByOperation(
            new[] { ColumnExpressions.Col("Group") },
            null,
            aggregations);

        var result = operation.Execute(input);

        Assert.That(result, Has.Count.EqualTo(4));
        Assert.That(result.ContainsKey("Group"), Is.True);
        Assert.That(result.ContainsKey("SumVal"), Is.True);
        Assert.That(result.ContainsKey("MaxVal"), Is.True);
        Assert.That(result.ContainsKey("MeanVal"), Is.True);
    }

    [Test]
    public void GroupByExtension_WithAggregations_IncludesAggregatedColumns()
    {
        var frame = NivaraFrame.Create(
            ("City", NivaraColumn<string>.Create(new[] { "NYC", "LA", "NYC", "LA", "NYC" })),
            ("Pop", NivaraColumn<int>.Create(new[] { 100, 200, 150, 250, 300 }))
        );

        var aggregations = new Dictionary<string, AggregationFunction>
        {
            ["Pop"] = AggregationFunctions.Sum()
        };

        var result = frame.GroupBy(["City"], aggregations);

        Assert.That(result.ColumnNames, Contains.Item("City"));
        Assert.That(result.ColumnNames, Contains.Item("Pop"));
        Assert.That(result.GetColumn("City").Length, Is.EqualTo(2));
        Assert.That(result.GetColumn("Pop").Length, Is.EqualTo(2));
    }

    [Test]
    public void GroupByExtension_WithAggregations_ResultValuesCorrect()
    {
        var frame = NivaraFrame.Create(
            ("Dept", NivaraColumn<string>.Create(new[] { "Eng", "HR", "Eng", "HR", "Eng" })),
            ("Salary", NivaraColumn<int>.Create(new[] { 100, 200, 150, 250, 300 }))
        );

        var aggregations = new Dictionary<string, AggregationFunction>
        {
            ["Salary"] = AggregationFunctions.Sum()
        };

        var result = frame.GroupBy(["Dept"], aggregations);

        var deptCol = result.GetColumn("Dept");
        var totalCol = result.GetColumn("Salary");

        var engIdx = -1;
        var hrIdx = -1;
        for (int i = 0; i < deptCol.Length; i++)
        {
            if ((string)deptCol.GetValue(i)! == "Eng") engIdx = i;
            if ((string)deptCol.GetValue(i)! == "HR") hrIdx = i;
        }

        Assert.That(engIdx, Is.GreaterThanOrEqualTo(0));
        Assert.That(hrIdx, Is.GreaterThanOrEqualTo(0));
        Assert.That(totalCol.GetValue(engIdx), Is.EqualTo(550L)); // 100+150+300
        Assert.That(totalCol.GetValue(hrIdx), Is.EqualTo(450L)); // 200+250
    }
}
