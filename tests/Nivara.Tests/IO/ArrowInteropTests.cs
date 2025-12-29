using Apache.Arrow;
using Apache.Arrow.Arrays;
using Apache.Arrow.Types;
using Nivara.IO;
using NUnit.Framework;

namespace Nivara.Tests.IO;

[TestFixture]
public class ArrowInteropTests
{
    [Test]
    public void ToArrowTable_EmptyFrame_ReturnsEmptyTable()
    {
        // Arrange
        var emptyColumn = NivaraColumn<int>.Create(System.Array.Empty<int>());
        var frame = NivaraFrame.Create(("TestColumn", emptyColumn));
        
        // Act
        var result = ArrowInterop.ToArrowTable(frame);
        
        // Assert
        Assert.That(result, Is.Not.Null);
        Assert.That(result.ColumnCount, Is.EqualTo(1));
        Assert.That(result.RowCount, Is.EqualTo(0));
    }

    [Test]
    public void ToArrowTable_SimpleIntColumn_ConvertsCorrectly()
    {
        // Arrange
        var data = new[] { 1, 2, 3, 4, 5 };
        var column = NivaraColumn<int>.Create(data);
        var frame = NivaraFrame.Create(("Numbers", column));
        
        // Act
        var result = ArrowInterop.ToArrowTable(frame);
        
        // Assert
        Assert.That(result, Is.Not.Null);
        Assert.That(result.ColumnCount, Is.EqualTo(1));
        Assert.That(result.RowCount, Is.EqualTo(5));
        
        var arrowColumn = result.Column(0);
        Assert.That(arrowColumn.Name, Is.EqualTo("Numbers"));
        Assert.That(arrowColumn.Data.DataType, Is.InstanceOf<Int32Type>());
    }

    [Test]
    public void ToArrowTable_MultipleColumns_ConvertsCorrectly()
    {
        // Arrange
        var intData = new[] { 1, 2, 3 };
        var stringData = new[] { "a", "b", "c" };
        var boolData = new[] { true, false, true };
        
        var intColumn = NivaraColumn<int>.Create(intData);
        var stringColumn = NivaraColumn<string>.Create(stringData);
        var boolColumn = NivaraColumn<bool>.Create(boolData);
        
        var frame = NivaraFrame.Create(
            ("Integers", intColumn),
            ("Strings", stringColumn),
            ("Booleans", boolColumn)
        );
        
        // Act
        var result = ArrowInterop.ToArrowTable(frame);
        
        // Assert
        Assert.That(result, Is.Not.Null);
        Assert.That(result.ColumnCount, Is.EqualTo(3));
        Assert.That(result.RowCount, Is.EqualTo(3));
        
        // Check column names and types
        Assert.That(result.Schema.GetFieldByIndex(0).Name, Is.EqualTo("Integers"));
        Assert.That(result.Schema.GetFieldByIndex(0).DataType, Is.InstanceOf<Int32Type>());
        
        Assert.That(result.Schema.GetFieldByIndex(1).Name, Is.EqualTo("Strings"));
        Assert.That(result.Schema.GetFieldByIndex(1).DataType, Is.InstanceOf<StringType>());
        
        Assert.That(result.Schema.GetFieldByIndex(2).Name, Is.EqualTo("Booleans"));
        Assert.That(result.Schema.GetFieldByIndex(2).DataType, Is.InstanceOf<BooleanType>());
    }

    [Test]
    public void FromArrowTable_SimpleTable_ConvertsCorrectly()
    {
        // Arrange
        var intBuilder = new Int32Array.Builder();
        intBuilder.Append(1);
        intBuilder.Append(2);
        intBuilder.Append(3);
        var intArray = intBuilder.Build();
        
        var field = new Field("Numbers", Int32Type.Default, true);
        var schema = new Apache.Arrow.Schema(new[] { field }, null);
        var recordBatch = new RecordBatch(schema, new IArrowArray[] { intArray }, 3);
        var table = Table.TableFromRecordBatches(schema, new[] { recordBatch });
        
        // Act
        var result = ArrowInterop.FromArrowTable(table);
        
        // Assert
        Assert.That(result, Is.Not.Null);
        Assert.That(result.ColumnCount, Is.EqualTo(1));
        Assert.That(result.RowCount, Is.EqualTo(3));
        Assert.That(result.ColumnNames, Contains.Item("Numbers"));
        
        var column = result.GetColumn("Numbers");
        Assert.That(column.ElementType, Is.EqualTo(typeof(int)));
        Assert.That(column.Length, Is.EqualTo(3));
    }

    [Test]
    public void ToArrowTable_WithNullValues_HandlesNullsCorrectly()
    {
        // Arrange
        var data = new int?[] { 1, null, 3, null, 5 };
        var column = NivaraColumn<int>.CreateFromNullable(data);
        var frame = NivaraFrame.Create(("NullableNumbers", column));
        
        // Act
        var result = ArrowInterop.ToArrowTable(frame);
        
        // Assert
        Assert.That(result, Is.Not.Null);
        Assert.That(result.ColumnCount, Is.EqualTo(1));
        Assert.That(result.RowCount, Is.EqualTo(5));
        
        var arrowColumn = result.Column(0);
        var chunk = arrowColumn.Data.Array(0) as Int32Array;
        Assert.That(chunk, Is.Not.Null);
        
        // Check null values
        Assert.That(chunk!.IsNull(1), Is.True);
        Assert.That(chunk.IsNull(3), Is.True);
        Assert.That(chunk.IsNull(0), Is.False);
        Assert.That(chunk.IsNull(2), Is.False);
        Assert.That(chunk.IsNull(4), Is.False);
    }

    [Test]
    public void RoundTrip_SimpleData_PreservesData()
    {
        // Arrange
        var originalData = new[] { 10, 20, 30, 40, 50 };
        var originalColumn = NivaraColumn<int>.Create(originalData);
        var originalFrame = NivaraFrame.Create(("Data", originalColumn));
        
        // Act - Convert to Arrow and back
        var arrowTable = ArrowInterop.ToArrowTable(originalFrame);
        var resultFrame = ArrowInterop.FromArrowTable(arrowTable);
        
        // Assert
        Assert.That(resultFrame.ColumnCount, Is.EqualTo(1));
        Assert.That(resultFrame.RowCount, Is.EqualTo(5));
        Assert.That(resultFrame.ColumnNames, Contains.Item("Data"));
        
        var resultColumn = resultFrame.GetColumn("Data");
        Assert.That(resultColumn.ElementType, Is.EqualTo(typeof(int)));
        Assert.That(resultColumn.Length, Is.EqualTo(5));
        
        // Check data values
        for (int i = 0; i < originalData.Length; i++)
        {
            var originalValue = originalColumn[i];
            var resultValue = resultColumn.GetValue(i);
            Assert.That(resultValue, Is.EqualTo(originalValue), $"Value at index {i} should be preserved");
        }
    }

    [Test]
    public void ToArrowArray_NullArgument_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => ArrowInterop.ToArrowArray<int>(null!));
    }

    [Test]
    public void FromArrowArray_NullArgument_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => ArrowInterop.FromArrowArray<int>(null!));
    }

    [Test]
    public void ToArrowArray_SimpleSeries_ConvertsCorrectly()
    {
        // Arrange
        var data = new[] { 10, 20, 30 };
        var series = NivaraSeries<int>.Create(data);
        
        // Act
        var result = ArrowInterop.ToArrowArray(series);
        
        // Assert
        Assert.That(result, Is.Not.Null);
        Assert.That(result.Length, Is.EqualTo(3));
        Assert.That(result, Is.InstanceOf<Int32Array>());
        
        var intArray = (Int32Array)result;
        Assert.That(intArray.GetValue(0), Is.EqualTo(10));
        Assert.That(intArray.GetValue(1), Is.EqualTo(20));
        Assert.That(intArray.GetValue(2), Is.EqualTo(30));
    }

    [Test]
    public void FromArrowArray_SimpleArray_ConvertsCorrectly()
    {
        // Arrange
        var builder = new Int32Array.Builder();
        builder.Append(100);
        builder.Append(200);
        builder.Append(300);
        var arrowArray = builder.Build();
        
        // Act
        var result = ArrowInterop.FromArrowArray<int>(arrowArray);
        
        // Assert
        Assert.That(result, Is.Not.Null);
        Assert.That(result.Length, Is.EqualTo(3));
        Assert.That(result[0], Is.EqualTo(100));
        Assert.That(result[1], Is.EqualTo(200));
        Assert.That(result[2], Is.EqualTo(300));
    }

    [Test]
    public void SeriesRoundTrip_SimpleData_PreservesData()
    {
        // Arrange
        var originalData = new[] { 5, 15, 25, 35 };
        var originalSeries = NivaraSeries<int>.Create(originalData);
        
        // Act - Convert to Arrow and back
        var arrowArray = ArrowInterop.ToArrowArray(originalSeries);
        var resultSeries = ArrowInterop.FromArrowArray<int>(arrowArray);
        
        // Assert
        Assert.That(resultSeries.Length, Is.EqualTo(4));
        
        for (int i = 0; i < originalData.Length; i++)
        {
            Assert.That(resultSeries[i], Is.EqualTo(originalData[i]), $"Value at index {i} should be preserved");
        }
    }

    [Test]
    public void ToArrowArray_EmptySeries_ReturnsEmptyArray()
    {
        // Arrange
        var emptySeries = NivaraSeries<int>.Create(System.Array.Empty<int>());
        
        // Act
        var result = ArrowInterop.ToArrowArray(emptySeries);
        
        // Assert
        Assert.That(result, Is.Not.Null);
        Assert.That(result.Length, Is.EqualTo(0));
        Assert.That(result, Is.InstanceOf<Int32Array>());
    }

    [Test]
    public void FromArrowArray_EmptyArray_ReturnsEmptySeries()
    {
        // Arrange
        var emptyArray = new Int32Array.Builder().Build();
        
        // Act
        var result = ArrowInterop.FromArrowArray<int>(emptyArray);
        
        // Assert
        Assert.That(result, Is.Not.Null);
        Assert.That(result.Length, Is.EqualTo(0));
    }
}
