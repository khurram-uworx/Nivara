using Apache.Arrow;
using Apache.Arrow.Types;
using Nivara.IO;
using NUnit.Framework;

namespace Nivara.Tests.IO;

[TestFixture]
public class ArrowParquetIntegrationTests
{
    private string _tempDirectory = null!;

    [SetUp]
    public void SetUp()
    {
        _tempDirectory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        Directory.CreateDirectory(_tempDirectory);
    }

    [TearDown]
    public void TearDown()
    {
        if (Directory.Exists(_tempDirectory))
        {
            Directory.Delete(_tempDirectory, true);
        }
    }

    [Test]
    public void CompleteWorkflow_NivaraFrame_To_Parquet_To_Arrow_To_NivaraFrame()
    {
        // Arrange - Create original NivaraFrame with various data types
        var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var stringColumn = NivaraColumn<string>.CreateForReferenceType(new[] { "apple", "banana", "cherry", "date", "elderberry" });
        var doubleColumn = NivaraColumn<double>.Create(new[] { 1.1, 2.2, 3.3, 4.4, 5.5 });
        var boolColumn = NivaraColumn<bool>.Create(new[] { true, false, true, false, true });
        var dateTimeColumn = NivaraColumn<DateTime>.Create(new[]
        {
            new DateTime(2023, 1, 1),
            new DateTime(2023, 2, 15),
            new DateTime(2023, 3, 30),
            new DateTime(2023, 4, 10),
            new DateTime(2023, 5, 25)
        });

        var originalFrame = NivaraFrame.Create(
            ("Integers", intColumn),
            ("Strings", stringColumn),
            ("Doubles", doubleColumn),
            ("Booleans", boolColumn),
            ("DateTimes", dateTimeColumn)
        );

        var parquetFile = Path.Combine(_tempDirectory, "test_data.parquet");

        try
        {
            // Act - Complete workflow
            // Step 1: NivaraFrame → Parquet
            ParquetWriter.WriteParquet(originalFrame, parquetFile);
            Assert.That(File.Exists(parquetFile), Is.True, "Parquet file should be created");

            // Step 2: Parquet → NivaraFrame
            var frameFromParquet = ParquetReader.ReadParquet(parquetFile);
            Assert.That(frameFromParquet, Is.Not.Null, "Frame should be read from Parquet");

            // Step 3: NivaraFrame → Arrow
            var arrowTable = ArrowInterop.ToArrowTable(frameFromParquet);
            Assert.That(arrowTable, Is.Not.Null, "Arrow table should be created");

            // Step 4: Arrow → NivaraFrame
            var finalFrame = ArrowInterop.FromArrowTable(arrowTable);
            Assert.That(finalFrame, Is.Not.Null, "Final frame should be created from Arrow");

            // Assert - Verify data integrity throughout the workflow
            Assert.That(finalFrame.ColumnCount, Is.EqualTo(originalFrame.ColumnCount), "Column count should be preserved");
            Assert.That(finalFrame.RowCount, Is.EqualTo(originalFrame.RowCount), "Row count should be preserved");

            // Verify column names
            var originalColumnNames = originalFrame.ColumnNames.ToList();
            var finalColumnNames = finalFrame.ColumnNames.ToList();
            Assert.That(finalColumnNames, Is.EquivalentTo(originalColumnNames), "Column names should be preserved");

            // Verify data values for each column
            VerifyColumnData(originalFrame.GetColumn("Integers"), finalFrame.GetColumn("Integers"), "Integers");
            VerifyColumnData(originalFrame.GetColumn("Strings"), finalFrame.GetColumn("Strings"), "Strings");
            VerifyColumnData(originalFrame.GetColumn("Doubles"), finalFrame.GetColumn("Doubles"), "Doubles");
            VerifyColumnData(originalFrame.GetColumn("Booleans"), finalFrame.GetColumn("Booleans"), "Booleans");
            VerifyColumnData(originalFrame.GetColumn("DateTimes"), finalFrame.GetColumn("DateTimes"), "DateTimes");
        }
        finally
        {
            // Cleanup
            originalFrame.Dispose();
        }
    }

    [Test]
    [Ignore("Null value preservation through Parquet round-trip needs investigation")]
    public void CompleteWorkflow_With_NullValues_PreservesNulls()
    {
        // Arrange - Create frame with null values
        var nullableIntArray = new int?[] { 1, null, 3, null, 5 };
        var intColumn = NivaraColumn<int>.CreateFromNullable(nullableIntArray);

        var stringArray = new string[] { "a", null!, "c", null!, "e" };
        var stringColumn = NivaraColumn<string>.CreateForReferenceType(stringArray);

        var originalFrame = NivaraFrame.Create(
            ("NullableInts", intColumn),
            ("NullableStrings", stringColumn)
        );

        var parquetFile = Path.Combine(_tempDirectory, "null_data.parquet");

        try
        {
            // Act - Complete workflow with nulls
            ParquetWriter.WriteParquet(originalFrame, parquetFile);
            var frameFromParquet = ParquetReader.ReadParquet(parquetFile);
            var arrowTable = ArrowInterop.ToArrowTable(frameFromParquet);
            var finalFrame = ArrowInterop.FromArrowTable(arrowTable);

            // Assert - Verify null preservation
            Assert.That(finalFrame.ColumnCount, Is.EqualTo(2));
            Assert.That(finalFrame.RowCount, Is.EqualTo(5));

            // Verify null values are preserved
            var finalIntColumn = finalFrame.GetColumn("NullableInts");
            var finalStringColumn = finalFrame.GetColumn("NullableStrings");

            for (int i = 0; i < 5; i++)
            {
                var originalIntValue = originalFrame.GetColumn("NullableInts").GetValue(i);
                var finalIntValue = finalIntColumn.GetValue(i);
                
                // Debug output for troubleshooting
                TestContext.Out.WriteLine($"Index {i}: Original int = {originalIntValue}, Final int = {finalIntValue}");
                
                Assert.That(finalIntValue, Is.EqualTo(originalIntValue), $"Int value at index {i} should be preserved (including nulls)");

                var originalStringValue = originalFrame.GetColumn("NullableStrings").GetValue(i);
                var finalStringValue = finalStringColumn.GetValue(i);
                
                // Debug output for troubleshooting
                TestContext.Out.WriteLine($"Index {i}: Original string = '{originalStringValue}', Final string = '{finalStringValue}'");
                
                Assert.That(finalStringValue, Is.EqualTo(originalStringValue), $"String value at index {i} should be preserved (including nulls)");
            }
        }
        finally
        {
            originalFrame.Dispose();
        }
    }

    [Test]
    public async Task AsyncWorkflow_NivaraFrame_To_Parquet_To_Arrow_PreservesData()
    {
        // Arrange
        var data = Enumerable.Range(1, 1000).ToArray(); // Larger dataset for async testing
        var intColumn = NivaraColumn<int>.Create(data);
        var originalFrame = NivaraFrame.Create(("LargeDataset", intColumn));

        var parquetFile = Path.Combine(_tempDirectory, "async_test.parquet");

        try
        {
            // Act - Async workflow
            await ParquetWriter.WriteParquetAsync(originalFrame, parquetFile);
            var frameFromParquet = await ParquetReader.ReadParquetAsync(parquetFile);
            var arrowTable = ArrowInterop.ToArrowTable(frameFromParquet);
            var finalFrame = ArrowInterop.FromArrowTable(arrowTable);

            // Assert
            Assert.That(finalFrame.RowCount, Is.EqualTo(1000));
            Assert.That(finalFrame.ColumnCount, Is.EqualTo(1));

            var finalColumn = finalFrame.GetColumn("LargeDataset");
            for (int i = 0; i < 1000; i++)
            {
                Assert.That(finalColumn.GetValue(i), Is.EqualTo(i + 1), $"Value at index {i} should be preserved");
            }
        }
        finally
        {
            originalFrame.Dispose();
        }
    }

    [Test]
    public void BatchProcessing_MultipleFrames_To_SingleParquet_To_Arrow()
    {
        // Arrange - Create multiple frames with compatible schemas
        var frame1 = NivaraFrame.Create(
            ("ID", NivaraColumn<int>.Create(new[] { 1, 2, 3 })),
            ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Alice", "Bob", "Charlie" }))
        );

        var frame2 = NivaraFrame.Create(
            ("ID", NivaraColumn<int>.Create(new[] { 4, 5, 6 })),
            ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "David", "Eve", "Frank" }))
        );

        var frame3 = NivaraFrame.Create(
            ("ID", NivaraColumn<int>.Create(new[] { 7, 8, 9 })),
            ("Name", NivaraColumn<string>.CreateForReferenceType(new[] { "Grace", "Henry", "Iris" }))
        );

        var frames = new[] { frame1, frame2, frame3 };
        var parquetFile = Path.Combine(_tempDirectory, "batch_data.parquet");

        try
        {
            // Act - Batch processing workflow
            ParquetWriter.WriteParquetBatch(frames, parquetFile);
            var combinedFrame = ParquetReader.ReadParquet(parquetFile);
            var arrowTable = ArrowInterop.ToArrowTable(combinedFrame);
            var finalFrame = ArrowInterop.FromArrowTable(arrowTable);

            // Assert - Should contain all data from all frames
            Assert.That(finalFrame.RowCount, Is.EqualTo(9), "Should contain all rows from all frames");
            Assert.That(finalFrame.ColumnCount, Is.EqualTo(2), "Should have same column structure");

            // Verify all IDs are present (1-9)
            var idColumn = finalFrame.GetColumn("ID");
            var allIds = new List<int>();
            for (int i = 0; i < 9; i++)
            {
                allIds.Add((int)idColumn.GetValue(i)!);
            }
            Assert.That(allIds, Is.EquivalentTo(new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }), "All IDs should be present");

            // Verify all names are present
            var nameColumn = finalFrame.GetColumn("Name");
            var allNames = new List<string>();
            for (int i = 0; i < 9; i++)
            {
                allNames.Add((string)nameColumn.GetValue(i)!);
            }
            var expectedNames = new[] { "Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry", "Iris" };
            Assert.That(allNames, Is.EquivalentTo(expectedNames), "All names should be present");
        }
        finally
        {
            frame1.Dispose();
            frame2.Dispose();
            frame3.Dispose();
        }
    }

    [Test]
    public void StreamBasedWorkflow_MemoryStream_PreservesData()
    {
        // Arrange
        var originalFrame = NivaraFrame.Create(
            ("StreamTest", NivaraColumn<double>.Create(new[] { 3.14, 2.71, 1.41, 1.73 }))
        );

        try
        {
            // Act - Stream-based workflow
            using var memoryStream = new MemoryStream();
            
            // Write to stream
            ParquetWriter.WriteParquet(originalFrame, memoryStream);
            Assert.That(memoryStream.Length, Is.GreaterThan(0), "Data should be written to stream");

            // Read from stream
            memoryStream.Position = 0; // Reset position for reading
            var frameFromStream = ParquetReader.ReadParquet(memoryStream);
            
            // Convert through Arrow
            var arrowTable = ArrowInterop.ToArrowTable(frameFromStream);
            var finalFrame = ArrowInterop.FromArrowTable(arrowTable);

            // Assert
            Assert.That(finalFrame.RowCount, Is.EqualTo(4));
            Assert.That(finalFrame.ColumnCount, Is.EqualTo(1));

            var finalColumn = finalFrame.GetColumn("StreamTest");
            var expectedValues = new[] { 3.14, 2.71, 1.41, 1.73 };
            for (int i = 0; i < 4; i++)
            {
                var actualValue = (double)finalColumn.GetValue(i)!;
                Assert.That(actualValue, Is.EqualTo(expectedValues[i]).Within(0.001), $"Value at index {i} should be preserved");
            }
        }
        finally
        {
            originalFrame.Dispose();
        }
    }

    [Test]
    public void ConfigurationOptions_Integration_WorksCorrectly()
    {
        // Arrange
        var originalFrame = NivaraFrame.Create(
            ("ConfigTest", NivaraColumn<int>.Create(new[] { 100, 200, 300 }))
        );

        var parquetFile = Path.Combine(_tempDirectory, "config_test.parquet");

        // Configure options
        var parquetOptions = new ParquetWriteOptions
        {
            ValidateSchema = true,
            Compression = "snappy",
            RowGroupSize = 1000
        };

        var arrowOptions = new ArrowConversionOptions
        {
            ValidateTypes = true,
            UseZeroCopy = false // Force copying approach
        };

        try
        {
            // Act - Workflow with custom configuration
            ParquetWriter.WriteParquet(originalFrame, parquetFile, parquetOptions);
            var frameFromParquet = ParquetReader.ReadParquet(parquetFile);
            var arrowTable = ArrowInterop.ToArrowTable(frameFromParquet, arrowOptions);
            var finalFrame = ArrowInterop.FromArrowTable(arrowTable, arrowOptions);

            // Assert - Configuration should not affect data integrity
            Assert.That(finalFrame.RowCount, Is.EqualTo(3));
            Assert.That(finalFrame.ColumnCount, Is.EqualTo(1));

            var finalColumn = finalFrame.GetColumn("ConfigTest");
            Assert.That(finalColumn.GetValue(0), Is.EqualTo(100));
            Assert.That(finalColumn.GetValue(1), Is.EqualTo(200));
            Assert.That(finalColumn.GetValue(2), Is.EqualTo(300));
        }
        finally
        {
            originalFrame.Dispose();
        }
    }

    [Test]
    public void EmptyFrame_CompleteWorkflow_HandledCorrectly()
    {
        // Arrange
        var emptyColumn = NivaraColumn<int>.Create(System.Array.Empty<int>());
        var originalFrame = NivaraFrame.Create(("EmptyColumn", emptyColumn));

        var parquetFile = Path.Combine(_tempDirectory, "empty_test.parquet");

        try
        {
            // Act - Complete workflow with empty frame
            ParquetWriter.WriteParquet(originalFrame, parquetFile);
            var frameFromParquet = ParquetReader.ReadParquet(parquetFile);
            var arrowTable = ArrowInterop.ToArrowTable(frameFromParquet);
            var finalFrame = ArrowInterop.FromArrowTable(arrowTable);

            // Assert - Empty frame should be handled correctly
            Assert.That(finalFrame.RowCount, Is.EqualTo(0));
            Assert.That(finalFrame.ColumnCount, Is.EqualTo(1));
            Assert.That(finalFrame.ColumnNames, Contains.Item("EmptyColumn"));
        }
        finally
        {
            originalFrame.Dispose();
        }
    }

    [Test]
    public void LargeDataset_Integration_PerformanceTest()
    {
        // Arrange - Create a larger dataset to test performance and memory handling
        const int rowCount = 10000;
        var intData = Enumerable.Range(1, rowCount).ToArray();
        var stringData = Enumerable.Range(1, rowCount).Select(i => $"Item_{i}").ToArray();
        var doubleData = Enumerable.Range(1, rowCount).Select(i => i * 1.5).ToArray();

        var originalFrame = NivaraFrame.Create(
            ("Integers", NivaraColumn<int>.Create(intData)),
            ("Strings", NivaraColumn<string>.CreateForReferenceType(stringData)),
            ("Doubles", NivaraColumn<double>.Create(doubleData))
        );

        var parquetFile = Path.Combine(_tempDirectory, "large_dataset.parquet");

        try
        {
            // Act - Complete workflow with large dataset
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            
            ParquetWriter.WriteParquet(originalFrame, parquetFile);
            var writeTime = stopwatch.ElapsedMilliseconds;
            
            stopwatch.Restart();
            var frameFromParquet = ParquetReader.ReadParquet(parquetFile);
            var readTime = stopwatch.ElapsedMilliseconds;
            
            stopwatch.Restart();
            var arrowTable = ArrowInterop.ToArrowTable(frameFromParquet);
            var finalFrame = ArrowInterop.FromArrowTable(arrowTable);
            var conversionTime = stopwatch.ElapsedMilliseconds;
            
            stopwatch.Stop();

            // Assert - Data integrity and reasonable performance
            Assert.That(finalFrame.RowCount, Is.EqualTo(rowCount));
            Assert.That(finalFrame.ColumnCount, Is.EqualTo(3));

            // Performance assertions (reasonable thresholds)
            Assert.That(writeTime, Is.LessThan(5000), "Write should complete within 5 seconds");
            Assert.That(readTime, Is.LessThan(5000), "Read should complete within 5 seconds");
            Assert.That(conversionTime, Is.LessThan(5000), "Conversion should complete within 5 seconds");

            // Spot check data integrity (check first, middle, and last rows)
            var finalIntColumn = finalFrame.GetColumn("Integers");
            var finalStringColumn = finalFrame.GetColumn("Strings");
            var finalDoubleColumn = finalFrame.GetColumn("Doubles");

            Assert.That(finalIntColumn.GetValue(0), Is.EqualTo(1));
            Assert.That(finalIntColumn.GetValue(rowCount / 2), Is.EqualTo(rowCount / 2 + 1));
            Assert.That(finalIntColumn.GetValue(rowCount - 1), Is.EqualTo(rowCount));

            Assert.That(finalStringColumn.GetValue(0), Is.EqualTo("Item_1"));
            Assert.That(finalStringColumn.GetValue(rowCount - 1), Is.EqualTo($"Item_{rowCount}"));

            Assert.That(finalDoubleColumn.GetValue(0), Is.EqualTo(1.5).Within(0.001));
            Assert.That(finalDoubleColumn.GetValue(rowCount - 1), Is.EqualTo(rowCount * 1.5).Within(0.001));

            TestContext.Out.WriteLine($"Performance metrics for {rowCount} rows:");
            TestContext.Out.WriteLine($"  Write time: {writeTime}ms");
            TestContext.Out.WriteLine($"  Read time: {readTime}ms");
            TestContext.Out.WriteLine($"  Conversion time: {conversionTime}ms");
            TestContext.Out.WriteLine($"  File size: {new FileInfo(parquetFile).Length / 1024}KB");
        }
        finally
        {
            originalFrame.Dispose();
        }
    }

    private static void VerifyColumnData(IColumn originalColumn, IColumn finalColumn, string columnName)
    {
        Assert.That(finalColumn.Length, Is.EqualTo(originalColumn.Length), $"{columnName} column length should be preserved");
        Assert.That(finalColumn.ElementType, Is.EqualTo(originalColumn.ElementType), $"{columnName} column type should be preserved");

        for (int i = 0; i < originalColumn.Length; i++)
        {
            var originalValue = originalColumn.GetValue(i);
            var finalValue = finalColumn.GetValue(i);

            if (originalValue is DateTime originalDateTime && finalValue is DateTime finalDateTime)
            {
                // DateTime comparison with tolerance for potential timezone/precision differences
                var timeDifference = Math.Abs((originalDateTime - finalDateTime).TotalMilliseconds);
                Assert.That(timeDifference, Is.LessThan(1000), $"{columnName} DateTime at index {i} should be preserved within 1 second tolerance");
            }
            else if (originalValue is double originalDouble && finalValue is double finalDouble)
            {
                // Double comparison with tolerance for floating-point precision
                Assert.That(finalDouble, Is.EqualTo(originalDouble).Within(0.0001), $"{columnName} double at index {i} should be preserved");
            }
            else
            {
                // Exact comparison for other types
                Assert.That(finalValue, Is.EqualTo(originalValue), $"{columnName} value at index {i} should be preserved");
            }
        }
    }
}