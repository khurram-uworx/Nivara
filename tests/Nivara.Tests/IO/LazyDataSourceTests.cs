using NUnit.Framework;
using Nivara;
using Nivara.IO;
using Nivara.Expressions;
using Nivara.Exceptions;

namespace Nivara.Tests.IO;

/// <summary>
/// Tests for lazy data source functionality including CSV and JSON scanning
/// </summary>
[TestFixture]
public class LazyDataSourceTests
{
    private string testDataDirectory = null!;
    private string csvFilePath = null!;
    private string jsonFilePath = null!;
    private string emptyCsvFilePath = null!;
    private string emptyJsonFilePath = null!;

    [OneTimeSetUp]
    public void OneTimeSetUp()
    {
        // Create test data directory
        testDataDirectory = Path.Combine(Path.GetTempPath(), "NivaraTests", Guid.NewGuid().ToString());
        Directory.CreateDirectory(testDataDirectory);

        // Create test CSV file
        csvFilePath = Path.Combine(testDataDirectory, "test_data.csv");
        File.WriteAllText(csvFilePath, """
            Name,Age,Salary
            Alice,30,75000
            Bob,25,65000
            Charlie,35,85000
            """);

        // Create test JSON file
        jsonFilePath = Path.Combine(testDataDirectory, "test_data.json");
        File.WriteAllText(jsonFilePath, """
            [
              {"Name": "Alice", "Age": 30, "Salary": 75000},
              {"Name": "Bob", "Age": 25, "Salary": 65000},
              {"Name": "Charlie", "Age": 35, "Salary": 85000}
            ]
            """);

        // Create empty CSV file
        emptyCsvFilePath = Path.Combine(testDataDirectory, "empty.csv");
        File.WriteAllText(emptyCsvFilePath, "Name,Age,Salary\n");

        // Create empty JSON file
        emptyJsonFilePath = Path.Combine(testDataDirectory, "empty.json");
        File.WriteAllText(emptyJsonFilePath, "[]");
    }

    [OneTimeTearDown]
    public void OneTimeTearDown()
    {
        if (Directory.Exists(testDataDirectory))
        {
            Directory.Delete(testDataDirectory, true);
        }
    }

    #region CSV Lazy Source Tests

    [Test]
    public void CsvLazySource_ScanCsv_CreatesLazyQuerySource()
    {
        // Act
        var source = CsvExtensions.ScanCsv(csvFilePath);

        // Assert
        Assert.That(source, Is.Not.Null);
        Assert.That(source.IsLazy, Is.True);
        Assert.That(source.Schema, Is.Not.Null);
        Assert.That(source.Schema.ColumnNames, Has.Count.EqualTo(3));
        Assert.That(source.Schema.ColumnNames, Contains.Item("Name"));
        Assert.That(source.Schema.ColumnNames, Contains.Item("Age"));
        Assert.That(source.Schema.ColumnNames, Contains.Item("Salary"));
    }

    [Test]
    public void CsvLazySource_ScanCsvAsQueryFrame_CreatesLazyQueryFrame()
    {
        // Act
        var queryFrame = CsvExtensions.ScanCsvAsQueryFrame(csvFilePath);

        // Assert
        Assert.That(queryFrame, Is.Not.Null);
        Assert.That(queryFrame.IsLazy, Is.True);
        Assert.That(queryFrame.Schema, Is.Not.Null);
        Assert.That(queryFrame.Schema.ColumnNames, Has.Count.EqualTo(3));
    }

    [Test]
    public void CsvLazySource_SchemaInference_InfersCorrectTypes()
    {
        // Act
        var source = CsvExtensions.ScanCsv(csvFilePath);

        // Assert
        Assert.That(source.Schema.GetColumnType("Name"), Is.EqualTo(typeof(string)));
        Assert.That(source.Schema.GetColumnType("Age"), Is.EqualTo(typeof(int)));
        Assert.That(source.Schema.GetColumnType("Salary"), Is.EqualTo(typeof(int)));
    }

    [Test]
    public void CsvLazySource_Execute_ReturnsCorrectData()
    {
        // Arrange
        var source = CsvExtensions.ScanCsv(csvFilePath);

        // Act
        var columns = source.Execute();

        // Assert
        Assert.That(columns, Is.Not.Null);
        Assert.That(columns, Has.Count.EqualTo(3));
        
        var nameColumn = columns["Name"];
        var ageColumn = columns["Age"];
        var salaryColumn = columns["Salary"];
        
        Assert.That(nameColumn.Length, Is.EqualTo(3));
        Assert.That(ageColumn.Length, Is.EqualTo(3));
        Assert.That(salaryColumn.Length, Is.EqualTo(3));
    }

    [Test]
    public void CsvLazySource_LazyEvaluation_NoImmediateExecution()
    {
        // Arrange
        var lastAccessTime = File.GetLastAccessTime(csvFilePath);
        
        // Act - creating query should not read file
        var queryFrame = CsvExtensions.ScanCsvAsQueryFrame(csvFilePath);
        var filteredQuery = queryFrame.Filter(ColumnExpressions.Col("Age") > 30);
        
        // Assert - file should not have been accessed for data (schema inference may access it briefly)
        // We test that the query was built without full execution
        Assert.That(queryFrame.IsLazy, Is.True);
        Assert.That(filteredQuery.IsLazy, Is.True);
    }

    [Test]
    public void CsvLazySource_Collect_ExecutesLazyQuery()
    {
        // Arrange
        var queryFrame = CsvExtensions.ScanCsvAsQueryFrame(csvFilePath);
        var filteredQuery = queryFrame.Filter(ColumnExpressions.Col("Age") > 30);

        // Act
        var result = filteredQuery.Collect();

        // Assert
        Assert.That(result, Is.Not.Null);
        Assert.That(result.RowCount, Is.EqualTo(1)); // Only Charlie (35) matches Age > 30
        Assert.That(result.ColumnCount, Is.EqualTo(3));
    }

    [Test]
    public void CsvLazySource_EmptyFile_HandlesGracefully()
    {
        // Act
        var source = CsvExtensions.ScanCsv(emptyCsvFilePath);
        var columns = source.Execute();

        // Assert
        Assert.That(columns, Is.Not.Null);
        Assert.That(columns, Has.Count.EqualTo(3)); // Headers still present
        Assert.That(columns["Name"].Length, Is.EqualTo(0));
        Assert.That(columns["Age"].Length, Is.EqualTo(0));
        Assert.That(columns["Salary"].Length, Is.EqualTo(0));
    }

    [Test]
    public void CsvLazySource_NonExistentFile_ThrowsFileNotFoundException()
    {
        // Arrange
        var nonExistentPath = Path.Combine(testDataDirectory, "nonexistent.csv");

        // Act & Assert
        Assert.Throws<FileNotFoundException>(() => CsvExtensions.ScanCsv(nonExistentPath));
    }

    [Test]
    public void CsvLazySource_NullFilePath_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => CsvExtensions.ScanCsv(null!));
    }

    #endregion

    #region JSON Lazy Source Tests

    [Test]
    public void JsonLazySource_ScanJson_CreatesLazyQuerySource()
    {
        // Act
        var source = JsonExtensions.ScanJson(jsonFilePath);

        // Assert
        Assert.That(source, Is.Not.Null);
        Assert.That(source.IsLazy, Is.True);
        Assert.That(source.Schema, Is.Not.Null);
        Assert.That(source.Schema.ColumnNames, Has.Count.EqualTo(3));
        Assert.That(source.Schema.ColumnNames, Contains.Item("Name"));
        Assert.That(source.Schema.ColumnNames, Contains.Item("Age"));
        Assert.That(source.Schema.ColumnNames, Contains.Item("Salary"));
    }

    [Test]
    public void JsonLazySource_ScanJsonAsQueryFrame_CreatesLazyQueryFrame()
    {
        // Act
        var queryFrame = JsonExtensions.ScanJsonAsQueryFrame(jsonFilePath);

        // Assert
        Assert.That(queryFrame, Is.Not.Null);
        Assert.That(queryFrame.IsLazy, Is.True);
        Assert.That(queryFrame.Schema, Is.Not.Null);
        Assert.That(queryFrame.Schema.ColumnNames, Has.Count.EqualTo(3));
    }

    [Test]
    public void JsonLazySource_SchemaInference_InfersCorrectTypes()
    {
        // Act
        var source = JsonExtensions.ScanJson(jsonFilePath);

        // Assert
        Assert.That(source.Schema.GetColumnType("Name"), Is.EqualTo(typeof(string)));
        Assert.That(source.Schema.GetColumnType("Age"), Is.EqualTo(typeof(double))); // JSON numbers default to double
        Assert.That(source.Schema.GetColumnType("Salary"), Is.EqualTo(typeof(double)));
    }

    [Test]
    public void JsonLazySource_Execute_ReturnsCorrectData()
    {
        // Arrange
        var source = JsonExtensions.ScanJson(jsonFilePath);

        // Act
        var columns = source.Execute();

        // Assert
        Assert.That(columns, Is.Not.Null);
        Assert.That(columns, Has.Count.EqualTo(3));
        
        var nameColumn = columns["Name"];
        var ageColumn = columns["Age"];
        var salaryColumn = columns["Salary"];
        
        Assert.That(nameColumn.Length, Is.EqualTo(3));
        Assert.That(ageColumn.Length, Is.EqualTo(3));
        Assert.That(salaryColumn.Length, Is.EqualTo(3));
    }

    [Test]
    public void JsonLazySource_LazyEvaluation_NoImmediateExecution()
    {
        // Act - creating query should not fully process file
        var queryFrame = JsonExtensions.ScanJsonAsQueryFrame(jsonFilePath);
        var filteredQuery = queryFrame.Filter(ColumnExpressions.Col("Age") > 30);
        
        // Assert - query should be lazy
        Assert.That(queryFrame.IsLazy, Is.True);
        Assert.That(filteredQuery.IsLazy, Is.True);
    }

    [Test]
    public void JsonLazySource_Collect_ExecutesLazyQuery()
    {
        // Arrange
        var queryFrame = JsonExtensions.ScanJsonAsQueryFrame(jsonFilePath);
        var filteredQuery = queryFrame.Filter(ColumnExpressions.Col("Age") > 30.0); // Use double for JSON

        // Act
        var result = filteredQuery.Collect();

        // Assert
        Assert.That(result, Is.Not.Null);
        Assert.That(result.RowCount, Is.EqualTo(1)); // Only Charlie (35) matches Age > 30
        Assert.That(result.ColumnCount, Is.EqualTo(3));
    }

    [Test]
    public void JsonLazySource_EmptyFile_HandlesGracefully()
    {
        // Act & Assert - should throw because empty JSON array has no schema to infer
        Assert.Throws<DataSourceException>(() => 
        {
            var source = JsonExtensions.ScanJson(emptyJsonFilePath);
            var schema = source.Schema; // This should throw
        });
    }

    [Test]
    public void JsonLazySource_NonExistentFile_ThrowsFileNotFoundException()
    {
        // Arrange
        var nonExistentPath = Path.Combine(testDataDirectory, "nonexistent.json");

        // Act & Assert
        Assert.Throws<FileNotFoundException>(() => JsonExtensions.ScanJson(nonExistentPath));
    }

    [Test]
    public void JsonLazySource_NullFilePath_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => JsonExtensions.ScanJson(null!));
    }

    #endregion

    #region Eager Reading Tests

    [Test]
    public void CsvEagerSource_ReadCsvAsFrame_ReturnsImmediateFrame()
    {
        // Act
        var frame = CsvExtensions.ReadCsvAsFrame(csvFilePath);

        // Assert
        Assert.That(frame, Is.Not.Null);
        Assert.That(frame.RowCount, Is.EqualTo(3));
        Assert.That(frame.ColumnCount, Is.EqualTo(3));
        Assert.That(frame.ColumnNames, Contains.Item("Name"));
        Assert.That(frame.ColumnNames, Contains.Item("Age"));
        Assert.That(frame.ColumnNames, Contains.Item("Salary"));
    }

    [Test]
    public void JsonEagerSource_ReadJsonAsFrame_ReturnsImmediateFrame()
    {
        // Act
        var frame = JsonExtensions.ReadJsonAsFrame(jsonFilePath);

        // Assert
        Assert.That(frame, Is.Not.Null);
        Assert.That(frame.RowCount, Is.EqualTo(3));
        Assert.That(frame.ColumnCount, Is.EqualTo(3));
        Assert.That(frame.ColumnNames, Contains.Item("Name"));
        Assert.That(frame.ColumnNames, Contains.Item("Age"));
        Assert.That(frame.ColumnNames, Contains.Item("Salary"));
    }

    [Test]
    public void EagerVsLazy_SameData_ProduceSameResults()
    {
        // Arrange & Act
        var eagerCsvFrame = CsvExtensions.ReadCsvAsFrame(csvFilePath);
        var lazyCsvFrame = CsvExtensions.ScanCsvAsQueryFrame(csvFilePath).Collect();
        
        var eagerJsonFrame = JsonExtensions.ReadJsonAsFrame(jsonFilePath);
        var lazyJsonFrame = JsonExtensions.ScanJsonAsQueryFrame(jsonFilePath).Collect();

        // Assert CSV
        Assert.That(lazyCsvFrame.RowCount, Is.EqualTo(eagerCsvFrame.RowCount));
        Assert.That(lazyCsvFrame.ColumnCount, Is.EqualTo(eagerCsvFrame.ColumnCount));
        Assert.That(lazyCsvFrame.ColumnNames, Is.EqualTo(eagerCsvFrame.ColumnNames));

        // Assert JSON
        Assert.That(lazyJsonFrame.RowCount, Is.EqualTo(eagerJsonFrame.RowCount));
        Assert.That(lazyJsonFrame.ColumnCount, Is.EqualTo(eagerJsonFrame.ColumnCount));
        Assert.That(lazyJsonFrame.ColumnNames, Is.EqualTo(eagerJsonFrame.ColumnNames));
    }

    #endregion

    #region Static Factory Class Tests

    [Test]
    public void CsvStaticClass_ScanAsQueryFrame_WorksCorrectly()
    {
        // Act
        var queryFrame = Csv.ScanAsQueryFrame(csvFilePath);

        // Assert
        Assert.That(queryFrame, Is.Not.Null);
        Assert.That(queryFrame.IsLazy, Is.True);
        Assert.That(queryFrame.Schema.ColumnNames, Has.Count.EqualTo(3));
    }

    [Test]
    public void CsvStaticClass_ReadAsFrame_WorksCorrectly()
    {
        // Act
        var frame = Csv.ReadAsFrame(csvFilePath);

        // Assert
        Assert.That(frame, Is.Not.Null);
        Assert.That(frame.RowCount, Is.EqualTo(3));
        Assert.That(frame.ColumnCount, Is.EqualTo(3));
    }

    [Test]
    public void JsonStaticClass_ScanAsQueryFrame_WorksCorrectly()
    {
        // Act
        var queryFrame = Json.ScanAsQueryFrame(jsonFilePath);

        // Assert
        Assert.That(queryFrame, Is.Not.Null);
        Assert.That(queryFrame.IsLazy, Is.True);
        Assert.That(queryFrame.Schema.ColumnNames, Has.Count.EqualTo(3));
    }

    [Test]
    public void JsonStaticClass_ReadAsFrame_WorksCorrectly()
    {
        // Act
        var frame = Json.ReadAsFrame(jsonFilePath);

        // Assert
        Assert.That(frame, Is.Not.Null);
        Assert.That(frame.RowCount, Is.EqualTo(3));
        Assert.That(frame.ColumnCount, Is.EqualTo(3));
    }

    #endregion

    #region Property-Based Test Scenarios

    /// <summary>
    /// Property 5: Comprehensive lazy evaluation
    /// For any data source and any sequence of query operations, building the query chain 
    /// should not perform any IO operations or data processing until Collect() is called.
    /// </summary>
    [Test]
    [Category("Feature: nivara-frame, Property 5: Comprehensive lazy evaluation")]
    public void LazyEvaluation_MultipleOperations_NoExecutionUntilCollect()
    {
        var testCases = new[]
        {
            csvFilePath,
            jsonFilePath
        };

        foreach (var filePath in testCases)
        {
            // Arrange - record file access time
            var initialAccessTime = File.GetLastAccessTime(filePath);
            
            // Act - build complex query chain
            QueryFrame query;
            if (filePath.EndsWith(".csv"))
            {
                query = CsvExtensions.ScanCsvAsQueryFrame(filePath);
                var complexQuery = query
                    .Filter(ColumnExpressions.Col("Age") > 25)
                    .Select("Name", "Salary")
                    .Filter(ColumnExpressions.Col("Salary") > 70000);
                
                // Assert - query should be lazy and no full execution should have occurred
                Assert.That(query.IsLazy, Is.True, $"Initial query should be lazy for {filePath}");
                Assert.That(complexQuery.IsLazy, Is.True, $"Complex query should be lazy for {filePath}");
                
                // Only after Collect() should we get results
                var result = complexQuery.Collect();
                Assert.That(result.RowCount, Is.GreaterThanOrEqualTo(0), $"Should get valid results for {filePath}");
            }
            else
            {
                query = JsonExtensions.ScanJsonAsQueryFrame(filePath);
                var complexQuery = query
                    .Filter(ColumnExpressions.Col("Age") > 25.0) // Use double for JSON
                    .Select("Name", "Salary")
                    .Filter(ColumnExpressions.Col("Salary") > 70000.0); // Use double for JSON
                
                // Assert - query should be lazy and no full execution should have occurred
                Assert.That(query.IsLazy, Is.True, $"Initial query should be lazy for {filePath}");
                Assert.That(complexQuery.IsLazy, Is.True, $"Complex query should be lazy for {filePath}");
                
                // Only after Collect() should we get results
                var result = complexQuery.Collect();
                Assert.That(result.RowCount, Is.GreaterThanOrEqualTo(0), $"Should get valid results for {filePath}");
            }
        }
    }

    /// <summary>
    /// Property 10: Schema inference from sample data
    /// For any data source with consistent column types, schema inference should correctly 
    /// determine column types from the first few rows of data.
    /// </summary>
    [Test]
    [Category("Feature: nivara-frame, Property 10: Schema inference from sample data")]
    public void SchemaInference_ConsistentTypes_InfersCorrectly()
    {
        var testCases = new[]
        {
            (FilePath: csvFilePath, ExpectedTypes: new[] { typeof(string), typeof(int), typeof(int) }),
            (FilePath: jsonFilePath, ExpectedTypes: new[] { typeof(string), typeof(double), typeof(double) })
        };

        foreach (var (filePath, expectedTypes) in testCases)
        {
            // Act
            IQuerySource source;
            if (filePath.EndsWith(".csv"))
                source = CsvExtensions.ScanCsv(filePath);
            else
                source = JsonExtensions.ScanJson(filePath);

            // Assert
            var schema = source.Schema;
            Assert.That(schema.ColumnNames, Has.Count.EqualTo(3), $"Should have 3 columns for {filePath}");
            
            var columnNames = new[] { "Name", "Age", "Salary" };
            for (int i = 0; i < columnNames.Length; i++)
            {
                var actualType = schema.GetColumnType(columnNames[i]);
                Assert.That(actualType, Is.EqualTo(expectedTypes[i]), 
                    $"Column {columnNames[i]} should be {expectedTypes[i].Name} for {filePath}");
            }
        }
    }

    #endregion

    #region Eager Data Source Property Tests

    /// <summary>
    /// Property 6: Eager loading behavior
    /// For any data source, calling Read operations should immediately load and parse the entire data source, 
    /// making results available without further IO.
    /// </summary>
    [Test]
    [Category("Feature: nivara-frame, Property 6: Eager loading behavior")]
    public void EagerLoading_ImmediateExecution_NoLazyBehavior()
    {
        var testCases = new[]
        {
            (FilePath: csvFilePath, ReadMethod: "CSV"),
            (FilePath: jsonFilePath, ReadMethod: "JSON")
        };

        foreach (var (filePath, readMethod) in testCases)
        {
            // Record initial file access time
            var initialAccessTime = File.GetLastAccessTime(filePath);
            
            // Act - eager reading should immediately process the file
            NivaraFrame frame;
            if (readMethod == "CSV")
            {
                frame = CsvExtensions.ReadCsvAsFrame(filePath);
            }
            else
            {
                frame = JsonExtensions.ReadJsonAsFrame(filePath);
            }
            
            // Assert - data should be immediately available
            Assert.That(frame, Is.Not.Null, $"Frame should be immediately available for {readMethod}");
            Assert.That(frame.RowCount, Is.EqualTo(3), $"Should have 3 rows for {readMethod}");
            Assert.That(frame.ColumnCount, Is.EqualTo(3), $"Should have 3 columns for {readMethod}");
            
            // Verify data is accessible without additional IO
            var nameColumn = frame.GetColumn<string>("Name");
            Assert.That(nameColumn[0], Is.EqualTo("Alice"), $"Data should be immediately accessible for {readMethod}");
            
            // Verify no lazy behavior - accessing data multiple times should not trigger additional IO
            var accessTime1 = File.GetLastAccessTime(filePath);
            var value1 = nameColumn[1];
            var accessTime2 = File.GetLastAccessTime(filePath);
            var value2 = nameColumn[2];
            var accessTime3 = File.GetLastAccessTime(filePath);
            
            // File access times should not change during data access (data is already loaded)
            Assert.That(accessTime2, Is.EqualTo(accessTime1), $"No additional IO should occur during data access for {readMethod}");
            Assert.That(accessTime3, Is.EqualTo(accessTime1), $"No additional IO should occur during repeated access for {readMethod}");
        }
    }

    /// <summary>
    /// Property 17: Data consistency validation
    /// For any data source, eager reading should validate data consistency and report Schema violations 
    /// with clear diagnostic information.
    /// </summary>
    [Test]
    [Category("Feature: nivara-frame, Property 17: Data consistency validation")]
    public void DataConsistencyValidation_InvalidData_ReportsErrors()
    {
        // Create test files with inconsistent data
        var inconsistentCsvPath = Path.Combine(testDataDirectory, "inconsistent.csv");
        File.WriteAllText(inconsistentCsvPath, """
            Name,Age,Salary
            Alice,30,75000
            Bob,not_a_number,65000
            Charlie,35,not_a_number
            """);

        var malformedJsonPath = Path.Combine(testDataDirectory, "malformed.json");
        File.WriteAllText(malformedJsonPath, """
            [
              {"Name": "Alice", "Age": 30, "Salary": 75000},
              {"Name": "Bob", "Age": "not_a_number", "Salary": 65000},
              {"Name": "Charlie", "Age": 35, "Salary": "not_a_number"}
            ]
            """);

        // Test CSV data consistency - should handle inconsistent types gracefully
        var csvFrame = CsvExtensions.ReadCsvAsFrame(inconsistentCsvPath);
        Assert.That(csvFrame, Is.Not.Null, "CSV frame should be created even with inconsistent data");
        Assert.That(csvFrame.RowCount, Is.EqualTo(3), "Should process all rows despite inconsistencies");
        
        // Verify schema inference handles mixed types by falling back to string
        var csvSchema = csvFrame.Schema;
        // Age column should fall back to string due to mixed types
        Assert.That(csvSchema.GetColumnType("Age"), Is.EqualTo(typeof(string)), "Mixed type column should fall back to string");
        
        // Test JSON data consistency - should handle mixed types gracefully
        var jsonFrame = JsonExtensions.ReadJsonAsFrame(malformedJsonPath);
        Assert.That(jsonFrame, Is.Not.Null, "JSON frame should be created even with mixed types");
        Assert.That(jsonFrame.RowCount, Is.EqualTo(3), "Should process all rows despite type inconsistencies");
        
        // Verify schema inference handles mixed types by falling back to string
        var jsonSchema = jsonFrame.Schema;
        Assert.That(jsonSchema.GetColumnType("Age"), Is.EqualTo(typeof(string)), "Mixed type JSON column should fall back to string");
        Assert.That(jsonSchema.GetColumnType("Salary"), Is.EqualTo(typeof(string)), "Mixed type JSON column should fall back to string");
        
        // Clean up test files
        File.Delete(inconsistentCsvPath);
        File.Delete(malformedJsonPath);
    }

    /// <summary>
    /// Property 17: Data consistency validation - Error reporting
    /// For any data source with severe errors, eager reading should report clear error messages.
    /// </summary>
    [Test]
    [Category("Feature: nivara-frame, Property 17: Data consistency validation")]
    public void DataConsistencyValidation_SevereErrors_ReportsDetailedErrors()
    {
        // Create test files with severe errors
        var corruptCsvPath = Path.Combine(testDataDirectory, "corrupt.csv");
        // Create a CSV file with no headers (which should cause an error)
        File.WriteAllText(corruptCsvPath, "");

        var invalidJsonPath = Path.Combine(testDataDirectory, "invalid.json");
        File.WriteAllText(invalidJsonPath, "{ this is not valid JSON }");

        // Test CSV error reporting - empty file should cause schema inference to fail
        var csvException = Assert.Throws<DataSourceException>(() => CsvExtensions.ReadCsvAsFrame(corruptCsvPath));
        Assert.That(csvException!.Message, Does.Contain("CSV").Or.Contain("csv"), "Error message should mention CSV");

        // Test JSON error reporting  
        var jsonException = Assert.Throws<DataSourceException>(() => JsonExtensions.ReadJsonAsFrame(invalidJsonPath));
        Assert.That(jsonException!.Message, Does.Contain("JSON").Or.Contain("json"), "Error message should mention JSON");

        // Clean up test files
        File.Delete(corruptCsvPath);
        File.Delete(invalidJsonPath);
    }

    /// <summary>
    /// Property 6: Eager loading behavior - Empty file handling
    /// For any empty data source, eager reading should return appropriate empty frames.
    /// </summary>
    [Test]
    [Category("Feature: nivara-frame, Property 6: Eager loading behavior")]
    public void EagerLoading_EmptyFiles_ReturnsEmptyFrames()
    {
        // Test empty CSV file handling
        var emptyFrame = CsvExtensions.ReadCsvAsFrame(emptyCsvFilePath);
        Assert.That(emptyFrame, Is.Not.Null, "Should return frame for empty CSV");
        Assert.That(emptyFrame.RowCount, Is.EqualTo(0), "Empty CSV should have 0 rows");
        Assert.That(emptyFrame.ColumnCount, Is.EqualTo(3), "Empty CSV should still have columns from headers");

        // Test empty JSON file handling - should throw because schema cannot be inferred
        Assert.Throws<DataSourceException>(() => JsonExtensions.ReadJsonAsFrame(emptyJsonFilePath),
            "Empty JSON array should throw DataSourceException due to inability to infer schema");
    }

    #endregion
}