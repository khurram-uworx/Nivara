using Nivara.Expressions;
using Nivara.Helpers;
using Nivara.IO;
using Nivara.Query;
using NUnit.Framework;

namespace Nivara.Tests;

/// <summary>
/// Property tests for resource management functionality.
/// Tests universal properties related to disposal patterns and resource cleanup.
/// </summary>
[TestFixture]
public class ResourceManagementPropertyTests
{
    /// <summary>
    /// Property 18: Resource disposal
    /// For any NivaraFrame or QueryFrame, proper disposal should clean up all underlying column resources
    /// and handle abandoned lazy queries appropriately.
    /// **Validates: Requirements 10.1, 10.2**
    /// </summary>
    [TestCase(1, 10)]
    [TestCase(5, 100)]
    [TestCase(3, 1000)]
    [TestCase(10, 50)]
    public void ResourceDisposal_ShouldCleanupAllUnderlyingResources(int columnCount, int rowCount)
    {
        // Arrange - Create frame with multiple columns
        var columns = new List<(string Name, IColumn Column)>();
        for (int i = 0; i < columnCount; i++)
        {
            var data = Enumerable.Range(0, rowCount).Select(x => x * (i + 1)).ToArray();
            var column = NivaraColumn<int>.Create(data);
            columns.Add(($"Column{i}", (IColumn)column));
        }

        NivaraFrame? frame = null;
        WeakReference? frameRef = null;

        // Act - Create and dispose frame
        frame = new NivaraFrame(columns);
        frameRef = new WeakReference(frame);

        // (Tracking removed in #1168; disposal is the only release path.)
        // Dispose the frame
        frame.Dispose();

        // Assert - Verify disposal behavior
        Assert.Throws<ObjectDisposedException>(() => _ = frame.RowCount,
            "Disposed frame should throw ObjectDisposedException when accessed");

        Assert.Throws<ObjectDisposedException>(() => _ = frame.ColumnCount,
            "Disposed frame should throw ObjectDisposedException when accessed");

        Assert.Throws<ObjectDisposedException>(() => frame.GetColumn<int>("Column0"),
            "Disposed frame should throw ObjectDisposedException when accessing columns");

        // Verify columns are also disposed
        foreach (var (_, column) in columns)
        {
            Assert.Throws<ObjectDisposedException>(() => _ = column.Length,
                "Disposed columns should throw ObjectDisposedException when accessed");
        }

        // Clear reference and force garbage collection
        frame = null;
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        // (Abandoned-resource cleanup removed in #1168; explicit disposal is the release path.)
    }

    /// <summary>
    /// Property 18: Resource disposal for QueryFrames
    /// For any QueryFrame, disposal should properly clean up lazy query resources.
    /// **Validates: Requirements 10.1, 10.2**
    /// </summary>
    [TestCase("test1.json")]
    [TestCase("test2.json")]
    [TestCase("test3.json")]
    public void QueryFrameDisposal_ShouldCleanupLazyQueryResources(string fileName)
    {
        // Arrange - Create a temporary JSON file for testing
        var tempFile = Path.Combine(Path.GetTempPath(), fileName);
        var testData = """
        [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
            {"id": 3, "name": "Charlie", "age": 35}
        ]
        """;

        try
        {
            File.WriteAllText(tempFile, testData);

            QueryFrame? queryFrame = null;
            WeakReference? queryRef = null;

            // Act - Create lazy query frame
            queryFrame = Json.ScanFrame(tempFile);
            queryRef = new WeakReference(queryFrame);

            // (Tracking removed in #1168; explicit disposal releases the handle.)
            // Dispose the query frame
            queryFrame.Dispose();

            // Assert - Verify disposal behavior
            Assert.Throws<ObjectDisposedException>(() => _ = queryFrame.Schema,
                "Disposed query frame should throw ObjectDisposedException when accessed");

            Assert.Throws<ObjectDisposedException>(() => queryFrame.Where(ColumnExpressions.Col("age") > 25),
                "Disposed query frame should throw ObjectDisposedException for operations");

            // Clear reference and force garbage collection
            queryFrame = null;
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            // Handle must be released so the file can be deleted (Windows).
            Assert.DoesNotThrow(() => File.Delete(tempFile),
                "Disposing the query frame must release the file handle");
        }
        finally
        {
            // Cleanup temp file
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
            }
        }
    }

    /// <summary>
    /// Property 19: IDisposable pattern implementation
    /// For any resource-holding type in the system, the IDisposable pattern should be properly implemented
    /// with appropriate cleanup behavior.
    /// **Validates: Requirements 10.4**
    /// </summary>
    [TestCase(typeof(NivaraFrame))]
    [TestCase(typeof(QueryFrame))]
    public void IDisposablePattern_ShouldBeProperlyImplemented(Type resourceType)
    {
        // Assert - Verify type implements IDisposable
        Assert.That(typeof(IDisposable).IsAssignableFrom(resourceType), Is.True,
            $"{resourceType.Name} should implement IDisposable");

        // Verify Dispose method exists and is public
        var disposeMethod = resourceType.GetMethod("Dispose", Type.EmptyTypes);
        Assert.That(disposeMethod, Is.Not.Null,
            $"{resourceType.Name} should have a public Dispose() method");

        Assert.That(disposeMethod!.IsPublic, Is.True,
            $"{resourceType.Name}.Dispose() should be public");

        // Verify multiple calls to Dispose are safe (idempotent)
        if (resourceType == typeof(NivaraFrame))
        {
            var column = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
            var frame = new NivaraFrame(new[] { ("test", (IColumn)column) });

            // Multiple dispose calls should not throw
            Assert.DoesNotThrow(() => frame.Dispose(),
                "First Dispose() call should not throw");
            Assert.DoesNotThrow(() => frame.Dispose(),
                "Second Dispose() call should not throw (idempotent)");
            Assert.DoesNotThrow(() => frame.Dispose(),
                "Third Dispose() call should not throw (idempotent)");
        }
    }

    /// <summary>
    /// Property 19: Column disposal pattern
    /// For any NivaraColumn, the IDisposable pattern should be properly implemented.
    /// **Validates: Requirements 10.4**
    /// </summary>
    [TestCase(new[] { 1, 2, 3, 4, 5 })]
    [TestCase(new[] { 10, 20, 30 })]
    [TestCase(new[] { 100 })]
    [TestCase(new int[0])]
    public void ColumnDisposablePattern_ShouldBeProperlyImplemented(int[] data)
    {
        // Arrange
        var column = NivaraColumn<int>.Create(data);

        // Act & Assert - Verify multiple dispose calls are safe
        Assert.DoesNotThrow(() => column.Dispose(),
            "First Dispose() call should not throw");
        Assert.DoesNotThrow(() => column.Dispose(),
            "Second Dispose() call should not throw (idempotent)");

        // Verify disposed column throws ObjectDisposedException
        Assert.Throws<ObjectDisposedException>(() => _ = column.Length,
            "Disposed column should throw ObjectDisposedException when accessed");

        if (data.Length > 0)
        {
            Assert.Throws<ObjectDisposedException>(() => _ = column[0],
                "Disposed column should throw ObjectDisposedException when indexed");
        }
    }

    /// <summary>
    /// Property 18: Memory management for large datasets
    /// For any large dataset operation, the system should provide appropriate memory management guidance.
    /// **Validates: Requirements 10.1**
    /// </summary>
    [TestCase(1000, "filter")]
    [TestCase(5000, "groupby")]
    [TestCase(10000, "join")]
    [TestCase(2000, "sort")]
    public void MemoryManagement_ShouldProvideGuidanceForLargeDatasets(int rowCount, string operationType)
    {
        // Arrange - Create a large frame
        var data = Enumerable.Range(0, rowCount).ToArray();
        var column = NivaraColumn<int>.Create(data);
        var frame = new NivaraFrame(new[] { ("data", (IColumn)column) });

        try
        {
            // Act - Get memory recommendations
            var recommendations = frame.GetMemoryRecommendations(operationType);

            // Assert - Verify recommendations are provided
            Assert.That(recommendations, Is.Not.Null,
                "Memory recommendations should be provided");

            Assert.That(recommendations.EstimatedMemoryUsage, Is.GreaterThan(0),
                "Estimated memory usage should be calculated");

            Assert.That(recommendations.AvailableMemory, Is.GreaterThan(0),
                "Available memory should be reported");

            // At megabyte-scale thresholds these row counts are not "large", so no size warning
            // is expected; operation-specific advice is still emitted where applicable.
            if (rowCount >= 5000)
            {
                Assert.That(recommendations.WarningLevel, Is.EqualTo(MemoryWarningLevel.None),
                    "Warning level should be None for sub-megabyte frames");
            }

            // Operation-specific recommendations should be relevant
            var recommendationText = string.Join(" ", recommendations.Recommendations);
            if (operationType == "groupby")
            {
                Assert.That(recommendationText.ToLowerInvariant().Contains("groupby") ||
                           recommendationText.ToLowerInvariant().Contains("memory"),
                    Is.True, "GroupBy operations should have relevant recommendations");
            }
        }
        finally
        {
            frame.Dispose();
        }
    }

}
