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
    [SetUp]
    public void Setup()
    {
        // Force cleanup before each test to ensure clean state
        NivaraResourceManager.ForceCleanup();
    }

    [TearDown]
    public void TearDown()
    {
        // Force cleanup after each test to prevent resource leaks
        NivaraResourceManager.ForceCleanup();
    }

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

        // Verify frame is tracked
        var statsBefore = NivaraResourceManager.GetResourceStatistics();
        Assert.That(statsBefore.TotalTrackedResources, Is.GreaterThan(0),
            "Frame and columns should be tracked before disposal");

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

        // Force cleanup of abandoned resources
        NivaraResourceManager.ForceCleanup();

        // Verify resources are cleaned up
        var statsAfter = NivaraResourceManager.GetResourceStatistics();
        Assert.That(statsAfter.AbandonedResourceCount, Is.EqualTo(0),
            "No abandoned resources should remain after disposal and cleanup");
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
            queryFrame = Json.ScanJsonAsQueryFrame(tempFile);
            queryRef = new WeakReference(queryFrame);

            // Verify query frame is tracked
            var statsBefore = NivaraResourceManager.GetResourceStatistics();
            Assert.That(statsBefore.TrackedResourcesByType.ContainsKey("LazyQueryFrame"), Is.True,
                "Lazy query frame should be tracked");

            // Dispose the query frame
            queryFrame.Dispose();

            // Assert - Verify disposal behavior
            Assert.Throws<ObjectDisposedException>(() => _ = queryFrame.Schema,
                "Disposed query frame should throw ObjectDisposedException when accessed");

            Assert.Throws<ObjectDisposedException>(() => queryFrame.Filter(ColumnExpressions.Col("age") > 25),
                "Disposed query frame should throw ObjectDisposedException for operations");

            // Clear reference and force garbage collection
            queryFrame = null;
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            // Force cleanup of abandoned resources
            NivaraResourceManager.ForceCleanup();

            // Verify resources are cleaned up
            var statsAfter = NivaraResourceManager.GetResourceStatistics();
            Assert.That(statsAfter.AbandonedResourceCount, Is.EqualTo(0),
                "No abandoned resources should remain after disposal and cleanup");
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
    /// Property 18: Abandoned query cleanup
    /// For any abandoned lazy query, the resource manager should automatically clean up resources.
    /// **Validates: Requirements 10.2**
    /// </summary>
    [TestCase(3)]
    [TestCase(5)]
    public void AbandonedQueryCleanup_ShouldAutomaticallyCleanupResources(int queryCount)
    {
        // Arrange - Create temporary JSON files
        var tempFiles = new List<string>();
        var testData = """
        [
            {"id": 1, "value": 100},
            {"id": 2, "value": 200}
        ]
        """;

        try
        {
            for (int i = 0; i < queryCount; i++)
            {
                var tempFile = Path.Combine(Path.GetTempPath(), $"test_abandoned_{i}_{Guid.NewGuid():N}.json");
                File.WriteAllText(tempFile, testData);
                tempFiles.Add(tempFile);
            }

            var weakRefs = new List<WeakReference>();

            // Act - Create lazy queries and abandon them in a separate method to ensure no strong references
            CreateAndAbandonQueries(tempFiles, weakRefs);

            // Verify queries are tracked
            var statsBefore = NivaraResourceManager.GetResourceStatistics();
            Assert.That(statsBefore.TrackedResourcesByType.ContainsKey("LazyQueryFrame"), Is.True,
                "Lazy query frames should be tracked");

            // Force multiple garbage collection cycles to ensure cleanup
            for (int i = 0; i < 5; i++)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                Thread.Sleep(10); // Give some time for cleanup
            }

            // Force cleanup of abandoned resources multiple times
            NivaraResourceManager.ForceCleanup();
            Thread.Sleep(50);
            NivaraResourceManager.ForceCleanup();

            // Assert - Check if most queries are garbage collected (allow for some variance in GC behavior)
            int collectedCount = 0;
            foreach (var weakRef in weakRefs)
            {
                if (weakRef.Target == null)
                    collectedCount++;
            }

            // At least half should be collected (GC is not deterministic in tests)
            Assert.That(collectedCount, Is.GreaterThanOrEqualTo(queryCount / 2),
                $"At least {queryCount / 2} out of {queryCount} abandoned query frames should be garbage collected");

            // Verify abandoned resources are cleaned up
            var statsAfter = NivaraResourceManager.GetResourceStatistics();
            Assert.That(statsAfter.AbandonedResourceCount, Is.LessThanOrEqualTo(queryCount / 2),
                "Most abandoned resources should be automatically cleaned up");
        }
        finally
        {
            // Cleanup temp files
            foreach (var tempFile in tempFiles)
            {
                if (File.Exists(tempFile))
                {
                    File.Delete(tempFile);
                }
            }
        }
    }

    /// <summary>
    /// Helper method to create and abandon queries without holding strong references
    /// </summary>
    private static void CreateAndAbandonQueries(List<string> tempFiles, List<WeakReference> weakRefs)
    {
        for (int i = 0; i < tempFiles.Count; i++)
        {
            var queryFrame = Json.ScanJsonAsQueryFrame(tempFiles[i]);
            weakRefs.Add(new WeakReference(queryFrame));
            // queryFrame goes out of scope here, making it eligible for GC
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

            // For large datasets, recommendations should be provided
            if (rowCount >= 5000)
            {
                Assert.That(recommendations.Recommendations.Count, Is.GreaterThan(0),
                    "Recommendations should be provided for large datasets");

                Assert.That(recommendations.WarningLevel, Is.Not.EqualTo(MemoryWarningLevel.None),
                    "Warning level should be set for large datasets");
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

    /// <summary>
    /// Property 19: Resource tracking statistics
    /// For any set of resource operations, the resource manager should provide accurate statistics.
    /// **Validates: Requirements 10.4**
    /// </summary>
    [TestCase(2, 3)]
    [TestCase(5, 10)]
    [TestCase(1, 100)]
    public void ResourceTracking_ShouldProvideAccurateStatistics(int frameCount, int rowCount)
    {
        // Arrange
        var frames = new List<NivaraFrame>();
        var statsBefore = NivaraResourceManager.GetResourceStatistics();

        try
        {
            // Act - Create multiple frames
            for (int i = 0; i < frameCount; i++)
            {
                var data = Enumerable.Range(0, rowCount).Select(x => x + i * 1000).ToArray();
                var column = NivaraColumn<int>.Create(data);
                var frame = new NivaraFrame(new[] { ($"data{i}", (IColumn)column) });
                frames.Add(frame);
            }

            // Get statistics after creation
            var statsAfter = NivaraResourceManager.GetResourceStatistics();

            // Assert - Verify statistics are accurate
            Assert.That(statsAfter.TotalTrackedResources,
                Is.GreaterThan(statsBefore.TotalTrackedResources),
                "Total tracked resources should increase after creating frames");

            Assert.That(statsAfter.TrackedResourcesByType.ContainsKey("NivaraFrame"), Is.True,
                "NivaraFrame resources should be tracked by type");

            Assert.That(statsAfter.TotalEstimatedMemoryUsage,
                Is.GreaterThan(statsBefore.TotalEstimatedMemoryUsage),
                "Total estimated memory usage should increase");

            // Dispose half the frames
            var framesToDispose = frameCount / 2;
            if (framesToDispose > 0)
            {
                for (int i = 0; i < framesToDispose; i++)
                {
                    frames[i].Dispose();
                }

                // Force cleanup and give time for tracking updates
                NivaraResourceManager.ForceCleanup();
                Thread.Sleep(10);

                // Verify statistics reflect disposal
                var statsAfterDisposal = NivaraResourceManager.GetResourceStatistics();

                // The total should be less than after creation, but may not be exactly equal to before
                // due to remaining frames and columns still being tracked
                Assert.That(statsAfterDisposal.TotalTrackedResources,
                    Is.LessThan(statsAfter.TotalTrackedResources),
                    "Total tracked resources should decrease after disposal");
            }
            else
            {
                // If no frames to dispose, just verify tracking is working
                Assert.That(statsAfter.TotalTrackedResources,
                    Is.GreaterThan(statsBefore.TotalTrackedResources),
                    "Resource tracking should be working");
            }
        }
        finally
        {
            // Cleanup remaining frames
            foreach (var frame in frames)
            {
                frame?.Dispose();
            }
        }
    }
}
