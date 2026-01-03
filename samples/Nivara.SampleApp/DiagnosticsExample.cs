using Nivara.Diagnostics;

namespace Nivara.SampleApp;

/// <summary>
/// Demonstrates the diagnostic capabilities of Nivara columns for performance analysis
/// </summary>
public static class DiagnosticsExample
{
    static void ShowColumnDiagnostics(string title, ColumnDiagnostics diagnostics)
    {
        Console.WriteLine($"{title}:");
        Console.WriteLine($"  Storage Type: {diagnostics.StorageType}");
        Console.WriteLine($"  Is Vectorizable: {diagnostics.IsVectorizable}");
        Console.WriteLine($"  Element Type: {diagnostics.ElementType.Name}");
        Console.WriteLine($"  Length: {diagnostics.Length}");
        Console.WriteLine($"  Has Nulls: {diagnostics.HasNulls}");
        Console.WriteLine($"  Hardware Accelerated: {diagnostics.IsHardwareAccelerated}");
        Console.WriteLine($"  Vector Size: {diagnostics.VectorSize} bytes");
        Console.WriteLine($"  Recommended Kernel: {diagnostics.RecommendedKernel}");
        Console.WriteLine($"  Estimated Memory: {diagnostics.EstimatedMemoryUsage:N0} bytes");

        var performance = diagnostics.Performance;
        Console.WriteLine($"  Performance:");
        Console.WriteLine($"    Throughput Multiplier: {performance.ThroughputMultiplier:F1}x");
        Console.WriteLine($"    Memory Efficiency: {performance.MemoryEfficiency:P1}");
        Console.WriteLine($"    Supports Vectorization: {performance.SupportsVectorization}");
        Console.WriteLine();
    }

    public static void RunExample()
    {
        Console.WriteLine("=== Nivara Diagnostics Example ===\n");

        // Create different types of columns
        var intColumn = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        var stringColumn = NivaraColumn<string>.CreateForReferenceType(new[] { "apple", "banana", "cherry" });
        var doubleColumn = NivaraColumn<double>.Create(new[] { 1.1, 2.2, 3.3, 4.4, 5.5 });

        // Demonstrate column diagnostics
        Console.WriteLine("Column Diagnostics:");
        Console.WriteLine("==================");

        ShowColumnDiagnostics("Integer Column", intColumn.Diagnostics);
        ShowColumnDiagnostics("String Column", stringColumn.Diagnostics);
        ShowColumnDiagnostics("Double Column", doubleColumn.Diagnostics);

        Console.WriteLine();

        // Enable operation tracking
        DiagnosticsTracker.IsEnabled = true;
        DiagnosticsTracker.ClearRecordedOperations();

        Console.WriteLine("Performing Operations with Tracking:");
        Console.WriteLine("===================================");

        // Perform various operations
        var result1 = intColumn.Add(intColumn);
        Console.WriteLine("✓ Element-wise addition performed");

        var result2 = intColumn.Multiply(2);
        Console.WriteLine("✓ Scalar multiplication performed");

        var result3 = doubleColumn.GreaterThan(3.0);
        Console.WriteLine("✓ Scalar comparison performed");

        var result4 = stringColumn.Equals("banana");
        Console.WriteLine("✓ String comparison performed");

        Console.WriteLine();

        // Show operation diagnostics
        var operations = DiagnosticsTracker.GetRecordedOperations();
        Console.WriteLine($"Recorded {operations.Length} operations:");
        Console.WriteLine("=====================================");

        foreach (var op in operations)
        {
            Console.WriteLine($"Operation: {op.OperationType}");
            Console.WriteLine($"  Kernel Used: {op.KernelUsed}");
            Console.WriteLine($"  Element Type: {op.ElementType.Name}");
            Console.WriteLine($"  Input Length: {op.InputLength}");
            Console.WriteLine($"  Had Nulls: {op.HadNulls}");
            Console.WriteLine($"  Selection Reason: {op.KernelSelectionReason}");
            Console.WriteLine($"  Timestamp: {op.Timestamp:HH:mm:ss.fff}");
            Console.WriteLine();
        }

        // Show summary statistics
        var summary = DiagnosticsTracker.GetSummary();
        Console.WriteLine("Operation Summary:");
        Console.WriteLine("=================");
        Console.WriteLine($"Total Operations: {summary.TotalOperations}");
        Console.WriteLine($"Vectorized Operations: {summary.VectorizedOperations}");
        Console.WriteLine($"Scalar Operations: {summary.ScalarOperations}");
        Console.WriteLine($"Operations with Nulls: {summary.OperationsWithNulls}");
        Console.WriteLine($"Vectorization Rate: {summary.VectorizationRate:F1}%");

        Console.WriteLine("\nOperation Types:");
        foreach (var kvp in summary.OperationTypes)
        {
            Console.WriteLine($"  {kvp.Key}: {kvp.Value}");
        }

        Console.WriteLine("\nKernel Types:");
        foreach (var kvp in summary.KernelTypes)
        {
            Console.WriteLine($"  {kvp.Key}: {kvp.Value}");
        }

        // Cleanup
        DiagnosticsTracker.IsEnabled = false;
        DiagnosticsTracker.ClearRecordedOperations();

        Console.WriteLine("\n=== Example Complete ===");
    }
}
