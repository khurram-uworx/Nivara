using Nivara;

namespace Nivara.SampleApp;

public static class AggregateExample
{
    public static void Run()
    {
        Console.WriteLine("=== Nivara Aggregate Functions Example ===");
        Console.WriteLine();

        // Test with integer data
        Console.WriteLine("Integer Series:");
        var intData = new[] { 1, 2, 3, 4, 5 };
        var intSeries = NivaraSeries<int>.Create(intData);

        Console.WriteLine($"Data: [{string.Join(", ", intData)}]");
        Console.WriteLine($"Sum: {intSeries.Sum()}");
        Console.WriteLine($"Average: {intSeries.Average()}");
        Console.WriteLine($"Min: {intSeries.Min()}");
        Console.WriteLine($"Max: {intSeries.Max()}");
        Console.WriteLine();

        // Test with floating point data (should use TensorPrimitives)
        Console.WriteLine("Float Series (Vectorized):");
        var floatData = new[] { 1.5f, 2.5f, 3.5f, 4.5f };
        var floatSeries = NivaraSeries<float>.Create(floatData);

        Console.WriteLine($"Data: [{string.Join(", ", floatData)}]");
        Console.WriteLine($"Sum: {floatSeries.Sum():F1}");
        Console.WriteLine($"Average: {floatSeries.Average():F1}");
        Console.WriteLine($"Min: {floatSeries.Min():F1}");
        Console.WriteLine($"Max: {floatSeries.Max():F1}");
        Console.WriteLine();

        // Test with double data (should use TensorPrimitives)
        Console.WriteLine("Double Series (Vectorized):");
        var doubleData = new[] { Math.PI, Math.E, 1.0, 10.0 };
        var doubleSeries = NivaraSeries<double>.Create(doubleData);

        Console.WriteLine($"Data: [π, e, 1.0, 10.0]");
        Console.WriteLine($"Sum: {doubleSeries.Sum():F3}");
        Console.WriteLine($"Average: {doubleSeries.Average():F3}");
        Console.WriteLine($"Min: {doubleSeries.Min():F3}");
        Console.WriteLine($"Max: {doubleSeries.Max():F3}");
        Console.WriteLine();

        // Test with null values
        Console.WriteLine("Series with Null Values:");
        var nullableData = new int?[] { 1, null, 3, null, 5 };
        var column = NivaraColumn<int>.CreateFromNullable(nullableData);
        var nullableSeries = new NivaraSeries<int>(column);

        Console.WriteLine($"Data: [1, null, 3, null, 5]");
        Console.WriteLine($"Sum: {nullableSeries.Sum()} (ignores nulls)");
        Console.WriteLine($"Average: {nullableSeries.Average()} (ignores nulls)");
        Console.WriteLine($"Min: {nullableSeries.Min()} (ignores nulls)");
        Console.WriteLine($"Max: {nullableSeries.Max()} (ignores nulls)");
        Console.WriteLine();

        // Test with string data (non-numeric, should work for Min/Max)
        Console.WriteLine("String Series (Comparison only):");
        var stringData = new[] { "zebra", "apple", "banana", "cherry" };
        var stringSeries = NivaraSeries<string>.Create(stringData);

        Console.WriteLine($"Data: [{string.Join(", ", stringData)}]");
        Console.WriteLine($"Min: {stringSeries.Min()} (lexicographic)");
        Console.WriteLine($"Max: {stringSeries.Max()} (lexicographic)");
        
        try
        {
            stringSeries.Sum();
        }
        catch (InvalidOperationException ex)
        {
            Console.WriteLine($"Sum: Not supported for strings - {ex.Message.Split('.')[0]}");
        }
        Console.WriteLine();

        Console.WriteLine("=== Example Complete ===");
    }
}