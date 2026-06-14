using Nivara.Diagnostics;
using System.Numerics;

namespace Nivara.AutoDiff;

static class AutoDiffDiagnostics
{
    public static void Measure<T>(
        string operationType,
        int inputLength,
        bool hadNulls,
        Action operation,
        string? notes = null)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(operation);

        var measurement = DiagnosticsTracker.StartMeasurement();
        operation();
        measurement.Record(
            operationType,
            KernelSelector.DetermineKernelType<T>(inputLength),
            inputLength,
            typeof(T),
            hadNulls,
            notes);
    }

    public static TResult Measure<T, TResult>(
        string operationType,
        int inputLength,
        bool hadNulls,
        Func<TResult> operation,
        string? notes = null)
        where T : struct, INumber<T>

        => DiagnosticsTracker.MeasureOperation(
            operationType,
            KernelSelector.DetermineKernelType<T>(inputLength),
            inputLength,
            typeof(T),
            hadNulls,
            operation,
            notes);

    public static string ShapeNote(string operation, ReadOnlySpan<int> shape)
        => $"AutoDiff={operation};Shape=[{string.Join(", ", shape.ToArray())}]";

    public static string MatrixNote(string operation, int rows, int inner, int cols)
        => $"AutoDiff={operation};Shape={rows}x{inner}->{rows}x{cols}";
}
