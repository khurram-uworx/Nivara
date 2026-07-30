using Nivara.Diagnostics;
using System.Numerics;

namespace Nivara.AutoDiff;

static class AutoDiffDiagnostics
{
    public static void Measure<T>(
        string operationType,
        int inputLength,
        Action operation,
        string? notes = null)
        where T : struct, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(operation);

        var measurement = DiagnosticsTracker.StartMeasurement();
        operation();
        measurement.Record(
            operationType,
            KernelSelector.DetermineKernelType<T>(inputLength),
            inputLength,
            typeof(T),
            false,
            notes);
    }

    public static TResult Measure<T, TResult>(
        string operationType,
        int inputLength,
        Func<TResult> operation,
        string? notes = null)
        where T : struct, IFloatingPointIeee754<T>

        => DiagnosticsTracker.MeasureOperation(
            operationType,
            KernelSelector.DetermineKernelType<T>(inputLength),
            inputLength,
            typeof(T),
            false,
            operation,
            notes);

    public static string ShapeNote(string operation, ReadOnlySpan<int> shape)
        => $"AutoDiff={operation};Shape=[{string.Join(", ", shape.ToArray())}]";

    public static string MatrixNote(string operation, int rows, int inner, int cols)
        => $"AutoDiff={operation};Shape={rows}x{inner}->{rows}x{cols}";
}
