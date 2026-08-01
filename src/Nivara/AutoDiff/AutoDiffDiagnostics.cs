using Nivara.Diagnostics;
using System.Numerics;

namespace Nivara.AutoDiff;

static class AutoDiffDiagnostics
{
    /// <summary>
    /// Enables AutoDiff operation diagnostics. Disabled by default so the inference
    /// hot path pays zero per-op lock acquisition; enable only when a diagnostics
    /// session is explicitly requested.
    /// </summary>
    public static bool Enabled { get; set; }

    public static void Measure<T>(
        string operationType,
        int inputLength,
        Action operation,
        string? notes = null)
        where T : struct, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(operation);

        if (!Enabled)
        {
            operation();
            return;
        }

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
    {
        ArgumentNullException.ThrowIfNull(operation);

        if (!Enabled)
            return operation();

        return DiagnosticsTracker.MeasureOperation(
            operationType,
            KernelSelector.DetermineKernelType<T>(inputLength),
            inputLength,
            typeof(T),
            false,
            operation,
            notes);
    }

    public static string ShapeNote(string operation, ReadOnlySpan<int> shape)
        => $"AutoDiff={operation};Shape=[{string.Join(", ", shape.ToArray())}]";

    public static string MatrixNote(string operation, int rows, int inner, int cols)
        => $"AutoDiff={operation};Shape={rows}x{inner}->{rows}x{cols}";
}
