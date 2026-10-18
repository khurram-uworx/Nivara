using Nivara;
using Nivara.Diagnostics;
using System.Numerics;

namespace Nivara.AI;

/// <summary>
/// Inference-side analogue of <c>AutoDiffDiagnostics</c>: records per-block timing
/// and allocation data for the fused (non-AutoDiff) CPU inference path, which
/// runs through <c>GradKernels</c>, <c>LayerNormKernel</c>, <c>AttentionKernels</c>,
/// and <c>TensorPrimitives</c> rather than <c>ReverseGradOperations</c>.
/// </summary>
internal static class InferenceDiagnostics
{
    static bool enabled;

    public static bool Enabled
    {
        get => enabled;
        set
        {
            enabled = value;
            if (value)
                DiagnosticsTracker.IsEnabled = true;
        }
    }

    public static bool IsEnabled => enabled;

    public static void Measure<T>(
        string operationType,
        int inputLength,
        Action operation,
        string? notes = null)
        where T : struct, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(operation);

        if (!enabled)
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

        if (!enabled)
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
}
