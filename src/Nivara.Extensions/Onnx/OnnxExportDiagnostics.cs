using Nivara.Diagnostics;

namespace Nivara.Onnx;

/// <summary>
/// Inference/export-side analogue of <c>AutoDiffDiagnostics</c>: records per-emitter
/// allocation and timing data for the ONNX export path through <see cref="DiagnosticsTracker"/>.
/// Disabled by default so the export hot path pays zero per-emitter overhead unless a
/// diagnostics session is explicitly requested.
/// </summary>
internal static class OnnxExportDiagnostics
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

    public static bool IsEnabled => enabled || DiagnosticsTracker.IsEnabled;
}
