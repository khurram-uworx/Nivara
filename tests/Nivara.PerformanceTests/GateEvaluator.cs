namespace Nivara.PerformanceTests;

/// <summary>
/// The per-row metrics the <c>--compare</c> gate consumes. The harness measures
/// <c>NsPerOp</c> too, but the gate policy has no use for it — keeping this record
/// minimal lets <see cref="GateEvaluator"/> be compiled into both the harness and
/// the test project as a shared source file (no project reference).
/// </summary>
internal sealed record GateRow(string Name, double OpsPerSec, double BytesPerOp, double Gen0PerOp);

/// <summary>
/// Pure per-row decision logic for the <c>--compare</c> no-regression gate (issue #420).
/// Bandwidth-bound rows (single-row GEMV / memory-streaming kernels — the Qwen decode
/// and prefill scenarios, ~30 GB/s effective DRAM ceiling) see ops/s swing ~2.5-3x with
/// machine state while B/op stays byte-stable, so their ops/s leg is gated at a wide
/// absolute floor; B/op and gen0 remain strict for every row. Shared with the test
/// project via a linked compile (see Nivara.Tests.csproj).
/// </summary>
internal static class GateEvaluator
{
    /// <summary>Default ops/s floor for stable rows, as a fraction of the baseline.</summary>
    public const double DefaultMinOpsFraction = 0.90;

    /// <summary>B/op floor — measured allocation must not exceed the baseline reading by more than 1%.</summary>
    public const double MaxAllocationFraction = 1.01;

    /// <summary>gen0 floor — GC scheduling is not allocation-proportional, so a small plus is tolerated.</summary>
    public const double Gen0Tolerance = 0.05;

    /// <summary>
    /// ops/s floor for bandwidth-bound rows. DRAM-bandwidth-bound kernels read 2.5-3x slower
    /// under machine load, which swamps the ~10% deltas real perf changes produce — so the
    /// floor is wide enough to absorb observed machine-state drift while still catching
    /// catastrophic (4x+) regressions.
    /// </summary>
    public const double BandwidthBoundMinOpsFraction = 0.25;

    /// <summary>Per-row gate verdict. Each leg's pass state is surfaced so the caller can render it.</summary>
    internal sealed record Verdict(bool Pass, bool OpsOk, bool BytesOk, bool Gen0Ok, double OpsFloor);

    /// <summary>Evaluates one measured row against its baseline row under the gate policy.</summary>
    internal static Verdict EvaluateRow(
        GateRow current,
        GateRow baseline,
        double minOpsFraction,
        bool bandwidthBound)
    {
        double opsFloor = bandwidthBound ? BandwidthBoundMinOpsFraction : minOpsFraction;
        bool opsOk = current.OpsPerSec >= baseline.OpsPerSec * opsFloor;
        bool bytesOk = current.BytesPerOp <= baseline.BytesPerOp * MaxAllocationFraction;
        bool gen0Ok = current.Gen0PerOp <= baseline.Gen0PerOp + Gen0Tolerance;
        return new Verdict(opsOk && bytesOk && gen0Ok, opsOk, bytesOk, gen0Ok, opsFloor);
    }
}