using Nivara.PerformanceTests;
using NUnit.Framework;

namespace Nivara.Tests.PerfGate;

[TestFixture]
public class GateEvaluatorTests
{
    const string BandwidthBoundRow = "Qwen LM head matmul [1x896 @ 151936x896]";
    const string StableRow = "ColumnAdd 1M x float";

    static GateRow Row(string name, double opsPerSec, double bytesPerOp = 0, double gen0PerOp = 0)
        => new(name, opsPerSec, bytesPerOp, gen0PerOp);

    [Test]
    public void EvaluateRow_BandwidthBoundThreeTimesSlowerUnderLoad_Passes()
    {
        var measured = Row(BandwidthBoundRow, opsPerSec: 100);
        var baseline = Row(BandwidthBoundRow, opsPerSec: 300);

        var verdict = GateEvaluator.EvaluateRow(measured, baseline, 0.90, bandwidthBound: true);

        Assert.That(verdict.Pass, Is.True);
        Assert.That(verdict.OpsFloor, Is.EqualTo(GateEvaluator.BandwidthBoundMinOpsFraction));
    }

    [Test]
    public void EvaluateRow_BandwidthBoundJustAboveFloor_Passes()
    {
        var measured = Row(BandwidthBoundRow, opsPerSec: 260);
        var baseline = Row(BandwidthBoundRow, opsPerSec: 1000);

        var verdict = GateEvaluator.EvaluateRow(measured, baseline, 0.90, bandwidthBound: true);

        Assert.That(verdict.OpsOk, Is.True);
        Assert.That(verdict.Pass, Is.True);
    }

    [Test]
    public void EvaluateRow_BandwidthBoundAtExactBandwidthFloor_Passes()
    {
        var measured = Row(BandwidthBoundRow, opsPerSec: 100);
        var baseline = Row(BandwidthBoundRow, opsPerSec: 400);

        var verdict = GateEvaluator.EvaluateRow(measured, baseline, 0.90, bandwidthBound: true);

        Assert.That(verdict.OpsOk, Is.True);
        Assert.That(verdict.Pass, Is.True);
    }

    [Test]
    public void EvaluateRow_BandwidthBoundBelowBandwidthFloor_Fails()
    {
        var measured = Row(BandwidthBoundRow, opsPerSec: 80);
        var baseline = Row(BandwidthBoundRow, opsPerSec: 400);

        var verdict = GateEvaluator.EvaluateRow(measured, baseline, 0.90, bandwidthBound: true);

        Assert.That(verdict.OpsOk, Is.False);
        Assert.That(verdict.Pass, Is.False);
    }

    [Test]
    public void EvaluateRow_BandwidthBoundOpsFineButAllocationOverFloor_Fails()
    {
        var measured = Row(BandwidthBoundRow, opsPerSec: 400, bytesPerOp: 1011, gen0PerOp: 0);
        var baseline = Row(BandwidthBoundRow, opsPerSec: 400, bytesPerOp: 1000, gen0PerOp: 0);

        var verdict = GateEvaluator.EvaluateRow(measured, baseline, 0.90, bandwidthBound: true);

        Assert.That(verdict.OpsOk, Is.True);
        Assert.That(verdict.BytesOk, Is.False);
        Assert.That(verdict.Pass, Is.False);
    }

    [Test]
    public void EvaluateRow_BandwidthBoundOpsFineButGen0OverTolerance_Fails()
    {
        var measured = Row(BandwidthBoundRow, opsPerSec: 400, bytesPerOp: 1000, gen0PerOp: 0.06);
        var baseline = Row(BandwidthBoundRow, opsPerSec: 400, bytesPerOp: 1000, gen0PerOp: 0);

        var verdict = GateEvaluator.EvaluateRow(measured, baseline, 0.90, bandwidthBound: true);

        Assert.That(verdict.OpsOk, Is.True);
        Assert.That(verdict.Gen0Ok, Is.False);
        Assert.That(verdict.Pass, Is.False);
    }

    [Test]
    public void EvaluateRow_StableRowThreeTimesSlower_Fails()
    {
        var measured = Row(StableRow, opsPerSec: 333);
        var baseline = Row(StableRow, opsPerSec: 1000);

        var verdict = GateEvaluator.EvaluateRow(measured, baseline, 0.90, bandwidthBound: false);

        Assert.That(verdict.Pass, Is.False);
        Assert.That(verdict.OpsOk, Is.False);
    }

    [Test]
    public void EvaluateRow_StableRowJustBelowOpsFloor_Fails()
    {
        var measured = Row(StableRow, opsPerSec: 899);
        var baseline = Row(StableRow, opsPerSec: 1000);

        var verdict = GateEvaluator.EvaluateRow(measured, baseline, 0.90, bandwidthBound: false);

        Assert.That(verdict.Pass, Is.False);
        Assert.That(verdict.OpsOk, Is.False);
    }

    [Test]
    public void EvaluateRow_StableRowAtExactOpsFloor_Passes()
    {
        var measured = Row(StableRow, opsPerSec: 900);
        var baseline = Row(StableRow, opsPerSec: 1000);

        var verdict = GateEvaluator.EvaluateRow(measured, baseline, 0.90, bandwidthBound: false);

        Assert.That(verdict.OpsOk, Is.True);
        Assert.That(verdict.Pass, Is.True);
    }

    [Test]
    public void EvaluateRow_StableRowAboveOpsFloor_Passes()
    {
        var measured = Row(StableRow, opsPerSec: 910);
        var baseline = Row(StableRow, opsPerSec: 1000);

        var verdict = GateEvaluator.EvaluateRow(measured, baseline, 0.90, bandwidthBound: false);

        Assert.That(verdict.Pass, Is.True);
        Assert.That(verdict.OpsFloor, Is.EqualTo(0.90).Within(1e-12));
    }

    [Test]
    public void EvaluateRow_CustomToleranceAppliesToStableRowsOnly()
    {
        var stable = GateEvaluator.EvaluateRow(
            Row(StableRow, opsPerSec: 810), Row(StableRow, opsPerSec: 1000), minOpsFraction: 0.80, bandwidthBound: false);
        var unstable = GateEvaluator.EvaluateRow(
            Row(BandwidthBoundRow, opsPerSec: 80), Row(BandwidthBoundRow, opsPerSec: 400), minOpsFraction: 0.80, bandwidthBound: true);

        Assert.That(stable.Pass, Is.True);
        Assert.That(stable.OpsFloor, Is.EqualTo(0.80).Within(1e-12));
        Assert.That(unstable.Pass, Is.False);
        Assert.That(unstable.OpsFloor, Is.EqualTo(GateEvaluator.BandwidthBoundMinOpsFraction));
    }
}