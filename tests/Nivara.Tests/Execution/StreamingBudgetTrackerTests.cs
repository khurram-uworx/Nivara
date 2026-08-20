using Nivara.Diagnostics;
using Nivara.Execution;
using NUnit.Framework;

namespace Nivara.Tests.Execution;

[TestFixture]
public class StreamingBudgetTrackerTests
{
    [Test]
    public void RecordFrame_AddsToEstimatedUsage()
    {
        var tracker = new StreamingBudgetTracker(long.MaxValue);
        var frame = NivaraFrame.Create(("A", NivaraColumn<int>.Create(new[] { 1, 2, 3 })));
        try
        {
            tracker.RecordFrame(frame);
            Assert.That(tracker.EstimatedMemoryUsage, Is.GreaterThan(0));
        }
        finally { frame.Dispose(); }
    }

    [Test]
    public void RecordFrame_AccumulatesMultipleFrames()
    {
        var tracker = new StreamingBudgetTracker(long.MaxValue);
        var frame1 = NivaraFrame.Create(("A", NivaraColumn<int>.Create(new[] { 1, 2, 3 })));
        var frame2 = NivaraFrame.Create(("A", NivaraColumn<int>.Create(new[] { 4, 5, 6 })));
        try
        {
            tracker.RecordFrame(frame1);
            var afterFirst = tracker.EstimatedMemoryUsage;
            tracker.RecordFrame(frame2);
            Assert.That(tracker.EstimatedMemoryUsage, Is.GreaterThan(afterFirst));
        }
        finally
        {
            frame1.Dispose();
            frame2.Dispose();
        }
    }

    [Test]
    public void IsMemoryBudgetExceeded_TrueWhenOverBudget()
    {
        var tracker = new StreamingBudgetTracker(memoryBudget: 1, budgetMultiplier: 1.0);
        var frame = NivaraFrame.Create(("A", NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 })));
        try
        {
            tracker.RecordFrame(frame);
            Assert.That(tracker.IsBudgetExceeded, Is.True,
                "Budget should be exceeded when a frame's estimated memory exceeds the threshold");
        }
        finally { frame.Dispose(); }
    }

    [Test]
    public void IsMemoryBudgetExceeded_FalseWhenUnderBudget()
    {
        var tracker = new StreamingBudgetTracker(memoryBudget: long.MaxValue);
        var frame = NivaraFrame.Create(("A", NivaraColumn<int>.Create(new[] { 1 })));
        try
        {
            tracker.RecordFrame(frame);
            Assert.That(tracker.IsBudgetExceeded, Is.False);
        }
        finally { frame.Dispose(); }
    }

    [Test]
    public void RecordWarning_EmitsOnce()
    {
        var tracker = new StreamingBudgetTracker(memoryBudget: 1, budgetMultiplier: 1.0);
        var diagnostics = new ExecutionDiagnostics();
        var frame = NivaraFrame.Create(("A", NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 })));
        try
        {
            tracker.RecordFrame(frame);
            tracker.RecordWarningIfExceeded(diagnostics);
            tracker.RecordWarningIfExceeded(diagnostics);
            tracker.RecordWarningIfExceeded(diagnostics);

            var budgetWarnings = diagnostics.Warnings.Where(w =>
                w.Message.Contains("budget", StringComparison.OrdinalIgnoreCase)).ToList();
            Assert.That(budgetWarnings.Count, Is.EqualTo(1),
                "Warning should be emitted exactly once");
        }
        finally { frame.Dispose(); }
    }

    [Test]
    public void RecordWarning_NoEmissionWhenUnderBudget()
    {
        var tracker = new StreamingBudgetTracker(memoryBudget: long.MaxValue);
        var diagnostics = new ExecutionDiagnostics();
        var frame = NivaraFrame.Create(("A", NivaraColumn<int>.Create(new[] { 1 })));
        try
        {
            tracker.RecordFrame(frame);
            tracker.RecordWarningIfExceeded(diagnostics);
            Assert.That(diagnostics.Warnings, Is.Empty);
        }
        finally { frame.Dispose(); }
    }

    [Test]
    public void RecordWarning_NoEmissionWhenNoDiagnostics()
    {
        var tracker = new StreamingBudgetTracker(memoryBudget: 1, budgetMultiplier: 1.0);
        var frame = NivaraFrame.Create(("A", NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 })));
        try
        {
            tracker.RecordFrame(frame);
            Assert.DoesNotThrow(() => tracker.RecordWarningIfExceeded(null));
        }
        finally { frame.Dispose(); }
    }

    [Test]
    public void Reset_ClearsState()
    {
        var tracker = new StreamingBudgetTracker(memoryBudget: 1, budgetMultiplier: 1.0);
        var frame = NivaraFrame.Create(("A", NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 })));
        try
        {
            tracker.RecordFrame(frame);
            Assert.That(tracker.EstimatedMemoryUsage, Is.GreaterThan(0));

            tracker.Reset();
            Assert.That(tracker.EstimatedMemoryUsage, Is.EqualTo(0));
            Assert.That(tracker.IsBudgetExceeded, Is.False);
        }
        finally { frame.Dispose(); }
    }

    [Test]
    public void WarningThreshold_IsBudgetTimesMultiplier()
    {
        var tracker = new StreamingBudgetTracker(memoryBudget: 1000, budgetMultiplier: 3.0);
        Assert.That(tracker.WarningThreshold, Is.EqualTo(3000));
    }

    [Test]
    public void WarningThreshold_DefaultMultiplierIsTwo()
    {
        var tracker = new StreamingBudgetTracker(memoryBudget: 500);
        Assert.That(tracker.WarningThreshold, Is.EqualTo(1000));
    }
}
