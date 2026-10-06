using System.Diagnostics;
using System.Reflection;
using NUnit.Framework;

namespace Nivara.Tests;

/// <summary>
/// Timing probes are only meaningful in an optimized build. This assembly is compiled
/// with the configuration it is tested in, but <c>System.Numerics.Tensors</c> ships
/// ReadyToRun and stays optimized regardless — so a Debug build compares an unoptimized
/// handwritten kernel against optimized framework code and reports a false regression.
/// See the <c>transpose</c> mode in <c>tests/Nivara.SimdProbe</c> (#482).
/// </summary>
static class TimingGuards
{
    internal static void RequireOptimizedBuildForTiming()
    {
        bool isOptimized = typeof(TimingGuards).Assembly
            .GetCustomAttributes(typeof(DebuggableAttribute), false)
            .Cast<DebuggableAttribute>()
            .Any(a => !a.IsJITOptimizerDisabled);

        if (!isOptimized)
            Assert.Ignore("Timing probe skipped: this is an unoptimized (Debug) build, where the "
                          + "comparison against ReadyToRun framework code is void. Run with -c Release.");
    }
}
