using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Pins that constructing and disposing a <see cref="SilkRuntime"/> without ever running a
/// kernel does not crash the test host (issue #1150): the OpenCL driver on Intel Arc
/// access-violates in <c>clReleaseCommandQueue</c> for an unflushed queue.
/// </summary>
[Category("Gpu")]
public sealed class SilkRuntimeLifecycleTests
{
    [Test]
    public void SilkRuntime_ConstructAndDispose_WithoutRunningKernel_DoesNotCrash()
    {
        SilkRuntime runtime;
        try
        {
            runtime = new SilkRuntime();
        }
        catch
        {
            Assert.Ignore("No OpenCL GPU available; skipping Silk runtime lifecycle verification.");
            return;
        }

        using (runtime) { }
    }
}
