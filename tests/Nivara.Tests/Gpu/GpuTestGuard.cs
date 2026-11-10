using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Shared skip/availability guard for GPU tests. By default a missing OpenCL device causes
/// <see cref="Assert.Ignore"/> (skipped). When <c>NIVARA_REQUIRE_GPU=1</c> (or <c>true</c>)
/// is set, a missing device fails the test instead, so CI consumers cannot mistake an
/// all-skipped run for green parity.
/// </summary>
internal static class GpuTestGuard
{
    public static void RequireGpu()
    {
        if (SilkRuntime.IsAvailable(out string? reason))
            return;

        if (IsRequired())
            Assert.Fail($"OpenCL GPU required but unavailable: {reason}");

        Assert.Ignore($"No OpenCL GPU available; skipping Silk verification. ({reason})");
    }

    static bool IsRequired()
    {
        string? value = Environment.GetEnvironmentVariable("NIVARA_REQUIRE_GPU");
        return string.Equals(value, "1", StringComparison.Ordinal)
            || string.Equals(value, "true", StringComparison.OrdinalIgnoreCase);
    }
}
