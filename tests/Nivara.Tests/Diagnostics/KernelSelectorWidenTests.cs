using Nivara.Diagnostics;
using Nivara.Primitives;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests;

[TestFixture]
public class KernelSelectorWidenTests
{
    [TearDown]
    public void ResetToggle() => NivaraPrimitives.UseWidenSimd = false;

    [Test]
    public void DetermineKernelType_NarrowFloat_ToggleOff_ReturnsScalar()
    {
        Assert.That(KernelSelector.DetermineKernelType<Half>(128), Is.EqualTo(KernelType.Scalar));
        Assert.That(KernelSelector.DetermineKernelType<BFloat16>(128), Is.EqualTo(KernelType.Scalar));
    }

    [Test]
    public void DetermineKernelType_NarrowFloat_ToggleOn_HardwareAboveThreshold_ReturnsWidenToFloatSimd()
    {
        if (!Vector.IsHardwareAccelerated)
            Assert.Ignore("Hardware acceleration not available");

        NivaraPrimitives.UseWidenSimd = true;
        var threshold = Vector<byte>.Count * 4;

        Assert.That(KernelSelector.DetermineKernelType<Half>(threshold), Is.EqualTo(KernelType.WidenToFloatSimd));
        Assert.That(KernelSelector.DetermineKernelType<BFloat16>(threshold), Is.EqualTo(KernelType.WidenToFloatSimd));
    }

    [Test]
    public void DetermineKernelType_NarrowFloat_ToggleOn_BelowThreshold_ReturnsScalar()
    {
        if (!Vector.IsHardwareAccelerated)
            Assert.Ignore("Hardware acceleration not available");

        NivaraPrimitives.UseWidenSimd = true;
        var threshold = Vector<byte>.Count * 4;

        Assert.That(KernelSelector.DetermineKernelType<Half>(threshold - 1), Is.EqualTo(KernelType.Scalar));
        Assert.That(KernelSelector.DetermineKernelType<BFloat16>(1), Is.EqualTo(KernelType.Scalar));
    }

    [Test]
    public void DetermineKernelType_WideFloat_NeverSelectsWidenToFloatSimd()
    {
        NivaraPrimitives.UseWidenSimd = true;
        var threshold = Vector<byte>.Count * 4;

        Assert.That(KernelSelector.DetermineKernelType<float>(threshold), Is.Not.EqualTo(KernelType.WidenToFloatSimd));
        Assert.That(KernelSelector.DetermineKernelType<double>(threshold), Is.Not.EqualTo(KernelType.WidenToFloatSimd));
    }

    [Test]
    public void ShouldWiden_NarrowFloat_ToggleOff_ReturnsFalse()
    {
        Assert.That(WidenPrimitives.ShouldWiden<Half>(1024), Is.False);
        Assert.That(WidenPrimitives.ShouldWiden<BFloat16>(1024), Is.False);
    }

    [Test]
    public void ShouldWiden_NarrowFloat_ToggleOn_RespectsLengthGate()
    {
        if (!Vector.IsHardwareAccelerated)
            Assert.Ignore("Hardware acceleration not available");

        NivaraPrimitives.UseWidenSimd = true;
        var threshold = Vector<byte>.Count * 4;

        Assert.That(WidenPrimitives.ShouldWiden<Half>(threshold), Is.True);
        Assert.That(WidenPrimitives.ShouldWiden<BFloat16>(threshold), Is.True);
        Assert.That(WidenPrimitives.ShouldWiden<Half>(threshold - 1), Is.False);
    }

    [Test]
    public void ShouldWiden_WideFloat_ToggleOn_ReturnsFalse()
    {
        if (!Vector.IsHardwareAccelerated)
            Assert.Ignore("Hardware acceleration not available");

        NivaraPrimitives.UseWidenSimd = true;
        Assert.That(WidenPrimitives.ShouldWiden<float>(4096), Is.False);
        Assert.That(WidenPrimitives.ShouldWiden<double>(4096), Is.False);
    }
}
