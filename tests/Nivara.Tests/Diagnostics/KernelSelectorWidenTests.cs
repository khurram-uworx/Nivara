using Nivara.Diagnostics;
using Nivara.Primitives;
using NUnit.Framework;
using System;
using System.Numerics;

namespace Nivara;

[TestFixture]
public class KernelSelectorWidenTests
{
    const string OptOut = "Nivara.Primitives.DisableWidenSimd";

    [TearDown]
    public void ResetToggle() => AppContext.SetSwitch(OptOut, false);

    [Test]
    public void DetermineKernelType_NarrowFloat_OptedOut_ReturnsScalar()
    {
        AppContext.SetSwitch(OptOut, true);
        Assert.That(NivaraPrimitives.DisableWidenSimd, Is.True);
        Assert.That(KernelSelector.DetermineKernelType<Half>(128), Is.EqualTo(KernelType.Scalar));
        Assert.That(KernelSelector.DetermineKernelType<BFloat16>(128), Is.EqualTo(KernelType.Scalar));
    }

    [Test]
    public void DetermineKernelType_NarrowFloat_DefaultOn_HardwareAboveThreshold_ReturnsWidenToFloatSimd()
    {
        if (!Vector.IsHardwareAccelerated)
            Assert.Ignore("Hardware acceleration not available");

        var threshold = Vector<byte>.Count * 4;

        Assert.That(KernelSelector.DetermineKernelType<Half>(threshold), Is.EqualTo(KernelType.WidenToFloatSimd));
        Assert.That(KernelSelector.DetermineKernelType<BFloat16>(threshold), Is.EqualTo(KernelType.WidenToFloatSimd));
    }

    [Test]
    public void DetermineKernelType_NarrowFloat_DefaultOn_BelowThreshold_ReturnsScalar()
    {
        if (!Vector.IsHardwareAccelerated)
            Assert.Ignore("Hardware acceleration not available");

        var threshold = Vector<byte>.Count * 4;

        Assert.That(KernelSelector.DetermineKernelType<Half>(threshold - 1), Is.EqualTo(KernelType.Scalar));
        Assert.That(KernelSelector.DetermineKernelType<BFloat16>(1), Is.EqualTo(KernelType.Scalar));
    }

    [Test]
    public void DetermineKernelType_WideFloat_NeverSelectsWidenToFloatSimd()
    {
        var threshold = Vector<byte>.Count * 4;

        Assert.That(KernelSelector.DetermineKernelType<float>(threshold), Is.Not.EqualTo(KernelType.WidenToFloatSimd));
        Assert.That(KernelSelector.DetermineKernelType<double>(threshold), Is.Not.EqualTo(KernelType.WidenToFloatSimd));
    }

    [Test]
    public void ShouldWiden_NarrowFloat_OptedOut_ReturnsFalse()
    {
        AppContext.SetSwitch(OptOut, true);
        Assert.That(WidenPrimitives.ShouldWiden<Half>(1024), Is.False);
        Assert.That(WidenPrimitives.ShouldWiden<BFloat16>(1024), Is.False);
    }

    [Test]
    public void ShouldWiden_NarrowFloat_DefaultOn_RespectsLengthGate()
    {
        if (!Vector.IsHardwareAccelerated)
            Assert.Ignore("Hardware acceleration not available");

        var threshold = Vector<byte>.Count * 4;

        Assert.That(WidenPrimitives.ShouldWiden<Half>(threshold), Is.True);
        Assert.That(WidenPrimitives.ShouldWiden<BFloat16>(threshold), Is.True);
        Assert.That(WidenPrimitives.ShouldWiden<Half>(threshold - 1), Is.False);
    }

    [Test]
    public void ShouldWiden_WideFloat_DefaultOn_ReturnsFalse()
    {
        if (!Vector.IsHardwareAccelerated)
            Assert.Ignore("Hardware acceleration not available");

        Assert.That(WidenPrimitives.ShouldWiden<float>(4096), Is.False);
        Assert.That(WidenPrimitives.ShouldWiden<double>(4096), Is.False);
    }
}
