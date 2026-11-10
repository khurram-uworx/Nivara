using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Device-free pin that the BF16/F16 Laya-head GEMM variants stay registered in
/// <see cref="SilkKernels.KernelNames"/> (so the OpenCL translation unit compiles
/// them) and remain reachable from <see cref="SilkLayaHeadGpuRunner"/> dispatch, so a
/// refactor cannot silently drop a case and revert to the F32 kernels.
/// </summary>
public sealed class SilkLayaHeadKernelRegistrationTests
{
    static readonly (string Base, string NarrowBf16, string NarrowF16, string ActivationBf16)[] Variants =
    [
        (SilkKernels.GemmQkv, SilkKernels.GemmQkvBf16, SilkKernels.GemmQkvF16, SilkKernels.GemmQkvBf16Act),
        (SilkKernels.GemmGelu, SilkKernels.GemmGeluBf16, SilkKernels.GemmGeluF16, SilkKernels.GemmGeluBf16Act),
        (SilkKernels.GemmRelu, SilkKernels.GemmReluBf16, SilkKernels.GemmReluF16, SilkKernels.GemmReluBf16Act),
        (SilkKernels.GemmBias, SilkKernels.GemmBiasBf16, SilkKernels.GemmBiasF16, SilkKernels.GemmBiasBf16Act),
    ];

    [Test]
    public void HeadGemmVariants_AreRegisteredInKernelNames()
    {
        foreach (var v in Variants)
        {
            Assert.That(SilkKernels.KernelNames, Does.Contain(v.NarrowBf16), v.NarrowBf16);
            Assert.That(SilkKernels.KernelNames, Does.Contain(v.NarrowF16), v.NarrowF16);
            Assert.That(SilkKernels.KernelNames, Does.Contain(v.ActivationBf16), v.ActivationBf16);
            Assert.That(SilkKernels.KernelNames, Does.Contain(v.Base), v.Base);
        }
    }

    [Test]
    public void HeadGemmVariants_AreDefinedInSource()
    {
        string source = SilkKernels.Source;
        foreach (var v in Variants)
        {
            Assert.That(source, Does.Contain($"__kernel void {v.NarrowBf16}("), v.NarrowBf16);
            Assert.That(source, Does.Contain($"__kernel void {v.NarrowF16}("), v.NarrowF16);
            Assert.That(source, Does.Contain($"__kernel void {v.ActivationBf16}("), v.ActivationBf16);
        }
    }

    [Test]
    public void HeadGemmDispatch_F32_UsesBaseKernel()
    {
        foreach (var v in Variants)
            Assert.That(SilkLayaHeadGpuRunner.ChooseKernel(v.Base), Is.EqualTo(v.Base));
    }

    [Test]
    public void HeadGemmDispatch_NarrowBf16_UsesBf16Variant()
    {
        foreach (var v in Variants)
            Assert.That(SilkLayaHeadGpuRunner.ChooseKernel(v.Base, narrowBf16: true), Is.EqualTo(v.NarrowBf16));
    }

    [Test]
    public void HeadGemmDispatch_NarrowF16_UsesF16Variant()
    {
        foreach (var v in Variants)
            Assert.That(SilkLayaHeadGpuRunner.ChooseKernel(v.Base, narrowF16: true), Is.EqualTo(v.NarrowF16));
    }

    [Test]
    public void HeadGemmDispatch_ActivationBf16_UsesBf16ActVariant()
    {
        foreach (var v in Variants)
            Assert.That(SilkLayaHeadGpuRunner.ChooseKernel(v.Base, activationBf16: true), Is.EqualTo(v.ActivationBf16));
    }

    [Test]
    public void HeadGemmDispatch_MappedKernel_IsRegistered()
    {
        foreach (var v in Variants)
        {
            string[] mapped =
            [
                SilkLayaHeadGpuRunner.ChooseKernel(v.Base),
                SilkLayaHeadGpuRunner.ChooseKernel(v.Base, narrowBf16: true),
                SilkLayaHeadGpuRunner.ChooseKernel(v.Base, narrowF16: true),
                SilkLayaHeadGpuRunner.ChooseKernel(v.Base, activationBf16: true),
            ];
            foreach (string name in mapped)
                Assert.That(SilkKernels.KernelNames, Does.Contain(name), name);
        }
    }
}
