using Nivara.Diagnostics;
using Nivara.Storage;
using System.Numerics;

namespace Nivara;

static class KernelSelector
{
    public static KernelType DetermineKernelType(int length, bool isVectorizable)
    {
        if (!isVectorizable)
            return KernelType.Scalar;

        if (!Vector.IsHardwareAccelerated)
            return KernelType.Scalar;

        var vectorSize = Vector<byte>.Count;
        if (length < vectorSize * 4)
            return KernelType.Scalar;

        return KernelType.Vectorized;
    }

    public static KernelType DetermineKernelType<T>(int length)
    {
        return DetermineKernelType(length, ColumnStorageFactory.IsVectorizable<T>());
    }

    public static KernelType DetermineBatchKernelType<T>()
    {
        var t = typeof(T);
        if ((t == typeof(float) || t == typeof(double)) && Vector.IsHardwareAccelerated)
            return KernelType.Vectorized;
        return KernelType.Scalar;
    }
}
