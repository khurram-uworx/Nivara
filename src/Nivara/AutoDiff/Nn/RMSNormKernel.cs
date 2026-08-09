using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Nivara.AutoDiff.Nn;

internal static class RMSNormKernel<T> where T : struct, IFloatingPointIeee754<T>
{
    internal static void PerRowRMSNormForwardKernel(
        T[] srcData, T[] resultData, int rows, int cols, double eps)
    {
        if (typeof(T) == typeof(float))
        {
            var srcFloat = Unsafe.As<T[], float[]>(ref srcData);
            var resFloat = Unsafe.As<T[], float[]>(ref resultData);
            for (int i = 0; i < rows; i++)
            {
                int baseIdx = i * cols;
                var row = srcFloat.AsSpan(baseIdx, cols);
                var dst = resFloat.AsSpan(baseIdx, cols);
                float sumSq = TensorPrimitives.Dot(row, row);
                float rms = MathF.Sqrt(sumSq / cols + (float)eps);
                TensorPrimitives.Multiply(row, 1.0f / rms, dst);
            }
        }
        else if (typeof(T) == typeof(double))
        {
            var srcDouble = Unsafe.As<T[], double[]>(ref srcData);
            var resDouble = Unsafe.As<T[], double[]>(ref resultData);
            for (int i = 0; i < rows; i++)
            {
                int baseIdx = i * cols;
                var row = srcDouble.AsSpan(baseIdx, cols);
                var dst = resDouble.AsSpan(baseIdx, cols);
                double sumSq = TensorPrimitives.Dot(row, row);
                double rms = Math.Sqrt(sumSq / cols + eps);
                TensorPrimitives.Multiply(row, 1.0 / rms, dst);
            }
        }
        else if (typeof(T) == typeof(Half))
        {
            var srcHalf = Unsafe.As<T[], Half[]>(ref srcData);
            var resHalf = Unsafe.As<T[], Half[]>(ref resultData);
            for (int i = 0; i < rows; i++)
            {
                int baseIdx = i * cols;
                var row = srcHalf.AsSpan(baseIdx, cols);
                var dst = resHalf.AsSpan(baseIdx, cols);
                float sumSq = float.CreateChecked(TensorPrimitives.Dot(row, row));
                float rms = MathF.Sqrt(sumSq / cols + (float)eps);
                TensorPrimitives.Multiply(row, (Half)(1.0f / rms), dst);
            }
        }
        else
        {
            for (int i = 0; i < rows; i++)
            {
                int baseIdx = i * cols;
                double sumSq = 0;
                for (int j = 0; j < cols; j++)
                {
                    double v = double.CreateChecked(srcData[baseIdx + j]);
                    sumSq += v * v;
                }
                double rms = Math.Sqrt(sumSq / cols + eps);
                double invRms = 1.0 / rms;
                for (int j = 0; j < cols; j++)
                    resultData[baseIdx + j] = T.CreateChecked(double.CreateChecked(srcData[baseIdx + j]) * invRms);
            }
        }
    }

    internal static void PerRowRMSNormBackwardKernel(
        T[] savedInput, T[] gradOut, T[] gradResult, int rows, int cols, double eps)
    {
        if (typeof(T) == typeof(float))
        {
            var tempArr = ArrayPool<float>.Shared.Rent(cols);
            try
            {
                var tempF = tempArr.AsSpan(0, cols);
                for (int i = 0; i < rows; i++)
                {
                    int baseIdx = i * cols;
                    var sF = MemoryMarshal.Cast<T, float>(savedInput.AsSpan(baseIdx, cols));
                    var gF = MemoryMarshal.Cast<T, float>(gradOut.AsSpan(baseIdx, cols));
                    var dF = MemoryMarshal.Cast<T, float>(gradResult.AsSpan(baseIdx, cols));

                    float sumSq = TensorPrimitives.Dot(sF, sF);
                    float rms = MathF.Sqrt(sumSq / cols + (float)eps);
                    float invRms = 1f / rms;
                    float rms3 = rms * rms * rms;
                    float sumGradX = TensorPrimitives.Dot(gF, sF);
                    float scale = sumGradX / (cols * rms3);

                    TensorPrimitives.Multiply(sF, -scale, tempF);
                    TensorPrimitives.MultiplyAdd(gF, invRms, tempF, dF);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(tempArr);
            }
        }
        else if (typeof(T) == typeof(double))
        {
            var tempArr = ArrayPool<double>.Shared.Rent(cols);
            try
            {
                var tempD = tempArr.AsSpan(0, cols);
                for (int i = 0; i < rows; i++)
                {
                    int baseIdx = i * cols;
                    var sD = MemoryMarshal.Cast<T, double>(savedInput.AsSpan(baseIdx, cols));
                    var gD = MemoryMarshal.Cast<T, double>(gradOut.AsSpan(baseIdx, cols));
                    var dD = MemoryMarshal.Cast<T, double>(gradResult.AsSpan(baseIdx, cols));

                    double sumSq = TensorPrimitives.Dot(sD, sD);
                    double rms = Math.Sqrt(sumSq / cols + eps);
                    double invRms = 1.0 / rms;
                    double rms3 = rms * rms * rms;
                    double sumGradX = TensorPrimitives.Dot(gD, sD);
                    double scale = sumGradX / (cols * rms3);

                    TensorPrimitives.Multiply(sD, -scale, tempD);
                    TensorPrimitives.MultiplyAdd(gD, invRms, tempD, dD);
                }
            }
            finally
            {
                ArrayPool<double>.Shared.Return(tempArr);
            }
        }
        else if (typeof(T) == typeof(Half))
        {
            var tempArr = ArrayPool<Half>.Shared.Rent(cols);
            try
            {
                var tempH = tempArr.AsSpan(0, cols);
                for (int i = 0; i < rows; i++)
                {
                    int baseIdx = i * cols;
                    var sH = MemoryMarshal.Cast<T, Half>(savedInput.AsSpan(baseIdx, cols));
                    var gH = MemoryMarshal.Cast<T, Half>(gradOut.AsSpan(baseIdx, cols));
                    var dH = MemoryMarshal.Cast<T, Half>(gradResult.AsSpan(baseIdx, cols));

                    float sumSq = float.CreateChecked(TensorPrimitives.Dot(sH, sH));
                    float rms = MathF.Sqrt(sumSq / cols + (float)eps);
                    float invRms = 1f / rms;
                    float rms3 = rms * rms * rms;
                    float sumGradX = float.CreateChecked(TensorPrimitives.Dot(gH, sH));
                    float scale = sumGradX / (cols * rms3);

                    TensorPrimitives.Multiply(sH, (Half)(-scale), tempH);
                    TensorPrimitives.MultiplyAdd(gH, (Half)invRms, tempH, dH);
                }
            }
            finally
            {
                ArrayPool<Half>.Shared.Return(tempArr);
            }
        }
        else
        {
            for (int i = 0; i < rows; i++)
            {
                int baseIdx = i * cols;

                double sumSq = 0;
                for (int j = 0; j < cols; j++)
                {
                    double v = double.CreateChecked(savedInput[baseIdx + j]);
                    sumSq += v * v;
                }
                double rms = Math.Sqrt(sumSq / cols + eps);
                double invRms = 1.0 / rms;
                double rms3 = rms * rms * rms;

                double sumGradX = 0;
                for (int j = 0; j < cols; j++)
                {
                    double g = double.CreateChecked(gradOut[baseIdx + j]);
                    double v = double.CreateChecked(savedInput[baseIdx + j]);
                    sumGradX += g * v;
                }

                double scale = sumGradX / (cols * rms3);

                for (int j = 0; j < cols; j++)
                {
                    double g = double.CreateChecked(gradOut[baseIdx + j]);
                    double v = double.CreateChecked(savedInput[baseIdx + j]);
                    gradResult[baseIdx + j] = T.CreateChecked(g * invRms - v * scale);
                }
            }
        }
    }
}
