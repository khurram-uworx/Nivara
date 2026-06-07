using Nivara;
using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;

namespace Nivara.Extensions.AutoDiff.Utilities;

/// <summary>
/// Utility functions for gradient management, memory cleanup, and debugging.
/// Provides helper methods for common gradient operations in reverse-mode automatic differentiation.
/// </summary>
public static class GradientUtils
{
    #region Gradient Management

    /// <summary>
    /// Clears all gradients in the computation graph reachable from the specified tensor.
    /// </summary>
    public static void ZeroGrad<T>(ReverseGradTensor<T> tensor) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        ComputationGraph.ZeroGrad(tensor);
    }

    /// <summary>
    /// Clears gradients for multiple tensors at once.
    /// </summary>
    public static void ZeroGrad<T>(IEnumerable<ReverseGradTensor<T>> tensors) where T : struct, INumber<T>
    {
        if (tensors == null)
            throw new ArgumentNullException(nameof(tensors));

        foreach (var tensor in tensors)
        {
            if (tensor != null)
            {
                tensor.ZeroGrad();
            }
        }
    }

    /// <summary>
    /// Detaches a tensor from the computation graph, returning a new tensor without gradient tracking.
    /// </summary>
    public static ReverseGradTensor<T> Detach<T>(ReverseGradTensor<T> tensor) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        return tensor.Detach();
    }

    /// <summary>
    /// Detaches multiple tensors from the computation graph.
    /// </summary>
    public static IEnumerable<ReverseGradTensor<T>> Detach<T>(IEnumerable<ReverseGradTensor<T>> tensors) where T : struct, INumber<T>
    {
        if (tensors == null)
            throw new ArgumentNullException(nameof(tensors));

        return tensors.Select(t => t?.Detach()).Where(t => t != null)!;
    }

    #endregion

    #region Gradient Clipping

    /// <summary>
    /// Clips gradient values to prevent exploding gradients.
    /// Values are clipped to the range [-maxValue, maxValue].
    /// </summary>
    public static void ClipGradValue<T>(ReverseGradTensor<T> tensor, T maxValue) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (maxValue <= T.Zero)
            throw new ArgumentException("Max value must be positive", nameof(maxValue));

        if (tensor.Grad == null)
            return;

        var grad = tensor.Grad;
        int n = grad.Length;
        var clippedData = ArrayPool<T>.Shared.Rent(n);
        var minValue = -maxValue;
        var span = grad.AsSpan();

        try
        {
            if (typeof(T) == typeof(float))
            {
                var fSpan = MemoryMarshal.Cast<T, float>(span);
                var fClipped = MemoryMarshal.Cast<T, float>(clippedData.AsSpan(0, n));
                for (int i = 0; i < n; i++)
                {
                    if (!grad.IsNull(i))
                    {
                        var v = fSpan[i];
                        fClipped[i] = v > (float)(object)maxValue! ? (float)(object)maxValue!
                            : v < (float)(object)minValue! ? (float)(object)minValue! : v;
                    }
                }
            }
            else if (typeof(T) == typeof(double))
            {
                var dSpan = MemoryMarshal.Cast<T, double>(span);
                var dClipped = MemoryMarshal.Cast<T, double>(clippedData.AsSpan(0, n));
                for (int i = 0; i < n; i++)
                {
                    if (!grad.IsNull(i))
                    {
                        var v = dSpan[i];
                        dClipped[i] = v > (double)(object)maxValue! ? (double)(object)maxValue!
                            : v < (double)(object)minValue! ? (double)(object)minValue! : v;
                    }
                }
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    if (!grad.IsNull(i))
                    {
                        var value = grad[i];
                        clippedData[i] = value > maxValue ? maxValue
                            : value < minValue ? minValue : value;
                    }
                }
            }

            tensor.Grad = NivaraColumn<T>.Create(clippedData.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(clippedData, clearArray: true);
        }
    }

    /// <summary>
    /// Clips gradient norm to prevent exploding gradients.
    /// If the gradient norm exceeds maxNorm, the gradient is scaled down proportionally.
    /// </summary>
    public static void ClipGradNorm<T>(ReverseGradTensor<T> tensor, double maxNorm) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (maxNorm <= 0)
            throw new ArgumentException("Max norm must be positive", nameof(maxNorm));

        if (tensor.Grad == null)
            return;

        var grad = tensor.Grad;
        int n = grad.Length;
        var span = grad.AsSpan();

        double normSquared = 0.0;
        if (typeof(T) == typeof(float))
        {
            var fSpan = MemoryMarshal.Cast<T, float>(span);
            for (int i = 0; i < n; i++)
                if (!grad.IsNull(i))
                    normSquared += fSpan[i] * fSpan[i];
        }
        else if (typeof(T) == typeof(double))
        {
            var dSpan = MemoryMarshal.Cast<T, double>(span);
            for (int i = 0; i < n; i++)
                if (!grad.IsNull(i))
                    normSquared += dSpan[i] * dSpan[i];
        }
        else
        {
            for (int i = 0; i < n; i++)
                if (!grad.IsNull(i))
                {
                    dynamic value = grad[i];
                    normSquared += (double)(value * value);
                }
        }

        var norm = Math.Sqrt(normSquared);

        if (norm > maxNorm)
        {
            var scale = maxNorm / norm;
            var clippedData = ArrayPool<T>.Shared.Rent(n);

            try
            {
                if (typeof(T) == typeof(float))
                {
                    var scaleFloat = (float)scale;
                    var fSpan = MemoryMarshal.Cast<T, float>(span);
                    var fClipped = MemoryMarshal.Cast<T, float>(clippedData.AsSpan(0, n));
                    for (int i = 0; i < n; i++)
                        fClipped[i] = !grad.IsNull(i) ? fSpan[i] * scaleFloat : 0f;
                }
                else if (typeof(T) == typeof(double))
                {
                    var dSpan = MemoryMarshal.Cast<T, double>(span);
                    var dClipped = MemoryMarshal.Cast<T, double>(clippedData.AsSpan(0, n));
                    for (int i = 0; i < n; i++)
                        dClipped[i] = !grad.IsNull(i) ? dSpan[i] * scale : 0.0;
                }
                else
                {
                    for (int i = 0; i < n; i++)
                    {
                        if (!grad.IsNull(i))
                        {
                            dynamic value = grad[i];
                            clippedData[i] = (T)(object)(value * scale)!;
                        }
                    }
                }

                tensor.Grad = NivaraColumn<T>.Create(clippedData.AsSpan(0, n));
            }
            finally
            {
                ArrayPool<T>.Shared.Return(clippedData, clearArray: true);
            }
        }
    }

    /// <summary>
    /// Clips gradients for multiple tensors by their global norm.
    /// </summary>
    public static void ClipGradNorm<T>(IEnumerable<ReverseGradTensor<T>> tensors, double maxNorm) where T : struct, INumber<T>
    {
        if (tensors == null)
            throw new ArgumentNullException(nameof(tensors));

        if (maxNorm <= 0)
            throw new ArgumentException("Max norm must be positive", nameof(maxNorm));

        var tensorList = tensors.Where(t => t != null && t.Grad != null).ToList();
        if (tensorList.Count == 0)
            return;

        double globalNormSquared = 0.0;
        foreach (var tensor in tensorList)
        {
            var grad = tensor.Grad!;
            int n = grad.Length;
            var span = grad.AsSpan();

            if (typeof(T) == typeof(float))
            {
                var fSpan = MemoryMarshal.Cast<T, float>(span);
                for (int i = 0; i < n; i++)
                    if (!grad.IsNull(i))
                        globalNormSquared += fSpan[i] * fSpan[i];
            }
            else if (typeof(T) == typeof(double))
            {
                var dSpan = MemoryMarshal.Cast<T, double>(span);
                for (int i = 0; i < n; i++)
                    if (!grad.IsNull(i))
                        globalNormSquared += dSpan[i] * dSpan[i];
            }
            else
            {
                for (int i = 0; i < n; i++)
                    if (!grad.IsNull(i))
                    {
                        dynamic value = grad[i];
                        globalNormSquared += (double)(value * value);
                    }
            }
        }

        var globalNorm = Math.Sqrt(globalNormSquared);

        if (globalNorm > maxNorm)
        {
            var scale = maxNorm / globalNorm;

            foreach (var tensor in tensorList)
            {
                var grad = tensor.Grad!;
                int n = grad.Length;
                var span = grad.AsSpan();
                var clippedData = ArrayPool<T>.Shared.Rent(n);

                try
                {
                    if (typeof(T) == typeof(float))
                    {
                        var scaleFloat = (float)scale;
                        var fSpan = MemoryMarshal.Cast<T, float>(span);
                        var fClipped = MemoryMarshal.Cast<T, float>(clippedData.AsSpan(0, n));
                        for (int i = 0; i < n; i++)
                            fClipped[i] = !grad.IsNull(i) ? fSpan[i] * scaleFloat : 0f;
                    }
                    else if (typeof(T) == typeof(double))
                    {
                        var dSpan = MemoryMarshal.Cast<T, double>(span);
                        var dClipped = MemoryMarshal.Cast<T, double>(clippedData.AsSpan(0, n));
                        for (int i = 0; i < n; i++)
                            dClipped[i] = !grad.IsNull(i) ? dSpan[i] * scale : 0.0;
                    }
                    else
                    {
                        for (int i = 0; i < n; i++)
                        {
                            if (!grad.IsNull(i))
                            {
                                dynamic value = grad[i];
                                clippedData[i] = (T)(object)(value * scale)!;
                            }
                        }
                    }

                    tensor.Grad = NivaraColumn<T>.Create(clippedData.AsSpan(0, n));
                }
                finally
                {
                    ArrayPool<T>.Shared.Return(clippedData, clearArray: true);
                }
            }
        }
    }

    #endregion

    #region Constant Tensor Creation

    /// <summary>
    /// Creates a constant tensor that doesn't require gradients.
    /// </summary>
    public static ReverseGradTensor<T> Constant<T>(T[] data) where T : struct, INumber<T>
    {
        if (data == null)
            throw new ArgumentNullException(nameof(data));

        return ReverseGradTensor<T>.FromArray(data, requiresGrad: false);
    }

    /// <summary>
    /// Creates a constant tensor from a NivaraColumn that doesn't require gradients.
    /// </summary>
    public static ReverseGradTensor<T> Constant<T>(NivaraColumn<T> column) where T : struct, INumber<T>
    {
        if (column == null)
            throw new ArgumentNullException(nameof(column));

        return ReverseGradTensor<T>.FromColumn(column, requiresGrad: false);
    }

    /// <summary>
    /// Creates a constant tensor filled with zeros.
    /// </summary>
    public static ReverseGradTensor<T> Zeros<T>(int length) where T : struct, INumber<T>
    {
        if (length <= 0)
            throw new ArgumentException("Length must be positive", nameof(length));

        var data = new T[length];
        Array.Fill(data, T.Zero);
        return ReverseGradTensor<T>.FromArray(data, requiresGrad: false);
    }

    /// <summary>
    /// Creates a constant tensor filled with ones.
    /// </summary>
    public static ReverseGradTensor<T> Ones<T>(int length) where T : struct, INumber<T>
    {
        if (length <= 0)
            throw new ArgumentException("Length must be positive", nameof(length));

        var data = new T[length];
        Array.Fill(data, T.One);
        return ReverseGradTensor<T>.FromArray(data, requiresGrad: false);
    }

    /// <summary>
    /// Creates a constant tensor filled with a specific value.
    /// </summary>
    public static ReverseGradTensor<T> Full<T>(int length, T value) where T : struct, INumber<T>
    {
        if (length <= 0)
            throw new ArgumentException("Length must be positive", nameof(length));

        var data = new T[length];
        Array.Fill(data, value);
        return ReverseGradTensor<T>.FromArray(data, requiresGrad: false);
    }

    #endregion

    #region Computation Graph Inspection

    /// <summary>
    /// Gets diagnostic information about the computation graph rooted at the specified tensor.
    /// </summary>
    public static Dictionary<string, object> GetGraphInfo<T>(ReverseGradTensor<T> tensor) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        return ComputationGraph.GetGraphInfo(tensor);
    }

    /// <summary>
    /// Prints a human-readable summary of the computation graph.
    /// </summary>
    public static string PrintGraphSummary<T>(ReverseGradTensor<T> tensor) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var info = GetGraphInfo(tensor);
        var sb = new StringBuilder();

        sb.AppendLine("Computation Graph Summary:");
        sb.AppendLine($"  Total Nodes: {info["TotalNodes"]}");
        sb.AppendLine($"  Is Leaf: {info["IsLeaf"]}");
        sb.AppendLine($"  Requires Grad: {info["RequiresGrad"]}");

        if (info["OperationCounts"] is Dictionary<string, int> opCounts && opCounts.Count > 0)
        {
            sb.AppendLine("  Operation Counts:");
            foreach (var kvp in opCounts.OrderByDescending(x => x.Value))
            {
                sb.AppendLine($"    {kvp.Key}: {kvp.Value}");
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// Checks if a tensor has any gradients computed.
    /// </summary>
    public static bool HasGradient<T>(ReverseGradTensor<T> tensor) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        return tensor.Grad != null;
    }

    /// <summary>
    /// Gets the gradient norm (L2 norm) for a tensor.
    /// </summary>
    public static double GetGradientNorm<T>(ReverseGradTensor<T> tensor) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (tensor.Grad == null)
            return 0.0;

        var grad = tensor.Grad;
        int n = grad.Length;
        var span = grad.AsSpan();
        double normSquared = 0.0;

        if (typeof(T) == typeof(float))
        {
            var fSpan = MemoryMarshal.Cast<T, float>(span);
            for (int i = 0; i < n; i++)
                if (!grad.IsNull(i))
                    normSquared += fSpan[i] * fSpan[i];
        }
        else if (typeof(T) == typeof(double))
        {
            var dSpan = MemoryMarshal.Cast<T, double>(span);
            for (int i = 0; i < n; i++)
                if (!grad.IsNull(i))
                    normSquared += dSpan[i] * dSpan[i];
        }
        else
        {
            for (int i = 0; i < n; i++)
                if (!grad.IsNull(i))
                {
                    dynamic value = grad[i];
                    normSquared += (double)(value * value);
                }
        }

        return Math.Sqrt(normSquared);
    }

    /// <summary>
    /// Gets the global gradient norm across multiple tensors.
    /// </summary>
    public static double GetGlobalGradientNorm<T>(IEnumerable<ReverseGradTensor<T>> tensors) where T : struct, INumber<T>
    {
        if (tensors == null)
            throw new ArgumentNullException(nameof(tensors));

        double globalNormSquared = 0.0;

        foreach (var tensor in tensors.Where(t => t != null && t.Grad != null))
        {
            var grad = tensor.Grad!;
            int n = grad.Length;
            var span = grad.AsSpan();

            if (typeof(T) == typeof(float))
            {
                var fSpan = MemoryMarshal.Cast<T, float>(span);
                for (int i = 0; i < n; i++)
                    if (!grad.IsNull(i))
                        globalNormSquared += fSpan[i] * fSpan[i];
            }
            else if (typeof(T) == typeof(double))
            {
                var dSpan = MemoryMarshal.Cast<T, double>(span);
                for (int i = 0; i < n; i++)
                    if (!grad.IsNull(i))
                        globalNormSquared += dSpan[i] * dSpan[i];
            }
            else
            {
                for (int i = 0; i < n; i++)
                    if (!grad.IsNull(i))
                    {
                        dynamic value = grad[i];
                        globalNormSquared += (double)(value * value);
                    }
            }
        }

        return Math.Sqrt(globalNormSquared);
    }

    /// <summary>
    /// Validates that a tensor is ready for backward pass.
    /// </summary>
    public static bool CanBackward<T>(ReverseGradTensor<T> tensor) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        return tensor.RequiresGrad && tensor.Length == 1;
    }

    /// <summary>
    /// Gets a detailed description of a tensor for debugging purposes.
    /// </summary>
    public static string DescribeTensor<T>(ReverseGradTensor<T> tensor) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var sb = new StringBuilder();
        sb.AppendLine($"ReverseGradTensor<{typeof(T).Name}>:");
        sb.AppendLine($"  Length: {tensor.Length}");
        sb.AppendLine($"  Requires Grad: {tensor.RequiresGrad}");
        sb.AppendLine($"  Has Gradient: {tensor.Grad != null}");
        sb.AppendLine($"  Is Leaf: {tensor.IsLeaf}");
        sb.AppendLine($"  Has Nulls: {tensor.HasNulls}");

        if (tensor.Grad != null)
        {
            sb.AppendLine($"  Gradient Norm: {GetGradientNorm(tensor):F6}");
        }

        if (!tensor.IsLeaf && tensor.GradFn != null)
        {
            sb.AppendLine($"  Operation: {tensor.GradFn.OperationName}");
        }

        return sb.ToString();
    }

    #endregion
}
