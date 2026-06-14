using Nivara;
using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Text;
using System.Threading;

namespace Nivara.AutoDiff.Utilities;

/// <summary>
/// Utility functions for gradient management, memory cleanup, and debugging.
/// Provides helper methods for common gradient operations in reverse-mode automatic differentiation.
/// </summary>
public static class GradientUtils
{
    private static readonly AsyncLocal<int> GradDepth = new();

    /// <summary>
    /// Gets whether reverse-mode operation tracking is currently enabled.
    /// Inference is the default; enter <see cref="Grad"/> when building a training graph.
    /// </summary>
    public static bool IsGradEnabled => GradDepth.Value > 0;

    /// <summary>
    /// Enables reverse-mode gradient tracking for the current async control flow.
    /// Dispose the returned scope to restore the previous inference-default behavior.
    /// </summary>
    public static IDisposable Grad()
    {
        GradDepth.Value++;
        return new GradScope();
    }

    internal static bool ShouldTrackGrad<T>(params ReverseGradTensor<T>[] tensors)
        where T : struct, INumber<T>
    {
        if (!IsGradEnabled)
            return false;

        foreach (var tensor in tensors)
        {
            if (tensor?.RequiresGrad == true)
                return true;
        }

        return false;
    }

    private sealed class GradScope : IDisposable
    {
        private bool disposed;

        public void Dispose()
        {
            if (disposed)
                return;

            if (GradDepth.Value > 0)
                GradDepth.Value--;

            disposed = true;
        }
    }

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
        var minValue = -maxValue;

        if (!grad.HasNulls)
        {
            grad.TryGetSpan(out var span);
            var clipped = new T[n];
            TensorPrimitives.Clamp(span, minValue, maxValue, clipped);
            tensor.Grad = NivaraColumn<T>.Create(clipped);
            return;
        }

        var buf = ArrayPool<T>.Shared.Rent(n);
        var nullMask = ArrayPool<bool>.Shared.Rent(n);

        try
        {
            grad.CopyTo(buf.AsSpan(0, n), T.Zero);
            grad.TryGetNullMask(out var mask);
            mask.CopyTo(nullMask.AsSpan(0, n));

            TensorPrimitives.Clamp(buf.AsSpan(0, n), minValue, maxValue, buf.AsSpan(0, n));

            tensor.Grad = NivaraColumn<T>.CreateFromSpans(buf.AsSpan(0, n), nullMask.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
            ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
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

        double normSquared;
        if (!grad.HasNulls)
        {
            grad.TryGetSpan(out var span);
            normSquared = double.CreateChecked(TensorPrimitives.SumOfSquares(span));
        }
        else
        {
            var buf = ArrayPool<T>.Shared.Rent(n);
            try
            {
                grad.CopyTo(buf.AsSpan(0, n), T.Zero);
                normSquared = double.CreateChecked(TensorPrimitives.SumOfSquares(buf.AsSpan(0, n)));
            }
            finally
            {
                ArrayPool<T>.Shared.Return(buf, clearArray: true);
            }
        }

        var norm = Math.Sqrt(normSquared);

        if (norm > maxNorm)
        {
            var scale = T.CreateChecked(maxNorm / norm);

            if (!grad.HasNulls)
            {
                grad.TryGetSpan(out var span);
                var clipped = new T[n];
                TensorPrimitives.Multiply(span, scale, clipped);
                tensor.Grad = NivaraColumn<T>.Create(clipped);
                return;
            }

            var buf = ArrayPool<T>.Shared.Rent(n);
            var nullMask = ArrayPool<bool>.Shared.Rent(n);

            try
            {
                grad.CopyTo(buf.AsSpan(0, n), T.Zero);
                grad.TryGetNullMask(out var mask);
                mask.CopyTo(nullMask.AsSpan(0, n));

                TensorPrimitives.Multiply(buf.AsSpan(0, n), scale, buf.AsSpan(0, n));

                tensor.Grad = NivaraColumn<T>.CreateFromSpans(buf.AsSpan(0, n), nullMask.AsSpan(0, n));
            }
            finally
            {
                ArrayPool<T>.Shared.Return(buf, clearArray: true);
                ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
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
            if (!grad.HasNulls)
            {
                grad.TryGetSpan(out var span);
                globalNormSquared += double.CreateChecked(TensorPrimitives.SumOfSquares(span));
            }
            else
            {
                int n = grad.Length;
                var buf = ArrayPool<T>.Shared.Rent(n);
                try
                {
                    grad.CopyTo(buf.AsSpan(0, n), T.Zero);
                    globalNormSquared += double.CreateChecked(TensorPrimitives.SumOfSquares(buf.AsSpan(0, n)));
                }
                finally
                {
                    ArrayPool<T>.Shared.Return(buf, clearArray: true);
                }
            }
        }

        var globalNorm = Math.Sqrt(globalNormSquared);

        if (globalNorm > maxNorm)
        {
            var scale = T.CreateChecked(maxNorm / globalNorm);

            foreach (var tensor in tensorList)
            {
                var grad = tensor.Grad!;
                int n = grad.Length;

                if (!grad.HasNulls)
                {
                    grad.TryGetSpan(out var span);
                    var clipped = new T[n];
                    TensorPrimitives.Multiply(span, scale, clipped);
                    tensor.Grad = NivaraColumn<T>.Create(clipped);
                }
                else
                {
                    var buf = ArrayPool<T>.Shared.Rent(n);
                    var nullMask = ArrayPool<bool>.Shared.Rent(n);

                    try
                    {
                        grad.CopyTo(buf.AsSpan(0, n), T.Zero);
                        grad.TryGetNullMask(out var mask);
                        mask.CopyTo(nullMask.AsSpan(0, n));

                        TensorPrimitives.Multiply(buf.AsSpan(0, n), scale, buf.AsSpan(0, n));

                        tensor.Grad = NivaraColumn<T>.CreateFromSpans(buf.AsSpan(0, n), nullMask.AsSpan(0, n));
                    }
                    finally
                    {
                        ArrayPool<T>.Shared.Return(buf, clearArray: true);
                        ArrayPool<bool>.Shared.Return(nullMask, clearArray: true);
                    }
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

        if (!grad.HasNulls)
        {
            grad.TryGetSpan(out var span);
            return Math.Sqrt(double.CreateChecked(TensorPrimitives.SumOfSquares(span)));
        }

        var buf = ArrayPool<T>.Shared.Rent(n);
        try
        {
            grad.CopyTo(buf.AsSpan(0, n), T.Zero);
            return Math.Sqrt(double.CreateChecked(TensorPrimitives.SumOfSquares(buf.AsSpan(0, n))));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buf, clearArray: true);
        }
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

            if (!grad.HasNulls)
            {
                grad.TryGetSpan(out var span);
                globalNormSquared += double.CreateChecked(TensorPrimitives.SumOfSquares(span));
            }
            else
            {
                int n = grad.Length;
                var buf = ArrayPool<T>.Shared.Rent(n);
                try
                {
                    grad.CopyTo(buf.AsSpan(0, n), T.Zero);
                    globalNormSquared += double.CreateChecked(TensorPrimitives.SumOfSquares(buf.AsSpan(0, n)));
                }
                finally
                {
                    ArrayPool<T>.Shared.Return(buf, clearArray: true);
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
