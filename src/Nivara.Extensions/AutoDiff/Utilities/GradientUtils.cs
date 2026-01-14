using System.Numerics;
using System.Text;
using Nivara;

namespace Nivara.Extensions.AutoDiff.Utilities;

/// <summary>
/// Utility functions for gradient management, memory cleanup, and debugging.
/// Provides helper methods for common gradient operations in automatic differentiation.
/// </summary>
public static class GradientUtils
{
    #region Gradient Management

    /// <summary>
    /// Clears all gradients in the computation graph reachable from the specified tensor.
    /// This is useful for resetting gradients between training iterations.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="tensor">The tensor to start clearing from</param>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null</exception>
    public static void ZeroGrad<T>(GradTensor<T> tensor) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        ComputationGraph.ZeroGrad(tensor);
    }

    /// <summary>
    /// Clears gradients for multiple tensors at once.
    /// This is useful for clearing gradients of all parameters in a model.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="tensors">The collection of tensors to clear gradients for</param>
    /// <exception cref="ArgumentNullException">Thrown when tensors collection is null</exception>
    public static void ZeroGrad<T>(IEnumerable<GradTensor<T>> tensors) where T : struct, INumber<T>
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
    /// This is useful for preventing gradient flow through certain operations.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="tensor">The tensor to detach</param>
    /// <returns>A new GradTensor with the same data but no gradient tracking</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null</exception>
    public static GradTensor<T> Detach<T>(GradTensor<T> tensor) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        return tensor.Detach();
    }

    /// <summary>
    /// Detaches multiple tensors from the computation graph.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="tensors">The collection of tensors to detach</param>
    /// <returns>A collection of detached tensors</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensors collection is null</exception>
    public static IEnumerable<GradTensor<T>> Detach<T>(IEnumerable<GradTensor<T>> tensors) where T : struct, INumber<T>
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
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="tensor">The tensor whose gradient to clip</param>
    /// <param name="maxValue">The maximum absolute value for gradient elements</param>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null</exception>
    /// <exception cref="ArgumentException">Thrown when maxValue is not positive</exception>
    public static void ClipGradValue<T>(GradTensor<T> tensor, T maxValue) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (maxValue <= T.Zero)
            throw new ArgumentException("Max value must be positive", nameof(maxValue));

        if (tensor.Grad == null)
            return;

        var grad = tensor.Grad;
        var clippedData = new T[grad.Length];
        var minValue = -maxValue;

        for (int i = 0; i < grad.Length; i++)
        {
            if (!grad.IsNull(i))
            {
                var value = grad[i];
                if (value > maxValue)
                    clippedData[i] = maxValue;
                else if (value < minValue)
                    clippedData[i] = minValue;
                else
                    clippedData[i] = value;
            }
        }

        tensor.Grad = NivaraColumn<T>.Create(clippedData);
    }

    /// <summary>
    /// Clips gradient norm to prevent exploding gradients.
    /// If the gradient norm exceeds maxNorm, the gradient is scaled down proportionally.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="tensor">The tensor whose gradient to clip</param>
    /// <param name="maxNorm">The maximum norm for the gradient</param>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null</exception>
    /// <exception cref="ArgumentException">Thrown when maxNorm is not positive</exception>
    public static void ClipGradNorm<T>(GradTensor<T> tensor, double maxNorm) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (maxNorm <= 0)
            throw new ArgumentException("Max norm must be positive", nameof(maxNorm));

        if (tensor.Grad == null)
            return;

        var grad = tensor.Grad;
        
        // Calculate gradient norm (L2 norm)
        double normSquared = 0.0;
        for (int i = 0; i < grad.Length; i++)
        {
            if (!grad.IsNull(i))
            {
                dynamic value = grad[i];
                normSquared += (double)(value * value);
            }
        }

        var norm = Math.Sqrt(normSquared);

        // Only clip if norm exceeds maxNorm
        if (norm > maxNorm)
        {
            var scale = maxNorm / norm;
            var clippedData = new T[grad.Length];

            // Convert scale to the appropriate type
            if (typeof(T) == typeof(float))
            {
                var scaleFloat = (float)scale;
                for (int i = 0; i < grad.Length; i++)
                {
                    if (!grad.IsNull(i))
                    {
                        var value = (float)(object)grad[i]!;
                        clippedData[i] = (T)(object)(value * scaleFloat);
                    }
                }
            }
            else if (typeof(T) == typeof(double))
            {
                for (int i = 0; i < grad.Length; i++)
                {
                    if (!grad.IsNull(i))
                    {
                        var value = (double)(object)grad[i]!;
                        clippedData[i] = (T)(object)(value * scale);
                    }
                }
            }
            else
            {
                // Fallback for other types
                for (int i = 0; i < grad.Length; i++)
                {
                    if (!grad.IsNull(i))
                    {
                        dynamic value = grad[i];
                        clippedData[i] = (T)(object)(value * scale)!;
                    }
                }
            }

            tensor.Grad = NivaraColumn<T>.Create(clippedData);
        }
    }

    /// <summary>
    /// Clips gradients for multiple tensors by their global norm.
    /// This is useful for clipping all parameters in a model together.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="tensors">The collection of tensors whose gradients to clip</param>
    /// <param name="maxNorm">The maximum global norm for all gradients</param>
    /// <exception cref="ArgumentNullException">Thrown when tensors collection is null</exception>
    /// <exception cref="ArgumentException">Thrown when maxNorm is not positive</exception>
    public static void ClipGradNorm<T>(IEnumerable<GradTensor<T>> tensors, double maxNorm) where T : struct, INumber<T>
    {
        if (tensors == null)
            throw new ArgumentNullException(nameof(tensors));

        if (maxNorm <= 0)
            throw new ArgumentException("Max norm must be positive", nameof(maxNorm));

        var tensorList = tensors.Where(t => t != null && t.Grad != null).ToList();
        if (tensorList.Count == 0)
            return;

        // Calculate global norm across all tensors
        double globalNormSquared = 0.0;
        foreach (var tensor in tensorList)
        {
            var grad = tensor.Grad!;
            for (int i = 0; i < grad.Length; i++)
            {
                if (!grad.IsNull(i))
                {
                    dynamic value = grad[i];
                    globalNormSquared += (double)(value * value);
                }
            }
        }

        var globalNorm = Math.Sqrt(globalNormSquared);

        // Only clip if global norm exceeds maxNorm
        if (globalNorm > maxNorm)
        {
            var scale = maxNorm / globalNorm;

            // Scale all gradients proportionally
            foreach (var tensor in tensorList)
            {
                var grad = tensor.Grad!;
                var clippedData = new T[grad.Length];

                // Convert scale to the appropriate type
                if (typeof(T) == typeof(float))
                {
                    var scaleFloat = (float)scale;
                    for (int i = 0; i < grad.Length; i++)
                    {
                        if (!grad.IsNull(i))
                        {
                            var value = (float)(object)grad[i]!;
                            clippedData[i] = (T)(object)(value * scaleFloat);
                        }
                    }
                }
                else if (typeof(T) == typeof(double))
                {
                    for (int i = 0; i < grad.Length; i++)
                    {
                        if (!grad.IsNull(i))
                        {
                            var value = (double)(object)grad[i]!;
                            clippedData[i] = (T)(object)(value * scale);
                        }
                    }
                }
                else
                {
                    // Fallback for other types
                    for (int i = 0; i < grad.Length; i++)
                    {
                        if (!grad.IsNull(i))
                        {
                            dynamic value = grad[i];
                            clippedData[i] = (T)(object)(value * scale)!;
                        }
                    }
                }

                tensor.Grad = NivaraColumn<T>.Create(clippedData);
            }
        }
    }

    #endregion

    #region Constant Tensor Creation

    /// <summary>
    /// Creates a constant tensor that doesn't require gradients.
    /// This is useful for creating tensors that should not participate in gradient computation.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="data">The data for the constant tensor</param>
    /// <returns>A new GradTensor with requiresGrad=false</returns>
    /// <exception cref="ArgumentNullException">Thrown when data is null</exception>
    public static GradTensor<T> Constant<T>(T[] data) where T : struct, INumber<T>
    {
        if (data == null)
            throw new ArgumentNullException(nameof(data));

        return GradTensor<T>.FromArray(data, requiresGrad: false);
    }

    /// <summary>
    /// Creates a constant tensor from a NivaraColumn that doesn't require gradients.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="column">The column data for the constant tensor</param>
    /// <returns>A new GradTensor with requiresGrad=false</returns>
    /// <exception cref="ArgumentNullException">Thrown when column is null</exception>
    public static GradTensor<T> Constant<T>(NivaraColumn<T> column) where T : struct, INumber<T>
    {
        if (column == null)
            throw new ArgumentNullException(nameof(column));

        return GradTensor<T>.FromColumn(column, requiresGrad: false);
    }

    /// <summary>
    /// Creates a constant tensor filled with zeros.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="length">The length of the tensor</param>
    /// <returns>A new GradTensor filled with zeros and requiresGrad=false</returns>
    /// <exception cref="ArgumentException">Thrown when length is not positive</exception>
    public static GradTensor<T> Zeros<T>(int length) where T : struct, INumber<T>
    {
        if (length <= 0)
            throw new ArgumentException("Length must be positive", nameof(length));

        var data = new T[length];
        Array.Fill(data, T.Zero);
        return GradTensor<T>.FromArray(data, requiresGrad: false);
    }

    /// <summary>
    /// Creates a constant tensor filled with ones.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="length">The length of the tensor</param>
    /// <returns>A new GradTensor filled with ones and requiresGrad=false</returns>
    /// <exception cref="ArgumentException">Thrown when length is not positive</exception>
    public static GradTensor<T> Ones<T>(int length) where T : struct, INumber<T>
    {
        if (length <= 0)
            throw new ArgumentException("Length must be positive", nameof(length));

        var data = new T[length];
        Array.Fill(data, T.One);
        return GradTensor<T>.FromArray(data, requiresGrad: false);
    }

    /// <summary>
    /// Creates a constant tensor filled with a specific value.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="length">The length of the tensor</param>
    /// <param name="value">The value to fill the tensor with</param>
    /// <returns>A new GradTensor filled with the specified value and requiresGrad=false</returns>
    /// <exception cref="ArgumentException">Thrown when length is not positive</exception>
    public static GradTensor<T> Full<T>(int length, T value) where T : struct, INumber<T>
    {
        if (length <= 0)
            throw new ArgumentException("Length must be positive", nameof(length));

        var data = new T[length];
        Array.Fill(data, value);
        return GradTensor<T>.FromArray(data, requiresGrad: false);
    }

    #endregion

    #region Computation Graph Inspection

    /// <summary>
    /// Gets diagnostic information about the computation graph rooted at the specified tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="tensor">The root tensor to analyze</param>
    /// <returns>A dictionary containing graph statistics</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null</exception>
    public static Dictionary<string, object> GetGraphInfo<T>(GradTensor<T> tensor) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        return ComputationGraph.GetGraphInfo(tensor);
    }

    /// <summary>
    /// Prints a human-readable summary of the computation graph.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="tensor">The root tensor to analyze</param>
    /// <returns>A formatted string describing the computation graph</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null</exception>
    public static string PrintGraphSummary<T>(GradTensor<T> tensor) where T : struct, INumber<T>
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
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="tensor">The tensor to check</param>
    /// <returns>True if the tensor has gradients, false otherwise</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null</exception>
    public static bool HasGradient<T>(GradTensor<T> tensor) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        return tensor.Grad != null;
    }

    /// <summary>
    /// Gets the gradient norm (L2 norm) for a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="tensor">The tensor to compute gradient norm for</param>
    /// <returns>The L2 norm of the gradient, or 0 if no gradient exists</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null</exception>
    public static double GetGradientNorm<T>(GradTensor<T> tensor) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (tensor.Grad == null)
            return 0.0;

        var grad = tensor.Grad;
        double normSquared = 0.0;

        for (int i = 0; i < grad.Length; i++)
        {
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
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="tensors">The collection of tensors to compute global gradient norm for</param>
    /// <returns>The global L2 norm of all gradients</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensors collection is null</exception>
    public static double GetGlobalGradientNorm<T>(IEnumerable<GradTensor<T>> tensors) where T : struct, INumber<T>
    {
        if (tensors == null)
            throw new ArgumentNullException(nameof(tensors));

        double globalNormSquared = 0.0;

        foreach (var tensor in tensors.Where(t => t != null && t.Grad != null))
        {
            var grad = tensor.Grad!;
            for (int i = 0; i < grad.Length; i++)
            {
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
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="tensor">The tensor to validate</param>
    /// <returns>True if the tensor can be used for backward pass, false otherwise</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null</exception>
    public static bool CanBackward<T>(GradTensor<T> tensor) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        return tensor.RequiresGrad && tensor.Length == 1;
    }

    /// <summary>
    /// Gets a detailed description of a tensor for debugging purposes.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="tensor">The tensor to describe</param>
    /// <returns>A formatted string with tensor details</returns>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null</exception>
    public static string DescribeTensor<T>(GradTensor<T> tensor) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var sb = new StringBuilder();
        sb.AppendLine($"GradTensor<{typeof(T).Name}>:");
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