using Nivara.Extensions.AutoDiff.Utilities;
using System.Numerics;

namespace Nivara.Extensions.AutoDiff;

/// <summary>
/// Reverse-mode gradient-enabled tensor that tracks gradients and computation history.
/// Built on top of GradTensor&lt;T&gt; to add reverse-mode automatic differentiation:
/// computation graph tracking, backward pass, and accumulated gradient storage.
/// </summary>
/// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
public sealed class ReverseGradTensor<T> : GradTensor<T> where T : struct, INumber<T>
{
    /// <summary>
    /// Gets or sets the gradient data as a NivaraColumn&lt;T&gt;
    /// </summary>
    public NivaraColumn<T>? Grad { get; set; }

    /// <summary>
    /// Gets a value indicating whether this tensor requires gradient computation
    /// </summary>
    public bool RequiresGrad { get; }

    /// <summary>
    /// Gets the computation graph node for this tensor (null for leaf tensors)
    /// </summary>
    internal OpNode<T>? GradFn { get; set; }

    /// <summary>
    /// Gets a value indicating whether this is a leaf tensor (no computation history)
    /// </summary>
    internal bool IsLeaf => GradFn == null;

    /// <summary>
    /// Initializes a new instance of ReverseGradTensor with the specified data and gradient requirements.
    /// Shape defaults to 1D: [data.Length].
    /// </summary>
    /// <param name="data">The underlying data as a NivaraColumn&lt;T&gt;</param>
    /// <param name="requiresGrad">Whether this tensor should track gradients</param>
    /// <exception cref="ArgumentNullException">Thrown when data is null</exception>
    /// <exception cref="AutoGradException">Thrown when T is not a supported type for automatic differentiation</exception>
    public ReverseGradTensor(NivaraColumn<T> data, bool requiresGrad = false)
        : base(data)
    {
        RequiresGrad = requiresGrad;
        GradFn = null;
    }

    /// <summary>
    /// Initializes a new instance of ReverseGradTensor with shape metadata.
    /// </summary>
    internal ReverseGradTensor(NivaraColumn<T> data, bool requiresGrad, int[] shape)
        : base(data, shape)
    {
        RequiresGrad = requiresGrad;
        GradFn = null;
    }

    /// <summary>
    /// Creates a ReverseGradTensor from a NivaraColumn&lt;T&gt;
    /// </summary>
    /// <param name="column">The column to wrap</param>
    /// <param name="requiresGrad">Whether the tensor should track gradients</param>
    /// <returns>A new ReverseGradTensor instance</returns>
    /// <exception cref="ArgumentNullException">Thrown when column is null</exception>
    public static ReverseGradTensor<T> FromColumn(NivaraColumn<T> column, bool requiresGrad = false)
    {
        if (column == null)
            throw new ArgumentNullException(nameof(column));

        return new ReverseGradTensor<T>(column, requiresGrad);
    }

    /// <summary>
    /// Creates a ReverseGradTensor from a NivaraSeries&lt;T&gt;
    /// </summary>
    /// <param name="series">The series to wrap</param>
    /// <param name="requiresGrad">Whether the tensor should track gradients</param>
    /// <returns>A new ReverseGradTensor instance</returns>
    /// <exception cref="ArgumentNullException">Thrown when series is null</exception>
    public static ReverseGradTensor<T> FromSeries(NivaraSeries<T> series, bool requiresGrad = false)
    {
        if (series == null)
            throw new ArgumentNullException(nameof(series));

        return new ReverseGradTensor<T>(series.Values, requiresGrad);
    }

    /// <summary>
    /// Creates a ReverseGradTensor from an array of values
    /// </summary>
    /// <param name="array">The array of values</param>
    /// <param name="requiresGrad">Whether the tensor should track gradients</param>
    /// <returns>A new ReverseGradTensor instance</returns>
    /// <exception cref="ArgumentNullException">Thrown when array is null</exception>
    public static ReverseGradTensor<T> FromArray(T[] array, bool requiresGrad = false)
    {
        if (array == null)
            throw new ArgumentNullException(nameof(array));

        var column = NivaraColumn<T>.Create(array);
        return new ReverseGradTensor<T>(column, requiresGrad);
    }

    /// <summary>
    /// Initiates backward pass computation from this tensor
    /// </summary>
    /// <param name="gradient">Optional gradient to use as starting point. If null, uses ones for scalar tensors</param>
    /// <exception cref="InvalidOperationException">Thrown when called on non-scalar tensors without gradient, tensors that don't require gradients, or when computation graph has issues</exception>
    /// <exception cref="ArgumentException">Thrown when gradient shape doesn't match tensor shape</exception>
    public void Backward(ReverseGradTensor<T>? gradient = null)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (!RequiresGrad)
        {
            throw new InvalidOperationException(
                "Cannot compute gradients for tensor that doesn't require gradients. " +
                "Set requiresGrad=true when creating the tensor.");
        }

        if (gradient == null && Length != 1)
        {
            throw new InvalidOperationException(
                $"Backward can only be called on scalar tensors (length=1) without providing a gradient. " +
                $"This tensor has length={Length}. " +
                $"For non-scalar tensors, provide a gradient argument with matching shape.");
        }

        if (gradient != null && gradient.Length != Length)
        {
            throw new ArgumentException(
                $"Gradient shape mismatch: expected length {Length}, got {gradient.Length}",
                nameof(gradient));
        }

        NivaraColumn<T> gradientData;
        if (gradient != null)
        {
            gradientData = gradient.Data;
        }
        else
        {
            gradientData = NivaraColumn<T>.Create(new T[] { T.One });
        }

        try
        {
            ComputationGraph.Backward(this, gradientData);
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains("Circular dependency"))
        {
            throw new InvalidOperationException(
                "Cannot perform backward pass: computation graph contains circular dependencies. " +
                "This typically indicates a bug in the operation implementation.", ex);
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains("Could not find output tensor"))
        {
            throw new InvalidOperationException(
                "Cannot perform backward pass: computation graph structure is invalid. " +
                "This typically indicates a bug in the operation implementation.", ex);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to compute gradients during backward pass: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Detaches this tensor from the computation graph, returning a new tensor without gradient tracking
    /// </summary>
    /// <returns>A new ReverseGradTensor with the same data but no gradient tracking</returns>
    public ReverseGradTensor<T> Detach()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return new ReverseGradTensor<T>(Data, requiresGrad: false);
    }

    /// <summary>
    /// Zeros the gradient of this tensor
    /// </summary>
    public void ZeroGrad()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        Grad = null;
    }

    /// <summary>
    /// Converts this ReverseGradTensor to a different numeric type.
    /// </summary>
    /// <typeparam name="TTarget">The target numeric type</typeparam>
    /// <param name="requiresGrad">Optional override for gradient tracking. If null, preserves current setting</param>
    /// <returns>A new ReverseGradTensor with the converted type</returns>
    /// <exception cref="AutoGradException">Thrown when conversion is not supported</exception>
    public ReverseGradTensor<TTarget> ConvertTo<TTarget>(bool? requiresGrad = null)
        where TTarget : struct, INumber<TTarget>
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return TypeConverter.Convert<T, TTarget>(this, requiresGrad);
    }

    /// <summary>
    /// Converts this ReverseGradTensor to float (single-precision).
    /// </summary>
    /// <param name="requiresGrad">Optional override for gradient tracking. If null, preserves current setting</param>
    /// <returns>A new ReverseGradTensor with float type</returns>
    public ReverseGradTensor<float> ToFloat(bool? requiresGrad = null)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return TypeConverter.ToFloat(this, requiresGrad);
    }

    /// <summary>
    /// Converts this ReverseGradTensor to double (double-precision).
    /// </summary>
    /// <param name="requiresGrad">Optional override for gradient tracking. If null, preserves current setting</param>
    /// <returns>A new ReverseGradTensor with double type</returns>
    public ReverseGradTensor<double> ToDouble(bool? requiresGrad = null)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return TypeConverter.ToDouble(this, requiresGrad);
    }

    /// <summary>
    /// Creates a string representation of this ReverseGradTensor
    /// </summary>
    /// <returns>A string representation showing data, gradient status, and computation graph info</returns>
    public override string ToString()
    {
        var shapeStr = string.Join("x", shape);
        var gradInfo = RequiresGrad ? (Grad != null ? "with grad" : "requires grad") : "no grad";
        var graphInfo = IsLeaf ? "leaf" : "non-leaf";
        return $"ReverseGradTensor<{typeof(T).Name}>[{shapeStr}] ({gradInfo}, {graphInfo})";
    }
}
