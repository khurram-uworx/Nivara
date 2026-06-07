using Nivara.Extensions.AutoDiff.Utilities;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.Extensions.AutoDiff;

/// <summary>
/// Gradient-enabled tensor wrapper that tracks gradients and computation history for automatic differentiation.
/// Built on top of NivaraColumn&lt;T&gt; to leverage existing vectorization, null handling, and storage backends.
/// </summary>
/// <typeparam name="T">The numeric type of the tensor that implements INumber&lt;T&gt;</typeparam>
public sealed class GradTensor<T> : IDisposable where T : struct, INumber<T>
{
    private bool disposed;

    /// <summary>
    /// Gets the underlying data as a NivaraColumn&lt;T&gt;
    /// </summary>
    public NivaraColumn<T> Data { get; }

    /// <summary>
    /// Gets or sets the gradient data as a NivaraColumn&lt;T&gt;
    /// </summary>
    public NivaraColumn<T>? Grad { get; set; }

    /// <summary>
    /// Gets a value indicating whether this tensor requires gradient computation
    /// </summary>
    public bool RequiresGrad { get; }

    /// <summary>
    /// Gets the number of elements in the tensor
    /// </summary>
    public int Length => Data.Length;

    /// <summary>
    /// Gets a value indicating whether this tensor contains any null values
    /// </summary>
    public bool HasNulls => Data.HasNulls;

    /// <summary>
    /// Gets the computation graph node for this tensor (null for leaf tensors)
    /// </summary>
    internal OpNode? GradFn { get; set; }

    /// <summary>
    /// Gets a value indicating whether this is a leaf tensor (no computation history)
    /// </summary>
    internal bool IsLeaf => GradFn == null;

    /// <summary>
    /// Initializes a new instance of GradTensor with the specified data and gradient requirements
    /// </summary>
    /// <param name="data">The underlying data as a NivaraColumn&lt;T&gt;</param>
    /// <param name="requiresGrad">Whether this tensor should track gradients</param>
    /// <exception cref="ArgumentNullException">Thrown when data is null</exception>
    /// <exception cref="AutoGradException">Thrown when T is not a supported type for automatic differentiation</exception>
    public GradTensor(NivaraColumn<T> data, bool requiresGrad = false)
    {
        // Validate that the type is supported for automatic differentiation
        TypeValidator.ValidateNumericType<T>();

        Data = data ?? throw new ArgumentNullException(nameof(data));
        RequiresGrad = requiresGrad;
        GradFn = null; // Leaf tensor by default
    }

    /// <summary>
    /// Creates a GradTensor from a NivaraColumn&lt;T&gt;
    /// </summary>
    /// <param name="column">The column to wrap</param>
    /// <param name="requiresGrad">Whether the tensor should track gradients</param>
    /// <returns>A new GradTensor instance</returns>
    /// <exception cref="ArgumentNullException">Thrown when column is null</exception>
    public static GradTensor<T> FromColumn(NivaraColumn<T> column, bool requiresGrad = false)
    {
        if (column == null)
            throw new ArgumentNullException(nameof(column));

        return new GradTensor<T>(column, requiresGrad);
    }

    /// <summary>
    /// Creates a GradTensor from a NivaraSeries&lt;T&gt;
    /// </summary>
    /// <param name="series">The series to wrap</param>
    /// <param name="requiresGrad">Whether the tensor should track gradients</param>
    /// <returns>A new GradTensor instance</returns>
    /// <exception cref="ArgumentNullException">Thrown when series is null</exception>
    public static GradTensor<T> FromSeries(NivaraSeries<T> series, bool requiresGrad = false)
    {
        if (series == null)
            throw new ArgumentNullException(nameof(series));

        return new GradTensor<T>(series.Values, requiresGrad);
    }

    /// <summary>
    /// Creates a GradTensor from an array of values
    /// </summary>
    /// <param name="array">The array of values</param>
    /// <param name="requiresGrad">Whether the tensor should track gradients</param>
    /// <returns>A new GradTensor instance</returns>
    /// <exception cref="ArgumentNullException">Thrown when array is null</exception>
    public static GradTensor<T> FromArray(T[] array, bool requiresGrad = false)
    {
        if (array == null)
            throw new ArgumentNullException(nameof(array));

        var column = NivaraColumn<T>.Create(array);
        return new GradTensor<T>(column, requiresGrad);
    }

    /// <summary>
    /// Converts this GradTensor back to a NivaraColumn&lt;T&gt;
    /// </summary>
    /// <returns>The underlying NivaraColumn&lt;T&gt;</returns>
    public NivaraColumn<T> ToColumn()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return Data;
    }

    /// <summary>
    /// Converts this GradTensor to a NivaraSeries&lt;T&gt;
    /// </summary>
    /// <returns>A new NivaraSeries&lt;T&gt; wrapping the data</returns>
    public NivaraSeries<T> ToSeries()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return new NivaraSeries<T>(Data);
    }

    /// <summary>
    /// Gets a tensor view of the data for operations requiring tensor semantics
    /// </summary>
    /// <returns>A Tensor&lt;T&gt; view of the data</returns>
    /// <exception cref="InvalidOperationException">Thrown when the data contains null values</exception>
    public Tensor<T> AsTensor()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (HasNulls)
        {
            throw new InvalidOperationException("Cannot create tensor view from data with null values. Use ToColumn() or ToSeries() for null-aware operations.");
        }

        // Convert to series first, then to tensor
        var series = ToSeries();
        return series.ToTensor();
    }

    /// <summary>
    /// Initiates backward pass computation from this tensor
    /// </summary>
    /// <param name="gradient">Optional gradient to use as starting point. If null, uses ones for scalar tensors</param>
    /// <exception cref="InvalidOperationException">Thrown when called on non-scalar tensors without gradient, tensors that don't require gradients, or when computation graph has issues</exception>
    /// <exception cref="ArgumentException">Thrown when gradient shape doesn't match tensor shape</exception>
    public void Backward(GradTensor<T>? gradient = null)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        // Validate that this tensor requires gradients
        if (!RequiresGrad)
        {
            throw new InvalidOperationException(
                "Cannot compute gradients for tensor that doesn't require gradients. " +
                "Set requiresGrad=true when creating the tensor.");
        }

        // Validate scalar tensor requirement when no gradient is provided
        if (gradient == null && Length != 1)
        {
            throw new InvalidOperationException(
                $"Backward can only be called on scalar tensors (length=1) without providing a gradient. " +
                $"This tensor has length={Length}. " +
                $"For non-scalar tensors, provide a gradient argument with matching shape.");
        }

        // Validate gradient shape if provided
        if (gradient != null && gradient.Length != Length)
        {
            throw new ArgumentException(
                $"Gradient shape mismatch: expected length {Length}, got {gradient.Length}",
                nameof(gradient));
        }

        // Prepare gradient data
        NivaraColumn<T> gradientData;
        if (gradient != null)
        {
            gradientData = gradient.Data;
        }
        else
        {
            // For scalar tensors, use ones as default gradient
            gradientData = NivaraColumn<T>.Create(new T[] { T.One });
        }

        try
        {
            // Use ComputationGraph to perform backward pass
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
    /// <returns>A new GradTensor with the same data but no gradient tracking</returns>
    public GradTensor<T> Detach()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return new GradTensor<T>(Data, requiresGrad: false);
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
    /// Gets the element at the specified index
    /// </summary>
    /// <param name="index">The zero-based index</param>
    /// <returns>The element at the specified index</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown when index is out of bounds</exception>
    public T this[int index]
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return Data[index];
        }
    }

    /// <summary>
    /// Determines whether the element at the specified index is null
    /// </summary>
    /// <param name="index">The zero-based index to check</param>
    /// <returns>true if the element is null; otherwise, false</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown when index is out of bounds</exception>
    public bool IsNull(int index)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return Data.IsNull(index);
    }

    /// <summary>
    /// Creates a string representation of this GradTensor
    /// </summary>
    /// <returns>A string representation showing data, gradient status, and computation graph info</returns>
    public override string ToString()
    {
        if (disposed)
            return "GradTensor<T> (disposed)";

        var gradInfo = RequiresGrad ? (Grad != null ? "with grad" : "requires grad") : "no grad";
        var graphInfo = IsLeaf ? "leaf" : "non-leaf";
        return $"GradTensor<{typeof(T).Name}>[{Length}] ({gradInfo}, {graphInfo})";
    }

    /// <summary>
    /// Converts this GradTensor to a different numeric type.
    /// </summary>
    /// <typeparam name="TTarget">The target numeric type</typeparam>
    /// <param name="requiresGrad">Optional override for gradient tracking. If null, preserves current setting</param>
    /// <returns>A new GradTensor with the converted type</returns>
    /// <exception cref="AutoGradException">Thrown when conversion is not supported</exception>
    public GradTensor<TTarget> ConvertTo<TTarget>(bool? requiresGrad = null)
        where TTarget : struct, INumber<TTarget>
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return TypeConverter.Convert<T, TTarget>(this, requiresGrad);
    }

    /// <summary>
    /// Converts this GradTensor to float (single-precision).
    /// </summary>
    /// <param name="requiresGrad">Optional override for gradient tracking. If null, preserves current setting</param>
    /// <returns>A new GradTensor with float type</returns>
    public GradTensor<float> ToFloat(bool? requiresGrad = null)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return TypeConverter.ToFloat(this, requiresGrad);
    }

    /// <summary>
    /// Converts this GradTensor to double (double-precision).
    /// </summary>
    /// <param name="requiresGrad">Optional override for gradient tracking. If null, preserves current setting</param>
    /// <returns>A new GradTensor with double type</returns>
    public GradTensor<double> ToDouble(bool? requiresGrad = null)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return TypeConverter.ToDouble(this, requiresGrad);
    }

    /// <summary>
    /// Releases all resources used by this GradTensor
    /// </summary>
    public void Dispose()
    {
        if (!disposed)
        {
            Data?.Dispose();
            Grad?.Dispose();
            disposed = true;
        }
    }
}