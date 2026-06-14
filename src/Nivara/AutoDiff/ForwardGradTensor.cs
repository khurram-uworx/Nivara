using System.Numerics;

namespace Nivara.AutoDiff;

/// <summary>
/// Forward-mode gradient-enabled tensor that carries a tangent (directional derivative)
/// alongside the primal value. Built on top of GradTensor&lt;T&gt; to add forward-mode
/// automatic differentiation: tangent propagation through operations without a
/// computation graph or backward pass.
/// </summary>
/// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
public sealed class ForwardGradTensor<T> : GradTensor<T> where T : struct, INumber<T>
{
    /// <summary>
    /// Gets the tangent (directional derivative) data as a NivaraColumn&lt;T&gt;.
    /// Null when no tangent is being tracked for this tensor.
    /// </summary>
    public NivaraColumn<T>? Tangent { get; }

    /// <summary>
    /// Gets a value indicating whether this tensor requires tangent computation
    /// </summary>
    public bool RequiresTangent { get; }

    /// <summary>
    /// Initializes a new instance of ForwardGradTensor with the specified data and optional tangent.
    /// Shape defaults to 1D: [data.Length].
    /// </summary>
    /// <param name="data">The underlying primal data as a NivaraColumn&lt;T&gt;</param>
    /// <param name="tangent">The tangent (directional derivative) data, or null if not tracking</param>
    /// <exception cref="ArgumentNullException">Thrown when data is null</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when tangent length doesn't match data length</exception>
    public ForwardGradTensor(NivaraColumn<T> data, NivaraColumn<T>? tangent = null)
        : base(data)
    {
        if (tangent != null && tangent.Length != data.Length)
            throw new ArgumentOutOfRangeException(nameof(tangent),
                $"Tangent length ({tangent.Length}) must match data length ({data.Length})");

        RequiresTangent = tangent != null;
        Tangent = tangent;
    }

    /// <summary>
    /// Initializes a new instance of ForwardGradTensor with shape metadata.
    /// </summary>
    internal ForwardGradTensor(NivaraColumn<T> data, NivaraColumn<T>? tangent, int[] shape)
        : base(data, shape)
    {
        if (tangent != null && tangent.Length != data.Length)
            throw new ArgumentOutOfRangeException(nameof(tangent),
                $"Tangent length ({tangent.Length}) must match data length ({data.Length})");

        RequiresTangent = tangent != null;
        Tangent = tangent;
    }

    /// <summary>
    /// Creates a ForwardGradTensor from a NivaraColumn&lt;T&gt;
    /// </summary>
    /// <param name="column">The column to wrap as primal data</param>
    /// <param name="tangent">Optional tangent column, or null if not tracking</param>
    /// <returns>A new ForwardGradTensor instance</returns>
    /// <exception cref="ArgumentNullException">Thrown when column is null</exception>
    public static ForwardGradTensor<T> FromColumn(NivaraColumn<T> column, NivaraColumn<T>? tangent = null)
    {
        if (column == null)
            throw new ArgumentNullException(nameof(column));

        return new ForwardGradTensor<T>(column, tangent);
    }

    /// <summary>
    /// Creates a ForwardGradTensor from a NivaraSeries&lt;T&gt;
    /// </summary>
    /// <param name="series">The series to wrap as primal data</param>
    /// <param name="tangent">Optional tangent series, or null if not tracking</param>
    /// <returns>A new ForwardGradTensor instance</returns>
    /// <exception cref="ArgumentNullException">Thrown when series is null</exception>
    public static ForwardGradTensor<T> FromSeries(NivaraSeries<T> series, NivaraSeries<T>? tangent = null)
    {
        if (series == null)
            throw new ArgumentNullException(nameof(series));

        var tangentCol = tangent?.Values;
        return new ForwardGradTensor<T>(series.Values, tangentCol);
    }

    /// <summary>
    /// Creates a ForwardGradTensor from an array of values
    /// </summary>
    /// <param name="array">The array of primal values</param>
    /// <param name="tangent">Optional array of tangent values, or null if not tracking</param>
    /// <returns>A new ForwardGradTensor instance</returns>
    /// <exception cref="ArgumentNullException">Thrown when array is null</exception>
    public static ForwardGradTensor<T> FromArray(T[] array, T[]? tangent = null)
    {
        if (array == null)
            throw new ArgumentNullException(nameof(array));

        var dataCol = NivaraColumn<T>.Create(array);
        NivaraColumn<T>? tangentCol = tangent != null ? NivaraColumn<T>.Create(tangent) : null;
        return new ForwardGradTensor<T>(dataCol, tangentCol);
    }

    /// <summary>
    /// Creates a ForwardGradTensor from row-major matrix data with explicit dimensions.
    /// The tensor's shape is set to [rows, cols] so callers can use shape-aware MatMul/Transpose
    /// without manually calling Reshape.
    /// </summary>
    /// <param name="rowMajorData">Flat array containing matrix data in row-major order</param>
    /// <param name="rows">Number of rows</param>
    /// <param name="cols">Number of columns</param>
    /// <param name="tangent">Optional flat array of tangent values (same layout), or null if not tracking</param>
    /// <returns>A new ForwardGradTensor instance with shape [rows, cols]</returns>
    /// <exception cref="ArgumentNullException">Thrown when rowMajorData is null</exception>
    /// <exception cref="ArgumentException">Thrown when data length doesn't match rows * cols</exception>
    public static ForwardGradTensor<T> FromMatrix(T[] rowMajorData, int rows, int cols, T[]? tangent = null)
    {
        if (rowMajorData == null)
            throw new ArgumentNullException(nameof(rowMajorData));

        if (rowMajorData.Length != rows * cols)
            throw new ArgumentException(
                $"Data length ({rowMajorData.Length}) must equal rows * cols ({rows} * {cols} = {rows * cols})");

        var column = NivaraColumn<T>.Create(rowMajorData);
        NivaraColumn<T>? tangentCol = tangent != null ? NivaraColumn<T>.Create(tangent) : null;
        var tensor = new ForwardGradTensor<T>(column, tangentCol);
        tensor.Reshape(rows, cols);
        return tensor;
    }

    /// <summary>
    /// Creates a string representation of this ForwardGradTensor
    /// </summary>
    /// <returns>A string representation showing data, tangent status, and shape info</returns>
    public override string ToString()
    {
        var shapeStr = string.Join("x", shape);
        var tanInfo = RequiresTangent ? "with tangent" : "no tangent";
        return $"ForwardGradTensor<{typeof(T).Name}>[{shapeStr}] ({tanInfo})";
    }
}
