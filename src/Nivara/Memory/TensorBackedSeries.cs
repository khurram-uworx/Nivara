using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.Memory;

/// <summary>
/// Internal storage class that uses Tensor&lt;T&gt; as the backing store for numeric types.
/// Provides zero-copy TensorSpan&lt;T&gt; access when no null values are present.
/// </summary>
/// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
internal sealed class TensorBackedSeries<T> : IDisposable where T : struct, INumber<T>
{
    Tensor<T>? tensor;
    readonly bool[]? validityMask;
    readonly bool ownsTensor;
    bool disposed;

    /// <summary>
    /// Gets the number of elements in this series.
    /// </summary>
    public int Length { get; }

    /// <summary>
    /// Gets whether this series uses tensor storage.
    /// </summary>
    public bool IsTensorBacked => tensor != null;

    /// <summary>
    /// Gets a read-only span of the values in this series.
    /// </summary>
    public ReadOnlySpan<T> Values
    {
        get
        {
            if (disposed)
                throw new ObjectDisposedException(nameof(TensorBackedSeries<T>));

            if (tensor == null)
                return ReadOnlySpan<T>.Empty;

            // For 1D TensorSpan, extract values manually
            var tensorSpan = tensor.AsTensorSpan();
            var values = new T[Length];
            for (int i = 0; i < Length; i++)
            {
                values[i] = tensorSpan[i];
            }
            return values.AsSpan();
        }
    }

    /// <summary>
    /// Gets a read-only span of the validity mask indicating which values are non-null.
    /// </summary>
    public ReadOnlySpan<bool> ValidityMask
    {
        get
        {
            if (disposed)
                throw new ObjectDisposedException(nameof(TensorBackedSeries<T>));

            if (validityMask == null)
                return ReadOnlySpan<bool>.Empty;

            return validityMask.AsSpan(0, Length);
        }
    }

    /// <summary>
    /// Gets the values as an array for reflection-based access.
    /// </summary>
    /// <returns>An array containing all values in this series.</returns>
    public T[] GetValuesArray()
    {
        if (tensor == null)
            return Array.Empty<T>();

        var result = new T[Length];
        var tensorSpan = tensor.AsTensorSpan();
        for (int i = 0; i < Length; i++)
        {
            result[i] = tensorSpan[i];
        }
        return result;
    }

    /// <summary>
    /// Gets the validity mask as an array for reflection-based access.
    /// </summary>
    /// <returns>An array containing the validity mask.</returns>
    public bool[] GetValidityMaskArray()
    {
        if (validityMask == null)
        {
            var result = new bool[Length];
            Array.Fill(result, true);
            return result;
        }

        var result2 = new bool[Length];
        Array.Copy(validityMask, result2, Length);
        return result2;
    }

    /// <summary>
    /// Gets a TensorSpan&lt;T&gt; for zero-copy operations. Only available when no null values are present.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when series contains null values.</exception>
    public TensorSpan<T> TensorSpan
    {
        get
        {
            if (disposed)
                throw new ObjectDisposedException(nameof(TensorBackedSeries<T>));

            if (tensor == null)
                throw new InvalidOperationException("Tensor is not available.");

            // Check if all values are valid
            if (validityMask != null)
            {
                for (int i = 0; i < Length; i++)
                {
                    if (!validityMask[i])
                    {
                        throw new InvalidOperationException("Cannot create TensorSpan from series with null values.");
                    }
                }
            }

            return tensor.AsTensorSpan();
        }
    }

    /// <summary>
    /// Initializes a new instance of the TensorBackedSeries class from a tensor.
    /// </summary>
    /// <param name="tensor">The tensor to use as backing storage.</param>
    /// <param name="validityMask">Optional validity mask. If null, all values are considered valid.</param>
    /// <param name="ownsTensor">Whether this instance owns the tensor and should dispose it.</param>
    public TensorBackedSeries(Tensor<T> tensor, bool[]? validityMask = null, bool ownsTensor = true)
    {
        ArgumentNullException.ThrowIfNull(tensor);

        if (tensor.Rank != 1)
        {
            throw new ArgumentException($"Only 1D tensors are supported. Tensor has {tensor.Rank} dimensions.");
        }

        var length = (int)tensor.Lengths[0];
        if (validityMask != null && validityMask.Length != length)
        {
            throw new ArgumentException("Validity mask length must match tensor length.", nameof(validityMask));
        }

        this.tensor = tensor;
        this.validityMask = validityMask;
        this.ownsTensor = ownsTensor;
        Length = length;
    }

    /// <summary>
    /// Initializes a new instance of the TensorBackedSeries class from data arrays.
    /// This constructor is designed for reflection-based instantiation.
    /// </summary>
    /// <param name="data">The data array to store in the tensor.</param>
    /// <param name="validityMask">Optional validity mask array. If null, all values are considered valid.</param>
    public TensorBackedSeries(T[] data, bool[]? validityMask = null)
        : this(new ReadOnlySpan<T>(data), validityMask != null ? new ReadOnlySpan<bool>(validityMask) : ReadOnlySpan<bool>.Empty)
    {
    }

    /// <summary>
    /// Initializes a new instance of the TensorBackedSeries class from data.
    /// </summary>
    /// <param name="data">The data to store in the tensor.</param>
    /// <param name="validityMask">Optional validity mask. If null, all values are considered valid.</param>
    public TensorBackedSeries(ReadOnlySpan<T> data, ReadOnlySpan<bool> validityMask = default)
    {
        Length = data.Length;

        if (Length == 0)
        {
            tensor = Tensor.Create<T>(Array.Empty<T>(), new ReadOnlySpan<nint>(new nint[] { 0 }));
            validityMask = ReadOnlySpan<bool>.Empty;
            return;
        }

        // Create validity mask if not provided
        bool[]? validityArray = null;
        if (validityMask.IsEmpty)
        {
            validityArray = new bool[Length];
            Array.Fill(validityArray, true);
        }
        else
        {
            if (validityMask.Length != Length)
                throw new ArgumentException("Validity mask length must match data length.", nameof(validityMask));

            validityArray = validityMask.ToArray();
        }

        // Copy data to array for tensor creation
        // Note: Tensor.Create<T>() in .NET 10 handles SIMD alignment internally
        // The underlying memory allocation is optimized for vector operations
        var dataArray = new T[Length];
        data.CopyTo(dataArray);

        // Create 1D tensor - Tensor.Create ensures proper memory alignment for SIMD operations
        var dimensions = new ReadOnlySpan<nint>(new nint[] { Length });
        tensor = Tensor.Create<T>(dataArray, dimensions);
        this.validityMask = validityArray;
        ownsTensor = true;
    }

    /// <summary>
    /// Gets the value at the specified index.
    /// </summary>
    /// <param name="index">The index of the value to get.</param>
    /// <returns>The value at the specified index, or null if the value is not valid.</returns>
    public T? GetValue(int index)
    {
        if (index < 0 || index >= Length)
            throw new ArgumentOutOfRangeException(nameof(index));

        if (tensor == null)
            return null;

        if (validityMask != null && !validityMask[index])
            return null;

        var tensorSpan = tensor.AsTensorSpan();
        return tensorSpan[index];
    }

    /// <summary>
    /// Gets the raw value at the specified index without checking validity.
    /// </summary>
    /// <param name="index">The index of the value to get.</param>
    /// <returns>The raw value at the specified index.</returns>
    public T GetRawValue(int index)
    {
        if (index < 0 || index >= Length)
            throw new ArgumentOutOfRangeException(nameof(index));

        if (tensor == null)
            throw new InvalidOperationException("Tensor is not available.");

        var tensorSpan = tensor.AsTensorSpan();
        return tensorSpan[index];
    }

    /// <summary>
    /// Checks if the value at the specified index is valid (not null).
    /// </summary>
    /// <param name="index">The index to check.</param>
    /// <returns>True if the value is valid, false otherwise.</returns>
    public bool IsValid(int index)
    {
        if (index < 0 || index >= Length)
            throw new ArgumentOutOfRangeException(nameof(index));

        if (validityMask == null)
            return true;

        return validityMask[index];
    }

    /// <summary>
    /// Creates a slice of this series without copying the underlying data.
    /// </summary>
    /// <param name="start">The starting index of the slice.</param>
    /// <param name="length">The length of the slice.</param>
    /// <returns>A new TensorBackedSeries that represents a slice of this series.</returns>
    public TensorBackedSeries<T> Slice(int start, int length)
    {
        if (start < 0 || start >= Length)
            throw new ArgumentOutOfRangeException(nameof(start));
        if (length < 0 || start + length > Length)
            throw new ArgumentOutOfRangeException(nameof(length));

        if (tensor == null)
            throw new InvalidOperationException("Cannot slice when tensor is not available.");

        // Create a new tensor from the slice
        var tensorSpan = tensor.AsTensorSpan();
        var sliceData = new T[length];
        var sliceValidity = validityMask != null ? new bool[length] : null;

        for (int i = 0; i < length; i++)
        {
            sliceData[i] = tensorSpan[start + i];
            if (sliceValidity != null)
            {
                sliceValidity[i] = validityMask![start + i];
            }
        }

        var sliceDimensions = new ReadOnlySpan<nint>(new nint[] { length });
        var sliceTensor = Tensor.Create<T>(sliceData, sliceDimensions);

        return new TensorBackedSeries<T>(sliceTensor, sliceValidity, ownsTensor: true);
    }

    /// <summary>
    /// Creates a TensorSpan&lt;T&gt; view of this series if no null values are present.
    /// </summary>
    /// <returns>A TensorSpan&lt;T&gt; view of the tensor data.</returns>
    /// <exception cref="InvalidOperationException">Thrown when series contains null values.</exception>
    public TensorSpan<T> GetTensorSpan()
    {
        if (tensor == null)
            throw new InvalidOperationException("Tensor is not available.");

        // Verify no null values
        if (validityMask != null)
        {
            for (int i = 0; i < Length; i++)
            {
                if (!validityMask[i])
                {
                    throw new InvalidOperationException("Cannot create TensorSpan from series with null values.");
                }
            }
        }

        return tensor.AsTensorSpan();
    }

    /// <summary>
    /// Gets the underlying tensor. The caller should not dispose it unless ownsTensor is false.
    /// </summary>
    /// <returns>The underlying tensor.</returns>
    public Tensor<T> GetTensor()
    {
        if (tensor == null)
            throw new InvalidOperationException("Tensor is not available.");

        return tensor;
    }

    /// <summary>
    /// Releases all resources used by the TensorBackedSeries.
    /// </summary>
    public void Dispose()
    {
        if (!disposed)
        {
            // Tensor<T> doesn't implement IDisposable, just clear the reference
            if (ownsTensor && tensor != null)
            {
                tensor = null;
            }

            disposed = true;
        }
    }
}

