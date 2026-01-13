using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using Nivara;
using Nivara.Tensors;

namespace Nivara.Extensions.AutoDiff.Operations;

/// <summary>
/// Gradient-aware operations for automatic differentiation.
/// Contains static methods for performing operations on GradTensors with automatic gradient computation.
/// Wraps existing NivaraColumn operations while adding gradient tracking and computation.
/// </summary>
public static class GradOperations
{
    #region Element-wise Operations

    /// <summary>
    /// Performs element-wise addition with gradient computation.
    /// Wraps existing NivaraColumn addition (a.Data + b.Data) with gradient tracking.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="a">The first operand</param>
    /// <param name="b">The second operand</param>
    /// <returns>A new GradTensor containing the element-wise sum with gradient tracking</returns>
    /// <exception cref="ArgumentNullException">Thrown when either operand is null</exception>
    /// <exception cref="ArgumentException">Thrown when operands have incompatible shapes</exception>
    public static GradTensor<T> Add<T>(GradTensor<T> a, GradTensor<T> b) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Cannot add tensors with different lengths: {a.Length} vs {b.Length}");
        }

        // Use existing NivaraColumn addition (already vectorized and optimized)
        var result = a.Data + b.Data;

        // Create result tensor
        var resultTensor = new GradTensor<T>(result, a.RequiresGrad || b.RequiresGrad);

        // Add gradient computation if needed
        if (a.RequiresGrad || b.RequiresGrad)
        {
            var gradFn = new OpNode("Add", new object[] { a, b }, gradOutput =>
            {
                // Convert gradOutput back to typed column for gradient accumulation
                var typedGradOutput = ConvertGradOutput<T>(gradOutput);

                // Gradient of addition: d/da (a + b) = 1, d/db (a + b) = 1
                if (a.RequiresGrad)
                {
                    AccumulateGradient(a, typedGradOutput);
                }
                if (b.RequiresGrad)
                {
                    AccumulateGradient(b, typedGradOutput);
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Performs element-wise subtraction with gradient computation.
    /// Implements subtraction using existing operations with gradient tracking.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="a">The minuend</param>
    /// <param name="b">The subtrahend</param>
    /// <returns>A new GradTensor containing the element-wise difference with gradient tracking</returns>
    /// <exception cref="ArgumentNullException">Thrown when either operand is null</exception>
    /// <exception cref="ArgumentException">Thrown when operands have incompatible shapes</exception>
    public static GradTensor<T> Subtract<T>(GradTensor<T> a, GradTensor<T> b) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Cannot subtract tensors with different lengths: {a.Length} vs {b.Length}");
        }

        // Implement subtraction using element-wise operations since NivaraColumn doesn't have Subtract
        var result = SubtractVectorized(a.Data, b.Data);

        // Create result tensor
        var resultTensor = new GradTensor<T>(result, a.RequiresGrad || b.RequiresGrad);

        // Add gradient computation if needed
        if (a.RequiresGrad || b.RequiresGrad)
        {
            var gradFn = new OpNode("Subtract", new object[] { a, b }, gradOutput =>
            {
                var typedGradOutput = ConvertGradOutput<T>(gradOutput);

                // Gradient of subtraction: d/da (a - b) = 1, d/db (a - b) = -1
                if (a.RequiresGrad)
                {
                    AccumulateGradient(a, typedGradOutput);
                }
                if (b.RequiresGrad)
                {
                    // Negate the gradient for subtraction
                    var negatedGrad = NegateVectorized(typedGradOutput);
                    AccumulateGradient(b, negatedGrad);
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Performs element-wise multiplication with gradient computation.
    /// Wraps existing NivaraColumn multiplication (a.Data * b.Data) with gradient tracking.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="a">The first operand</param>
    /// <param name="b">The second operand</param>
    /// <returns>A new GradTensor containing the element-wise product with gradient tracking</returns>
    /// <exception cref="ArgumentNullException">Thrown when either operand is null</exception>
    /// <exception cref="ArgumentException">Thrown when operands have incompatible shapes</exception>
    public static GradTensor<T> Multiply<T>(GradTensor<T> a, GradTensor<T> b) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Cannot multiply tensors with different lengths: {a.Length} vs {b.Length}");
        }

        // Use existing NivaraColumn multiplication (already vectorized and optimized)
        var result = a.Data * b.Data;

        // Create result tensor
        var resultTensor = new GradTensor<T>(result, a.RequiresGrad || b.RequiresGrad);

        // Add gradient computation if needed
        if (a.RequiresGrad || b.RequiresGrad)
        {
            var gradFn = new OpNode("Multiply", new object[] { a, b }, gradOutput =>
            {
                var typedGradOutput = ConvertGradOutput<T>(gradOutput);

                // Gradient of multiplication: d/da (a * b) = b, d/db (a * b) = a
                if (a.RequiresGrad)
                {
                    var aGrad = MultiplyVectorized(typedGradOutput, b.Data);
                    AccumulateGradient(a, aGrad);
                }
                if (b.RequiresGrad)
                {
                    var bGrad = MultiplyVectorized(typedGradOutput, a.Data);
                    AccumulateGradient(b, bGrad);
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Performs element-wise division with gradient computation.
    /// Implements division using existing operations with gradient tracking.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="a">The dividend</param>
    /// <param name="b">The divisor</param>
    /// <returns>A new GradTensor containing the element-wise quotient with gradient tracking</returns>
    /// <exception cref="ArgumentNullException">Thrown when either operand is null</exception>
    /// <exception cref="ArgumentException">Thrown when operands have incompatible shapes</exception>
    /// <exception cref="DivideByZeroException">Thrown when divisor contains zero values</exception>
    public static GradTensor<T> Divide<T>(GradTensor<T> a, GradTensor<T> b) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Cannot divide tensors with different lengths: {a.Length} vs {b.Length}");
        }

        // Check for division by zero
        for (int i = 0; i < b.Length; i++)
        {
            if (!b.IsNull(i) && b[i] == T.Zero)
            {
                throw new DivideByZeroException($"Division by zero at index {i}");
            }
        }

        // Implement division using element-wise operations since NivaraColumn doesn't have Divide
        var result = DivideVectorized(a.Data, b.Data);

        // Create result tensor
        var resultTensor = new GradTensor<T>(result, a.RequiresGrad || b.RequiresGrad);

        // Add gradient computation if needed
        if (a.RequiresGrad || b.RequiresGrad)
        {
            var gradFn = new OpNode("Divide", new object[] { a, b }, gradOutput =>
            {
                var typedGradOutput = ConvertGradOutput<T>(gradOutput);

                // Gradient of division: d/da (a / b) = 1/b, d/db (a / b) = -a/(b^2)
                if (a.RequiresGrad)
                {
                    var aGrad = DivideVectorized(typedGradOutput, b.Data);
                    AccumulateGradient(a, aGrad);
                }
                if (b.RequiresGrad)
                {
                    // -a / (b^2) = -(a / b) / b
                    var quotient = DivideVectorized(a.Data, b.Data);
                    var bGradPositive = DivideVectorized(quotient, b.Data);
                    var bGrad = NegateVectorized(bGradPositive);
                    var finalBGrad = MultiplyVectorized(bGrad, typedGradOutput);
                    AccumulateGradient(b, finalBGrad);
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    #endregion

    #region Matrix Operations

    /// <summary>
    /// Performs matrix multiplication with gradient computation.
    /// Uses Data.AsTensor() for tensor operations and converts results back to NivaraColumn&lt;T&gt;.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="a">The left matrix (as a flattened GradTensor)</param>
    /// <param name="b">The right matrix (as a flattened GradTensor)</param>
    /// <param name="aRows">Number of rows in matrix a</param>
    /// <param name="aCols">Number of columns in matrix a (must equal bRows)</param>
    /// <param name="bCols">Number of columns in matrix b</param>
    /// <returns>A new GradTensor containing the matrix multiplication result with gradient tracking</returns>
    /// <exception cref="ArgumentNullException">Thrown when either operand is null</exception>
    /// <exception cref="ArgumentException">Thrown when matrix dimensions are incompatible</exception>
    public static GradTensor<T> MatMul<T>(GradTensor<T> a, GradTensor<T> b, int aRows, int aCols, int bCols) 
        where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        var bRows = aCols; // For matrix multiplication, a.cols must equal b.rows

        if (a.Length != aRows * aCols)
        {
            throw new ArgumentException($"Matrix a dimensions don't match: expected {aRows * aCols} elements, got {a.Length}");
        }

        if (b.Length != bRows * bCols)
        {
            throw new ArgumentException($"Matrix b dimensions don't match: expected {bRows * bCols} elements, got {b.Length}");
        }

        // Perform matrix multiplication using tensor operations
        var result = MatMulVectorized(a.Data, b.Data, aRows, aCols, bCols);

        // Create result tensor
        var resultTensor = new GradTensor<T>(result, a.RequiresGrad || b.RequiresGrad);

        // Add gradient computation if needed
        if (a.RequiresGrad || b.RequiresGrad)
        {
            var gradFn = new OpNode("MatMul", new object[] { a, b }, gradOutput =>
            {
                var typedGradOutput = ConvertGradOutput<T>(gradOutput);

                // Gradient of matrix multiplication:
                // d/da (a @ b) = gradOutput @ b^T
                // d/db (a @ b) = a^T @ gradOutput
                if (a.RequiresGrad)
                {
                    var bTransposed = TransposeVectorized(b.Data, bRows, bCols);
                    var aGrad = MatMulVectorized(typedGradOutput, bTransposed, aRows, bCols, bRows);
                    AccumulateGradient(a, aGrad);
                }
                if (b.RequiresGrad)
                {
                    var aTransposed = TransposeVectorized(a.Data, aRows, aCols);
                    var bGrad = MatMulVectorized(aTransposed, typedGradOutput, aCols, aRows, bCols);
                    AccumulateGradient(b, bGrad);
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Performs matrix transpose with gradient computation.
    /// Wraps existing tensor transpose operations with gradient tracking.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="a">The matrix to transpose (as a flattened GradTensor)</param>
    /// <param name="rows">Number of rows in the original matrix</param>
    /// <param name="cols">Number of columns in the original matrix</param>
    /// <returns>A new GradTensor containing the transposed matrix with gradient tracking</returns>
    /// <exception cref="ArgumentNullException">Thrown when operand is null</exception>
    /// <exception cref="ArgumentException">Thrown when matrix dimensions don't match tensor length</exception>
    public static GradTensor<T> Transpose<T>(GradTensor<T> a, int rows, int cols) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (a.Length != rows * cols)
        {
            throw new ArgumentException($"Matrix dimensions don't match: expected {rows * cols} elements, got {a.Length}");
        }

        // Perform transpose operation
        var result = TransposeVectorized(a.Data, rows, cols);

        // Create result tensor
        var resultTensor = new GradTensor<T>(result, a.RequiresGrad);

        // Add gradient computation if needed
        if (a.RequiresGrad)
        {
            var gradFn = new OpNode("Transpose", new object[] { a }, gradOutput =>
            {
                var typedGradOutput = ConvertGradOutput<T>(gradOutput);

                // Gradient of transpose: d/da (a^T) = (gradOutput)^T
                // Transpose the gradient back to original shape
                var aGrad = TransposeVectorized(typedGradOutput, cols, rows);
                AccumulateGradient(a, aGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Converts object-typed gradient output back to typed NivaraColumn for gradient accumulation
    /// </summary>
    private static NivaraColumn<T> ConvertGradOutput<T>(NivaraColumn<object> gradOutput) where T : struct, INumber<T>
    {
        var typedData = new T[gradOutput.Length];
        for (int i = 0; i < gradOutput.Length; i++)
        {
            if (!gradOutput.IsNull(i))
            {
                typedData[i] = (T)gradOutput[i];
            }
        }
        return NivaraColumn<T>.Create(typedData);
    }

    /// <summary>
    /// Accumulates gradient into a tensor's gradient field
    /// </summary>
    private static void AccumulateGradient<T>(GradTensor<T> tensor, NivaraColumn<T> gradient) where T : struct, INumber<T>
    {
        if (tensor.Grad == null)
        {
            tensor.Grad = gradient;
        }
        else
        {
            // Add to existing gradient
            tensor.Grad = tensor.Grad + gradient;
        }
    }

    /// <summary>
    /// Performs vectorized subtraction using TensorPrimitives when possible
    /// </summary>
    private static NivaraColumn<T> SubtractVectorized<T>(NivaraColumn<T> a, NivaraColumn<T> b) where T : struct, INumber<T>
    {
        var result = new T[a.Length];

        // Use TensorPrimitives for optimized operations when available
        if (typeof(T) == typeof(float))
        {
            var aSpan = MemoryMarshal.Cast<T, float>(GetDataSpan(a));
            var bSpan = MemoryMarshal.Cast<T, float>(GetDataSpan(b));
            var resultSpan = MemoryMarshal.Cast<T, float>(result.AsSpan());

            TensorPrimitives.Subtract(aSpan, bSpan, resultSpan);
        }
        else if (typeof(T) == typeof(double))
        {
            var aSpan = MemoryMarshal.Cast<T, double>(GetDataSpan(a));
            var bSpan = MemoryMarshal.Cast<T, double>(GetDataSpan(b));
            var resultSpan = MemoryMarshal.Cast<T, double>(result.AsSpan());

            TensorPrimitives.Subtract(aSpan, bSpan, resultSpan);
        }
        else
        {
            // Fallback to element-wise subtraction for other types
            for (int i = 0; i < a.Length; i++)
            {
                if (!a.IsNull(i) && !b.IsNull(i))
                {
                    result[i] = a[i] - b[i];
                }
            }
        }

        return NivaraColumn<T>.Create(result);
    }

    /// <summary>
    /// Performs vectorized division using TensorPrimitives when possible
    /// </summary>
    private static NivaraColumn<T> DivideVectorized<T>(NivaraColumn<T> a, NivaraColumn<T> b) where T : struct, INumber<T>
    {
        var result = new T[a.Length];

        // Use TensorPrimitives for optimized operations when available
        if (typeof(T) == typeof(float))
        {
            var aSpan = MemoryMarshal.Cast<T, float>(GetDataSpan(a));
            var bSpan = MemoryMarshal.Cast<T, float>(GetDataSpan(b));
            var resultSpan = MemoryMarshal.Cast<T, float>(result.AsSpan());

            TensorPrimitives.Divide(aSpan, bSpan, resultSpan);
        }
        else if (typeof(T) == typeof(double))
        {
            var aSpan = MemoryMarshal.Cast<T, double>(GetDataSpan(a));
            var bSpan = MemoryMarshal.Cast<T, double>(GetDataSpan(b));
            var resultSpan = MemoryMarshal.Cast<T, double>(result.AsSpan());

            TensorPrimitives.Divide(aSpan, bSpan, resultSpan);
        }
        else
        {
            // Fallback to element-wise division for other types
            for (int i = 0; i < a.Length; i++)
            {
                if (!a.IsNull(i) && !b.IsNull(i))
                {
                    result[i] = a[i] / b[i];
                }
            }
        }

        return NivaraColumn<T>.Create(result);
    }

    /// <summary>
    /// Performs vectorized negation using TensorPrimitives when possible
    /// </summary>
    private static NivaraColumn<T> NegateVectorized<T>(NivaraColumn<T> a) where T : struct, INumber<T>
    {
        var result = new T[a.Length];

        // Use TensorPrimitives for optimized operations when available
        if (typeof(T) == typeof(float))
        {
            var aSpan = MemoryMarshal.Cast<T, float>(GetDataSpan(a));
            var resultSpan = MemoryMarshal.Cast<T, float>(result.AsSpan());

            TensorPrimitives.Negate(aSpan, resultSpan);
        }
        else if (typeof(T) == typeof(double))
        {
            var aSpan = MemoryMarshal.Cast<T, double>(GetDataSpan(a));
            var resultSpan = MemoryMarshal.Cast<T, double>(result.AsSpan());

            TensorPrimitives.Negate(aSpan, resultSpan);
        }
        else
        {
            // Fallback to element-wise negation for other types
            for (int i = 0; i < a.Length; i++)
            {
                if (!a.IsNull(i))
                {
                    result[i] = -a[i];
                }
            }
        }

        return NivaraColumn<T>.Create(result);
    }

    /// <summary>
    /// Performs vectorized multiplication using existing NivaraColumn operations
    /// </summary>
    private static NivaraColumn<T> MultiplyVectorized<T>(NivaraColumn<T> a, NivaraColumn<T> b) where T : struct, INumber<T>
    {
        // Use existing NivaraColumn multiplication which is already vectorized
        return a * b;
    }

    /// <summary>
    /// Gets data span from NivaraColumn for vectorized operations
    /// </summary>
    private static Span<T> GetDataSpan<T>(NivaraColumn<T> column) where T : struct, INumber<T>
    {
        var data = new T[column.Length];
        for (int i = 0; i < column.Length; i++)
        {
            data[i] = column[i];
        }
        return data.AsSpan();
    }

    /// <summary>
    /// Performs vectorized matrix multiplication using tensor operations
    /// </summary>
    private static NivaraColumn<T> MatMulVectorized<T>(NivaraColumn<T> a, NivaraColumn<T> b, int aRows, int aCols, int bCols) 
        where T : struct, INumber<T>
    {
        var result = new T[aRows * bCols];

        // Manual matrix multiplication (could be optimized with BLAS in the future)
        for (int i = 0; i < aRows; i++)
        {
            for (int j = 0; j < bCols; j++)
            {
                T sum = T.Zero;
                for (int k = 0; k < aCols; k++)
                {
                    var aIndex = i * aCols + k;
                    var bIndex = k * bCols + j;
                    sum += a[aIndex] * b[bIndex];
                }
                result[i * bCols + j] = sum;
            }
        }

        return NivaraColumn<T>.Create(result);
    }

    /// <summary>
    /// Performs vectorized matrix transpose
    /// </summary>
    private static NivaraColumn<T> TransposeVectorized<T>(NivaraColumn<T> a, int rows, int cols) where T : struct, INumber<T>
    {
        var result = new T[rows * cols];

        // Transpose: result[j * rows + i] = a[i * cols + j]
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[j * rows + i] = a[i * cols + j];
            }
        }

        return NivaraColumn<T>.Create(result);
    }

    #endregion
}