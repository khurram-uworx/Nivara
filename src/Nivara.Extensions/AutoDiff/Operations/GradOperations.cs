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

    #region Reduction Operations

    /// <summary>
    /// Computes the sum of all elements in the tensor with gradient computation.
    /// Uses existing NivaraSeries.Sum() method for the forward pass.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="a">The tensor to sum</param>
    /// <returns>A scalar GradTensor containing the sum with gradient tracking</returns>
    /// <exception cref="ArgumentNullException">Thrown when operand is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when the tensor is empty</exception>
    public static GradTensor<T> Sum<T>(GradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (a.Length == 0)
        {
            throw new InvalidOperationException("Cannot compute sum of empty tensor");
        }

        // Use existing NivaraSeries.Sum() for forward pass
        var series = a.ToSeries();
        var sumValue = series.Sum();

        // Create scalar result tensor
        var resultData = NivaraColumn<T>.Create(new T[] { sumValue });
        var resultTensor = new GradTensor<T>(resultData, a.RequiresGrad);

        // Add gradient computation if needed
        if (a.RequiresGrad)
        {
            var gradFn = new OpNode("Sum", new object[] { a }, gradOutput =>
            {
                var typedGradOutput = ConvertGradOutput<T>(gradOutput);

                // Gradient of sum: d/da (sum(a)) = ones_like(a) * gradOutput
                // The gradient broadcasts back to the original tensor shape
                var aGrad = BroadcastGradient(typedGradOutput, a.Length);
                AccumulateGradient(a, aGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Computes the mean (average) of all elements in the tensor with gradient computation.
    /// Uses existing NivaraSeries.Average() method for the forward pass.
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="a">The tensor to average</param>
    /// <returns>A scalar GradTensor containing the mean with gradient tracking</returns>
    /// <exception cref="ArgumentNullException">Thrown when operand is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when the tensor is empty or contains only null values</exception>
    public static GradTensor<T> Mean<T>(GradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        if (a.Length == 0)
        {
            throw new InvalidOperationException("Cannot compute mean of empty tensor");
        }

        // Use existing NivaraSeries.Average() for forward pass
        var series = a.ToSeries();
        var meanValue = series.Average();

        // Create scalar result tensor
        var resultData = NivaraColumn<T>.Create(new T[] { meanValue });
        var resultTensor = new GradTensor<T>(resultData, a.RequiresGrad);

        // Add gradient computation if needed
        if (a.RequiresGrad)
        {
            var gradFn = new OpNode("Mean", new object[] { a }, gradOutput =>
            {
                var typedGradOutput = ConvertGradOutput<T>(gradOutput);

                // Gradient of mean: d/da (mean(a)) = (1/n) * ones_like(a) * gradOutput
                // where n is the number of elements
                var aGrad = BroadcastGradient(typedGradOutput, a.Length);

                // Divide by the number of elements
                var scaledGrad = DivideByScalar(aGrad, a.Length);
                AccumulateGradient(a, scaledGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    #endregion

    #region Activation Functions

    /// <summary>
    /// Applies the ReLU (Rectified Linear Unit) activation function with gradient computation.
    /// ReLU(x) = max(0, x)
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="a">The input tensor</param>
    /// <returns>A new GradTensor with ReLU applied element-wise with gradient tracking</returns>
    /// <exception cref="ArgumentNullException">Thrown when operand is null</exception>
    public static GradTensor<T> Relu<T>(GradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        // Apply ReLU: max(0, x)
        var result = ApplyRelu(a.Data);

        // Create result tensor
        var resultTensor = new GradTensor<T>(result, a.RequiresGrad);

        // Add gradient computation if needed
        if (a.RequiresGrad)
        {
            var gradFn = new OpNode("Relu", new object[] { a }, gradOutput =>
            {
                var typedGradOutput = ConvertGradOutput<T>(gradOutput);

                // Gradient of ReLU: d/dx ReLU(x) = 1 if x > 0, else 0
                var aGrad = ApplyReluGradient(a.Data, typedGradOutput);
                AccumulateGradient(a, aGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Applies the Sigmoid activation function with gradient computation.
    /// Sigmoid(x) = 1 / (1 + exp(-x))
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="a">The input tensor</param>
    /// <returns>A new GradTensor with Sigmoid applied element-wise with gradient tracking</returns>
    /// <exception cref="ArgumentNullException">Thrown when operand is null</exception>
    public static GradTensor<T> Sigmoid<T>(GradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        // Apply Sigmoid: 1 / (1 + exp(-x))
        var result = ApplySigmoid(a.Data);

        // Create result tensor
        var resultTensor = new GradTensor<T>(result, a.RequiresGrad);

        // Add gradient computation if needed
        if (a.RequiresGrad)
        {
            var gradFn = new OpNode("Sigmoid", new object[] { a }, gradOutput =>
            {
                var typedGradOutput = ConvertGradOutput<T>(gradOutput);

                // Gradient of Sigmoid: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
                // We already have sigmoid(x) in result, so we can reuse it
                var aGrad = ApplySigmoidGradient(result, typedGradOutput);
                AccumulateGradient(a, aGrad);
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    /// <summary>
    /// Applies the Tanh (Hyperbolic Tangent) activation function with gradient computation.
    /// Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    /// </summary>
    /// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
    /// <param name="a">The input tensor</param>
    /// <returns>A new GradTensor with Tanh applied element-wise with gradient tracking</returns>
    /// <exception cref="ArgumentNullException">Thrown when operand is null</exception>
    public static GradTensor<T> Tanh<T>(GradTensor<T> a) where T : struct, INumber<T>
    {
        if (a == null) throw new ArgumentNullException(nameof(a));

        // Apply Tanh
        var result = ApplyTanh(a.Data);

        // Create result tensor
        var resultTensor = new GradTensor<T>(result, a.RequiresGrad);

        // Add gradient computation if needed
        if (a.RequiresGrad)
        {
            var gradFn = new OpNode("Tanh", new object[] { a }, gradOutput =>
            {
                var typedGradOutput = ConvertGradOutput<T>(gradOutput);

                // Gradient of Tanh: d/dx tanh(x) = 1 - tanh(x)^2
                // We already have tanh(x) in result, so we can reuse it
                var aGrad = ApplyTanhGradient(result, typedGradOutput);
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
        var nullMask = new bool[a.Length];
        bool hasNulls = false;

        // Use TensorPrimitives for optimized operations when available
        if (typeof(T) == typeof(float))
        {
            var aSpan = MemoryMarshal.Cast<T, float>(GetDataSpan(a));
            var bSpan = MemoryMarshal.Cast<T, float>(GetDataSpan(b));
            var resultSpan = MemoryMarshal.Cast<T, float>(result.AsSpan());

            TensorPrimitives.Subtract(aSpan, bSpan, resultSpan);
            
            // Propagate null mask
            for (int i = 0; i < a.Length; i++)
            {
                nullMask[i] = a.IsNull(i) || b.IsNull(i);
                if (nullMask[i]) hasNulls = true;
            }
        }
        else if (typeof(T) == typeof(double))
        {
            var aSpan = MemoryMarshal.Cast<T, double>(GetDataSpan(a));
            var bSpan = MemoryMarshal.Cast<T, double>(GetDataSpan(b));
            var resultSpan = MemoryMarshal.Cast<T, double>(result.AsSpan());

            TensorPrimitives.Subtract(aSpan, bSpan, resultSpan);
            
            // Propagate null mask
            for (int i = 0; i < a.Length; i++)
            {
                nullMask[i] = a.IsNull(i) || b.IsNull(i);
                if (nullMask[i]) hasNulls = true;
            }
        }
        else
        {
            // Fallback to element-wise subtraction for other types
            for (int i = 0; i < a.Length; i++)
            {
                nullMask[i] = a.IsNull(i) || b.IsNull(i);
                if (nullMask[i])
                {
                    hasNulls = true;
                }
                else
                {
                    result[i] = a[i] - b[i];
                }
            }
        }

        return CreateColumnWithNullMask(result, nullMask, hasNulls);
    }

    /// <summary>
    /// Performs vectorized division using TensorPrimitives when possible
    /// </summary>
    private static NivaraColumn<T> DivideVectorized<T>(NivaraColumn<T> a, NivaraColumn<T> b) where T : struct, INumber<T>
    {
        var result = new T[a.Length];
        var nullMask = new bool[a.Length];
        bool hasNulls = false;

        // Use TensorPrimitives for optimized operations when available
        if (typeof(T) == typeof(float))
        {
            var aSpan = MemoryMarshal.Cast<T, float>(GetDataSpan(a));
            var bSpan = MemoryMarshal.Cast<T, float>(GetDataSpan(b));
            var resultSpan = MemoryMarshal.Cast<T, float>(result.AsSpan());

            TensorPrimitives.Divide(aSpan, bSpan, resultSpan);
            
            // Propagate null mask
            for (int i = 0; i < a.Length; i++)
            {
                nullMask[i] = a.IsNull(i) || b.IsNull(i);
                if (nullMask[i]) hasNulls = true;
            }
        }
        else if (typeof(T) == typeof(double))
        {
            var aSpan = MemoryMarshal.Cast<T, double>(GetDataSpan(a));
            var bSpan = MemoryMarshal.Cast<T, double>(GetDataSpan(b));
            var resultSpan = MemoryMarshal.Cast<T, double>(result.AsSpan());

            TensorPrimitives.Divide(aSpan, bSpan, resultSpan);
            
            // Propagate null mask
            for (int i = 0; i < a.Length; i++)
            {
                nullMask[i] = a.IsNull(i) || b.IsNull(i);
                if (nullMask[i]) hasNulls = true;
            }
        }
        else
        {
            // Fallback to element-wise division for other types
            for (int i = 0; i < a.Length; i++)
            {
                nullMask[i] = a.IsNull(i) || b.IsNull(i);
                if (nullMask[i])
                {
                    hasNulls = true;
                }
                else
                {
                    result[i] = a[i] / b[i];
                }
            }
        }

        return CreateColumnWithNullMask(result, nullMask, hasNulls);
    }

    /// <summary>
    /// Performs vectorized negation using TensorPrimitives when possible
    /// </summary>
    private static NivaraColumn<T> NegateVectorized<T>(NivaraColumn<T> a) where T : struct, INumber<T>
    {
        var result = new T[a.Length];
        var nullMask = new bool[a.Length];
        bool hasNulls = false;

        // Use TensorPrimitives for optimized operations when available
        if (typeof(T) == typeof(float))
        {
            var aSpan = MemoryMarshal.Cast<T, float>(GetDataSpan(a));
            var resultSpan = MemoryMarshal.Cast<T, float>(result.AsSpan());

            TensorPrimitives.Negate(aSpan, resultSpan);
            
            // Propagate null mask
            for (int i = 0; i < a.Length; i++)
            {
                nullMask[i] = a.IsNull(i);
                if (nullMask[i]) hasNulls = true;
            }
        }
        else if (typeof(T) == typeof(double))
        {
            var aSpan = MemoryMarshal.Cast<T, double>(GetDataSpan(a));
            var resultSpan = MemoryMarshal.Cast<T, double>(result.AsSpan());

            TensorPrimitives.Negate(aSpan, resultSpan);
            
            // Propagate null mask
            for (int i = 0; i < a.Length; i++)
            {
                nullMask[i] = a.IsNull(i);
                if (nullMask[i]) hasNulls = true;
            }
        }
        else
        {
            // Fallback to element-wise negation for other types
            for (int i = 0; i < a.Length; i++)
            {
                nullMask[i] = a.IsNull(i);
                if (nullMask[i])
                {
                    hasNulls = true;
                }
                else
                {
                    result[i] = -a[i];
                }
            }
        }

        return CreateColumnWithNullMask(result, nullMask, hasNulls);
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

    /// <summary>
    /// Broadcasts a scalar gradient to match the shape of the original tensor
    /// </summary>
    private static NivaraColumn<T> BroadcastGradient<T>(NivaraColumn<T> scalarGrad, int targetLength) where T : struct, INumber<T>
    {
        if (scalarGrad.Length != 1)
        {
            throw new ArgumentException($"Expected scalar gradient with length 1, got {scalarGrad.Length}");
        }

        var gradValue = scalarGrad[0];
        var result = new T[targetLength];
        for (int i = 0; i < targetLength; i++)
        {
            result[i] = gradValue;
        }

        return NivaraColumn<T>.Create(result);
    }

    /// <summary>
    /// Divides all elements in a column by a scalar value
    /// </summary>
    private static NivaraColumn<T> DivideByScalar<T>(NivaraColumn<T> column, int divisor) where T : struct, INumber<T>
    {
        var result = new T[column.Length];

        // Convert divisor to type T
        T divisorT = ConvertToT<T>(divisor);

        for (int i = 0; i < column.Length; i++)
        {
            if (!column.IsNull(i))
            {
                result[i] = column[i] / divisorT;
            }
        }

        return NivaraColumn<T>.Create(result);
    }

    /// <summary>
    /// Converts an integer to type T for scalar operations
    /// </summary>
    private static T ConvertToT<T>(int value) where T : struct, INumber<T>
    {
        var type = typeof(T);

        if (type == typeof(float))
        {
            return (T)(object)(float)value;
        }
        else if (type == typeof(double))
        {
            return (T)(object)(double)value;
        }
        else if (type == typeof(int))
        {
            return (T)(object)value;
        }
        else if (type == typeof(long))
        {
            return (T)(object)(long)value;
        }
        else
        {
            // Fall back to dynamic conversion for other types
            return (T)(object)((dynamic)value)!;
        }
    }

    /// <summary>
    /// Applies ReLU activation function: max(0, x)
    /// </summary>
    private static NivaraColumn<T> ApplyRelu<T>(NivaraColumn<T> input) where T : struct, INumber<T>
    {
        var result = new T[input.Length];
        var nullMask = new bool[input.Length];
        bool hasNulls = false;

        // Use TensorPrimitives for optimized operations when available
        if (typeof(T) == typeof(float))
        {
            var inputSpan = GetDataSpan(input);
            var floatSpan = MemoryMarshal.Cast<T, float>(inputSpan);
            var resultSpan = MemoryMarshal.Cast<T, float>(result.AsSpan());

            // ReLU: max(0, x)
            for (int i = 0; i < floatSpan.Length; i++)
            {
                resultSpan[i] = Math.Max(0.0f, floatSpan[i]);
                nullMask[i] = input.IsNull(i);
                if (nullMask[i]) hasNulls = true;
            }
        }
        else if (typeof(T) == typeof(double))
        {
            var inputSpan = GetDataSpan(input);
            var doubleSpan = MemoryMarshal.Cast<T, double>(inputSpan);
            var resultSpan = MemoryMarshal.Cast<T, double>(result.AsSpan());

            // ReLU: max(0, x)
            for (int i = 0; i < doubleSpan.Length; i++)
            {
                resultSpan[i] = Math.Max(0.0, doubleSpan[i]);
                nullMask[i] = input.IsNull(i);
                if (nullMask[i]) hasNulls = true;
            }
        }
        else
        {
            // Fallback for other types
            for (int i = 0; i < input.Length; i++)
            {
                nullMask[i] = input.IsNull(i);
                if (nullMask[i])
                {
                    hasNulls = true;
                }
                else
                {
                    result[i] = input[i] > T.Zero ? input[i] : T.Zero;
                }
            }
        }

        return CreateColumnWithNullMask(result, nullMask, hasNulls);
    }

    /// <summary>
    /// Applies ReLU gradient: 1 if x > 0, else 0
    /// </summary>
    private static NivaraColumn<T> ApplyReluGradient<T>(NivaraColumn<T> input, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        var result = new T[input.Length];

        for (int i = 0; i < input.Length; i++)
        {
            if (!input.IsNull(i) && !gradOutput.IsNull(i))
            {
                // Gradient is gradOutput if input > 0, else 0
                result[i] = input[i] > T.Zero ? gradOutput[i] : T.Zero;
            }
        }

        return NivaraColumn<T>.Create(result);
    }

    /// <summary>
    /// Applies Sigmoid activation function: 1 / (1 + exp(-x))
    /// </summary>
    private static NivaraColumn<T> ApplySigmoid<T>(NivaraColumn<T> input) where T : struct, INumber<T>
    {
        var result = new T[input.Length];
        var nullMask = new bool[input.Length];
        bool hasNulls = false;

        // Use TensorPrimitives for optimized operations when available
        if (typeof(T) == typeof(float))
        {
            var inputSpan = GetDataSpan(input);
            var floatSpan = MemoryMarshal.Cast<T, float>(inputSpan);
            var resultSpan = MemoryMarshal.Cast<T, float>(result.AsSpan());

            // Sigmoid: 1 / (1 + exp(-x))
            for (int i = 0; i < floatSpan.Length; i++)
            {
                resultSpan[i] = 1.0f / (1.0f + MathF.Exp(-floatSpan[i]));
                nullMask[i] = input.IsNull(i);
                if (nullMask[i]) hasNulls = true;
            }
        }
        else if (typeof(T) == typeof(double))
        {
            var inputSpan = GetDataSpan(input);
            var doubleSpan = MemoryMarshal.Cast<T, double>(inputSpan);
            var resultSpan = MemoryMarshal.Cast<T, double>(result.AsSpan());

            // Sigmoid: 1 / (1 + exp(-x))
            for (int i = 0; i < doubleSpan.Length; i++)
            {
                resultSpan[i] = 1.0 / (1.0 + Math.Exp(-doubleSpan[i]));
                nullMask[i] = input.IsNull(i);
                if (nullMask[i]) hasNulls = true;
            }
        }
        else
        {
            // Fallback for other types (using dynamic)
            for (int i = 0; i < input.Length; i++)
            {
                nullMask[i] = input.IsNull(i);
                if (nullMask[i])
                {
                    hasNulls = true;
                }
                else
                {
                    dynamic x = input[i];
                    result[i] = (T)(object)(1.0 / (1.0 + Math.Exp(-x)))!;
                }
            }
        }

        return CreateColumnWithNullMask(result, nullMask, hasNulls);
    }

    /// <summary>
    /// Applies Sigmoid gradient: sigmoid(x) * (1 - sigmoid(x)) * gradOutput
    /// </summary>
    private static NivaraColumn<T> ApplySigmoidGradient<T>(NivaraColumn<T> sigmoidOutput, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        var result = new T[sigmoidOutput.Length];

        for (int i = 0; i < sigmoidOutput.Length; i++)
        {
            if (!sigmoidOutput.IsNull(i) && !gradOutput.IsNull(i))
            {
                // Gradient: sigmoid(x) * (1 - sigmoid(x)) * gradOutput
                var sig = sigmoidOutput[i];
                var oneMinus = T.One - sig;
                result[i] = sig * oneMinus * gradOutput[i];
            }
        }

        return NivaraColumn<T>.Create(result);
    }

    /// <summary>
    /// Applies Tanh activation function
    /// </summary>
    private static NivaraColumn<T> ApplyTanh<T>(NivaraColumn<T> input) where T : struct, INumber<T>
    {
        var result = new T[input.Length];
        var nullMask = new bool[input.Length];
        bool hasNulls = false;

        // Use TensorPrimitives for optimized operations when available
        if (typeof(T) == typeof(float))
        {
            var inputSpan = GetDataSpan(input);
            var floatSpan = MemoryMarshal.Cast<T, float>(inputSpan);
            var resultSpan = MemoryMarshal.Cast<T, float>(result.AsSpan());

            // Tanh
            for (int i = 0; i < floatSpan.Length; i++)
            {
                resultSpan[i] = MathF.Tanh(floatSpan[i]);
                nullMask[i] = input.IsNull(i);
                if (nullMask[i]) hasNulls = true;
            }
        }
        else if (typeof(T) == typeof(double))
        {
            var inputSpan = GetDataSpan(input);
            var doubleSpan = MemoryMarshal.Cast<T, double>(inputSpan);
            var resultSpan = MemoryMarshal.Cast<T, double>(result.AsSpan());

            // Tanh
            for (int i = 0; i < doubleSpan.Length; i++)
            {
                resultSpan[i] = Math.Tanh(doubleSpan[i]);
                nullMask[i] = input.IsNull(i);
                if (nullMask[i]) hasNulls = true;
            }
        }
        else
        {
            // Fallback for other types (using dynamic)
            for (int i = 0; i < input.Length; i++)
            {
                nullMask[i] = input.IsNull(i);
                if (nullMask[i])
                {
                    hasNulls = true;
                }
                else
                {
                    dynamic x = input[i];
                    result[i] = (T)(object)Math.Tanh(x)!;
                }
            }
        }

        return CreateColumnWithNullMask(result, nullMask, hasNulls);
    }

    /// <summary>
    /// Applies Tanh gradient: (1 - tanh(x)^2) * gradOutput
    /// </summary>
    private static NivaraColumn<T> ApplyTanhGradient<T>(NivaraColumn<T> tanhOutput, NivaraColumn<T> gradOutput) where T : struct, INumber<T>
    {
        var result = new T[tanhOutput.Length];

        for (int i = 0; i < tanhOutput.Length; i++)
        {
            if (!tanhOutput.IsNull(i) && !gradOutput.IsNull(i))
            {
                // Gradient: (1 - tanh(x)^2) * gradOutput
                var tanh = tanhOutput[i];
                var tanhSquared = tanh * tanh;
                var oneMinus = T.One - tanhSquared;
                result[i] = oneMinus * gradOutput[i];
            }
        }

        return NivaraColumn<T>.Create(result);
    }

    /// <summary>
    /// Helper method to create a NivaraColumn with an explicit null mask
    /// </summary>
    private static NivaraColumn<T> CreateColumnWithNullMask<T>(T[] data, bool[] nullMask, bool hasNulls) where T : struct, INumber<T>
    {
        if (!hasNulls)
        {
            return NivaraColumn<T>.Create(data);
        }

        // Convert to nullable array for CreateFromNullable
        var nullableData = new T?[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            nullableData[i] = nullMask[i] ? null : data[i];
        }

        return NivaraColumn<T>.CreateFromNullable(nullableData);
    }

    #endregion
}