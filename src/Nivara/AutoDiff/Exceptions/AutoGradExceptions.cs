using System.Text;

namespace Nivara.AutoDiff.Exceptions;

/// <summary>
/// Base exception for all automatic differentiation errors.
/// Provides detailed context about the operation, tensor shapes, and debugging information.
/// </summary>
public class AutoGradException : Exception
{
    /// <summary>
    /// Initializes a new instance of AutoGradException
    /// </summary>
    /// <param name="message">The error message</param>
    public AutoGradException(string message) : base(message)
    {
    }

    /// <summary>
    /// Initializes a new instance of AutoGradException with an inner exception
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="innerException">The inner exception that caused this error</param>
    public AutoGradException(string message, Exception innerException) : base(message, innerException)
    {
    }

    /// <summary>
    /// Initializes a new instance of AutoGradException with operation context
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="operationContext">The context of the operation that failed</param>
    protected AutoGradException(string message, string? operationContext) : base(message)
    {
        OperationContext = operationContext;
    }

    /// <summary>
    /// Initializes a new instance of AutoGradException with operation context and inner exception
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="operationContext">The context of the operation that failed</param>
    /// <param name="innerException">The inner exception that caused this error</param>
    protected AutoGradException(string message, string? operationContext, Exception innerException)
        : base(message, innerException)
    {
        OperationContext = operationContext;
    }

    /// <summary>
    /// Gets the context of the operation that failed, if available
    /// </summary>
    public string? OperationContext { get; }

    /// <summary>
    /// Gets the shapes of tensors involved in the operation, if available
    /// </summary>
    public IReadOnlyList<int[]>? InvolvedShapes { get; protected init; }

    /// <summary>
    /// Gets detailed context information about the failure for debugging
    /// </summary>
    /// <returns>A formatted string with detailed error context</returns>
    public virtual string GetDetailedContext()
    {
        var context = new StringBuilder();
        context.AppendLine($"Exception Type: {GetType().Name}");
        context.AppendLine($"Message: {Message}");

        if (!string.IsNullOrEmpty(OperationContext))
        {
            context.AppendLine($"Operation Context: {OperationContext}");
        }

        if (InvolvedShapes != null && InvolvedShapes.Count > 0)
        {
            context.AppendLine("Involved Tensor Shapes:");
            for (int i = 0; i < InvolvedShapes.Count; i++)
            {
                context.AppendLine($"  Tensor {i}: [{string.Join(", ", InvolvedShapes[i])}]");
            }
        }

        if (InnerException != null)
        {
            context.AppendLine($"Inner Exception: {InnerException.GetType().Name}: {InnerException.Message}");
        }

        return context.ToString();
    }
}

/// <summary>
/// Exception thrown when tensor shapes are incompatible for an operation.
/// Provides detailed information about expected vs actual shapes.
/// </summary>
public sealed class ShapeIncompatibilityException : AutoGradException
{
    /// <summary>
    /// Initializes a new instance of ShapeIncompatibilityException
    /// </summary>
    /// <param name="message">The error message</param>
    public ShapeIncompatibilityException(string message) : base(message)
    {
    }

    /// <summary>
    /// Initializes a new instance of ShapeIncompatibilityException with shape details
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="operationName">The name of the operation that encountered shape incompatibility</param>
    /// <param name="expectedShape">The expected tensor shape</param>
    /// <param name="actualShape">The actual tensor shape</param>
    public ShapeIncompatibilityException(string message, string operationName, int[] expectedShape, int[] actualShape)
        : base(message, operationName)
    {
        ExpectedShape = expectedShape;
        ActualShape = actualShape;
        InvolvedShapes = new[] { expectedShape, actualShape };
    }

    /// <summary>
    /// Initializes a new instance of ShapeIncompatibilityException with multiple tensor shapes
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="operationName">The name of the operation that encountered shape incompatibility</param>
    /// <param name="tensorShapes">The shapes of all tensors involved in the operation</param>
    public ShapeIncompatibilityException(string message, string operationName, params int[][] tensorShapes)
        : base(message, operationName)
    {
        if (tensorShapes.Length >= 2)
        {
            ExpectedShape = tensorShapes[0];
            ActualShape = tensorShapes[1];
        }
        InvolvedShapes = tensorShapes;
    }

    /// <summary>
    /// Initializes a new instance of ShapeIncompatibilityException with inner exception
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="operationName">The name of the operation that encountered shape incompatibility</param>
    /// <param name="expectedShape">The expected tensor shape</param>
    /// <param name="actualShape">The actual tensor shape</param>
    /// <param name="innerException">The inner exception that caused the shape incompatibility</param>
    public ShapeIncompatibilityException(string message, string operationName, int[] expectedShape, int[] actualShape, Exception innerException)
        : base(message, operationName, innerException)
    {
        ExpectedShape = expectedShape;
        ActualShape = actualShape;
        InvolvedShapes = new[] { expectedShape, actualShape };
    }

    /// <summary>
    /// Gets the expected tensor shape, if provided
    /// </summary>
    public int[]? ExpectedShape { get; }

    /// <summary>
    /// Gets the actual tensor shape, if provided
    /// </summary>
    public int[]? ActualShape { get; }

    /// <summary>
    /// Gets detailed context information about the shape incompatibility
    /// </summary>
    /// <returns>A formatted string with detailed error context</returns>
    public override string GetDetailedContext()
    {
        var context = new StringBuilder();
        context.AppendLine(base.GetDetailedContext());

        if (ExpectedShape != null)
        {
            context.AppendLine($"Expected Shape: [{string.Join(", ", ExpectedShape)}]");
        }

        if (ActualShape != null)
        {
            context.AppendLine($"Actual Shape: [{string.Join(", ", ActualShape)}]");
        }

        if (ExpectedShape != null && ActualShape != null)
        {
            context.AppendLine("Shape Differences:");
            if (ExpectedShape.Length != ActualShape.Length)
            {
                context.AppendLine($"  • Dimension count mismatch: expected {ExpectedShape.Length}, got {ActualShape.Length}");
            }
            else
            {
                for (int i = 0; i < ExpectedShape.Length; i++)
                {
                    if (ExpectedShape[i] != ActualShape[i])
                    {
                        context.AppendLine($"  • Dimension {i}: expected {ExpectedShape[i]}, got {ActualShape[i]}");
                    }
                }
            }
        }

        return context.ToString();
    }
}
