using System.Text;

namespace Nivara.Extensions.AutoDiff.Exceptions;

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
/// Exception thrown when gradient computation fails during backward pass.
/// Provides information about the failing operation and computation graph node.
/// </summary>
public sealed class GradientComputationException : AutoGradException
{
    /// <summary>
    /// Initializes a new instance of GradientComputationException
    /// </summary>
    /// <param name="message">The error message</param>
    public GradientComputationException(string message) : base(message)
    {
    }

    /// <summary>
    /// Initializes a new instance of GradientComputationException with an inner exception
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="innerException">The inner exception that caused the gradient computation to fail</param>
    public GradientComputationException(string message, Exception innerException)
        : base(message, innerException)
    {
    }

    /// <summary>
    /// Initializes a new instance of GradientComputationException with operation details
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="failedOperation">The name of the operation that failed</param>
    /// <param name="operationContext">Additional context about the operation</param>
    public GradientComputationException(string message, string failedOperation, string? operationContext = null)
        : base(message, operationContext)
    {
        FailedOperation = failedOperation;
    }

    /// <summary>
    /// Initializes a new instance of GradientComputationException with full context
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="failedOperation">The name of the operation that failed</param>
    /// <param name="operationContext">Additional context about the operation</param>
    /// <param name="innerException">The inner exception that caused the gradient computation to fail</param>
    public GradientComputationException(string message, string failedOperation, string? operationContext, Exception innerException)
        : base(message, operationContext, innerException)
    {
        FailedOperation = failedOperation;
    }

    /// <summary>
    /// Gets the name of the operation that failed during gradient computation
    /// </summary>
    public string? FailedOperation { get; }

    /// <summary>
    /// Gets the number of inputs to the failing operation, if available
    /// </summary>
    public int? FailingNodeInputCount { get; init; }

    /// <summary>
    /// Gets detailed context information about the gradient computation failure
    /// </summary>
    /// <returns>A formatted string with detailed error context</returns>
    public override string GetDetailedContext()
    {
        var context = new StringBuilder();
        context.AppendLine(base.GetDetailedContext());

        if (!string.IsNullOrEmpty(FailedOperation))
        {
            context.AppendLine($"Failed Operation: {FailedOperation}");
        }

        if (FailingNodeInputCount.HasValue)
        {
            context.AppendLine($"Node Inputs: {FailingNodeInputCount.Value}");
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

/// <summary>
/// Exception thrown when circular dependencies are detected in the computation graph.
/// Provides information about the cycle path for debugging.
/// </summary>
public sealed class CircularDependencyException : AutoGradException
{
    /// <summary>
    /// Initializes a new instance of CircularDependencyException
    /// </summary>
    /// <param name="message">The error message</param>
    public CircularDependencyException(string message) : base(message)
    {
    }

    /// <summary>
    /// Initializes a new instance of CircularDependencyException with cycle information
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="cycleOperationNames">The names of operations forming the circular dependency</param>
    public CircularDependencyException(string message, IReadOnlyList<string> cycleOperationNames)
        : base(message, "Computation Graph Validation")
    {
        CycleOperationNames = cycleOperationNames;
    }

    /// <summary>
    /// Initializes a new instance of CircularDependencyException with cycle information and inner exception
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="cycleOperationNames">The names of operations forming the circular dependency</param>
    /// <param name="innerException">The inner exception that led to detecting the circular dependency</param>
    public CircularDependencyException(string message, IReadOnlyList<string> cycleOperationNames, Exception innerException)
        : base(message, "Computation Graph Validation", innerException)
    {
        CycleOperationNames = cycleOperationNames;
    }

    /// <summary>
    /// Gets the names of operations forming the circular dependency
    /// </summary>
    public IReadOnlyList<string>? CycleOperationNames { get; }

    /// <summary>
    /// Gets detailed context information about the circular dependency
    /// </summary>
    /// <returns>A formatted string with detailed error context</returns>
    public override string GetDetailedContext()
    {
        var context = new StringBuilder();
        context.AppendLine(base.GetDetailedContext());

        if (CycleOperationNames != null && CycleOperationNames.Count > 0)
        {
            context.AppendLine("Circular Dependency Path:");
            for (int i = 0; i < CycleOperationNames.Count; i++)
            {
                context.AppendLine($"  {i + 1}. {CycleOperationNames[i]}");
            }
            context.AppendLine($"  → Back to: {CycleOperationNames[0]}");
        }

        return context.ToString();
    }
}

/// <summary>
/// Exception thrown when backward() is called on an invalid tensor.
/// Provides guidance on proper usage of the backward pass.
/// </summary>
public sealed class InvalidBackwardCallException : AutoGradException
{
    /// <summary>
    /// Initializes a new instance of InvalidBackwardCallException
    /// </summary>
    /// <param name="message">The error message</param>
    public InvalidBackwardCallException(string message) : base(message)
    {
    }

    /// <summary>
    /// Initializes a new instance of InvalidBackwardCallException with tensor details
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="tensorShape">The shape of the tensor on which backward was called</param>
    /// <param name="requiresGrad">Whether the tensor has requiresGrad enabled</param>
    public InvalidBackwardCallException(string message, int[] tensorShape, bool requiresGrad)
        : base(message, "Backward Pass")
    {
        TensorShape = tensorShape;
        RequiresGrad = requiresGrad;
        InvolvedShapes = new[] { tensorShape };
    }

    /// <summary>
    /// Gets the shape of the tensor on which backward was called
    /// </summary>
    public int[]? TensorShape { get; }

    /// <summary>
    /// Gets whether the tensor has requiresGrad enabled
    /// </summary>
    public bool RequiresGrad { get; }

    /// <summary>
    /// Gets detailed context information about the invalid backward call
    /// </summary>
    /// <returns>A formatted string with detailed error context</returns>
    public override string GetDetailedContext()
    {
        var context = new StringBuilder();
        context.AppendLine(base.GetDetailedContext());

        if (TensorShape != null)
        {
            context.AppendLine($"Tensor Shape: [{string.Join(", ", TensorShape)}]");
            context.AppendLine($"Is Scalar: {TensorShape.Length == 1 && TensorShape[0] == 1}");
        }

        context.AppendLine($"Requires Grad: {RequiresGrad}");

        context.AppendLine("\nGuidance:");
        context.AppendLine("  • backward() can only be called on scalar tensors (shape [1])");
        context.AppendLine("  • The tensor must have requiresGrad=true");
        context.AppendLine("  • For non-scalar tensors, use backward(gradient) with an explicit gradient");

        return context.ToString();
    }
}

/// <summary>
/// Exception thrown when type validation fails for automatic differentiation operations.
/// Provides information about expected vs actual types.
/// </summary>
public sealed class TypeValidationException : AutoGradException
{
    /// <summary>
    /// Initializes a new instance of TypeValidationException
    /// </summary>
    /// <param name="message">The error message</param>
    public TypeValidationException(string message) : base(message)
    {
    }

    /// <summary>
    /// Initializes a new instance of TypeValidationException with type details
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="operationName">The name of the operation that encountered type validation failure</param>
    /// <param name="expectedType">The expected type</param>
    /// <param name="actualType">The actual type</param>
    public TypeValidationException(string message, string operationName, Type expectedType, Type actualType)
        : base(message, operationName)
    {
        ExpectedType = expectedType;
        ActualType = actualType;
    }

    /// <summary>
    /// Initializes a new instance of TypeValidationException with inner exception
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="operationName">The name of the operation that encountered type validation failure</param>
    /// <param name="expectedType">The expected type</param>
    /// <param name="actualType">The actual type</param>
    /// <param name="innerException">The inner exception that caused the type validation failure</param>
    public TypeValidationException(string message, string operationName, Type expectedType, Type actualType, Exception innerException)
        : base(message, operationName, innerException)
    {
        ExpectedType = expectedType;
        ActualType = actualType;
    }

    /// <summary>
    /// Gets the expected type
    /// </summary>
    public Type? ExpectedType { get; }

    /// <summary>
    /// Gets the actual type
    /// </summary>
    public Type? ActualType { get; }

    /// <summary>
    /// Gets detailed context information about the type validation failure
    /// </summary>
    /// <returns>A formatted string with detailed error context</returns>
    public override string GetDetailedContext()
    {
        var context = new StringBuilder();
        context.AppendLine(base.GetDetailedContext());

        if (ExpectedType != null)
        {
            context.AppendLine($"Expected Type: {ExpectedType.Name}");
        }

        if (ActualType != null)
        {
            context.AppendLine($"Actual Type: {ActualType.Name}");
        }

        return context.ToString();
    }
}