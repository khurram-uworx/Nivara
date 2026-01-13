namespace Nivara.Extensions.AutoDiff.Exceptions;

/// <summary>
/// Forward declarations for automatic differentiation exception hierarchy.
/// These will be fully implemented in later tasks.
/// </summary>

/// <summary>
/// Base exception for automatic differentiation errors
/// </summary>
public class AutoGradException : Exception
{
    // Implementation will be added in task 9.1
}

/// <summary>
/// Exception thrown when gradient computation fails
/// </summary>
public class GradientComputationException : AutoGradException
{
    // Implementation will be added in task 9.1
}

/// <summary>
/// Exception thrown when tensor shapes are incompatible for operations
/// </summary>
public class ShapeIncompatibilityException : AutoGradException
{
    // Implementation will be added in task 9.1
}

/// <summary>
/// Exception thrown when circular dependencies are detected in computation graph
/// </summary>
public class CircularDependencyException : AutoGradException
{
    // Implementation will be added in task 9.1
}