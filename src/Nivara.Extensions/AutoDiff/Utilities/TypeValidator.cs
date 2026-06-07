using Nivara.Extensions.AutoDiff.Exceptions;
using System.Numerics;

namespace Nivara.Extensions.AutoDiff.Utilities;

/// <summary>
/// Provides compile-time and runtime type validation for automatic differentiation operations.
/// Ensures that only supported numeric types are used with GradTensor operations.
/// </summary>
public static class TypeValidator
{
    /// <summary>
    /// Validates that the specified type is supported for automatic differentiation.
    /// </summary>
    /// <typeparam name="T">The type to validate</typeparam>
    /// <exception cref="AutoGradException">Thrown when the type is not supported for automatic differentiation</exception>
    public static void ValidateNumericType<T>() where T : struct, INumber<T>
    {
        var type = typeof(T);

        if (!IsSupportedType(type))
        {
            throw new TypeValidationException(
                $"Type {type.Name} is not supported for automatic differentiation. " +
                $"Supported types are: float, double. " +
                $"For other numeric types, consider converting to float or double first.",
                "GradTensor constructor",
                typeof(float), // expected (representative)
                type);         // actual
        }
    }

    /// <summary>
    /// Checks if the specified type is supported for automatic differentiation.
    /// </summary>
    /// <typeparam name="T">The type to check</typeparam>
    /// <returns>True if the type is supported; otherwise, false</returns>
    public static bool IsSupported<T>() where T : struct, INumber<T>
    {
        return IsSupportedType(typeof(T));
    }

    /// <summary>
    /// Checks if the specified type is supported for automatic differentiation.
    /// </summary>
    /// <param name="type">The type to check</param>
    /// <returns>True if the type is supported; otherwise, false</returns>
    public static bool IsSupportedType(Type type)
    {
        // Currently, we support float and double for automatic differentiation
        // These types have well-defined gradient semantics and are commonly used in ML
        return type == typeof(float) || type == typeof(double);
    }

    /// <summary>
    /// Gets a list of all supported types for automatic differentiation.
    /// </summary>
    /// <returns>An array of supported types</returns>
    public static Type[] GetSupportedTypes()
    {
        return new[] { typeof(float), typeof(double) };
    }

    /// <summary>
    /// Validates that two tensors have compatible types for operations.
    /// </summary>
    /// <typeparam name="T">The type of the first tensor</typeparam>
    /// <typeparam name="U">The type of the second tensor</typeparam>
    /// <exception cref="AutoGradException">Thrown when types are incompatible</exception>
    public static void ValidateCompatibleTypes<T, U>()
        where T : struct, INumber<T>
        where U : struct, INumber<U>
    {
        if (typeof(T) != typeof(U))
        {
            throw new AutoGradException(
                $"Type mismatch: cannot perform operations between tensors of type {typeof(T).Name} and {typeof(U).Name}. " +
                $"Both tensors must have the same numeric type.");
        }
    }

    /// <summary>
    /// Validates that a tensor's shape is compatible with the expected shape for an operation.
    /// </summary>
    /// <param name="actualLength">The actual length of the tensor</param>
    /// <param name="expectedLength">The expected length of the tensor</param>
    /// <param name="operationName">The name of the operation being performed</param>
    /// <exception cref="ShapeIncompatibilityException">Thrown when shapes are incompatible</exception>
    public static void ValidateShapeCompatibility(int actualLength, int expectedLength, string operationName)
    {
        if (actualLength != expectedLength)
        {
            throw new ShapeIncompatibilityException(
                $"Shape mismatch in {operationName}: expected length {expectedLength}, but got {actualLength}",
                operationName,
                new[] { expectedLength },
                new[] { actualLength });
        }
    }

    /// <summary>
    /// Validates that a tensor is scalar (length 1) for operations that require scalar inputs.
    /// </summary>
    /// <param name="length">The length of the tensor</param>
    /// <param name="operationName">The name of the operation being performed</param>
    /// <exception cref="AutoGradException">Thrown when the tensor is not scalar</exception>
    public static void ValidateScalar(int length, string operationName)
    {
        if (length != 1)
        {
            throw new AutoGradException(
                $"{operationName} requires a scalar tensor (length=1), but got tensor with length={length}");
        }
    }

    /// <summary>
    /// Validates that a tensor is non-empty for operations that require non-empty inputs.
    /// </summary>
    /// <param name="length">The length of the tensor</param>
    /// <param name="operationName">The name of the operation being performed</param>
    /// <exception cref="AutoGradException">Thrown when the tensor is empty</exception>
    public static void ValidateNonEmpty(int length, string operationName)
    {
        if (length == 0)
        {
            throw new AutoGradException(
                $"{operationName} cannot be performed on empty tensors");
        }
    }
}
