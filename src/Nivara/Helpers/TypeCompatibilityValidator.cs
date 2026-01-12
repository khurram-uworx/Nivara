using Nivara.Exceptions;

namespace Nivara.Helpers;

/// <summary>
/// Provides comprehensive type compatibility validation for operations and expressions.
/// Ensures type safety across all query operations and frame manipulations.
/// </summary>
public static class TypeCompatibilityValidator
{
    /// <summary>
    /// Validates that two types are compatible for arithmetic operations
    /// </summary>
    /// <param name="leftType">The left operand type</param>
    /// <param name="rightType">The right operand type</param>
    /// <param name="operationName">The name of the operation for error messages</param>
    /// <exception cref="ColumnTypeMismatchException">Thrown when types are incompatible</exception>
    public static void ValidateArithmeticCompatibility(Type leftType, Type rightType, string operationName)
    {
        if (leftType == null)
            throw new ArgumentNullException(nameof(leftType));
        if (rightType == null)
            throw new ArgumentNullException(nameof(rightType));

        if (!AreArithmeticCompatible(leftType, rightType))
        {
            throw new ColumnTypeMismatchException(
                operationName,
                GetCompatibleTypes(leftType).FirstOrDefault() ?? leftType,
                rightType);
        }
    }

    /// <summary>
    /// Validates that a type supports comparison operations
    /// </summary>
    /// <param name="type">The type to validate</param>
    /// <param name="operationName">The name of the operation for error messages</param>
    /// <exception cref="ColumnTypeMismatchException">Thrown when type doesn't support comparison</exception>
    public static void ValidateComparisonSupport(Type type, string operationName)
    {
        if (type == null)
            throw new ArgumentNullException(nameof(type));

        if (!SupportsComparison(type))
        {
            var supportedTypes = GetComparisonSupportedTypes();
            throw new ColumnTypeMismatchException(
                operationName,
                supportedTypes.FirstOrDefault() ?? typeof(object),
                type);
        }
    }

    /// <summary>
    /// Validates that two types are compatible for comparison operations
    /// </summary>
    /// <param name="leftType">The left operand type</param>
    /// <param name="rightType">The right operand type</param>
    /// <param name="operationName">The name of the operation for error messages</param>
    /// <exception cref="ColumnTypeMismatchException">Thrown when types are incompatible</exception>
    public static void ValidateComparisonCompatibility(Type leftType, Type rightType, string operationName)
    {
        if (leftType == null)
            throw new ArgumentNullException(nameof(leftType));
        if (rightType == null)
            throw new ArgumentNullException(nameof(rightType));

        ValidateComparisonSupport(leftType, operationName);
        ValidateComparisonSupport(rightType, operationName);

        if (!AreComparisonCompatible(leftType, rightType))
        {
            throw new ColumnTypeMismatchException(
                operationName,
                leftType,
                rightType);
        }
    }

    /// <summary>
    /// Validates that a column type is supported for the specified operation
    /// </summary>
    /// <param name="columnType">The column type to validate</param>
    /// <param name="operationName">The name of the operation</param>
    /// <param name="supportedTypes">The types supported by the operation</param>
    /// <exception cref="ColumnTypeMismatchException">Thrown when type is not supported</exception>
    public static void ValidateOperationSupport(Type columnType, string operationName, IEnumerable<Type> supportedTypes)
    {
        if (columnType == null)
            throw new ArgumentNullException(nameof(columnType));
        if (supportedTypes == null)
            throw new ArgumentNullException(nameof(supportedTypes));

        var supportedTypesList = supportedTypes.ToList();

        if (!supportedTypesList.Contains(columnType) && !supportedTypesList.Any(t => AreTypesCompatible(columnType, t)))
        {
            var supportedTypeNames = string.Join(", ", supportedTypesList.Select(t => t.Name));
            throw new ColumnTypeMismatchException(
                operationName,
                supportedTypesList.FirstOrDefault() ?? typeof(object),
                columnType);
        }
    }

    /// <summary>
    /// Validates that all columns in a frame have compatible types for the specified operation
    /// </summary>
    /// <param name="frame">The frame to validate</param>
    /// <param name="operationName">The name of the operation</param>
    /// <param name="requiredCompatibility">The type of compatibility required</param>
    /// <exception cref="SchemaValidationException">Thrown when frame has incompatible types</exception>
    public static void ValidateFrameTypeCompatibility(NivaraFrame frame, string operationName, TypeCompatibilityRequirement requiredCompatibility)
    {
        if (frame == null)
            throw new ArgumentNullException(nameof(frame));

        var schema = frame.Schema;
        var columnTypes = schema.ColumnTypes.Values.ToList();

        switch (requiredCompatibility)
        {
            case TypeCompatibilityRequirement.AllSameType:
                ValidateAllSameType(columnTypes, operationName);
                break;
            case TypeCompatibilityRequirement.AllNumeric:
                ValidateAllNumeric(columnTypes, operationName);
                break;
            case TypeCompatibilityRequirement.AllComparable:
                ValidateAllComparable(columnTypes, operationName);
                break;
            case TypeCompatibilityRequirement.AllArithmeticCompatible:
                ValidateAllArithmeticCompatible(columnTypes, operationName);
                break;
        }
    }

    /// <summary>
    /// Checks if two types are compatible for arithmetic operations
    /// </summary>
    /// <param name="type1">The first type</param>
    /// <param name="type2">The second type</param>
    /// <returns>True if types are arithmetic compatible</returns>
    public static bool AreArithmeticCompatible(Type type1, Type type2)
    {
        if (type1 == type2)
            return true;

        var numericTypes = GetNumericTypes();
        return numericTypes.Contains(type1) && numericTypes.Contains(type2);
    }

    /// <summary>
    /// Checks if two types are compatible for comparison operations
    /// </summary>
    /// <param name="type1">The first type</param>
    /// <param name="type2">The second type</param>
    /// <returns>True if types are comparison compatible</returns>
    public static bool AreComparisonCompatible(Type type1, Type type2)
    {
        if (type1 == type2)
            return true;

        // Numeric types can be compared with each other
        var numericTypes = GetNumericTypes();
        if (numericTypes.Contains(type1) && numericTypes.Contains(type2))
            return true;

        // String comparisons
        if (type1 == typeof(string) && type2 == typeof(string))
            return true;

        // DateTime comparisons
        if (type1 == typeof(DateTime) && type2 == typeof(DateTime))
            return true;

        return false;
    }

    /// <summary>
    /// Checks if two types are compatible (general compatibility)
    /// </summary>
    /// <param name="type1">The first type</param>
    /// <param name="type2">The second type</param>
    /// <returns>True if types are compatible</returns>
    public static bool AreTypesCompatible(Type type1, Type type2)
    {
        if (type1 == type2)
            return true;

        // Numeric type compatibility
        var numericTypes = GetNumericTypes();
        return numericTypes.Contains(type1) && numericTypes.Contains(type2);
    }

    /// <summary>
    /// Checks if a type supports comparison operations
    /// </summary>
    /// <param name="type">The type to check</param>
    /// <returns>True if the type supports comparison</returns>
    public static bool SupportsComparison(Type type)
    {
        // All numeric types support comparison
        if (GetNumericTypes().Contains(type))
            return true;

        // String supports comparison
        if (type == typeof(string))
            return true;

        // DateTime supports comparison
        if (type == typeof(DateTime) || type == typeof(DateTimeOffset))
            return true;

        // Boolean supports equality comparison
        if (type == typeof(bool))
            return true;

        // Check if type implements IComparable
        return typeof(IComparable).IsAssignableFrom(type) ||
               type.GetInterfaces().Any(i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(IComparable<>));
    }

    /// <summary>
    /// Gets the numeric types supported for arithmetic operations
    /// </summary>
    /// <returns>Collection of numeric types</returns>
    public static IReadOnlyList<Type> GetNumericTypes()
    {
        return new[]
        {
            typeof(byte), typeof(sbyte), typeof(short), typeof(ushort),
            typeof(int), typeof(uint), typeof(long), typeof(ulong),
            typeof(float), typeof(double), typeof(decimal)
        };
    }

    /// <summary>
    /// Gets the types that support comparison operations
    /// </summary>
    /// <returns>Collection of comparison-supported types</returns>
    public static IReadOnlyList<Type> GetComparisonSupportedTypes()
    {
        var numericTypes = GetNumericTypes();
        var additionalTypes = new[] { typeof(string), typeof(DateTime), typeof(DateTimeOffset), typeof(bool) };
        return numericTypes.Concat(additionalTypes).ToList();
    }

    /// <summary>
    /// Gets compatible types for the specified type
    /// </summary>
    /// <param name="type">The type to find compatible types for</param>
    /// <returns>Collection of compatible types</returns>
    public static IReadOnlyList<Type> GetCompatibleTypes(Type type)
    {
        if (GetNumericTypes().Contains(type))
            return GetNumericTypes();

        return new[] { type };
    }

    private static void ValidateAllSameType(IList<Type> types, string operationName)
    {
        if (types.Count <= 1)
            return;

        var firstType = types[0];
        for (int i = 1; i < types.Count; i++)
        {
            if (types[i] != firstType)
            {
                throw new SchemaValidationException(
                    $"Operation '{operationName}' requires all columns to have the same type. " +
                    $"Found {firstType.Name} and {types[i].Name}.");
            }
        }
    }

    private static void ValidateAllNumeric(IList<Type> types, string operationName)
    {
        var numericTypes = GetNumericTypes();

        foreach (var type in types)
        {
            if (!numericTypes.Contains(type))
            {
                var supportedTypeNames = string.Join(", ", numericTypes.Select(t => t.Name));
                throw new SchemaValidationException(
                    $"Operation '{operationName}' requires all columns to be numeric types. " +
                    $"Column type {type.Name} is not supported. Supported types: {supportedTypeNames}.");
            }
        }
    }

    private static void ValidateAllComparable(IList<Type> types, string operationName)
    {
        foreach (var type in types)
        {
            if (!SupportsComparison(type))
            {
                var supportedTypes = GetComparisonSupportedTypes();
                var supportedTypeNames = string.Join(", ", supportedTypes.Select(t => t.Name));
                throw new SchemaValidationException(
                    $"Operation '{operationName}' requires all columns to support comparison operations. " +
                    $"Column type {type.Name} is not supported. Supported types: {supportedTypeNames}.");
            }
        }
    }

    private static void ValidateAllArithmeticCompatible(IList<Type> types, string operationName)
    {
        if (types.Count <= 1)
            return;

        var firstType = types[0];
        for (int i = 1; i < types.Count; i++)
        {
            if (!AreArithmeticCompatible(firstType, types[i]))
            {
                throw new SchemaValidationException(
                    $"Operation '{operationName}' requires all columns to have arithmetic-compatible types. " +
                    $"Types {firstType.Name} and {types[i].Name} are not compatible.");
            }
        }
    }
}

/// <summary>
/// Defines the type compatibility requirements for operations
/// </summary>
public enum TypeCompatibilityRequirement
{
    /// <summary>
    /// All columns must have exactly the same type
    /// </summary>
    AllSameType,

    /// <summary>
    /// All columns must be numeric types
    /// </summary>
    AllNumeric,

    /// <summary>
    /// All columns must support comparison operations
    /// </summary>
    AllComparable,

    /// <summary>
    /// All columns must be arithmetic-compatible with each other
    /// </summary>
    AllArithmeticCompatible
}