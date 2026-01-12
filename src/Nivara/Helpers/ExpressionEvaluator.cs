using Nivara.Exceptions;
using Nivara.Expressions;

namespace Nivara.Helpers;

/// <summary>
/// Evaluates column expressions against input data to produce result columns
/// </summary>
internal sealed class ExpressionEvaluator
{
    /// <summary>
    /// Evaluates a column expression and returns the result column
    /// </summary>
    /// <param name="expression">The expression to evaluate</param>
    /// <param name="input">The input columns</param>
    /// <returns>The result column</returns>
    /// <exception cref="QueryExecutionException">Thrown when evaluation fails</exception>
    public IColumn Evaluate(ColumnExpression expression, IReadOnlyDictionary<string, IColumn> input)
    {
        ArgumentNullException.ThrowIfNull(expression);
        ArgumentNullException.ThrowIfNull(input);

        try
        {
            return expression switch
            {
                ColumnReference columnRef => EvaluateColumnReference(columnRef, input),
                LiteralExpression literal => EvaluateLiteral(literal, input),
                BinaryExpression binary => EvaluateBinaryExpression(binary, input),
                ComparisonExpression comparison => EvaluateComparisonExpression(comparison, input),
                ScalarExpression scalar => EvaluateScalarExpression(scalar, input),
                _ => throw new NotSupportedException($"Expression type {expression.GetType().Name} is not supported")
            };
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Failed to evaluate expression '{expression.Name}': {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Evaluates a column expression that should return a boolean result
    /// </summary>
    /// <param name="expression">The expression to evaluate</param>
    /// <param name="input">The input columns</param>
    /// <returns>A boolean column with the evaluation results</returns>
    /// <exception cref="QueryExecutionException">Thrown when evaluation fails or result is not boolean</exception>
    public NivaraColumn<bool> EvaluateBoolean(ColumnExpression expression, IReadOnlyDictionary<string, IColumn> input)
    {
        var result = Evaluate(expression, input);

        if (result is not NivaraColumn<bool> boolColumn)
        {
            throw new QueryExecutionException($"Expression '{expression.Name}' must evaluate to a boolean column, but got {result.ElementType.Name}");
        }

        return boolColumn;
    }

    /// <summary>
    /// Evaluates a column reference expression
    /// </summary>
    /// <param name="columnRef">The column reference</param>
    /// <param name="input">The input columns</param>
    /// <returns>The referenced column</returns>
    static IColumn EvaluateColumnReference(ColumnReference columnRef, IReadOnlyDictionary<string, IColumn> input)
    {
        if (!input.TryGetValue(columnRef.ColumnName, out var column))
        {
            var availableColumns = string.Join(", ", input.Keys);
            throw new QueryExecutionException($"Column '{columnRef.ColumnName}' not found. Available columns: {availableColumns}");
        }

        return column;
    }

    /// <summary>
    /// Evaluates a literal expression by creating a constant column
    /// </summary>
    /// <param name="literal">The literal expression</param>
    /// <param name="input">The input columns (used to determine result length)</param>
    /// <returns>A constant column with the literal value</returns>
    static IColumn EvaluateLiteral(LiteralExpression literal, IReadOnlyDictionary<string, IColumn> input)
    {
        // Get the length from any input column
        var length = input.Values.FirstOrDefault()?.Length ?? 1;

        // Create a constant column with the literal value
        return CreateConstantColumn(literal.Value, length);
    }

    /// <summary>
    /// Evaluates a binary expression between two column expressions
    /// </summary>
    /// <param name="binary">The binary expression</param>
    /// <param name="input">The input columns</param>
    /// <returns>The result column</returns>
    IColumn EvaluateBinaryExpression(BinaryExpression binary, IReadOnlyDictionary<string, IColumn> input)
    {
        var leftColumn = Evaluate(binary.Left, input);
        var rightColumn = Evaluate(binary.Right, input);

        return binary.Operator switch
        {
            BinaryOperator.Add => ApplyBinaryOperation(leftColumn, rightColumn, (l, r) => AddValues(l, r)),
            BinaryOperator.Subtract => ApplyBinaryOperation(leftColumn, rightColumn, (l, r) => SubtractValues(l, r)),
            BinaryOperator.Multiply => ApplyBinaryOperation(leftColumn, rightColumn, (l, r) => MultiplyValues(l, r)),
            BinaryOperator.Divide => ApplyBinaryOperation(leftColumn, rightColumn, (l, r) => DivideValues(l, r)),
            BinaryOperator.And => ApplyBinaryOperation(leftColumn, rightColumn, (l, r) => AndValues(l, r)),
            BinaryOperator.Or => ApplyBinaryOperation(leftColumn, rightColumn, (l, r) => OrValues(l, r)),
            _ => throw new NotSupportedException($"Binary operator {binary.Operator} is not supported")
        };
    }

    /// <summary>
    /// Evaluates a comparison expression between two column expressions
    /// </summary>
    /// <param name="comparison">The comparison expression</param>
    /// <param name="input">The input columns</param>
    /// <returns>A boolean column with comparison results</returns>
    IColumn EvaluateComparisonExpression(ComparisonExpression comparison, IReadOnlyDictionary<string, IColumn> input)
    {
        var leftColumn = Evaluate(comparison.Left, input);
        var rightColumn = Evaluate(comparison.Right, input);

        return comparison.Operator switch
        {
            ComparisonOperator.Equal => ApplyComparisonOperation(leftColumn, rightColumn, (l, r) => CompareEqual(l, r)),
            ComparisonOperator.NotEqual => ApplyComparisonOperation(leftColumn, rightColumn, (l, r) => !CompareEqual(l, r)),
            ComparisonOperator.GreaterThan => ApplyComparisonOperation(leftColumn, rightColumn, (l, r) => CompareGreaterThan(l, r)),
            ComparisonOperator.LessThan => ApplyComparisonOperation(leftColumn, rightColumn, (l, r) => CompareLessThan(l, r)),
            ComparisonOperator.GreaterThanOrEqual => ApplyComparisonOperation(leftColumn, rightColumn, (l, r) => CompareGreaterThanOrEqual(l, r)),
            ComparisonOperator.LessThanOrEqual => ApplyComparisonOperation(leftColumn, rightColumn, (l, r) => CompareLessThanOrEqual(l, r)),
            _ => throw new NotSupportedException($"Comparison operator {comparison.Operator} is not supported")
        };
    }

    /// <summary>
    /// Evaluates a scalar expression (column with scalar value)
    /// </summary>
    /// <param name="scalar">The scalar expression</param>
    /// <param name="input">The input columns</param>
    /// <returns>The result column</returns>
    IColumn EvaluateScalarExpression(ScalarExpression scalar, IReadOnlyDictionary<string, IColumn> input)
    {
        var column = Evaluate(scalar.Column, input);
        var scalarColumn = CreateConstantColumn(scalar.Scalar, column.Length);

        return scalar.Operator switch
        {
            BinaryOperator.Add => ApplyBinaryOperation(column, scalarColumn, (l, r) => AddValues(l, r)),
            BinaryOperator.Subtract => ApplyBinaryOperation(column, scalarColumn, (l, r) => SubtractValues(l, r)),
            BinaryOperator.Multiply => ApplyBinaryOperation(column, scalarColumn, (l, r) => MultiplyValues(l, r)),
            BinaryOperator.Divide => ApplyBinaryOperation(column, scalarColumn, (l, r) => DivideValues(l, r)),
            BinaryOperator.And => ApplyBinaryOperation(column, scalarColumn, (l, r) => AndValues(l, r)),
            BinaryOperator.Or => ApplyBinaryOperation(column, scalarColumn, (l, r) => OrValues(l, r)),
            _ => throw new NotSupportedException($"Scalar operator {scalar.Operator} is not supported")
        };
    }

    /// <summary>
    /// Creates a constant column with the specified value repeated for the given length
    /// </summary>
    /// <param name="value">The constant value</param>
    /// <param name="length">The length of the column</param>
    /// <returns>A constant column</returns>
    static IColumn CreateConstantColumn(object? value, int length)
    {
        if (value == null)
        {
            // Create a column of nullable objects
            var nullArray = new object?[length];
            return NivaraColumn<object?>.Create(nullArray);
        }

        // Use dynamic dispatch to create the appropriate column type
        return value switch
        {
            int intValue => CreateConstantColumnTyped(intValue, length),
            double doubleValue => CreateConstantColumnTyped(doubleValue, length),
            float floatValue => CreateConstantColumnTyped(floatValue, length),
            long longValue => CreateConstantColumnTyped(longValue, length),
            string stringValue => CreateConstantColumnTyped(stringValue, length),
            bool boolValue => CreateConstantColumnTyped(boolValue, length),
            decimal decimalValue => CreateConstantColumnTyped(decimalValue, length),
            byte byteValue => CreateConstantColumnTyped(byteValue, length),
            short shortValue => CreateConstantColumnTyped(shortValue, length),
            DateTime dateTimeValue => CreateConstantColumnTyped(dateTimeValue, length),
            _ => CreateConstantColumnGeneric(value, length)
        };
    }

    /// <summary>
    /// Creates a constant column for a specific type
    /// </summary>
    static IColumn CreateConstantColumnTyped<T>(T value, int length)
    {
        var array = new T[length];
        Array.Fill(array, value);
        return NivaraColumn<T>.Create(array);
    }

    /// <summary>
    /// Creates a constant column for unknown types using object column
    /// </summary>
    static IColumn CreateConstantColumnGeneric(object value, int length)
    {
        var array = new object[length];
        Array.Fill(array, value);
        return NivaraColumn<object>.Create(array);
    }

    /// <summary>
    /// Applies a binary operation to two columns element-wise
    /// </summary>
    /// <param name="left">The left column</param>
    /// <param name="right">The right column</param>
    /// <param name="operation">The operation to apply</param>
    /// <returns>The result column</returns>
    static IColumn ApplyBinaryOperation(IColumn left, IColumn right, Func<object?, object?, object?> operation)
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Columns must have the same length for binary operations");

        var resultArray = new object?[left.Length];

        for (int i = 0; i < left.Length; i++)
        {
            var leftValue = left.GetValue(i);
            var rightValue = right.GetValue(i);
            resultArray[i] = operation(leftValue, rightValue);
        }

        return NivaraColumn<object?>.Create(resultArray);
    }

    /// <summary>
    /// Applies a comparison operation to two columns element-wise
    /// </summary>
    /// <param name="left">The left column</param>
    /// <param name="right">The right column</param>
    /// <param name="operation">The comparison operation to apply</param>
    /// <returns>A boolean column with comparison results</returns>
    static NivaraColumn<bool> ApplyComparisonOperation(IColumn left, IColumn right, Func<object?, object?, bool> operation)
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Columns must have the same length for comparison operations");

        var resultArray = new bool[left.Length];

        for (int i = 0; i < left.Length; i++)
        {
            var leftValue = left.GetValue(i);
            var rightValue = right.GetValue(i);
            resultArray[i] = operation(leftValue, rightValue);
        }

        return NivaraColumn<bool>.Create(resultArray);
    }

    // Arithmetic operation implementations
    static object? AddValues(object? left, object? right)
    {
        if (left == null || right == null) return null;

        return (left, right) switch
        {
            (int l, int r) => l + r,
            (double l, double r) => l + r,
            (float l, float r) => l + r,
            (long l, long r) => l + r,
            (decimal l, decimal r) => l + r,
            _ => Convert.ToDouble(left) + Convert.ToDouble(right)
        };
    }

    static object? SubtractValues(object? left, object? right)
    {
        if (left == null || right == null) return null;

        return (left, right) switch
        {
            (int l, int r) => l - r,
            (double l, double r) => l - r,
            (float l, float r) => l - r,
            (long l, long r) => l - r,
            (decimal l, decimal r) => l - r,
            _ => Convert.ToDouble(left) - Convert.ToDouble(right)
        };
    }

    static object? MultiplyValues(object? left, object? right)
    {
        if (left == null || right == null) return null;

        return (left, right) switch
        {
            (int l, int r) => l * r,
            (double l, double r) => l * r,
            (float l, float r) => l * r,
            (long l, long r) => l * r,
            (decimal l, decimal r) => l * r,
            _ => Convert.ToDouble(left) * Convert.ToDouble(right)
        };
    }

    static object? DivideValues(object? left, object? right)
    {
        if (left == null || right == null) return null;

        return (left, right) switch
        {
            (int l, int r) => r != 0 ? (double)l / r : throw new DivideByZeroException(),
            (double l, double r) => r != 0 ? l / r : throw new DivideByZeroException(),
            (float l, float r) => r != 0 ? l / r : throw new DivideByZeroException(),
            (long l, long r) => r != 0 ? (double)l / r : throw new DivideByZeroException(),
            (decimal l, decimal r) => r != 0 ? l / r : throw new DivideByZeroException(),
            _ => Convert.ToDouble(right) != 0 ? Convert.ToDouble(left) / Convert.ToDouble(right) : throw new DivideByZeroException()
        };
    }

    static object? AndValues(object? left, object? right)
    {
        if (left == null || right == null) return null;

        return (left, right) switch
        {
            (bool l, bool r) => l && r,
            _ => Convert.ToBoolean(left) && Convert.ToBoolean(right)
        };
    }

    static object? OrValues(object? left, object? right)
    {
        if (left == null || right == null) return null;

        return (left, right) switch
        {
            (bool l, bool r) => l || r,
            _ => Convert.ToBoolean(left) || Convert.ToBoolean(right)
        };
    }

    // Comparison operation implementations
    static bool CompareEqual(object? left, object? right)
    {
        if (left == null && right == null) return true;
        if (left == null || right == null) return false;

        return left.Equals(right);
    }

    static bool CompareGreaterThan(object? left, object? right)
    {
        if (left == null || right == null) return false;

        if (left is IComparable leftComparable)
            return leftComparable.CompareTo(right) > 0;

        throw new InvalidOperationException($"Cannot compare values of type {left.GetType().Name}");
    }

    static bool CompareLessThan(object? left, object? right)
    {
        if (left == null || right == null) return false;

        if (left is IComparable leftComparable)
            return leftComparable.CompareTo(right) < 0;

        throw new InvalidOperationException($"Cannot compare values of type {left.GetType().Name}");
    }

    static bool CompareGreaterThanOrEqual(object? left, object? right)
    {
        if (left == null || right == null) return false;

        if (left is IComparable leftComparable)
            return leftComparable.CompareTo(right) >= 0;

        throw new InvalidOperationException($"Cannot compare values of type {left.GetType().Name}");
    }

    static bool CompareLessThanOrEqual(object? left, object? right)
    {
        if (left == null || right == null) return false;

        if (left is IComparable leftComparable)
            return leftComparable.CompareTo(right) <= 0;

        throw new InvalidOperationException($"Cannot compare values of type {left.GetType().Name}");
    }
}
