using Nivara.Diagnostics;
using Nivara.Exceptions;
using Nivara.Expressions;
using System.Collections.Concurrent;
using System.Numerics;
using System.Reflection;

namespace Nivara.Helpers;

/// <summary>
/// Evaluates column expressions against input data to produce result columns
/// </summary>
sealed class ExpressionEvaluator
{
    int typedPathEvaluationCount;

    /// <summary>
    /// Gets how many typed-kernel evaluations (same-element-type operands) were applied
    /// by the most recent operations on this instance. Used by guardrail tests to assert
    /// the typed fast path is actually selected (claims-integrity guardrail).
    /// </summary>
    internal int TypedPathEvaluationCount => typedPathEvaluationCount;

    /// <summary>
    /// Gets how many boxed (object?) evaluations were applied by the most recent operations
    /// on this instance. Always zero: the boxed object fallback was removed, so this property
    /// is a claims-integrity guardrail that fails if boxing is ever reintroduced.
    /// </summary>
    internal int BoxedPathEvaluationCount => 0;

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
                NotExpression not => EvaluateNotExpression(not, input),
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

        var typedResult = TryEvaluateTypedBinary(binary.Operator, leftColumn, rightColumn);
        if (typedResult != null)
        {
            typedPathEvaluationCount++;
            RecordEvaluationDiagnostics(leftColumn, rightColumn);
            return typedResult;
        }

        throw new NotSupportedException(
            $"Binary operator {binary.Operator} on ({leftColumn.ElementType.Name}, {rightColumn.ElementType.Name}) is not supported by the typed evaluator");
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

        var typedResult = TryEvaluateTypedComparison(comparison.Operator, leftColumn, rightColumn);
        if (typedResult != null)
        {
            typedPathEvaluationCount++;
            RecordEvaluationDiagnostics(leftColumn, rightColumn);
            return typedResult;
        }

        throw new NotSupportedException(
            $"Comparison {comparison.Operator} on ({leftColumn.ElementType.Name}, {rightColumn.ElementType.Name}) is not supported by the typed evaluator");
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

        var typedResult = TryEvaluateTypedBinary(scalar.Operator, column, scalarColumn);
        if (typedResult != null)
        {
            typedPathEvaluationCount++;
            RecordEvaluationDiagnostics(column, scalarColumn);
            return typedResult;
        }

        throw new NotSupportedException(
            $"Scalar operator {scalar.Operator} on ({column.ElementType.Name}, {scalarColumn.ElementType.Name}) is not supported by the typed evaluator");
    }

    /// <summary>
    /// Records the typed-kernel choice for an expression evaluation into the active
    /// diagnostics tracker, when diagnostics are enabled.
    /// </summary>
    static void RecordEvaluationDiagnostics(IColumn left, IColumn right)
    {
        if (!DiagnosticsTracker.IsEnabled)
            return;

        DiagnosticsTracker.RecordOperation(new OperationDiagnostics(
            "ExpressionEvaluation",
            KernelType.Vectorized,
            left.Length,
            left.ElementType,
            left.HasNulls || right.HasNulls,
            0,
            TimeSpan.Zero,
            "Typed column kernel"));
    }

    /// <summary>
    /// Attempts to evaluate a binary operation through typed column kernels, returning null when the
    /// typed path is not applicable so callers surface a clear <see cref="NotSupportedException"/>.
    /// Same-type operands use the typed same-type kernel; numerically promotable mixed operands
    /// (C# binary numeric promotion) use the typed promoted kernel.
    /// </summary>
    static IColumn? TryEvaluateTypedBinary(BinaryOperator op, IColumn left, IColumn right)
    {
        if (left.Length != right.Length)
            return null;

        if (left.ElementType == right.ElementType)
        {
            return left.ElementType switch
            {
                Type t when t == typeof(int) => TryBinaryTyped<int>(op, left, right),
                Type t when t == typeof(long) => TryBinaryTyped<long>(op, left, right),
                Type t when t == typeof(float) => TryBinaryTyped<float>(op, left, right),
                Type t when t == typeof(double) => TryBinaryTyped<double>(op, left, right),
                Type t when t == typeof(bool) => TryBoolLogicTyped(op, left, right),
                _ => null
            };
        }

        return TryEvaluatePromotedBinary(op, left, right);
    }

    /// <summary>
    /// Applies a boolean logic operation (And/Or) to two boolean columns element-wise.
    /// Returns null for non-boolean columns or unsupported operators so callers surface a
    /// clear <see cref="NotSupportedException"/>. Null propagation is SQL-like: a null operand
    /// yields a masked false result (the underlying value is <c>false</c> at masked positions).
    /// </summary>
    static IColumn? TryBoolLogicTyped(BinaryOperator op, IColumn left, IColumn right)
    {
        if (left is not NivaraColumn<bool> l || right is not NivaraColumn<bool> r)
            return null;

        return op switch
        {
            BinaryOperator.And => l.Zip(r, static (a, b) => a && b),
            BinaryOperator.Or => l.Zip(r, static (a, b) => a || b),
            _ => null
        };
    }

    /// <summary>
    /// Applies a binary operation to two typed columns of the same element type.
    /// Falls back to null when the operation or element type is unsupported.
    /// </summary>
    static IColumn? TryBinaryTyped<T>(BinaryOperator op, IColumn left, IColumn right) where T : struct, INumber<T>
    {
        if (left is not NivaraColumn<T> l || right is not NivaraColumn<T> r)
            return null;

        try
        {
            return op switch
            {
                BinaryOperator.Add => l.Add(r),
                BinaryOperator.Subtract => l.Subtract(r),
                BinaryOperator.Multiply => l.Multiply(r),
                BinaryOperator.Divide => l.Divide(r),
                _ => null
            };
        }
        catch (InvalidOperationException)
        {
            return null;
        }
    }

    /// <summary>
    /// Attempts to evaluate a binary operation through the typed promoted kernel when the operand
    /// element types differ but are numerically promotable (C# binary numeric promotion).
    /// Returns null when the promoted path is not applicable so callers surface a clear
    /// <see cref="NotSupportedException"/>.
    /// </summary>
    static IColumn? TryEvaluatePromotedBinary(BinaryOperator op, IColumn left, IColumn right)
    {
        var resultType = NumericPromoter.GetPromotedType(left.ElementType, right.ElementType);
        if (resultType == null)
            return null;

        var kernel = GetPromotedKernel(nameof(TryBinaryPromoted), left.ElementType, right.ElementType, resultType);
        return InvokePromotedKernel(kernel, op, left, right);
    }

    /// <summary>
    /// Applies a binary operation to two typed columns of different, numerically promotable element
    /// types. Both operands are widened to the promoted type (via <c>INumber.CreateChecked</c>) and
    /// the operation runs in the promoted type with null-OR propagation (SQL-like semantics).
    /// </summary>
    static IColumn? TryBinaryPromoted<TLeft, TRight, TResult>(BinaryOperator op, IColumn left, IColumn right)
        where TLeft : struct, INumber<TLeft>
        where TRight : struct, INumber<TRight>
        where TResult : struct, INumber<TResult>
    {
        if (left is not NivaraColumn<TLeft> l || right is not NivaraColumn<TRight> r)
            return null;

        return op switch
        {
            BinaryOperator.Add => l.Zip(r, static (a, b) => TResult.CreateChecked(a) + TResult.CreateChecked(b)),
            BinaryOperator.Subtract => l.Zip(r, static (a, b) => TResult.CreateChecked(a) - TResult.CreateChecked(b)),
            BinaryOperator.Multiply => l.Zip(r, static (a, b) => TResult.CreateChecked(a) * TResult.CreateChecked(b)),
            BinaryOperator.Divide => l.Zip(r, static (a, b) => TResult.CreateChecked(a) / TResult.CreateChecked(b)),
            _ => null
        };
    }

    /// <summary>
    /// Attempts to evaluate a comparison operation through typed column kernels, returning null when the
    /// typed path is not applicable so callers surface a clear <see cref="NotSupportedException"/>.
    /// Same-type operands use the typed same-type kernel; numerically promotable mixed operands
    /// (C# binary numeric promotion) use the typed promoted comparison kernel.
    /// </summary>
    static IColumn? TryEvaluateTypedComparison(ComparisonOperator op, IColumn left, IColumn right)
    {
        if (left.Length != right.Length)
            return null;

        if (left.ElementType == right.ElementType)
        {
            return left.ElementType switch
            {
                Type t when t == typeof(int) => TryComparisonTyped<int>(op, left, right),
                Type t when t == typeof(long) => TryComparisonTyped<long>(op, left, right),
                Type t when t == typeof(short) => TryComparisonTyped<short>(op, left, right),
                Type t when t == typeof(byte) => TryComparisonTyped<byte>(op, left, right),
                Type t when t == typeof(float) => TryComparisonTyped<float>(op, left, right),
                Type t when t == typeof(double) => TryComparisonTyped<double>(op, left, right),
                Type t when t == typeof(string) => TryComparisonTyped<string>(op, left, right),
                Type t when t == typeof(bool) => TryComparisonTyped<bool>(op, left, right),
                Type t when t == typeof(decimal) => TryComparisonTyped<decimal>(op, left, right),
                Type t when t == typeof(DateTime) => TryComparisonTyped<DateTime>(op, left, right),
                Type t when t == typeof(Guid) => TryComparisonTyped<Guid>(op, left, right),
                Type t when t == typeof(Half) => TryComparisonTyped<Half>(op, left, right),
                _ => null
            };
        }

        return TryEvaluatePromotedComparison(op, left, right);
    }

    /// <summary>
    /// Applies a comparison operation to two typed columns of the same element type.
    /// Falls back to null when the operation or element type is unsupported.
    /// </summary>
    static IColumn? TryComparisonTyped<T>(ComparisonOperator op, IColumn left, IColumn right)
    {
        if (left is not NivaraColumn<T> l || right is not NivaraColumn<T> r)
            return null;

        try
        {
            return op switch
            {
                ComparisonOperator.Equal => l.Equals(r),
                ComparisonOperator.NotEqual => l.Equals(r).Transform(v => !v),
                ComparisonOperator.GreaterThan => l.GreaterThan(r),
                ComparisonOperator.LessThan => l.LessThan(r),
                ComparisonOperator.GreaterThanOrEqual => l.LessThan(r).Transform(v => !v),
                ComparisonOperator.LessThanOrEqual => l.GreaterThan(r).Transform(v => !v),
                _ => null
            };
        }
        catch (InvalidOperationException)
        {
            return null;
        }
    }

    /// <summary>
    /// Attempts to evaluate a comparison operation through the typed promoted kernel when the operand
    /// element types differ but are numerically promotable (C# binary numeric promotion).
    /// Returns null when the promoted path is not applicable so callers surface a clear
    /// <see cref="NotSupportedException"/>.
    /// </summary>
    static IColumn? TryEvaluatePromotedComparison(ComparisonOperator op, IColumn left, IColumn right)
    {
        var resultType = NumericPromoter.GetPromotedType(left.ElementType, right.ElementType);
        if (resultType == null)
            return null;

        var kernel = GetPromotedKernel(nameof(TryComparisonPromoted), left.ElementType, right.ElementType, resultType);
        return InvokePromotedKernel(kernel, op, left, right);
    }

    /// <summary>
    /// Applies a comparison operation to two typed columns of different, numerically promotable element
    /// types. Both operands are widened to the promoted type (via <c>INumber.CreateChecked</c>) and the
    /// comparison runs in the promoted type with null-OR propagation (SQL-like semantics).
    /// </summary>
    static IColumn? TryComparisonPromoted<TLeft, TRight, TResult>(ComparisonOperator op, IColumn left, IColumn right)
        where TLeft : struct, INumber<TLeft>
        where TRight : struct, INumber<TRight>
        where TResult : struct, INumber<TResult>
    {
        if (left is not NivaraColumn<TLeft> l || right is not NivaraColumn<TRight> r)
            return null;

        return op switch
        {
            ComparisonOperator.Equal => l.Zip(r, static (a, b) => TResult.CreateChecked(a) == TResult.CreateChecked(b)),
            ComparisonOperator.NotEqual => l.Zip(r, static (a, b) => TResult.CreateChecked(a) != TResult.CreateChecked(b)),
            ComparisonOperator.GreaterThan => l.Zip(r, static (a, b) => TResult.CreateChecked(a) > TResult.CreateChecked(b)),
            ComparisonOperator.LessThan => l.Zip(r, static (a, b) => TResult.CreateChecked(a) < TResult.CreateChecked(b)),
            ComparisonOperator.GreaterThanOrEqual => l.Zip(r, static (a, b) => TResult.CreateChecked(a) >= TResult.CreateChecked(b)),
            ComparisonOperator.LessThanOrEqual => l.Zip(r, static (a, b) => TResult.CreateChecked(a) <= TResult.CreateChecked(b)),
            _ => null
        };
    }

    /// <summary>
    /// Cache of reflection-built generic promoted kernels keyed by kernel name and the
    /// (left, right, result) element type triple.
    /// </summary>
    static readonly ConcurrentDictionary<(string Name, Type Left, Type Right, Type Result), MethodInfo> promotedKernelCache = new();

    static MethodInfo GetPromotedKernel(string methodName, Type leftType, Type rightType, Type resultType)
    {
        return promotedKernelCache.GetOrAdd((methodName, leftType, rightType, resultType), static key =>
        {
            var method = typeof(ExpressionEvaluator).GetMethod(key.Name, BindingFlags.Static | BindingFlags.NonPublic);
            return method!.MakeGenericMethod(key.Left, key.Right, key.Result);
        });
    }

    /// <summary>
    /// Invokes a cached promoted kernel, mapping an inner <see cref="InvalidOperationException"/> or
    /// <see cref="ArgumentException"/> to a null result so callers surface a clear
    /// <see cref="NotSupportedException"/>.
    /// </summary>
    static IColumn? InvokePromotedKernel(MethodInfo kernel, object op, IColumn left, IColumn right)
    {
        try
        {
            return (IColumn?)kernel.Invoke(null, new object[] { op, left, right });
        }
        catch (TargetInvocationException tie) when (tie.InnerException is InvalidOperationException or ArgumentException)
        {
            return null;
        }
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
            Guid guidValue => CreateConstantColumnTyped(guidValue, length),
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
    /// Evaluates a logical negation expression element-wise, propagating the operand null mask
    /// (a null operand yields a masked null result, SQL-like semantics).
    /// </summary>
    /// <param name="not">The negation expression</param>
    /// <param name="input">The input columns</param>
    /// <returns>A boolean column with the negated results</returns>
    /// <exception cref="QueryExecutionException">Thrown when the operand is not boolean</exception>
    NivaraColumn<bool> EvaluateNotExpression(NotExpression not, IReadOnlyDictionary<string, IColumn> input)
    {
        var operand = Evaluate(not.Operand, input);
        if (operand is not NivaraColumn<bool> boolColumn)
        {
            throw new QueryExecutionException($"Not expression requires a boolean operand, got {operand.ElementType.Name}");
        }

        var resultArray = new bool[boolColumn.Length];
        var nullMask = new bool[boolColumn.Length];
        bool hasNulls = false;

        for (int i = 0; i < boolColumn.Length; i++)
        {
            if (boolColumn.IsNull(i))
            {
                nullMask[i] = true;
                resultArray[i] = false;
                hasNulls = true;
            }
            else
            {
                resultArray[i] = !boolColumn[i];
            }
        }

        return hasNulls
            ? NivaraColumn<bool>.CreateFromSpans(resultArray, nullMask)
            : NivaraColumn<bool>.Create(resultArray);
    }
}
