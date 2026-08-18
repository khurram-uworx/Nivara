using Nivara.Expressions;

namespace Nivara.IO;

/// <summary>
/// Evaluates <see cref="ColumnExpression"/> filter predicates against
/// row-group column statistics (min/max) to determine which
/// row groups can be safely skipped.
/// </summary>
internal static class RowGroupFilterEvaluator
{
    internal static bool CanEvaluate(ColumnExpression expression, Schema sourceSchema)
    {
        return expression switch
        {
            ComparisonExpression cmp => IsPushdownEligibleComparison(cmp, sourceSchema),
            BinaryExpression binary when binary.Operator == BinaryOperator.And =>
                CanEvaluate(binary.Left, sourceSchema) && CanEvaluate(binary.Right, sourceSchema),
            _ => false
        };
    }

    /// <summary>
    /// Returns true if the row group CAN contain matching rows.
    /// Conservative: returns true for any un-evaluable sub-expression.
    /// </summary>
    internal static bool EvaluateRowGroup(
        ColumnExpression expression,
        Func<string, RowGroupColumnStats?> statsProvider,
        Schema sourceSchema)
    {
        return expression switch
        {
            ComparisonExpression cmp => EvaluateComparison(cmp, statsProvider, sourceSchema),
            BinaryExpression binary when binary.Operator == BinaryOperator.And =>
                EvaluateRowGroup(binary.Left, statsProvider, sourceSchema)
                && EvaluateRowGroup(binary.Right, statsProvider, sourceSchema),
            _ => true
        };
    }

    static bool IsPushdownEligibleComparison(ComparisonExpression cmp, Schema sourceSchema)
    {
        return cmp.Left is ColumnReference leftCol
            && cmp.Right is LiteralExpression
            && sourceSchema.HasColumn(leftCol.ColumnName);
    }

    static bool EvaluateComparison(
        ComparisonExpression cmp,
        Func<string, RowGroupColumnStats?> statsProvider,
        Schema sourceSchema)
    {
        if (cmp.Left is not ColumnReference leftCol || cmp.Right is not LiteralExpression literal)
            return true;

        var columnName = leftCol.ColumnName;
        var stats = statsProvider(columnName);
        if (stats == null)
            return true;

        var literalValue = literal.Value;
        if (literalValue == null)
            return true;

        var columnType = sourceSchema.GetColumnType(columnName);
        var underlyingType = Nullable.GetUnderlyingType(columnType) ?? columnType;

        try
        {
            if (stats.MinValue == null && stats.MaxValue == null)
                return true;

            return cmp.Operator switch
            {
                ComparisonOperator.GreaterThan => stats.MinValue != null && CompareValues(literalValue, stats.MinValue, underlyingType) < 0,
                ComparisonOperator.GreaterThanOrEqual => stats.MinValue != null && CompareValues(literalValue, stats.MinValue, underlyingType) <= 0,
                ComparisonOperator.LessThan => stats.MaxValue != null && CompareValues(literalValue, stats.MaxValue, underlyingType) > 0,
                ComparisonOperator.LessThanOrEqual => stats.MaxValue != null && CompareValues(literalValue, stats.MaxValue, underlyingType) >= 0,
                ComparisonOperator.Equal => stats.MinValue != null && stats.MaxValue != null
                    && CompareValues(literalValue, stats.MinValue, underlyingType) >= 0
                    && CompareValues(literalValue, stats.MaxValue, underlyingType) <= 0,
                ComparisonOperator.NotEqual => true,
                _ => true
            };
        }
        catch
        {
            return true;
        }
    }

    static int CompareValues(object a, object b, Type targetType)
    {
        var convertedA = Convert.ChangeType(a, targetType);
        var convertedB = Convert.ChangeType(b, targetType);

        if (convertedA is IComparable comparableA)
            return comparableA.CompareTo(convertedB);

        return string.Compare(convertedA?.ToString(), convertedB?.ToString(), StringComparison.Ordinal);
    }

    internal sealed class RowGroupColumnStats
    {
        public object? MinValue { get; init; }
        public object? MaxValue { get; init; }
    }
}
