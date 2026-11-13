using Nivara.Helpers;
using System.Globalization;
using System.Numerics;

namespace Nivara.Expressions;

/// <summary>
/// Binds a column-reference leaf of a fused expression to the input column it evaluates against.
/// </summary>
internal sealed record FusedColumnBinding(ColumnReference Reference, IColumn Column);

/// <summary>
/// Describes a fusable expression: the result element type, whether generic math applies to it,
/// whether any leaf column has nulls, a structural signature used as the compiled-kernel cache key,
/// and the leaf column bindings.
/// </summary>
internal sealed record FusedExpressionPlan(
    Type ResultType,
    bool IsGenericMath,
    bool HasNulls,
    string Signature,
    IReadOnlyList<FusedColumnBinding> Columns);

/// <summary>
/// Infers whether a validated <see cref="ColumnExpression"/> can run through the fused evaluator and,
/// when it can, the unified result type and leaf bindings needed to build the fused kernel.
/// Object-typed columns, null literals, and type mixes that cannot be unified are not fusable.
/// </summary>
internal static class ExpressionTypeInferer
{
    /// <summary>
    /// Attempts to infer a fusable evaluation plan for the expression against the given input columns.
    /// </summary>
    /// <param name="expression">The validated expression to analyze</param>
    /// <param name="input">The input columns the expression evaluates against</param>
    /// <returns>The fused plan, or null when the expression is not fusable</returns>
    public static FusedExpressionPlan? TryInfer(ColumnExpression expression, IReadOnlyDictionary<string, IColumn> input)
    {
        ArgumentNullException.ThrowIfNull(expression);
        ArgumentNullException.ThrowIfNull(input);

        var leaves = new List<FusedColumnBinding>();
        var resultType = InferNode(expression, input, leaves);
        if (resultType == null)
            return null;

        var hasNulls = false;
        foreach (var leaf in leaves)
        {
            if (leaf.Column.HasNulls)
            {
                hasNulls = true;
                break;
            }
        }

        return new FusedExpressionPlan(
            resultType,
            IsGenericMathType(resultType),
            hasNulls,
            BuildSignature(expression, input),
            leaves);
    }

    /// <summary>
    /// Gets whether the type participates in generic math (<see cref="INumber{T}"/>), which is what the
    /// generic node-tree kernel requires. Mirrors the AutoDiff <c>IFloatingPointIeee754&lt;T&gt;</c> domain
    /// validation: float, double, Half, and BFloat16 pass. BFloat16 was admitted to the
    /// AutoDiff domain per issue #137 and is now exercised by the fused expression engine
    /// (the column/query layer gained BFloat16 support in the Phase 2 BFloat16 work).
    /// </summary>
    /// <param name="type">The element type to check</param>
    /// <returns>True when the type implements <see cref="INumber{T}"/></returns>
    public static bool IsGenericMathType(Type type)
    {
        ArgumentNullException.ThrowIfNull(type);
        if (type.IsGenericTypeDefinition || type.ContainsGenericParameters || type.IsPointer || type.IsByRef || type.IsInterface)
            return false;

        return type.GetInterfaces().Any(i =>
            i.IsGenericType &&
            i.GetGenericTypeDefinition() == typeof(INumber<>) &&
            i.GetGenericArguments()[0] == type);
    }

    static Type? InferNode(ColumnExpression node, IReadOnlyDictionary<string, IColumn> input, List<FusedColumnBinding> leaves)
    {
        return node switch
        {
            ColumnReference columnRef => InferColumnReference(columnRef, input, leaves),
            LiteralExpression literal => InferLiteral(literal),
            BinaryExpression binary => InferBinary(binary, input, leaves),
            ScalarExpression scalar => InferScalar(scalar, input, leaves),
            ComparisonExpression comparison => InferComparison(comparison, input, leaves),
            NotExpression not => InferNot(not, input, leaves),
            ConditionalExpression conditional => InferConditional(conditional, input, leaves),
            _ => null
        };
    }

    static Type? InferColumnReference(ColumnReference columnRef, IReadOnlyDictionary<string, IColumn> input, List<FusedColumnBinding> leaves)
    {
        if (!input.TryGetValue(columnRef.ColumnName, out var column))
            return null;

        if (column.ElementType == typeof(object))
            return null;

        leaves.Add(new FusedColumnBinding(columnRef, column));
        return column.ElementType;
    }

    static Type? InferLiteral(LiteralExpression literal)
    {
        if (literal.Value == null)
            return null;

        return literal.Value.GetType();
    }

    static Type? InferBinary(BinaryExpression binary, IReadOnlyDictionary<string, IColumn> input, List<FusedColumnBinding> leaves)
    {
        if (binary.Operator is BinaryOperator.And or BinaryOperator.Or)
        {
            var left = InferNode(binary.Left, input, leaves);
            var right = InferNode(binary.Right, input, leaves);
            return left == typeof(bool) && right == typeof(bool) ? typeof(bool) : null;
        }

        var leftType = InferNode(binary.Left, input, leaves);
        var rightType = InferNode(binary.Right, input, leaves);
        if (leftType == null || rightType == null)
            return null;

        return NumericPromoter.GetPromotedType(leftType, rightType);
    }

    static Type? InferScalar(ScalarExpression scalar, IReadOnlyDictionary<string, IColumn> input, List<FusedColumnBinding> leaves)
    {
        var columnType = InferNode(scalar.Column, input, leaves);
        if (columnType == null || scalar.Scalar == null)
            return null;

        return NumericPromoter.GetPromotedType(columnType, scalar.Scalar.GetType());
    }

    static Type? InferComparison(ComparisonExpression comparison, IReadOnlyDictionary<string, IColumn> input, List<FusedColumnBinding> leaves)
    {
        var leftType = InferNode(comparison.Left, input, leaves);
        var rightType = InferNode(comparison.Right, input, leaves);
        if (leftType == null || rightType == null)
            return null;

        // Numeric pairs unify through C# binary numeric promotion; non-numeric pairs must already be
        // same-typed so a single comparer can order them (string/DateTime/Guid/bool/...).
        if (NumericPromoter.GetPromotedType(leftType, rightType) == null && leftType != rightType)
            return null;

        return typeof(bool);
    }

    static Type? InferNot(NotExpression not, IReadOnlyDictionary<string, IColumn> input, List<FusedColumnBinding> leaves)
    {
        var operandType = InferNode(not.Operand, input, leaves);
        return operandType == typeof(bool) ? typeof(bool) : null;
    }

    static Type? InferConditional(ConditionalExpression conditional, IReadOnlyDictionary<string, IColumn> input, List<FusedColumnBinding> leaves)
    {
        var testType = InferNode(conditional.Test, input, leaves);
        if (testType != typeof(bool))
            return null;

        var trueType = InferNode(conditional.TrueValue, input, leaves);
        var falseType = InferNode(conditional.FalseValue, input, leaves);
        if (trueType == null || falseType == null)
            return null;

        return NumericPromoter.GetPromotedType(trueType, falseType);
    }

    static string BuildSignature(ColumnExpression node, IReadOnlyDictionary<string, IColumn> input)
    {
        return node switch
        {
            ColumnReference columnRef => $"C:{columnRef.ColumnName}:{TypeName(input.TryGetValue(columnRef.ColumnName, out var column) ? column.ElementType : null)}",
            LiteralExpression literal => $"L:{FormatValue(literal.Value)}",
            BinaryExpression binary => $"({BuildSignature(binary.Left, input)} {binary.Operator} {BuildSignature(binary.Right, input)})",
            ScalarExpression scalar => $"({BuildSignature(scalar.Column, input)} {scalar.Operator} {FormatValue(scalar.Scalar)})",
            ComparisonExpression comparison => $"({BuildSignature(comparison.Left, input)} {comparison.Operator} {BuildSignature(comparison.Right, input)})",
            NotExpression not => $"!({BuildSignature(not.Operand, input)})",
            ConditionalExpression conditional => $"({BuildSignature(conditional.Test, input)} ? {BuildSignature(conditional.TrueValue, input)} : {BuildSignature(conditional.FalseValue, input)})",
            _ => node.GetType().Name
        };
    }

    static string TypeName(Type? type) => type?.FullName ?? "?";

    /// <summary>
    /// Formats a literal value into a cache-safe, round-trip-safe, injective signature token.
    /// Numeric components are used in preference to formatted text so that sub-second temporal
    /// precision (and floating-point ULPs) are preserved: the previous <c>IFormattable.ToString(null, …)</c>
    /// path rendered <see cref="DateTime"/>/<see cref="DateTimeOffset"/>/<see cref="TimeOnly"/>
    /// without fractional seconds, which let two literals differing only below one second share a
    /// compiled-delegate cache entry (issue #1169). The text is length-prefixed so that a literal
    /// whose value contains the signature's own delimiters cannot forge another expression's key.
    /// </summary>
    static string FormatValue(object? value)
    {
        if (value == null)
            return "null";

        // Encode the value's *bits* (not its formatted text) wherever that is exact, so the token
        // is round-trip-safe regardless of culture or machine. DateTime/DateTimeOffset encode Kind
        // and offset as ticks rather than via "O" formatting, because "O" for Kind=Local embeds the
        // machine's offset and would make signatures environment-dependent.
        var text = value switch
        {
            DateTime dt => $"{dt.Ticks}:{(int)dt.Kind}",
            DateTimeOffset dto => $"{dto.UtcTicks}:{dto.Offset.Ticks}",
            TimeOnly t => $"{t.Ticks}",
            DateOnly d => $"{d.DayNumber}",
            TimeSpan ts => $"{ts.Ticks}",
            Guid g => g.ToString("D", CultureInfo.InvariantCulture),
            float f => $"{BitConverter.SingleToInt32Bits(f)}",
            double d => $"{BitConverter.DoubleToInt64Bits(d)}",
            Half h => $"{BitConverter.HalfToInt16Bits(h)}",
            decimal m => m.ToString(CultureInfo.InvariantCulture),
            string s => s,
            IFormattable f => f.ToString(null, CultureInfo.InvariantCulture) ?? string.Empty,
            _ => value.ToString() ?? string.Empty,
        };

        // Include the literal runtime type: two literals that stringify identically
        // (0.1f vs 0.1, 1.1m vs 1.1, nint vs int) must not share a signature, or the
        // compiled-delegate cache would reuse the wrong typed kernel (issue #246).
        // The length prefix makes the concatenation injective against delimiter injection.
        return $"{text.Length}:{text}:{value.GetType().FullName}";
    }
}
