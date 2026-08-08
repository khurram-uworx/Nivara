namespace Nivara.Helpers;

/// <summary>
/// Computes the common promoted type for mixed numeric operand pairs using
/// C# binary numeric promotion rules (C# spec §12.4.7.3). The promoted type is
/// the result type of arithmetic on the pair and the type the operator runs in.
/// </summary>
internal static class NumericPromoter
{
    /// <summary>
    /// Gets the promoted type for a binary numeric operand pair, or null when the
    /// pair is not numerically promotable.
    /// </summary>
    /// <param name="left">The left operand type</param>
    /// <param name="right">The right operand type</param>
    /// <returns>The promoted result type, or null when either type is non-numeric</returns>
    public static Type? GetPromotedType(Type left, Type right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        if (left == right)
            return left;

        var numericTypes = TypeCompatibilityValidator.GetNumericTypes();
        if (!numericTypes.Contains(left) || !numericTypes.Contains(right))
            return null;

        // C# rule 1: decimal wins over integrals. float/double has no implicit
        // conversion to decimal (a binding-time error in C#); resolve to double.
        if (left == typeof(decimal) || right == typeof(decimal))
        {
            var other = left == typeof(decimal) ? right : left;
            return other == typeof(float) || other == typeof(double)
                ? typeof(double)
                : typeof(decimal);
        }

        // C# rule 2/3: floating types dominate.
        if (left == typeof(double) || right == typeof(double))
            return typeof(double);

        if (left == typeof(float) || right == typeof(float))
            return typeof(float);

        // C# rule 4: ulong dominates unsigned, but is rejected for signed types
        // (a binding-time error in C#); resolve signed pairs to double.
        if (left == typeof(ulong) || right == typeof(ulong))
        {
            var other = left == typeof(ulong) ? right : left;
            return other == typeof(sbyte) || other == typeof(short) || other == typeof(int) || other == typeof(long)
                ? typeof(double)
                : typeof(ulong);
        }

        // C# rule 5: long dominates signed integrals and uint (uint implicitly
        // converts to long).
        if (left == typeof(long) || right == typeof(long))
            return typeof(long);

        // C# rule 6: uint + signed integral promotes both to long.
        if (left == typeof(uint) || right == typeof(uint))
            return typeof(long);

        // Remaining small integral pairs promote to int.
        return typeof(int);
    }
}
