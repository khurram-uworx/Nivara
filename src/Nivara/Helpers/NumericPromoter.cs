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

        var numericTypes = TypeCompatibilityValidator.GetNumericTypes();
        if (!numericTypes.Contains(left) || !numericTypes.Contains(right))
            return null;

        if (left == right)
            return IsSmallIntegralType(left) ? typeof(int) : left;

        // Half implicitly converts to float/double; it has no implicit conversion to/from
        // integrals or decimal, so promote those pairs to double (safe superset).
        if (left == typeof(Half) || right == typeof(Half))
        {
            var other = left == typeof(Half) ? right : left;
            return other == typeof(float) || other == typeof(double) ? other : typeof(double);
        }

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

        // C# rule 7: uint + byte/ushort/char promotes to uint.
        if (left == typeof(uint) || right == typeof(uint))
        {
            var other = left == typeof(uint) ? right : left;
            return other == typeof(byte) || other == typeof(ushort) || other == typeof(char)
                ? typeof(uint)
                : typeof(long);
        }

        // Remaining small integral pairs promote to int.
        return typeof(int);
    }

    static bool IsSmallIntegralType(Type type)
    {
        // C# spec §12.4.7.3 rule 1: when both operands share one of these types, the
        // promoted type is int (the operands are converted to int before the operation).
        return type == typeof(sbyte)
            || type == typeof(byte)
            || type == typeof(short)
            || type == typeof(ushort)
            || type == typeof(char);
    }
}
