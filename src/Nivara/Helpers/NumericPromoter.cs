namespace Nivara.Helpers;

/// <summary>
/// Computes the common promoted type for mixed numeric operand pairs using
/// C# binary numeric promotion rules (C# spec §12.4.7.3). The promoted type is
/// the result type of arithmetic on the pair and the type the operator runs in.
/// Pairs that are binding-time errors in C# (e.g. <c>long + ulong</c>,
/// <c>nint + ulong</c>, <c>UInt128 + int</c>, <c>Half + int</c>) resolve to the
/// safe superset <c>double</c>, so magnitude is never lost silently.
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
        // Int128/UInt128 likewise cannot convert to decimal, and decimal cannot hold
        // their full range, so those pairs also resolve to double (safe superset).
        if (left == typeof(decimal) || right == typeof(decimal))
        {
            var other = left == typeof(decimal) ? right : left;
            return other == typeof(float) || other == typeof(double)
                || other == typeof(Int128) || other == typeof(UInt128)
                ? typeof(double)
                : typeof(decimal);
        }

        // Native-size (nint/nuint) and 128-bit (Int128/UInt128) integers follow C#
        // implicit-conversion search (§10.4.7.3): the widest target both operands can
        // convert to, or double (safe superset) when no common implicit target exists.
        if (IsNativeOrWide(left) || IsNativeOrWide(right))
        {
            if (left == typeof(Int128) || right == typeof(Int128))
            {
                var other = left == typeof(Int128) ? right : left;
                return other == typeof(UInt128) || other == typeof(float) || other == typeof(double)
                    ? typeof(double)
                    : typeof(Int128);
            }

            if (left == typeof(UInt128) || right == typeof(UInt128))
            {
                var other = left == typeof(UInt128) ? right : left;
                return other == typeof(Int128)
                    || other == typeof(float) || other == typeof(double)
                    || other == typeof(sbyte) || other == typeof(short) || other == typeof(int)
                    || other == typeof(long) || other == typeof(nint)
                    ? typeof(double)
                    : typeof(UInt128);
            }

            if (left == typeof(nint) || right == typeof(nint))
            {
                var other = left == typeof(nint) ? right : left;
                if (other == typeof(nuint) || other == typeof(ulong))
                    return typeof(double);
                if (other == typeof(uint) || other == typeof(long))
                    return typeof(long);
                return other == typeof(byte) || other == typeof(sbyte) || other == typeof(ushort)
                    || other == typeof(short) || other == typeof(char) || other == typeof(int)
                    ? typeof(nint)
                    : other;
            }

            var otherOperand = left == typeof(nuint) ? right : left;
            if (otherOperand == typeof(nint) || otherOperand == typeof(sbyte) || otherOperand == typeof(short)
                || otherOperand == typeof(int) || otherOperand == typeof(long))
                return typeof(double);
            if (otherOperand == typeof(byte) || otherOperand == typeof(ushort) || otherOperand == typeof(char)
                || otherOperand == typeof(uint))
                return typeof(nuint);
            return otherOperand == typeof(ulong) ? typeof(ulong) : otherOperand;
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

    static bool IsNativeOrWide(Type type)
    {
        // nint/nuint (native-size integers) and Int128/UInt128 participate in binary
        // numeric promotion through implicit-conversion search rather than the fixed
        // integral ladder, so they get their own arms (issue #250).
        return type == typeof(nint)
            || type == typeof(nuint)
            || type == typeof(Int128)
            || type == typeof(UInt128);
    }
}
