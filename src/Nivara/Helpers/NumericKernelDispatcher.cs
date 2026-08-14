using System.Collections.Concurrent;
using System.Numerics;
using System.Reflection;

namespace Nivara.Helpers;

/// <summary>
/// Centralized numeric kernel dispatch for <see cref="Nivara.NivaraColumn{T}"/> and
/// <see cref="Nivara.NivaraSeries{T}"/>. Replaces the per-method <c>typeof(T)</c> chains
/// that repeated the full numeric domain in every arithmetic/comparison helper.
/// <typeparamref name="T"/> on the entry points is unconstrained, so each (type, operation,
/// shape) is resolved once via <c>MakeGenericMethod</c> over a constrained generic builder
/// and cached as a typed delegate. Delegates are built (not invoked) through reflection, so
/// no <see cref="Span{T}"/> ever crosses a <see cref="MethodInfo.Invoke"/> boundary.
/// </summary>
static class NumericKernelDispatcher
{
    enum Operation
    {
        Add,
        Subtract,
        Multiply,
        Divide,
        Sum,
        Min,
        Max,
        DivideByCount,
        Equals,
        GreaterThan,
        LessThan
    }

    enum Shape
    {
        Span,
        ScalarRight,
        ScalarLeft
    }

    static readonly ConcurrentDictionary<(Type Type, Operation Op, Shape Shape), Delegate> cache = new();

    static readonly HashSet<Type> arithmeticDomain =
    [
        typeof(float), typeof(double), typeof(int), typeof(long), typeof(short),
        typeof(ushort), typeof(uint), typeof(ulong), typeof(byte), typeof(sbyte),
        typeof(char), typeof(decimal), typeof(Half), typeof(nint), typeof(nuint),
        typeof(Int128), typeof(UInt128)
    ];

    static readonly HashSet<Type> comparisonDomain =
    [
        typeof(float), typeof(double), typeof(int), typeof(long), typeof(short),
        typeof(ushort), typeof(uint), typeof(ulong), typeof(byte), typeof(sbyte),
        typeof(char)
    ];

    static string arithmeticMessage(Type type)
        => $"Arithmetic on type {type.Name} is not supported by the typed kernel dispatch";

    static string sumMessage(Type type)
        => $"Sum on type {type.Name} is not supported by the typed kernel dispatch";

    static string minMessage(Type type)
        => $"Min on type {type.Name} is not supported by the typed kernel dispatch";

    static string maxMessage(Type type)
        => $"Max on type {type.Name} is not supported by the typed kernel dispatch";

    static string averageMessage(Type type)
        => $"Average on type {type.Name} is not supported by the typed kernel dispatch";

    public static void Add<T>(ReadOnlySpan<T> x, T y, Span<T> destination)
        => ((Action<ReadOnlySpan<T>, T, Span<T>>)getArithmetic(typeof(T), Operation.Add, Shape.ScalarRight, arithmeticMessage))(x, y, destination);

    public static void Add<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
        => ((Action<ReadOnlySpan<T>, ReadOnlySpan<T>, Span<T>>)getArithmetic(typeof(T), Operation.Add, Shape.Span, arithmeticMessage))(x, y, destination);

    public static void Subtract<T>(ReadOnlySpan<T> x, T y, Span<T> destination)
        => ((Action<ReadOnlySpan<T>, T, Span<T>>)getArithmetic(typeof(T), Operation.Subtract, Shape.ScalarRight, arithmeticMessage))(x, y, destination);

    public static void Subtract<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
        => ((Action<ReadOnlySpan<T>, ReadOnlySpan<T>, Span<T>>)getArithmetic(typeof(T), Operation.Subtract, Shape.Span, arithmeticMessage))(x, y, destination);

    public static void SubtractFrom<T>(T scalar, ReadOnlySpan<T> x, Span<T> destination)
        => ((Action<T, ReadOnlySpan<T>, Span<T>>)getArithmetic(typeof(T), Operation.Subtract, Shape.ScalarLeft, arithmeticMessage))(scalar, x, destination);

    public static void Multiply<T>(ReadOnlySpan<T> x, T y, Span<T> destination)
        => ((Action<ReadOnlySpan<T>, T, Span<T>>)getArithmetic(typeof(T), Operation.Multiply, Shape.ScalarRight, arithmeticMessage))(x, y, destination);

    public static void Multiply<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
        => ((Action<ReadOnlySpan<T>, ReadOnlySpan<T>, Span<T>>)getArithmetic(typeof(T), Operation.Multiply, Shape.Span, arithmeticMessage))(x, y, destination);

    public static void Divide<T>(ReadOnlySpan<T> x, T y, Span<T> destination)
        => ((Action<ReadOnlySpan<T>, T, Span<T>>)getArithmetic(typeof(T), Operation.Divide, Shape.ScalarRight, arithmeticMessage))(x, y, destination);

    public static void Divide<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
        => ((Action<ReadOnlySpan<T>, ReadOnlySpan<T>, Span<T>>)getArithmetic(typeof(T), Operation.Divide, Shape.Span, arithmeticMessage))(x, y, destination);

    public static void DivideBy<T>(T scalar, ReadOnlySpan<T> x, Span<T> destination)
        => ((Action<T, ReadOnlySpan<T>, Span<T>>)getArithmetic(typeof(T), Operation.Divide, Shape.ScalarLeft, arithmeticMessage))(scalar, x, destination);

    public static T Sum<T>(ReadOnlySpan<T> values)
        => ((Func<ReadOnlySpan<T>, T>)getArithmetic(typeof(T), Operation.Sum, Shape.Span, sumMessage))(values);

    public static T Min<T>(ReadOnlySpan<T> values)
        => ((Func<ReadOnlySpan<T>, T>)getArithmetic(typeof(T), Operation.Min, Shape.Span, minMessage))(values);

    public static T Max<T>(ReadOnlySpan<T> values)
        => ((Func<ReadOnlySpan<T>, T>)getArithmetic(typeof(T), Operation.Max, Shape.Span, maxMessage))(values);

    public static T DivideByCount<T>(T sum, int count)
        => ((Func<T, int, T>)getArithmetic(typeof(T), Operation.DivideByCount, Shape.Span, averageMessage))(sum, count);

    public static bool TryEquals<T>(ReadOnlySpan<T> x, T y, Span<bool> destination)
        => tryInvokeComparison<T>(Operation.Equals, Shape.ScalarRight, x, y, destination);

    public static bool TryEquals<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
        => tryInvokeComparison<T>(Operation.Equals, Shape.Span, x, y, destination);

    public static bool TryGreaterThan<T>(ReadOnlySpan<T> x, T y, Span<bool> destination)
        => tryInvokeComparison<T>(Operation.GreaterThan, Shape.ScalarRight, x, y, destination);

    public static bool TryGreaterThan<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
        => tryInvokeComparison<T>(Operation.GreaterThan, Shape.Span, x, y, destination);

    public static bool TryLessThan<T>(ReadOnlySpan<T> x, T y, Span<bool> destination)
        => tryInvokeComparison<T>(Operation.LessThan, Shape.ScalarRight, x, y, destination);

    public static bool TryLessThan<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
        => tryInvokeComparison<T>(Operation.LessThan, Shape.Span, x, y, destination);

    static bool tryInvokeComparison<T>(Operation op, Shape shape, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
    {
        if (!tryGetComparison(typeof(T), op, shape, out var kernel))
            return false;

        ((Action<ReadOnlySpan<T>, ReadOnlySpan<T>, Span<bool>>)kernel)(x, y, destination);
        return true;
    }

    static bool tryInvokeComparison<T>(Operation op, Shape shape, ReadOnlySpan<T> x, T y, Span<bool> destination)
    {
        if (!tryGetComparison(typeof(T), op, shape, out var kernel))
            return false;

        ((Action<ReadOnlySpan<T>, T, Span<bool>>)kernel)(x, y, destination);
        return true;
    }

    static Delegate getArithmetic(Type type, Operation op, Shape shape, Func<Type, string> message)
    {
        if (!arithmeticDomain.Contains(type))
            throw new NotSupportedException(message(type));

        return cache.GetOrAdd((type, op, shape), static key => buildArithmetic(key.Type, key.Op, key.Shape));
    }

    static bool tryGetComparison(Type type, Operation op, Shape shape, out Delegate kernel)
    {
        if (!comparisonDomain.Contains(type))
        {
            kernel = null!;
            return false;
        }

        kernel = cache.GetOrAdd((type, op, shape), static key => buildComparison(key.Type, key.Op, key.Shape));
        return true;
    }

    static Delegate buildArithmetic(Type type, Operation op, Shape shape)
        => build(buildArithmeticMethod, type, op, shape);

    static Delegate buildComparison(Type type, Operation op, Shape shape)
        => build(buildComparisonMethod, type, op, shape);

    static Delegate build(MethodInfo builder, Type type, Operation op, Shape shape)
        => (Delegate)builder.MakeGenericMethod(type).Invoke(null, new object?[] { op, shape })!;

    static readonly MethodInfo buildArithmeticMethod = typeof(NumericKernelDispatcher)
        .GetMethod(nameof(createArithmetic), BindingFlags.NonPublic | BindingFlags.Static)!;

    static readonly MethodInfo buildComparisonMethod = typeof(NumericKernelDispatcher)
        .GetMethod(nameof(createComparison), BindingFlags.NonPublic | BindingFlags.Static)!;

    static Delegate createArithmetic<U>(Operation op, Shape shape)
        where U : INumber<U>
        => op switch
        {
            Operation.Add => shape switch
            {
                Shape.Span => new Action<ReadOnlySpan<U>, ReadOnlySpan<U>, Span<U>>(NumericTensorKernels<U>.Add),
                Shape.ScalarRight => new Action<ReadOnlySpan<U>, U, Span<U>>(NumericTensorKernels<U>.Add),
                _ => throw new ArgumentOutOfRangeException(nameof(shape)),
            },
            Operation.Subtract => shape switch
            {
                Shape.Span => new Action<ReadOnlySpan<U>, ReadOnlySpan<U>, Span<U>>(NumericTensorKernels<U>.Subtract),
                Shape.ScalarRight => new Action<ReadOnlySpan<U>, U, Span<U>>(NumericTensorKernels<U>.Subtract),
                Shape.ScalarLeft => new Action<U, ReadOnlySpan<U>, Span<U>>(NumericTensorKernels<U>.SubtractFrom),
                _ => throw new ArgumentOutOfRangeException(nameof(shape)),
            },
            Operation.Multiply => shape switch
            {
                Shape.Span => new Action<ReadOnlySpan<U>, ReadOnlySpan<U>, Span<U>>(NumericTensorKernels<U>.Multiply),
                Shape.ScalarRight => new Action<ReadOnlySpan<U>, U, Span<U>>(NumericTensorKernels<U>.Multiply),
                _ => throw new ArgumentOutOfRangeException(nameof(shape)),
            },
            Operation.Divide => shape switch
            {
                Shape.Span => new Action<ReadOnlySpan<U>, ReadOnlySpan<U>, Span<U>>(NumericTensorKernels<U>.Divide),
                Shape.ScalarRight => new Action<ReadOnlySpan<U>, U, Span<U>>(NumericTensorKernels<U>.Divide),
                Shape.ScalarLeft => new Action<U, ReadOnlySpan<U>, Span<U>>(NumericTensorKernels<U>.DivideBy),
                _ => throw new ArgumentOutOfRangeException(nameof(shape)),
            },
            Operation.Sum => new Func<ReadOnlySpan<U>, U>(NumericTensorKernels<U>.Sum),
            Operation.Min => new Func<ReadOnlySpan<U>, U>(NumericTensorKernels<U>.Min),
            Operation.Max => new Func<ReadOnlySpan<U>, U>(NumericTensorKernels<U>.Max),
            Operation.DivideByCount => new Func<U, int, U>(NumericTensorKernels<U>.DivideByCount),
            _ => throw new ArgumentOutOfRangeException(nameof(op)),
        };

    static Delegate createComparison<U>(Operation op, Shape shape)
        where U : INumber<U>
        => op switch
        {
            Operation.Equals => shape switch
            {
                Shape.Span => new Action<ReadOnlySpan<U>, ReadOnlySpan<U>, Span<bool>>(NumericTensorKernels<U>.Equals),
                Shape.ScalarRight => new Action<ReadOnlySpan<U>, U, Span<bool>>(NumericTensorKernels<U>.Equals),
                _ => throw new ArgumentOutOfRangeException(nameof(shape)),
            },
            Operation.GreaterThan => shape switch
            {
                Shape.Span => new Action<ReadOnlySpan<U>, ReadOnlySpan<U>, Span<bool>>(NumericTensorKernels<U>.GreaterThan),
                Shape.ScalarRight => new Action<ReadOnlySpan<U>, U, Span<bool>>(NumericTensorKernels<U>.GreaterThan),
                _ => throw new ArgumentOutOfRangeException(nameof(shape)),
            },
            Operation.LessThan => shape switch
            {
                Shape.Span => new Action<ReadOnlySpan<U>, ReadOnlySpan<U>, Span<bool>>(NumericTensorKernels<U>.LessThan),
                Shape.ScalarRight => new Action<ReadOnlySpan<U>, U, Span<bool>>(NumericTensorKernels<U>.LessThan),
                _ => throw new ArgumentOutOfRangeException(nameof(shape)),
            },
            _ => throw new ArgumentOutOfRangeException(nameof(op)),
        };
}
