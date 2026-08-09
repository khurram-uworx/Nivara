using Nivara.Extensions;
using Nivara.Helpers;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara;

/// <summary>
/// Abstract base class for aggregation functions that can be applied to grouped data
/// </summary>
public abstract class AggregationFunction
{
    /// <summary>
    /// Gets the name of the aggregation function
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Gets the result type for the given input type
    /// </summary>
    /// <param name="inputType">The input column type</param>
    /// <returns>The result type after aggregation</returns>
    public abstract Type GetResultType(Type inputType);

    /// <summary>
    /// Applies the aggregation function to a column for a specific group
    /// </summary>
    /// <param name="column">The source column</param>
    /// <param name="groupIndices">The indices of rows in this group</param>
    /// <returns>The aggregated value</returns>
    public abstract object? Apply(IColumn column, IReadOnlyList<int> groupIndices);

    /// <summary>
    /// Applies the aggregation function to multiple groups and returns a column of results
    /// </summary>
    /// <param name="column">The source column</param>
    /// <param name="groups">The groups with their indices</param>
    /// <returns>A column containing the aggregated values for each group</returns>
    public virtual IColumn ApplyToGroups(IColumn column, IEnumerable<(GroupKey Key, IReadOnlyList<int> Indices)> groups)
    {
        if (column == null)
            throw new ArgumentNullException(nameof(column));
        if (groups == null)
            throw new ArgumentNullException(nameof(groups));

        var groupList = groups.ToList();
        var resultType = GetResultType(column.ElementType);
        var results = new object?[groupList.Count];

        for (int i = 0; i < groupList.Count; i++)
        {
            results[i] = Apply(column, groupList[i].Indices);
        }

        return CreateColumnFromValues(resultType, results);
    }

    /// <summary>
    /// Creates a column from an array of values with proper type handling
    /// </summary>
    /// <param name="elementType">The element type</param>
    /// <param name="values">The values</param>
    /// <returns>A new column</returns>
    protected static IColumn CreateColumnFromValues(Type elementType, object?[] values)
    {
        return elementType switch
        {
            Type t when t == typeof(int) => NivaraColumn<int>.Create(values.Cast<int>().ToArray()),
            Type t when t == typeof(double) => NivaraColumn<double>.Create(values.Cast<double>().ToArray()),
            Type t when t == typeof(float) => NivaraColumn<float>.Create(values.Cast<float>().ToArray()),
            Type t when t == typeof(long) => NivaraColumn<long>.Create(values.Cast<long>().ToArray()),
            Type t when t == typeof(string) => NivaraColumn<string>.Create(values.Cast<string>().ToArray()),
            Type t when t == typeof(bool) => NivaraColumn<bool>.Create(values.Cast<bool>().ToArray()),
            Type t when t == typeof(decimal) => NivaraColumn<decimal>.Create(values.Cast<decimal>().ToArray()),
            Type t when t == typeof(ulong) => NivaraColumn<ulong>.Create(values.Cast<ulong>().ToArray()),
            Type t when t == typeof(Int128) => NivaraColumn<Int128>.Create(values.Cast<Int128>().ToArray()),
            Type t when t == typeof(UInt128) => NivaraColumn<UInt128>.Create(values.Cast<UInt128>().ToArray()),
            Type t when t == typeof(byte) => NivaraColumn<byte>.Create(values.Cast<byte>().ToArray()),
            Type t when t == typeof(short) => NivaraColumn<short>.Create(values.Cast<short>().ToArray()),
            Type t when t == typeof(DateTime) => NivaraColumn<DateTime>.Create(values.Cast<DateTime>().ToArray()),
            _ => NivaraColumn<object>.Create(values.Where(v => v != null).ToArray()!)
        };
    }

    /// <summary>
    /// Validates that the input type is supported by this aggregation function
    /// </summary>
    /// <param name="inputType">The input type to validate</param>
    /// <exception cref="ArgumentException">Thrown when the input type is not supported</exception>
    protected virtual void ValidateInputType(Type inputType)
    {
        // Default implementation allows all types
        // Derived classes can override to restrict supported types
    }

    /// <summary>
    /// Helper method to check if a type supports comparison operations
    /// </summary>
    protected static bool IsComparableType(Type type)
    {
        // All numeric types support comparison
        if (type.IsNumericType())
            return true;

        // String supports comparison
        if (type == typeof(string))
            return true;

        // DateTime and other common comparable types
        if (type == typeof(DateTime) || type == typeof(DateTimeOffset) || type == typeof(TimeSpan))
            return true;

        // Guid supports comparison
        if (type == typeof(Guid))
            return true;

        // Check if type implements IComparable<T> or IComparable
        return typeof(IComparable<>).MakeGenericType(type).IsAssignableFrom(type) ||
               typeof(IComparable).IsAssignableFrom(type);
    }

    /// <inheritdoc />
    public override string ToString() => Name;
}

/// <summary>
/// Count aggregation function that counts non-null values in each group
/// </summary>
public sealed class CountAggregation : AggregationFunction
{
    /// <inheritdoc />
    public override string Name => "Count";

    /// <inheritdoc />
    public override Type GetResultType(Type inputType) => typeof(long);

    /// <inheritdoc />
    public override object? Apply(IColumn column, IReadOnlyList<int> groupIndices)
    {
        if (column == null)
            throw new ArgumentNullException(nameof(column));
        if (groupIndices == null)
            throw new ArgumentNullException(nameof(groupIndices));

        long count = 0;
        foreach (var index in groupIndices)
            if (column.GetValue(index) != null)
                count++;

        return count;
    }
}

/// <summary>
/// Row-count aggregation function that counts every row in each group regardless of nulls.
/// Maps to LINQ's <c>group.Count()</c> without a selector.
/// </summary>
public sealed class RowCountAggregation : AggregationFunction
{
    /// <inheritdoc />
    public override string Name => "Count";

    /// <inheritdoc />
    public override Type GetResultType(Type inputType) => typeof(long);

    /// <inheritdoc />
    public override object? Apply(IColumn column, IReadOnlyList<int> groupIndices)
    {
        if (groupIndices == null)
            throw new ArgumentNullException(nameof(groupIndices));

        return (long)groupIndices.Count;
    }
}

/// <summary>
/// Sum aggregation function that sums numeric values in each group using vectorized operations
/// </summary>
public sealed class SumAggregation : AggregationFunction
{
    /// <inheritdoc />
    public override string Name => "Sum";

    /// <inheritdoc />
    public override Type GetResultType(Type inputType)
    {
        ValidateInputType(inputType);

        // Handle nullable types by checking the underlying type
        var underlyingType = Nullable.GetUnderlyingType(inputType) ?? inputType;

        // Follow NivaraSeries result-type promotion rules for the full numeric domain
        return underlyingType switch
        {
            Type t when t == typeof(byte) => typeof(long),
            Type t when t == typeof(sbyte) => typeof(long),
            Type t when t == typeof(short) => typeof(long),
            Type t when t == typeof(ushort) => typeof(long),
            Type t when t == typeof(int) => typeof(long),
            Type t when t == typeof(uint) => typeof(long),
            Type t when t == typeof(char) => typeof(long),
            Type t when t == typeof(bool) => typeof(long),
            Type t when t == typeof(long) => typeof(long),
            Type t when t == typeof(ulong) => typeof(ulong),
            Type t when t == typeof(nint) => typeof(Int128),
            Type t when t == typeof(nuint) => typeof(UInt128),
            Type t when t == typeof(Int128) => typeof(Int128),
            Type t when t == typeof(UInt128) => typeof(UInt128),
            Type t when t == typeof(float) => typeof(double),
            Type t when t == typeof(Half) => typeof(double),
            Type t when t == typeof(double) => typeof(double),
            Type t when t == typeof(decimal) => typeof(decimal),
            _ => throw new ArgumentException($"Sum aggregation not supported for type {inputType.Name}")
        };
    }

    /// <inheritdoc />
    public override object? Apply(IColumn column, IReadOnlyList<int> groupIndices)
    {
        if (column == null)
            throw new ArgumentNullException(nameof(column));
        if (groupIndices == null)
            throw new ArgumentNullException(nameof(groupIndices));

        ValidateInputType(column.ElementType);

        // Extract valid values for this group
        var validValues = ExtractValidValues(column, groupIndices);
        if (validValues.Count == 0)
            return GetZeroValue(GetResultType(column.ElementType));

        // Handle nullable types by checking the underlying type
        var elementType = Nullable.GetUnderlyingType(column.ElementType) ?? column.ElementType;

        return elementType switch
        {
            Type t when t == typeof(int) => SumVectorized<int, long>(validValues),
            Type t when t == typeof(byte) => SumVectorized<byte, long>(validValues),
            Type t when t == typeof(sbyte) => SumVectorized<sbyte, long>(validValues),
            Type t when t == typeof(short) => SumVectorized<short, long>(validValues),
            Type t when t == typeof(ushort) => SumVectorized<ushort, long>(validValues),
            Type t when t == typeof(uint) => SumVectorized<uint, long>(validValues),
            Type t when t == typeof(char) => SumVectorized<char, long>(validValues),
            Type t when t == typeof(bool) => SumVectorizedBool<long>(validValues),
            Type t when t == typeof(long) => SumVectorized<long, long>(validValues),
            Type t when t == typeof(ulong) => SumVectorized<ulong, ulong>(validValues),
            Type t when t == typeof(nint) => SumVectorized<nint, Int128>(validValues),
            Type t when t == typeof(nuint) => SumVectorized<nuint, UInt128>(validValues),
            Type t when t == typeof(Int128) => SumVectorized<Int128, Int128>(validValues),
            Type t when t == typeof(UInt128) => SumVectorized<UInt128, UInt128>(validValues),
            Type t when t == typeof(float) => SumVectorized<float, double>(validValues),
            Type t when t == typeof(Half) => SumVectorized<Half, double>(validValues),
            Type t when t == typeof(double) => SumVectorized<double, double>(validValues),
            Type t when t == typeof(decimal) => SumScalarDecimal(validValues),
            _ => throw new ArgumentException($"Sum aggregation not supported for type {column.ElementType.Name}")
        };
    }

    /// <inheritdoc />
    protected override void ValidateInputType(Type inputType)
    {
        var underlying = Nullable.GetUnderlyingType(inputType) ?? inputType;
        var supported = TypeCompatibilityValidator.GetNumericTypes().Append(typeof(bool));
        if (!supported.Contains(underlying))
            throw new ArgumentException($"Sum aggregation requires numeric type, got {inputType.Name}");
    }

    /// <summary>
    /// Extracts valid (non-null) values from a column for the specified indices
    /// </summary>
    static List<object> ExtractValidValues(IColumn column, IReadOnlyList<int> groupIndices)
    {
        var validValues = new List<object>();
        foreach (var index in groupIndices)
        {
            var value = column.GetValue(index);
            if (value != null)
                validValues.Add(value);
        }
        return validValues;
    }

    /// <summary>
    /// Performs vectorized sum for numeric values using generic TensorPrimitives after
    /// widening each boxed value into the promoted result type via typed CreateChecked
    /// conversion (byte/sbyte/short/ushort/int/uint/char → long, nint → Int128, nuint →
    /// UInt128, float/Half → double), preserving the documented result-type promotion rules.
    /// </summary>
    static object SumVectorized<TSource, TResult>(List<object> validValues)
        where TSource : INumberBase<TSource>
        where TResult : unmanaged, INumber<TResult>
    {
        var widened = new TResult[validValues.Count];
        for (int i = 0; i < validValues.Count; i++)
            widened[i] = TResult.CreateChecked((TSource)validValues[i]);

        return TensorPrimitives.Sum(widened.AsSpan());
    }

    /// <summary>
    /// Performs vectorized sum of boolean values, counting true values as ones (bool is not an
    /// INumberBase type, so it is converted explicitly before widening into the result type).
    /// </summary>
    static object SumVectorizedBool<TResult>(List<object> validValues)
        where TResult : unmanaged, INumber<TResult>
    {
        var widened = new TResult[validValues.Count];
        for (int i = 0; i < validValues.Count; i++)
            widened[i] = TResult.CreateChecked((bool)validValues[i] ? 1 : 0);

        return TensorPrimitives.Sum(widened.AsSpan());
    }

    /// <summary>
    /// Performs scalar decimal sum aggregation (decimal is not a supported TensorPrimitives element type)
    /// </summary>
    static object SumScalarDecimal(List<object> validValues)
    {
        decimal sum = 0m;
        foreach (var value in validValues)
            sum += (decimal)value;

        return sum;
    }

    /// <summary>
    /// Gets the zero value for a given type
    /// </summary>
    static object GetZeroValue(Type type)
    {
        return type switch
        {
            Type t when t == typeof(long) => 0L,
            Type t when t == typeof(ulong) => 0UL,
            Type t when t == typeof(Int128) => Int128.Zero,
            Type t when t == typeof(UInt128) => UInt128.Zero,
            Type t when t == typeof(double) => 0.0,
            Type t when t == typeof(decimal) => 0m,
            _ => Activator.CreateInstance(type)!
        };
    }
}

/// <summary>
/// Min aggregation function that finds the minimum value in each group
/// </summary>
public sealed class MinAggregation : AggregationFunction
{
    /// <inheritdoc />
    public override string Name => "Min";

    /// <inheritdoc />
    public override Type GetResultType(Type inputType) => inputType;

    /// <inheritdoc />
    public override object? Apply(IColumn column, IReadOnlyList<int> groupIndices)
    {
        if (column == null)
            throw new ArgumentNullException(nameof(column));
        if (groupIndices == null)
            throw new ArgumentNullException(nameof(groupIndices));

        // Extract valid values for this group
        var validValues = new List<object>();
        foreach (var index in groupIndices)
        {
            var value = column.GetValue(index);
            if (value != null)
                validValues.Add(value);
        }

        if (validValues.Count == 0)
            return null;

        // Handle nullable types by checking the underlying type
        var elementType = Nullable.GetUnderlyingType(column.ElementType) ?? column.ElementType;

        // Use vectorized operations for supported types
        return elementType switch
        {
            Type t when t == typeof(float) => MinVectorized<float>(validValues),
            Type t when t == typeof(double) => MinVectorized<double>(validValues),
            Type t when t == typeof(int) => MinVectorized<int>(validValues),
            Type t when t == typeof(long) => MinVectorized<long>(validValues),
            Type t when t == typeof(short) => MinVectorized<short>(validValues),
            Type t when t == typeof(ushort) => MinVectorized<ushort>(validValues),
            Type t when t == typeof(uint) => MinVectorized<uint>(validValues),
            Type t when t == typeof(ulong) => MinVectorized<ulong>(validValues),
            Type t when t == typeof(byte) => MinVectorized<byte>(validValues),
            Type t when t == typeof(sbyte) => MinVectorized<sbyte>(validValues),
            Type t when t == typeof(decimal) => MinVectorized<decimal>(validValues),
            _ => MinScalar(validValues)
        };
    }

    /// <summary>
    /// Performs vectorized min for numeric values using generic TensorPrimitives
    /// </summary>
    static object MinVectorized<T>(List<object> validValues)
        where T : unmanaged, INumber<T>
    {
        var typedValues = new T[validValues.Count];
        for (int i = 0; i < validValues.Count; i++)
            typedValues[i] = (T)validValues[i];

        return TensorPrimitives.Min(typedValues.AsSpan());
    }

    /// <summary>
    /// Performs scalar min for non-vectorizable types
    /// </summary>
    static object MinScalar(List<object> validValues)
    {
        object min = validValues[0];
        var comparer = Comparer<object>.Default;

        for (int i = 1; i < validValues.Count; i++)
            if (comparer.Compare(validValues[i], min) < 0)
                min = validValues[i];

        return min;
    }
}

/// <summary>
/// Max aggregation function that finds the maximum value in each group
/// </summary>
public sealed class MaxAggregation : AggregationFunction
{
    /// <inheritdoc />
    public override string Name => "Max";

    /// <inheritdoc />
    public override Type GetResultType(Type inputType) => inputType;

    /// <inheritdoc />
    public override object? Apply(IColumn column, IReadOnlyList<int> groupIndices)
    {
        if (column == null)
            throw new ArgumentNullException(nameof(column));
        if (groupIndices == null)
            throw new ArgumentNullException(nameof(groupIndices));

        // Extract valid values for this group
        var validValues = new List<object>();
        foreach (var index in groupIndices)
        {
            var value = column.GetValue(index);
            if (value != null)
                validValues.Add(value);
        }

        if (validValues.Count == 0)
            return null;

        // Handle nullable types by checking the underlying type
        var elementType = Nullable.GetUnderlyingType(column.ElementType) ?? column.ElementType;

        // Use vectorized operations for supported types
        return elementType switch
        {
            Type t when t == typeof(float) => MaxVectorized<float>(validValues),
            Type t when t == typeof(double) => MaxVectorized<double>(validValues),
            Type t when t == typeof(int) => MaxVectorized<int>(validValues),
            Type t when t == typeof(long) => MaxVectorized<long>(validValues),
            Type t when t == typeof(short) => MaxVectorized<short>(validValues),
            Type t when t == typeof(ushort) => MaxVectorized<ushort>(validValues),
            Type t when t == typeof(uint) => MaxVectorized<uint>(validValues),
            Type t when t == typeof(ulong) => MaxVectorized<ulong>(validValues),
            Type t when t == typeof(byte) => MaxVectorized<byte>(validValues),
            Type t when t == typeof(sbyte) => MaxVectorized<sbyte>(validValues),
            Type t when t == typeof(decimal) => MaxVectorized<decimal>(validValues),
            _ => MaxScalar(validValues)
        };
    }

    /// <summary>
    /// Performs vectorized max for numeric values using generic TensorPrimitives
    /// </summary>
    static object MaxVectorized<T>(List<object> validValues)
        where T : unmanaged, INumber<T>
    {
        var typedValues = new T[validValues.Count];
        for (int i = 0; i < validValues.Count; i++)
            typedValues[i] = (T)validValues[i];

        return TensorPrimitives.Max(typedValues.AsSpan());
    }

    /// <summary>
    /// Performs scalar max for non-vectorizable types
    /// </summary>
    static object MaxScalar(List<object> validValues)
    {
        object max = validValues[0];
        var comparer = Comparer<object>.Default;

        for (int i = 1; i < validValues.Count; i++)
            if (comparer.Compare(validValues[i], max) > 0)
                max = validValues[i];

        return max;
    }
}

/// <summary>
/// Mean (average) aggregation function that computes the arithmetic mean of numeric values in each group
/// </summary>
public sealed class MeanAggregation : AggregationFunction
{
    /// <inheritdoc />
    public override string Name => "Mean";

    /// <inheritdoc />
    public override Type GetResultType(Type inputType)
    {
        ValidateInputType(inputType);

        // Mean always returns double for numeric types
        return typeof(double);
    }

    /// <inheritdoc />
    public override object? Apply(IColumn column, IReadOnlyList<int> groupIndices)
    {
        if (column == null)
            throw new ArgumentNullException(nameof(column));
        if (groupIndices == null)
            throw new ArgumentNullException(nameof(groupIndices));

        ValidateInputType(column.ElementType);

        // Extract valid values for this group
        var validValues = new List<object>();
        foreach (var index in groupIndices)
        {
            var value = column.GetValue(index);
            if (value != null)
                validValues.Add(value);
        }

        if (validValues.Count == 0)
            return null;

        // Calculate sum and divide by count
        var sumAggregation = new SumAggregation();
        var sum = sumAggregation.Apply(column, groupIndices);

        if (sum == null)
            return null;

        // Convert sum to double and divide by count
        var doubleSum = ToDouble(sum);
        return doubleSum / validValues.Count;
    }

    /// <summary>
    /// Converts a boxed aggregation sum to double. Int128/UInt128 (and thus nint/nuint, whose
    /// sums are Int128/UInt128) do not implement IConvertible, so Convert.ChangeType would throw.
    /// </summary>
    static double ToDouble(object sum)
    {
        return sum switch
        {
            long value => value,
            ulong value => value,
            Int128 value => (double)value,
            UInt128 value => (double)value,
            double value => value,
            decimal value => (double)value,
            _ => throw new ArgumentException($"Cannot convert sum of type {sum.GetType().Name} to double")
        };
    }

    /// <inheritdoc />
    protected override void ValidateInputType(Type inputType)
    {
        var underlying = Nullable.GetUnderlyingType(inputType) ?? inputType;
        var supported = TypeCompatibilityValidator.GetNumericTypes().Append(typeof(bool));
        if (!supported.Contains(underlying))
            throw new ArgumentException($"Mean aggregation requires numeric type, got {inputType.Name}");
    }
}

/// <summary>
/// Factory class for creating standard aggregation functions
/// </summary>
public static class AggregationFunctions
{
    /// <summary>
    /// Creates a Count aggregation function
    /// </summary>
    /// <returns>A new CountAggregation instance</returns>
    public static CountAggregation Count() => new();

    /// <summary>
    /// Creates a row-count aggregation function that counts every row in a group
    /// </summary>
    /// <returns>A new RowCountAggregation instance</returns>
    public static RowCountAggregation RowCount() => new();

    /// <summary>
    /// Creates a Sum aggregation function
    /// </summary>
    /// <returns>A new SumAggregation instance</returns>
    public static SumAggregation Sum() => new();

    /// <summary>
    /// Creates a Min aggregation function
    /// </summary>
    /// <returns>A new MinAggregation instance</returns>
    public static MinAggregation Min() => new();

    /// <summary>
    /// Creates a Max aggregation function
    /// </summary>
    /// <returns>A new MaxAggregation instance</returns>
    public static MaxAggregation Max() => new();

    /// <summary>
    /// Creates a Mean aggregation function
    /// </summary>
    /// <returns>A new MeanAggregation instance</returns>
    public static MeanAggregation Mean() => new();

    /// <summary>
    /// Gets all standard aggregation functions
    /// </summary>
    /// <returns>A collection of standard aggregation functions</returns>
    public static IReadOnlyList<AggregationFunction> GetStandardFunctions()
    {
        return new AggregationFunction[]
        {
            Count(),
            Sum(),
            Min(),
            Max(),
            Mean()
        };
    }
}
