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
    internal virtual IColumn ApplyToGroups(IColumn column, IEnumerable<(GroupKey Key, IReadOnlyList<int> Indices)> groups)
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
        return ColumnFactory.Create(elementType, values);
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

    /// <summary>
    /// Reads the non-null values of a typed column via the generic <see cref="IColumn{T}"/> indexer
    /// (no per-element boxing) into a compact array, skipping null positions.
    /// </summary>
    protected static T[] ExtractValidTyped<T>(IColumn column, IReadOnlyList<int> groupIndices)
        where T : struct
    {
        var typed = (IColumn<T>)column;
        int count = 0;
        foreach (var idx in groupIndices)
            if (!column.IsNull(idx))
                count++;
        var values = new T[count];
        int pos = 0;
        foreach (var idx in groupIndices)
            if (!column.IsNull(idx))
                values[pos++] = typed[idx];
        return values;
    }
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
            Type t when t == typeof(BFloat16) => typeof(double),
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

        // Handle nullable types by checking the underlying type
        var elementType = Nullable.GetUnderlyingType(column.ElementType) ?? column.ElementType;

        return elementType switch
        {
            Type t when t == typeof(int) => SumVectorized<int, long>(column, groupIndices),
            Type t when t == typeof(byte) => SumVectorized<byte, long>(column, groupIndices),
            Type t when t == typeof(sbyte) => SumVectorized<sbyte, long>(column, groupIndices),
            Type t when t == typeof(short) => SumVectorized<short, long>(column, groupIndices),
            Type t when t == typeof(ushort) => SumVectorized<ushort, long>(column, groupIndices),
            Type t when t == typeof(uint) => SumVectorized<uint, long>(column, groupIndices),
            Type t when t == typeof(char) => SumVectorized<char, long>(column, groupIndices),
            Type t when t == typeof(bool) => SumVectorizedBool<long>(column, groupIndices),
            Type t when t == typeof(long) => SumVectorized<long, long>(column, groupIndices),
            Type t when t == typeof(ulong) => SumVectorized<ulong, ulong>(column, groupIndices),
            Type t when t == typeof(nint) => SumVectorized<nint, Int128>(column, groupIndices),
            Type t when t == typeof(nuint) => SumVectorized<nuint, UInt128>(column, groupIndices),
            Type t when t == typeof(Int128) => SumVectorized<Int128, Int128>(column, groupIndices),
            Type t when t == typeof(UInt128) => SumVectorized<UInt128, UInt128>(column, groupIndices),
            Type t when t == typeof(float) => SumVectorized<float, double>(column, groupIndices),
            Type t when t == typeof(Half) => SumVectorized<Half, double>(column, groupIndices),
            Type t when t == typeof(BFloat16) => SumVectorized<BFloat16, double>(column, groupIndices),
            Type t when t == typeof(double) => SumVectorized<double, double>(column, groupIndices),
            Type t when t == typeof(decimal) => SumScalarDecimal(column, groupIndices),
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
    /// Performs vectorized sum for numeric values using generic TensorPrimitives after widening
    /// each value into the promoted result type via typed CreateChecked conversion
    /// (byte/sbyte/short/ushort/int/uint/char → long, nint → Int128, nuint → UInt128,
    /// float/Half → double), preserving the documented result-type promotion rules. Values are
    /// read through the typed IColumn&lt;TSource&gt; indexer so there is no per-element boxing.
    /// </summary>
    static object SumVectorized<TSource, TResult>(IColumn column, IReadOnlyList<int> groupIndices)
        where TSource : struct, INumberBase<TSource>
        where TResult : unmanaged, INumber<TResult>
    {
        var values = ExtractValidTyped<TSource>(column, groupIndices);
        var widened = new TResult[values.Length];
        for (int i = 0; i < values.Length; i++)
            widened[i] = TResult.CreateChecked(values[i]);

        return TensorPrimitives.Sum(widened.AsSpan());
    }

    /// <summary>
    /// Performs vectorized sum of boolean values, counting true values as ones (bool is not an
    /// INumberBase type, so it is converted explicitly before widening into the result type).
    /// Values are read through the typed IColumn&lt;bool&gt; indexer so there is no per-element boxing.
    /// </summary>
    static object SumVectorizedBool<TResult>(IColumn column, IReadOnlyList<int> groupIndices)
        where TResult : unmanaged, INumber<TResult>
    {
        var values = ExtractValidTyped<bool>(column, groupIndices);
        var widened = new TResult[values.Length];
        for (int i = 0; i < values.Length; i++)
            widened[i] = TResult.CreateChecked(values[i] ? 1 : 0);

        return TensorPrimitives.Sum(widened.AsSpan());
    }

    /// <summary>
    /// Performs scalar decimal sum aggregation (decimal is not a supported TensorPrimitives element
    /// type). Values are read through the typed IColumn&lt;decimal&gt; indexer so there is no per-element boxing.
    /// </summary>
    static object SumScalarDecimal(IColumn column, IReadOnlyList<int> groupIndices)
    {
        var values = ExtractValidTyped<decimal>(column, groupIndices);
        decimal sum = 0m;
        foreach (var value in values)
            sum += value;

        return sum;
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

        // Handle nullable types by checking the underlying type
        var elementType = Nullable.GetUnderlyingType(column.ElementType) ?? column.ElementType;

        // Vectorized for INumber<T> types; typed scalar for char/bool; boxed fallback for
        // genuinely non-numeric element types (string/DateTime/custom structs).
        return elementType switch
        {
            Type t when t == typeof(float) => MinVectorized<float>(column, groupIndices),
            Type t when t == typeof(double) => MinVectorized<double>(column, groupIndices),
            Type t when t == typeof(int) => MinVectorized<int>(column, groupIndices),
            Type t when t == typeof(long) => MinVectorized<long>(column, groupIndices),
            Type t when t == typeof(short) => MinVectorized<short>(column, groupIndices),
            Type t when t == typeof(ushort) => MinVectorized<ushort>(column, groupIndices),
            Type t when t == typeof(uint) => MinVectorized<uint>(column, groupIndices),
            Type t when t == typeof(ulong) => MinVectorized<ulong>(column, groupIndices),
            Type t when t == typeof(byte) => MinVectorized<byte>(column, groupIndices),
            Type t when t == typeof(sbyte) => MinVectorized<sbyte>(column, groupIndices),
            Type t when t == typeof(decimal) => MinVectorized<decimal>(column, groupIndices),
            Type t when t == typeof(nint) => MinVectorized<nint>(column, groupIndices),
            Type t when t == typeof(nuint) => MinVectorized<nuint>(column, groupIndices),
            Type t when t == typeof(Int128) => MinVectorized<Int128>(column, groupIndices),
            Type t when t == typeof(UInt128) => MinVectorized<UInt128>(column, groupIndices),
            Type t when t == typeof(Half) => MinVectorized<Half>(column, groupIndices),
            Type t when t == typeof(BFloat16) => MinVectorized<BFloat16>(column, groupIndices),
            Type t when t == typeof(char) => MinScalar<char>(column, groupIndices),
            Type t when t == typeof(bool) => MinScalar<bool>(column, groupIndices),
            _ => MinScalarBoxed(column, groupIndices)
        };
    }

    /// <summary>
    /// Performs vectorized min for numeric values using generic TensorPrimitives. Values are read
    /// through the typed IColumn&lt;T&gt; indexer so there is no per-element boxing.
    /// </summary>
    static object? MinVectorized<T>(IColumn column, IReadOnlyList<int> groupIndices)
        where T : unmanaged, INumber<T>
    {
        var values = ExtractValidTyped<T>(column, groupIndices);
        if (values.Length == 0)
            return null;

        return TensorPrimitives.Min(values.AsSpan());
    }

    /// <summary>
    /// Performs typed scalar min for char/bool via Comparer&lt;T&gt;.Default (no per-element boxing).
    /// </summary>
    static object? MinScalar<T>(IColumn column, IReadOnlyList<int> groupIndices)
        where T : struct, IComparable<T>
    {
        var values = ExtractValidTyped<T>(column, groupIndices);
        if (values.Length == 0)
            return null;

        var comparer = Comparer<T>.Default;
        T min = values[0];
        for (int i = 1; i < values.Length; i++)
            if (comparer.Compare(values[i], min) < 0)
                min = values[i];

        return min;
    }

    /// <summary>
    /// Boxed scalar fallback for genuinely non-numeric element types (string/DateTime/custom
    /// structs), preserving the prior Comparer&lt;object&gt; behavior.
    /// </summary>
    static object? MinScalarBoxed(IColumn column, IReadOnlyList<int> groupIndices)
    {
        var validValues = new List<object>();
        foreach (var index in groupIndices)
        {
            var value = column.GetValue(index);
            if (value != null)
                validValues.Add(value);
        }

        if (validValues.Count == 0)
            return null;

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

        // Handle nullable types by checking the underlying type
        var elementType = Nullable.GetUnderlyingType(column.ElementType) ?? column.ElementType;

        // Vectorized for INumber<T> types; typed scalar for char/bool; boxed fallback for
        // genuinely non-numeric element types (string/DateTime/custom structs).
        return elementType switch
        {
            Type t when t == typeof(float) => MaxVectorized<float>(column, groupIndices),
            Type t when t == typeof(double) => MaxVectorized<double>(column, groupIndices),
            Type t when t == typeof(int) => MaxVectorized<int>(column, groupIndices),
            Type t when t == typeof(long) => MaxVectorized<long>(column, groupIndices),
            Type t when t == typeof(short) => MaxVectorized<short>(column, groupIndices),
            Type t when t == typeof(ushort) => MaxVectorized<ushort>(column, groupIndices),
            Type t when t == typeof(uint) => MaxVectorized<uint>(column, groupIndices),
            Type t when t == typeof(ulong) => MaxVectorized<ulong>(column, groupIndices),
            Type t when t == typeof(byte) => MaxVectorized<byte>(column, groupIndices),
            Type t when t == typeof(sbyte) => MaxVectorized<sbyte>(column, groupIndices),
            Type t when t == typeof(decimal) => MaxVectorized<decimal>(column, groupIndices),
            Type t when t == typeof(nint) => MaxVectorized<nint>(column, groupIndices),
            Type t when t == typeof(nuint) => MaxVectorized<nuint>(column, groupIndices),
            Type t when t == typeof(Int128) => MaxVectorized<Int128>(column, groupIndices),
            Type t when t == typeof(UInt128) => MaxVectorized<UInt128>(column, groupIndices),
            Type t when t == typeof(Half) => MaxVectorized<Half>(column, groupIndices),
            Type t when t == typeof(BFloat16) => MaxVectorized<BFloat16>(column, groupIndices),
            Type t when t == typeof(char) => MaxScalar<char>(column, groupIndices),
            Type t when t == typeof(bool) => MaxScalar<bool>(column, groupIndices),
            _ => MaxScalarBoxed(column, groupIndices)
        };
    }

    /// <summary>
    /// Performs vectorized max for numeric values using generic TensorPrimitives. Values are read
    /// through the typed IColumn&lt;T&gt; indexer so there is no per-element boxing.
    /// </summary>
    static object? MaxVectorized<T>(IColumn column, IReadOnlyList<int> groupIndices)
        where T : unmanaged, INumber<T>
    {
        var values = ExtractValidTyped<T>(column, groupIndices);
        if (values.Length == 0)
            return null;

        return TensorPrimitives.Max(values.AsSpan());
    }

    /// <summary>
    /// Performs typed scalar max for char/bool via Comparer&lt;T&gt;.Default (no per-element boxing).
    /// </summary>
    static object? MaxScalar<T>(IColumn column, IReadOnlyList<int> groupIndices)
        where T : struct, IComparable<T>
    {
        var values = ExtractValidTyped<T>(column, groupIndices);
        if (values.Length == 0)
            return null;

        var comparer = Comparer<T>.Default;
        T max = values[0];
        for (int i = 1; i < values.Length; i++)
            if (comparer.Compare(values[i], max) > 0)
                max = values[i];

        return max;
    }

    /// <summary>
    /// Boxed scalar fallback for genuinely non-numeric element types (string/DateTime/custom
    /// structs), preserving the prior Comparer&lt;object&gt; behavior.
    /// </summary>
    static object? MaxScalarBoxed(IColumn column, IReadOnlyList<int> groupIndices)
    {
        var validValues = new List<object>();
        foreach (var index in groupIndices)
        {
            var value = column.GetValue(index);
            if (value != null)
                validValues.Add(value);
        }

        if (validValues.Count == 0)
            return null;

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

        int validCount = 0;
        foreach (var idx in groupIndices)
            if (!column.IsNull(idx))
                validCount++;

        if (validCount == 0)
            return null;

        // Reuse the box-free Sum aggregation for the widened sum, then divide by the count.
        var sum = new SumAggregation().Apply(column, groupIndices);
        if (sum == null)
            return null;

        return ToDouble(sum) / validCount;
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
/// Quantile aggregation function that computes the q-th quantile of numeric values in each group
/// using linear interpolation (numpy/polars "linear", Hyndman-Fan type 7). Groups with no valid
/// values aggregate to null, matching Mean/Min/Max.
/// </summary>
public sealed class QuantileAggregation : AggregationFunction
{
    readonly double q;

    /// <summary>
    /// Initializes a quantile aggregation with the requested quantile.
    /// </summary>
    /// <param name="q">The quantile to compute, in [0, 1]</param>
    public QuantileAggregation(double q)
    {
        if (double.IsNaN(q) || q < 0d || q > 1d)
            throw new ArgumentOutOfRangeException(nameof(q), "Quantile must be in the range [0, 1].");
        this.q = q;
    }

    /// <summary>
    /// Gets the quantile this aggregation computes.
    /// </summary>
    public double Q => q;

    /// <inheritdoc />
    public override string Name => $"Quantile({q})";

    /// <inheritdoc />
    public override Type GetResultType(Type inputType)
    {
        ValidateInputType(inputType);

        // Quantile of a numeric column is a fractional value regardless of input width.
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

        return QuantileKernel.ComputeFromColumn(column, groupIndices, q);
    }

    /// <inheritdoc />
    protected override void ValidateInputType(Type inputType)
    {
        var underlying = Nullable.GetUnderlyingType(inputType) ?? inputType;
        if (!TypeCompatibilityValidator.GetNumericTypes().Contains(underlying))
            throw new ArgumentException($"Quantile aggregation requires numeric type, got {inputType.Name}");
    }
}

/// <summary>
/// Median aggregation function that computes the 0.5 quantile of numeric values in each group
/// using the same linear-interpolation definition as <see cref="QuantileAggregation"/>.
/// </summary>
public sealed class MedianAggregation : AggregationFunction
{
    /// <inheritdoc />
    public override string Name => "Median";

    /// <inheritdoc />
    public override Type GetResultType(Type inputType)
    {
        ValidateInputType(inputType);

        // Median of a numeric column is a fractional value regardless of input width.
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

        return QuantileKernel.ComputeFromColumn(column, groupIndices, 0.5);
    }

    /// <inheritdoc />
    protected override void ValidateInputType(Type inputType)
    {
        var underlying = Nullable.GetUnderlyingType(inputType) ?? inputType;
        if (!TypeCompatibilityValidator.GetNumericTypes().Contains(underlying))
            throw new ArgumentException($"Median aggregation requires numeric type, got {inputType.Name}");
    }
}

/// <summary>
/// Standard-deviation aggregation function. Follows numpy <c>ddof</c> semantics: 0 is the
/// population standard deviation (divide by n), 1 is the sample standard deviation (divide by
/// n - 1). Groups with no valid values aggregate to null.
/// </summary>
public sealed class StdDevAggregation : AggregationFunction
{
    readonly int ddof;

    /// <summary>
    /// Initializes a standard-deviation aggregation with the requested delta degrees of freedom.
    /// </summary>
    /// <param name="ddof">Delta degrees of freedom (0 = population, 1 = sample)</param>
    public StdDevAggregation(int ddof = 0)
    {
        if (ddof < 0)
            throw new ArgumentOutOfRangeException(nameof(ddof), "ddof must be >= 0.");
        this.ddof = ddof;
    }

    /// <summary>
    /// Gets the delta degrees of freedom this aggregation uses.
    /// </summary>
    public int Ddof => ddof;

    /// <inheritdoc />
    public override string Name => ddof == 0 ? "StdDev" : $"StdDev({ddof})";

    /// <inheritdoc />
    public override Type GetResultType(Type inputType)
    {
        ValidateInputType(inputType);

        // Standard deviation of a numeric column is a fractional value regardless of input width.
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

        return MomentsKernel.ComputeFromColumn(column, groupIndices, ddof, variance: false);
    }

    /// <inheritdoc />
    protected override void ValidateInputType(Type inputType)
    {
        var underlying = Nullable.GetUnderlyingType(inputType) ?? inputType;
        if (!TypeCompatibilityValidator.GetNumericTypes().Contains(underlying))
            throw new ArgumentException($"StdDev aggregation requires numeric type, got {inputType.Name}");
    }
}

/// <summary>
/// Variance aggregation function. Follows numpy <c>ddof</c> semantics: 0 is the population
/// variance (divide by n), 1 is the sample variance (divide by n - 1). Groups with no valid
/// values aggregate to null.
/// </summary>
public sealed class VarianceAggregation : AggregationFunction
{
    readonly int ddof;

    /// <summary>
    /// Initializes a variance aggregation with the requested delta degrees of freedom.
    /// </summary>
    /// <param name="ddof">Delta degrees of freedom (0 = population, 1 = sample)</param>
    public VarianceAggregation(int ddof = 0)
    {
        if (ddof < 0)
            throw new ArgumentOutOfRangeException(nameof(ddof), "ddof must be >= 0.");
        this.ddof = ddof;
    }

    /// <summary>
    /// Gets the delta degrees of freedom this aggregation uses.
    /// </summary>
    public int Ddof => ddof;

    /// <inheritdoc />
    public override string Name => ddof == 0 ? "Variance" : $"Variance({ddof})";

    /// <inheritdoc />
    public override Type GetResultType(Type inputType)
    {
        ValidateInputType(inputType);

        // Variance of a numeric column is a fractional value regardless of input width.
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

        return MomentsKernel.ComputeFromColumn(column, groupIndices, ddof, variance: true);
    }

    /// <inheritdoc />
    protected override void ValidateInputType(Type inputType)
    {
        var underlying = Nullable.GetUnderlyingType(inputType) ?? inputType;
        if (!TypeCompatibilityValidator.GetNumericTypes().Contains(underlying))
            throw new ArgumentException($"Variance aggregation requires numeric type, got {inputType.Name}");
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
    /// Creates a quantile aggregation function computing the q-th quantile with linear interpolation.
    /// </summary>
    /// <param name="q">The quantile to compute, in [0, 1]</param>
    /// <returns>A new QuantileAggregation instance</returns>
    public static QuantileAggregation Quantile(double q) => new(q);

    /// <summary>
    /// Creates a median aggregation function computing the 0.5 quantile.
    /// </summary>
    /// <returns>A new MedianAggregation instance</returns>
    public static MedianAggregation Median() => new();

    /// <summary>
    /// Creates a standard-deviation aggregation function.
    /// </summary>
    /// <param name="ddof">Delta degrees of freedom (0 = population, 1 = sample)</param>
    /// <returns>A new StdDevAggregation instance</returns>
    public static StdDevAggregation StdDev(int ddof = 0) => new(ddof);

    /// <summary>
    /// Creates a variance aggregation function.
    /// </summary>
    /// <param name="ddof">Delta degrees of freedom (0 = population, 1 = sample)</param>
    /// <returns>A new VarianceAggregation instance</returns>
    public static VarianceAggregation Variance(int ddof = 0) => new(ddof);

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
            Mean(),
            Median(),
            Quantile(0.25),
            Quantile(0.5),
            Quantile(0.75)
        };
    }
}
