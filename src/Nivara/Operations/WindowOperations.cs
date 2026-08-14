using Nivara.Exceptions;
using Nivara.Expressions;
using Nivara.Query;
using Nivara.Tensors;

namespace Nivara.Operations;

/// <summary>
/// Base class for window-function operations (rolling, cumulative, shift).
/// Appends a computed result column while preserving all input columns.
/// </summary>
/// <remarks>Added as part of issue #135 window functions delivery.</remarks>
abstract class WindowOperationBase : IQueryOperation
{
    /// <summary>
    /// Initializes a new instance of WindowOperationBase over a named source column
    /// </summary>
    /// <param name="source">The source column name</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    protected WindowOperationBase(string source, string resultColumn)
        : this(source: source ?? throw new ArgumentNullException(nameof(source)), sourceExpression: null, resultColumn)
    {
    }

    /// <summary>
    /// Initializes a new instance of WindowOperationBase over a named source column with a
    /// window specification (partition/order keys)
    /// </summary>
    /// <param name="source">The source column name</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="spec">The window specification</param>
    protected WindowOperationBase(string source, string resultColumn, WindowSpec spec)
        : this(source: source ?? throw new ArgumentNullException(nameof(source)), sourceExpression: null, resultColumn, spec)
    {
    }

    /// <summary>
    /// Initializes a new instance of WindowOperationBase over a computed source expression
    /// </summary>
    /// <param name="sourceExpression">The computed source column expression</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    protected WindowOperationBase(ColumnExpression sourceExpression, string resultColumn)
        : this(source: null, sourceExpression: sourceExpression ?? throw new ArgumentNullException(nameof(sourceExpression)), resultColumn)
    {
    }

    WindowOperationBase(string? source, ColumnExpression? sourceExpression, string resultColumn, WindowSpec? spec = null)
    {
        if (source is not null && string.IsNullOrWhiteSpace(source))
            throw new ArgumentException("Source column name cannot be null or whitespace", nameof(source));

        Source = source;
        SourceExpression = sourceExpression;
        ResultColumn = resultColumn ?? throw new ArgumentNullException(nameof(resultColumn));
        Spec = spec;
    }

    /// <summary>
    /// Gets the source column name (null when a computed source expression is used)
    /// </summary>
    public string? Source { get; }

    /// <summary>
    /// Gets the computed source column expression (null when a named source column is used)
    /// </summary>
    public ColumnExpression? SourceExpression { get; }

    /// <summary>
    /// Gets the optional window specification (partition/order keys). Null or empty = unpartitioned row order.
    /// </summary>
    public WindowSpec? Spec { get; }

    /// <summary>
    /// Gets the result column name
    /// </summary>
    public string ResultColumn { get; }

    public abstract string OperationType { get; }

    /// <inheritdoc />
    public Schema TransformSchema(Schema inputSchema)
    {
        if (inputSchema == null)
            throw new ArgumentNullException(nameof(inputSchema));

        Type sourceType;
        if (SourceExpression is not null)
        {
            try
            {
                SourceExpression.Validate(inputSchema);
            }
            catch (SchemaValidationException ex)
            {
                throw new SchemaValidationException($"Window source expression validation failed: {ex.Message}");
            }

            sourceType = SourceExpression.ResultType;
        }
        else
        {
            if (!inputSchema.HasColumn(Source!))
                throw new SchemaValidationException(
                    $"Window source column '{Source}' not found in schema. Available columns: {string.Join(", ", inputSchema.ColumnNames)}");

            sourceType = inputSchema.GetColumnType(Source!);
        }

        if (inputSchema.HasColumn(ResultColumn))
            throw new ArgumentException($"Result column '{ResultColumn}' already exists in the schema", nameof(ResultColumn));

        if (Spec is { IsEmpty: false })
        {
            foreach (var partition in Spec.PartitionColumns)
            {
                if (!inputSchema.HasColumn(partition))
                    throw new SchemaValidationException(
                        $"Partition column '{partition}' not found in schema. Available columns: {string.Join(", ", inputSchema.ColumnNames)}");
            }

            foreach (var sortKey in Spec.OrderKeys)
            {
                if (!inputSchema.HasColumn(sortKey.ColumnName))
                    throw new SchemaValidationException(
                        $"Order column '{sortKey.ColumnName}' not found in schema. Available columns: {string.Join(", ", inputSchema.ColumnNames)}");

                var columnType = inputSchema.GetColumnType(sortKey.ColumnName);
                if (!SortOperation.IsComparableType(columnType))
                    throw new SchemaValidationException(
                        $"Order column '{sortKey.ColumnName}' of type '{columnType.Name}' is not comparable and cannot be used for the window");
            }
        }

        var resultType = GetResultType(sourceType);
        return inputSchema.WithColumn(ResultColumn, resultType);
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        try
        {
            IColumn sourceColumn;
            if (SourceExpression is not null)
            {
                sourceColumn = new FusedExpressionEvaluator().Evaluate(SourceExpression, input);
            }
            else if (!input.TryGetValue(Source!, out sourceColumn!))
            {
                throw new ColumnNotFoundException(Source!, input.Keys);
            }

            var resultColumn = Spec is { IsEmpty: false }
                ? PartitionedWindowEngine.Compute(input, sourceColumn, Spec, Compute)
                : Compute(sourceColumn);

            var result = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
            foreach (var kvp in input)
                result[kvp.Key] = kvp.Value;
            result[ResultColumn] = resultColumn;
            return result;
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Window operation failed: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Computes the result column type from the source column type
    /// </summary>
    /// <param name="sourceType">The source column type</param>
    /// <returns>The result column type</returns>
    protected abstract Type GetResultType(Type sourceType);

    /// <summary>
    /// Computes the result column from the source column
    /// </summary>
    /// <param name="sourceColumn">The source column</param>
    /// <returns>The computed result column</returns>
    protected abstract IColumn Compute(IColumn sourceColumn);

    /// <inheritdoc />
    public override string ToString()
    {
        var baseStr = $"{OperationType}({(SourceExpression is not null ? SourceExpression.Name : Source)} -> {ResultColumn})";
        return Spec is { IsEmpty: false } ? $"{baseStr} {Spec}" : baseStr;
    }
}

/// <summary>
/// Rolling-window aggregate operation (sum / mean / min / max)
/// </summary>
sealed class RollingOperation : WindowOperationBase
{
    /// <summary>
    /// Initializes a new instance of RollingOperation
    /// </summary>
    /// <param name="source">The source column name</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="windowSize">The rolling window size</param>
    /// <param name="minPeriods">The minimum number of valid observations required</param>
    /// <param name="nullHandler">Optional null-replacement handler</param>
    /// <param name="kind">The rolling aggregate kind</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when windowSize is not positive</exception>
    public RollingOperation(string source, string resultColumn, int windowSize, int? minPeriods, Func<object?>? nullHandler, NivaraFrameExtensions.RollingKind kind)
        : base(source, resultColumn)
    {
        if (windowSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(windowSize), "Window size must be positive");

        WindowSize = windowSize;
        MinPeriods = minPeriods;
        NullHandler = nullHandler;
        Kind = kind;
    }

    /// <summary>
    /// Initializes a new instance of RollingOperation over a computed source expression
    /// </summary>
    /// <param name="sourceExpression">The computed source column expression</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="windowSize">The rolling window size</param>
    /// <param name="minPeriods">The minimum number of valid observations required</param>
    /// <param name="nullHandler">Optional null-replacement handler</param>
    /// <param name="kind">The rolling aggregate kind</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when windowSize is not positive</exception>
    public RollingOperation(ColumnExpression sourceExpression, string resultColumn, int windowSize, int? minPeriods = null, Func<object?>? nullHandler = null, NivaraFrameExtensions.RollingKind kind = NivaraFrameExtensions.RollingKind.Sum)
        : base(sourceExpression, resultColumn)
    {
        if (windowSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(windowSize), "Window size must be positive");

        WindowSize = windowSize;
        MinPeriods = minPeriods;
        NullHandler = nullHandler;
        Kind = kind;
    }

    /// <summary>
    /// Initializes a new instance of RollingOperation over a named source column with a
    /// window specification (partition/order keys)
    /// </summary>
    /// <param name="source">The source column name</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="windowSize">The rolling window size</param>
    /// <param name="spec">The window specification</param>
    /// <param name="minPeriods">The minimum number of valid observations required</param>
    /// <param name="nullHandler">Optional null-replacement handler</param>
    /// <param name="kind">The rolling aggregate kind</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when windowSize is not positive</exception>
    public RollingOperation(string source, string resultColumn, int windowSize, WindowSpec spec, int? minPeriods = null, Func<object?>? nullHandler = null, NivaraFrameExtensions.RollingKind kind = NivaraFrameExtensions.RollingKind.Sum)
        : base(source, resultColumn, spec ?? throw new ArgumentNullException(nameof(spec)))
    {
        if (windowSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(windowSize), "Window size must be positive");

        WindowSize = windowSize;
        MinPeriods = minPeriods;
        NullHandler = nullHandler;
        Kind = kind;
    }

    /// <summary>
    /// Gets the rolling window size
    /// </summary>
    public int WindowSize { get; }

    /// <summary>
    /// Gets the minimum number of valid observations required
    /// </summary>
    public int? MinPeriods { get; }

    /// <summary>
    /// Gets the optional null-replacement handler
    /// </summary>
    public Func<object?>? NullHandler { get; }

    /// <summary>
    /// Gets the rolling aggregate kind
    /// </summary>
    public NivaraFrameExtensions.RollingKind Kind { get; }

    public override string OperationType => Query.OperationType.Rolling;

    /// <inheritdoc />
    protected override Type GetResultType(Type sourceType)
        => Kind == NivaraFrameExtensions.RollingKind.Mean ? typeof(double) : sourceType;

    /// <inheritdoc />
    protected override IColumn Compute(IColumn sourceColumn)
        => NivaraFrameExtensions.CalculateRolling(sourceColumn, WindowSize, MinPeriods, NullHandler, Kind);

    /// <inheritdoc />
    public override string ToString()
        => $"Rolling{Kind}({(SourceExpression is not null ? SourceExpression.Name : Source)} -> {ResultColumn}, WindowSize: {WindowSize})";
}

/// <summary>
/// Cumulative aggregate operation (sum / max / min / product / count)
/// </summary>
sealed class CumulativeOperation : WindowOperationBase
{
    /// <summary>
    /// Initializes a new instance of CumulativeOperation
    /// </summary>
    /// <param name="source">The source column name</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="nullHandler">Optional null-replacement handler</param>
    /// <param name="kind">The cumulative aggregate kind</param>
    /// <param name="isCount">Whether this is a running count-of-non-null operation</param>
    public CumulativeOperation(string source, string resultColumn, Func<object?>? nullHandler, NivaraFrameExtensions.CumulativeKind kind, bool isCount = false)
        : base(source, resultColumn)
    {
        NullHandler = nullHandler;
        Kind = kind;
        IsCount = isCount;
    }

    /// <summary>
    /// Initializes a new instance of CumulativeOperation over a computed source expression
    /// </summary>
    /// <param name="sourceExpression">The computed source column expression</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="nullHandler">Optional null-replacement handler</param>
    /// <param name="kind">The cumulative aggregate kind</param>
    /// <param name="isCount">Whether this is a running count-of-non-null operation</param>
    public CumulativeOperation(ColumnExpression sourceExpression, string resultColumn, Func<object?>? nullHandler = null, NivaraFrameExtensions.CumulativeKind kind = NivaraFrameExtensions.CumulativeKind.Sum, bool isCount = false)
        : base(sourceExpression, resultColumn)
    {
        NullHandler = nullHandler;
        Kind = kind;
        IsCount = isCount;
    }

    /// <summary>
    /// Initializes a new instance of CumulativeOperation over a named source column with a
    /// window specification (partition/order keys)
    /// </summary>
    /// <param name="source">The source column name</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="spec">The window specification</param>
    /// <param name="nullHandler">Optional null-replacement handler</param>
    /// <param name="kind">The cumulative aggregate kind</param>
    /// <param name="isCount">Whether this is a running count-of-non-null operation</param>
    public CumulativeOperation(string source, string resultColumn, WindowSpec spec, Func<object?>? nullHandler = null, NivaraFrameExtensions.CumulativeKind kind = NivaraFrameExtensions.CumulativeKind.Sum, bool isCount = false)
        : base(source, resultColumn, spec ?? throw new ArgumentNullException(nameof(spec)))
    {
        NullHandler = nullHandler;
        Kind = kind;
        IsCount = isCount;
    }

    /// <summary>
    /// Gets the optional null-replacement handler
    /// </summary>
    public Func<object?>? NullHandler { get; }

    /// <summary>
    /// Gets the cumulative aggregate kind
    /// </summary>
    public NivaraFrameExtensions.CumulativeKind Kind { get; }

    /// <summary>
    /// Gets a value indicating whether this is a running count-of-non-null operation
    /// </summary>
    public bool IsCount { get; }

    public override string OperationType => Query.OperationType.Cumulative;

    /// <inheritdoc />
    protected override Type GetResultType(Type sourceType)
        => IsCount ? typeof(long) : sourceType;

    /// <inheritdoc />
    protected override IColumn Compute(IColumn sourceColumn)
        => IsCount
            ? NivaraFrameExtensions.CalculateCumulativeCount(sourceColumn)
            : NivaraFrameExtensions.CalculateCumulative(sourceColumn, NullHandler, Kind);

    /// <inheritdoc />
    public override string ToString()
        => $"Cumulative{Kind}({(SourceExpression is not null ? SourceExpression.Name : Source)} -> {ResultColumn})";
}

/// <summary>
/// Shift (lag) operation: <c>output[i] = input[i - periods]</c>, boundary positions null or <c>fillValue</c>.
/// <see cref="Nivara.Query.QueryFrame.Lead"/> is represented as a <see cref="ShiftOperation"/> with negated periods.
/// </summary>
sealed class ShiftOperation : WindowOperationBase
{
    /// <summary>
    /// Initializes a new instance of ShiftOperation
    /// </summary>
    /// <param name="source">The source column name</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="periods">The number of positions to shift by (negative = lead)</param>
    /// <param name="fillValue">Optional fill value for boundary positions</param>
    public ShiftOperation(string source, string resultColumn, int periods, object? fillValue = null)
        : base(source, resultColumn)
    {
        Periods = periods;
        FillValue = fillValue;
    }

    /// <summary>
    /// Initializes a new instance of ShiftOperation over a computed source expression
    /// </summary>
    /// <param name="sourceExpression">The computed source column expression</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="periods">The number of positions to shift by (negative = lead)</param>
    /// <param name="fillValue">Optional fill value for boundary positions</param>
    public ShiftOperation(ColumnExpression sourceExpression, string resultColumn, int periods, object? fillValue = null)
        : base(sourceExpression, resultColumn)
    {
        Periods = periods;
        FillValue = fillValue;
    }

    /// <summary>
    /// Initializes a new instance of ShiftOperation over a named source column with a
    /// window specification (partition/order keys)
    /// </summary>
    /// <param name="source">The source column name</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    /// <param name="periods">The number of positions to shift by (negative = lead)</param>
    /// <param name="spec">The window specification</param>
    /// <param name="fillValue">Optional fill value for boundary positions</param>
    public ShiftOperation(string source, string resultColumn, int periods, WindowSpec spec, object? fillValue = null)
        : base(source, resultColumn, spec ?? throw new ArgumentNullException(nameof(spec)))
    {
        Periods = periods;
        FillValue = fillValue;
    }

    /// <summary>
    /// Gets the number of positions to shift by (negative = lead)
    /// </summary>
    public int Periods { get; }

    /// <summary>
    /// Gets the optional fill value for boundary positions
    /// </summary>
    public object? FillValue { get; }

    public override string OperationType => Query.OperationType.Shift;

    /// <inheritdoc />
    protected override Type GetResultType(Type sourceType) => sourceType;

    /// <inheritdoc />
    protected override IColumn Compute(IColumn sourceColumn)
        => NivaraFrameExtensions.CalculateShift(sourceColumn, Periods, FillValue);

    /// <inheritdoc />
    public override string ToString()
        => $"Shift({(SourceExpression is not null ? SourceExpression.Name : Source)} -> {ResultColumn}, Periods: {Periods})";
}
