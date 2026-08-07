using Nivara.Exceptions;
using Nivara.Query;

namespace Nivara.Operations;

/// <summary>
/// Base class for window-function operations (rolling, cumulative, shift).
/// Appends a computed result column while preserving all input columns.
/// </summary>
/// <remarks>Added as part of issue #135 window functions delivery.</remarks>
abstract class WindowOperationBase : IQueryOperation
{
    /// <summary>
    /// Initializes a new instance of WindowOperationBase
    /// </summary>
    /// <param name="source">The source column name</param>
    /// <param name="resultColumn">The name of the appended result column</param>
    protected WindowOperationBase(string source, string resultColumn)
    {
        Source = source ?? throw new ArgumentNullException(nameof(source));
        ResultColumn = resultColumn ?? throw new ArgumentNullException(nameof(resultColumn));
    }

    /// <summary>
    /// Gets the source column name
    /// </summary>
    public string Source { get; }

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

        if (!inputSchema.HasColumn(Source))
            throw new SchemaValidationException(
                $"Window source column '{Source}' not found in schema. Available columns: {string.Join(", ", inputSchema.ColumnNames)}");

        if (inputSchema.HasColumn(ResultColumn))
            throw new ArgumentException($"Result column '{ResultColumn}' already exists in the schema", nameof(ResultColumn));

        var sourceType = inputSchema.GetColumnType(Source);
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
            if (!input.TryGetValue(Source, out var sourceColumn))
                throw new ColumnNotFoundException(Source, input.Keys);

            var resultColumn = Compute(sourceColumn);

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
        => $"{OperationType}({Source} -> {ResultColumn})";
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
        => $"Rolling{Kind}({Source} -> {ResultColumn}, WindowSize: {WindowSize})";
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
        => $"Cumulative{Kind}({Source} -> {ResultColumn})";
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
        => $"Shift({Source} -> {ResultColumn}, Periods: {Periods})";
}
