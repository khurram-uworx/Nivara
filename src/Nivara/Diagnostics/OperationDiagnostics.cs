using Nivara.Storage;
using System.Diagnostics;

namespace Nivara.Diagnostics;

/// <summary>
/// Provides diagnostic information about column operations, including kernel selection and performance metrics.
/// Used for performance analysis and optimization decisions.
/// </summary>
public sealed class OperationDiagnostics
{
    /// <summary>
    /// Initializes a new instance of OperationDiagnostics
    /// </summary>
    /// <param name="operationType">The type of operation performed</param>
    /// <param name="kernelUsed">The kernel type that was used</param>
    /// <param name="inputLength">The length of input data</param>
    /// <param name="elementType">The element type being operated on</param>
    /// <param name="hadNulls">Whether the operation involved null values</param>
    internal OperationDiagnostics(
        string operationType,
        KernelType kernelUsed,
        int inputLength,
        Type elementType,
        bool hadNulls)
        : this(operationType, kernelUsed, inputLength, elementType, hadNulls, 0, TimeSpan.Zero, null)
    {
    }

    /// <summary>
    /// Initializes a new instance of OperationDiagnostics with measured allocation and timing data.
    /// </summary>
    internal OperationDiagnostics(
        string operationType,
        KernelType kernelUsed,
        int inputLength,
        Type elementType,
        bool hadNulls,
        long allocatedBytes,
        TimeSpan elapsed,
        string? notes = null)
    {
        OperationType = operationType ?? throw new ArgumentNullException(nameof(operationType));
        KernelUsed = kernelUsed;
        InputLength = inputLength;
        ElementType = elementType ?? throw new ArgumentNullException(nameof(elementType));
        HadNulls = hadNulls;
        AllocatedBytes = Math.Max(0, allocatedBytes);
        Elapsed = elapsed;
        Notes = notes;
        Timestamp = DateTime.UtcNow;
    }

    /// <summary>
    /// Gets the type of operation that was performed
    /// </summary>
    public string OperationType { get; }

    /// <summary>
    /// Gets the kernel type that was actually used for the operation
    /// </summary>
    public KernelType KernelUsed { get; }

    /// <summary>
    /// Gets the length of the input data
    /// </summary>
    public int InputLength { get; }

    /// <summary>
    /// Gets the element type that was operated on
    /// </summary>
    public Type ElementType { get; }

    /// <summary>
    /// Gets a value indicating whether the operation involved null values
    /// </summary>
    public bool HadNulls { get; }

    /// <summary>
    /// Gets the managed bytes allocated by the measured operation, when available.
    /// </summary>
    public long AllocatedBytes { get; }

    /// <summary>
    /// Gets the elapsed time for the measured operation, when available.
    /// </summary>
    public TimeSpan Elapsed { get; }

    /// <summary>
    /// Gets optional operation-specific diagnostic notes.
    /// </summary>
    public string? Notes { get; }

    /// <summary>
    /// Gets the timestamp when the operation was performed
    /// </summary>
    public DateTime Timestamp { get; }

    /// <summary>
    /// Gets the reason why a particular kernel was selected
    /// </summary>
    public string KernelSelectionReason
    {
        get
        {
            if (KernelUsed == KernelType.Scalar)
            {
                if (!ColumnStorageFactory.IsVectorizable<object>()) // We can't use generic T here
                    return $"Type {ElementType.Name} is not vectorizable";
                if (!System.Numerics.Vector.IsHardwareAccelerated)
                    return "Hardware acceleration not available";
                if (InputLength < System.Numerics.Vector<byte>.Count * 4)
                    return $"Input length {InputLength} too small for vectorization overhead";
                return "Scalar kernel selected for unknown reason";
            }
            else
                return $"Vectorized kernel selected for {ElementType.Name} with length {InputLength}";
        }
    }

    /// <summary>
    /// Returns a string representation of the operation diagnostics
    /// </summary>
    /// <returns>A formatted string with operation details</returns>
    public override string ToString()
        => $"OperationDiagnostics {{ " +
        $"Operation: {OperationType}, " +
        $"Kernel: {KernelUsed}, " +
        $"Type: {ElementType.Name}, " +
        $"Length: {InputLength:N0}, " +
        $"HadNulls: {HadNulls}, " +
        $"AllocatedBytes: {AllocatedBytes:N0}, " +
        $"Elapsed: {Elapsed.TotalMilliseconds:F3}ms, " +
        (string.IsNullOrWhiteSpace(Notes) ? string.Empty : $"Notes: {Notes}, ") +
        $"Reason: {KernelSelectionReason} " +
        $"}}";
}

/// <summary>
/// Sink that receives recorded operation diagnostics.
/// </summary>
public interface IOperationDiagnosticsSink
{
    /// <summary>
    /// Gets a value indicating whether recording is enabled.
    /// </summary>
    bool IsEnabled { get; }

    /// <summary>
    /// Records an operation diagnostic.
    /// </summary>
    void RecordOperation(OperationDiagnostics diagnostic);
}

/// <summary>
/// A scoped, bounded collector of operation diagnostics.
/// </summary>
public sealed class OperationDiagnosticsCollector : IOperationDiagnosticsSink
{
    readonly object lockObject = new object();
    readonly OperationDiagnostics[] buffer;
    int head;
    int count;
    long overflowCount;

    /// <summary>
    /// Initializes a new collector with the default capacity (10,000) and enabled.
    /// </summary>
    public OperationDiagnosticsCollector()
        : this(10000, true)
    {
    }

    /// <summary>
    /// Initializes a new collector with the specified capacity.
    /// </summary>
    public OperationDiagnosticsCollector(int capacity)
        : this(capacity, true)
    {
    }

    /// <summary>
    /// Initializes a new collector with the specified enabled state.
    /// </summary>
    public OperationDiagnosticsCollector(bool isEnabled)
        : this(10000, isEnabled)
    {
    }

    /// <summary>
    /// Initializes a new collector with the specified capacity and enabled state.
    /// </summary>
    public OperationDiagnosticsCollector(int capacity, bool isEnabled)
    {
        if (capacity <= 0)
            throw new ArgumentOutOfRangeException(nameof(capacity));
        Capacity = capacity;
        buffer = new OperationDiagnostics[capacity];
        IsEnabled = isEnabled;
    }

    /// <summary>
    /// Initializes a new collector with the specified enabled state and capacity.
    /// </summary>
    public OperationDiagnosticsCollector(bool isEnabled, int capacity)
        : this(capacity, isEnabled)
    {
    }

    /// <summary>
    /// Gets the maximum number of entries retained.
    /// </summary>
    public int Capacity { get; }

    /// <summary>
    /// Gets the maximum number of entries retained (alias of <see cref="Capacity"/>).
    /// </summary>
    public int MaxEntries => Capacity;

    /// <summary>
    /// Gets or sets a value indicating whether recording is enabled.
    /// </summary>
    public bool IsEnabled { get; set; }

    /// <summary>
    /// Gets the number of entries dropped due to capacity.
    /// </summary>
    public long Overflowed => overflowCount;

    /// <summary>
    /// Gets the number of entries dropped due to capacity (alias).
    /// </summary>
    public long OverflowCount => overflowCount;

    /// <summary>
    /// Gets the number of entries dropped due to capacity (alias).
    /// </summary>
    public long DroppedCount => overflowCount;

    /// <summary>
    /// Gets the number of entries dropped due to capacity (alias).
    /// </summary>
    public long DroppedEntries => overflowCount;

    /// <summary>
    /// Gets the number of entries dropped due to capacity (alias).
    /// </summary>
    public long DroppedOperations => overflowCount;

    /// <inheritdoc />
    public void RecordOperation(OperationDiagnostics diagnostic)
    {
        if (!IsEnabled || diagnostic == null)
            return;

        lock (lockObject)
        {
            if (count < Capacity)
            {
                buffer[(head + count) % Capacity] = diagnostic;
                count++;
            }
            else
            {
                buffer[head] = diagnostic;
                head = (head + 1) % Capacity;
                overflowCount++;
            }
        }
    }

    /// <summary>
    /// Gets a snapshot of all recorded operations.
    /// </summary>
    public OperationDiagnostics[] GetRecordedOperations()
    {
        lock (lockObject)
        {
            var result = new OperationDiagnostics[count];
            for (int i = 0; i < count; i++)
                result[i] = buffer[(head + i) % Capacity];
            return result;
        }
    }

    /// <summary>
    /// Clears all recorded operations and the overflow counter.
    /// </summary>
    public void ClearRecordedOperations()
    {
        lock (lockObject)
        {
            Array.Clear(buffer, 0, buffer.Length);
            head = 0;
            count = 0;
            overflowCount = 0;
        }
    }

    /// <summary>
    /// Gets summary statistics about recorded operations.
    /// </summary>
    public OperationSummary GetSummary()
        => DiagnosticsTracker.BuildSummary(GetRecordedOperations());
}

/// <summary>
/// Captures allocation and timing measurements for a diagnostics operation.
/// </summary>
internal readonly struct OperationMeasurement
{
    readonly IOperationDiagnosticsSink? sink;
    readonly long initialAllocatedBytes;
    readonly Stopwatch? stopwatch;

    internal OperationMeasurement(IOperationDiagnosticsSink? sink)
    {
        this.sink = sink != null && sink.IsEnabled ? sink : null;
        initialAllocatedBytes = this.sink != null ? GC.GetAllocatedBytesForCurrentThread() : 0;
        stopwatch = this.sink != null ? Stopwatch.StartNew() : null;
    }

    public void Record(
        string operationType,
        KernelType kernelUsed,
        int inputLength,
        Type elementType,
        bool hadNulls,
        string? notes = null)
    {
        if (sink == null || stopwatch == null)
            return;

        stopwatch.Stop();
        var allocatedBytes = Math.Max(0, GC.GetAllocatedBytesForCurrentThread() - initialAllocatedBytes);
        sink.RecordOperation(new OperationDiagnostics(
            operationType,
            kernelUsed,
            inputLength,
            elementType,
            hadNulls,
            allocatedBytes,
            stopwatch.Elapsed,
            notes));
    }
}

/// <summary>
/// Tracks operation diagnostics for performance analysis.
/// Provides a thread-safe way to collect diagnostic information about column operations.
/// </summary>
public static class DiagnosticsTracker
{
    static readonly object globalLock = new object();
    static bool globalEnabled = false;
    static OperationDiagnosticsCollector globalCollector = new OperationDiagnosticsCollector(10000, true);
    static readonly AsyncLocal<IOperationDiagnosticsSink?> currentSink = new AsyncLocal<IOperationDiagnosticsSink?>();

    /// <summary>
    /// Gets or sets a value indicating whether diagnostic tracking is enabled (global sink).
    /// </summary>
    public static bool IsEnabled
    {
        get
        {
            var sink = currentSink.Value;
            if (sink != null)
                return sink.IsEnabled;
            return Volatile.Read(ref globalEnabled);
        }
        set
        {
            Volatile.Write(ref globalEnabled, value);
        }
    }

    /// <summary>
    /// Gets the currently active diagnostics sink, if any.
    /// </summary>
    public static IOperationDiagnosticsSink? CurrentSink => currentSink.Value;

    /// <summary>
    /// Gets or sets the maximum number of entries retained by the global sink.
    /// </summary>
    public static int MaxEntries
    {
        get => globalCollector.Capacity;
        set
        {
            lock (globalLock)
            {
                var replacement = new OperationDiagnosticsCollector(value, true);
                foreach (var op in globalCollector.GetRecordedOperations())
                    replacement.RecordOperation(op);
                globalCollector = replacement;
            }
        }
    }

    /// <summary>
    /// Gets or sets the maximum number of entries retained by the global sink (alias).
    /// </summary>
    public static int Capacity
    {
        get => MaxEntries;
        set => MaxEntries = value;
    }

    /// <summary>
    /// Gets the number of entries dropped by the global sink due to capacity.
    /// </summary>
    public static long Overflowed => globalCollector.Overflowed;

    /// <summary>
    /// Gets the number of entries dropped by the global sink due to capacity (alias).
    /// </summary>
    public static long OverflowCount => globalCollector.OverflowCount;

    /// <summary>
    /// Gets the number of entries dropped by the global sink due to capacity (alias).
    /// </summary>
    public static long DroppedCount => globalCollector.DroppedCount;

    /// <summary>
    /// Gets the number of entries dropped by the global sink due to capacity (alias).
    /// </summary>
    public static long DroppedEntries => globalCollector.DroppedEntries;

    /// <summary>
    /// Gets the number of entries dropped by the global sink due to capacity (alias).
    /// </summary>
    public static long DroppedOperations => globalCollector.DroppedOperations;

    /// <summary>
    /// Begins a diagnostics scope that routes recording to the given sink.
    /// </summary>
    public static IDisposable BeginScope(IOperationDiagnosticsSink sink)
    {
        if (sink == null)
            throw new ArgumentNullException(nameof(sink));

        var previous = currentSink.Value;
        currentSink.Value = sink;
        return new DiagnosticsScope(previous);
    }

    /// <summary>
    /// Begins a diagnostics scope that routes recording to the given collector.
    /// </summary>
    public static IDisposable BeginScope(OperationDiagnosticsCollector collector)
        => BeginScope((IOperationDiagnosticsSink)collector);

    /// <summary>
    /// Begins a diagnostics scope that routes recording to the given execution diagnostics.
    /// </summary>
    public static IDisposable BeginScope(ExecutionDiagnostics diagnostics)
        => BeginScope((IOperationDiagnosticsSink)diagnostics);

    sealed class DiagnosticsScope : IDisposable
    {
        readonly IOperationDiagnosticsSink? previous;
        bool disposed;

        internal DiagnosticsScope(IOperationDiagnosticsSink? previous)
        {
            this.previous = previous;
        }

        public void Dispose()
        {
            if (disposed)
                return;
            currentSink.Value = previous;
            disposed = true;
        }
    }

    /// <summary>
    /// Records an operation for diagnostic tracking
    /// </summary>
    /// <param name="diagnostic">The operation diagnostic to record</param>
    internal static void RecordOperation(OperationDiagnostics diagnostic)
    {
        if (diagnostic == null)
            return;

        var sink = currentSink.Value;
        if (sink != null)
        {
            sink.RecordOperation(diagnostic);
            return;
        }

        if (!Volatile.Read(ref globalEnabled))
            return;

        globalCollector.RecordOperation(diagnostic);
    }

    /// <summary>
    /// Starts a measured operation when diagnostics are enabled; otherwise returns a no-op measurement.
    /// </summary>
    internal static OperationMeasurement StartMeasurement()
    {
        var sink = currentSink.Value;
        if (sink != null)
            return new OperationMeasurement(sink);

        if (!Volatile.Read(ref globalEnabled))
            return new OperationMeasurement(null);

        return new OperationMeasurement(globalCollector);
    }

    /// <summary>
    /// Executes and records a measured operation when diagnostics are enabled.
    /// </summary>
    internal static TResult MeasureOperation<TResult>(
        string operationType,
        KernelType kernelUsed,
        int inputLength,
        Type elementType,
        bool hadNulls,
        Func<TResult> operation,
        string? notes = null)
    {
        ArgumentNullException.ThrowIfNull(operation);

        var measurement = StartMeasurement();
        var result = operation();
        measurement.Record(operationType, kernelUsed, inputLength, elementType, hadNulls, notes);
        return result;
    }

    /// <summary>
    /// Gets a copy of all recorded operations
    /// </summary>
    /// <returns>An array of all recorded operation diagnostics</returns>
    public static OperationDiagnostics[] GetRecordedOperations()
    {
        var sink = currentSink.Value;
        if (sink is OperationDiagnosticsCollector collector)
            return collector.GetRecordedOperations();
        if (sink is ExecutionDiagnostics diagnostics)
            return diagnostics.GetRecordedOperations();
        return globalCollector.GetRecordedOperations();
    }

    /// <summary>
    /// Clears all recorded operations
    /// </summary>
    public static void ClearRecordedOperations()
    {
        var sink = currentSink.Value;
        if (sink is OperationDiagnosticsCollector collector)
        {
            collector.ClearRecordedOperations();
            return;
        }
        if (sink is ExecutionDiagnostics diagnostics)
        {
            diagnostics.ClearRecordedOperations();
            return;
        }
        globalCollector.ClearRecordedOperations();
    }

    /// <summary>
    /// Gets summary statistics about recorded operations
    /// </summary>
    /// <returns>A summary of operation statistics</returns>
    public static OperationSummary GetSummary()
    {
        var sink = currentSink.Value;
        if (sink is OperationDiagnosticsCollector collector)
            return collector.GetSummary();
        if (sink is ExecutionDiagnostics diagnostics)
            return diagnostics.GetOperationSummary();
        return globalCollector.GetSummary();
    }

    /// <summary>
    /// Builds an <see cref="OperationSummary"/> from a snapshot of diagnostics.
    /// </summary>
    internal static OperationSummary BuildSummary(OperationDiagnostics[] operations)
    {
        if (operations.Length == 0)
            return new OperationSummary(0, 0, 0, 0, 0, new Dictionary<string, int>(), new Dictionary<KernelType, int>(), new Dictionary<string, long>());

        var totalOperations = operations.Length;
        var vectorizedOperations = operations.Count(op => op.KernelUsed == KernelType.Vectorized);
        var scalarOperations = operations.Count(op => op.KernelUsed == KernelType.Scalar);
        var operationsWithNulls = operations.Count(op => op.HadNulls);
        var totalAllocatedBytes = operations.Sum(op => op.AllocatedBytes);

        var operationTypes = operations
            .GroupBy(op => op.OperationType)
            .ToDictionary(g => g.Key, g => g.Count());

        var kernelTypes = operations
            .GroupBy(op => op.KernelUsed)
            .ToDictionary(g => g.Key, g => g.Count());

        var allocatedBytesByOperationType = operations
            .GroupBy(op => op.OperationType)
            .ToDictionary(g => g.Key, g => g.Sum(op => op.AllocatedBytes));

        return new OperationSummary(
            totalOperations,
            vectorizedOperations,
            scalarOperations,
            operationsWithNulls,
            totalAllocatedBytes,
            operationTypes,
            kernelTypes,
            allocatedBytesByOperationType);
    }
}

/// <summary>
/// Provides summary statistics about recorded operations
/// </summary>
public sealed class OperationSummary
{
    /// <summary>
    /// Initializes a new instance of OperationSummary
    /// </summary>
    internal OperationSummary(
        int totalOperations,
        int vectorizedOperations,
        int scalarOperations,
        int operationsWithNulls,
        long totalAllocatedBytes,
        Dictionary<string, int> operationTypes,
        Dictionary<KernelType, int> kernelTypes,
        Dictionary<string, long> allocatedBytesByOperationType)
    {
        TotalOperations = totalOperations;
        VectorizedOperations = vectorizedOperations;
        ScalarOperations = scalarOperations;
        OperationsWithNulls = operationsWithNulls;
        TotalAllocatedBytes = Math.Max(0, totalAllocatedBytes);
        OperationTypes = operationTypes ?? new Dictionary<string, int>();
        KernelTypes = kernelTypes ?? new Dictionary<KernelType, int>();
        AllocatedBytesByOperationType = allocatedBytesByOperationType ?? new Dictionary<string, long>();
    }

    /// <summary>
    /// Gets the total number of operations recorded
    /// </summary>
    public int TotalOperations { get; }

    /// <summary>
    /// Gets the number of operations that used vectorized kernels
    /// </summary>
    public int VectorizedOperations { get; }

    /// <summary>
    /// Gets the number of operations that used scalar kernels
    /// </summary>
    public int ScalarOperations { get; }

    /// <summary>
    /// Gets the number of operations that involved null values
    /// </summary>
    public int OperationsWithNulls { get; }

    /// <summary>
    /// Gets the total managed bytes allocated by recorded measured operations.
    /// </summary>
    public long TotalAllocatedBytes { get; }

    /// <summary>
    /// Gets the average managed bytes allocated per recorded operation.
    /// </summary>
    public double AverageAllocatedBytes => TotalOperations > 0 ? (double)TotalAllocatedBytes / TotalOperations : 0;

    /// <summary>
    /// Gets the count of operations by type
    /// </summary>
    public Dictionary<string, int> OperationTypes { get; }

    /// <summary>
    /// Gets the count of operations by kernel type
    /// </summary>
    public Dictionary<KernelType, int> KernelTypes { get; }

    /// <summary>
    /// Gets allocated bytes grouped by operation type.
    /// </summary>
    public Dictionary<string, long> AllocatedBytesByOperationType { get; }

    /// <summary>
    /// Gets the percentage of operations that used vectorization
    /// </summary>
    public double VectorizationRate => TotalOperations > 0 ? (double)VectorizedOperations / TotalOperations * 100 : 0;

    /// <summary>
    /// Returns a string representation of the operation summary
    /// </summary>
    /// <returns>A formatted string with summary statistics</returns>
    public override string ToString()
        => $"OperationSummary {{ " +
        $"Total: {TotalOperations:N0}, " +
        $"Vectorized: {VectorizedOperations:N0} ({VectorizationRate:F1}%), " +
        $"Scalar: {ScalarOperations:N0}, " +
        $"WithNulls: {OperationsWithNulls:N0}, " +
        $"AllocatedBytes: {TotalAllocatedBytes:N0} " +
        $"}}";
}
