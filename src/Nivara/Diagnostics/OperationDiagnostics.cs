using Nivara.Storage;

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
    {
        OperationType = operationType ?? throw new ArgumentNullException(nameof(operationType));
        KernelUsed = kernelUsed;
        InputLength = inputLength;
        ElementType = elementType ?? throw new ArgumentNullException(nameof(elementType));
        HadNulls = hadNulls;
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
                {
                    return $"Type {ElementType.Name} is not vectorizable";
                }
                if (!System.Numerics.Vector.IsHardwareAccelerated)
                {
                    return "Hardware acceleration not available";
                }
                if (InputLength < System.Numerics.Vector<byte>.Count * 4)
                {
                    return $"Input length {InputLength} too small for vectorization overhead";
                }
                return "Scalar kernel selected for unknown reason";
            }
            else
            {
                return $"Vectorized kernel selected for {ElementType.Name} with length {InputLength}";
            }
        }
    }

    /// <summary>
    /// Returns a string representation of the operation diagnostics
    /// </summary>
    /// <returns>A formatted string with operation details</returns>
    public override string ToString()
    {
        return $"OperationDiagnostics {{ " +
               $"Operation: {OperationType}, " +
               $"Kernel: {KernelUsed}, " +
               $"Type: {ElementType.Name}, " +
               $"Length: {InputLength:N0}, " +
               $"HadNulls: {HadNulls}, " +
               $"Reason: {KernelSelectionReason} " +
               $"}}";
    }
}

/// <summary>
/// Tracks operation diagnostics for performance analysis.
/// Provides a thread-safe way to collect diagnostic information about column operations.
/// </summary>
public static class DiagnosticsTracker
{
    private static readonly object lockObject = new object();
    private static readonly List<OperationDiagnostics> operations = new List<OperationDiagnostics>();
    private static bool isEnabled = false;

    /// <summary>
    /// Gets or sets a value indicating whether diagnostic tracking is enabled
    /// </summary>
    public static bool IsEnabled
    {
        get
        {
            lock (lockObject)
            {
                return isEnabled;
            }
        }
        set
        {
            lock (lockObject)
            {
                isEnabled = value;
            }
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

        lock (lockObject)
        {
            if (isEnabled)
            {
                operations.Add(diagnostic);
            }
        }
    }

    /// <summary>
    /// Gets a copy of all recorded operations
    /// </summary>
    /// <returns>An array of all recorded operation diagnostics</returns>
    public static OperationDiagnostics[] GetRecordedOperations()
    {
        lock (lockObject)
        {
            return operations.ToArray();
        }
    }

    /// <summary>
    /// Clears all recorded operations
    /// </summary>
    public static void ClearRecordedOperations()
    {
        lock (lockObject)
        {
            operations.Clear();
        }
    }

    /// <summary>
    /// Gets summary statistics about recorded operations
    /// </summary>
    /// <returns>A summary of operation statistics</returns>
    public static OperationSummary GetSummary()
    {
        lock (lockObject)
        {
            if (operations.Count == 0)
            {
                return new OperationSummary(0, 0, 0, 0, new Dictionary<string, int>(), new Dictionary<KernelType, int>());
            }

            var totalOperations = operations.Count;
            var vectorizedOperations = operations.Count(op => op.KernelUsed == KernelType.Vectorized);
            var scalarOperations = operations.Count(op => op.KernelUsed == KernelType.Scalar);
            var operationsWithNulls = operations.Count(op => op.HadNulls);

            var operationTypes = operations
                .GroupBy(op => op.OperationType)
                .ToDictionary(g => g.Key, g => g.Count());

            var kernelTypes = operations
                .GroupBy(op => op.KernelUsed)
                .ToDictionary(g => g.Key, g => g.Count());

            return new OperationSummary(
                totalOperations,
                vectorizedOperations,
                scalarOperations,
                operationsWithNulls,
                operationTypes,
                kernelTypes);
        }
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
        Dictionary<string, int> operationTypes,
        Dictionary<KernelType, int> kernelTypes)
    {
        TotalOperations = totalOperations;
        VectorizedOperations = vectorizedOperations;
        ScalarOperations = scalarOperations;
        OperationsWithNulls = operationsWithNulls;
        OperationTypes = operationTypes ?? new Dictionary<string, int>();
        KernelTypes = kernelTypes ?? new Dictionary<KernelType, int>();
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
    /// Gets the count of operations by type
    /// </summary>
    public Dictionary<string, int> OperationTypes { get; }

    /// <summary>
    /// Gets the count of operations by kernel type
    /// </summary>
    public Dictionary<KernelType, int> KernelTypes { get; }

    /// <summary>
    /// Gets the percentage of operations that used vectorization
    /// </summary>
    public double VectorizationRate => TotalOperations > 0 ? (double)VectorizedOperations / TotalOperations * 100 : 0;

    /// <summary>
    /// Returns a string representation of the operation summary
    /// </summary>
    /// <returns>A formatted string with summary statistics</returns>
    public override string ToString()
    {
        return $"OperationSummary {{ " +
               $"Total: {TotalOperations:N0}, " +
               $"Vectorized: {VectorizedOperations:N0} ({VectorizationRate:F1}%), " +
               $"Scalar: {ScalarOperations:N0}, " +
               $"WithNulls: {OperationsWithNulls:N0} " +
               $"}}";
    }
}