using Nivara.Diagnostics;
using Nivara.Exceptions;
using Nivara.Helpers;
using Nivara.Operations;
using Nivara.Storage;
using Nivara.Tensors;
using System.Collections.Concurrent;
using System.Linq.Expressions;
using System.Numerics;
using System.Reflection;
using System.Runtime.InteropServices;

namespace Nivara.Expressions;

/// <summary>
/// Fused evaluator: lowers a validated <see cref="ColumnExpression"/> AST into a single-pass kernel
/// over the whole column with no intermediate columns and no per-element <c>object?</c> boxing.
///
/// The primary target compiles the tree with <see cref="System.Linq.Expressions"/> into a cached
/// delegate over typed leaf arrays. Ref structs (<c>Span&lt;T&gt;</c>) are prohibited in expression
/// trees (confirmed via MS Learn), so the delegate consumes <c>T[]</c> arrays rather than spans; the
/// straight-line loop body is emitted by the JIT and auto-vectorizes for numeric leaves. A generic
/// node-tree kernel (<see cref="FusedKernel"/>) is the fallback when the compiled target cannot be
/// built. Expressions that cannot be fused at all throw (no boxed fallback). Null masks are OR'd from
/// the leaf masks in a separate pass; comparisons produce masked-false at nulls (SQL-like semantics),
/// matching the legacy evaluator.
/// </summary>
sealed class FusedExpressionEvaluator
{
    int fusedPathEvaluationCount;
    int compiledPathEvaluationCount;
    int spanKernelPathEvaluationCount;
    int tensorPrimitivesPathEvaluationCount;

    /// <summary>
    /// Gets how many fused evaluations were applied by the most recent operations on this instance.
    /// Guardrail tests assert the fused path is actually selected (claims-integrity guardrail).
    /// </summary>
    internal int FusedPathEvaluationCount => fusedPathEvaluationCount;

    /// <summary>
    /// Gets how many of those evaluations ran through the compiled (expression-tree) target.
    /// </summary>
    internal int CompiledPathEvaluationCount => compiledPathEvaluationCount;

    /// <summary>
    /// Gets how many of those evaluations ran through the generic span-kernel interpreter.
    /// </summary>
    internal int SpanKernelPathEvaluationCount => spanKernelPathEvaluationCount;

    /// <summary>
    /// Gets how many of those evaluations ran through the TensorPrimitives single-op backend.
    /// </summary>
    internal int TensorPrimitivesPathEvaluationCount => tensorPrimitivesPathEvaluationCount;

    delegate void CompiledFusedInvoke(object[] leafArrays, Array dest, int start, int count, int destStart, bool[]? mask);

    static readonly ConcurrentDictionary<string, CompiledFusedInvoke> compiledKernelCache = new();

    static readonly ConcurrentDictionary<Type, MethodInfo> snapshotKernelCache = new();

    static readonly ConcurrentDictionary<Type, MethodInfo> createColumnKernelCache = new();

    static readonly ConcurrentDictionary<Type, MethodInfo> createFromSpansKernelCache = new();

    static readonly ConcurrentDictionary<Type, MethodInfo> allocateArrayKernelCache = new();

    static readonly ConcurrentDictionary<(Type Source, Type Target), MethodInfo> numericConvertKernelCache = new();

    /// <summary>
    /// Evaluates a column expression and returns the result column through the fused path.
    /// </summary>
    /// <param name="expression">The expression to evaluate</param>
    /// <param name="input">The input columns</param>
    /// <returns>The result column</returns>
    /// <exception cref="QueryExecutionException">Thrown when evaluation fails</exception>
    public IColumn Evaluate(ColumnExpression expression, IReadOnlyDictionary<string, IColumn> input)
    {
        ArgumentNullException.ThrowIfNull(expression);
        ArgumentNullException.ThrowIfNull(input);

        try
        {
            return EvaluateCore(expression, input, null);
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Failed to evaluate expression '{expression.Name}': {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Evaluates a column expression in row-batches of <paramref name="chunkSize"/> through the fused
    /// kernels, producing a single result column. Leaf data is never copied: each backend slices the
    /// existing column storage per chunk (issue #167). Bit-identical to whole-column evaluation.
    /// </summary>
    /// <param name="expression">The expression to evaluate</param>
    /// <param name="input">The input columns</param>
    /// <param name="chunkSize">The row-batch size</param>
    /// <returns>The result column</returns>
    /// <exception cref="QueryExecutionException">Thrown when evaluation fails</exception>
    internal IColumn EvaluateChunked(ColumnExpression expression, IReadOnlyDictionary<string, IColumn> input, int chunkSize)
    {
        ArgumentNullException.ThrowIfNull(expression);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(chunkSize);

        try
        {
            return EvaluateCore(expression, input, chunkSize);
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Failed to evaluate expression '{expression.Name}': {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Evaluates a column expression that should return a boolean result.
    /// </summary>
    /// <param name="expression">The expression to evaluate</param>
    /// <param name="input">The input columns</param>
    /// <returns>A boolean column with the evaluation results</returns>
    /// <exception cref="QueryExecutionException">Thrown when evaluation fails or result is not boolean</exception>
    public NivaraColumn<bool> EvaluateBoolean(ColumnExpression expression, IReadOnlyDictionary<string, IColumn> input)
    {
        if (IsVacuousOverEmptyColumns(expression, input))
            return NivaraColumn<bool>.Create(Array.Empty<bool>());

        var result = Evaluate(expression, input);

        if (result is not NivaraColumn<bool> boolColumn)
        {
            throw new QueryExecutionException($"Expression '{expression.Name}' must evaluate to a boolean column, but got {result.ElementType.Name}");
        }

        return boolColumn;
    }

    /// <summary>
    /// Determines whether a boolean expression is vacuously empty because every column it references
    /// exists and has zero rows. The legacy evaluator produced an empty boolean column through its
    /// boxed loop in this case; the fused evaluator would reject non-fusable operand combinations
    /// (e.g. a string vs int comparison on an empty CSV column) even though there is no data to
    /// compare, so short-circuit to an empty result to preserve legacy semantics.
    /// </summary>
    static bool IsVacuousOverEmptyColumns(ColumnExpression expression, IReadOnlyDictionary<string, IColumn> input)
    {
        var references = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        CollectColumnReferences(expression, references);

        if (references.Count == 0)
            return false;

        foreach (var name in references)
        {
            if (!input.TryGetValue(name, out var column) || column.Length != 0)
                return false;
        }

        return true;
    }

    static void CollectColumnReferences(ColumnExpression node, HashSet<string> references)
    {
        switch (node)
        {
            case ColumnReference columnRef:
                references.Add(columnRef.ColumnName);
                break;
            case ScalarExpression scalar:
                CollectColumnReferences(scalar.Column, references);
                break;
            case BinaryExpression binary:
                CollectColumnReferences(binary.Left, references);
                CollectColumnReferences(binary.Right, references);
                break;
            case ComparisonExpression comparison:
                CollectColumnReferences(comparison.Left, references);
                CollectColumnReferences(comparison.Right, references);
                break;
            case NotExpression not:
                CollectColumnReferences(not.Operand, references);
                break;
            case WindowExpression window:
                if (window.Source is not null)
                    CollectColumnReferences(window.Source, references);
                foreach (var key in window.OrderBy)
                    CollectColumnReferences(key.Key, references);
                foreach (var partition in window.PartitionBy)
                    CollectColumnReferences(partition, references);
                break;
        }
    }

    IColumn EvaluateCore(ColumnExpression expression, IReadOnlyDictionary<string, IColumn> input, int? chunkSize)
    {
        // Hydrate window expressions before any kernel planning: each window node is materialized
        // through its own evaluation (rolling/cumulative/shift/rank kernels over the computed source
        // or key expressions) and replaced with a reference to a synthetic column injected into the
        // input dictionary. The surrounding elementwise expression then fuses over the materialized
        // window column in a single pass. Nested windows compose because the inner evaluations recurse
        // through this same method.
        if (ContainsWindowExpression(expression))
        {
            var synthetic = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
            expression = HydrateWindows(expression, input, synthetic);

            if (synthetic.Count > 0)
            {
                var combined = new Dictionary<string, IColumn>(input, StringComparer.OrdinalIgnoreCase);
                foreach (var kvp in synthetic)
                    combined[kvp.Key] = kvp.Value;
                input = combined;
            }
        }

        switch (expression)
        {
            // Trivial passthroughs: a bare column reference or literal needs no kernel.
            case ColumnReference columnRef:
                {
                    var direct = input[columnRef.ColumnName];
                    fusedPathEvaluationCount++;
                    RecordDiagnostics(direct.Length, direct.ElementType, direct.HasNulls, ColumnStorageFactory.IsVectorizable(direct.ElementType), "Direct column passthrough");
                    return direct;
                }

            case LiteralExpression literal:
                {
                    var constantLength = input.Values.FirstOrDefault()?.Length ?? 1;
                    var constant = CreateConstantColumn(literal.Value, constantLength);
                    fusedPathEvaluationCount++;
                    RecordDiagnostics(constantLength, constant.ElementType, constant.HasNulls, ColumnStorageFactory.IsVectorizable(constant.ElementType), "Constant column");
                    return constant;
                }
        }

        var plan = ExpressionTypeInferer.TryInfer(expression, input);
        if (plan == null)
        {
            throw new NotSupportedException(
                $"Expression '{expression.Name}' cannot run through the fused evaluator: unsupported operand combination");
        }

        // Literal-only expressions have no leaf columns to size or route against (e.g. `Lit(2) * 2`).
        // They constant-fold through the compiled target at the input length instead of being rejected
        // as unsupported (issue #249); the span and TensorPrimitives backends both require a column leaf.
        if (plan.Columns.Count == 0)
        {
            var constantLength = input.Values.FirstOrDefault()?.Length ?? 1;
            var constantPlan = KernelLowerer.Lower(expression, plan);
            return EvaluateCompiled(expression, plan, constantPlan, constantLength, chunkSize);
        }

        ValidateLeafLengths(plan);
        var length = plan.Columns[0].Column.Length;

        var kernelPlan = KernelLowerer.Lower(expression, plan);

        // Span kernel first: null-bearing uniform generic-math plans fuse values and the OR'd null
        // mask in a single zero-copy pass — no leaf snapshots, no separate mask pass. Null-free
        // single-op uniform plans skip straight to the TensorPrimitives SIMD kernel. Everything else
        // (null-free chains, heterogeneous, and every bool-result plan) stays on the compiled path,
        // preserving the JIT-vectorized fused win for the common null-free case.
        if (kernelPlan.IsUniformNumeric && kernelPlan.HasNulls)
        {
            return EvaluateNodeTree(expression, plan, kernelPlan, length, chunkSize);
        }

        if (kernelPlan.IsTensorPrimitivesCandidate && !kernelPlan.HasNulls)
        {
            return EvaluateTensorPrimitives(expression, plan, kernelPlan, length, chunkSize);
        }

        return EvaluateCompiled(expression, plan, kernelPlan, length, chunkSize);
    }

    /// <summary>
    /// Evaluates through the compiled expression-tree target: whole-column in one call, or batched by
    /// <paramref name="chunkSize"/> into the shared result array through the offset-based delegate.
    /// </summary>
    IColumn EvaluateCompiled(ColumnExpression expression, FusedExpressionPlan plan, KernelPlan kernelPlan, int length, int? chunkSize)
    {
        var leafArrays = SnapshotLeaves(plan);

        CompiledFusedInvoke invoke;
        try
        {
            invoke = compiledKernelCache.GetOrAdd(plan.Signature, key => BuildCompiledDelegate(kernelPlan));
            compiledPathEvaluationCount++;
        }
        catch (NotSupportedException)
        {
            return EvaluateNodeTree(expression, plan, kernelPlan, length, null);
        }

        var resultArray = AllocateResultArray(plan.ResultType, length);

        // OR the leaf null masks up front (null-propagation rule) and hand them to the delegate:
        // the compiled loop short-circuits masked positions to default(T) instead of computing a
        // value there. The span kernel already does this; without the mask the compiled path would
        // evaluate every position (e.g. masked decimalCol / intCol with a null int → DivideByZero)
        // and write real values behind the mask (issue #247).
        var mask = plan.HasNulls ? ComputeMask(plan, length) : null;

        if (chunkSize == null || chunkSize.Value >= length)
        {
            invoke(leafArrays, resultArray, 0, length, 0, mask);
        }
        else
        {
            for (var start = 0; start < length; start += chunkSize.Value)
            {
                var count = Math.Min(chunkSize.Value, length - start);
                invoke(leafArrays, resultArray, start, count, start, mask);
            }
        }

        fusedPathEvaluationCount++;
        var vectorizable = plan.Columns.All(l => ColumnStorageFactory.IsVectorizable(l.Column.ElementType))
            && ColumnStorageFactory.IsVectorizable(plan.ResultType);
        RecordDiagnostics(length, plan.ResultType, plan.HasNulls, vectorizable, "Compiled fused kernel");
        return CreateResultColumn(plan.ResultType, resultArray, mask);
    }

    /// <summary>
    /// Creates a typed result array for the compiled delegate to write into. The delegate is shared
    /// across evaluations, so the caller owns the allocation (whole-column and per-chunk alike).
    /// </summary>
    static Array AllocateResultArray(Type elementType, int length)
    {
        var kernel = allocateArrayKernelCache.GetOrAdd(elementType, static t =>
            typeof(FusedExpressionEvaluator).GetMethod(nameof(AllocateResultArrayOf), BindingFlags.Static | BindingFlags.NonPublic)!
                .MakeGenericMethod(t));
        return (Array)kernel.Invoke(null, new object?[] { length })!;
    }

    static T[] AllocateResultArrayOf<T>(int length) => new T[length];

    internal static bool ContainsWindowExpression(ColumnExpression node)
    {
        switch (node)
        {
            case WindowExpression:
                return true;
            case ScalarExpression scalar:
                return ContainsWindowExpression(scalar.Column);
            case BinaryExpression binary:
                return ContainsWindowExpression(binary.Left) || ContainsWindowExpression(binary.Right);
            case ComparisonExpression comparison:
                return ContainsWindowExpression(comparison.Left) || ContainsWindowExpression(comparison.Right);
            case NotExpression not:
                return ContainsWindowExpression(not.Operand);
            default:
                return false;
        }
    }

    /// <summary>
    /// Rewrites a tree, replacing each window node with a reference to a synthetic column that
    /// holds the materialized window result. The rewritten tree contains only plain elementwise
    /// nodes, so it flows through the standard fused kernel planning unchanged.
    /// </summary>
    ColumnExpression HydrateWindows(ColumnExpression node, IReadOnlyDictionary<string, IColumn> input, Dictionary<string, IColumn> synthetic)
    {
        switch (node)
        {
            case WindowExpression window:
                {
                    var result = MaterializeWindow(window, input);
                    var name = SyntheticWindowPrefix + synthetic.Count;
                    synthetic[name] = result;
                    return new ColumnReference(name, result.ElementType);
                }

            case ScalarExpression scalar:
                return new ScalarExpression(scalar.Operator, HydrateWindows(scalar.Column, input, synthetic), scalar.Scalar);

            case BinaryExpression binary:
                return new BinaryExpression(binary.Operator, HydrateWindows(binary.Left, input, synthetic), HydrateWindows(binary.Right, input, synthetic));

            case ComparisonExpression comparison:
                return new ComparisonExpression(comparison.Operator, HydrateWindows(comparison.Left, input, synthetic), HydrateWindows(comparison.Right, input, synthetic));

            case NotExpression not:
                return new NotExpression(HydrateWindows(not.Operand, input, synthetic));

            default:
                return node;
        }
    }

    /// <summary>
    /// Materializes a window expression into its result column using the same kernels as the eager
    /// frame path and the pipeline operations. Rank-family kinds materialize each order/partition
    /// key expression into a synthetic column and feed them to <see cref="RankKernel"/>.
    /// </summary>
    IColumn MaterializeWindow(WindowExpression window, IReadOnlyDictionary<string, IColumn> input)
    {
        if (WindowFunctionHelpers.IsRankFamily(window.Kind))
        {
            var keyColumns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
            var orderKeys = new List<SortKey>();
            var partitionNames = new List<string>();
            var index = 0;

            foreach (var orderKey in window.OrderBy)
            {
                var name = SyntheticWindowPrefix + index++;
                keyColumns[name] = Evaluate(orderKey.Key, input);
                orderKeys.Add(new SortKey(name, orderKey.Direction, orderKey.NullOrdering));
            }

            foreach (var partition in window.PartitionBy)
            {
                var name = SyntheticWindowPrefix + index++;
                keyColumns[name] = Evaluate(partition, input);
                partitionNames.Add(name);
            }

            return RankKernel.Compute(keyColumns, partitionNames.ToArray(), orderKeys, WindowFunctionHelpers.ToRankKind(window.Kind));
        }

        var source = Evaluate(window.Source!, input);
        switch (window.Kind)
        {
            case WindowFunctionKind.RollingSum:
            case WindowFunctionKind.RollingMean:
            case WindowFunctionKind.RollingMin:
            case WindowFunctionKind.RollingMax:
                return NivaraFrameExtensions.CalculateRolling(source, window.WindowSize!.Value, window.MinPeriods, window.NullHandler, WindowFunctionHelpers.ToRollingKind(window.Kind));

            case WindowFunctionKind.CumulativeCount:
                return NivaraFrameExtensions.CalculateCumulativeCount(source);

            case WindowFunctionKind.CumulativeSum:
            case WindowFunctionKind.CumulativeMax:
            case WindowFunctionKind.CumulativeMin:
            case WindowFunctionKind.CumulativeProduct:
                return NivaraFrameExtensions.CalculateCumulative(source, window.NullHandler, WindowFunctionHelpers.ToCumulativeKind(window.Kind));

            case WindowFunctionKind.Shift:
            case WindowFunctionKind.Lead:
                return NivaraFrameExtensions.CalculateShift(source, window.Kind == WindowFunctionKind.Lead ? -window.Periods!.Value : window.Periods!.Value, window.FillValue);

            default:
                throw new NotSupportedException($"Window kind {window.Kind} is not supported by the fused evaluator");
        }
    }

    static string SyntheticWindowPrefix => "__window_";

    /// <summary>
    /// Evaluates through the generic span kernel: primary path for null-bearing uniform generic-math
    /// plans, and fallback when the expression-tree-compiled target cannot be built. Requires generic
    /// math and leaf columns already sharing the result element type. The null mask is fused into the
    /// kernel (OR semantics), so no separate <see cref="ComputeMask"/> pass runs.
    /// </summary>
    IColumn EvaluateNodeTree(ColumnExpression expression, FusedExpressionPlan plan, KernelPlan kernelPlan, int length, int? chunkSize)
    {
        if (!kernelPlan.IsUniformNumeric)
        {
            throw new NotSupportedException(
                $"Expression '{expression.Name}' is not supported by the fused evaluator for element type {plan.ResultType.Name}");
        }

        var runner = GetNodeTreeRunner(plan.ResultType);
        var result = runner(kernelPlan, chunkSize);
        spanKernelPathEvaluationCount++;
        fusedPathEvaluationCount++;
        RecordDiagnostics(length, plan.ResultType, plan.HasNulls, ColumnStorageFactory.IsVectorizable(plan.ResultType), "Span fused kernel");
        return result;
    }

    static class FusedNodeTreeRunner<T>
        where T : struct, INumber<T>
    {
        internal static IColumn Run(KernelPlan plan, int? chunkSize)
            => FusedKernel.Evaluate<T>(plan, chunkSize);
    }

    delegate IColumn FusedNodeTreeInvoker(KernelPlan plan, int? chunkSize);

    static readonly ConcurrentDictionary<Type, FusedNodeTreeInvoker> nodeTreeRunnerCache = new();

    static FusedNodeTreeInvoker GetNodeTreeRunner(Type elementType)
    {
        return nodeTreeRunnerCache.GetOrAdd(elementType, static t =>
        {
            var method = typeof(FusedNodeTreeRunner<>).MakeGenericType(t)
                .GetMethod(nameof(FusedNodeTreeRunner<int>.Run), BindingFlags.Static | BindingFlags.NonPublic)!;
            return (FusedNodeTreeInvoker)method.CreateDelegate(typeof(FusedNodeTreeInvoker));
        });
    }

    /// <summary>
    /// Evaluates through the TensorPrimitives single-op backend: null-free uniform plans that are a
    /// single Add/Subtract/Multiply/Divide over leaves and literals dispatch to the SIMD-vectorized
    /// <see cref="System.Numerics.Tensors.TensorPrimitives"/> overloads in one call.
    /// </summary>
    IColumn EvaluateTensorPrimitives(ColumnExpression expression, FusedExpressionPlan plan, KernelPlan kernelPlan, int length, int? chunkSize)
    {
        if (!TensorPrimitivesKernel.IsDispatchable(kernelPlan))
        {
            throw new NotSupportedException(
                $"Expression '{expression.Name}' is not dispatchable to the TensorPrimitives single-op backend");
        }

        var runner = GetTensorPrimitivesRunner(plan.ResultType);
        var result = runner(kernelPlan, chunkSize);
        tensorPrimitivesPathEvaluationCount++;
        fusedPathEvaluationCount++;
        RecordDiagnostics(length, plan.ResultType, plan.HasNulls, ColumnStorageFactory.IsVectorizable(plan.ResultType), "TensorPrimitives fused kernel");
        return result;
    }

    static class TensorPrimitivesRunner<T>
        where T : struct, INumber<T>
    {
        internal static IColumn Run(KernelPlan plan, int? chunkSize)
        {
            var leaves = new NivaraColumn<T>[plan.Columns.Count];
            for (int i = 0; i < plan.Columns.Count; i++)
                leaves[i] = (NivaraColumn<T>)plan.Columns[i].Column;

            var length = leaves.Length == 0 ? 1 : leaves[0].Length;
            var ok = chunkSize == null
                ? TensorPrimitivesKernel.TryEvaluate<T>(plan, leaves, length, out var result)
                : TensorPrimitivesKernel.TryEvaluateChunked<T>(plan, leaves, length, chunkSize.Value, out result);
            if (!ok)
                throw new NotSupportedException($"Plan is not dispatchable to the TensorPrimitives single-op backend for element type {typeof(T).Name}");
            return result;
        }
    }

    delegate IColumn TensorPrimitivesInvoker(KernelPlan plan, int? chunkSize);

    static readonly ConcurrentDictionary<Type, TensorPrimitivesInvoker> tensorPrimitivesRunnerCache = new();

    static TensorPrimitivesInvoker GetTensorPrimitivesRunner(Type elementType)
    {
        return tensorPrimitivesRunnerCache.GetOrAdd(elementType, static t =>
        {
            var method = typeof(TensorPrimitivesRunner<>).MakeGenericType(t)
                .GetMethod(nameof(TensorPrimitivesRunner<int>.Run), BindingFlags.Static | BindingFlags.NonPublic)!;
            return (TensorPrimitivesInvoker)method.CreateDelegate(typeof(TensorPrimitivesInvoker));
        });
    }

    /// <summary>
    /// Ensures every leaf column has the same length before the fused kernel runs.
    /// </summary>
    static void ValidateLeafLengths(FusedExpressionPlan plan)
    {
        var length = plan.Columns[0].Column.Length;
        foreach (var leaf in plan.Columns)
        {
            if (leaf.Column.Length != length)
            {
                throw new ArgumentException("Columns must have the same length for fused expression evaluation");
            }
        }
    }

    /// <summary>
    /// Snaps each leaf column to a typed array the compiled delegate can index, reading zero-copy
    /// when the leaf's backing array is contiguous at offset 0. Null positions hold <c>default(T)</c>
    /// in the backing data, so a zero-copy view is always value-correct; the null mask is computed
    /// separately. Only sliced columns (offset &gt; 0) require a snapshot copy.
    /// </summary>
    static object[] SnapshotLeaves(FusedExpressionPlan plan)
    {
        var leafArrays = new object[plan.Columns.Count];
        for (int i = 0; i < plan.Columns.Count; i++)
        {
            var column = plan.Columns[i].Column;
            var kernel = GetSnapshotKernel(column.ElementType);
            leafArrays[i] = kernel.Invoke(null, new object?[] { column })!;
        }

        return leafArrays;
    }

    static MethodInfo GetSnapshotKernel(Type elementType)
    {
        return snapshotKernelCache.GetOrAdd(elementType, static t =>
            typeof(FusedExpressionEvaluator).GetMethod(nameof(SnapshotLeaf), BindingFlags.Static | BindingFlags.NonPublic)!
                .MakeGenericMethod(t));
    }

    static T[] SnapshotLeaf<T>(NivaraColumn<T> column)
    {
        if (MemoryMarshal.TryGetArray(column.Storage.Data, out var segment) && segment.Array is not null && segment.Offset == 0)
            return segment.Array;

        var array = new T[column.Length];
        column.AsSpan().CopyTo(array);
        return array;
    }

    /// <summary>
    /// ORs the null masks of leaf columns into a single result mask (null-propagation rule).
    /// </summary>
    static bool[]? ComputeMask(FusedExpressionPlan plan, int length)
    {
        var mask = new bool[length];
        foreach (var leaf in plan.Columns)
        {
            if (!leaf.Column.HasNulls)
                continue;

            for (int i = 0; i < length; i++)
                mask[i] |= leaf.Column.IsNull(i);
        }

        return mask;
    }

    /// <summary>
    /// Creates the typed result column from the fused kernel output.
    /// </summary>
    static IColumn CreateResultColumn(Type elementType, Array data, bool[]? mask)
    {
        if (mask == null)
        {
            var kernel = createColumnKernelCache.GetOrAdd(elementType, static t =>
                typeof(FusedExpressionEvaluator).GetMethod(nameof(CreateColumn), BindingFlags.Static | BindingFlags.NonPublic)!
                    .MakeGenericMethod(t));
            return (IColumn)kernel.Invoke(null, new object?[] { data })!;
        }

        var spansKernel = createFromSpansKernelCache.GetOrAdd(elementType, static t =>
            typeof(FusedExpressionEvaluator).GetMethod(nameof(CreateColumnFromSpans), BindingFlags.Static | BindingFlags.NonPublic)!
                .MakeGenericMethod(t));
        return (IColumn)spansKernel.Invoke(null, new object?[] { data, mask })!;
    }

    static NivaraColumn<T> CreateColumn<T>(T[] data) => NivaraColumn<T>.Create(data);

    static NivaraColumn<T> CreateColumnFromSpans<T>(T[] data, bool[] mask) => NivaraColumn<T>.CreateFromSpans(data, mask);

    /// <summary>
    /// Creates a constant column with the specified value repeated for the given length.
    /// </summary>
    static IColumn CreateConstantColumn(object? value, int length)
    {
        if (value == null)
        {
            var nullArray = new object?[length];
            return NivaraColumn<object?>.Create(nullArray);
        }

        if (value is string stringValue)
        {
            var array = new string[length];
            Array.Fill(array, stringValue);
            return NivaraColumn<string>.Create(array);
        }

        var constantValues = new object[length];
        Array.Fill(constantValues, value);
        return ColumnFactory.Create(value.GetType(), constantValues);
    }

    /// <summary>
    /// Records the kernel route for an evaluation into the active diagnostics tracker.
    /// </summary>
    static void RecordDiagnostics(int length, Type elementType, bool hasNulls, bool vectorizable, string message)
    {
        if (!DiagnosticsTracker.IsEnabled)
            return;

        DiagnosticsTracker.RecordOperation(new OperationDiagnostics(
            "FusedExpressionEvaluation",
            vectorizable ? KernelType.Vectorized : KernelType.Scalar,
            length,
            elementType,
            hasNulls,
            0,
            TimeSpan.Zero,
            message));
    }

    /// <summary>
    /// Builds a compiled <c>Action&lt;T1[], ..., TN[], R[], int, int, int&gt;</c> delegate that runs the
    /// whole expression over the leaf arrays in a single loop, writing into a caller-provided result
    /// array at an offset, plus a cached typed invocation wrapper
    /// (<see cref="CompiledFusedInvoke"/>) that casts the leaf arrays and calls the concrete action
    /// directly (no per-evaluation <see cref="Delegate.DynamicInvoke"/>). The loop reads
    /// <c>leaf[i + start]</c> and writes <c>dest[destStart + i]</c>, so one cached delegate serves
    /// whole-column and chunked execution alike (issue #167).
    /// </summary>
    static CompiledFusedInvoke BuildCompiledDelegate(KernelPlan plan)
    {
        var leafParams = new ParameterExpression[plan.Columns.Count];
        for (int i = 0; i < plan.Columns.Count; i++)
            leafParams[i] = Expression.Parameter(plan.Columns[i].Column.ElementType.MakeArrayType(), "leaf" + i);

        var destParam = Expression.Parameter(plan.ResultType.MakeArrayType(), "dest");
        var startParam = Expression.Parameter(typeof(int), "start");
        var countParam = Expression.Parameter(typeof(int), "count");
        var destStartParam = Expression.Parameter(typeof(int), "destStart");
        var maskParam = Expression.Parameter(typeof(bool[]), "mask");
        var indexVar = Expression.Parameter(typeof(int), "i");

        var value = BuildCompiledNode(plan, plan.RootNode, leafParams, indexVar, startParam);
        var loopBody = BuildOffsetForLoop(indexVar, value, destParam, destStartParam, countParam, maskParam, plan.ResultType);

        var paramTypes = new Type[plan.Columns.Count + 5];
        for (int i = 0; i < plan.Columns.Count; i++)
            paramTypes[i] = leafParams[i].Type;
        paramTypes[plan.Columns.Count] = destParam.Type;
        paramTypes[plan.Columns.Count + 1] = startParam.Type;
        paramTypes[plan.Columns.Count + 2] = countParam.Type;
        paramTypes[plan.Columns.Count + 3] = destStartParam.Type;
        paramTypes[plan.Columns.Count + 4] = maskParam.Type;

        var actionType = Expression.GetActionType(paramTypes);
        var body = Expression.Block(new[] { indexVar }, loopBody);
        var lambda = Expression.Lambda(actionType, body, leafParams.Append(destParam).Append(startParam).Append(countParam).Append(destStartParam).Append(maskParam));
        var typedAction = lambda.Compile();

        var argsParam = Expression.Parameter(typeof(object[]), "args");
        var destObjectParam = Expression.Parameter(typeof(Array), "dest");
        var startObjectParam = Expression.Parameter(typeof(int), "start");
        var countObjectParam = Expression.Parameter(typeof(int), "count");
        var destStartObjectParam = Expression.Parameter(typeof(int), "destStart");
        var maskObjectParam = Expression.Parameter(typeof(bool[]), "mask");

        var callArgs = new Expression[leafParams.Length + 5];
        for (int i = 0; i < leafParams.Length; i++)
            callArgs[i] = Expression.Convert(Expression.ArrayIndex(argsParam, Expression.Constant(i)), leafParams[i].Type);
        callArgs[leafParams.Length] = Expression.Convert(destObjectParam, destParam.Type);
        callArgs[leafParams.Length + 1] = startObjectParam;
        callArgs[leafParams.Length + 2] = countObjectParam;
        callArgs[leafParams.Length + 3] = destStartObjectParam;
        callArgs[leafParams.Length + 4] = maskObjectParam;

        var wrapperBody = Expression.Invoke(Expression.Constant(typedAction), callArgs);

        return Expression.Lambda<CompiledFusedInvoke>(wrapperBody, argsParam, destObjectParam, startObjectParam, countObjectParam, destStartObjectParam, maskObjectParam).Compile();
    }

    /// <summary>
    /// Builds the per-row value expression for an IR node, returning an expression of the node's
    /// compute type. Numeric nodes promote their operands (C# binary numeric promotion via
    /// <see cref="NumericPromoter"/>) and insert widening conversions on the leaf/literal operands.
    /// Leaf reads use <c>leaf[i + start]</c> so one cached delegate serves whole-column and chunked
    /// execution alike (issue #167).
    /// </summary>
    static Expression BuildCompiledNode(KernelPlan plan, int nodeIndex, ParameterExpression[] leafParams, ParameterExpression indexVar, ParameterExpression startParam)
    {
        var node = plan.Nodes[nodeIndex];
        switch (node.Op)
        {
            case KernelOp.Column:
                return Expression.ArrayAccess(leafParams[node.Left], Expression.Add(indexVar, startParam));

            case KernelOp.Literal:
                return Expression.Constant(node.Value, node.Value!.GetType());

            case KernelOp.Add:
            case KernelOp.Subtract:
            case KernelOp.Multiply:
            case KernelOp.Divide:
            case KernelOp.Modulo:
                {
                    var left = BuildCompiledNode(plan, node.Left, leafParams, indexVar, startParam);
                    var right = BuildCompiledNode(plan, node.Right, leafParams, indexVar, startParam);
                    return ApplyArithmetic(MapToBinaryOperator(node.Op), ConvertTo(left, node.ComputeType), ConvertTo(right, node.ComputeType));
                }

            case KernelOp.And:
            case KernelOp.Or:
                {
                    var left = BuildCompiledNode(plan, node.Left, leafParams, indexVar, startParam);
                    var right = BuildCompiledNode(plan, node.Right, leafParams, indexVar, startParam);
                    return node.Op == KernelOp.And ? Expression.AndAlso(left, right) : Expression.OrElse(left, right);
                }

            case KernelOp.Equal:
            case KernelOp.NotEqual:
            case KernelOp.GreaterThan:
            case KernelOp.LessThan:
            case KernelOp.GreaterThanOrEqual:
            case KernelOp.LessThanOrEqual:
                {
                    var left = BuildCompiledNode(plan, node.Left, leafParams, indexVar, startParam);
                    var right = BuildCompiledNode(plan, node.Right, leafParams, indexVar, startParam);
                    var operandType = NumericPromoter.GetPromotedType(left.Type, right.Type) ?? left.Type;
                    return BuildComparison(MapToComparisonOperator(node.Op), ConvertTo(left, operandType), ConvertTo(right, operandType));
                }

            case KernelOp.Not:
                return Expression.Not(BuildCompiledNode(plan, node.Left, leafParams, indexVar, startParam));

            default:
                throw new NotSupportedException($"Kernel op {node.Op} is not supported by the compiled fused target");
        }
    }

    static BinaryOperator MapToBinaryOperator(KernelOp op)
    {
        return op switch
        {
            KernelOp.Add => BinaryOperator.Add,
            KernelOp.Subtract => BinaryOperator.Subtract,
            KernelOp.Multiply => BinaryOperator.Multiply,
            KernelOp.Divide => BinaryOperator.Divide,
            KernelOp.Modulo => BinaryOperator.Modulo,
            _ => throw new NotSupportedException($"Kernel op {op} is not an arithmetic operator")
        };
    }

    static ComparisonOperator MapToComparisonOperator(KernelOp op)
    {
        return op switch
        {
            KernelOp.Equal => ComparisonOperator.Equal,
            KernelOp.NotEqual => ComparisonOperator.NotEqual,
            KernelOp.GreaterThan => ComparisonOperator.GreaterThan,
            KernelOp.LessThan => ComparisonOperator.LessThan,
            KernelOp.GreaterThanOrEqual => ComparisonOperator.GreaterThanOrEqual,
            KernelOp.LessThanOrEqual => ComparisonOperator.LessThanOrEqual,
            _ => throw new NotSupportedException($"Kernel op {op} is not a comparison operator")
        };
    }

    static Expression ConvertTo(Expression expression, Type target)
    {
        if (expression.Type == target)
            return expression;

        try
        {
            return Expression.Convert(expression, target);
        }
        catch (InvalidOperationException)
        {
            // Extended numeric domain: nint/nuint/Int128/UInt128/Half/decimal have no built-in CLR
            // conversion to the promoted compute type (e.g. nint -> double, UInt128 -> double). Fall
            // back to a typed INumber<T>.CreateChecked dispatch matching C# promotion (issue #250).
            var source = expression.Type;
            var convert = numericConvertKernelCache.GetOrAdd((source, target), static pair =>
                typeof(FusedExpressionEvaluator).GetMethod(nameof(ConvertNumeric), BindingFlags.Static | BindingFlags.NonPublic)!
                    .MakeGenericMethod(pair.Source, pair.Target));
            return Expression.Call(convert, expression);
        }
    }

    /// <summary>
    /// Converts a numeric value between the extended numeric domain via
    /// <c>TResult.CreateChecked</c>, mirroring the runtime type-switch in
    /// <see cref="FusedKernel.CoerceLiteral{T}"/>. Only emitted by the compiled target when
    /// <see cref="Expression.Convert"/> has no built-in conversion for the pair.
    /// </summary>
    static TResult ConvertNumeric<TSource, TResult>(TSource value)
        where TSource : struct, INumber<TSource>
        where TResult : struct, INumber<TResult>
        => TResult.CreateChecked(value);

    static Expression ApplyArithmetic(BinaryOperator op, Expression left, Expression right)
    {
        return op switch
        {
            BinaryOperator.Add => Expression.Add(left, right),
            BinaryOperator.Subtract => Expression.Subtract(left, right),
            BinaryOperator.Multiply => Expression.Multiply(left, right),
            BinaryOperator.Divide => Expression.Divide(left, right),
            BinaryOperator.Modulo => Expression.Modulo(left, right),
            _ => throw new NotSupportedException($"Binary operator {op} is not supported by the fused evaluator")
        };
    }

    /// <summary>
    /// Builds a comparison. Numeric types (incl. decimal) use direct operator IL for SIMD-friendly
    /// kernels; other comparable types (string, DateTime, Guid, bool, ...) use
    /// <see cref="EqualityComparer{T}"/> for equality and <see cref="Comparer{T}"/> for ordering,
    /// matching the legacy boxed path and <see cref="NivaraColumn{T}"/> semantics.
    /// </summary>
    static Expression BuildComparison(ComparisonOperator op, Expression left, Expression right)
    {
        if (NumericPromoter.GetPromotedType(left.Type, left.Type) != null)
        {
            return op switch
            {
                ComparisonOperator.Equal => Expression.Equal(left, right),
                ComparisonOperator.NotEqual => Expression.NotEqual(left, right),
                ComparisonOperator.GreaterThan => Expression.GreaterThan(left, right),
                ComparisonOperator.LessThan => Expression.LessThan(left, right),
                ComparisonOperator.GreaterThanOrEqual => Expression.GreaterThanOrEqual(left, right),
                ComparisonOperator.LessThanOrEqual => Expression.LessThanOrEqual(left, right),
                _ => throw new NotSupportedException($"Comparison operator {op} is not supported by the fused evaluator")
            };
        }

        if (op is ComparisonOperator.Equal or ComparisonOperator.NotEqual)
        {
            var equalityComparerType = typeof(EqualityComparer<>).MakeGenericType(left.Type);
            var equalityComparer = Expression.Property(null, equalityComparerType.GetProperty(nameof(EqualityComparer<int>.Default), BindingFlags.Public | BindingFlags.Static)!);
            var equals = Expression.Call(equalityComparer, equalityComparerType.GetMethod(nameof(EqualityComparer<int>.Equals), new[] { left.Type, left.Type })!, left, right);
            return op == ComparisonOperator.Equal ? equals : Expression.Not(equals);
        }

        var comparerType = typeof(Comparer<>).MakeGenericType(left.Type);
        var comparer = Expression.Property(null, comparerType.GetProperty(nameof(Comparer<int>.Default), BindingFlags.Public | BindingFlags.Static)!);
        var compare = Expression.Call(comparer, comparerType.GetMethod(nameof(Comparer<int>.Compare), new[] { left.Type, left.Type })!, left, right);
        var zero = Expression.Constant(0);
        return op switch
        {
            ComparisonOperator.GreaterThan => Expression.GreaterThan(compare, zero),
            ComparisonOperator.LessThan => Expression.LessThan(compare, zero),
            ComparisonOperator.GreaterThanOrEqual => Expression.GreaterThanOrEqual(compare, zero),
            ComparisonOperator.LessThanOrEqual => Expression.LessThanOrEqual(compare, zero),
            _ => throw new NotSupportedException($"Comparison operator {op} is not supported by the fused evaluator")
        };
    }

    /// <summary>
    /// Builds a <c>for (i = 0; i &lt; count; i++) dest[destStart + i] = ...;</c> loop writing into a
    /// caller-provided result array at an offset, with leaf reads at <c>[start + i]</c>. When
    /// <paramref name="mask"/> is provided, positions whose mask bit is set are short-circuited to
    /// <c>default(T)</c> instead of computing a value there (the span kernel's masked-position
    /// semantics; the OR'd mask itself was computed before the value pass).
    /// </summary>
    static Expression BuildOffsetForLoop(ParameterExpression index, Expression value, ParameterExpression dest, ParameterExpression destStart, ParameterExpression count, ParameterExpression? mask, Type elementType)
    {
        var breakLabel = Expression.Label();
        var destPosition = Expression.ArrayAccess(dest, Expression.Add(destStart, index));
        var write = Expression.Assign(destPosition, value);

        Expression loopStep = write;
        if (mask != null)
        {
            var masked = Expression.Assign(destPosition, Expression.Default(elementType));
            var isMasked = Expression.AndAlso(
                Expression.NotEqual(mask, Expression.Constant(null, mask.Type)),
                Expression.ArrayAccess(mask, Expression.Add(destStart, index)));
            loopStep = Expression.Condition(isMasked, masked, write);
        }

        return Expression.Loop(
            Expression.IfThenElse(
                Expression.LessThan(index, count),
                Expression.Block(loopStep, Expression.PostIncrementAssign(index)),
                Expression.Break(breakLabel)),
            breakLabel);
    }
}
