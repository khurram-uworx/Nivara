using Nivara.Diagnostics;
using Nivara.Exceptions;
using Nivara.Helpers;
using Nivara.Storage;
using System.Collections.Concurrent;
using System.Linq.Expressions;
using System.Numerics;
using System.Reflection;

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
    int nodeTreePathEvaluationCount;

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
    /// Gets how many of those evaluations ran through the generic node-tree kernel fallback.
    /// </summary>
    internal int NodeTreePathEvaluationCount => nodeTreePathEvaluationCount;

    static readonly ConcurrentDictionary<string, Delegate> compiledKernelCache = new();

    static readonly ConcurrentDictionary<Type, MethodInfo> snapshotKernelCache = new();

    static readonly ConcurrentDictionary<Type, MethodInfo> createColumnKernelCache = new();

    static readonly ConcurrentDictionary<Type, MethodInfo> createFromSpansKernelCache = new();

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
            return EvaluateCore(expression, input);
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
        }
    }

    IColumn EvaluateCore(ColumnExpression expression, IReadOnlyDictionary<string, IColumn> input)
    {
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
        if (plan == null || plan.Columns.Count == 0)
        {
            throw new NotSupportedException(
                $"Expression '{expression.Name}' cannot run through the fused evaluator: unsupported operand combination");
        }

        ValidateLeafLengths(plan);
        var length = plan.Columns[0].Column.Length;
        var leafArrays = SnapshotLeaves(plan);

        Delegate action;
        try
        {
            action = compiledKernelCache.GetOrAdd(plan.Signature, key => BuildCompiledDelegate(expression, plan));
            compiledPathEvaluationCount++;
        }
        catch (NotSupportedException)
        {
            return EvaluateNodeTree(expression, plan, length);
        }

        var resultArray = ExecuteCompiled(action, leafArrays, plan.ResultType, length);
        var mask = plan.HasNulls ? ComputeMask(plan, length) : null;

        if (mask != null && plan.ResultType == typeof(bool))
        {
            var boolResult = (bool[])resultArray;
            for (int i = 0; i < length; i++)
                if (mask[i])
                    boolResult[i] = false;
        }

        fusedPathEvaluationCount++;
        var vectorizable = plan.Columns.All(l => ColumnStorageFactory.IsVectorizable(l.Column.ElementType))
            && ColumnStorageFactory.IsVectorizable(plan.ResultType);
        RecordDiagnostics(length, plan.ResultType, plan.HasNulls, vectorizable, "Compiled fused kernel");
        return CreateResultColumn(plan.ResultType, resultArray, mask);
    }

    /// <summary>
    /// Runs the compiled delegate, writing results into a freshly allocated typed array.
    /// </summary>
    static Array ExecuteCompiled(Delegate action, object[] leafArrays, Type resultType, int length)
    {
        var result = Array.CreateInstance(resultType, length);
        var args = new object[leafArrays.Length + 1];
        Array.Copy(leafArrays, args, leafArrays.Length);
        args[leafArrays.Length] = result;
        action.DynamicInvoke(args);
        return result;
    }

    /// <summary>
    /// Evaluates through the generic node-tree kernel when the compiled target could not be built.
    /// Requires generic math and leaf columns already sharing the result element type.
    /// </summary>
    IColumn EvaluateNodeTree(ColumnExpression expression, FusedExpressionPlan plan, int length)
    {
        if (!plan.IsGenericMath)
        {
            throw new NotSupportedException(
                $"Expression '{expression.Name}' is not supported by the fused evaluator for element type {plan.ResultType.Name}");
        }

        foreach (var leaf in plan.Columns)
        {
            if (leaf.Column.ElementType != plan.ResultType)
            {
                throw new NotSupportedException(
                    $"Expression '{expression.Name}' mixes element types that the generic node-tree kernel cannot fuse");
            }
        }

        var mask = plan.HasNulls ? ComputeMask(plan, length) : null;
        var kernel = GetNodeTreeRunner(plan.ResultType);
        var result = kernel.Invoke(null, new object?[] { expression, plan.Columns, mask });
        nodeTreePathEvaluationCount++;
        fusedPathEvaluationCount++;
        RecordDiagnostics(length, plan.ResultType, plan.HasNulls, ColumnStorageFactory.IsVectorizable(plan.ResultType), "Generic node-tree kernel");
        return (IColumn)result!;
    }

    static class FusedNodeTreeRunner<T>
        where T : struct, INumber<T>
    {
        internal static IColumn Run(ColumnExpression expression, IReadOnlyList<FusedColumnBinding> leaves, bool[]? mask)
            => FusedKernel.Evaluate<T>(expression, leaves, mask);
    }

    static readonly ConcurrentDictionary<Type, MethodInfo> nodeTreeRunnerCache = new();

    static MethodInfo GetNodeTreeRunner(Type elementType)
    {
        return nodeTreeRunnerCache.GetOrAdd(elementType, static t =>
            typeof(FusedNodeTreeRunner<>).MakeGenericType(t)
                .GetMethod(nameof(FusedNodeTreeRunner<int>.Run), BindingFlags.Static | BindingFlags.NonPublic)!);
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
    /// Snapshots each leaf column into a typed array (default values at null positions; the null mask
    /// is computed separately), so the compiled delegate can index the backing arrays directly.
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
        var array = new T[column.Length];
        if (column.TryGetSpan(out var span))
        {
            span.CopyTo(array);
        }
        else
        {
            for (int i = 0; i < column.Length; i++)
                array[i] = column[i];
        }

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

        return value switch
        {
            int intValue => FillConstant(intValue, length),
            double doubleValue => FillConstant(doubleValue, length),
            float floatValue => FillConstant(floatValue, length),
            long longValue => FillConstant(longValue, length),
            bool boolValue => FillConstant(boolValue, length),
            decimal decimalValue => FillConstant(decimalValue, length),
            byte byteValue => FillConstant(byteValue, length),
            short shortValue => FillConstant(shortValue, length),
            DateTime dateTimeValue => FillConstant(dateTimeValue, length),
            _ => FillConstantObject(value, length)
        };
    }

    static IColumn FillConstant<T>(T value, int length)
        where T : struct
    {
        var array = new T[length];
        Array.Fill(array, value);
        return NivaraColumn<T>.Create(array);
    }

    static IColumn FillConstantObject(object value, int length)
    {
        var array = new object[length];
        Array.Fill(array, value);
        return NivaraColumn<object>.Create(array);
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
    /// Builds a compiled <c>Action&lt;T1[], ..., TN[], R[]&gt;</c> delegate that runs the whole
    /// expression over the leaf arrays in a single loop, writing into the result array.
    /// </summary>
    static Delegate BuildCompiledDelegate(ColumnExpression expression, FusedExpressionPlan plan)
    {
        var leafIndex = new Dictionary<ColumnReference, int>(ReferenceEqualityComparer.Instance);
        for (int i = 0; i < plan.Columns.Count; i++)
            leafIndex[plan.Columns[i].Reference] = i;

        var leafParams = new ParameterExpression[plan.Columns.Count];
        for (int i = 0; i < plan.Columns.Count; i++)
            leafParams[i] = Expression.Parameter(plan.Columns[i].Column.ElementType.MakeArrayType(), "leaf" + i);

        var destParam = Expression.Parameter(plan.ResultType.MakeArrayType(), "dest");
        var indexVar = Expression.Parameter(typeof(int), "i");

        var value = BuildNode(expression, leafParams, leafIndex, indexVar);
        var loopBody = BuildForLoop(indexVar, value, destParam);

        var paramTypes = new Type[plan.Columns.Count + 1];
        for (int i = 0; i < plan.Columns.Count; i++)
            paramTypes[i] = leafParams[i].Type;
        paramTypes[plan.Columns.Count] = destParam.Type;

        var actionType = Expression.GetActionType(paramTypes);
        var body = Expression.Block(new[] { indexVar }, loopBody);
        var lambda = Expression.Lambda(actionType, body, leafParams.Append(destParam));
        return lambda.Compile();
    }

    /// <summary>
    /// Builds the per-row value expression for a node, returning an expression of the node's compute
    /// type. Numeric nodes promote their operands (C# binary numeric promotion via
    /// <see cref="NumericPromoter"/>) and insert widening conversions on the leaf/literal operands.
    /// </summary>
    static Expression BuildNode(ColumnExpression node, ParameterExpression[] leafParams, IReadOnlyDictionary<ColumnReference, int> leafIndex, ParameterExpression indexVar)
    {
        switch (node)
        {
            case ColumnReference columnRef:
                {
                    var leafParam = leafParams[leafIndex[columnRef]];
                    return Expression.ArrayAccess(leafParam, indexVar);
                }

            case LiteralExpression literal:
                return Expression.Constant(literal.Value, literal.Value!.GetType());

            case ScalarExpression scalar:
                {
                    var columnValue = BuildNode(scalar.Column, leafParams, leafIndex, indexVar);
                    var scalarType = scalar.Scalar!.GetType();
                    var promoted = NumericPromoter.GetPromotedType(columnValue.Type, scalarType)!;
                    var left = ConvertTo(columnValue, promoted);
                    var right = ConvertTo(Expression.Constant(scalar.Scalar, scalarType), promoted);
                    return ApplyArithmetic(scalar.Operator, left, right);
                }

            case BinaryExpression binary when binary.Operator is not (BinaryOperator.And or BinaryOperator.Or):
                {
                    var left = BuildNode(binary.Left, leafParams, leafIndex, indexVar);
                    var right = BuildNode(binary.Right, leafParams, leafIndex, indexVar);
                    var promoted = NumericPromoter.GetPromotedType(left.Type, right.Type)!;
                    return ApplyArithmetic(binary.Operator, ConvertTo(left, promoted), ConvertTo(right, promoted));
                }

            case BinaryExpression binary:
                return binary.Operator == BinaryOperator.And
                    ? Expression.AndAlso(BuildNode(binary.Left, leafParams, leafIndex, indexVar), BuildNode(binary.Right, leafParams, leafIndex, indexVar))
                    : Expression.OrElse(BuildNode(binary.Left, leafParams, leafIndex, indexVar), BuildNode(binary.Right, leafParams, leafIndex, indexVar));

            case ComparisonExpression comparison:
                {
                    var left = BuildNode(comparison.Left, leafParams, leafIndex, indexVar);
                    var right = BuildNode(comparison.Right, leafParams, leafIndex, indexVar);
                    var operandType = NumericPromoter.GetPromotedType(left.Type, right.Type) ?? left.Type;
                    return BuildComparison(comparison.Operator, ConvertTo(left, operandType), ConvertTo(right, operandType));
                }

            case NotExpression not:
                return Expression.Not(BuildNode(not.Operand, leafParams, leafIndex, indexVar));

            default:
                throw new NotSupportedException($"Expression type {node.GetType().Name} is not supported by the fused evaluator");
        }
    }

    static Expression ConvertTo(Expression expression, Type target)
        => expression.Type == target ? expression : Expression.Convert(expression, target);

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
    /// Builds a <c>for (i = 0; i &lt; dest.Length; i++) dest[i] = value;</c> loop over the result array.
    /// </summary>
    static Expression BuildForLoop(ParameterExpression index, Expression value, ParameterExpression dest)
    {
        var breakLabel = Expression.Label();
        return Expression.Loop(
            Expression.IfThenElse(
                Expression.LessThan(index, Expression.ArrayLength(dest)),
                Expression.Block(
                    Expression.Assign(Expression.ArrayAccess(dest, index), value),
                    Expression.PostIncrementAssign(index)),
                Expression.Break(breakLabel)),
            breakLabel);
    }
}
