using Nivara.Exceptions;
using Nivara.Expressions;
using Nivara.Operations;
using Nivara.Query;
using System.Linq.Expressions;
using System.Reflection;

namespace Nivara.Linq;

/// <summary>
/// A typed LINQ query over a frame. The query is lazy: predicates, projections, and ordering are
/// translated to the expression engine at build time (fail fast) and executed only when
/// <see cref="Collect"/>, <see cref="ToObjects"/>, or <see cref="ToList"/> is called.
/// </summary>
/// <typeparam name="T">The row type</typeparam>
public sealed class NivaraQuery<T>
{
    readonly QueryFrame frame;

    internal NivaraQuery(QueryFrame frame)
    {
        this.frame = frame ?? throw new ArgumentNullException(nameof(frame));
    }

    internal QueryFrame Frame => frame;

    /// <summary>
    /// Gets a value indicating whether this query uses a lazy data source
    /// </summary>
    public bool IsLazy => frame.IsLazy;

    /// <summary>
    /// Gets the schema that will result from executing this query
    /// </summary>
    public Schema Schema => frame.Schema;

    /// <summary>
    /// Returns a string describing the query plan for debugging
    /// </summary>
    public string ExplainPlan() => frame.ExplainPlan();

    /// <summary>
    /// Returns the underlying lazy query frame for advanced composition
    /// </summary>
    internal QueryFrame AsQueryFrame() => frame;

    /// <summary>
    /// Filters the rows using the given predicate
    /// </summary>
    /// <param name="predicate">The predicate to filter by</param>
    /// <returns>A new query with the filter applied</returns>
    /// <exception cref="ArgumentNullException">Thrown when predicate is null</exception>
    /// <exception cref="UnsupportedQueryExpressionException">Thrown when the predicate uses unsupported expressions</exception>
    public NivaraQuery<T> Where(Expression<Func<T, bool>> predicate)
    {
        ArgumentNullException.ThrowIfNull(predicate);

        var condition = CreateTranslator().Translate(predicate.Body);
        return new NivaraQuery<T>(frame.Filter(condition));
    }

    /// <summary>
    /// Projects each row into a new shape using the given selector
    /// </summary>
    /// <typeparam name="TResult">The result row type</typeparam>
    /// <param name="selector">The projection selector (an anonymous type or member initializer)</param>
    /// <returns>A new query producing the projected rows</returns>
    /// <exception cref="ArgumentNullException">Thrown when selector is null</exception>
    /// <exception cref="UnsupportedQueryExpressionException">Thrown when the projection is not supported</exception>
    public NivaraQuery<TResult> Select<TResult>(Expression<Func<T, TResult>> selector)
    {
        ArgumentNullException.ThrowIfNull(selector);

        var projection = TypedProjectionBuilder.Build(selector.Body, CreateTranslator());
        var selectOperation = new SelectOperation(projection.Columns, projection.OutputNames);
        return new NivaraQuery<TResult>(frame.WithOperation(selectOperation));
    }

    /// <summary>
    /// Sorts the rows by the given key in ascending order
    /// </summary>
    /// <param name="keySelector">The sort key selector</param>
    /// <returns>A new query with the sort applied</returns>
    public NivaraQuery<T> OrderBy(Expression<Func<T, object?>> keySelector)
        => SortByCore(keySelector, SortDirection.Ascending);

    /// <summary>
    /// Sorts the rows by the given key in descending order
    /// </summary>
    /// <param name="keySelector">The sort key selector</param>
    /// <returns>A new query with the sort applied</returns>
    public NivaraQuery<T> OrderByDescending(Expression<Func<T, object?>> keySelector)
        => SortByCore(keySelector, SortDirection.Descending);

    /// <summary>
    /// Appends a secondary ascending sort key
    /// </summary>
    /// <param name="keySelector">The sort key selector</param>
    /// <returns>A new query with the secondary sort applied</returns>
    public NivaraQuery<T> ThenBy(Expression<Func<T, object?>> keySelector)
        => ThenByCore(keySelector, SortDirection.Ascending);

    /// <summary>
    /// Appends a secondary descending sort key
    /// </summary>
    /// <param name="keySelector">The sort key selector</param>
    /// <returns>A new query with the secondary sort applied</returns>
    public NivaraQuery<T> ThenByDescending(Expression<Func<T, object?>> keySelector)
        => ThenByCore(keySelector, SortDirection.Descending);

    NivaraQuery<T> SortByCore(Expression<Func<T, object?>> keySelector, SortDirection direction)
    {
        ArgumentNullException.ThrowIfNull(keySelector);

        var key = CreateTranslator().Translate(keySelector.Body);
        return new NivaraQuery<T>(frame.SortByExpression(key, direction));
    }

    NivaraQuery<T> ThenByCore(Expression<Func<T, object?>> keySelector, SortDirection direction)
    {
        ArgumentNullException.ThrowIfNull(keySelector);

        var key = CreateTranslator().Translate(keySelector.Body);
        return new NivaraQuery<T>(frame.ThenBy(key, direction));
    }

    /// <summary>
    /// Groups the rows by the given key, returning a grouped query on which a final Select (with
    /// aggregates) or a bare Collect may be performed
    /// </summary>
    /// <typeparam name="TKey">The group key type</typeparam>
    /// <param name="keySelector">The key selector</param>
    /// <returns>A grouped query</returns>
    /// <exception cref="ArgumentNullException">Thrown when keySelector is null</exception>
    /// <exception cref="UnsupportedQueryExpressionException">Thrown when the key selector is not supported</exception>
    public NivaraGroupedQuery<TKey, T> GroupBy<TKey>(Expression<Func<T, TKey>> keySelector)
    {
        ArgumentNullException.ThrowIfNull(keySelector);

        var preGroupSchema = frame.Schema;
        var key = CreateTranslator(preGroupSchema).Translate(keySelector.Body);
        var keyColumnName = GetKeyOutputName(key);

        return new NivaraGroupedQuery<TKey, T>(frame, key, keyColumnName, preGroupSchema);
    }

    /// <summary>
    /// Omits the first <paramref name="count"/> rows
    /// </summary>
    public NivaraQuery<T> Skip(int count) => new(frame.Skip(count));

    /// <summary>
    /// Keeps only the first <paramref name="count"/> rows
    /// </summary>
    public NivaraQuery<T> Take(int count) => new(frame.Take(count));

    /// <summary>
    /// Applies a skip-and-take window to the rows
    /// </summary>
    public NivaraQuery<T> Slice(int skip, int take) => new(frame.Slice(skip, take));

    /// <summary>
    /// Executes the query and returns a materialized frame
    /// </summary>
    /// <returns>A NivaraFrame with the query results</returns>
    public NivaraFrame Collect() => frame.Collect();

    /// <summary>
    /// Executes the query and materializes the result rows as objects
    /// </summary>
    /// <returns>A read-only list of typed rows</returns>
    public IReadOnlyList<T> ToObjects()
    {
        var result = Collect();
        var factory = TypedRowFactory<T>.GetFactory(result.Schema);
        var columns = result.ColumnNames.Select(name => result.GetColumn(name)).ToArray();

        var rows = new List<T>(result.RowCount);
        for (int i = 0; i < result.RowCount; i++)
            rows.Add(factory(columns, i));

        return rows;
    }

    /// <summary>
    /// Executes the query and materializes the result rows as objects
    /// </summary>
    /// <returns>A list of typed rows</returns>
    public List<T> ToList() => ToObjects().ToList();

    /// <summary>
    /// Executes the query and materializes the result rows as objects (alias for <see cref="ToObjects"/>)
    /// </summary>
    /// <returns>A read-only list of typed rows</returns>
    public IReadOnlyList<T> ToRows() => ToObjects();

    TypedExpressionTranslator CreateTranslator() => CreateTranslator(frame.Schema);

    TypedExpressionTranslator CreateTranslator(Schema schema) => TypedLinqMetadata.CreateTranslator(typeof(T), schema);

    static string GetKeyOutputName(ColumnExpression key)
    {
        return key is ColumnReference columnReference ? columnReference.ColumnName : key.Name;
    }
}

/// <summary>
/// A typed LINQ query that has been grouped. Only a final Select projection (using <c>g.Key</c> and
/// group aggregates such as <c>g.Count()</c>, <c>g.Sum(...)</c>, <c>g.Average(...)</c>,
/// <c>g.Min(...)</c>, and <c>g.Max(...)</c>) or a bare <see cref="Collect"/> (distinct keys) is
/// supported; any other chained operation fails fast with a clear diagnostic.
/// </summary>
/// <typeparam name="TKey">The group key type</typeparam>
/// <typeparam name="T">The row type being grouped</typeparam>
public sealed class NivaraGroupedQuery<TKey, T>
{
    readonly QueryFrame baseFrame;
    readonly ColumnExpression keyExpression;
    readonly string keyColumnName;
    readonly Schema preGroupSchema;

    internal NivaraGroupedQuery(QueryFrame baseFrame, ColumnExpression keyExpression, string keyColumnName, Schema preGroupSchema)
    {
        this.baseFrame = baseFrame ?? throw new ArgumentNullException(nameof(baseFrame));
        this.keyExpression = keyExpression ?? throw new ArgumentNullException(nameof(keyExpression));
        this.keyColumnName = keyColumnName ?? throw new ArgumentNullException(nameof(keyColumnName));
        this.preGroupSchema = preGroupSchema ?? throw new ArgumentNullException(nameof(preGroupSchema));
    }

    /// <summary>
    /// Executes the query and returns the distinct group keys as a frame
    /// </summary>
    /// <returns>A NivaraFrame with one row per distinct key</returns>
    public NivaraFrame Collect()
    {
        return baseFrame.WithOperation(BuildGroupByOperation(aggregations: null)).Collect();
    }

    /// <summary>
    /// Projects each group into a final result shape using a projection over the group marker
    /// (<c>g.Key</c> plus aggregate calls)
    /// </summary>
    /// <typeparam name="TResult">The result row type</typeparam>
    /// <param name="selector">The projection selector</param>
    /// <returns>A new query producing one row per group</returns>
    /// <exception cref="ArgumentNullException">Thrown when selector is null</exception>
    /// <exception cref="UnsupportedQueryExpressionException">Thrown when the projection uses unsupported constructs</exception>
    public NivaraQuery<TResult> Select<TResult>(Expression<Func<Grouping<TKey, T>, TResult>> selector)
    {
        ArgumentNullException.ThrowIfNull(selector);

        var body = selector.Body;
        var translator = TypedLinqMetadata.CreateTranslator(typeof(T), preGroupSchema);

        var columns = new List<ColumnExpression>();
        var outputNames = new List<string>();
        var aggregations = new List<GroupedAggregation>();

        foreach (var (memberName, expression) in EnumerateProjectionMembers(body))
        {
            var translated = TranslateGroupMember(expression, translator);

            switch (translated)
            {
                case GroupAggregate aggregate:
                    aggregations.Add(new GroupedAggregation(memberName, aggregate.Source, aggregate.Function));
                    columns.Add(new ColumnReference(memberName));
                    outputNames.Add(memberName);
                    break;

                case ColumnExpression columnExpression:
                    columns.Add(columnExpression);
                    outputNames.Add(memberName);
                    break;
            }
        }

        var groupByOperation = BuildGroupByOperation(aggregations);
        var selectOperation = new SelectOperation(columns.ToArray(), outputNames.ToArray());

        return new NivaraQuery<TResult>(
            baseFrame.WithOperation(groupByOperation).WithOperation(selectOperation));
    }

    GroupByOperation BuildGroupByOperation(IReadOnlyList<GroupedAggregation>? aggregations)
    {
        return new GroupByOperation(new[] { keyExpression }, new[] { keyColumnName }, aggregations);
    }

    static IEnumerable<(string MemberName, Expression Expression)> EnumerateProjectionMembers(Expression body)
    {
        switch (body)
        {
            case NewExpression newExpression when newExpression.Members is { Count: > 0 }:
                for (int i = 0; i < newExpression.Arguments.Count; i++)
                {
                    var member = newExpression.Members[i]
                        ?? throw new UnsupportedQueryExpressionException("Grouped Select members must be named.");
                    yield return (member.Name, newExpression.Arguments[i]);
                }
                yield break;

            case MemberInitExpression memberInit:
                foreach (var binding in memberInit.Bindings)
                {
                    if (binding is not MemberAssignment assignment)
                        throw new UnsupportedQueryExpressionException(
                            $"Unsupported grouped Select member '{binding.Member.Name}': only property assignments are supported.");
                    yield return (binding.Member.Name, assignment.Expression);
                }
                yield break;

            default:
                throw new UnsupportedQueryExpressionException(
                    $"Unsupported grouped Select projection '{body}': projections must construct an object using g.Key and group aggregates.");
        }
    }

    /// <summary>
    /// Translates a member of a grouped projection. A reference to <c>g.Key</c> maps to the group
    /// key column; an aggregate method call (Count/Sum/Average/Min/Max) on the group marker maps to
    /// a group aggregation; anything else is rejected.
    /// </summary>
    object TranslateGroupMember(Expression expression, TypedExpressionTranslator translator)
    {
        switch (expression)
        {
            case MemberExpression member when IsKeyAccess(member):
                return new ColumnReference(keyColumnName);

            case MethodCallExpression call:
                return TranslateGroupAggregate(call, translator);

            default:
                throw new UnsupportedQueryExpressionException(
                    $"Unsupported grouped Select expression '{expression}': only g.Key and group aggregates are supported.");
        }
    }

    static bool IsKeyAccess(MemberExpression member)
    {
        return member.Member is PropertyInfo property
            && string.Equals(property.Name, nameof(Grouping<TKey, T>.Key), StringComparison.Ordinal)
            && member.Expression is ParameterExpression parameter
            && parameter.Type.IsGenericType
            && parameter.Type.GetGenericTypeDefinition() == typeof(Grouping<,>);
    }

    GroupAggregate TranslateGroupAggregate(MethodCallExpression call, TypedExpressionTranslator translator)
    {
        var methodName = call.Method.Name;

        if (string.Equals(methodName, nameof(Grouping<TKey, T>.Count), StringComparison.Ordinal))
        {
            if (call.Arguments.Count != 0)
                throw new UnsupportedQueryExpressionException("g.Count() does not accept a predicate; filter rows before grouping instead.");

            // The count source is unused; reference the key column so the expression is valid against the pre-group schema.
            return new GroupAggregate(new ColumnReference(keyColumnName), new RowCountAggregation());
        }

        if (call.Arguments.Count != 1 || call.Arguments[0] is not LambdaExpression selector)
            throw new UnsupportedQueryExpressionException($"g.{methodName}(...) requires a single selector lambda.");

        var function = methodName switch
        {
            nameof(Grouping<TKey, T>.Sum) => (AggregationFunction)new SumAggregation(),
            nameof(Grouping<TKey, T>.Average) => new MeanAggregation(),
            nameof(Grouping<TKey, T>.Min) => new MinAggregation(),
            nameof(Grouping<TKey, T>.Max) => new MaxAggregation(),
            _ => throw new UnsupportedQueryExpressionException($"Group aggregate '{methodName}' is not supported. Supported aggregates: Count, Sum, Average, Min, Max.")
        };

        var source = translator.Translate(selector.Body);
        return new GroupAggregate(source, function);
    }

    readonly record struct GroupAggregate(ColumnExpression Source, AggregationFunction Function);
}
