using Nivara.Operations;

namespace Nivara.Execution;

public interface IParallelSortOperation
{
    IReadOnlyList<SortKey> SortKeys { get; }
    bool IsStable { get; }
}

public interface IParallelGroupByOperation
{
    IReadOnlyList<Nivara.Expressions.ColumnExpression> GroupByColumns { get; }
}

public interface IParallelJoinOperation
{
    IReadOnlyDictionary<string, IColumn> RightColumns { get; }
    JoinKey[] JoinKeys { get; }
    JoinIndices ComputeJoinIndicesWithHashMap(Dictionary<CompositeKey, List<int>> rightHashMap);
    IReadOnlyDictionary<string, IColumn> MaterializeResult(JoinIndices joinIndices);
}

public interface IParallelConcatenationOperation
{
    IReadOnlyList<IReadOnlyDictionary<string, IColumn>> Sources { get; }
    ConcatenationDirection Direction { get; }
    ConcatenationMismatchHandling MismatchHandling { get; }
    IColumn CreateNullColumn(Type elementType, int length);
}
