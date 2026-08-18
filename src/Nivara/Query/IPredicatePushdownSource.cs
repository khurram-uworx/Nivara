using Nivara;
using Nivara.Expressions;

namespace Nivara.Query;

/// <summary>
/// Opt-in interface for data sources that can evaluate filter predicates
/// against storage-level statistics (e.g., Parquet row-group min/max) to
/// skip data that cannot match the filter.
/// </summary>
internal interface IPredicatePushdownSource
{
    /// <summary>
    /// Returns true if the source can evaluate the given filter condition
    /// against its storage-level statistics for row-group-level pruning.
    /// </summary>
    bool CanPushdownFilter(ColumnExpression condition, Schema sourceSchema);

    /// <summary>
    /// Evaluates the filter condition against storage-level statistics and
    /// records which row groups / chunks can be skipped during execution.
    /// Must be called before <see cref="IQuerySource.Execute"/> or
    /// <see cref="IQuerySource.ExecuteAsync"/>.
    /// </summary>
    void ApplyFilterPredicate(ColumnExpression condition, Schema sourceSchema);
}
