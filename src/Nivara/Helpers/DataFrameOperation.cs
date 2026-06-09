using Nivara.Query;

namespace Nivara.Helpers;

public abstract class DataFrameOperation : IQueryOperation<NivaraFrame>
{
    protected DataFrameOperation(QueryPlan plan)
    {
        Plan = plan ?? throw new ArgumentNullException(nameof(plan));
    }

    public QueryPlan Plan { get; }
    public abstract string OperationType { get; }
    public abstract NivaraFrame Execute();

    public override string ToString() => $"{OperationType}Operation";
}
