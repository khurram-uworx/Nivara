using Nivara.AI;

namespace NivaraChat.Helpers;

public sealed class AdaptiveTrainingContext : IDisposable
{
    public LayaDecisionHead<float, float> Head { get; }
    public IReadOnlyList<AdaptiveTrainingExample> Examples { get; }

    public AdaptiveTrainingContext(
        LayaDecisionHead<float, float> head,
        IReadOnlyList<AdaptiveTrainingExample> examples)
    {
        ArgumentNullException.ThrowIfNull(head);
        ArgumentNullException.ThrowIfNull(examples);
        Head = head;
        Examples = examples;
    }

    public void Dispose() => Head.Dispose();
}
