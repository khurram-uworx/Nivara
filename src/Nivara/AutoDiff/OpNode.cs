using System.Numerics;

namespace Nivara.AutoDiff;

/// <summary>
/// A node in the reverse-mode computation graph: the backward function for one
/// operation, plus the tensors it is responsible for differentiating.
/// </summary>
sealed class OpNode<T> where T : struct, IFloatingPointIeee754<T>
{
    public string OperationName { get; }

    /// <summary>
    /// Gets every tensor whose gradient this node's backward function is responsible
    /// for accumulating into, including module parameters.
    /// </summary>
    /// <remarks>
    /// This list is the graph edge set, not a hint. <see cref="ComputationGraph"/> walks
    /// it to build the backward plan and to clear gradients, so a tensor absent from it
    /// is unreachable to <c>Backward</c> and invisible to
    /// <see cref="Utilities.GradientUtils.ZeroGrad{T}(ReverseGradTensor{T})"/>.
    /// <para>
    /// Therefore: an argument the backward deliberately does not differentiate must be
    /// rejected loudly with <c>GradientUtils.RequireConstant</c> rather than quietly
    /// omitted, and a tensor is never silently ignored. Forward mode has the
    /// equivalent obligation against its tracking predicate, where there is no graph
    /// and no <c>Backward</c> to fail later.
    /// </para>
    /// </remarks>
    public IReadOnlyList<ReverseGradTensor<T>> Inputs { get; }

    public Action<NivaraColumn<T>> BackwardFunction { get; }

    public OpNode(
        string operationName,
        IReadOnlyList<ReverseGradTensor<T>> inputs,
        Action<NivaraColumn<T>> backwardFunction)
    {
        if (string.IsNullOrEmpty(operationName))
            throw new ArgumentException("Operation name cannot be null or empty", nameof(operationName));

        OperationName = operationName;
        Inputs = inputs ?? throw new ArgumentNullException(nameof(inputs));
        BackwardFunction = backwardFunction ?? throw new ArgumentNullException(nameof(backwardFunction));
    }

    public void Apply(NivaraColumn<T> gradOutput)
    {
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        try
        {
            BackwardFunction(gradOutput);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Error applying backward function for operation '{OperationName}': {ex.Message}", ex);
        }
    }

    public override string ToString()
    {
        return $"OpNode({OperationName}, inputs: {Inputs.Count})";
    }
}
