using System.Numerics;

namespace Nivara.AutoDiff;

sealed class OpNode<T> where T : struct, INumber<T>
{
    public string OperationName { get; }
    public IReadOnlyList<object> Inputs { get; }
    public Action<NivaraColumn<T>, bool> BackwardFunction { get; }

    public OpNode(
        string operationName,
        IReadOnlyList<object> inputs,
        Action<NivaraColumn<T>, bool> backwardFunction)
    {
        if (string.IsNullOrEmpty(operationName))
            throw new ArgumentException("Operation name cannot be null or empty", nameof(operationName));

        OperationName = operationName;
        Inputs = inputs ?? throw new ArgumentNullException(nameof(inputs));
        BackwardFunction = backwardFunction ?? throw new ArgumentNullException(nameof(backwardFunction));
    }

    public void Apply(NivaraColumn<T> gradOutput, bool stripGradientNulls)
    {
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));

        try
        {
            BackwardFunction(gradOutput, stripGradientNulls);
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
