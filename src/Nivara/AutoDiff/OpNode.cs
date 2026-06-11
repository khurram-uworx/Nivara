using System.Numerics;

namespace Nivara.AutoDiff;

sealed class OpNode<T> where T : struct, INumber<T>
{
    public string OperationName { get; }
    public IReadOnlyList<object> Inputs { get; }
    public Action<NivaraColumn<T>> BackwardFunction { get; }
    public bool ShouldSaveForBackward { get; }
    public Dictionary<string, object>? SavedValues { get; }

    public OpNode(
        string operationName,
        IReadOnlyList<object> inputs,
        Action<NivaraColumn<T>> backwardFunction,
        bool shouldSaveForBackward = false,
        Dictionary<string, object>? savedValues = null)
    {
        if (string.IsNullOrEmpty(operationName))
            throw new ArgumentException("Operation name cannot be null or empty", nameof(operationName));

        OperationName = operationName;
        Inputs = inputs ?? throw new ArgumentNullException(nameof(inputs));
        BackwardFunction = backwardFunction ?? throw new ArgumentNullException(nameof(backwardFunction));
        ShouldSaveForBackward = shouldSaveForBackward;
        SavedValues = savedValues;
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
