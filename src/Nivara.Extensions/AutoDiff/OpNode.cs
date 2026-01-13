using System.Numerics;
using Nivara;

namespace Nivara.Extensions.AutoDiff;

/// <summary>
/// Represents a node in the computation graph for automatic differentiation.
/// Contains operation metadata and backward functions for gradient computation.
/// </summary>
internal sealed class OpNode
{
    /// <summary>
    /// Gets the name of the operation this node represents
    /// </summary>
    public string OperationName { get; }

    /// <summary>
    /// Gets the input tensors that were used to create this operation
    /// </summary>
    public IReadOnlyList<object> Inputs { get; }

    /// <summary>
    /// Gets the backward function that computes gradients for this operation
    /// </summary>
    public Action<NivaraColumn<object>> BackwardFunction { get; }

    /// <summary>
    /// Gets a value indicating whether this operation should save intermediate values for backward pass
    /// </summary>
    public bool ShouldSaveForBackward { get; }

    /// <summary>
    /// Gets any saved values needed for the backward pass
    /// </summary>
    public Dictionary<string, object>? SavedValues { get; }

    /// <summary>
    /// Initializes a new instance of OpNode with the specified operation details
    /// </summary>
    /// <param name="operationName">The name of the operation</param>
    /// <param name="inputs">The input tensors used in this operation</param>
    /// <param name="backwardFunction">The function to compute gradients</param>
    /// <param name="shouldSaveForBackward">Whether to save intermediate values</param>
    /// <param name="savedValues">Optional saved values for backward pass</param>
    /// <exception cref="ArgumentNullException">Thrown when required parameters are null</exception>
    /// <exception cref="ArgumentException">Thrown when operationName is empty</exception>
    public OpNode(
        string operationName,
        IReadOnlyList<object> inputs,
        Action<NivaraColumn<object>> backwardFunction,
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

    /// <summary>
    /// Applies the backward function with the given gradient output
    /// </summary>
    /// <param name="gradOutput">The gradient flowing back through this operation</param>
    /// <exception cref="ArgumentNullException">Thrown when gradOutput is null</exception>
    public void Apply(NivaraColumn<object> gradOutput)
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

    /// <summary>
    /// Creates a string representation of this OpNode
    /// </summary>
    /// <returns>A string representation showing operation name and input count</returns>
    public override string ToString()
    {
        return $"OpNode({OperationName}, inputs: {Inputs.Count})";
    }
}