using System.Numerics;
using Nivara;

namespace Nivara.Extensions.AutoDiff;

/// <summary>
/// Manages the computation graph and backward pass for automatic differentiation.
/// Provides static methods for graph construction, traversal, and gradient computation.
/// </summary>
public sealed class ComputationGraph
{
    /// <summary>
    /// Adds a computation node to the graph for the specified output tensor
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor</typeparam>
    /// <param name="output">The output tensor that results from this operation</param>
    /// <param name="node">The operation node to add</param>
    /// <exception cref="ArgumentNullException">Thrown when output or node is null</exception>
    internal static void AddNode<T>(GradTensor<T> output, OpNode node) where T : struct, INumber<T>
    {
        if (output == null)
            throw new ArgumentNullException(nameof(output));
        if (node == null)
            throw new ArgumentNullException(nameof(node));

        // Only add nodes for tensors that require gradients
        if (output.RequiresGrad)
        {
            output.GradFn = node;
        }
    }

    /// <summary>
    /// Performs backward pass computation starting from the specified tensor
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor</typeparam>
    /// <param name="tensor">The tensor to start backward pass from</param>
    /// <param name="gradient">The initial gradient (optional, defaults to ones for scalar tensors)</param>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when tensor doesn't require gradients or graph has cycles</exception>
    internal static void Backward<T>(GradTensor<T> tensor, NivaraColumn<T>? gradient = null) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (!tensor.RequiresGrad)
        {
            throw new InvalidOperationException("Cannot perform backward pass on tensor that doesn't require gradients");
        }

        // Validate graph structure
        ValidateGraph(tensor);

        // Initialize gradient if not provided
        if (gradient == null)
        {
            if (tensor.Length == 1)
            {
                gradient = NivaraColumn<T>.Create(new T[] { T.One });
            }
            else
            {
                throw new InvalidOperationException("Gradient must be provided for non-scalar tensors");
            }
        }

        // Set initial gradient
        tensor.Grad = gradient;

        // Get topological ordering of nodes
        var nodes = TopologicalSort(tensor);

        // Apply backward functions in reverse topological order
        foreach (var node in nodes.AsEnumerable().Reverse())
        {
            // Find the tensor this node belongs to and apply gradients
            ApplyNodeGradients(node, tensor);
        }
    }

    /// <summary>
    /// Performs topological sort of the computation graph starting from the given tensor
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor</typeparam>
    /// <param name="root">The root tensor to start traversal from</param>
    /// <returns>A list of nodes in topological order</returns>
    /// <exception cref="ArgumentNullException">Thrown when root is null</exception>
    internal static List<OpNode> TopologicalSort<T>(GradTensor<T> root) where T : struct, INumber<T>
    {
        if (root == null)
            throw new ArgumentNullException(nameof(root));

        var visited = new HashSet<OpNode>();
        var result = new List<OpNode>();
        var visiting = new HashSet<OpNode>(); // For cycle detection

        void Visit(OpNode? node)
        {
            if (node == null || visited.Contains(node))
                return;

            if (visiting.Contains(node))
            {
                throw new InvalidOperationException($"Circular dependency detected in computation graph at operation '{node.OperationName}'");
            }

            visiting.Add(node);

            // Visit all input nodes first
            foreach (var input in node.Inputs)
            {
                if (input is GradTensor<T> gradTensor && gradTensor.GradFn != null)
                {
                    Visit(gradTensor.GradFn);
                }
            }

            visiting.Remove(node);
            visited.Add(node);
            result.Add(node);
        }

        Visit(root.GradFn);
        return result;
    }

    /// <summary>
    /// Validates the computation graph for cycles and other structural issues
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor</typeparam>
    /// <param name="root">The root tensor to validate from</param>
    /// <exception cref="ArgumentNullException">Thrown when root is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when graph has structural issues</exception>
    internal static void ValidateGraph<T>(GradTensor<T> root) where T : struct, INumber<T>
    {
        if (root == null)
            throw new ArgumentNullException(nameof(root));

        try
        {
            // Topological sort will detect cycles
            TopologicalSort(root);
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains("Circular dependency"))
        {
            throw new InvalidOperationException("Computation graph contains circular dependencies, which would cause infinite loops during backward pass", ex);
        }
    }

    /// <summary>
    /// Applies gradients from a computation node to its input tensors
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor</typeparam>
    /// <param name="node">The computation node to apply</param>
    /// <param name="outputTensor">The output tensor that this node produced</param>
    private static void ApplyNodeGradients<T>(OpNode node, GradTensor<T> outputTensor) where T : struct, INumber<T>
    {
        if (outputTensor.Grad == null)
            return;

        try
        {
            // Convert gradient to object column for the backward function
            // This is a temporary approach - in a full implementation, we'd have type-safe backward functions
            var gradAsObjects = new object[outputTensor.Grad.Length];
            for (int i = 0; i < outputTensor.Grad.Length; i++)
            {
                gradAsObjects[i] = outputTensor.Grad[i]!;
            }
            var gradColumn = NivaraColumn<object>.CreateForReferenceType(gradAsObjects);

            // Apply the backward function
            node.Apply(gradColumn);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to apply gradients for operation '{node.OperationName}': {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Clears all gradients in the computation graph reachable from the specified tensor
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor</typeparam>
    /// <param name="tensor">The tensor to start clearing from</param>
    /// <exception cref="ArgumentNullException">Thrown when tensor is null</exception>
    public static void ZeroGrad<T>(GradTensor<T> tensor) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var visited = new HashSet<object>();

        void ClearGradients(GradTensor<T> t)
        {
            if (visited.Contains(t))
                return;

            visited.Add(t);
            t.ZeroGrad();

            // Recursively clear gradients of input tensors
            if (t.GradFn != null)
            {
                foreach (var input in t.GradFn.Inputs)
                {
                    if (input is GradTensor<T> inputTensor)
                    {
                        ClearGradients(inputTensor);
                    }
                }
            }
        }

        ClearGradients(tensor);
    }

    /// <summary>
    /// Gets diagnostic information about the computation graph
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor</typeparam>
    /// <param name="root">The root tensor to analyze</param>
    /// <returns>A dictionary containing graph statistics</returns>
    /// <exception cref="ArgumentNullException">Thrown when root is null</exception>
    public static Dictionary<string, object> GetGraphInfo<T>(GradTensor<T> root) where T : struct, INumber<T>
    {
        if (root == null)
            throw new ArgumentNullException(nameof(root));

        var nodes = TopologicalSort(root);
        var operationCounts = new Dictionary<string, int>();

        foreach (var node in nodes)
        {
            operationCounts[node.OperationName] = operationCounts.GetValueOrDefault(node.OperationName, 0) + 1;
        }

        return new Dictionary<string, object>
        {
            ["TotalNodes"] = nodes.Count,
            ["IsLeaf"] = root.IsLeaf,
            ["RequiresGrad"] = root.RequiresGrad,
            ["OperationCounts"] = operationCounts
        };
    }
}