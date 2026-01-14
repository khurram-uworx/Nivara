using System.Numerics;

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

        // Validate graph structure (will throw if cycles detected)
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

        // Set initial gradient for the root tensor
        tensor.Grad = gradient;

        // Build a map to track which tensor each OpNode produces
        // This is needed because OpNode stores inputs but we need to know the output
        var nodeToOutputMap = new Dictionary<OpNode, GradTensor<T>>();
        BuildNodeToOutputMap(tensor, nodeToOutputMap);

        // Get topological ordering of nodes (from inputs to outputs)
        var nodes = TopologicalSort(tensor);

        // Apply backward functions in reverse topological order (from output to inputs)
        // This ensures we process gradients from the output back to the inputs
        foreach (var node in nodes.AsEnumerable().Reverse())
        {
            // Find the output tensor that this node produced
            if (!nodeToOutputMap.TryGetValue(node, out var outputTensor))
            {
                // This shouldn't happen if the graph is properly constructed
                throw new InvalidOperationException($"Could not find output tensor for operation '{node.OperationName}'");
            }

            // Get the gradient for the output tensor
            var outputGrad = outputTensor.Grad;
            if (outputGrad == null)
            {
                // If there's no gradient for this output, skip it
                // This can happen for intermediate tensors that don't contribute to the final output
                continue;
            }

            try
            {
                // Convert gradient to object column for the backward function
                var gradAsObjects = new object[outputGrad.Length];
                for (int i = 0; i < outputGrad.Length; i++)
                {
                    gradAsObjects[i] = outputGrad[i]!;
                }
                var gradColumn = NivaraColumn<object>.CreateForReferenceType(gradAsObjects);

                // Apply the backward function - this will accumulate gradients into input tensors
                node.Apply(gradColumn);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to apply gradients for operation '{node.OperationName}': {ex.Message}", ex);
            }
        }
    }

    /// <summary>
    /// Builds a map from OpNode to the output tensor it produces
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor</typeparam>
    /// <param name="tensor">The tensor to start mapping from</param>
    /// <param name="nodeToOutputMap">The map to populate</param>
    private static void BuildNodeToOutputMap<T>(GradTensor<T> tensor, Dictionary<OpNode, GradTensor<T>> nodeToOutputMap)
        where T : struct, INumber<T>
    {
        var visited = new HashSet<GradTensor<T>>();

        void Visit(GradTensor<T> t)
        {
            if (visited.Contains(t))
                return;

            visited.Add(t);

            // If this tensor has a GradFn, it means it was produced by an operation
            if (t.GradFn != null)
            {
                nodeToOutputMap[t.GradFn] = t;

                // Recursively visit input tensors
                foreach (var input in t.GradFn.Inputs)
                {
                    if (input is GradTensor<T> inputTensor)
                    {
                        Visit(inputTensor);
                    }
                }
            }
        }

        Visit(tensor);
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