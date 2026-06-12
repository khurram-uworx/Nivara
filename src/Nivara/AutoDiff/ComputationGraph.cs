using System.Numerics;

namespace Nivara.AutoDiff;

public sealed class ComputationGraph
{
    internal static void AddNode<T>(ReverseGradTensor<T> output, OpNode<T> node) where T : struct, INumber<T>
    {
        if (output == null)
            throw new ArgumentNullException(nameof(output));
        if (node == null)
            throw new ArgumentNullException(nameof(node));

        if (output.RequiresGrad)
        {
            output.GradFn = node;
        }
    }

    internal static void Backward<T>(ReverseGradTensor<T> tensor, NivaraColumn<T>? gradient = null, bool stripGradientNulls = true) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (!tensor.RequiresGrad)
        {
            throw new InvalidOperationException("Cannot perform backward pass on tensor that doesn't require gradients");
        }

        ValidateGraph(tensor);

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

        tensor.Grad = gradient;

        var nodeToOutputMap = new Dictionary<OpNode<T>, ReverseGradTensor<T>>();
        BuildNodeToOutputMap(tensor, nodeToOutputMap);

        var nodes = TopologicalSort(tensor);

        foreach (var node in nodes.AsEnumerable().Reverse())
        {
            if (!nodeToOutputMap.TryGetValue(node, out var outputTensor))
            {
                throw new InvalidOperationException($"Could not find output tensor for operation '{node.OperationName}'");
            }

            var outputGrad = outputTensor.Grad;
            if (outputGrad == null)
            {
                continue;
            }

            try
            {
                node.Apply(outputGrad, stripGradientNulls);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to apply gradients for operation '{node.OperationName}': {ex.Message}", ex);
            }
        }
    }

    private static void BuildNodeToOutputMap<T>(ReverseGradTensor<T> tensor, Dictionary<OpNode<T>, ReverseGradTensor<T>> nodeToOutputMap)
        where T : struct, INumber<T>
    {
        var visited = new HashSet<ReverseGradTensor<T>>();

        void Visit(ReverseGradTensor<T> t)
        {
            if (visited.Contains(t))
                return;

            visited.Add(t);

            if (t.GradFn != null)
            {
                nodeToOutputMap[t.GradFn] = t;

                foreach (var input in t.GradFn.Inputs)
                {
                    if (input is ReverseGradTensor<T> inputTensor)
                    {
                        Visit(inputTensor);
                    }
                }
            }
        }

        Visit(tensor);
    }

    internal static List<OpNode<T>> TopologicalSort<T>(ReverseGradTensor<T> root) where T : struct, INumber<T>
    {
        if (root == null)
            throw new ArgumentNullException(nameof(root));

        var visited = new HashSet<OpNode<T>>();
        var result = new List<OpNode<T>>();
        var visiting = new HashSet<OpNode<T>>();

        void Visit(OpNode<T>? node)
        {
            if (node == null || visited.Contains(node))
                return;

            if (visiting.Contains(node))
            {
                throw new InvalidOperationException($"Circular dependency detected in computation graph at operation '{node.OperationName}'");
            }

            visiting.Add(node);

            foreach (var input in node.Inputs)
            {
                if (input is ReverseGradTensor<T> gradTensor && gradTensor.GradFn != null)
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

    internal static void ValidateGraph<T>(ReverseGradTensor<T> root) where T : struct, INumber<T>
    {
        if (root == null)
            throw new ArgumentNullException(nameof(root));

        try
        {
            TopologicalSort(root);
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains("Circular dependency"))
        {
            throw new InvalidOperationException("Computation graph contains circular dependencies, which would cause infinite loops during backward pass", ex);
        }
    }

    public static void ZeroGrad<T>(ReverseGradTensor<T> tensor) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var visited = new HashSet<object>();

        void ClearGradients(ReverseGradTensor<T> t)
        {
            if (visited.Contains(t))
                return;

            visited.Add(t);
            t.ZeroGrad();

            if (t.GradFn != null)
            {
                foreach (var input in t.GradFn.Inputs)
                {
                    if (input is ReverseGradTensor<T> inputTensor)
                    {
                        ClearGradients(inputTensor);
                    }
                }
            }
        }

        ClearGradients(tensor);
    }

    public static Dictionary<string, object> GetGraphInfo<T>(ReverseGradTensor<T> root) where T : struct, INumber<T>
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
