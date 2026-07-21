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

        var plan = BuildBackwardPlan(tensor);

        for (int i = plan.Nodes.Count - 1; i >= 0; i--)
        {
            var node = plan.Nodes[i];
            var outputTensor = plan.NodeToOutputMap[node];
            var outputGrad = outputTensor.Grad;

            if (outputGrad == null)
                continue;

            node.Apply(outputGrad, stripGradientNulls);
        }
    }

    readonly struct BackwardPlan<T> where T : struct, INumber<T>
    {
        public readonly List<OpNode<T>> Nodes;
        public readonly Dictionary<OpNode<T>, ReverseGradTensor<T>> NodeToOutputMap;

        public BackwardPlan(List<OpNode<T>> nodes, Dictionary<OpNode<T>, ReverseGradTensor<T>> nodeToOutputMap)
        {
            Nodes = nodes;
            NodeToOutputMap = nodeToOutputMap;
        }
    }

    private static BackwardPlan<T> BuildBackwardPlan<T>(ReverseGradTensor<T> root) where T : struct, INumber<T>
    {
        var visited = new HashSet<OpNode<T>>();
        var visiting = new HashSet<OpNode<T>>();
        var result = new List<OpNode<T>>();
        var nodeToOutput = new Dictionary<OpNode<T>, ReverseGradTensor<T>>();

        void VisitTensors(ReverseGradTensor<T> t)
        {
            if (t.GradFn == null)
                return;

            nodeToOutput[t.GradFn] = t;

            foreach (var input in t.GradFn.Inputs)
            {
                if (input is ReverseGradTensor<T> inputTensor)
                    VisitTensors(inputTensor);
            }
        }

        void VisitNodes(OpNode<T>? node)
        {
            if (node == null || visited.Contains(node))
                return;

            if (visiting.Contains(node))
                throw new InvalidOperationException(
                    $"Circular dependency detected in computation graph at operation '{node.OperationName}'");

            visiting.Add(node);

            foreach (var input in node.Inputs)
            {
                if (input is ReverseGradTensor<T> gradTensor && gradTensor.GradFn != null)
                    VisitNodes(gradTensor.GradFn);
            }

            visiting.Remove(node);
            visited.Add(node);
            result.Add(node);
        }

        VisitTensors(root);
        VisitNodes(root.GradFn);

        return new BackwardPlan<T>(result, nodeToOutput);
    }

    internal static List<OpNode<T>> TopologicalSort<T>(ReverseGradTensor<T> root) where T : struct, INumber<T>
    {
        if (root == null)
            throw new ArgumentNullException(nameof(root));

        return BuildBackwardPlan(root).Nodes;
    }

    internal static void ValidateGraph<T>(ReverseGradTensor<T> root) where T : struct, INumber<T>
    {
        if (root == null)
            throw new ArgumentNullException(nameof(root));

        BuildBackwardPlan(root);
    }

    public static void ZeroGrad<T>(ReverseGradTensor<T> tensor) where T : struct, INumber<T>
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var visited = new HashSet<ReverseGradTensor<T>>();

        void ClearGradients(ReverseGradTensor<T> t)
        {
            if (!visited.Add(t))
                return;

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
