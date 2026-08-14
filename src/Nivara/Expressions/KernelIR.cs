using Nivara.Helpers;

namespace Nivara.Expressions;

/// <summary>
/// Flat kernel-plan operations for fused expression evaluation. The plan is the single
/// representation all fused backends consume: the compiled expression-tree target, the
/// generic span interpreter, and the <see cref="System.Numerics.Tensors.TensorPrimitives"/>
/// single-op dispatcher. Each backend implements the subset of operations it supports and the
/// planner selects the best one (issue #167).
/// </summary>
internal enum KernelOp
{
    Column,
    Literal,
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    And,
    Or,
    Not
}

/// <summary>
/// One node of a post-order kernel plan. Binary nodes reference child node indices; a
/// <see cref="KernelOp.Column"/> node's <see cref="KernelNode.Left"/> is the leaf index into
/// <see cref="KernelPlan.Columns"/>; a <see cref="KernelOp.Literal"/> node's
/// <see cref="KernelNode.Value"/> is the original literal value (each backend coerces once).
/// <see cref="KernelNode.ComputeType"/> is the promoted type the operation runs in.
/// </summary>
internal readonly record struct KernelNode(KernelOp Op, int Left, int Right, object? Value, Type ComputeType);

/// <summary>
/// A lowered fused-expression plan: a flat post-order node graph over the expression's leaf
/// columns plus the routing flags the planner uses to select a backend (issue #167).
/// </summary>
internal sealed class KernelPlan
{
    /// <summary>
    /// Initializes a new KernelPlan from the lowered nodes and the fused plan it came from.
    /// </summary>
    /// <param name="nodes">Post-order nodes; the root is the last node</param>
    /// <param name="columns">The leaf column bindings (order matches <see cref="KernelOp.Column"/> Left values)</param>
    /// <param name="resultType">The expression result element type</param>
    /// <param name="hasNulls">Whether any leaf column carries a null mask</param>
    /// <param name="signature">Structural cache key (from <see cref="FusedExpressionPlan.Signature"/>)</param>
    /// <param name="isGenericMath">Whether the result type implements generic math</param>
    /// <param name="rootNode">Index of the plan root node</param>
    public KernelPlan(
        IReadOnlyList<KernelNode> nodes,
        IReadOnlyList<FusedColumnBinding> columns,
        Type resultType,
        bool hasNulls,
        string signature,
        bool isGenericMath,
        int rootNode)
    {
        Nodes = nodes;
        Columns = columns;
        ResultType = resultType;
        HasNulls = hasNulls;
        Signature = signature;
        RootNode = rootNode;

        IsUniformNumeric = isGenericMath
            && resultType != typeof(bool)
            && columns.All(c => c.Column.ElementType == resultType);

        IsTensorPrimitivesCandidate = IsUniformNumeric && IsSingleDispatachableBinary(nodes);

        MaxStackDepth = nodes.Count(n => n.Op is KernelOp.Column or KernelOp.Literal);
    }

    /// <summary>
    /// Gets the post-order node list. The root is the last node.
    /// </summary>
    public IReadOnlyList<KernelNode> Nodes { get; }

    /// <summary>
    /// Gets the leaf column bindings, indexed by <see cref="KernelOp.Column"/> node Left values.
    /// </summary>
    public IReadOnlyList<FusedColumnBinding> Columns { get; }

    /// <summary>
    /// Gets the expression result element type.
    /// </summary>
    public Type ResultType { get; }

    /// <summary>
    /// Gets whether any leaf column has nulls (mask must be OR'd into the result).
    /// </summary>
    public bool HasNulls { get; }

    /// <summary>
    /// Gets the structural cache key for this plan shape.
    /// </summary>
    public string Signature { get; }

    /// <summary>
    /// Gets the index of the plan root node in <see cref="Nodes"/>.
    /// </summary>
    public int RootNode { get; }

    /// <summary>
    /// Gets whether the plan is uniform numeric generic-math: every leaf shares the result element
    /// type and the result is not bool. This is the domain of the generic span interpreter.
    /// </summary>
    public bool IsUniformNumeric { get; }

    /// <summary>
    /// Gets whether the plan is a single element-wise Add/Subtract/Multiply/Divide over leaves and
    /// literals on a uniform numeric domain, eligible for direct <see cref="System.Numerics.Tensors.TensorPrimitives"/>
    /// dispatch. Modulo is excluded (no generic <see cref="System.Numerics.Tensors.TensorPrimitives"/>
    /// overload in the pinned BCL version).
    /// </summary>
    public bool IsTensorPrimitivesCandidate { get; }

    /// <summary>
    /// Gets a safe upper bound on the interpreter value-stack depth for post-order evaluation
    /// (every Column/Literal node pushes one value).
    /// </summary>
    public int MaxStackDepth { get; }

    static bool IsSingleDispatachableBinary(IReadOnlyList<KernelNode> nodes)
    {
        var computeCount = 0;
        var computeOp = KernelOp.Column;
        foreach (var node in nodes)
        {
            if (node.Op is KernelOp.Add or KernelOp.Subtract or KernelOp.Multiply or KernelOp.Divide or KernelOp.Modulo)
            {
                computeCount++;
                computeOp = node.Op;
            }
        }

        return computeCount == 1
            && computeOp is KernelOp.Add or KernelOp.Subtract or KernelOp.Multiply or KernelOp.Divide;
    }
}
