using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;
using System.Reflection;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Enforces the <c>OpNode.Inputs</c> contract across both automatic differentiation
/// modes: an operation lists every tensor whose gradient its backward is responsible
/// for, and an argument it deliberately excludes is rejected loudly rather than
/// silently ignored.
///
/// The defect shape these tests exist for (#487) needs at least two tensor
/// arguments. Setting exactly one of them to require a gradient forces the op either
/// to track it or to reject it; setting every argument at once would pass even when
/// one is excluded, which is why each row isolates a single argument.
///
/// Coverage is deliberately scoped to that defect shape, and the three completeness
/// tests below enforce the scope rather than leaving it to drift:
/// <list type="bullet">
/// <item>Every public static op taking two or more tensor arguments, plus
/// <c>Concat</c> (one argument that is an array of tensors), must have an op row.</item>
/// <item>Every public <see cref="Module{T}"/> subclass owning a
/// <see cref="Parameter{T}"/> must have a module row, because those are the sites
/// whose backward can accumulate into a parameter the node forgot to list.</item>
/// <item>Arguments excluded by type, not by choice, are pinned explicitly.</item>
/// </list>
/// Single-tensor-argument ops are outside this scope: with only one tensor there is
/// nothing for a node to omit, so they cannot exhibit the defect.
/// </summary>
[TestFixture]
public class OpNodeInputContractTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    public sealed record ReverseRow(
        string Op,
        string ArgName,
        Func<ReverseGradTensor<float>> Tracked,
        Func<ReverseGradTensor<float>, ReverseGradTensor<float>> Call)
    {
        public override string ToString() => $"{Op}/{ArgName}";
    }

    public sealed record ForwardRow(
        string Op,
        string ArgName,
        Func<ForwardGradTensor<float>> Tracked,
        Func<ForwardGradTensor<float>, ForwardGradTensor<float>> Call)
    {
        public override string ToString() => $"{Op}/{ArgName}";
    }

    public sealed record ModuleRow(string Module, Func<ModuleRowContext> Build)
    {
        public override string ToString() => Module;
    }

    public /// <summary>
    /// One module's forward graph plus the parameters that graph must reach.
    /// <paramref name="InputIsGraphInput"/> is false for modules whose input is an
    /// index constant consumed as integers (Embedding, SparseEmbedding): the tensor
    /// never enters the graph, so ZeroGrad correctly never touches it.
    /// </summary>
    sealed record ModuleRowContext(
        ReverseGradTensor<float> Input,
        ReverseGradTensor<float> Output,
        ReverseGradTensor<float>[] ExpectedParameters,
        bool InputIsGraphInput = true);

    // ---------------------------------------------------------------------
    // Reverse mode: the contract itself
    // ---------------------------------------------------------------------

    [TestCaseSource(nameof(ReverseRows))]
    public void ReverseOp_SoleGradBearingArgument_IsTrackedOrRejected(ReverseRow row)
    {
        var tracked = row.Tracked();

        ReverseGradTensor<float>? result = null;
        ArgumentException? rejection = null;
        try
        {
            result = row.Call(tracked);
        }
        catch (ArgumentException ex)
        {
            rejection = ex;
        }

        if (rejection != null)
        {
            Assert.That(rejection!.Message, Does.Contain("non-differentiable"),
                $"{row} rejected '{row.ArgName}' but not as a constant, so the contract does not explain why.");
            Assert.That(rejection.ParamName, Is.EqualTo(row.ArgName));
            return;
        }

        Assert.That(result, Is.Not.Null, $"{row} returned null.");
        Assert.That(result!.RequiresGrad, Is.True,
            $"{row} accepted a requires-grad '{row.ArgName}' yet produced an output that requires no gradient, " +
            "so Backward can never reach that argument.");
        Assert.That(result.GradFn, Is.Not.Null, $"{row} produced no graph node for a requires-grad '{row.ArgName}'.");
        Assert.That(result.GradFn!.Inputs, Does.Contain(tracked),
            $"{row} accumulated into '{row.ArgName}' but omitted it from Inputs, so its gradient is unreachable.");
    }

    // ---------------------------------------------------------------------
    // Forward mode: the same contract on the tangent predicate
    // ---------------------------------------------------------------------

    [TestCaseSource(nameof(ForwardRows))]
    public void ForwardOp_SoleTangentBearingArgument_IsTrackedOrRejected(ForwardRow row)
    {
        var tracked = row.Tracked();

        ForwardGradTensor<float>? result = null;
        ArgumentException? rejection = null;
        try
        {
            result = row.Call(tracked);
        }
        catch (ArgumentException ex)
        {
            rejection = ex;
        }

        if (rejection != null)
        {
            Assert.That(rejection!.Message, Does.Contain("non-differentiable"),
                $"{row} rejected '{row.ArgName}' but not as a constant, so the contract does not explain why.");
            Assert.That(rejection.ParamName, Is.EqualTo(row.ArgName));
            return;
        }

        Assert.That(result, Is.Not.Null, $"{row} returned null.");
        Assert.That(result!.RequiresTangent, Is.True,
            $"{row} accepted a tangent-bearing '{row.ArgName}' yet dropped it from its tracking predicate. " +
            "Forward mode has no graph and no Backward to fail later, so the JVP silently loses that contribution.");
    }

    // ---------------------------------------------------------------------
    // Completeness: every op with two or more tensor arguments must have a row
    // ---------------------------------------------------------------------

    [Test]
    public void ReverseOps_WithMultipleTensorArguments_EveryOneHasAContractRow()
        => AssertEveryMultiTensorOpCovered(typeof(ReverseGradTensor<>), ReverseRows().Select(r => r.Op));

    [Test]
    public void ForwardOps_WithMultipleTensorArguments_EveryOneHasAContractRow()
        => AssertEveryMultiTensorOpCovered(typeof(ForwardGradTensor<>), ForwardRows().Select(r => r.Op));

    // ---------------------------------------------------------------------
    // Completeness: every Parameter-owning module must have a module row
    // ---------------------------------------------------------------------

    [Test]
    public void ParameterOwningModules_EveryOneHasAContractRow()
    {
        // Types come back as open generics, so "Conv1d`1" has to match the row's "Conv1d".
        var covered = ModuleRows().Select(r => r.Module).ToHashSet(StringComparer.Ordinal);
        var uncovered = ParameterOwningModules()
            .Where(m => !covered.Contains(ModuleRowNameOf(m)))
            .Select(ModuleRowNameOf)
            .OrderBy(n => n, StringComparer.Ordinal)
            .ToList();

        Assert.That(uncovered, Is.Empty,
            "These public Module<T> subclasses own a Parameter<T>, so their forward graph must reach it, " +
            "but no module contract row covers them. Add a row to ModuleRows().");
    }

    // ---------------------------------------------------------------------
    // Module rows: parameters are reachable, and ZeroGrad reaches them
    // ---------------------------------------------------------------------

    [TestCaseSource(nameof(ModuleRows))]
    public void Module_ParameterTensors_AreReachableFromOutput(ModuleRow row)
    {
        var context = row.Build();

        Assert.That(context.Output.RequiresGrad, Is.True, $"{row} lost the requires-grad input.");

        var reachable = ReachableTensors(context.Output);
        foreach (var parameter in context.ExpectedParameters)
            Assert.That(reachable, Does.Contain(parameter),
                $"{row} accumulates into a parameter its output graph never reaches, so no gradient can flow to it.");
    }

    [TestCaseSource(nameof(ModuleRows))]
    public void Module_ZeroGrad_ClearsParameterGradients(ModuleRow row)
    {
        var context = row.Build();

        context.Input.Grad = Ones(context.Input.Length);
        foreach (var parameter in context.ExpectedParameters)
            parameter.Grad = Ones(parameter.Length);

        GradientUtils.ZeroGrad(context.Output);

        if (context.InputIsGraphInput)
            Assert.That(context.Input.Grad, Is.Null, $"{row}: ZeroGrad left the input gradient behind.");
        else
            Assert.That(context.Input.Grad, Is.Not.Null,
                $"{row}: the input is an index constant that never enters the graph, so ZeroGrad " +
                "should not have touched it. If it now does, the input stopped being a constant.");
        foreach (var parameter in context.ExpectedParameters)
            Assert.That(parameter.Grad, Is.Null,
                $"{row}: GradientUtils.ZeroGrad does not reach this parameter, contradicting its own doc.");
    }

    [Test]
    public void BatchNorm_EvalPath_ListsAffineParameters()
    {
        // #494 resolved this in PyTorch's favour. The eval forward still computes
        // gamma * xhat + beta, so gamma and beta stay in the differentiable path and
        // their backward must produce gradients -- the eval path is not a frozen one.
        //
        // This assertion previously pinned the opposite conclusion ("the eval backward
        // accumulates into input alone, so [input] is contract-correct"), which was true
        // of the closure at the time but wrong about the math. Do not restore it: if the
        // eval path ever stops reaching its parameters again, that is the bug, not the
        // contract.
        var bn = new BatchNorm1d<float>(numFeatures: 3);
        bn.Eval();
        var input = Tensor3D(Rand(6, 901), 2, 3, 1, requiresGrad: true);

        var output = bn.Forward(input);

        Assert.That(output.GradFn, Is.Not.Null);
        Assert.That(output.GradFn!.Inputs, Is.EqualTo(new[] { input, bn.Weight!.Tensor, bn.Bias!.Tensor }));
    }

    [Test]
    public void BatchNorm_EvalPath_AffineFalse_ListsOnlyTheInput()
    {
        // Without affine there is no gamma or beta to differentiate, so the eval node
        // really does have nothing but the input to list.
        var bn = new BatchNorm1d<float>(numFeatures: 3, affine: false);
        bn.Eval();
        var input = Tensor3D(Rand(6, 902), 2, 3, 1, requiresGrad: true);

        var output = bn.Forward(input);

        Assert.That(output.GradFn, Is.Not.Null);
        Assert.That(output.GradFn!.Inputs, Is.EqualTo(new[] { input }));
    }

    [Test]
    public void BatchNorm2d_EvalPath_ListsAffineParameters()
    {
        var bn = new BatchNorm2d<float>(numFeatures: 3);
        bn.Eval();
        var input = Tensor4D(Rand(24, 903), 2, 3, 2, 2, requiresGrad: true);

        var output = bn.Forward(input);

        Assert.That(output.GradFn, Is.Not.Null);
        Assert.That(output.GradFn!.Inputs, Is.EqualTo(new[] { input, bn.Weight!.Tensor, bn.Bias!.Tensor }));
    }

    // ---------------------------------------------------------------------
    // Arguments excluded by type rather than by choice
    // ---------------------------------------------------------------------

    [Test]
    public void NonTensorArguments_AreExcludedStructurally()
    {
        // Gather takes int[] indices and DropoutWithMask takes a ReadOnlySpan<bool>
        // keep-mask, so neither can carry a gradient at all and neither can silently
        // drop one. Recorded because "excluded by type" is a different claim from
        // "excluded by choice", and only the second needs a guard.
        var reverseGather = typeof(ReverseGradOperations).GetMethod(nameof(ReverseGradOperations.Gather));
        Assert.That(reverseGather, Is.Not.Null);
        Assert.That(reverseGather!.GetParameters()[1].ParameterType, Is.EqualTo(typeof(int[])));

        var forwardGather = typeof(ForwardGradOperations).GetMethod(nameof(ForwardGradOperations.Gather));
        Assert.That(forwardGather, Is.Not.Null);
        Assert.That(forwardGather!.GetParameters()[1].ParameterType, Is.EqualTo(typeof(int[])));

        var dropoutWithMask = typeof(ReverseGradOperations).GetMethod(
            nameof(ReverseGradOperations.DropoutWithMask), BindingFlags.NonPublic | BindingFlags.Static);
        Assert.That(dropoutWithMask, Is.Not.Null, "DropoutWithMask was expected to exist as an internal op.");
        Assert.That(dropoutWithMask!.GetParameters()[1].ParameterType, Is.EqualTo(typeof(ReadOnlySpan<bool>)));
    }

    // ---------------------------------------------------------------------
    // Reverse op rows
    // ---------------------------------------------------------------------

    static IEnumerable<ReverseRow> ReverseRows()
    {
        foreach (var row in ReverseBinaryRows("Add", 1001, (a, b) => ReverseGradOperations.Add(a, b)))
            yield return row;
        foreach (var row in ReverseBinaryRows("Subtract", 1003, (a, b) => ReverseGradOperations.Subtract(a, b)))
            yield return row;
        foreach (var row in ReverseBinaryRows("Multiply", 1005, (a, b) => ReverseGradOperations.Multiply(a, b)))
            yield return row;
        foreach (var row in ReverseBinaryRows("Divide", 1007, (a, b) => ReverseGradOperations.Divide(a, b)))
            yield return row;
        foreach (var row in ReverseBinaryRows("MatMul", 1009, (a, b) => ReverseGradOperations.MatMul(a, b)))
            yield return row;
        foreach (var row in ReverseBinaryRows("MatMulTransposedB", 1011,
                     (a, b) => ReverseGradOperations.MatMulTransposedB(a, b)))
            yield return row;

        // AddBias takes a per-column bias, so its second argument is not the same shape as its first.
        yield return new ReverseRow("AddBias", "a",
            () => Mat2D(Rand(4, 1013), 2, 2, requiresGrad: true),
            tracked => ReverseGradOperations.AddBias(tracked, Vec(Rand(2, 1014), requiresGrad: false)));
        yield return new ReverseRow("AddBias", "bias",
            () => Vec(Rand(2, 1015), requiresGrad: true),
            tracked => ReverseGradOperations.AddBias(Mat2D(Rand(4, 1016), 2, 2, requiresGrad: false), tracked));

        foreach (var row in ReverseBroadcastRows("BroadcastMultiply", 1020,
                     (input, scale) => ReverseGradOperations.BroadcastMultiply(input, scale)))
            yield return row;
        foreach (var row in ReverseBroadcastRows("BroadcastAdd", 1025,
                     (input, bias) => ReverseGradOperations.BroadcastAdd(input, bias)))
            yield return row;

        foreach (var row in ReverseNormalPairRows("KlDivergence", 1030,
                     (mean, logVar) => ReverseGradOperations.KlDivergence(mean, logVar)))
            yield return row;
        foreach (var row in ReverseNormalPairRows("SampleNormal", 1035,
                     (mean, logVar) => ReverseGradOperations.SampleNormal(mean, logVar, seed: 7)))
            yield return row;

        // Concat takes an array of tensors, so its "argument" is the array itself.
        yield return new ReverseRow("Concat", "tensors[1]",
            () => Mat2D(Rand(4, 1040), 2, 2, requiresGrad: true),
            tracked => ReverseGradOperations.Concat(
                [Mat2D(Rand(4, 1041), 2, 2, requiresGrad: false), tracked]));

        foreach (var row in ReverseAttentionRows(1050))
            yield return row;

        // indices carries integer row selectors, so it is a non-differentiable constant.
        yield return new ReverseRow("SparseEmbeddingBag", "indices",
            () => Indices2D(Vec([0f, 1f, 1f, 0f], requiresGrad: true)),
            tracked => ReverseGradOperations.SparseEmbeddingBag(
                Mat2D(Rand(8, 1070), 4, 2, requiresGrad: false), tracked));

        yield return new ReverseRow("SparseEmbeddingBag", "weight",
            () => Mat2D(Rand(8, 1071), 4, 2, requiresGrad: true),
            tracked => ReverseGradOperations.SparseEmbeddingBag(
                tracked, Indices2D(Vec([0f, 1f, 1f, 0f], requiresGrad: false))));
    }

    static IEnumerable<ReverseRow> ReverseBinaryRows(
        string op, int seed,
        Func<ReverseGradTensor<float>, ReverseGradTensor<float>, ReverseGradTensor<float>> call)
    {
        yield return new ReverseRow(op, "a",
            () => Mat2D(Rand(4, seed), 2, 2, requiresGrad: true),
            tracked => call(tracked, Mat2D(Rand(4, seed + 1), 2, 2, requiresGrad: false)));

        yield return new ReverseRow(op, "b",
            () => Mat2D(Rand(4, seed + 1), 2, 2, requiresGrad: true),
            tracked => call(Mat2D(Rand(4, seed), 2, 2, requiresGrad: false), tracked));
    }

    static IEnumerable<ReverseRow> ReverseBroadcastRows(
        string op, int seed,
        Func<ReverseGradTensor<float>, ReverseGradTensor<float>, ReverseGradTensor<float>> call)
    {
        yield return new ReverseRow(op, "input",
            () => Tensor3D(Rand(8, seed), 2, 2, 2, requiresGrad: true),
            tracked => call(tracked, Vec(Rand(2, seed + 1), requiresGrad: false)));

        yield return new ReverseRow(op, "scale",
            () => Vec(Rand(2, seed + 2), requiresGrad: true),
            tracked => call(Tensor3D(Rand(8, seed + 3), 2, 2, 2, requiresGrad: false), tracked));
    }

    static IEnumerable<ReverseRow> ReverseNormalPairRows(
        string op, int seed,
        Func<ReverseGradTensor<float>, ReverseGradTensor<float>, ReverseGradTensor<float>> call)
    {
        yield return new ReverseRow(op, "mean",
            () => Vec(Rand(4, seed), requiresGrad: true),
            tracked => call(tracked, Vec(Rand(4, seed + 1), requiresGrad: false)));

        yield return new ReverseRow(op, "logVar",
            () => Vec(Rand(4, seed + 2), requiresGrad: true),
            tracked => call(Vec(Rand(4, seed + 3), requiresGrad: false), tracked));
    }

    static IEnumerable<ReverseRow> ReverseAttentionRows(int seed)
    {
        for (int i = 0; i < 3; i++)
        {
            var argName = new[] { "query", "key", "value" }[i];
            yield return new ReverseRow("MultiHeadAttention", argName,
                () => Mat2D(Rand(16, seed), 4, 4, requiresGrad: true),
                tracked => ReverseGradOperations.MultiHeadAttention(
                    AttentionArg(argName, 0, tracked, Mat2D(Rand(16, seed + 1), 4, 4, requiresGrad: false)),
                    AttentionArg(argName, 1, tracked, Mat2D(Rand(16, seed + 2), 4, 4, requiresGrad: false)),
                    AttentionArg(argName, 2, tracked, Mat2D(Rand(16, seed + 3), 4, 4, requiresGrad: false)),
                    numHeads: 2, scale: 0.5f));

            yield return new ReverseRow("BatchedMultiHeadAttention", argName,
                () => Tensor3D(Rand(32, seed + 10), 2, 4, 4, requiresGrad: true),
                tracked => ReverseGradOperations.BatchedMultiHeadAttention(
                    BatchAttentionArg(argName, 0, tracked, Tensor3D(Rand(32, seed + 11), 2, 4, 4, requiresGrad: false)),
                    BatchAttentionArg(argName, 1, tracked, Tensor3D(Rand(32, seed + 12), 2, 4, 4, requiresGrad: false)),
                    BatchAttentionArg(argName, 2, tracked, Tensor3D(Rand(32, seed + 13), 2, 4, 4, requiresGrad: false)),
                    numHeads: 2, scale: 0.5f));
        }

        // The mask is excluded from the tracking predicate, so a gradient on it must throw.
        yield return new ReverseRow("MultiHeadAttention", "mask",
            () => Mat2D(Rand(16, seed + 4), 4, 4, requiresGrad: true),
            tracked => ReverseGradOperations.MultiHeadAttention(
                Mat2D(Rand(16, seed + 5), 4, 4, requiresGrad: false),
                Mat2D(Rand(16, seed + 6), 4, 4, requiresGrad: false),
                Mat2D(Rand(16, seed + 7), 4, 4, requiresGrad: false),
                numHeads: 2, scale: 0.5f, mask: tracked));

        yield return new ReverseRow("BatchedMultiHeadAttention", "mask",
            () => Tensor3D(Rand(32, seed + 14), 2, 4, 4, requiresGrad: true),
            tracked => ReverseGradOperations.BatchedMultiHeadAttention(
                Tensor3D(Rand(32, seed + 15), 2, 4, 4, requiresGrad: false),
                Tensor3D(Rand(32, seed + 16), 2, 4, 4, requiresGrad: false),
                Tensor3D(Rand(32, seed + 17), 2, 4, 4, requiresGrad: false),
                numHeads: 2, scale: 0.5f, mask: tracked));
    }

    static ReverseGradTensor<float> AttentionArg(
        string argName, int position, ReverseGradTensor<float> tracked, ReverseGradTensor<float> other)
        => argName == new[] { "query", "key", "value" }[position] ? tracked : other;

    static ReverseGradTensor<float> BatchAttentionArg(
        string argName, int position, ReverseGradTensor<float> tracked, ReverseGradTensor<float> other)
        => AttentionArg(argName, position, tracked, other);

    // ---------------------------------------------------------------------
    // Forward op rows
    // ---------------------------------------------------------------------

    static IEnumerable<ForwardRow> ForwardRows()
    {
        foreach (var row in ForwardBinaryRows("Add", 2001, (a, b) => ForwardGradOperations.Add(a, b)))
            yield return row;
        foreach (var row in ForwardBinaryRows("Subtract", 2003, (a, b) => ForwardGradOperations.Subtract(a, b)))
            yield return row;
        foreach (var row in ForwardBinaryRows("Multiply", 2005, (a, b) => ForwardGradOperations.Multiply(a, b)))
            yield return row;
        foreach (var row in ForwardBinaryRows("Divide", 2007, (a, b) => ForwardGradOperations.Divide(a, b)))
            yield return row;
        foreach (var row in ForwardBinaryRows("MatMul", 2009, (a, b) => ForwardGradOperations.MatMul(a, b)))
            yield return row;
        foreach (var row in ForwardBinaryRows("MatMulTransposedB", 2011,
                     (a, b) => ForwardGradOperations.MatMulTransposedB(a, b)))
            yield return row;

        yield return new ForwardRow("AddBias", "a",
            () => FwdMat2D(Rand(4, 2013), 2, 2, tangent: true),
            tracked => ForwardGradOperations.AddBias(tracked, FwdVec(Rand(2, 2014), tangent: false)));
        yield return new ForwardRow("AddBias", "bias",
            () => FwdVec(Rand(2, 2015), tangent: true),
            tracked => ForwardGradOperations.AddBias(FwdMat2D(Rand(4, 2016), 2, 2, tangent: false), tracked));

        foreach (var row in ForwardBroadcastRows("BroadcastMultiply", 2020,
                     (input, scale) => ForwardGradOperations.BroadcastMultiply(input, scale)))
            yield return row;
        foreach (var row in ForwardBroadcastRows("BroadcastAdd", 2025,
                     (input, bias) => ForwardGradOperations.BroadcastAdd(input, bias)))
            yield return row;

        foreach (var row in ForwardNormalPairRows("KlDivergence", 2030,
                     (mean, logVar) => ForwardGradOperations.KlDivergence(mean, logVar)))
            yield return row;
        foreach (var row in ForwardNormalPairRows("SampleNormal", 2035,
                     (mean, logVar) => ForwardGradOperations.SampleNormal(mean, logVar, seed: 7)))
            yield return row;

        yield return new ForwardRow("Concat", "tensors[1]",
            () => FwdMat2D(Rand(4, 2040), 2, 2, tangent: true),
            tracked => ForwardGradOperations.Concat(
                [FwdMat2D(Rand(4, 2041), 2, 2, tangent: false), tracked]));

        foreach (var row in ForwardAttentionRows(2050))
            yield return row;

        yield return new ForwardRow("SparseEmbeddingBag", "indices",
            () => FwdIndices2D(FwdVec([0f, 1f, 1f, 0f], tangent: true)),
            tracked => ForwardGradOperations.SparseEmbeddingBag(
                FwdMat2D(Rand(8, 2070), 4, 2, tangent: false), tracked));

        yield return new ForwardRow("SparseEmbeddingBag", "weight",
            () => FwdMat2D(Rand(8, 2071), 4, 2, tangent: true),
            tracked => ForwardGradOperations.SparseEmbeddingBag(
                tracked, FwdIndices2D(FwdVec([0f, 1f, 1f, 0f], tangent: false))));
    }

    static IEnumerable<ForwardRow> ForwardBinaryRows(
        string op, int seed,
        Func<ForwardGradTensor<float>, ForwardGradTensor<float>, ForwardGradTensor<float>> call)
    {
        yield return new ForwardRow(op, "a",
            () => FwdMat2D(Rand(4, seed), 2, 2, tangent: true),
            tracked => call(tracked, FwdMat2D(Rand(4, seed + 1), 2, 2, tangent: false)));

        yield return new ForwardRow(op, "b",
            () => FwdMat2D(Rand(4, seed + 1), 2, 2, tangent: true),
            tracked => call(FwdMat2D(Rand(4, seed), 2, 2, tangent: false), tracked));
    }

    static IEnumerable<ForwardRow> ForwardBroadcastRows(
        string op, int seed,
        Func<ForwardGradTensor<float>, ForwardGradTensor<float>, ForwardGradTensor<float>> call)
    {
        yield return new ForwardRow(op, "input",
            () => FwdTensor3D(Rand(8, seed), 2, 2, 2, tangent: true),
            tracked => call(tracked, FwdVec(Rand(2, seed + 1), tangent: false)));

        yield return new ForwardRow(op, "scale",
            () => FwdVec(Rand(2, seed + 2), tangent: true),
            tracked => call(FwdTensor3D(Rand(8, seed + 3), 2, 2, 2, tangent: false), tracked));
    }

    static IEnumerable<ForwardRow> ForwardNormalPairRows(
        string op, int seed,
        Func<ForwardGradTensor<float>, ForwardGradTensor<float>, ForwardGradTensor<float>> call)
    {
        yield return new ForwardRow(op, "mean",
            () => FwdVec(Rand(4, seed), tangent: true),
            tracked => call(tracked, FwdVec(Rand(4, seed + 1), tangent: false)));

        yield return new ForwardRow(op, "logVar",
            () => FwdVec(Rand(4, seed + 2), tangent: true),
            tracked => call(FwdVec(Rand(4, seed + 3), tangent: false), tracked));
    }

    static IEnumerable<ForwardRow> ForwardAttentionRows(int seed)
    {
        for (int i = 0; i < 3; i++)
        {
            var argName = new[] { "query", "key", "value" }[i];
            yield return new ForwardRow("MultiHeadAttention", argName,
                () => FwdMat2D(Rand(16, seed), 4, 4, tangent: true),
                tracked => ForwardGradOperations.MultiHeadAttention(
                    FwdAttentionArg(argName, 0, tracked, FwdMat2D(Rand(16, seed + 1), 4, 4, tangent: false)),
                    FwdAttentionArg(argName, 1, tracked, FwdMat2D(Rand(16, seed + 2), 4, 4, tangent: false)),
                    FwdAttentionArg(argName, 2, tracked, FwdMat2D(Rand(16, seed + 3), 4, 4, tangent: false)),
                    numHeads: 2, scale: 0.5f));

            yield return new ForwardRow("BatchedMultiHeadAttention", argName,
                () => FwdTensor3D(Rand(32, seed + 10), 2, 4, 4, tangent: true),
                tracked => ForwardGradOperations.BatchedMultiHeadAttention(
                    FwdAttentionArg(argName, 0, tracked, FwdTensor3D(Rand(32, seed + 11), 2, 4, 4, tangent: false)),
                    FwdAttentionArg(argName, 1, tracked, FwdTensor3D(Rand(32, seed + 12), 2, 4, 4, tangent: false)),
                    FwdAttentionArg(argName, 2, tracked, FwdTensor3D(Rand(32, seed + 13), 2, 4, 4, tangent: false)),
                    numHeads: 2, scale: 0.5f));
        }

        yield return new ForwardRow("MultiHeadAttention", "mask",
            () => FwdMat2D(Rand(16, seed + 4), 4, 4, tangent: true),
            tracked => ForwardGradOperations.MultiHeadAttention(
                FwdMat2D(Rand(16, seed + 5), 4, 4, tangent: false),
                FwdMat2D(Rand(16, seed + 6), 4, 4, tangent: false),
                FwdMat2D(Rand(16, seed + 7), 4, 4, tangent: false),
                numHeads: 2, scale: 0.5f, mask: tracked));

        yield return new ForwardRow("BatchedMultiHeadAttention", "mask",
            () => FwdTensor3D(Rand(32, seed + 14), 2, 4, 4, tangent: true),
            tracked => ForwardGradOperations.BatchedMultiHeadAttention(
                FwdTensor3D(Rand(32, seed + 15), 2, 4, 4, tangent: false),
                FwdTensor3D(Rand(32, seed + 16), 2, 4, 4, tangent: false),
                FwdTensor3D(Rand(32, seed + 17), 2, 4, 4, tangent: false),
                numHeads: 2, scale: 0.5f, mask: tracked));
    }

    static ForwardGradTensor<float> FwdAttentionArg(
        string argName, int position, ForwardGradTensor<float> tracked, ForwardGradTensor<float> other)
        => argName == new[] { "query", "key", "value" }[position] ? tracked : other;

    // ---------------------------------------------------------------------
    // Module rows
    // ---------------------------------------------------------------------

    static IEnumerable<ModuleRow> ModuleRows()
    {
        yield return new ModuleRow("Conv1d", () =>
        {
            var module = new Conv1d<float>(inChannels: 2, outChannels: 3, kernelSize: 2);
            var input = Tensor3D(Rand(12, 3001), 2, 2, 3, requiresGrad: true);
            var output = module.Forward(input);
            return new ModuleRowContext(input, output, [module.Weight!.Tensor, module.Bias!.Tensor]);
        });

        yield return new ModuleRow("Conv2d", () =>
        {
            var module = new Conv2d<float>(2, 3, 2, 1, 0, 0, 0, 0, bias: true);
            var input = Tensor4D(Rand(8, 3011), 1, 2, 2, 2, requiresGrad: true);
            var output = module.Forward(input);
            return new ModuleRowContext(input, output, [module.Weight!.Tensor, module.Bias!.Tensor]);
        });

        yield return new ModuleRow("ConvTranspose2d", () =>
        {
            var module = new ConvTranspose2d<float>(inChannels: 2, outChannels: 3, kernelSize: 2);
            var input = Tensor4D(Rand(8, 3021), 1, 2, 2, 2, requiresGrad: true);
            var output = module.Forward(input);
            return new ModuleRowContext(input, output, [module.Weight!.Tensor, module.Bias!.Tensor]);
        });

        yield return new ModuleRow("LayerNorm", () =>
        {
            var module = new LayerNorm<float>(normalizedShape: 4);
            var input = Mat2D(Rand(8, 3031), 2, 4, requiresGrad: true);
            var output = module.Forward(input);
            return new ModuleRowContext(input, output, [module.Weight!.Tensor, module.Bias!.Tensor]);
        });

        yield return new ModuleRow("RMSNorm", () =>
        {
            var module = new RMSNorm<float>(normalizedShape: 4);
            var input = Mat2D(Rand(8, 3041), 2, 4, requiresGrad: true);
            var output = module.Forward(input);
            return new ModuleRowContext(input, output, [module.Weight!.Tensor]);
        });

        yield return new ModuleRow("BatchNorm1d", () =>
        {
            var module = new BatchNorm1d<float>(numFeatures: 2, trackRunningStats: false);
            var input = Tensor3D(Rand(8, 3051), 2, 2, 2, requiresGrad: true);
            var output = module.Forward(input);
            return new ModuleRowContext(input, output, [module.Weight!.Tensor, module.Bias!.Tensor]);
        });

        yield return new ModuleRow("BatchNorm2d", () =>
        {
            var module = new BatchNorm2d<float>(numFeatures: 2, trackRunningStats: false);
            var input = Tensor4D(Rand(4, 3061), 1, 2, 2, 1, requiresGrad: true);
            var output = module.Forward(input);
            return new ModuleRowContext(input, output, [module.Weight!.Tensor, module.Bias!.Tensor]);
        });

        // These own parameters but build no node of their own, so each parameter is reached
        // through a delegated op. The reachability assertion is what pins that.
        yield return new ModuleRow("Linear", () =>
        {
            var module = new Linear<float>(inFeatures: 4, outFeatures: 3, bias: true);
            var input = Mat2D(Rand(8, 3071), 2, 4, requiresGrad: true);
            var output = module.Forward(input);
            return new ModuleRowContext(input, output, [module.Weight!.Tensor, module.Bias!.Tensor]);
        });

        // Embedding and SparseEmbedding consume their input as integer selectors via
        // int.CreateChecked, so the input is a non-differentiable constant and correctly
        // never enters the graph. Requires-grad on it is a caller error, not a contract
        // breach by the module, which is why these rows assert parameter reachability only.
        yield return new ModuleRow("Embedding", () =>
        {
            var module = new Embedding<float>(numEmbeddings: 4, embeddingDim: 3);
            var input = ReverseGradTensor<float>.FromArray([0f, 1f, 2f, 3f], requiresGrad: false);
            var output = module.Forward(input);
            return new ModuleRowContext(input, output, [module.Weight!.Tensor], InputIsGraphInput: false);
        });

        yield return new ModuleRow("SparseEmbedding", () =>
        {
            var module = new SparseEmbedding<float>(numEmbeddings: 4, embeddingDim: 2);
            var input = Indices2D(Vec([0f, 1f, 1f, 0f], requiresGrad: false));
            var output = module.Forward(input);
            return new ModuleRowContext(input, output, [module.Weight!.Tensor], InputIsGraphInput: false);
        });

        // VAE's beta is registered with requiresGrad: false and consumed as a scalar
        // multiplier on the KL term, so it is only reachable through ElboLoss, never
        // through Forward. That is consistent with the contract: beta's backward IS
        // Multiply's backward, and Multiply already lists every tensor argument.
        yield return new ModuleRow("VAE", () =>
        {
            var module = new VAE<float>(inputDim: 3, latentDim: 2, hiddenDim: 4);
            var input = Mat2D(Rand(3, 3081), 1, 3, requiresGrad: true);
            var (mu, logVar) = module.Encode(input);
            var recon = Vec(Rand(3, 3082), requiresGrad: false);
            var original = Vec(Rand(3, 3083), requiresGrad: false);
            var loss = module.ElboLoss(recon, original, mu, logVar, ElboLossType.KldBeta);
            return new ModuleRowContext(input, loss, [BetaOf(module)]);
        });
    }

    static ReverseGradTensor<float> BetaOf(VAE<float> vae)
        => vae.GetParameters().Single(p => p.Key.EndsWith("Beta", StringComparison.Ordinal)).Value.Tensor;

    // ---------------------------------------------------------------------
    // Reflection completeness helpers
    // ---------------------------------------------------------------------

    static void AssertEveryMultiTensorOpCovered(Type openTensorType, IEnumerable<string> covered)
    {
        var ops = typeof(ReverseGradOperations)
            .GetMethods(BindingFlags.Public | BindingFlags.Static)
            .Where(m => m.IsGenericMethodDefinition)
            .Where(m => TensorArgCount(m, openTensorType) >= 2)
            .Select(m => m.Name)
            .ToHashSet(StringComparer.Ordinal);

        // Concat takes a single argument that is an array of tensors, so a plain
        // "two or more" count would miss it even though a caller can pass a mixed array.
        ops.Add("Concat");

        var coveredOps = covered.ToHashSet(StringComparer.Ordinal);
        var uncovered = ops
            .Where(o => !coveredOps.Contains(o))
            .OrderBy(o => o, StringComparer.Ordinal)
            .ToList();

        Assert.That(uncovered, Is.Empty,
            "These public static ops take two or more tensor arguments (or one array of tensors), so an " +
            "argument can be silently dropped, but no contract row covers them. Add rows for them.");
    }

    static int TensorArgCount(MethodInfo method, Type openTensorType)
    {
        var count = 0;
        foreach (var parameter in method.GetParameters())
        {
            var type = parameter.ParameterType;
            if (type.IsArray)
                type = type.GetElementType()!;
            if (type.IsGenericType && type.GetGenericTypeDefinition() == openTensorType)
                count++;
        }
        return count;
    }

    static IEnumerable<Type> ParameterOwningModules()
    {
        var moduleDefinition = typeof(Module<>);
        var parameterDefinition = typeof(Parameter<>);

        return typeof(Conv1d<>).Assembly
            .GetExportedTypes()
            .Where(t => t.IsClass && !t.IsAbstract)
            .Where(t => DerivesFromModule(t, moduleDefinition))
            .Where(t => OwnsParameterField(t, parameterDefinition))
            .OrderBy(t => t.Name, StringComparer.Ordinal);
    }

    static string ModuleRowNameOf(Type moduleType)
    {
        var name = moduleType.Name;
        var arity = name.IndexOf('`');
        return arity < 0 ? name : name[..arity];
    }

    static bool DerivesFromModule(Type type, Type moduleDefinition)
    {
        for (var current = type.BaseType; current != null; current = current.BaseType)
            if (current.IsGenericType && current.GetGenericTypeDefinition() == moduleDefinition)
                return true;
        return false;
    }

    static bool OwnsParameterField(Type type, Type parameterDefinition)
    {
        for (var current = type; current != null && current != typeof(object); current = current.BaseType)
            foreach (var field in current.GetFields(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public))
                if (field.FieldType.IsGenericType
                    && field.FieldType.GetGenericTypeDefinition() == parameterDefinition)
                    return true;
        return false;
    }

    // ---------------------------------------------------------------------
    // Tensor helpers
    // ---------------------------------------------------------------------

    static HashSet<ReverseGradTensor<float>> ReachableTensors(ReverseGradTensor<float> root)
    {
        var seen = new HashSet<ReverseGradTensor<float>>();
        var pending = new Queue<ReverseGradTensor<float>>();
        pending.Enqueue(root);

        while (pending.Count > 0)
        {
            var tensor = pending.Dequeue();
            if (!seen.Add(tensor) || tensor.GradFn == null)
                continue;

            foreach (var input in tensor.GradFn.Inputs)
                pending.Enqueue(input);
        }

        return seen;
    }

    static float[] Rand(int count, int seed)
    {
        var rng = new Random(seed);
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2 - 1);
        return arr;
    }

    static NivaraColumn<float> Ones(int count)
    {
        var arr = new float[count];
        Array.Fill(arr, 1f);
        return NivaraColumn<float>.Create(arr);
    }

    static ReverseGradTensor<float> Vec(float[] data, bool requiresGrad)
        => ReverseGradTensor<float>.FromArray(data, requiresGrad);

    static ReverseGradTensor<float> Mat2D(float[] data, int rows, int cols, bool requiresGrad)
        => ReverseGradTensor<float>.FromMatrix(data, rows, cols, requiresGrad);

    static ReverseGradTensor<float> Tensor3D(float[] data, int b, int l, int d, bool requiresGrad)
        => Reshape(new ReverseGradTensor<float>(NivaraColumn<float>.Create(data), requiresGrad), b, l, d);

    static ReverseGradTensor<float> Tensor4D(float[] data, int b, int c, int h, int w, bool requiresGrad)
        => Reshape(new ReverseGradTensor<float>(NivaraColumn<float>.Create(data), requiresGrad), b, c, h, w);

    static ReverseGradTensor<float> Reshape(ReverseGradTensor<float> tensor, params int[] shape)
    {
        tensor.Reshape(shape);
        return tensor;
    }

    static ReverseGradTensor<float> Indices2D(ReverseGradTensor<float> flat)
        => Reshape(flat, 2, 2);

    static ForwardGradTensor<float> FwdVec(float[] data, bool tangent)
        => ForwardGradTensor<float>.FromArray(data, tangent ? TangentOf(data.Length) : null);

    static ForwardGradTensor<float> FwdMat2D(float[] data, int rows, int cols, bool tangent)
        => ForwardGradTensor<float>.FromMatrix(data, rows, cols, tangent ? TangentOf(data.Length) : null);

    static ForwardGradTensor<float> FwdTensor3D(float[] data, int b, int l, int d, bool tangent)
    {
        var tensor = new ForwardGradTensor<float>(
            NivaraColumn<float>.Create(data),
            tangent ? NivaraColumn<float>.Create(TangentOf(data.Length)) : null);
        tensor.Reshape(b, l, d);
        return tensor;
    }

    static ForwardGradTensor<float> FwdIndices2D(ForwardGradTensor<float> flat)
    {
        flat.Reshape(2, 2);
        return flat;
    }

    static float[] TangentOf(int count)
    {
        var arr = new float[count];
        Array.Fill(arr, 1f);
        return arr;
    }
}
