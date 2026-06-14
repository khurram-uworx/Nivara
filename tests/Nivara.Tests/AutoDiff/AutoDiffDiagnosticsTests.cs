using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Optimizer;
using Nivara.Diagnostics;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class AutoDiffDiagnosticsTests
{
    [SetUp]
    public void SetUp()
    {
        DiagnosticsTracker.IsEnabled = true;
        DiagnosticsTracker.ClearRecordedOperations();
    }

    [TearDown]
    public void TearDown()
    {
        DiagnosticsTracker.IsEnabled = false;
        DiagnosticsTracker.ClearRecordedOperations();
    }

    [Test]
    public void Backward_RepeatedCalls_RecordAllocationDiagnostics()
    {
        for (int i = 0; i < 2; i++)
        {
            var x = new ReverseGradTensor<float>(
                NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f }),
                requiresGrad: true);
            var y = ReverseGradOperations.Sum(ReverseGradOperations.Multiply(x, x));

            y.Backward();
        }

        var backwardOps = DiagnosticsTracker.GetRecordedOperations()
            .Where(op => op.OperationType == "AutoDiffBackward")
            .ToArray();

        Assert.That(backwardOps, Has.Length.EqualTo(2));
        Assert.That(backwardOps.All(op => op.AllocatedBytes >= 0), Is.True);
        Assert.That(backwardOps.All(op => op.Elapsed >= TimeSpan.Zero), Is.True);
        Assert.That(backwardOps.All(op => op.Notes?.Contains("AutoDiff=Backward") == true), Is.True);
    }

    [Test]
    public void ActivationGradient_WithNulls_RecordsNullAwareDiagnostics()
    {
        var x = new ReverseGradTensor<float>(
            NivaraColumn<float>.CreateFromNullable(new float?[] { -1f, null, 1f }),
            requiresGrad: true);

        var sigmoid = ReverseGradOperations.Sigmoid(x);
        var grad = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 1f, 1f }),
            requiresGrad: false);

        sigmoid.Backward(grad, stripGradientNulls: false);

        var operations = DiagnosticsTracker.GetRecordedOperations();
        var sigmoidOp = AssertSingleOperation(operations, "AutoDiffSigmoid");
        var backwardOp = AssertSingleOperation(operations, "AutoDiffBackward");

        Assert.That(sigmoidOp.HadNulls, Is.True);
        Assert.That(backwardOp.HadNulls, Is.True);
        Assert.That(sigmoidOp.Notes, Does.Contain("AutoDiff=Sigmoid"));
        Assert.That(backwardOp.Notes, Does.Contain("StripGradientNulls=False"));
        Assert.That(x.Grad, Is.Not.Null);
        Assert.That(x.Grad!.IsNull(1), Is.True);
    }

    [Test]
    public void SgdUpdate_RecordsAllocationDiagnosticsAndPreservesNullSkipSemantics()
    {
        var tensor = new ReverseGradTensor<float>(
            NivaraColumn<float>.CreateFromNullable(new float?[] { 1f, null, 3f }),
            requiresGrad: true);
        tensor.Grad = NivaraColumn<float>.CreateFromNullable(new float?[] { 0.1f, 0.2f, null });

        var updated = SGD<float>.SgdUpdate(tensor, 0.5f, weightDecay: 0.1f);

        var op = AssertSingleOperation(DiagnosticsTracker.GetRecordedOperations(), "AutoDiffSgdUpdate");
        Assert.That(op.HadNulls, Is.True);
        Assert.That(op.AllocatedBytes, Is.GreaterThanOrEqualTo(0));
        Assert.That(op.Elapsed, Is.GreaterThanOrEqualTo(TimeSpan.Zero));
        Assert.That(op.Notes, Does.Contain("WeightDecay=True"));
        Assert.That(updated.IsNull(1), Is.True);
        Assert.That(updated.IsNull(2), Is.True);
    }

    [Test]
    public void MatrixOperations_RecordAutoDiffDiagnostics()
    {
        var a = ReverseGradTensor<float>.FromMatrix(new float[] { 1f, 2f, 3f, 4f }, 2, 2, requiresGrad: true);
        var b = ReverseGradTensor<float>.FromMatrix(new float[] { 5f, 6f, 7f, 8f }, 2, 2, requiresGrad: true);

        var product = ReverseGradOperations.MatMul(a, b);
        var transposed = ReverseGradOperations.Transpose(product);

        var operations = DiagnosticsTracker.GetRecordedOperations();
        var matMulOp = AssertSingleOperation(operations, "AutoDiffMatMul");
        var transposeOp = AssertSingleOperation(operations, "AutoDiffTranspose");

        Assert.That(matMulOp.Notes, Does.Contain("AutoDiff=MatMul"));
        Assert.That(transposeOp.Notes, Does.Contain("AutoDiff=Transpose"));
        Assert.That(matMulOp.InputLength, Is.EqualTo(8));
        Assert.That(transposed.Shape, Is.EqualTo(new[] { 2, 2 }));
    }

    [Test]
    public void AutoDiffDiagnostics_WhenDisabled_DoNotRecordOperations()
    {
        DiagnosticsTracker.IsEnabled = false;
        DiagnosticsTracker.ClearRecordedOperations();

        var x = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1f, 2f }),
            requiresGrad: true);
        var y = ReverseGradOperations.Sum(ReverseGradOperations.Relu(x));

        y.Backward();

        Assert.That(DiagnosticsTracker.GetRecordedOperations(), Is.Empty);
    }

    static OperationDiagnostics AssertSingleOperation(OperationDiagnostics[] operations, string operationType)
    {
        var matches = operations.Where(op => op.OperationType == operationType).ToArray();
        Assert.That(matches, Has.Length.EqualTo(1), $"Expected exactly one {operationType} diagnostic.");
        return matches[0];
    }
}
