using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Cross-validation tests comparing forward-mode JVP (ForwardGradOperations)
/// with reverse-mode gradients (GradOperations + Backward).
///
/// For element-wise ops seeded with tangent on one input:
///   Forward JVP result = ∂f/∂input  (element-wise)
///   Backward gradient (via Sum backward) = ∂Sum(f)/∂input  (element-wise)
///   These are identical because ∂Sum/∂x = 1 for all elements.
/// </summary>
[TestFixture]
public class ForwardParityTests
{
    [Test]
    public void Add_ForwardTangent_EqualsBackwardGradient()
    {
        // Backward: y = a + b, sum(y).Backward() → grad_a = [1, 1]
        // Forward:  t_a = [1, 1], t_b = none  → JVP = t_a = [1, 1]
        var aData = NivaraColumn<float>.Create(new float[] { 1f, 2f });
        var bData = NivaraColumn<float>.Create(new float[] { 3f, 4f });

        var ra = new ReverseGradTensor<float>(aData, requiresGrad: true);
        var rb = new ReverseGradTensor<float>(bData, requiresGrad: false);
        ReverseGradOperations.Sum(ReverseGradOperations.Add(ra, rb)).Backward();
        var expected = ra.Grad!;

        var fa = new ForwardGradTensor<float>(aData, NivaraColumn<float>.Create(new float[] { 1f, 1f }));
        var fb = new ForwardGradTensor<float>(bData);
        var result = ForwardGradOperations.Add(fa, fb);

        Assert.That(result.RequiresTangent, Is.True);
        for (int i = 0; i < expected.Length; i++)
            Assert.That(result.Tangent![i], Is.EqualTo(expected[i]).Within(1e-6f));
    }

    [Test]
    public void Subtract_ForwardTangent_EqualsBackwardGradient()
    {
        var aData = NivaraColumn<float>.Create(new float[] { 10f, 8f });
        var bData = NivaraColumn<float>.Create(new float[] { 3f, 2f });

        var ra = new ReverseGradTensor<float>(aData, requiresGrad: true);
        var rb = new ReverseGradTensor<float>(bData, requiresGrad: false);
        ReverseGradOperations.Sum(ReverseGradOperations.Subtract(ra, rb)).Backward();
        var expected = ra.Grad!;

        var fa = new ForwardGradTensor<float>(aData, NivaraColumn<float>.Create(new float[] { 1f, 1f }));
        var fb = new ForwardGradTensor<float>(bData);
        var result = ForwardGradOperations.Subtract(fa, fb);

        for (int i = 0; i < expected.Length; i++)
            Assert.That(result.Tangent![i], Is.EqualTo(expected[i]).Within(1e-6f));
    }

    [Test]
    public void Multiply_ForwardTangent_EqualsBackwardGradient()
    {
        var aData = NivaraColumn<float>.Create(new float[] { 2f, 3f });
        var bData = NivaraColumn<float>.Create(new float[] { 4f, 5f });

        var ra = new ReverseGradTensor<float>(aData, requiresGrad: true);
        var rb = new ReverseGradTensor<float>(bData, requiresGrad: false);
        ReverseGradOperations.Sum(ReverseGradOperations.Multiply(ra, rb)).Backward();
        var expected = ra.Grad!;

        var fa = new ForwardGradTensor<float>(aData, NivaraColumn<float>.Create(new float[] { 1f, 1f }));
        var fb = new ForwardGradTensor<float>(bData);
        var result = ForwardGradOperations.Multiply(fa, fb);

        for (int i = 0; i < expected.Length; i++)
            Assert.That(result.Tangent![i], Is.EqualTo(expected[i]).Within(1e-6f));
    }

    [Test]
    public void Divide_ForwardTangent_EqualsBackwardGradient()
    {
        var aData = NivaraColumn<float>.Create(new float[] { 12f, 15f });
        var bData = NivaraColumn<float>.Create(new float[] { 3f, 5f });

        var ra = new ReverseGradTensor<float>(aData, requiresGrad: true);
        var rb = new ReverseGradTensor<float>(bData, requiresGrad: false);
        ReverseGradOperations.Sum(ReverseGradOperations.Divide(ra, rb)).Backward();
        var expected = ra.Grad!;

        var fa = new ForwardGradTensor<float>(aData, NivaraColumn<float>.Create(new float[] { 1f, 1f }));
        var fb = new ForwardGradTensor<float>(bData);
        var result = ForwardGradOperations.Divide(fa, fb);

        for (int i = 0; i < expected.Length; i++)
            Assert.That(result.Tangent![i], Is.EqualTo(expected[i]).Within(1e-6f));
    }

    [Test]
    public void Relu_ForwardTangent_EqualsBackwardGradient()
    {
        var xData = NivaraColumn<float>.Create(new float[] { -1f, 0f, 1f, 2f });

        var rx = new ReverseGradTensor<float>(xData, requiresGrad: true);
        ReverseGradOperations.Sum(ReverseGradOperations.Relu(rx)).Backward();
        var expected = rx.Grad!;

        var fx = new ForwardGradTensor<float>(xData, NivaraColumn<float>.Create(new float[] { 1f, 1f, 1f, 1f }));
        var result = ForwardGradOperations.Relu(fx);

        for (int i = 0; i < expected.Length; i++)
            Assert.That(result.Tangent![i], Is.EqualTo(expected[i]).Within(1e-6f));
    }

    [Test]
    public void Sigmoid_ForwardTangent_EqualsBackwardGradient()
    {
        var xData = NivaraColumn<float>.Create(new float[] { -1f, 0f, 1f });

        var rx = new ReverseGradTensor<float>(xData, requiresGrad: true);
        ReverseGradOperations.Sum(ReverseGradOperations.Sigmoid(rx)).Backward();
        var expected = rx.Grad!;

        var fx = new ForwardGradTensor<float>(xData, NivaraColumn<float>.Create(new float[] { 1f, 1f, 1f }));
        var result = ForwardGradOperations.Sigmoid(fx);

        for (int i = 0; i < expected.Length; i++)
            Assert.That(result.Tangent![i], Is.EqualTo(expected[i]).Within(1e-6f));
    }

    [Test]
    public void Tanh_ForwardTangent_EqualsBackwardGradient()
    {
        var xData = NivaraColumn<float>.Create(new float[] { -1f, 0f, 1f });

        var rx = new ReverseGradTensor<float>(xData, requiresGrad: true);
        ReverseGradOperations.Sum(ReverseGradOperations.Tanh(rx)).Backward();
        var expected = rx.Grad!;

        var fx = new ForwardGradTensor<float>(xData, NivaraColumn<float>.Create(new float[] { 1f, 1f, 1f }));
        var result = ForwardGradOperations.Tanh(fx);

        for (int i = 0; i < expected.Length; i++)
            Assert.That(result.Tangent![i], Is.EqualTo(expected[i]).Within(1e-6f));
    }

    [Test]
    public void Negate_ForwardTangent_EqualsBackwardGradient()
    {
        var xData = NivaraColumn<float>.Create(new float[] { 1f, -2f, 3f });

        var rx = new ReverseGradTensor<float>(xData, requiresGrad: true);
        ReverseGradOperations.Sum(ReverseGradOperations.Negate(rx)).Backward();
        var expected = rx.Grad!;

        var fx = new ForwardGradTensor<float>(xData, NivaraColumn<float>.Create(new float[] { 1f, 1f, 1f }));
        var result = ForwardGradOperations.Negate(fx);

        for (int i = 0; i < expected.Length; i++)
            Assert.That(result.Tangent![i], Is.EqualTo(expected[i]).Within(1e-6f));
    }

    [Test]
    public void Abs_ForwardTangent_EqualsBackwardGradient()
    {
        var xData = NivaraColumn<float>.Create(new float[] { -2f, 0f, 3f });

        var rx = new ReverseGradTensor<float>(xData, requiresGrad: true);
        ReverseGradOperations.Sum(ReverseGradOperations.Abs(rx)).Backward();
        var expected = rx.Grad!;

        var fx = new ForwardGradTensor<float>(xData, NivaraColumn<float>.Create(new float[] { 1f, 1f, 1f }));
        var result = ForwardGradOperations.Abs(fx);

        for (int i = 0; i < expected.Length; i++)
            Assert.That(result.Tangent![i], Is.EqualTo(expected[i]).Within(1e-6f));
    }

    [Test]
    public void Exp_ForwardTangent_EqualsBackwardGradient()
    {
        var xData = NivaraColumn<float>.Create(new float[] { 0f, 1f });

        var rx = new ReverseGradTensor<float>(xData, requiresGrad: true);
        ReverseGradOperations.Sum(ReverseGradOperations.Exp(rx)).Backward();
        var expected = rx.Grad!;

        var fx = new ForwardGradTensor<float>(xData, NivaraColumn<float>.Create(new float[] { 1f, 1f }));
        var result = ForwardGradOperations.Exp(fx);

        for (int i = 0; i < expected.Length; i++)
            Assert.That(result.Tangent![i], Is.EqualTo(expected[i]).Within(1e-6f));
    }

    [Test]
    public void Log_ForwardTangent_EqualsBackwardGradient()
    {
        var xData = NivaraColumn<float>.Create(new float[] { 1f, 2f });

        var rx = new ReverseGradTensor<float>(xData, requiresGrad: true);
        ReverseGradOperations.Sum(ReverseGradOperations.Log(rx)).Backward();
        var expected = rx.Grad!;

        var fx = new ForwardGradTensor<float>(xData, NivaraColumn<float>.Create(new float[] { 1f, 1f }));
        var result = ForwardGradOperations.Log(fx);

        for (int i = 0; i < expected.Length; i++)
            Assert.That(result.Tangent![i], Is.EqualTo(expected[i]).Within(1e-6f));
    }

    [Test]
    public void Sum_ForwardTangent_MatchesBackwardGradientMagnitude()
    {
        // Forward Sum(x) with t_x=[1,...,1]: JVP = sum(t_x) = n
        // Backward Sum(Sum(x)): grad_x = [1,...,1], sum(grad_x) = n
        var xData = NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f, 4f });

        var rx = new ReverseGradTensor<float>(xData, requiresGrad: true);
        ReverseGradOperations.Sum(ReverseGradOperations.Sum(rx)).Backward();
        var backSum = new NivaraSeries<float>(rx.Grad!).Sum();

        var fx = new ForwardGradTensor<float>(xData, NivaraColumn<float>.Create(new float[] { 1f, 1f, 1f, 1f }));
        var fsum = ForwardGradOperations.Sum(fx);

        Assert.That(fsum.Tangent![0], Is.EqualTo(backSum).Within(1e-6f));
    }

    [Test]
    public void Mean_ForwardTangent_MatchesBackwardGradientMagnitude()
    {
        var xData = NivaraColumn<float>.Create(new float[] { 2f, 4f, 6f, 8f });

        var rx = new ReverseGradTensor<float>(xData, requiresGrad: true);
        ReverseGradOperations.Sum(ReverseGradOperations.Mean(rx)).Backward();
        var backSum = new NivaraSeries<float>(rx.Grad!).Sum();

        var fx = new ForwardGradTensor<float>(xData, NivaraColumn<float>.Create(new float[] { 1f, 1f, 1f, 1f }));
        var fmean = ForwardGradOperations.Mean(fx);

        // Forward: JVP = sum(t_x) / n  = 4 / 4 = 1
        // Backward: grad_x = [0.25, 0.25, 0.25, 0.25], sum = 1
        Assert.That(fmean.Tangent![0], Is.EqualTo(backSum).Within(1e-6f));
    }

    [Test]
    public void ChainedOps_ForwardTangent_MatchesBackwardGradient()
    {
        // y = relu(w * x + b), sum(y).Backward()
        // Forward: seed tangent only on w (t_w=[1,1,1]), no tangent on b or x
        // JVP at y = relu'(z) * (x * t_w)  where z = w*x + b
        // Backward grad_w = relu'(z) * x (via chain rule of d(sum)/dw)
        // These should match element-wise.
        var xData = NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f });
        var wData = NivaraColumn<float>.Create(new float[] { 0.5f, 0.5f, 0.5f });
        var bData = NivaraColumn<float>.Create(new float[] { -1f, 0f, 1f });

        // Backward path: sum(relu(wx + b)).Backward() → grad_w
        var rx = new ReverseGradTensor<float>(xData, requiresGrad: false);
        var rw = new ReverseGradTensor<float>(wData, requiresGrad: true);
        var rb = new ReverseGradTensor<float>(bData, requiresGrad: false);
        var ry = ReverseGradOperations.Relu(ReverseGradOperations.Add(ReverseGradOperations.Multiply(rx, rw), rb));
        ReverseGradOperations.Sum(ry).Backward();
        var expectedW = rw.Grad!;

        // Forward path: seed tangent only on w
        var fw = new ForwardGradTensor<float>(wData, NivaraColumn<float>.Create(new float[] { 1f, 1f, 1f }));
        var fx = new ForwardGradTensor<float>(xData);
        var fb = new ForwardGradTensor<float>(bData);
        var fresult = ForwardGradOperations.Relu(ForwardGradOperations.Add(ForwardGradOperations.Multiply(fx, fw), fb));

        Assert.That(fresult.RequiresTangent, Is.True);
        for (int i = 0; i < expectedW.Length; i++)
            Assert.That(fresult.Tangent![i], Is.EqualTo(expectedW[i]).Within(1e-6f));
    }

    [Test]
    public void MatMul_ForwardTangent_MatchesBackwardGradient()
    {
        // f(A,B) = Sum(A @ B).  Use symmetric B so B = B^T.
        // Forward JVP (seed t_A = ones):  t_A @ B
        // Backward grad_A (via Sum backward):  ones(2,2) @ B^T = ones(2,2) @ B
        // With t_A = ones, these are equal element-wise.
        var aData = NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f, 4f });
        var bData = NivaraColumn<float>.Create(new float[] { 5f, 6f, 6f, 8f }); // symmetric B

        // Backward
        var ra = new ReverseGradTensor<float>(aData, requiresGrad: true);
        var rb = new ReverseGradTensor<float>(bData, requiresGrad: false);
        ra.Reshape(2, 2);
        rb.Reshape(2, 2);
        ReverseGradOperations.Sum(ReverseGradOperations.MatMul(ra, rb)).Backward();

        // Forward: tangent on A = ones(2x2), no tangent on B
        var fa = new ForwardGradTensor<float>(aData, NivaraColumn<float>.Create(new float[] { 1f, 1f, 1f, 1f }));
        var fb = new ForwardGradTensor<float>(bData);
        fa.Reshape(2, 2);
        fb.Reshape(2, 2);
        var fresult = ForwardGradOperations.MatMul(fa, fb);

        Assert.That(fresult.Tangent, Is.Not.Null);
        Assert.That(ra.Grad, Is.Not.Null);
        for (int i = 0; i < 4; i++)
        {
            Assert.That(fresult.Tangent![i], Is.EqualTo(ra.Grad![i]).Within(1e-5f),
                $"Mismatch at position {i}: forward={fresult.Tangent[i]}, backward={ra.Grad[i]}");
        }
    }

    [Test]
    public void Transpose_ForwardTangent_MatchesBackwardGradient()
    {
        var xData = NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f, 4f, 5f, 6f });

        var rx = new ReverseGradTensor<float>(xData, requiresGrad: true);
        rx.Reshape(2, 3);
        ReverseGradOperations.Sum(ReverseGradOperations.Transpose(rx)).Backward();
        var expected = rx.Grad!;

        var fx = new ForwardGradTensor<float>(xData, NivaraColumn<float>.Create(new float[] { 1f, 1f, 1f, 1f, 1f, 1f }));
        fx.Reshape(2, 3);
        var fresult = ForwardGradOperations.Transpose(fx);

        // Forward tangent = transpose(t_x) = all ones (shape 3x2 flattened)
        // Backward grad_x (via Sum(Transpose))) = all ones (shape 2x3 flattened)
        // Not directly comparable because shapes differ after transpose.
        // Instead verify both are non-null and finite.
        Assert.That(fresult.Tangent, Is.Not.Null);
        Assert.That(expected, Is.Not.Null);
        Assert.That(fresult.Tangent!.Length, Is.EqualTo(6));
        Assert.That(expected.Length, Is.EqualTo(6));
        // Forward and backward tangents/gradients should have consistent sum
        Assert.That(fresult.Tangent.Sum(), Is.EqualTo(new NivaraSeries<float>(expected).Sum()).Within(1e-6f));
    }

    [Test]
    public void DoubleType_ForwardTangent_MatchesBackwardGradient()
    {
        var xData = NivaraColumn<double>.Create(new double[] { -2.0, 0.0, 3.0 });

        var rx = new ReverseGradTensor<double>(xData, requiresGrad: true);
        ReverseGradOperations.Sum(ReverseGradOperations.Abs(rx)).Backward();
        var expected = rx.Grad!;

        var fx = new ForwardGradTensor<double>(xData, NivaraColumn<double>.Create(new double[] { 1.0, 1.0, 1.0 }));
        var result = ForwardGradOperations.Abs(fx);

        for (int i = 0; i < expected.Length; i++)
            Assert.That(result.Tangent![i], Is.EqualTo(expected[i]).Within(1e-12));
    }
}
