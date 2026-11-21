using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class LoRALinearTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    static ReverseGradTensor<float> MakeInput(int batch, int inFeatures)
    {
        var data = new float[batch * inFeatures];
        for (int i = 0; i < data.Length; i++) data[i] = (i + 1) * 0.1f;
        var t = new ReverseGradTensor<float>(NivaraColumn<float>.Create(data), requiresGrad: false);
        t.Reshape(batch, inFeatures);
        return t;
    }

    [Test]
    public void LoRALinear_Step0_AdaptedEqualsBase_Exact()
    {
        using var lora = new LoRALinear<float>(4, 3, r: 2);
        var x = MakeInput(2, 4);

        var baseOut = lora.Base.Forward(x);
        var adaptedOut = lora.Forward(x);

        Assert.That(adaptedOut.Length, Is.EqualTo(baseOut.Length));
        for (int i = 0; i < baseOut.Length; i++)
            Assert.That(adaptedOut[i], Is.EqualTo(baseOut[i]));
    }

    [Test]
    public void LoRALinear_FrozenBase_HasNoGradients_AfterBackward()
    {
        using var lora = new LoRALinear<float>(4, 3, r: 2);
        var x = MakeInput(2, 4);

        var y = lora.Forward(x);
        var grad = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(Enumerable.Repeat(1f, y.Length).ToArray()), requiresGrad: false);
        grad.Reshape(y.Shape[0], y.Shape[1]);
        y.Backward(grad);

        foreach (var (name, param) in lora.Base.GetParameters())
            Assert.That(param.Tensor.Grad, Is.Null,
                $"Frozen base parameter '{name}' should have no gradient");

        Assert.That(lora.A.Tensor.Grad, Is.Not.Null);
        Assert.That(lora.B.Tensor.Grad, Is.Not.Null);
    }

    [Test]
    public void LoRALinear_AdapterGradients_MatchFiniteDifferences()
    {
        const int inF = 3, outF = 2, r = 2, batch = 2;
        using var lora = new LoRALinear<float>(inF, outF, r, alpha: 4f);
        var x = MakeInput(batch, inF);

        // Autodiff gradients
        float[] gradA, gradB;
        using (var gs = GradientUtils.Grad())
        {
            var y = lora.Forward(x);
            var ones = new ReverseGradTensor<float>(
                NivaraColumn<float>.Create(Enumerable.Repeat(1f, y.Length).ToArray()), requiresGrad: false);
            ones.Reshape(y.Shape[0], y.Shape[1]);
            y.Backward(ones);

            gradA = new float[lora.A.Tensor.Length];
            for (int i = 0; i < gradA.Length; i++) gradA[i] = lora.A.Tensor.Grad![i];
            gradB = new float[lora.B.Tensor.Length];
            for (int i = 0; i < gradB.Length; i++) gradB[i] = lora.B.Tensor.Grad![i];
        }

        float Loss()
        {
            var y = lora.Forward(x);
            float s = 0f;
            for (int i = 0; i < y.Length; i++) s += y[i];
            return s;
        }

        void SetParam(Parameter<float> p, int idx, float val)
        {
            var t = p.Tensor;
            var data = new float[t.Length];
            for (int i = 0; i < data.Length; i++) data[i] = t[i];
            data[idx] = val;
            var col = NivaraColumn<float>.Create(data);
            var ten = new ReverseGradTensor<float>(col, requiresGrad: true);
            ten.Reshape(t.Shape[0], t.Shape[1]);
            p.Tensor = ten;
        }

        const float eps = 1e-3f;

        // A
        for (int idx = 0; idx < lora.A.Tensor.Length; idx++)
        {
            var t = lora.A.Tensor;
            float orig = t[idx];
            SetParam(lora.A, idx, orig + eps);
            float lp = Loss();
            SetParam(lora.A, idx, orig - eps);
            float lm = Loss();
            SetParam(lora.A, idx, orig);
            float fd = (lp - lm) / (2f * eps);
            Assert.That(gradA[idx], Is.EqualTo(fd).Within(1e-2f),
                $"A grad mismatch at {idx}: autodiff={gradA[idx]} fd={fd}");
        }

        // B
        for (int idx = 0; idx < lora.B.Tensor.Length; idx++)
        {
            var t = lora.B.Tensor;
            float orig = t[idx];
            SetParam(lora.B, idx, orig + eps);
            float lp = Loss();
            SetParam(lora.B, idx, orig - eps);
            float lm = Loss();
            SetParam(lora.B, idx, orig);
            float fd = (lp - lm) / (2f * eps);
            Assert.That(gradB[idx], Is.EqualTo(fd).Within(1e-2f),
                $"B grad mismatch at {idx}: autodiff={gradB[idx]} fd={fd}");
        }
    }

    [Test]
    public void LoRALinear_Merge_EquivalentToUnmerged()
    {
        using var lora = new LoRALinear<float>(4, 3, r: 2, alpha: 4f);
        var x = MakeInput(2, 4);

        var before = new float[lora.Forward(x).Length];
        var outBefore = lora.Forward(x);
        for (int i = 0; i < before.Length; i++) before[i] = outBefore[i];

        lora.Merge();

        var after = lora.Forward(x);
        for (int i = 0; i < before.Length; i++)
            Assert.That(after[i], Is.EqualTo(before[i]).Within(1e-5f));
    }
}
