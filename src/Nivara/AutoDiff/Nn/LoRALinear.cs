using Nivara.AutoDiff.Nn.Initializers;
using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// A low-rank adapter wrapped around a frozen base <see cref="Linear{T}"/>:
/// <c>y = base(x) + (alpha / r) * ((x @ A) @ B)</c>, with <c>A: [in, r]</c> and
/// <c>B: [r, out]</c>. <c>A</c> is Kaiming-initialized and <c>B</c> is zero, so the
/// adapted output equals the base output exactly at step 0. Only <c>A</c> and <c>B</c>
/// are trainable; the base weights are frozen.
/// </summary>
public sealed class LoRALinear<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly Linear<T> baseLinear;
    readonly int r;
    readonly T alpha;
    readonly T scale;
    Parameter<T> a;
    Parameter<T> b;
    bool merged;

    /// <summary>Gets the frozen base linear layer.</summary>
    public Linear<T> Base => baseLinear;

    /// <summary>Gets the frozen base linear layer (alias).</summary>
    public Linear<T> BaseLinear => baseLinear;

    /// <summary>Gets the adapter matrix <c>A</c> of shape <c>[inFeatures, r]</c>.</summary>
    public Parameter<T> A => a;

    /// <summary>Gets the adapter matrix <c>A</c> (alias).</summary>
    public Parameter<T> LoraA => a;

    /// <summary>Gets the adapter matrix <c>A</c> (alias).</summary>
    public Parameter<T> AdapterA => a;

    /// <summary>Gets the adapter matrix <c>B</c> of shape <c>[r, outFeatures]</c>.</summary>
    public Parameter<T> B => b;

    /// <summary>Gets the adapter matrix <c>B</c> (alias).</summary>
    public Parameter<T> LoraB => b;

    /// <summary>Gets the adapter matrix <c>B</c> (alias).</summary>
    public Parameter<T> AdapterB => b;

    /// <summary>Gets the rank <c>r</c>.</summary>
    public int Rank => r;

    /// <summary>Gets the scaling factor <c>alpha</c>.</summary>
    public T Alpha => alpha;

    /// <summary>Gets whether the adapter has been merged into the base.</summary>
    public bool IsMerged => merged;

    public LoRALinear(int inFeatures, int outFeatures, int r = 8, T? alpha = null, bool useBias = true)
        : this(new Linear<T>(inFeatures, outFeatures, useBias), r, alpha)
    {
    }

    public LoRALinear(Linear<T> baseLinear, int r = 8, T? alpha = null)
    {
        if (baseLinear is null) throw new ArgumentNullException(nameof(baseLinear));
        if (r <= 0) throw new ArgumentOutOfRangeException(nameof(r));

        this.r = r;
        this.alpha = alpha ?? T.CreateChecked(16);
        this.scale = this.alpha / T.CreateChecked(r);
        this.baseLinear = baseLinear;
        baseLinear.Freeze();
        RegisterModules(baseLinear);

        int inFeatures = baseLinear.InFeatures;
        int outFeatures = baseLinear.OutFeatures;

        var aData = new T[inFeatures * r];
        var aTensor = ReverseGradTensor<T>.FromMatrix(aData, inFeatures, r, requiresGrad: true);
        a = new Parameter<T>("A", aTensor);
        KaimingUniformInitializer<T>.Instance.Initialize(a);
        // Ensure requiresGrad survives initialization (initializer preserves it, but be explicit).
        if (!a.Tensor.RequiresGrad)
            a.Tensor = new ReverseGradTensor<T>(a.Tensor.Data, requiresGrad: true, a.Tensor.Shape);
        RegisterParameters(a);

        var bData = new T[r * outFeatures];
        var bTensor = ReverseGradTensor<T>.FromMatrix(bData, r, outFeatures, requiresGrad: true);
        b = new Parameter<T>("B", bTensor);
        RegisterParameters(b);
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        var baseOut = baseLinear.Forward(input);

        if (merged)
            return baseOut;

        var xA = ReverseGradOperations.MatMul(input, a.Tensor);
        var xAB = ReverseGradOperations.MatMul(xA, b.Tensor);

        var scaleArr = new T[xAB.Length];
        for (int i = 0; i < scaleArr.Length; i++)
            scaleArr[i] = scale;
        var scaleTensor = ReverseGradTensor<T>.FromArray(scaleArr, requiresGrad: false);
        var scaled = ReverseGradOperations.Multiply(xAB, scaleTensor);

        return ReverseGradOperations.Add(baseOut, scaled);
    }

    /// <summary>
    /// Folds the adapter into the base weight so inference has no adapter overhead.
    /// After merging, <see cref="Forward"/> returns only the base output.
    /// </summary>
    public void Merge()
    {
        if (merged) return;

        int inFeatures = baseLinear.InFeatures;
        int outFeatures = baseLinear.OutFeatures;

        var weight = baseLinear.Weight!.Tensor;
        var wData = new T[outFeatures * inFeatures];
        weight.Data.CopyTo(wData, T.Zero);

        var aSpan = a.Tensor.AsSpan();
        var bSpan = b.Tensor.AsSpan();

        for (int j = 0; j < outFeatures; j++)
        {
            for (int i = 0; i < inFeatures; i++)
            {
                T sum = T.Zero;
                for (int k = 0; k < r; k++)
                    sum += aSpan[i * r + k] * bSpan[k * outFeatures + j];
                wData[j * inFeatures + i] += scale * sum;
            }
        }

        var newWeight = ReverseGradTensor<T>.FromMatrix(wData, outFeatures, inFeatures, requiresGrad: false);
        baseLinear.BindWeight(newWeight);

        merged = true;
    }
}

/// <summary>
/// A low-rank adapter wrapped around a frozen mixed-dtype base <see cref="Linear{TModel, TWeight}"/>:
/// <c>y = base(x) + (alpha / r) * ((x @ A) @ B)</c>, with <c>A: [in, r]</c> and <c>B: [r, out]</c>. The adapter
/// matrices live in the compute type <typeparamref name="TModel"/> (F32) while the base weight is stored as
/// <typeparamref name="TWeight"/> (for example BF16); the base is frozen and only the F32 adapters train.
/// <c>A</c> is Kaiming-initialized and <c>B</c> is zero, so the adapted output equals the base output
/// exactly at step 0.
/// </summary>
public sealed class LoRALinear<TModel, TWeight> : Module<TModel>
    where TModel : struct, IFloatingPointIeee754<TModel>
    where TWeight : struct, IFloatingPointIeee754<TWeight>
{
    readonly Linear<TModel, TWeight> baseLinear;
    readonly int r;
    readonly TModel alpha;
    readonly TModel scale;
    Parameter<TModel> a;
    Parameter<TModel> b;
    bool merged;

    /// <summary>Gets the frozen base linear layer.</summary>
    public Linear<TModel, TWeight> Base => baseLinear;

    /// <summary>Gets the frozen base linear layer (alias).</summary>
    public Linear<TModel, TWeight> BaseLinear => baseLinear;

    /// <summary>Gets the adapter matrix <c>A</c> of shape <c>[in, r]</c>.</summary>
    public Parameter<TModel> A => a;

    /// <summary>Gets the adapter matrix <c>A</c> (alias).</summary>
    public Parameter<TModel> LoraA => a;

    /// <summary>Gets the adapter matrix <c>B</c> of shape <c>[r, out]</c>.</summary>
    public Parameter<TModel> B => b;

    /// <summary>Gets the adapter matrix <c>B</c> (alias).</summary>
    public Parameter<TModel> LoraB => b;

    /// <summary>Gets the adapter matrix <c>B</c> (alias).</summary>
    public Parameter<TModel> AdapterB => b;

    /// <summary>Gets the adapter matrix <c>A</c> (alias).</summary>
    public Parameter<TModel> AdapterA => a;

    /// <summary>Gets the rank <c>r</c>.</summary>
    public int Rank => r;

    /// <summary>Gets the scaling factor <c>alpha</c>.</summary>
    public TModel Alpha => alpha;

    /// <summary>Gets whether the adapter has been merged into the base.</summary>
    public bool IsMerged => merged;

    public LoRALinear(int inFeatures, int outFeatures, int r = 8, TModel? alpha = null, bool useBias = true)
        : this(new Linear<TModel, TWeight>(inFeatures, outFeatures, useBias), r, alpha)
    {
    }

    public LoRALinear(Linear<TModel, TWeight> baseLinear, int r = 8, TModel? alpha = null)
    {
        if (baseLinear is null) throw new ArgumentNullException(nameof(baseLinear));
        if (r <= 0) throw new ArgumentOutOfRangeException(nameof(r));

        this.r = r;
        this.alpha = alpha ?? TModel.CreateChecked(16);
        this.scale = this.alpha / TModel.CreateChecked(r);
        this.baseLinear = baseLinear;
        baseLinear.Freeze();
        RegisterModules(baseLinear);

        int inFeatures = baseLinear.InFeatures;
        int outFeatures = baseLinear.OutFeatures;

        var aData = new TModel[inFeatures * r];
        var aTensor = ReverseGradTensor<TModel>.FromMatrix(aData, inFeatures, r, requiresGrad: true);
        a = new Parameter<TModel>("A", aTensor);
        KaimingUniformInitializer<TModel>.Instance.Initialize(a);
        if (!a.Tensor.RequiresGrad)
            a.Tensor = new ReverseGradTensor<TModel>(a.Tensor.Data, requiresGrad: true, a.Tensor.Shape);
        RegisterParameters(a);

        var bData = new TModel[r * outFeatures];
        var bTensor = ReverseGradTensor<TModel>.FromMatrix(bData, r, outFeatures, requiresGrad: true);
        b = new Parameter<TModel>("B", bTensor);
        RegisterParameters(b);
    }

    public override ReverseGradTensor<TModel> Forward(ReverseGradTensor<TModel> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        var baseOut = baseLinear.Forward(input);

        if (merged)
            return baseOut;

        var xA = ReverseGradOperations.MatMul(input, a.Tensor);
        var xAB = ReverseGradOperations.MatMul(xA, b.Tensor);

        var scaleArr = new TModel[xAB.Length];
        for (int i = 0; i < scaleArr.Length; i++)
            scaleArr[i] = scale;
        var scaleTensor = ReverseGradTensor<TModel>.FromArray(scaleArr, requiresGrad: false);
        var scaled = ReverseGradOperations.Multiply(xAB, scaleTensor);

        return ReverseGradOperations.Add(baseOut, scaled);
    }

    /// <summary>
    /// Folds the adapter into the base weight so inference has no adapter overhead.
    /// After merging, <see cref="Forward"/> returns only the base output.
    /// </summary>
    public void Merge()
    {
        if (merged) return;

        int inFeatures = baseLinear.InFeatures;
        int outFeatures = baseLinear.OutFeatures;

        var weight = baseLinear.Weight!.Tensor;
        var wData = new TWeight[outFeatures * inFeatures];
        weight.Data.CopyTo(wData, TWeight.Zero);

        var aSpan = a.Tensor.AsSpan();
        var bSpan = b.Tensor.AsSpan();

        for (int j = 0; j < outFeatures; j++)
        {
            for (int i = 0; i < inFeatures; i++)
            {
                TModel sum = TModel.Zero;
                for (int k = 0; k < r; k++)
                    sum += aSpan[i * r + k] * bSpan[k * outFeatures + j];
                wData[j * inFeatures + i] += TWeight.CreateChecked(scale * sum);
            }
        }

        var newWeight = ReverseGradTensor<TWeight>.FromMatrix(wData, outFeatures, inFeatures, requiresGrad: false);
        baseLinear.BindWeight(newWeight);

        merged = true;
    }
}
