using Nivara.AutoDiff.Nn.Initializers;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using Nivara.Primitives;
using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Nivara.AutoDiff.Nn;

public class Linear<TModel, TWeight> : Module<TModel>
    where TModel : struct, IFloatingPointIeee754<TModel>
    where TWeight : struct, IFloatingPointIeee754<TWeight>
{
    readonly int inFeatures;
    readonly int outFeatures;
    readonly bool useBias;
    readonly Parameter<TWeight> weight;
    readonly Parameter<TModel>? bias;

    public int InFeatures => inFeatures;
    public int OutFeatures => outFeatures;
    public Parameter<TWeight>? Weight => weight;
    public Parameter<TModel>? Bias => bias;

    public Linear(int inFeatures, int outFeatures, bool bias = true,
        IInitializer<TWeight>? weightInitializer = null,
        IInitializer<TModel>? biasInitializer = null)
    {
        if (inFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(inFeatures));
        if (outFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(outFeatures));

        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        useBias = bias;

        var weightData = new TWeight[outFeatures * inFeatures];
        var weightTensor = ReverseGradTensor<TWeight>.FromMatrix(weightData, outFeatures, inFeatures, requiresGrad: true);
        weight = new Parameter<TWeight>("Weight", weightTensor);
        if (typeof(TWeight) == typeof(TModel))
            RegisterParameterObject(weight);
        else
            RegisterMixedParameters(weight);

        if (bias)
        {
            var biasData = new TModel[outFeatures];
            var biasTensor = ReverseGradTensor<TModel>.FromMatrix(biasData, 1, outFeatures, requiresGrad: true);
            this.bias = new Parameter<TModel>("Bias", biasTensor);
            RegisterParameters(this.bias);
        }

        (weightInitializer ?? KaimingUniformInitializer<TWeight>.Instance).Initialize(weight);

        if (bias && biasInitializer != null)
            biasInitializer.Initialize(this.bias!);
    }

    public override ReverseGradTensor<TModel> Forward(ReverseGradTensor<TModel> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        int aRows = input.shape[0];
        int aCols = input.shape[1];
        int bCols = outFeatures;

        input.Data.TryGetSpan(out var inputSpan);
        weight.Tensor.Data.TryGetSpan(out var weightSpan);

        var resultArr = new TModel[aRows * bCols];
        MixedMatMulTransposedB(inputSpan, weightSpan, resultArr, aRows, aCols, bCols);

        var shouldTrack = GradientUtils.IsGradEnabled &&
            (input.RequiresGrad || weight.Tensor.RequiresGrad || (bias?.Tensor.RequiresGrad ?? false));

        var resultTensor = new ReverseGradTensor<TModel>(
            NivaraColumn<TModel>.CreateFromOwnedArray(resultArr),
            shouldTrack,
            new[] { aRows, bCols });

        if (shouldTrack)
        {
            ReverseGradTensor<TModel>[] nodeInputs = typeof(TWeight) == typeof(TModel)
                ? [input, (ReverseGradTensor<TModel>)(object)weight.Tensor]
                : [input];
            var gradFn = new OpNode<TModel>("LinearMixed", nodeInputs, (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gradSpan);
                weight.Tensor.Data.TryGetSpan(out var weightSpanLocal);
                input.Data.TryGetSpan(out var inputSpanLocal);

                if (input.RequiresGrad)
                {
                    var aGradArr = new TModel[aRows * aCols];
                    MixedMatMul(gradSpan, weightSpanLocal, aGradArr, aRows, bCols, aCols);
                    ReverseGradOperations.AccumulateGradient(input, NivaraColumn<TModel>.CreateFromOwnedArray(aGradArr));
                }

                if (weight.Tensor.RequiresGrad)
                {
                    var gTArr = ArrayPool<TModel>.Shared.Rent(Math.Max(bCols * aRows, 1));
                    try
                    {
                        GradKernels.Transpose<TModel>(gradSpan, gTArr.AsSpan(0, bCols * aRows), aRows, bCols);
                        var bGradArr = new TModel[bCols * aCols];
                        GradKernels.MatMul<TModel>(gTArr.AsSpan(0, bCols * aRows), inputSpanLocal, bGradArr, bCols, aRows, aCols);

                        var narrowed = new TWeight[bCols * aCols];
                        for (int i = 0; i < narrowed.Length; i++)
                            narrowed[i] = TWeight.CreateChecked(bGradArr[i]);
                        ReverseGradOperations.AccumulateGradient(weight.Tensor, NivaraColumn<TWeight>.CreateFromOwnedArray(narrowed));
                    }
                    finally
                    {
                        ArrayPool<TModel>.Shared.Return(gTArr, clearArray: true);
                    }
                }
            });
            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        if (useBias && bias != null)
            resultTensor = ReverseGradOperations.AddBias(resultTensor, bias.Tensor);

        return resultTensor;
    }

    public override Dictionary<string, ReverseGradTensor<TModel>> StateDict()
    {
        var state = base.StateDict();
        var weightTensor = TypeConverter.Convert<TWeight, TModel>(weight.Tensor, weight.Tensor.RequiresGrad);
        state["Weight"] = Module<TModel>.CloneTensor(weightTensor);
        return state;
    }

    public override void LoadStateDict(IReadOnlyDictionary<string, ReverseGradTensor<TModel>> stateDict, bool strict = false)
    {
        ArgumentNullException.ThrowIfNull(stateDict);

        var remaining = new Dictionary<string, ReverseGradTensor<TModel>>();

        foreach (var (name, source) in stateDict)
        {
            if (name == "Weight")
            {
                ValidateShapePublic(name, source.Shape, weight.Shape);
                var converted = TypeConverter.Convert<TModel, TWeight>(source, weight.Tensor.RequiresGrad);
                var values = new TWeight[converted.Length];
                converted.Data.CopyTo(values, TWeight.Zero);
                var data = NivaraColumn<TWeight>.Create(values);
                weight.Tensor = new ReverseGradTensor<TWeight>(data, converted.RequiresGrad, converted.Shape);
            }
            else
                remaining[name] = source;
        }

        base.LoadStateDict(remaining, strict);
    }

    static void ValidateShapePublic(string name, int[] sourceShape, int[] targetShape)
    {
        if (sourceShape.Length != targetShape.Length)
            throw new InvalidOperationException(
                $"Parameter '{name}' shape rank mismatch: " +
                $"state has {sourceShape.Length}D, model has {targetShape.Length}D.");

        for (int i = 0; i < sourceShape.Length; i++)
        {
            if (sourceShape[i] != targetShape[i])
                throw new InvalidOperationException(
                    $"Parameter '{name}' shape mismatch: " +
                    $"state has [{string.Join(", ", sourceShape)}], " +
                    $"model has [{string.Join(", ", targetShape)}].");
        }
    }

    static void MixedMatMulTransposedB(
        ReadOnlySpan<TModel> a, ReadOnlySpan<TWeight> b, TModel[] output,
        int aRows, int aCols, int bCols)
    {
        if (typeof(TWeight) == typeof(TModel))
        {
            GradKernels.MatMulTransposedB<TModel>(a, MemoryMarshal.Cast<TWeight, TModel>(b), output, aRows, aCols, bCols);
            return;
        }
        if (typeof(TWeight) == typeof(BFloat16) && typeof(TModel) == typeof(float))
        {
            var aF = MemoryMarshal.Cast<TModel, float>(a);
            var bBf = MemoryMarshal.Cast<TWeight, BFloat16>(b);
            var oF = MemoryMarshal.Cast<TModel, float>(output.AsSpan());
            NarrowFloatKernels.MatMulTransposedB(aF, bBf, oF, aRows, aCols, bCols);
            return;
        }
        if (typeof(TWeight) == typeof(Half) && typeof(TModel) == typeof(float))
        {
            var aF = MemoryMarshal.Cast<TModel, float>(a);
            var bH = MemoryMarshal.Cast<TWeight, Half>(b);
            var oF = MemoryMarshal.Cast<TModel, float>(output.AsSpan());
            NarrowFloatKernels.MatMulTransposedB(aF, bH, oF, aRows, aCols, bCols);
            return;
        }

        var widened = Array.ConvertAll(b.ToArray(), x => TModel.CreateChecked(x));
        GradKernels.MatMulTransposedB<TModel>(a, widened, output, aRows, aCols, bCols);
    }

    static void MixedMatMul(
        ReadOnlySpan<TModel> a, ReadOnlySpan<TWeight> b, TModel[] output,
        int aRows, int aCols, int bCols)
    {
        if (typeof(TWeight) == typeof(TModel))
        {
            GradKernels.MatMul<TModel>(a, MemoryMarshal.Cast<TWeight, TModel>(b), output, aRows, aCols, bCols);
            return;
        }
        if (typeof(TWeight) == typeof(BFloat16) && typeof(TModel) == typeof(float))
        {
            var aF = MemoryMarshal.Cast<TModel, float>(a);
            var bBf = MemoryMarshal.Cast<TWeight, BFloat16>(b);
            var oF = MemoryMarshal.Cast<TModel, float>(output.AsSpan());
            NarrowFloatKernels.MatMul(aF, bBf, oF, aRows, aCols, bCols);
            return;
        }
        if (typeof(TWeight) == typeof(Half) && typeof(TModel) == typeof(float))
        {
            var widened = Array.ConvertAll(b.ToArray(), x => TModel.CreateChecked(x));
            GradKernels.MatMul<TModel>(a, widened, output, aRows, aCols, bCols);
            return;
        }

        var w = Array.ConvertAll(b.ToArray(), x => TModel.CreateChecked(x));
        GradKernels.MatMul<TModel>(a, w, output, aRows, aCols, bCols);
    }
}

public sealed class Linear<T> : Linear<T, T>
    where T : struct, IFloatingPointIeee754<T>
{
    public Linear(int inFeatures, int outFeatures, bool bias = true,
        IInitializer<T>? weightInitializer = null,
        IInitializer<T>? biasInitializer = null)
        : base(inFeatures, outFeatures, bias, weightInitializer, biasInitializer)
    {
    }
}
