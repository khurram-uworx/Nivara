using Nivara.AutoDiff.Nn.Initializers;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Buffers;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public class Linear<TModel, TWeight> : Module<TModel>
    where TModel : struct, IFloatingPointIeee754<TModel>
    where TWeight : struct, IFloatingPointIeee754<TWeight>
{
    readonly int inFeatures;
    readonly int outFeatures;
    readonly bool useBias;
    Parameter<TWeight>? weight;
    readonly Parameter<TModel>? bias;

    public int InFeatures => inFeatures;
    public int OutFeatures => outFeatures;
    public Parameter<TWeight>? Weight => weight;
    public Parameter<TModel>? Bias => bias;

    public Linear(int inFeatures, int outFeatures, bool bias = true,
        IInitializer<TWeight>? weightInitializer = null,
        IInitializer<TModel>? biasInitializer = null)
        : this(inFeatures, outFeatures, bias, weightInitializer, biasInitializer, false)
    {
    }

    internal Linear(int inFeatures, int outFeatures, bool bias,
        IInitializer<TWeight>? weightInitializer,
        IInitializer<TModel>? biasInitializer,
        bool skipWeightInitialization)
    {
        if (inFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(inFeatures));
        if (outFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(outFeatures));

        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        useBias = bias;

        if (!skipWeightInitialization)
        {
            var weightData = new TWeight[outFeatures * inFeatures];
            var weightTensor = ReverseGradTensor<TWeight>.FromMatrix(weightData, outFeatures, inFeatures, requiresGrad: true);
            weight = new Parameter<TWeight>("Weight", weightTensor);
            if (typeof(TWeight) == typeof(TModel))
                RegisterParameterObject(weight);
            else
                RegisterMixedParameters(weight);

            (weightInitializer ?? KaimingUniformInitializer<TWeight>.Instance).Initialize(weight);
        }

        if (bias)
        {
            var biasData = new TModel[outFeatures];
            var biasTensor = ReverseGradTensor<TModel>.FromMatrix(biasData, 1, outFeatures, requiresGrad: true);
            this.bias = new Parameter<TModel>("Bias", biasTensor);
            RegisterParameters(this.bias);

            if (biasInitializer != null)
                biasInitializer.Initialize(this.bias);
        }
    }

    /// <summary>
    /// Creates a <see cref="Linear{TModel, TWeight}"/> whose weight is left uninitialised
    /// (and unregistered) so that a checkpoint can be bound into it without allocating the
    /// weight-sized placeholder. Use <see cref="BindWeight"/> or <see cref="Module{T}.LoadStateDict"/>
    /// to supply the weight before use.
    /// </summary>
    public static Linear<TModel, TWeight> Uninitialized(int inFeatures, int outFeatures, bool bias = true)
    {
        if (inFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(inFeatures));
        if (outFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(outFeatures));
        return new Linear<TModel, TWeight>(inFeatures, outFeatures, bias, null, null, true);
    }

    /// <summary>
    /// Binds a weight tensor into this module, creating the weight parameter if it has not
    /// been initialised yet (the <see cref="Uninitialized"/> load path). A subsequent call
    /// replaces the bound tensor. The supplied tensor must have shape
    /// <c>[outFeatures, inFeatures]</c>.
    /// </summary>
    public void BindWeight(ReverseGradTensor<TWeight> weightTensor)
    {
        ArgumentNullException.ThrowIfNull(weightTensor);
        if (weightTensor.Shape.Length != 2 || weightTensor.Shape[0] != outFeatures || weightTensor.Shape[1] != inFeatures)
            throw new ArgumentException(
                $"Weight shape must be [{outFeatures}, {inFeatures}], but was [{string.Join(", ", weightTensor.Shape)}].",
                nameof(weightTensor));

        if (weight is null)
        {
            weight = new Parameter<TWeight>("Weight", weightTensor);
            if (typeof(TWeight) == typeof(TModel))
                RegisterParameterObject(weight);
            else
                RegisterMixedParameters(weight);
        }
        else
        {
            weight.Tensor = weightTensor;
        }
    }

    public override ReverseGradTensor<TModel> Forward(ReverseGradTensor<TModel> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (weight is null) throw new InvalidOperationException("Linear weight has not been loaded.");

        int aRows = input.shape[0];
        int aCols = input.shape[1];
        int bCols = outFeatures;

        input.Data.TryGetSpan(out var inputSpan);
        weight!.Tensor.Data.TryGetSpan(out var weightSpan);

        var resultArr = new TModel[aRows * bCols];
        MixedMatMulTransposedB(inputSpan, weightSpan, resultArr, aRows, aCols, bCols);

        var shouldTrack = GradientUtils.IsGradEnabled &&
            (input.RequiresGrad || weight!.Tensor.RequiresGrad || (bias?.Tensor.RequiresGrad ?? false));

        var resultTensor = new ReverseGradTensor<TModel>(
            NivaraColumn<TModel>.CreateFromOwnedArray(resultArr),
            shouldTrack,
            new[] { aRows, bCols });

        if (shouldTrack)
        {
            ReverseGradTensor<TModel>[] nodeInputs = typeof(TWeight) == typeof(TModel)
                ? [input, (ReverseGradTensor<TModel>)(object)weight!.Tensor]
                : [input];
            var gradFn = new OpNode<TModel>("LinearMixed", nodeInputs, (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gradSpan);
                weight!.Tensor.Data.TryGetSpan(out var weightSpanLocal);
                input.Data.TryGetSpan(out var inputSpanLocal);

                if (input.RequiresGrad)
                {
                    var aGradArr = new TModel[aRows * aCols];
                    MixedMatMul(gradSpan, weightSpanLocal, aGradArr, aRows, bCols, aCols);
                    ReverseGradOperations.AccumulateGradient(input, NivaraColumn<TModel>.CreateFromOwnedArray(aGradArr));
                }

                if (weight!.Tensor.RequiresGrad)
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
                        ReverseGradOperations.AccumulateGradient(weight!.Tensor, NivaraColumn<TWeight>.CreateFromOwnedArray(narrowed));
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

    public override string ExportToOnnx(IOnnxExportContext<TModel> context, string inputName)
    {
        if (weight is null) throw new InvalidOperationException("Linear weight has not been loaded.");
        return context.Linear<TWeight>(inputName, this);
    }

    public override Dictionary<string, ReverseGradTensor<TModel>> StateDict()
    {
        if (weight is null) throw new InvalidOperationException("Linear weight has not been loaded.");
        var state = base.StateDict();
        var weightTensor = TypeConverter.Convert<TWeight, TModel>(weight.Tensor, weight.Tensor.RequiresGrad);
        state["Weight"] = Module<TModel>.CloneTensor(weightTensor);
        return state;
    }

    public override void LoadStateDict(IReadOnlyDictionary<string, ReverseGradTensor<TModel>> stateDict, bool strict = false)
    {
        ArgumentNullException.ThrowIfNull(stateDict);

        var remaining = new Dictionary<string, ReverseGradTensor<TModel>>();
        bool weightSeen = false;

        foreach (var (name, source) in stateDict)
        {
            if (name == "Weight")
            {
                ValidateShapePublic(name, source.Shape, new[] { outFeatures, inFeatures });
                if (typeof(TWeight) == typeof(TModel))
                {
                    var cloned = Module<TModel>.CloneTensor(source);
                    BindWeight((ReverseGradTensor<TWeight>)(object)cloned);
                }
                else
                {
                    var converted = TypeConverter.Convert<TModel, TWeight>(source, weight?.Tensor.RequiresGrad ?? true);
                    var values = new TWeight[converted.Length];
                    converted.Data.CopyTo(values, TWeight.Zero);
                    var data = NivaraColumn<TWeight>.Create(values);
                    BindWeight(new ReverseGradTensor<TWeight>(data, converted.RequiresGrad, converted.Shape));
                }
                weightSeen = true;
                continue;
            }
            remaining[name] = source;
        }

        base.LoadStateDict(remaining, false);

        if (strict)
        {
            var modelParameters = GetParameters();
            var missing = modelParameters.Keys
                .Where(k => !(k == "Weight" ? weightSeen : remaining.ContainsKey(k)))
                .ToArray();
            if (missing.Length > 0)
                throw new InvalidOperationException(
                    $"State dictionary is missing model parameters: [{string.Join(", ", missing)}]");
        }
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
        => GradKernels.MixedMatMulTransposedB(a, b, output, aRows, aCols, bCols);

    static void MixedMatMul(
        ReadOnlySpan<TModel> a, ReadOnlySpan<TWeight> b, TModel[] output,
        int aRows, int aCols, int bCols)
        => GradKernels.MixedMatMul(a, b, output, aRows, aCols, bCols);
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

    internal Linear(int inFeatures, int outFeatures, bool bias, bool skipWeightInitialization)
        : base(inFeatures, outFeatures, bias, null, null, skipWeightInitialization)
    {
    }

    public new static Linear<T> Uninitialized(int inFeatures, int outFeatures, bool bias = true)
    {
        if (inFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(inFeatures));
        if (outFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(outFeatures));
        return new Linear<T>(inFeatures, outFeatures, bias, true);
    }

    public override string ExportToOnnx(IOnnxExportContext<T> context, string inputName)
    {
        if (Weight is null) throw new InvalidOperationException("Linear weight has not been loaded.");
        return context.Linear(inputName, this);
    }
}
