using Nivara.AutoDiff.Nn.Initializers;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Learnable lookup table mapping token IDs to dense vectors. Supports both single-token and
/// batched lookups via <see cref="ReverseGradOperations.Gather"/>. Weights are initialized
/// with a normal distribution (std 0.02).
/// </summary>
/// <typeparam name="TModel">The compute (activation) floating-point type.</typeparam>
/// <typeparam name="TWeight">The storage (weight) floating-point type.</typeparam>
public class Embedding<TModel, TWeight> : Module<TModel>
    where TModel : struct, IFloatingPointIeee754<TModel>
    where TWeight : struct, IFloatingPointIeee754<TWeight>
{
    readonly int numEmbeddings;
    readonly int embeddingDim;
    Parameter<TWeight>? weight;

    /// <summary>Gets the size of the vocabulary (number of embedding rows).</summary>
    public int NumEmbeddings => numEmbeddings;
    /// <summary>Gets the dimension of each embedding vector.</summary>
    public int EmbeddingDim => embeddingDim;
    /// <summary>Gets the weight parameter (shape <c>[numEmbeddings, embeddingDim]</c>).</summary>
    public Parameter<TWeight>? Weight => weight;

    public Embedding(int numEmbeddings, int embeddingDim)
        : this(numEmbeddings, embeddingDim, false)
    {
    }

    internal Embedding(int numEmbeddings, int embeddingDim, bool skipWeightInitialization)
    {
        if (numEmbeddings <= 0) throw new ArgumentOutOfRangeException(nameof(numEmbeddings));
        if (embeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDim));

        this.numEmbeddings = numEmbeddings;
        this.embeddingDim = embeddingDim;

        if (!skipWeightInitialization)
        {
            var data = new TWeight[numEmbeddings * embeddingDim];
            var tensor = ReverseGradTensor<TWeight>.FromMatrix(data, numEmbeddings, embeddingDim, requiresGrad: true);
            weight = new Parameter<TWeight>("Weight", tensor);
            if (typeof(TWeight) == typeof(TModel))
                RegisterParameterObject(weight);
            else
                RegisterMixedParameters(weight);

            var init = new NormalInitializer<TWeight>(TWeight.Zero, TWeight.CreateChecked(0.02));
            init.Initialize(weight);
        }
    }

    /// <summary>
    /// Creates an <see cref="Embedding{TModel, TWeight}"/> whose weight is left uninitialised
    /// so that a checkpoint can be bound into it without allocating the weight-sized placeholder.
    /// Use <see cref="BindWeight"/> or <see cref="Module{T}.LoadStateDict"/> to supply the
    /// weight before use.
    /// </summary>
    public static Embedding<TModel, TWeight> Uninitialized(int numEmbeddings, int embeddingDim)
    {
        if (numEmbeddings <= 0) throw new ArgumentOutOfRangeException(nameof(numEmbeddings));
        if (embeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDim));
        return new Embedding<TModel, TWeight>(numEmbeddings, embeddingDim, true);
    }

    /// <summary>
    /// Binds a weight tensor into this module, creating the weight parameter if it has not
    /// been initialised yet (the <see cref="Uninitialized"/> load path). A subsequent call
    /// replaces the bound tensor. The supplied tensor must have shape
    /// <c>[numEmbeddings, embeddingDim]</c>.
    /// </summary>
    public void BindWeight(ReverseGradTensor<TWeight> weightTensor)
    {
        ArgumentNullException.ThrowIfNull(weightTensor);
        if (weightTensor.Shape.Length != 2 || weightTensor.Shape[0] != numEmbeddings || weightTensor.Shape[1] != embeddingDim)
            throw new ArgumentException(
                $"Weight shape must be [{numEmbeddings}, {embeddingDim}], but was [{string.Join(", ", weightTensor.Shape)}].",
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

    /// <summary>
    /// Looks up the embedding vector for each token ID in the input, producing a tensor of
    /// shape <c>inputShape + [embeddingDim]</c> (or <c>[numTokens, embeddingDim]</c> for flat input).
    /// </summary>
    /// <param name="input">Tensor of token IDs, each in <c>[0, numEmbeddings)</c></param>
    /// <returns>The embedded tensor</returns>
    public override ReverseGradTensor<TModel> Forward(ReverseGradTensor<TModel> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (weight is null) throw new InvalidOperationException("Embedding weight has not been loaded.");

        if (input.Length == 0)
            throw new ArgumentException("Embedding.Forward input tensor is empty.", nameof(input));

        int[] originalShape = input.Shape;
        int totalTokens = input.Length;

        var tokenIds = new int[totalTokens];
        for (int i = 0; i < totalTokens; i++)
            tokenIds[i] = int.CreateChecked(input.Data[i]);

        for (int i = 0; i < totalTokens; i++)
        {
            if (tokenIds[i] < 0 || tokenIds[i] >= numEmbeddings)
                throw new ArgumentOutOfRangeException(
                    nameof(input),
                    $"Token ID at position {i} is {tokenIds[i]}, " +
                    $"must be in range [0, {numEmbeddings}).");
        }

        return ForwardCore(tokenIds, originalShape, totalTokens > 0 && originalShape.Length > 1);
    }

    /// <summary>
    /// Looks up embedding vectors for a batch of exact token IDs. Unlike
    /// <see cref="Forward(ReverseGradTensor{TModel})"/>, this path preserves token IDs
    /// exactly regardless of the compute dtype <typeparamref name="TModel"/>, so it is the
    /// correct overload for narrow-precision dtypes (e.g. <see cref="BFloat16"/>,
    /// <see cref="Half"/>) whose representable integer range (&le; 256) is far smaller
    /// than typical vocabularies (30k+). Token IDs are stored as <see cref="int"/> and
    /// never round-tripped through <typeparamref name="TModel"/>.
    /// </summary>
    /// <param name="tokenIds">Exact token IDs, each in <c>[0, numEmbeddings)</c></param>
    /// <returns>The embedded tensor of shape <c>[tokenIds.Length, embeddingDim]</c></returns>
    public ReverseGradTensor<TModel> Forward(int[] tokenIds)
    {
        if (tokenIds == null) throw new ArgumentNullException(nameof(tokenIds));
        if (weight is null) throw new InvalidOperationException("Embedding weight has not been loaded.");
        if (tokenIds.Length == 0)
            throw new ArgumentException("Embedding.Forward input is empty.", nameof(tokenIds));

        for (int i = 0; i < tokenIds.Length; i++)
        {
            if (tokenIds[i] < 0 || tokenIds[i] >= numEmbeddings)
                throw new ArgumentOutOfRangeException(
                    nameof(tokenIds),
                    $"Token ID at position {i} is {tokenIds[i]}, " +
                    $"must be in range [0, {numEmbeddings}).");
        }

        return ForwardCore(tokenIds, null, false);
    }

    /// <summary>
    /// Looks up the embedding vector for a single token ID.
    /// </summary>
    /// <param name="tokenId">The token ID in <c>[0, numEmbeddings)</c></param>
    /// <returns>The embedding vector of length <c>embeddingDim</c></returns>
    public ReverseGradTensor<TModel> Forward(int tokenId)
    {
        if (tokenId < 0 || tokenId >= numEmbeddings)
            throw new ArgumentOutOfRangeException(nameof(tokenId));
        if (weight is null) throw new InvalidOperationException("Embedding weight has not been loaded.");

        return ForwardCore(new[] { tokenId }, null, false);
    }

    ReverseGradTensor<TModel> ForwardCore(int[] tokenIds, int[]? originalShape, bool reshape)
    {
        if (weight is null) throw new InvalidOperationException("Embedding weight has not been loaded.");

        int stride = embeddingDim;
        int totalTokens = tokenIds.Length;
        var resultArr = new TModel[totalTokens * stride];

        weight!.Tensor.Data.TryGetSpan(out var wSpan);

        for (int i = 0; i < totalTokens; i++)
        {
            int srcOffset = tokenIds[i] * stride;
            int dstOffset = i * stride;
            for (int j = 0; j < stride; j++)
                resultArr[dstOffset + j] = TModel.CreateChecked(wSpan[srcOffset + j]);
        }

        var resultCol = NivaraColumn<TModel>.CreateFromOwnedArray(resultArr);
        var resultShape = new[] { totalTokens, embeddingDim };
        var resultTensor = new ReverseGradTensor<TModel>(resultCol, ShouldTrack(), resultShape);

        bool shouldTrack = ShouldTrack();
        // ComputationGraph node attachment (mirror Linear pattern).
        if (shouldTrack)
        {
            var savedTokenIds = (int[])tokenIds.Clone();
            var nodeInputs = typeof(TWeight) == typeof(TModel)
                ? (IReadOnlyList<ReverseGradTensor<TModel>>)
                    new[] { (ReverseGradTensor<TModel>)(object)weight!.Tensor }
                : (IReadOnlyList<ReverseGradTensor<TModel>>)Array.Empty<ReverseGradTensor<TModel>>();
            var gradFn = new OpNode<TModel>("EmbeddingMixed", nodeInputs, (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gradSpan);
                if (!weight!.Tensor.RequiresGrad)
                    return;

                var gAcc = new TModel[numEmbeddings * embeddingDim];
                for (int i = 0; i < savedTokenIds.Length; i++)
                {
                    int srcRow = savedTokenIds[i] * stride;
                    int dstRow = i * stride;
                    for (int j = 0; j < stride; j++)
                        gAcc[srcRow + j] += gradSpan[dstRow + j];
                }

                var narrowed = new TWeight[numEmbeddings * embeddingDim];
                for (int i = 0; i < narrowed.Length; i++)
                    narrowed[i] = TWeight.CreateChecked(gAcc[i]);

                ReverseGradOperations.AccumulateGradient(
                    weight!.Tensor,
                    NivaraColumn<TWeight>.CreateFromOwnedArray(narrowed));
            });
            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        if (reshape && originalShape != null && originalShape.Length > 1)
            resultTensor.Reshape(originalShape.Append(embeddingDim).ToArray());

        return resultTensor;
    }

    /// <summary>
    /// Gathers embedding rows for the given token ids into <paramref name="destination"/> in the
    /// compute dtype, reading the weight storage dtype and converting per element. For
    /// <typeparamref name="TWeight"/> == <typeparamref name="TModel"/> this degenerates to a
    /// <see cref="Span{T}.CopyTo"/> per row (no conversion). Used by the encoder's inference
    /// path so it never widens the whole embedding table to read a handful of rows.
    /// </summary>
    internal void GatherRows(int[] tokenIds, Span<TModel> destination)
    {
        if (tokenIds == null) throw new ArgumentNullException(nameof(tokenIds));
        if (weight is null) throw new InvalidOperationException("Embedding weight has not been loaded.");
        if (destination.Length < tokenIds.Length * embeddingDim)
            throw new ArgumentException("Destination span is too small.", nameof(destination));

        weight!.Tensor.Data.TryGetSpan(out var wSpan);
        int stride = embeddingDim;
        if (typeof(TWeight) == typeof(TModel))
        {
            var wModel = MemoryMarshal.Cast<TWeight, TModel>(wSpan);
            for (int i = 0; i < tokenIds.Length; i++)
                wModel.Slice(tokenIds[i] * stride, stride).CopyTo(destination.Slice(i * stride, stride));
            return;
        }
        for (int i = 0; i < tokenIds.Length; i++)
        {
            int srcOffset = tokenIds[i] * stride;
            int dstOffset = i * stride;
            for (int j = 0; j < stride; j++)
                destination[dstOffset + j] = TModel.CreateChecked(wSpan[srcOffset + j]);
        }
    }

    bool ShouldTrack() => GradientUtils.IsGradEnabled && weight!.Tensor.RequiresGrad;

    public override string ExportToOnnx(IOnnxExportContext<TModel> context, string inputName)
    {
        if (weight is null) throw new InvalidOperationException("Embedding weight has not been loaded.");
        return context.Embedding<TWeight>(inputName, this);
    }

    public override Dictionary<string, ReverseGradTensor<TModel>> StateDict()
    {
        if (weight is null) throw new InvalidOperationException("Embedding weight has not been loaded.");
        var state = base.StateDict();
        var weightTensor = TypeConverter.Convert<TWeight, TModel>(weight.Tensor, weight.Tensor.RequiresGrad);
        state["Weight"] = Module<TModel>.CloneTensor(weightTensor);
        return state;
    }

    public override void LoadStateDict(
        IReadOnlyDictionary<string, ReverseGradTensor<TModel>> stateDict,
        bool strict = false)
    {
        ArgumentNullException.ThrowIfNull(stateDict);

        var remaining = new Dictionary<string, ReverseGradTensor<TModel>>();
        bool weightSeen = false;

        foreach (var (name, source) in stateDict)
        {
            if (name == "Weight")
            {
                if (source.Shape.Length != 2 || source.Shape[0] != numEmbeddings || source.Shape[1] != embeddingDim)
                    throw new InvalidOperationException(
                        $"Parameter 'Weight' shape mismatch: " +
                        $"state has [{string.Join(", ", source.Shape)}], " +
                        $"model has [{numEmbeddings}, {embeddingDim}].");
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
}

public sealed class Embedding<T> : Embedding<T, T>
    where T : struct, IFloatingPointIeee754<T>
{
    public Embedding(int numEmbeddings, int embeddingDim)
        : base(numEmbeddings, embeddingDim)
    {
    }

    internal Embedding(int numEmbeddings, int embeddingDim, bool skipWeightInitialization)
        : base(numEmbeddings, embeddingDim, skipWeightInitialization)
    {
    }

    public new static Embedding<T> Uninitialized(int numEmbeddings, int embeddingDim)
        => new Embedding<T>(numEmbeddings, embeddingDim, true);

    public override string ExportToOnnx(IOnnxExportContext<T> context, string inputName)
        => context.Embedding(inputName, this);
}
