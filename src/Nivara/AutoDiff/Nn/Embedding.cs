using Nivara.AutoDiff.Nn.Initializers;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Numerics;

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
    readonly Parameter<TWeight> weight;

    /// <summary>Gets the size of the vocabulary (number of embedding rows).</summary>
    public int NumEmbeddings => numEmbeddings;
    /// <summary>Gets the dimension of each embedding vector.</summary>
    public int EmbeddingDim => embeddingDim;
    /// <summary>Gets the weight parameter (shape <c>[numEmbeddings, embeddingDim]</c>).</summary>
    public Parameter<TWeight>? Weight => weight;

    /// <summary>
    /// Creates an embedding table.
    /// </summary>
    /// <param name="numEmbeddings">Size of the vocabulary (must be positive)</param>
    /// <param name="embeddingDim">Dimension of each embedding vector (must be positive)</param>
    public Embedding(int numEmbeddings, int embeddingDim)
    {
        if (numEmbeddings <= 0) throw new ArgumentOutOfRangeException(nameof(numEmbeddings));
        if (embeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDim));

        this.numEmbeddings = numEmbeddings;
        this.embeddingDim = embeddingDim;

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

    /// <summary>
    /// Looks up the embedding vector for each token ID in the input, producing a tensor of
    /// shape <c>inputShape + [embeddingDim]</c> (or <c>[numTokens, embeddingDim]</c> for flat input).
    /// </summary>
    /// <param name="input">Tensor of token IDs, each in <c>[0, numEmbeddings)</c></param>
    /// <returns>The embedded tensor</returns>
    public override ReverseGradTensor<TModel> Forward(ReverseGradTensor<TModel> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

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

        return ForwardCore(new[] { tokenId }, null, false);
    }

    ReverseGradTensor<TModel> ForwardCore(int[] tokenIds, int[]? originalShape, bool reshape)
    {
        int stride = embeddingDim;
        int totalTokens = tokenIds.Length;
        var resultArr = new TModel[totalTokens * stride];

        weight.Tensor.Data.TryGetSpan(out var wSpan);

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
                    new[] { (ReverseGradTensor<TModel>)(object)weight.Tensor }
                : (IReadOnlyList<ReverseGradTensor<TModel>>)Array.Empty<ReverseGradTensor<TModel>>();
            var gradFn = new OpNode<TModel>("EmbeddingMixed", nodeInputs, (typedGradOutput) =>
            {
                typedGradOutput.TryGetSpan(out var gradSpan);
                if (!weight.Tensor.RequiresGrad)
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
                    weight.Tensor,
                    NivaraColumn<TWeight>.CreateFromOwnedArray(narrowed));
            });
            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        if (reshape && originalShape != null && originalShape.Length > 1)
            resultTensor.Reshape(originalShape.Append(embeddingDim).ToArray());

        return resultTensor;
    }

    bool ShouldTrack() => GradientUtils.IsGradEnabled && weight.Tensor.RequiresGrad;

    public override Dictionary<string, ReverseGradTensor<TModel>> StateDict()
    {
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

        foreach (var (name, source) in stateDict)
        {
            if (name == "Weight" && typeof(TWeight) != typeof(TModel))
            {
                ValidateShape(name, source.Shape, weight.Shape);
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

    static void ValidateShape(string name, int[] sourceShape, int[] targetShape)
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
}

public sealed class Embedding<T> : Embedding<T, T>
    where T : struct, IFloatingPointIeee754<T>
{
    public Embedding(int numEmbeddings, int embeddingDim)
        : base(numEmbeddings, embeddingDim)
    {
    }
}
