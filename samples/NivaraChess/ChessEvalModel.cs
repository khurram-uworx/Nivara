using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;

namespace NivaraChess;

public abstract class ChessEvalModelBase : Module<float>
{
    public abstract int Phase { get; }
    public abstract float PredictCentipawns(ChessBoard board);
}

public sealed class ChessEvalModel : ChessEvalModelBase
{
    readonly Linear<float> fc1;
    readonly Linear<float> fc2;
    readonly Linear<float> output;

    public override int Phase => 1;

    public ChessEvalModel(int hiddenDim)
    {
        if (hiddenDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(hiddenDim), "Hidden dimension must be positive.");

        fc1 = new Linear<float>(ChessFeatures.FeatureCount, hiddenDim);
        fc2 = new Linear<float>(hiddenDim, Math.Max(4, hiddenDim / 2));
        output = new Linear<float>(Math.Max(4, hiddenDim / 2), 1);

        RegisterModules(fc1, fc2, output);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input)
    {
        var x = fc1.Forward(input);
        x = Activation.Relu(x);
        x = fc2.Forward(x);
        x = Activation.Relu(x);
        return output.Forward(x);
    }

    public override float PredictCentipawns(ChessBoard board)
    {
        ArgumentNullException.ThrowIfNull(board);

        var tensor = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(board.ToFeatureVector()),
            requiresGrad: false);
        tensor.Reshape(1, ChessFeatures.FeatureCount);

        var prediction = Forward(tensor);
        return prediction.Data[0];
    }
}

public sealed class NnueChessEvalModel : ChessEvalModelBase
{
    readonly SparseEmbedding<float> featureTransformer;
    readonly Linear<float> hidden;
    readonly Linear<float> output;

    public override int Phase => 2;

    public NnueChessEvalModel(int featureDim)
    {
        if (featureDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(featureDim), "Feature dimension must be positive.");

        featureTransformer = new SparseEmbedding<float>(ChessFeatures.HalfKpFeatureCount, featureDim);
        hidden = new Linear<float>(featureDim, 32);
        output = new Linear<float>(32, 1);

        RegisterModules(featureTransformer, hidden, output);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input)
    {
        var x = featureTransformer.Forward(input);
        x = Activation.Relu(x);
        x = hidden.Forward(x);
        x = Activation.Relu(x);
        return output.Forward(x);
    }

    public override float PredictCentipawns(ChessBoard board)
    {
        ArgumentNullException.ThrowIfNull(board);

        var tensor = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(board.ToHalfKpFeatureVector()),
            requiresGrad: false);
        tensor.Reshape(1, ChessFeatures.MaxActiveFeatures);

        var prediction = Forward(tensor);
        return prediction.Data[0];
    }
}
