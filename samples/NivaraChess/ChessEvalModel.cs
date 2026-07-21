using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;

namespace NivaraChess;

public sealed class ChessEvalModel : Module<float>
{
    readonly Linear<float> fc1;
    readonly Linear<float> fc2;
    readonly Linear<float> output;

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

    public float PredictCentipawns(ChessBoard board)
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
