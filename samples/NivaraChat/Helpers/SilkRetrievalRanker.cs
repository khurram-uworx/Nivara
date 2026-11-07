using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using NivaraChat.Helpers;

namespace NivaraChat.Helpers;

internal sealed class SilkFusionRanker : IFusionRanker
{
    readonly SilkRetrievalTrainer trainer;

    public SilkFusionRanker(SilkRetrievalTrainer trainer)
    {
        this.trainer = trainer;
    }

    public SilkRetrievalTrainer Trainer => trainer;

    public int FeatureCount => trainer.FeatureCount;

    public float[] Weights => trainer.Weights;

    public IReadOnlyList<Parameter<float>> TrainableParameters => Array.Empty<Parameter<float>>();

    public float[] FeatureVector(TrainingQuery query) => trainer.FeatureVector(query);

    public ReverseGradTensor<float> ForwardMasked(TrainingQuery query)
    {
        var scores = trainer.Scores(query);
        return ReverseGradTensor<float>.FromMatrix(scores, query.RowCount, 1, requiresGrad: false);
    }

    public float[] WeightsFor(TrainingQuery query) => trainer.WeightsFor(query);
}
