using Nivara.AI;

namespace NivaraChat.Helpers;

public sealed class AdaptiveFeedbackCollector
{
    readonly LayaDecisionHead<float, float> head;
    readonly string weightsPath;
    readonly int batchThreshold;
    readonly int epochs;
    readonly float learningRate;
    readonly List<AdaptiveTrainingExample> buffer = [];

    public int BufferCount => buffer.Count;
    public int TotalCollected { get; private set; }
    public int SuccessfulUpdates { get; private set; }

    public AdaptiveFeedbackCollector(
        LayaDecisionHead<float, float> head,
        string weightsPath,
        int batchThreshold = 10,
        int epochs = AdaptiveTraining.DefaultEpochs,
        float learningRate = AdaptiveTraining.DefaultLearningRate)
    {
        ArgumentNullException.ThrowIfNull(head);
        ArgumentException.ThrowIfNullOrWhiteSpace(weightsPath);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchThreshold);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(epochs);
        if (!float.IsFinite(learningRate) || learningRate <= 0f)
            throw new ArgumentOutOfRangeException(nameof(learningRate));

        this.head = head;
        this.weightsPath = weightsPath;
        this.batchThreshold = batchThreshold;
        this.epochs = epochs;
        this.learningRate = learningRate;
    }

    public bool Add(AdaptiveTrainingExample example)
    {
        ArgumentNullException.ThrowIfNull(example);
        AdaptiveTraining.ValidateExample(head, example);
        buffer.Add(example);
        TotalCollected++;
        return Flush();
    }

    public bool Flush()
    {
        if (buffer.Count < batchThreshold)
            return false;

        var previousState = head.StateDict();
        try
        {
            AdaptiveTraining.Update(head, buffer, epochs, learningRate);
            AdaptiveHeadPersistence.Save(head, weightsPath);
        }
        catch (Exception updateError)
        {
            try
            {
                head.LoadStateDict(previousState, strict: true);
            }
            catch (Exception restoreError)
            {
                throw new AggregateException(
                    "Adaptive feedback update failed and the in-memory head could not be restored.",
                    updateError,
                    restoreError);
            }

            throw;
        }

        buffer.Clear();
        SuccessfulUpdates++;
        return true;
    }
}
