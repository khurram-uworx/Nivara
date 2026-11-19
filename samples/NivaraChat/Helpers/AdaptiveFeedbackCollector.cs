using Nivara.AI;
using System.Diagnostics;

namespace NivaraChat.Helpers;

public sealed class AdaptiveFeedbackCollector
{
    readonly LayaDecisionHead<float, float> head;
    readonly string weightsPath;
    readonly int batchThreshold;
    readonly int epochs;
    readonly float learningRate;
    readonly bool persist;
    readonly Action<LayaDecisionHead<float, float>, IReadOnlyList<AdaptiveTrainingExample>, int, float>? updateAction;
    readonly List<AdaptiveTrainingExample> buffer = [];

    public int BufferCount => buffer.Count;
    public int TotalCollected { get; private set; }
    public int SuccessfulUpdates { get; private set; }
    public bool Persist { get; }
    public long UpdateMilliseconds { get; private set; }
    public long PersistMilliseconds { get; private set; }

    public AdaptiveFeedbackCollector(
        LayaDecisionHead<float, float> head,
        string weightsPath,
        int batchThreshold = 10,
        int epochs = AdaptiveTraining.DefaultEpochs,
        float learningRate = AdaptiveTraining.DefaultLearningRate,
        bool persist = true,
        Action<LayaDecisionHead<float, float>, IReadOnlyList<AdaptiveTrainingExample>, int, float>? updateAction = null)
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
        this.persist = persist;
        this.updateAction = updateAction;
        Persist = persist;
    }

    public bool Add(AdaptiveTrainingExample example)
    {
        ArgumentNullException.ThrowIfNull(example);
        AdaptiveTraining.ValidateExample(head, example);
        buffer.Add(example);
        TotalCollected++;
        return Flush();
    }

    public bool Flush() => TrainAndPersist(requireFullBatch: true);

    public bool FlushRemaining() => TrainAndPersist(requireFullBatch: false);

    bool TrainAndPersist(bool requireFullBatch)
    {
        if (buffer.Count == 0)
            return false;
        if (requireFullBatch && buffer.Count < batchThreshold)
            return false;

        var previousState = head.StateDict();
        try
        {
            var updateSw = Stopwatch.StartNew();
            (updateAction ?? AdaptiveTraining.Update)(head, buffer, epochs, learningRate);
            updateSw.Stop();
            UpdateMilliseconds += updateSw.ElapsedMilliseconds;

            if (persist)
            {
                var persistSw = Stopwatch.StartNew();
                AdaptiveHeadPersistence.Save(head, weightsPath);
                persistSw.Stop();
                PersistMilliseconds += persistSw.ElapsedMilliseconds;
            }
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
