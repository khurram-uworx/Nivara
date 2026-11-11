using Nivara.AI;
using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

[Category("Gpu")]
public sealed class SilkDistilBertTrainerBf16SmokeTests
{
    [Test]
    public void Train_Bf16Checkpoint_LossDecreases()
    {
        var runtime = SilkBertBf16Fixtures.Device();
        using (runtime)
        {
            var config = SilkBertBf16Fixtures.Config();
            var bf16Tensors = SilkBertBf16Fixtures.ToBf16(SilkBertBf16Fixtures.Checkpoint(BertGpuNaming.DistilBert));

            using var trainer = new SilkDistilBertTrainer(runtime, config, bf16Tensors, BertGpuNaming.DistilBert,
                maxBatch: 4, maxSeqLen: 16);

            var (tokens, mask, labels) = SilkBertBf16Fixtures.TrainingBatch(samples: 64, seed: 37);

            var epochs = trainer.Train(tokens, mask, labels, batchSize: 4, seqLen: SilkBertBf16Fixtures.SeqLen, epochs: 4,
                learningRate: 0.1f, weightDecay: 0.01f, seed: 41);

            Assert.That(epochs.Count, Is.EqualTo(4));
            foreach (var e in epochs) Assert.That(float.IsFinite(e.Loss), Is.True);
            Assert.That(epochs[^1].Loss, Is.LessThan(0.75f * epochs[0].Loss));
        }
    }
}
