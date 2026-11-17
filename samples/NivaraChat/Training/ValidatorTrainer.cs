using Nivara;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Training;
using Nivara.Metrics;
using Nivara.Samples;
using Nivara.Silk;
using NivaraChat.Data;

namespace NivaraChat.Training;

public static class ValidatorTrainer
{
    public static (TextClassifierModel<float> model, TextTokenizer tokenizer) Train(
        int epochs = 20, int batchSize = 32, int numSamples = 1000, string saveDir = "models", int seed = 42, bool useSilk = false)
    {
        var (inputs, labels) = SyntheticDataGenerator.GenerateValidatorData(numSamples, seed);
        var tokenizer = TextTokenizer.FromDocuments(inputs, maxVocabSize: 5000);

        int maxSeqLen = 40;
        int trainCount = (int)(inputs.Length * 0.8);
        var trainInputs = inputs.AsSpan(0, trainCount).ToArray();
        var trainLabels = labels.AsSpan(0, trainCount).ToArray();
        var testInputs = inputs.AsSpan(trainCount).ToArray();
        var testLabels = labels.AsSpan(trainCount).ToArray();

        var trainTokens = new int[trainCount * maxSeqLen];
        for (int i = 0; i < trainCount; i++)
        {
            var encoded = tokenizer.Encode(trainInputs[i], fixedLength: maxSeqLen);
            Array.Copy(encoded, 0, trainTokens, i * maxSeqLen, maxSeqLen);
        }

        if (useSilk)
        {
            GpuTextLinearClassifier? silk = null;
            try { silk = new GpuTextLinearClassifier(tokenizer.VocabSize, 32, 64, seed, 2); }
            catch (InvalidOperationException ex) { Console.WriteLine($"  Silk/OpenCL unavailable ({ex.Message}); using CPU trainer."); }
            if (silk is not null)
            {
                Console.WriteLine("  Training validator on Silk (device forward/backward)...");
                var silkEpochs = silk.Train(trainTokens, trainLabels, batchSize, maxSeqLen, epochs, 0.001f, seed);
                foreach (var e in silkEpochs) Console.WriteLine($"    epoch {e.Epoch}: loss={e.Loss:F4} ({e.Elapsed.TotalSeconds:F1}s)");

                var layout = silk.ExportCpuLayout();
                using var silkModel = new TextClassifierModel<float>(tokenizer.VocabSize, 32, 64, numClasses: 2, maxSeqLen);
                silkModel.SetWeights(layout.Embedding, layout.Fc1W, layout.Fc1B, layout.Fc2W, layout.Fc2B);

                var silkPredicted = new int[testLabels.Length];
                var silkActual = (int[])testLabels.Clone();
                for (int i = 0; i < testLabels.Length; i++)
                {
                    var encoded = tokenizer.Encode(testInputs[i], fixedLength: maxSeqLen);
                    var preds = silkModel.Predict(encoded);
                    silkPredicted[i] = preds[0];
                }
                var silkMatrix = NivaraMetrics.ConfusionMatrixFrame(
                    NivaraColumn<int>.Create(silkPredicted),
                    NivaraColumn<int>.Create(silkActual));
                long silkCorrect = 0;
                for (int r = 0; r < silkMatrix.RowCount; r++) silkCorrect += silkMatrix.GetColumn<long>("Correct")[r];
                Console.WriteLine($"  Test accuracy: {(double)silkCorrect / testLabels.Length:P1}");

                Directory.CreateDirectory(saveDir);
                ModelSerializer.Save(silkModel, Path.Combine(saveDir, "validator_model.json"));
                tokenizer.Save(Path.Combine(saveDir, "validator_tokenizer.json"));
                Console.WriteLine($"  Saved to {saveDir}/");
                silk.Dispose();
                return (silkModel, tokenizer);
            }
        }

        using var model = new TextClassifierModel<float>(tokenizer.VocabSize, 32, 64, numClasses: 2, maxSeqLen);
        using var optimizer = new Adam<float>(learningRate: 0.001f);
        optimizer.AddParameterGroup(model.GetParameters().Values);

        var lossFn = new CrossEntropyLoss<float>();
        var trainFrame = FrameBuilder.BuildDocumentClassificationFrame(trainTokens, trainLabels, trainCount, maxSeqLen);
        var featureColumns = Enumerable.Range(0, maxSeqLen).Select(d => $"tok_{d}").ToArray();
        var trainDataset = new TensorDataset<float>(trainFrame, featureColumns, ["label"]);
        var trainLoader = new DataLoader<float>(trainDataset, batchSize, shuffle: true, seed: seed);

        Console.WriteLine($"  Training validator: {trainCount} samples, {tokenizer.VocabSize} vocab, 2 classes");

        var trainLoop = new TrainingLoop<float>(
            model, trainLoader,
            (logits, lbls) =>
            {
                int bs = logits.Length / 2;
                var targets = new int[bs];
                for (int i = 0; i < bs; i++)
                    targets[i] = int.CreateChecked(lbls.Data[i]);
                return lossFn.Forward(logits, targets);
            },
            optimizer, epochs);

        var result = trainLoop.Run();

        var predictedLabels = new int[testLabels.Length];
        var actualLabels = (int[])testLabels.Clone();
        for (int i = 0; i < testLabels.Length; i++)
        {
            var encoded = tokenizer.Encode(testInputs[i], fixedLength: maxSeqLen);
            var preds = model.Predict(encoded);
            predictedLabels[i] = preds[0];
        }

        var accuracyMatrix = NivaraMetrics.ConfusionMatrixFrame(
            NivaraColumn<int>.Create(predictedLabels),
            NivaraColumn<int>.Create(actualLabels));
        long correct = 0;
        for (int r = 0; r < accuracyMatrix.RowCount; r++)
            correct += accuracyMatrix.GetColumn<long>("Correct")[r];

        Console.WriteLine($"  Test accuracy: {(double)correct / testLabels.Length:P1}");

        Directory.CreateDirectory(saveDir);
        ModelSerializer.Save(model, Path.Combine(saveDir, "validator_model.json"));
        tokenizer.Save(Path.Combine(saveDir, "validator_tokenizer.json"));
        Console.WriteLine($"  Saved to {saveDir}/");

        return (model, tokenizer);
    }
}
