using Nivara;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Training;
using NivaraChat.Data;

namespace NivaraChat.Training;

public static class ValidatorTrainer
{
    public static (TextClassifierModel<float> model, TextTokenizer tokenizer) Train(
        int epochs = 20, int batchSize = 32, int numSamples = 1000, string saveDir = "models", int seed = 42)
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

        using var model = new TextClassifierModel<float>(tokenizer.VocabSize, 32, 64, numClasses: 2, maxSeqLen);
        using var optimizer = new Adam<float>(learningRate: 0.001f);
        optimizer.AddParameterGroup(model.GetParameters().Values);

        var lossFn = new CrossEntropyLoss<float>();
        var trainFrame = BuildFrame(trainTokens, trainLabels, trainCount, maxSeqLen);
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

        int correct = 0;
        for (int i = 0; i < testLabels.Length; i++)
        {
            var encoded = tokenizer.Encode(testInputs[i], fixedLength: maxSeqLen);
            var preds = model.Predict(encoded);
            if (preds[0] == testLabels[i]) correct++;
        }
        Console.WriteLine($"  Test accuracy: {(double)correct / testLabels.Length:P1}");

        Directory.CreateDirectory(saveDir);
        ModelSerializer.Save(model, Path.Combine(saveDir, "validator_model.json"));
        tokenizer.Save(Path.Combine(saveDir, "validator_tokenizer.json"));
        Console.WriteLine($"  Saved to {saveDir}/");

        return (model, tokenizer);
    }

    static NivaraFrame BuildFrame(int[] tokens, int[] labels, int count, int seqLen)
    {
        var columns = new List<(string Name, IColumn Column)>();
        for (int d = 0; d < seqLen; d++)
        {
            var colData = new float[count];
            for (int i = 0; i < count; i++)
                colData[i] = tokens[i * seqLen + d];
            columns.Add(($"tok_{d}", NivaraColumn<float>.Create(colData)));
        }
        var labelData = new float[count];
        for (int i = 0; i < count; i++)
            labelData[i] = labels[i];
        columns.Add(("label", NivaraColumn<float>.Create(labelData)));
        return NivaraFrame.Create(columns.ToArray());
    }
}
