using Nivara;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Serialization;
using Nivara.AutoDiff.Training;
using NivaraChat.Data;

namespace NivaraChat.Training;

public static class EntityTrainer
{
    static readonly string[] EntityClasses = ["O", "B-person", "B-org", "B-date", "B-location"];

    public static (TokenClassifierModel<float> model, TextTokenizer tokenizer) Train(
        int epochs = 20, int batchSize = 32, int numSamples = 1000, string saveDir = "models", int seed = 42)
    {
        var (texts, labelSequences) = SyntheticDataGenerator.GenerateEntityData(numSamples, seed);
        var tokenizer = TextTokenizer.FromDocuments(texts, maxVocabSize: 5000);

        int maxSeqLen = 20;
        int numClasses = EntityClasses.Length;
        int trainCount = (int)(texts.Length * 0.8);
        var trainTexts = texts.AsSpan(0, trainCount).ToArray();
        var trainLabels = labelSequences.AsSpan(0, trainCount).ToArray();
        var testTexts = texts.AsSpan(trainCount).ToArray();
        var testLabels = labelSequences.AsSpan(trainCount).ToArray();

        var trainTokens = new int[trainCount * maxSeqLen];
        var trainLabelIds = new int[trainCount * maxSeqLen];
        for (int i = 0; i < trainCount; i++)
        {
            var encoded = tokenizer.Encode(trainTexts[i], fixedLength: maxSeqLen, addBosEos: false);
            Array.Copy(encoded, 0, trainTokens, i * maxSeqLen, maxSeqLen);
            var labelTokens = trainLabels[i].Split(' ');
            for (int j = 0; j < maxSeqLen; j++)
            {
                int labelId = j < labelTokens.Length
                    ? Array.IndexOf(EntityClasses, labelTokens[j])
                    : 0;
                trainLabelIds[i * maxSeqLen + j] = labelId >= 0 ? labelId : 0;
            }
        }

        using var model = new TokenClassifierModel<float>(tokenizer.VocabSize, 32, 64, numClasses, maxSeqLen);
        using var optimizer = new Adam<float>(learningRate: 0.001f);
        optimizer.AddParameterGroup(model.GetParameters().Values);

        var lossFn = new CrossEntropyLoss<float>();
        var trainFrame = BuildFrame(trainTokens, trainLabelIds, trainCount, maxSeqLen);
        var featureColumns = Enumerable.Range(0, maxSeqLen).Select(d => $"tok_{d}").ToArray();
        var trainDataset = new TensorDataset<float>(trainFrame, featureColumns, Enumerable.Range(0, maxSeqLen).Select(d => $"lbl_{d}").ToArray());
        var trainLoader = new DataLoader<float>(trainDataset, batchSize, shuffle: true, seed: seed);

        Console.WriteLine($"  Training entity extractor: {trainCount} samples, {tokenizer.VocabSize} vocab, {numClasses} classes");

        var trainLoop = new TrainingLoop<float>(
            model, trainLoader,
            (logits, lbls) =>
            {
                int totalTokens = logits.Length / numClasses;
                var targets = new int[totalTokens];
                for (int i = 0; i < totalTokens; i++)
                    targets[i] = int.CreateChecked(lbls.Data[i]);
                return lossFn.Forward(logits, targets);
            },
            optimizer, epochs);

        var result = trainLoop.Run();

        Directory.CreateDirectory(saveDir);
        ModelSerializer.Save(model, Path.Combine(saveDir, "entity_model.json"));
        tokenizer.Save(Path.Combine(saveDir, "entity_tokenizer.json"));
        Console.WriteLine($"  Saved to {saveDir}/");

        return (model, tokenizer);
    }

    static NivaraFrame BuildFrame(int[] tokens, int[] labels, int count, int seqLen)
    {
        var columns = new List<(string Name, IColumn Column)>();
        for (int d = 0; d < seqLen; d++)
        {
            var tokData = new float[count];
            var lblData = new float[count];
            for (int i = 0; i < count; i++)
            {
                tokData[i] = tokens[i * seqLen + d];
                lblData[i] = labels[i * seqLen + d];
            }
            columns.Add(($"tok_{d}", NivaraColumn<float>.Create(tokData)));
            columns.Add(($"lbl_{d}", NivaraColumn<float>.Create(lblData)));
        }
        return NivaraFrame.Create(columns.ToArray());
    }
}
