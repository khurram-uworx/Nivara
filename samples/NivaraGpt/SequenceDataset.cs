using Nivara;

namespace NivaraGpt;

public static class SequenceDataset
{
    public static NivaraFrame BuildFrame(
        Tokenizer tokenizer,
        List<string> docs,
        int blockSize)
    {
        var allTokens = new List<int>();
        foreach (var doc in docs)
        {
            var tokens = tokenizer.Encode(doc);
            allTokens.AddRange(tokens);
            allTokens.Add(tokenizer.EOS);
        }

        int nSamples = (allTokens.Count - 1) / blockSize;
        var allInputs = new float[nSamples * blockSize];
        var allTargets = new float[nSamples * blockSize];

        for (int i = 0; i < nSamples; i++)
        {
            for (int j = 0; j < blockSize; j++)
            {
                int idx = i * blockSize + j;
                allInputs[idx] = allTokens[idx];
                allTargets[idx] = allTokens[idx + 1];
            }
        }

        var inputCol = NivaraColumn<float>.Create(allInputs);
        var targetCol = NivaraColumn<float>.Create(allTargets);
        return NivaraFrame.Create(("input_ids", inputCol), ("target_ids", targetCol));
    }
}
