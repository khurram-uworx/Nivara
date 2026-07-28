using Nivara.Samples;
using Microsoft.ML.Tokenizers;

namespace NivaraFineTuning;

public sealed class Sst2Example
{
    public string Sentence { get; init; } = "";
    public int Label { get; init; }
}

public sealed class TokenizedExample
{
    public float[] TokenIds { get; init; } = [];
    public float[] AttentionMask { get; init; } = [];
    public int Label { get; init; }
}

public sealed class Sst2Batch
{
    public float[] TokenIds { get; init; } = [];
    public float[] AttentionMask { get; init; } = [];
    public int[] Labels { get; init; } = [];
    public int BatchSize { get; init; }
    public int SeqLen { get; init; }
}

public sealed class Sst2Dataset
{
    public List<Sst2Example> Train { get; }
    public List<Sst2Example> Dev { get; }

    Sst2Dataset(List<Sst2Example> train, List<Sst2Example> dev)
    {
        Train = train;
        Dev = dev;
    }

    public static Sst2Dataset Load(string dataDir)
    {
        var trainPath = Path.Combine(dataDir, "train.tsv");
        var devPath = Path.Combine(dataDir, "dev.tsv");

        if (!File.Exists(trainPath)) throw new FileNotFoundException($"SST-2 train.tsv not found at {trainPath}");
        if (!File.Exists(devPath)) throw new FileNotFoundException($"SST-2 dev.tsv not found at {devPath}");

        var train = ParseTsv(trainPath);
        var dev = ParseTsv(devPath);

        return new Sst2Dataset(train, dev);
    }

    static List<Sst2Example> ParseTsv(string path)
    {
        return File.ReadAllLines(path)
            .Skip(1)
            .Where(line => !string.IsNullOrWhiteSpace(line))
            .Select(line =>
            {
                var parts = line.Split('\t');
                return new Sst2Example
                {
                    Sentence = parts[0].Trim('"'),
                    Label = int.Parse(parts[1])
                };
            })
            .ToList();
    }

    public static List<TokenizedExample> Tokenize(
        BertTokenizer tokenizer,
        List<Sst2Example> entries,
        int maxLen = 128)
    {
        var results = new List<TokenizedExample>(entries.Count);
        foreach (var entry in entries)
        {
            var (tokenIds, mask, _) = MiniLMTokenizer.Encode(tokenizer, entry.Sentence, maxLen);
            results.Add(new TokenizedExample
            {
                TokenIds = tokenIds,
                AttentionMask = mask,
                Label = entry.Label
            });
        }
        return results;
    }

    public static IEnumerable<Sst2Batch> CreateBatches(
        List<TokenizedExample> data,
        int batchSize,
        bool shuffle = false,
        Random? rng = null)
    {
        rng ??= Random.Shared;
        var indices = Enumerable.Range(0, data.Count).ToList();
        if (shuffle)
        {
            for (int i = indices.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        int seqLen = data[0].TokenIds.Length;
        for (int start = 0; start < indices.Count; start += batchSize)
        {
            int currentBatchSize = Math.Min(batchSize, indices.Count - start);
            var tokenIds = new float[currentBatchSize * seqLen];
            var masks = new float[currentBatchSize * seqLen];
            var labels = new int[currentBatchSize];

            for (int i = 0; i < currentBatchSize; i++)
            {
                var example = data[indices[start + i]];
                Array.Copy(example.TokenIds, 0, tokenIds, i * seqLen, seqLen);
                Array.Copy(example.AttentionMask, 0, masks, i * seqLen, seqLen);
                labels[i] = example.Label;
            }

            yield return new Sst2Batch
            {
                TokenIds = tokenIds,
                AttentionMask = masks,
                Labels = labels,
                BatchSize = currentBatchSize,
                SeqLen = seqLen
            };
        }
    }
}
