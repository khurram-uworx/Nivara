namespace NivaraFineTuning;

public sealed class Sst2Example
{
    public string Sentence { get; init; } = "";
    public int Label { get; init; }
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
            .Skip(1) // header
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
}
