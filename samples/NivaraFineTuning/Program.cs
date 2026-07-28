using Nivara.Samples;
using NivaraFineTuning;

var modelPath = FindPath("distilbert-base-uncased");
var dataPath = FindPath("sst2");

if (modelPath == null)
{
    Console.Error.WriteLine("DistilBERT model weights not found. Run Python/download_model.py first.");
    return 1;
}

if (dataPath == null)
{
    Console.Error.WriteLine("SST-2 data not found. Run Python/download_data.py first.");
    return 1;
}

Console.WriteLine($"Model: {modelPath}");
Console.WriteLine($"Data:  {dataPath}");

var dataset = Sst2Dataset.Load(dataPath);
Console.WriteLine($"Train examples: {dataset.Train.Count}");
Console.WriteLine($"Dev examples:   {dataset.Dev.Count}");
return 0;

static string? FindPath(string name)
{
    var dir = AppContext.BaseDirectory;
    for (int i = 0; i < 5; i++)
    {
        dir = Path.GetDirectoryName(dir);
        if (dir == null) break;
        var candidate = Path.Combine(dir, "data", name);
        if (Directory.Exists(candidate)) return candidate;
    }
    return null;
}
