using Nivara.Samples.Incident;

var mode = args.Length > 0 ? args[0] : "";
string datasetPath = "./data/incident-lab";
string scenario = "A";
int scale = 1;
int chunkSize = 100_000;

for (int i = 1; i < args.Length; i++)
{
    if (args[i] == "--dataset" && i + 1 < args.Length) datasetPath = args[++i];
    if (args[i] == "--scenario" && i + 1 < args.Length) scenario = args[++i];
    if (args[i] == "--scale" && i + 1 < args.Length) scale = int.Parse(args[++i]);
    if (args[i] == "--chunk-size" && i + 1 < args.Length) chunkSize = int.Parse(args[++i]);
}

switch (mode)
{
    case "generate":
        Console.WriteLine($"Generating dataset for scenario {scenario} at scale {scale}...");
        Console.WriteLine($"Output: {datasetPath}");
        break;
    case "analyze":
        Console.WriteLine($"Analyzing dataset from {datasetPath} (scenario {scenario})...");
        break;
    case "replay":
        Console.WriteLine($"Replaying dataset from {datasetPath} (scenario {scenario}, chunk size {chunkSize})...");
        break;
    default:
        PrintUsage();
        break;
}

static void PrintUsage()
{
    Console.WriteLine("Nivara Incident Lab");
    Console.WriteLine();
    Console.WriteLine("Usage:");
    Console.WriteLine("  NivaraIncident.Cli generate  --dataset <path> --scenario <A|B|C|D> --scale <N>");
    Console.WriteLine("  NivaraIncident.Cli analyze   --dataset <path> --scenario <A|B|C|D> --stream --chunk-size <N>");
    Console.WriteLine("  NivaraIncident.Cli replay    --dataset <path> --scenario <A|B|C|D> --chunk-size <N>");
}
