using System.Diagnostics;

namespace NivaraInference;

class Program
{
    static int Main(string[] args)
    {
        Console.WriteLine("=== Nivara HuggingFace Inference ===");
        Console.WriteLine();

        string modelPath = args.Length > 0 ? args[0] : "";
        if (string.IsNullOrEmpty(modelPath))
        {
            Console.WriteLine("Usage: NivaraInference <path-to-model.safetensors>");
            Console.WriteLine();
            Console.WriteLine("Available models in samples/data/:");
            Console.WriteLine("  mobilenet_v2/model.safetensors  (~13.5 MB, 3.4M params)");
            Console.WriteLine("  resnet18/model.safetensors      (~44.6 MB, 11.7M params)");
            return 1;
        }

        if (!File.Exists(modelPath))
        {
            Console.Error.WriteLine($"Model file not found: {modelPath}");
            return 1;
        }

        var fileInfo = new FileInfo(modelPath);
        Console.WriteLine($"Loading weights from {fileInfo.Name} ({fileInfo.Length / (1024.0 * 1024.0):F1} MB)...");

        var sw = Stopwatch.StartNew();
        var tensors = SafeTensorsLoader.Read(modelPath);
        sw.Stop();
        Console.WriteLine($"Loaded {tensors.Count} tensors in {sw.ElapsedMilliseconds} ms");
        Console.WriteLine();

        Console.WriteLine("Tensor summary:");
        long totalParams = 0;
        foreach (var (name, (data, shape)) in tensors.OrderBy(kvp => kvp.Key))
        {
            Console.WriteLine($"  {name,-60} [{string.Join(", ", shape)}] {data.Length,10} params");
            totalParams += data.Length;
        }
        Console.WriteLine();
        Console.WriteLine($"Total: {totalParams:N0} parameters across {tensors.Count} tensors");

        return 0;
    }
}
