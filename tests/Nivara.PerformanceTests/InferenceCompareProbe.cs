using System.Diagnostics;
using Nivara.AI;
using NivaraInference;

namespace Nivara.PerformanceTests;

internal static class InferenceCompareProbe
{
    public static int Run(string[] args)
    {
        string? model = null;
        bool diag = false;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i].Equals("--inference-compare", StringComparison.OrdinalIgnoreCase))
            {
                if (i + 1 >= args.Length) { Console.Error.WriteLine("--inference-compare needs a value."); return 1; }
                model = args[++i]; diag = false;
            }
            else if (args[i].Equals("--inference-compare-diag", StringComparison.OrdinalIgnoreCase))
            {
                if (i + 1 >= args.Length) { Console.Error.WriteLine("--inference-compare-diag needs a value."); return 1; }
                model = args[++i]; diag = true;
            }
        }

        if (string.IsNullOrWhiteSpace(model))
        {
            Console.Error.WriteLine("Usage: --inference-compare <model> | --inference-compare-diag <model>");
            Console.Error.WriteLine("Models: mobilenet_v2, resnet18, minilm, distilbert, distilbert_sst, modernbert, laya");
            return 1;
        }

        string modelDir = Path.Combine("samples", "data", model);
        string modelPath = Path.Combine(modelDir, "model.safetensors");
        if (!File.Exists(modelPath))
        {
            Console.Error.WriteLine($"Model file not found: {modelPath}");
            return 1;
        }

        Console.WriteLine($"Loading weights from {modelPath}...");
        var tensors = SafeTensorsLoader.Read(modelPath);

        if (diag)
        {
            return model switch
            {
                "mobilenet_v2" => InferenceCompare.VisionCompareDiag(tensors, "mobilenet_v2"),
                "resnet18" => InferenceCompare.VisionCompareDiag(tensors, "resnet18"),
                "modernbert" => InferenceCompare.ModernBertCompareDiag(tensors, modelDir),
                _ => UnsupportedDiag(model),
            };
        }

        return model switch
        {
            "mobilenet_v2" => InferenceCompare.VisionCompare(tensors, "mobilenet_v2"),
            "resnet18" => InferenceCompare.VisionCompare(tensors, "resnet18"),
            "minilm" => InferenceCompare.MiniLMCompare(tensors),
            "distilbert" => InferenceCompare.DistilBertCompare(tensors),
            "distilbert_sst" => InferenceCompare.DistilBertSstCompare(tensors),
            "modernbert" => InferenceCompare.ModernBertCompare(tensors, modelDir),
            "laya" => InferenceCompare.LayaCompare(tensors, modelDir),
            _ => Unsupported(model),
        };
    }

    private static int Unsupported(string model)
    {
        Console.Error.WriteLine($"No CPU compare entry point for '{model}'.");
        return 1;
    }

    private static int UnsupportedDiag(string model)
    {
        Console.Error.WriteLine($"No compare_diag entry point for '{model}' (modernbert, mobilenet_v2, resnet18 only).");
        return 1;
    }
}
