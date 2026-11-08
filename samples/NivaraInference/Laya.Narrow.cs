using Nivara.AI;
using System.Diagnostics;
using System.Numerics;

namespace NivaraInference;

public static partial class Laya
{
    static string ParameterLine(Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors)
    {
        long parameters = tensors.Values.Sum(t => (long)t.Data.Length);
        return $"Parameters: {parameters:N0} ({parameters * 2.0 / (1024.0 * 1024.0):F1} MB as BFloat16)";
    }

    static string ParameterLine(Dictionary<string, (Half[] Data, int[] Shape)> tensors)
    {
        long parameters = tensors.Values.Sum(t => (long)t.Data.Length);
        return $"Parameters: {parameters:N0} ({parameters * 2.0 / (1024.0 * 1024.0):F1} MB as Half)";
    }

    public static int RunGpu(
        Dictionary<string, (BFloat16[] Data, int[] Shape)> tensors,
        string modelDir)
    {
        var agent = LoadAgentConfig(modelDir);
        var config = LoadEncoderConfig(modelDir);

        Console.WriteLine($"=== {ModelTypeName} Inference (GPU) ===");
        Console.WriteLine("Encoder and decision head both on the accelerator. BFloat16 native.");
        Console.WriteLine();
        PrintSetup(config, agent);

        var buildSw = Stopwatch.StartNew();
        using Nivara.AI.ILayaGpuSession session = Nivara.Silk.SilkLayaSession.Open(config, tensors, agent.HeadLayers);
        buildSw.Stop();
        Console.WriteLine($"Device: {session.DeviceName}, precision BFloat16");
        Console.WriteLine($"GPU model build: {buildSw.ElapsedMilliseconds} ms");
        Console.WriteLine(ParameterLine(tensors));
        Console.WriteLine();

        var tokenizer = LoadTokenizer(modelDir);
        var calibration = new LayaCalibration(agent.Temperature, agent.TemperatureByOptions);
        var stateIds = tokenizer.Encode(FixtureState.Replace("[MASK]", " ", StringComparison.Ordinal));

        Console.WriteLine($"State ({stateIds.Count} tokens): {Truncate(FixtureState, 72)}");
        Console.WriteLine();

        foreach (var question in FixtureQuestions)
        {
            var sequence = LayaPromptBuilder.BuildSequence(
                tokenizer, FixtureState, question, agent.MaxLen, agent.HeadMaxLen, stateIds: stateIds);

            var sw = Stopwatch.StartNew();
            var output = session.Forward(sequence.Ids, sequence.Length, question.Type, sequence.Markers);
            sw.Stop();

            var decision = calibration.Decode(question, output.Logits, output.ActionProbability);
            PrintDecision(question, sequence, decision, sw.ElapsedMilliseconds);
        }

        Console.WriteLine();
        Console.WriteLine("Run 'laya --silk compare' for the GPU-vs-CPU parity gate.");
        return 0;
    }

    public static int RunGpu(
        Dictionary<string, (Half[] Data, int[] Shape)> tensors,
        string modelDir)
    {
        var agent = LoadAgentConfig(modelDir);
        var config = LoadEncoderConfig(modelDir);

        Console.WriteLine($"=== {ModelTypeName} Inference (GPU) ===");
        Console.WriteLine("Encoder and decision head both on the accelerator. Half native.");
        Console.WriteLine();
        PrintSetup(config, agent);

        var buildSw = Stopwatch.StartNew();
        using Nivara.AI.ILayaGpuSession session = Nivara.Silk.SilkLayaSession.Open(config, tensors, agent.HeadLayers);
        buildSw.Stop();
        Console.WriteLine($"Device: {session.DeviceName}, precision Half");
        Console.WriteLine($"GPU model build: {buildSw.ElapsedMilliseconds} ms");
        Console.WriteLine(ParameterLine(tensors));
        Console.WriteLine();

        var tokenizer = LoadTokenizer(modelDir);
        var calibration = new LayaCalibration(agent.Temperature, agent.TemperatureByOptions);
        var stateIds = tokenizer.Encode(FixtureState.Replace("[MASK]", " ", StringComparison.Ordinal));

        Console.WriteLine($"State ({stateIds.Count} tokens): {Truncate(FixtureState, 72)}");
        Console.WriteLine();

        foreach (var question in FixtureQuestions)
        {
            var sequence = LayaPromptBuilder.BuildSequence(
                tokenizer, FixtureState, question, agent.MaxLen, agent.HeadMaxLen, stateIds: stateIds);

            var sw = Stopwatch.StartNew();
            var output = session.Forward(sequence.Ids, sequence.Length, question.Type, sequence.Markers);
            sw.Stop();

            var decision = calibration.Decode(question, output.Logits, output.ActionProbability);
            PrintDecision(question, sequence, decision, sw.ElapsedMilliseconds);
        }

        Console.WriteLine();
        Console.WriteLine("Run 'laya --silk compare' for the GPU-vs-CPU parity gate.");
        return 0;
    }
}