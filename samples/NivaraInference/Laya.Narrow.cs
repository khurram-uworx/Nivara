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
        using Nivara.AI.ILayaGpuSession session = Nivara.AI.GpuSeamFactories.LayaBf16(config, tensors, agent.HeadLayers);
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
        using Nivara.AI.ILayaGpuSession session = Nivara.AI.GpuSeamFactories.LayaFp16(config, tensors, agent.HeadLayers);
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

    public static int RunBenchmarkGpu(
        Dictionary<string, (Half[] Data, int[] Shape)> tensors,
        string modelDir,
        Nivara.AI.OpenLayaSessionFp16 open)
    {
        var agent = LoadAgentConfig(modelDir);
        var config = LoadEncoderConfig(modelDir);

        Console.WriteLine($"=== {ModelTypeName} Benchmark (GPU) ===");
        PrintSetup(config, agent);

        var buildSw = Stopwatch.StartNew();
        using Nivara.AI.ILayaGpuSession session = open(config, tensors, agent.HeadLayers);
        buildSw.Stop();
        Console.WriteLine($"Device: {session.DeviceName}, precision Half");
        Console.WriteLine($"GPU model build: {buildSw.ElapsedMilliseconds} ms");
        Console.WriteLine();
        Console.WriteLine(ParameterLine(tensors));
        Console.WriteLine();

        var tokenizer = LoadTokenizer(modelDir);
        var stateIds = tokenizer.Encode(FixtureState.Replace("[MASK]", " ", StringComparison.Ordinal));

        Console.WriteLine($"1 warmup + 3 timed passes at max_len {agent.MaxLen}, median reported.");
        Console.WriteLine();
        Console.WriteLine($"  {"question",-10} {"valid",5} {"markers",7} {"median",10} {"min",8} {"max",8}");

        foreach (var question in FixtureQuestions)
        {
            var sequence = LayaPromptBuilder.BuildSequence(
                tokenizer, FixtureState, question, agent.MaxLen, agent.HeadMaxLen, stateIds: stateIds);
            var (ids, validLength) = PadTo(sequence.Ids, agent.MaxLen, config.PadTokenId);

            session.Forward(ids, validLength, question.Type, sequence.Markers);

            const int iterations = 3;
            var samples = new long[iterations];
            for (int i = 0; i < iterations; i++)
            {
                var sw = Stopwatch.StartNew();
                session.Forward(ids, validLength, question.Type, sequence.Markers);
                sw.Stop();
                samples[i] = sw.ElapsedMilliseconds;
            }

            Array.Sort(samples);
            Console.WriteLine(
                $"  {question.Id,-10} {validLength,5} {sequence.Markers.Length,7} {samples[iterations / 2],8} ms {samples[0],6} ms {samples[^1],6} ms");
        }

        Console.WriteLine();
        Console.WriteLine("Encoder plus head, one pass each, no KV cache — cost scales with the padded");
        Console.WriteLine("sequence length, not with tokens generated. Run 'laya benchmark' for the CPU table.");
        return 0;
    }

    public static int CompareGpu(
        Dictionary<string, (Half[] Data, int[] Shape)> tensors,
        string modelDir,
        Nivara.AI.OpenLayaSessionFp16 open)
    {
        var agent = LoadAgentConfig(modelDir);
        var config = LoadEncoderConfig(modelDir);

        Console.WriteLine($"=== {ModelTypeName} GPU vs CPU Gate ===");
        Console.WriteLine("Reference: the in-process CPU encoder + head in native (F16 storage, F32 compute) mode.");
        Console.WriteLine();
        PrintSetup(config, agent);

        var cpuBuildSw = Stopwatch.StartNew();
        var (cpuEncoder, cpuHead, _) = LoadModels<float, Half>(tensors, config, agent);
        cpuBuildSw.Stop();
        Console.WriteLine($"CPU model load: {cpuBuildSw.ElapsedMilliseconds} ms");

        var buildSw = Stopwatch.StartNew();
        using Nivara.AI.ILayaGpuSession session = open(config, tensors, agent.HeadLayers);
        buildSw.Stop();
        Console.WriteLine($"Device: {session.DeviceName}");
        Console.WriteLine($"GPU model build: {buildSw.ElapsedMilliseconds} ms");
        Console.WriteLine(ParameterLine(tensors));
        Console.WriteLine();

        var tokenizer = LoadTokenizer(modelDir);
        var stateIds = tokenizer.Encode(FixtureState.Replace("[MASK]", " ", StringComparison.Ordinal));

        bool logitsOk = true;
        bool clampOk = true;
        double worstScaled = 0.0;
        string worstQuestion = "-";
        foreach (var question in FixtureQuestions)
        {
            var sequence = LayaPromptBuilder.BuildSequence(
                tokenizer, FixtureState, question, agent.MaxLen, agent.HeadMaxLen, stateIds: stateIds);
            var (ids, validLength) = PadTo(sequence.Ids, agent.MaxLen, config.PadTokenId);

            var cpuSw = Stopwatch.StartNew();
            var cpuHidden = cpuEncoder.Forward(ids, validLength);
            var cpuOutput = cpuHead.Forward(cpuHidden, question.Type, sequence.Markers, validLength);
            cpuSw.Stop();

            var gpuSw = Stopwatch.StartNew();
            var gpuOutput = session.Forward(ids, validLength, question.Type, sequence.Markers);
            gpuSw.Stop();

            var row = CompareQuestion(
                question.Id, sequence.Markers.Length, gpuOutput, cpuOutput,
                cpuSw.ElapsedMilliseconds, gpuSw.ElapsedMilliseconds);
            logitsOk &= row.ValuesOk;
            clampOk &= row.ClampOk;
            if (row.Scaled > worstScaled)
            {
                worstScaled = row.Scaled;
                worstQuestion = question.Id;
            }
        }

        Console.WriteLine();
        Console.WriteLine($"Value parity:  {(logitsOk ? "PASS" : "FAIL")}   " +
                          $"worst {worstScaled:E3} of the {GateRelTol:G} bound " +
                          $"({100.0 * worstScaled / GateRelTol:F1}% used, on {worstQuestion})");
        Console.WriteLine($"Mask clamp:   {(clampOk ? "PASS" : "FAIL (a fully-masked row produced a non-finite value)")}");

        if (!logitsOk || !clampOk) return 1;

        Console.WriteLine();
        Console.WriteLine("Gate passed. The Laya GPU encoder + decision head (F16 storage) matches the CPU native path.");
        return 0;
    }
}
