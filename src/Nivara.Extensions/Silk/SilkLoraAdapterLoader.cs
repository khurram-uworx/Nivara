using System.Runtime.InteropServices;

namespace Nivara.Silk;

public static class SilkLoraAdapterLoader
{
    public readonly record struct ExpectedSlot(int Layer, SilkLlamaLoRATarget Target, int InFeatures, int OutFeatures);
    public readonly record struct ParsedAdapter(float[] A, float[] B);

    public static IReadOnlyList<ExpectedSlot> BuildExpectedSlots(SilkLlamaLoRATarget targets, int numLayers, int hiddenDim, int qDim, int kvDim, int intermediateDim)
    {
        var slots = new List<ExpectedSlot>();
        for (int i = 0; i < numLayers; i++)
        {
            if (targets.HasFlag(SilkLlamaLoRATarget.Q)) slots.Add(new(i, SilkLlamaLoRATarget.Q, hiddenDim, qDim));
            if (targets.HasFlag(SilkLlamaLoRATarget.K)) slots.Add(new(i, SilkLlamaLoRATarget.K, hiddenDim, kvDim));
            if (targets.HasFlag(SilkLlamaLoRATarget.V)) slots.Add(new(i, SilkLlamaLoRATarget.V, hiddenDim, kvDim));
            if (targets.HasFlag(SilkLlamaLoRATarget.O)) slots.Add(new(i, SilkLlamaLoRATarget.O, qDim, hiddenDim));
            if (targets.HasFlag(SilkLlamaLoRATarget.Gate)) slots.Add(new(i, SilkLlamaLoRATarget.Gate, hiddenDim, intermediateDim));
            if (targets.HasFlag(SilkLlamaLoRATarget.Up)) slots.Add(new(i, SilkLlamaLoRATarget.Up, hiddenDim, intermediateDim));
            if (targets.HasFlag(SilkLlamaLoRATarget.Down)) slots.Add(new(i, SilkLlamaLoRATarget.Down, intermediateDim, hiddenDim));
        }
        return slots;
    }

    public static SilkLlamaLoRATarget ParseTargets(string targets)
    {
        var result = SilkLlamaLoRATarget.None;
        foreach (var raw in targets.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var part = raw.Trim();
            if (part.Contains('.', StringComparison.Ordinal)) part = part[(part.LastIndexOf('.') + 1)..];
            result |= part.ToLowerInvariant() switch
            {
                "q" => SilkLlamaLoRATarget.Q,
                "k" => SilkLlamaLoRATarget.K,
                "v" => SilkLlamaLoRATarget.V,
                "o" => SilkLlamaLoRATarget.O,
                "gate" => SilkLlamaLoRATarget.Gate,
                "up" => SilkLlamaLoRATarget.Up,
                "down" => SilkLlamaLoRATarget.Down,
                "ffn" => SilkLlamaLoRATarget.FFN,
                _ => throw new ArgumentException($"Unknown LoRA target '{raw}'. Expected q,k,v,o,gate,up,down,ffn.")
            };
        }
        if (result == SilkLlamaLoRATarget.None) result = SilkLlamaLoRATarget.Q | SilkLlamaLoRATarget.V;
        return result;
    }

    public static ParsedAdapter[] ValidateAndParse(LoraSftAdapterBundle bundle, IReadOnlyList<ExpectedSlot> expected)
    {
        ArgumentNullException.ThrowIfNull(bundle);
        ArgumentNullException.ThrowIfNull(expected);

        if (bundle.Rank <= 0)
            throw new InvalidOperationException($"Invalid adapter bundle rank {bundle.Rank}.");
        if (!float.IsFinite(bundle.Alpha) || bundle.Alpha <= 0f)
            throw new InvalidOperationException($"Invalid adapter bundle alpha {bundle.Alpha}.");
        if (bundle.Adapters is null)
            throw new InvalidOperationException("Adapter bundle is missing the 'adapters' array.");

        if (bundle.Adapters.Length != expected.Count)
            throw new InvalidOperationException($"Adapter count mismatch: file has {bundle.Adapters.Length}, model has {expected.Count}.");

        var parsed = new ParsedAdapter[expected.Count];
        for (int i = 0; i < expected.Count; i++)
        {
            var entry = bundle.Adapters[i];
            var slot = expected[i];

            if (entry.Index >= 0 && entry.Index != i)
                throw new InvalidOperationException($"Adapter index mismatch at position {i}: file index={entry.Index}.");

            if (entry.AShape is { Length: 2 } && (entry.AShape[0] != slot.InFeatures || entry.AShape[1] != bundle.Rank))
                throw new InvalidOperationException($"Adapter {i} A shape mismatch.");
            if (entry.BShape is { Length: 2 } && (entry.BShape[0] != bundle.Rank || entry.BShape[1] != slot.OutFeatures))
                throw new InvalidOperationException($"Adapter {i} B shape mismatch.");

            var aBytes = Convert.FromBase64String(entry.A ?? "");
            var bBytes = Convert.FromBase64String(entry.B ?? "");
            var a = MemoryMarshal.Cast<byte, float>(aBytes).ToArray();
            var b = MemoryMarshal.Cast<byte, float>(bBytes).ToArray();
            if (a.Length != slot.InFeatures * bundle.Rank || b.Length != bundle.Rank * slot.OutFeatures)
                throw new InvalidOperationException($"Adapter {i} data length mismatch.");

            parsed[i] = new ParsedAdapter(a, b);
        }

        return parsed;
    }
}
