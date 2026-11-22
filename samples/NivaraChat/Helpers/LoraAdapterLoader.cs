using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.Samples;
using Nivara.Silk;
using System.Numerics;

namespace NivaraChat.Helpers;

/// <summary>
/// Sample-local helper that loads a <see cref="LoraSftAdapterBundle"/> produced by
/// <c>NivaraFineTuning --mode ft-lora</c> into a CPU <see cref="LlamaForCausalLM{TModel,TWeight}"/>
/// by attaching LoRA adapters and copying the stored A/B matrices. This is the
/// consumption touchpoint for the FT showcase (issue #1419): it does not alter RAG modes.
/// </summary>
public static class LoraAdapterLoader
{
    /// <summary>
    /// Attaches LoRA adapters to <paramref name="model"/> according to the bundle header and
    /// copies the stored A/B matrices into them. Validates rank, alpha, count, and shapes.
    /// </summary>
    public static void Apply<TWeight>(
        LlamaForCausalLM<float, TWeight> model,
        string adapterPath)
        where TWeight : struct, IFloatingPointIeee754<TWeight>
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(adapterPath);

        var bundle = LoraSftAdapterBundle.Load(adapterPath);

        var targets = ParseTargets(bundle.Targets);
        int rank = bundle.Rank;
        float alpha = bundle.Alpha;

        var adapters = model.AttachLora(targets, rank, alpha);

        if (bundle.Adapters is null)
            throw new InvalidOperationException("Adapter bundle is missing the 'adapters' array.");

        if (rank > 0 && adapters.Count > 0 && bundle.Rank != adapters[0].Rank)
            throw new InvalidOperationException($"Adapter rank mismatch: file rank={bundle.Rank}, model rank={adapters[0].Rank}.");

        if (adapters.Count > 0 && bundle.Alpha != 0f && MathF.Abs(bundle.Alpha - adapters[0].Alpha) > 1e-6f)
            throw new InvalidOperationException($"Adapter alpha mismatch: file alpha={bundle.Alpha}, model alpha={adapters[0].Alpha}.");

        if (bundle.Adapters.Length != adapters.Count)
            throw new InvalidOperationException($"Adapter count mismatch: file has {bundle.Adapters.Length}, model has {adapters.Count}.");

        for (int i = 0; i < adapters.Count; i++)
        {
            var entry = bundle.Adapters[i];
            if (entry.Index >= 0 && entry.Index != i)
                throw new InvalidOperationException($"Adapter index mismatch at position {i}: file index={entry.Index}.");

            if (entry.AShape is { Length: 2 })
            {
                if (entry.AShape[0] != adapters[i].A.Tensor.Shape[0] || entry.AShape[1] != adapters[i].A.Tensor.Shape[1])
                    throw new InvalidOperationException($"Adapter {i} A shape mismatch.");
            }
            if (entry.BShape is { Length: 2 })
            {
                if (entry.BShape[0] != adapters[i].B.Tensor.Shape[0] || entry.BShape[1] != adapters[i].B.Tensor.Shape[1])
                    throw new InvalidOperationException($"Adapter {i} B shape mismatch.");
            }

            var aBytes = Convert.FromBase64String(entry.A);
            var bBytes = Convert.FromBase64String(entry.B);
            var a = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(aBytes).ToArray();
            var b = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(bBytes).ToArray();
            if (a.Length != adapters[i].A.Tensor.Length || b.Length != adapters[i].B.Tensor.Length)
                throw new InvalidOperationException($"Adapter {i} data length mismatch.");

            adapters[i].A.Tensor = ReverseGradTensor<float>.FromMatrix(a, adapters[i].A.Tensor.Shape[0], adapters[i].A.Tensor.Shape[1], requiresGrad: true);
            adapters[i].B.Tensor = ReverseGradTensor<float>.FromMatrix(b, adapters[i].B.Tensor.Shape[0], adapters[i].B.Tensor.Shape[1], requiresGrad: true);
        }
    }

    static LlamaLoRATarget ParseTargets(string targets)
    {
        var result = LlamaLoRATarget.None;
        foreach (var part in targets.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            result |= part.ToLowerInvariant() switch
            {
                "q" => LlamaLoRATarget.Q,
                "k" => LlamaLoRATarget.K,
                "v" => LlamaLoRATarget.V,
                "o" => LlamaLoRATarget.O,
                "ffn" => LlamaLoRATarget.FFN,
                _ => throw new ArgumentException($"Unknown LoRA target '{part}'. Expected q,k,v,o,ffn.")
            };
        }
        if (result == LlamaLoRATarget.None) result = LlamaLoRATarget.Q | LlamaLoRATarget.V;
        return result;
    }
}
