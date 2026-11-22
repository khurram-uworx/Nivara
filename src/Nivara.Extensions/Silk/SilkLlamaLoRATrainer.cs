using Nivara.AutoDiff.Nn;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace Nivara.Silk;

[Flags]
public enum SilkLlamaLoRATarget
{
    None = 0,
    Q = 1,
    K = 2,
    V = 4,
    O = 8,
    Gate = 16,
    Up = 32,
    Down = 64,
    FFN = Gate | Up | Down,
}

/// <summary>
/// Silk/OpenCL LoRA trainer for Llama-family causal LMs (Qwen2.5/SmolLM).
/// Supports LoRA forward, prompt-masked cross-entropy backward, transformer
/// backward, Adam update, and adapter bundle save/load. CPU/Silk parity is
/// confirmed by controlled synthetic GPU-gated tests.
/// </summary>
public sealed class SilkLlamaLoRATrainer : IDisposable
{
    public sealed record Options(
        int VocabSize,
        int HiddenDim,
        int IntermediateDim,
        int NumLayers,
        int NumHeads,
        int NumKvHeads,
        int MaxSeqLen,
        float RmsNormEps = 1e-6f,
        float RopeTheta = 10000f,
        int Rank = 8,
        float Alpha = 16f,
        SilkLlamaLoRATarget Targets = SilkLlamaLoRATarget.Q | SilkLlamaLoRATarget.V);

    sealed class LoraPair
    {
        public required nint A; // [in, rank]
        public required nint B; // [rank, out]
        public required nint GradA;
        public required nint GradB;
        public required nint MA;
        public required nint VA;
        public required nint MB;
        public required nint VB;
        public required int InFeatures;
        public required int OutFeatures;
    }

    sealed class LayerWeights
    {
        public required nint InputNormW;
        public required nint PostNormW;
        public required nint QW;
        public required nint KW;
        public required nint VW;
        public required nint OW;
        public required nint GateW;
        public required nint UpW;
        public required nint DownW;
        public nint QBias;
        public nint KBias;
        public nint VBias;
        public bool HasQkvBias;
        public LoraPair? QLora;
        public LoraPair? KLora;
        public LoraPair? VLora;
        public LoraPair? OLora;
        public LoraPair? GateLora;
        public LoraPair? UpLora;
        public LoraPair? DownLora;
    }

    sealed class LayerActivations
    {
        public required nint Input;
        public required nint Normed1;
        public required nint Q;
        public required nint K;
        public required nint V;
        public required nint AttnOut;
        public required nint AttnProj;
        public required nint PostAttn;
        public required nint Normed2;
        public required nint Gate;
        public required nint Up;
        public required nint Mlp;
    }

    readonly SilkRuntime runtime;
    readonly Options options;
    readonly int headDim;
    readonly int qDim;
    readonly int kvDim;
    readonly float loraScale;
    readonly float attnScale;
    readonly List<nint> ownedBuffers = new();
    readonly LayerWeights[] layers;
    readonly LayerActivations[] activations;
    readonly RotaryEmbedding<float> rotary;

    nint tokEmb;
    nint tokEmbTransposed;
    nint finalNormW;
    nint xBuf;
    nint normedBuf;
    nint finalNormInputBuf;
    nint logitsBuf;
    nint dLogitsBuf;
    nint dCurBuf;
    nint dTmpBuf;
    nint dLnAccBuf; // [maxSeqLen, hiddenDim] for dgamma accumulation
    nint targetIdsBuf;
    nint cosBuf;
    nint sinBuf;
    nint posIdsBuf;
    nint loraTmpBuf; // [maxSeqLen, rank]
    nint loraOutBuf; // [maxSeqLen, max(outDim)]
    nint loraDxBuf;  // [maxSeqLen, max(inDim)] scratch for LoRA backward
    nint dQBuf;
    nint dKBuf;
    nint dVBuf;
    nint dAttnOutBuf;
    nint attnProbsBuf;
    nint attnDeltaBuf;
    nint dGateBuf;
    nint dUpBuf;
    nint dMlpBuf;
    nint loraGradABuf; // [max(in,rank)] scratch
    nint loraGradBBuf; // [max(rank,out)] scratch
    int ropeTableRows;
    int adamStep;
    int[]? posIdHost;

    bool disposed;

    public SilkLlamaLoRATrainer(SilkRuntime runtime, Options options)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(options);
        if (options.Rank <= 0) throw new ArgumentOutOfRangeException(nameof(options.Rank));
        this.runtime = runtime;
        this.options = options;
        headDim = options.HiddenDim / options.NumHeads;
        qDim = options.NumHeads * headDim;
        kvDim = options.NumKvHeads * headDim;
        loraScale = options.Alpha / options.Rank;
        attnScale = 1f / MathF.Sqrt(headDim);
        rotary = new RotaryEmbedding<float>(headDim, options.MaxSeqLen, options.RopeTheta);
        layers = new LayerWeights[options.NumLayers];
        activations = new LayerActivations[options.NumLayers];
        for (int i = 0; i < layers.Length; i++)
            layers[i] = new LayerWeights { InputNormW = 0, PostNormW = 0, QW = 0, KW = 0, VW = 0, OW = 0, GateW = 0, UpW = 0, DownW = 0 };

        AllocateCommonBuffers();
        AllocateLayerActivations();
    }

    public int MaxSeqLen => options.MaxSeqLen;

    public void SetLoraWeights(int layerIndex, SilkLlamaLoRATarget target, float[] aData, float[] bData)
    {
        ThrowIfDisposed();
        var layer = layers[layerIndex];
        var pair = GetLoraPair(layer, target) ?? throw new ArgumentException($"No LoRA for {target} on layer {layerIndex}.");
        if (aData.Length != pair.InFeatures * options.Rank) throw new ArgumentException("A size mismatch.", nameof(aData));
        if (bData.Length != options.Rank * pair.OutFeatures) throw new ArgumentException("B size mismatch.", nameof(bData));
        runtime.Upload(pair.A, aData);
        runtime.Upload(pair.B, bData);
    }

    public (float[] A, float[] B) DownloadLoraWeights(int layerIndex, SilkLlamaLoRATarget target)
    {
        ThrowIfDisposed();
        var layer = layers[layerIndex];
        var pair = GetLoraPair(layer, target) ?? throw new ArgumentException($"No LoRA for {target} on layer {layerIndex}.");
        return (runtime.Download(pair.A, pair.InFeatures * options.Rank), runtime.Download(pair.B, options.Rank * pair.OutFeatures));
    }

    public (float[] GradA, float[] GradB) DownloadLoraGrads(int layerIndex, SilkLlamaLoRATarget target)
    {
        ThrowIfDisposed();
        var layer = layers[layerIndex];
        var pair = GetLoraPair(layer, target) ?? throw new ArgumentException($"No LoRA for {target} on layer {layerIndex}.");
        return (runtime.Download(pair.GradA, pair.InFeatures * options.Rank), runtime.Download(pair.GradB, options.Rank * pair.OutFeatures));
    }

    public void SaveAdapterBundle(string path, string baseModelId, string template)
    {
        ThrowIfDisposed();
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var entries = EnumerateLoraPairs().Select((p, i) => new LoraSftAdapterEntry
        {
            Index = i,
            Layer = p.LayerIndex,
            Target = p.Target,
            AShape = new[] { p.Pair.InFeatures, options.Rank },
            BShape = new[] { options.Rank, p.Pair.OutFeatures },
            A = Convert.ToBase64String(MemoryMarshal.AsBytes(runtime.Download(p.Pair.A, p.Pair.InFeatures * options.Rank).AsSpan())),
            B = Convert.ToBase64String(MemoryMarshal.AsBytes(runtime.Download(p.Pair.B, options.Rank * p.Pair.OutFeatures).AsSpan())),
        }).ToArray();

        var bundle = new LoraSftAdapterBundle
        {
            BaseModelId = baseModelId,
            Template = template,
            Rank = options.Rank,
            Alpha = options.Alpha,
            Targets = options.Targets.ToString(),
            TrainableParams = entries.Sum(e => e.AShape.Aggregate(1, (a, b) => a * b) + e.BShape.Aggregate(1, (a, b) => a * b)),
            Adapters = entries,
        };
        LoraSftAdapterBundle.Save(path, bundle);
    }

    public void LoadAdapterBundle(string path)
    {
        ThrowIfDisposed();
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var bundle = LoraSftAdapterBundle.Load(path);
        if (bundle.Rank != options.Rank)
            throw new InvalidOperationException("Rank mismatch.");
        if (bundle.Alpha != options.Alpha)
            throw new InvalidOperationException("Alpha mismatch.");

        var adapters = EnumerateLoraPairs().ToList();
        var entries = bundle.Adapters;
        if (entries.Length != adapters.Count)
            throw new InvalidOperationException($"Adapter count mismatch: file has {entries.Length}, trainer has {adapters.Count}.");

        for (int i = 0; i < adapters.Count; i++)
        {
            var entry = entries[i];
            var pair = adapters[i].Pair;
            var aBytes = Convert.FromBase64String(entry.A);
            var bBytes = Convert.FromBase64String(entry.B);
            var a = MemoryMarshal.Cast<byte, float>(aBytes).ToArray();
            var b = MemoryMarshal.Cast<byte, float>(bBytes).ToArray();
            if (a.Length != pair.InFeatures * options.Rank || b.Length != options.Rank * pair.OutFeatures)
                throw new InvalidOperationException($"Adapter {i} shape mismatch.");
            runtime.Upload(pair.A, a);
            runtime.Upload(pair.B, b);
        }
    }

    IEnumerable<(int LayerIndex, string Target, LoraPair Pair)> EnumerateLoraPairs()
    {
        for (int i = 0; i < options.NumLayers; i++)
        {
            var layer = layers[i];
            if (layer.QLora is not null) yield return (i, "Q", layer.QLora);
            if (layer.KLora is not null) yield return (i, "K", layer.KLora);
            if (layer.VLora is not null) yield return (i, "V", layer.VLora);
            if (layer.OLora is not null) yield return (i, "O", layer.OLora);
            if (layer.GateLora is not null) yield return (i, "Gate", layer.GateLora);
            if (layer.UpLora is not null) yield return (i, "Up", layer.UpLora);
            if (layer.DownLora is not null) yield return (i, "Down", layer.DownLora);
        }
    }

    static LoraPair? GetLoraPair(LayerWeights layer, SilkLlamaLoRATarget target) => target switch
    {
        SilkLlamaLoRATarget.Q => layer.QLora,
        SilkLlamaLoRATarget.K => layer.KLora,
        SilkLlamaLoRATarget.V => layer.VLora,
        SilkLlamaLoRATarget.O => layer.OLora,
        SilkLlamaLoRATarget.Gate => layer.GateLora,
        SilkLlamaLoRATarget.Up => layer.UpLora,
        SilkLlamaLoRATarget.Down => layer.DownLora,
        _ => throw new ArgumentException($"Single target expected, got {target}.", nameof(target)),
    };

    public void UploadBaseWeights(Dictionary<string, (float[] Data, int[] Shape)> tensors)
    {
        ThrowIfDisposed();
        tokEmb = Upload(tensors, "model.embed_tokens.weight");
        tokEmbTransposed = UploadTransposed(tensors, "model.embed_tokens.weight");
        finalNormW = Upload(tensors, "model.norm.weight");

        for (int i = 0; i < options.NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            var layer = layers[i];
            layer.InputNormW = Upload(tensors, $"{p}.input_layernorm.weight");
            layer.PostNormW = Upload(tensors, $"{p}.post_attention_layernorm.weight");
            layer.QW = UploadTransposed(tensors, $"{p}.self_attn.q_proj.weight");
            layer.KW = UploadTransposed(tensors, $"{p}.self_attn.k_proj.weight");
            layer.VW = UploadTransposed(tensors, $"{p}.self_attn.v_proj.weight");
            layer.OW = UploadTransposed(tensors, $"{p}.self_attn.o_proj.weight");
            layer.GateW = UploadTransposed(tensors, $"{p}.mlp.gate_proj.weight");
            layer.UpW = UploadTransposed(tensors, $"{p}.mlp.up_proj.weight");
            layer.DownW = UploadTransposed(tensors, $"{p}.mlp.down_proj.weight");

            if (tensors.TryGetValue($"{p}.self_attn.q_proj.bias", out var qb))
            {
                layer.QBias = UploadRaw(qb.Data);
                layer.KBias = UploadRaw(tensors[$"{p}.self_attn.k_proj.bias"].Data);
                layer.VBias = UploadRaw(tensors[$"{p}.self_attn.v_proj.bias"].Data);
                layer.HasQkvBias = true;
            }

            AttachLoraIfRequested(layer);
        }
    }

    /// <summary>Runs forward pass and returns logits [seqLen, vocabSize].</summary>
    public float[] Forward(int[] tokenIds)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(tokenIds);
        int seqLen = tokenIds.Length;
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(tokenIds));
        if (seqLen > options.MaxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(tokenIds), $"seqLen {seqLen} exceeds MaxSeqLen {options.MaxSeqLen}.");

        ForwardDevice(tokenIds, seqLen);
        return runtime.Download(logitsBuf, seqLen * options.VocabSize);
    }

    /// <summary>Runs forward pass and returns last-token logits [vocabSize].</summary>
    public float[] ForwardLastLogits(int[] tokenIds)
    {
        var all = Forward(tokenIds);
        int seqLen = tokenIds.Length;
        var last = new float[options.VocabSize];
        Array.Copy(all, (seqLen - 1) * options.VocabSize, last, 0, options.VocabSize);
        return last;
    }

    /// <summary>
    /// Computes prompt-masked cross-entropy loss and runs backward through LM head
    /// and final RMSNorm. Returns (loss, validTokenCount). dCurBuf holds dX at finalNorm input.
    /// </summary>
    public (float Loss, int ValidCount) BackwardPromptMaskedCrossEntropy(int[] labels, int seqLen)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(labels);
        if (seqLen <= 0 || seqLen > options.MaxSeqLen) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (labels.Length < seqLen) throw new ArgumentException("labels too short.", nameof(labels));

        int valid = 0;
        for (int i = 0; i < seqLen; i++)
            if (labels[i] != -100) valid++;
        if (valid == 0) throw new ArgumentException("No valid labels (all -100).", nameof(labels));

        // Loss on host (logits already in logitsBuf from forward)
        var logits = runtime.Download(logitsBuf, seqLen * options.VocabSize);
        float loss = ComputeMaskedCrossEntropy(logits, labels, seqLen, options.VocabSize);

        float scale = 1f / valid;
        runtime.Upload(targetIdsBuf, labels.AsSpan(0, seqLen));
        // dLogits = softmax(logits) - onehot(labels), scaled, with ignoreIndex
        SilkLauncher.SoftmaxCrossEntropyBackward(runtime, logitsBuf, targetIdsBuf, dLogitsBuf,
            seqLen, options.VocabSize, scale, -100);

        // LM head backward: logits = normed @ tokEmbTransposed
        // dNormed = dLogits @ tokEmb  (tokEmb is [vocab, hidden])
        SilkLauncher.Gemm(runtime, dLogitsBuf, tokEmb, dTmpBuf, seqLen, options.VocabSize, options.HiddenDim);

        // Final RMSNorm backward: dTmpBuf is dNormed, finalNormInputBuf is x, finalNormW is gamma
        // dCurBuf = dX, dLnAccBuf = dGammaAcc (ignored; base frozen)
        SilkLauncher.RmsNormBackward(runtime, dTmpBuf, finalNormInputBuf, finalNormW, dCurBuf, dLnAccBuf,
            seqLen, options.HiddenDim, options.RmsNormEps);

        return (loss, valid);
    }

    public void ZeroLoraGradients()
    {
        ThrowIfDisposed();
        for (int i = 0; i < options.NumLayers; i++)
            ZeroLayerLoraGrads(layers[i]);
    }

    public void AdamUpdateLora(float learningRate, float weightDecay = 0f, float beta1 = 0.9f, float beta2 = 0.999f)
    {
        ThrowIfDisposed();
        adamStep++;
        float c1 = 1f - MathF.Pow(beta1, adamStep);
        float c2 = 1f - MathF.Pow(beta2, adamStep);
        for (int i = 0; i < options.NumLayers; i++)
            AdamLayerLora(layers[i], learningRate, c1, c2, weightDecay);
    }

    /// <summary>Runs forward, prompt-masked CE backward, full transformer backward, and LoRA Adam update.</summary>
    public float TrainStep(int[] inputIds, int[] labels, float learningRate)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(inputIds);
        ArgumentNullException.ThrowIfNull(labels);
        int seqLen = inputIds.Length;
        if (seqLen <= 0 || seqLen > options.MaxSeqLen) throw new ArgumentOutOfRangeException(nameof(inputIds));
        if (labels.Length < seqLen) throw new ArgumentException("labels too short.", nameof(labels));

        ZeroLoraGradients();
        ForwardDevice(inputIds, seqLen);
        var (loss, _) = BackwardPromptMaskedCrossEntropy(labels, seqLen);
        BackwardTransformerBlocks(seqLen);
        AdamUpdateLora(learningRate);
        return loss;
    }

    /// <summary>Backward through all transformer blocks. Assumes dCurBuf holds grad at finalNorm input.</summary>
    public void BackwardTransformerBlocks(int seqLen)
    {
        ThrowIfDisposed();
        int h = options.HiddenDim;
        int inter = options.IntermediateDim;

        for (int li = options.NumLayers - 1; li >= 0; li--)
        {
            var layer = layers[li];
            var act = activations[li];

            // dCurBuf = dOut (grad into block output). Residual: dPostAttn += dOut, dMlpProj = dOut.
            // Save dOut in dTmpBuf as dMlpProj source; dPostAttn starts as copy of dOut.
            runtime.CopyBuffer(dCurBuf, dTmpBuf, 0, 0, seqLen * h); // dTmpBuf = dMlpProj/dOut

            // Backward through down proj: dMlp = dOut @ DownW^T (+ LoRA)
            BackwardLinearWithLora(act.Mlp, layer.DownW, layer.DownLora, dTmpBuf, dMlpBuf, seqLen, inter, h, addToDx: false);

            // SwiGLU backward: gate,up -> dGate,dUp
            SilkLauncher.SwiGluBackward(runtime, dMlpBuf, act.Gate, act.Up, dGateBuf, dUpBuf, seqLen, inter);

            // Backward through up/gate projs into normed2 grad. Accumulate both into dTmpBuf(normed2-grad).
            BackwardLinearWithLora(act.Normed2, layer.UpW, layer.UpLora, dUpBuf, dTmpBuf, seqLen, h, inter, addToDx: false);
            BackwardLinearWithLora(act.Normed2, layer.GateW, layer.GateLora, dGateBuf, dTmpBuf, seqLen, h, inter, addToDx: true);

            // RMSNorm backward on post-attention norm: input=PostAttn, gamma=PostNormW, dNormed=dTmpBuf
            SilkLauncher.RmsNormBackward(runtime, dTmpBuf, act.PostAttn, layer.PostNormW, dMlpBuf, dLnAccBuf, seqLen, h, options.RmsNormEps);
            // dMlpBuf now holds dPostAttn from FFN path. Add residual dOut.
            SilkLauncher.Add(runtime, dMlpBuf, dCurBuf, dMlpBuf, seqLen * h);

            // Attention residual split: dMlpBuf is dPostAttn total. dX_in_attn_res = dPostAttn, dAttnProj = dPostAttn.
            runtime.CopyBuffer(dMlpBuf, dTmpBuf, 0, 0, seqLen * h); // dTmpBuf = dAttnProj

            // Backward through O proj
            BackwardLinearWithLora(act.AttnOut, layer.OW, layer.OLora, dTmpBuf, dAttnOutBuf, seqLen, qDim, h, addToDx: false);

            // Attention backward: dAttnOutBuf is [seq,qDim]
            SilkLauncher.CausalGqaAttentionBackwardPrep(runtime, act.Q, act.K, act.V, dAttnOutBuf, attnProbsBuf, attnDeltaBuf,
                seqLen, options.NumHeads, options.NumKvHeads, headDim, attnScale);
            SilkLauncher.CausalGqaAttentionBackwardDq(runtime, act.K, act.V, dAttnOutBuf, attnProbsBuf, attnDeltaBuf, dQBuf,
                seqLen, options.NumHeads, options.NumKvHeads, headDim, attnScale);
            SilkLauncher.CausalGqaAttentionBackwardDkv(runtime, act.Q, act.V, dAttnOutBuf, attnProbsBuf, attnDeltaBuf, dKBuf, dVBuf,
                seqLen, options.NumHeads, options.NumKvHeads, headDim, attnScale);

            // RoPE backward (inverse rotation)
            SilkLauncher.RotaryBackward(runtime, dQBuf, posIdsBuf, cosBuf, sinBuf, dQBuf, seqLen, options.NumHeads, headDim, headDim / 2);
            SilkLauncher.RotaryBackward(runtime, dKBuf, posIdsBuf, cosBuf, sinBuf, dKBuf, seqLen, options.NumKvHeads, headDim, headDim / 2);

            // Backward through Q/K/V projs into normed1 grad
            BackwardLinearWithLora(act.Normed1, layer.QW, layer.QLora, dQBuf, dTmpBuf, seqLen, h, qDim, addToDx: false);
            BackwardLinearWithLora(act.Normed1, layer.KW, layer.KLora, dKBuf, dTmpBuf, seqLen, h, kvDim, addToDx: true);
            BackwardLinearWithLora(act.Normed1, layer.VW, layer.VLora, dVBuf, dTmpBuf, seqLen, h, kvDim, addToDx: true);

            // Input RMSNorm backward
            SilkLauncher.RmsNormBackward(runtime, dTmpBuf, act.Input, layer.InputNormW, dCurBuf, dLnAccBuf, seqLen, h, options.RmsNormEps);
            // Add residual from post-attn total
            SilkLauncher.Add(runtime, dCurBuf, dMlpBuf, dCurBuf, seqLen * h);
        }
    }

    void BackwardLinearWithLora(nint x, nint baseW, LoraPair? lora, nint dY, nint dX, int rows, int inFeatures, int outFeatures, bool addToDx)
    {
        // Base path dX contribution: dY @ W^T
        if (addToDx)
        {
            SilkLauncher.GemmNtResidual(runtime, dY, baseW, dX, dX, rows, outFeatures, inFeatures);
        }
        else
        {
            SilkLauncher.GemmNt(runtime, dY, baseW, dX, rows, outFeatures, inFeatures);
        }

        if (lora is null) return;

        // dLoraOut = loraScale * dY
        SilkLauncher.Scale(runtime, dY, loraOutBuf, loraScale, rows * outFeatures);

        // dB += loraTmp^T @ dLoraOut, loraTmp = x @ A
        SilkLauncher.Gemm(runtime, x, lora.A, loraTmpBuf, rows, inFeatures, options.Rank);
        SilkLauncher.GemmTn(runtime, loraTmpBuf, loraOutBuf, loraGradBBuf, options.Rank, rows, outFeatures);
        SilkLauncher.Add(runtime, lora.GradB, loraGradBBuf, lora.GradB, options.Rank * outFeatures);

        // dLoraTmp = dLoraOut @ B^T
        SilkLauncher.GemmNt(runtime, loraOutBuf, lora.B, loraTmpBuf, rows, outFeatures, options.Rank);

        // dA += x^T @ dLoraTmp
        SilkLauncher.GemmTn(runtime, x, loraTmpBuf, loraGradABuf, inFeatures, rows, options.Rank);
        SilkLauncher.Add(runtime, lora.GradA, loraGradABuf, lora.GradA, inFeatures * options.Rank);

        // dX += dLoraTmp @ A^T
        SilkLauncher.GemmNtResidual(runtime, loraTmpBuf, lora.A, dX, dX, rows, options.Rank, inFeatures);
    }

    void ZeroLayerLoraGrads(LayerWeights layer)
    {
        ZeroLoraPair(layer.QLora);
        ZeroLoraPair(layer.KLora);
        ZeroLoraPair(layer.VLora);
        ZeroLoraPair(layer.OLora);
        ZeroLoraPair(layer.GateLora);
        ZeroLoraPair(layer.UpLora);
        ZeroLoraPair(layer.DownLora);
    }

    void ZeroLoraPair(LoraPair? p)
    {
        if (p is null) return;
        SilkLauncher.Fill(runtime, p.GradA, 0f, p.InFeatures * options.Rank);
        SilkLauncher.Fill(runtime, p.GradB, 0f, options.Rank * p.OutFeatures);
    }

    void AdamLayerLora(LayerWeights layer, float lr, float c1, float c2, float weightDecay)
    {
        AdamLoraPair(layer.QLora, lr, c1, c2, weightDecay);
        AdamLoraPair(layer.KLora, lr, c1, c2, weightDecay);
        AdamLoraPair(layer.VLora, lr, c1, c2, weightDecay);
        AdamLoraPair(layer.OLora, lr, c1, c2, weightDecay);
        AdamLoraPair(layer.GateLora, lr, c1, c2, weightDecay);
        AdamLoraPair(layer.UpLora, lr, c1, c2, weightDecay);
        AdamLoraPair(layer.DownLora, lr, c1, c2, weightDecay);
    }

    void AdamLoraPair(LoraPair? p, float lr, float c1, float c2, float weightDecay)
    {
        if (p is null) return;
        SilkLauncher.AdamUpdate(runtime, p.A, p.GradA, p.MA, p.VA, lr, c1, c2, 1e-8f, weightDecay, p.InFeatures * options.Rank);
        SilkLauncher.AdamUpdate(runtime, p.B, p.GradB, p.MB, p.VB, lr, c1, c2, 1e-8f, weightDecay, options.Rank * p.OutFeatures);
    }

    static float ComputeMaskedCrossEntropy(float[] logits, int[] labels, int seqLen, int vocab)
    {
        float sum = 0;
        int valid = 0;
        for (int t = 0; t < seqLen; t++)
        {
            int label = labels[t];
            if (label < 0 || label >= vocab) continue;
            float max = float.NegativeInfinity;
            int off = t * vocab;
            for (int v = 0; v < vocab; v++) max = MathF.Max(max, logits[off + v]);
            float s = 0;
            for (int v = 0; v < vocab; v++) s += MathF.Exp(logits[off + v] - max);
            float logp = logits[off + label] - max - MathF.Log(s);
            sum += -logp;
            valid++;
        }
        return valid == 0 ? 0f : sum / valid;
    }

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        foreach (var buf in ownedBuffers)
        {
            if (buf != 0)
                runtime.Release(buf);
        }
        ownedBuffers.Clear();
    }

    void ForwardDevice(int[] tokenIds, int seqLen)
    {
        EnsureRopeTables(seqLen);
        EnsurePosIds(seqLen);
        runtime.Upload(posIdsBuf, posIdHost.AsSpan(0, seqLen));
        runtime.Upload(targetIdsBuf, tokenIds.AsSpan(0, seqLen)); // reuse as id buffer
        SilkLauncher.Gather(runtime, tokEmb, targetIdsBuf, xBuf, seqLen * options.HiddenDim, options.HiddenDim);

        for (int li = 0; li < options.NumLayers; li++)
        {
            var layer = layers[li];
            var act = activations[li];

            runtime.CopyBuffer(xBuf, act.Input, 0, 0, seqLen * options.HiddenDim);

            SilkLauncher.RmsNorm(runtime, xBuf, layer.InputNormW, act.Normed1, seqLen, options.HiddenDim, options.RmsNormEps);

            GemmWithLora(act.Normed1, layer.QW, layer.QLora, act.Q, seqLen, options.HiddenDim, qDim);
            GemmWithLora(act.Normed1, layer.KW, layer.KLora, act.K, seqLen, options.HiddenDim, kvDim);
            GemmWithLora(act.Normed1, layer.VW, layer.VLora, act.V, seqLen, options.HiddenDim, kvDim);
            if (layer.HasQkvBias)
            {
                SilkLauncher.AddBias(runtime, act.Q, layer.QBias, act.Q, seqLen, qDim);
                SilkLauncher.AddBias(runtime, act.K, layer.KBias, act.K, seqLen, kvDim);
                SilkLauncher.AddBias(runtime, act.V, layer.VBias, act.V, seqLen, kvDim);
            }

            SilkLauncher.Rotary(runtime, act.Q, posIdsBuf, cosBuf, sinBuf, seqLen, options.NumHeads, headDim, headDim / 2);
            SilkLauncher.Rotary(runtime, act.K, posIdsBuf, cosBuf, sinBuf, seqLen, options.NumKvHeads, headDim, headDim / 2);

            SilkLauncher.CausalGqaAttention(runtime, act.Q, act.K, act.V, act.AttnOut, seqLen, options.NumHeads, options.NumKvHeads, headDim, attnScale);

            GemmWithLora(act.AttnOut, layer.OW, layer.OLora, act.AttnProj, seqLen, qDim, options.HiddenDim);
            SilkLauncher.Add(runtime, xBuf, act.AttnProj, xBuf, seqLen * options.HiddenDim);

            runtime.CopyBuffer(xBuf, act.PostAttn, 0, 0, seqLen * options.HiddenDim);

            SilkLauncher.RmsNorm(runtime, xBuf, layer.PostNormW, act.Normed2, seqLen, options.HiddenDim, options.RmsNormEps);

            GemmWithLora(act.Normed2, layer.GateW, layer.GateLora, act.Gate, seqLen, options.HiddenDim, options.IntermediateDim);
            GemmWithLora(act.Normed2, layer.UpW, layer.UpLora, act.Up, seqLen, options.HiddenDim, options.IntermediateDim);
            SilkLauncher.SwiGlu(runtime, act.Gate, act.Up, act.Mlp, seqLen, options.IntermediateDim);

            GemmWithLora(act.Mlp, layer.DownW, layer.DownLora, act.AttnProj, seqLen, options.IntermediateDim, options.HiddenDim);
            SilkLauncher.Add(runtime, xBuf, act.AttnProj, xBuf, seqLen * options.HiddenDim);
        }

        runtime.CopyBuffer(xBuf, finalNormInputBuf, 0, 0, seqLen * options.HiddenDim);
        SilkLauncher.RmsNorm(runtime, xBuf, finalNormW, normedBuf, seqLen, options.HiddenDim, options.RmsNormEps);
        SilkLauncher.Gemm(runtime, normedBuf, tokEmbTransposed, logitsBuf, seqLen, options.HiddenDim, options.VocabSize);
    }

    void GemmWithLora(nint input, nint baseWeight, LoraPair? lora, nint output, int rows, int inFeatures, int outFeatures)
    {
        SilkLauncher.Gemm(runtime, input, baseWeight, output, rows, inFeatures, outFeatures);
        if (lora is null) return;

        SilkLauncher.Gemm(runtime, input, lora.A, loraTmpBuf, rows, inFeatures, options.Rank);
        SilkLauncher.Gemm(runtime, loraTmpBuf, lora.B, loraOutBuf, rows, options.Rank, outFeatures);
        SilkLauncher.Scale(runtime, loraOutBuf, loraOutBuf, loraScale, rows * outFeatures);
        SilkLauncher.Add(runtime, output, loraOutBuf, output, rows * outFeatures);
    }

    void AllocateCommonBuffers()
    {
        int cap = options.MaxSeqLen;
        int h = options.HiddenDim;
        int v = options.VocabSize;
        xBuf = Alloc(cap * h);
        normedBuf = Alloc(cap * h);
        finalNormInputBuf = Alloc(cap * h);
        logitsBuf = Alloc(cap * v);
        dLogitsBuf = Alloc(cap * v);
        dCurBuf = Alloc(cap * h);
        dTmpBuf = Alloc(cap * h);
        dLnAccBuf = Alloc(cap * h);
        targetIdsBuf = Alloc(cap);
        posIdsBuf = Alloc(cap);
        cosBuf = Alloc(cap * headDim / 2);
        sinBuf = Alloc(cap * headDim / 2);
        loraTmpBuf = Alloc(cap * options.Rank);
        int maxOut = Math.Max(h, Math.Max(qDim, Math.Max(kvDim, options.IntermediateDim)));
        loraOutBuf = Alloc(cap * maxOut);
        loraDxBuf = Alloc(cap * maxOut);
        dQBuf = Alloc(cap * qDim);
        dKBuf = Alloc(cap * kvDim);
        dVBuf = Alloc(cap * kvDim);
        dAttnOutBuf = Alloc(cap * qDim);
        attnProbsBuf = Alloc(cap * options.NumHeads * cap);
        attnDeltaBuf = Alloc(cap * options.NumHeads);
        dGateBuf = Alloc(cap * options.IntermediateDim);
        dUpBuf = Alloc(cap * options.IntermediateDim);
        dMlpBuf = Alloc(cap * options.IntermediateDim);
        int maxIn = Math.Max(h, Math.Max(qDim, Math.Max(kvDim, options.IntermediateDim)));
        loraGradABuf = Alloc(maxIn * options.Rank);
        loraGradBBuf = Alloc(options.Rank * Math.Max(maxOut, maxIn));
    }

    void AllocateLayerActivations()
    {
        int cap = options.MaxSeqLen;
        int h = options.HiddenDim;
        for (int i = 0; i < options.NumLayers; i++)
        {
            activations[i] = new LayerActivations
            {
                Input = Alloc(cap * h),
                Normed1 = Alloc(cap * h),
                Q = Alloc(cap * qDim),
                K = Alloc(cap * kvDim),
                V = Alloc(cap * kvDim),
                AttnOut = Alloc(cap * qDim),
                AttnProj = Alloc(cap * h),
                PostAttn = Alloc(cap * h),
                Normed2 = Alloc(cap * h),
                Gate = Alloc(cap * options.IntermediateDim),
                Up = Alloc(cap * options.IntermediateDim),
                Mlp = Alloc(cap * options.IntermediateDim),
            };
        }
    }

    void AttachLoraIfRequested(LayerWeights layer)
    {
        if (options.Targets.HasFlag(SilkLlamaLoRATarget.Q)) layer.QLora = CreateLora(options.HiddenDim, qDim);
        if (options.Targets.HasFlag(SilkLlamaLoRATarget.K)) layer.KLora = CreateLora(options.HiddenDim, kvDim);
        if (options.Targets.HasFlag(SilkLlamaLoRATarget.V)) layer.VLora = CreateLora(options.HiddenDim, kvDim);
        if (options.Targets.HasFlag(SilkLlamaLoRATarget.O)) layer.OLora = CreateLora(qDim, options.HiddenDim);
        if (options.Targets.HasFlag(SilkLlamaLoRATarget.FFN))
        {
            layer.GateLora = CreateLora(options.HiddenDim, options.IntermediateDim);
            layer.UpLora = CreateLora(options.HiddenDim, options.IntermediateDim);
            layer.DownLora = CreateLora(options.IntermediateDim, options.HiddenDim);
        }
    }

    LoraPair CreateLora(int inFeatures, int outFeatures)
    {
        int aLen = inFeatures * options.Rank;
        int bLen = options.Rank * outFeatures;
        var a = Alloc(aLen);
        var b = Alloc(bLen);
        var gradA = Alloc(aLen);
        var gradB = Alloc(bLen);
        var mA = Alloc(aLen);
        var vA = Alloc(aLen);
        var mB = Alloc(bLen);
        var vB = Alloc(bLen);

        var rng = new Random(42);
        var aHost = new float[aLen];
        float bound = MathF.Sqrt(6f / inFeatures);
        for (int i = 0; i < aHost.Length; i++)
            aHost[i] = (float)(rng.NextDouble() * 2.0 * bound - bound);
        runtime.Upload(a, aHost);
        runtime.Upload(b, new float[bLen]);
        runtime.Upload(gradA, new float[aLen]);
        runtime.Upload(gradB, new float[bLen]);
        runtime.Upload(mA, new float[aLen]);
        runtime.Upload(vA, new float[aLen]);
        runtime.Upload(mB, new float[bLen]);
        runtime.Upload(vB, new float[bLen]);

        return new LoraPair
        {
            A = a,
            B = b,
            GradA = gradA,
            GradB = gradB,
            MA = mA,
            VA = vA,
            MB = mB,
            VB = vB,
            InFeatures = inFeatures,
            OutFeatures = outFeatures
        };
    }

    void EnsureRopeTables(int seqLen)
    {
        if (ropeTableRows >= seqLen) return;
        int half = headDim / 2;
        var cos = new float[seqLen * half];
        var sin = new float[seqLen * half];
        rotary.GetPositionTables(0, seqLen, out var cosMem, out var sinMem);
        cosMem.CopyTo(cos);
        sinMem.CopyTo(sin);
        runtime.Upload(cosBuf, cos.AsSpan(0, seqLen * half));
        runtime.Upload(sinBuf, sin.AsSpan(0, seqLen * half));
        ropeTableRows = seqLen;
    }

    void EnsurePosIds(int seqLen)
    {
        if (posIdHost is not null && posIdHost.Length >= seqLen) return;
        posIdHost = new int[seqLen];
        for (int i = 0; i < seqLen; i++) posIdHost[i] = i;
    }

    nint Alloc(int elements)
    {
        var h = runtime.Allocate(elements);
        ownedBuffers.Add(h);
        return h;
    }

    nint Upload(Dictionary<string, (float[] Data, int[] Shape)> tensors, string name)
    {
        if (!tensors.TryGetValue(name, out var t))
            throw new KeyNotFoundException($"Missing tensor '{name}'.");
        return UploadRaw(t.Data);
    }

    nint UploadRaw(float[] data)
    {
        var h = Alloc(data.Length);
        runtime.Upload(h, data);
        return h;
    }

    nint UploadTransposed(Dictionary<string, (float[] Data, int[] Shape)> tensors, string name)
    {
        if (!tensors.TryGetValue(name, out var t))
            throw new KeyNotFoundException($"Missing tensor '{name}'.");
        if (t.Shape.Length != 2) throw new InvalidOperationException($"Tensor '{name}' must be 2D.");
        int rows = t.Shape[0];
        int cols = t.Shape[1];
        var transposed = new float[rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                transposed[c * rows + r] = t.Data[r * cols + c];
        var h = Alloc(transposed.Length);
        runtime.Upload(h, transposed);
        return h;
    }

    void ThrowIfDisposed()
    {
        if (disposed) throw new ObjectDisposedException(nameof(SilkLlamaLoRATrainer));
    }
}
