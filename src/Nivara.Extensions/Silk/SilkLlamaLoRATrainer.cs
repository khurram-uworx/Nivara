using Nivara.AutoDiff.Nn;
using System.Numerics;

namespace Nivara.Silk;

[Flags]
public enum SilkLlamaLoRATarget
{
    None = 0,
    Q = 1,
    K = 2,
    V = 4,
    O = 8,
    FFN = 16,
}

/// <summary>
/// D2a: Silk/OpenCL LoRA trainer for Llama-family causal LMs (Qwen2.5/SmolLM).
/// Goal: CPU/Silk parity for release readiness. Current implementation is an active
/// scaffold: it owns device buffers/LoRA parameters and exposes the training surface,
/// but the full backward chain and parity-verified train loop are still being completed.
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
        public LoraPair? QLora;
        public LoraPair? KLora;
        public LoraPair? VLora;
        public LoraPair? OLora;
        public LoraPair? GateLora;
        public LoraPair? UpLora;
        public LoraPair? DownLora;
    }

    readonly SilkRuntime runtime;
    readonly Options options;
    readonly int headDim;
    readonly float loraScale;
    readonly List<nint> ownedBuffers = new();
    readonly LayerWeights[] layers;

    nint tokEmb;
    nint tokEmbTransposed;
    nint finalNormW;
    nint xBuf;
    nint normedBuf;
    nint logitsBuf;
    nint dLogitsBuf;
    nint targetIdsBuf;
    nint cosBuf;
    nint sinBuf;
    nint posIdsBuf;

    bool disposed;

    public SilkLlamaLoRATrainer(SilkRuntime runtime, Options options)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(options);
        if (options.Rank <= 0) throw new ArgumentOutOfRangeException(nameof(options.Rank));
        this.runtime = runtime;
        this.options = options;
        headDim = options.HiddenDim / options.NumHeads;
        loraScale = options.Alpha / options.Rank;
        layers = new LayerWeights[options.NumLayers];
        for (int i = 0; i < layers.Length; i++)
            layers[i] = new LayerWeights { InputNormW = 0, PostNormW = 0, QW = 0, KW = 0, VW = 0, OW = 0, GateW = 0, UpW = 0, DownW = 0 };

        AllocateCommonBuffers();
    }

    public bool IsParityReady => false;

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

            AttachLoraIfRequested(layer);
        }
    }

    public float TrainStep(int[] inputIds, int[] labels, float learningRate)
    {
        ThrowIfDisposed();
        throw new NotImplementedException(
            "D2a Silk Llama LoRA backward/parity path is scaffolded but not complete yet. " +
            "CPU reference is available; Silk parity is required before release readiness.");
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

    void AllocateCommonBuffers()
    {
        int cap = options.MaxSeqLen;
        int h = options.HiddenDim;
        int v = options.VocabSize;
        xBuf = Alloc(cap * h);
        normedBuf = Alloc(cap * h);
        logitsBuf = Alloc(cap * v);
        dLogitsBuf = Alloc(cap * v);
        targetIdsBuf = Alloc(cap);
        posIdsBuf = Alloc(cap);
        cosBuf = Alloc(cap * headDim / 2);
        sinBuf = Alloc(cap * headDim / 2);
    }

    void AttachLoraIfRequested(LayerWeights layer)
    {
        if (options.Targets.HasFlag(SilkLlamaLoRATarget.Q)) layer.QLora = CreateLora(options.HiddenDim, options.HiddenDim);
        if (options.Targets.HasFlag(SilkLlamaLoRATarget.K)) layer.KLora = CreateLora(options.HiddenDim, options.NumKvHeads * headDim);
        if (options.Targets.HasFlag(SilkLlamaLoRATarget.V)) layer.VLora = CreateLora(options.HiddenDim, options.NumKvHeads * headDim);
        if (options.Targets.HasFlag(SilkLlamaLoRATarget.O)) layer.OLora = CreateLora(options.HiddenDim, options.HiddenDim);
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

        // A: Kaiming uniform-ish seeded by zeros+small pattern is not ideal; initialize on host for now.
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
            A = a, B = b, GradA = gradA, GradB = gradB,
            MA = mA, VA = vA, MB = mB, VB = vB,
            InFeatures = inFeatures, OutFeatures = outFeatures
        };
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
        var h = Alloc(t.Data.Length);
        runtime.Upload(h, t.Data);
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
