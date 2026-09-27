using ILGPU;
using ILGPU.Runtime;

namespace Nivara.Samples.Gpu;

/// <summary>Result of a BERT-family encoder GPU forward: the encoder's final hidden state
/// ([batch*seqLen, hiddenDim]) and, when the classification head weights are present
/// (pre_classifier/classifier keys), the [batch, numClasses] logits.</summary>
public sealed record BertEncoderGpuResult(float[] Hidden, float[]? Logits);

/// <summary>Weight-key naming style for a BERT-family encoder: HuggingFace 'distilbert.*' keys
/// (DistilBERT), or BERT-style keys already stripped of the 'bert.' module prefix during
/// provisioning (MiniLM — the sample weights hold 'embeddings.*' / 'encoder.layer.N.*' keys).</summary>
public enum BertGpuNaming
{
    DistilBert,
    Bert
}

/// <summary>
/// Resolves the model-specific weight key names to the roles the runner consumes (word/position
/// embeddings + embed LayerNorm, per-layer q/k/v/o projections, ffn1/ffn2, ln1/ln2). Both naming
/// styles map to the same layer roles, so the runner is config-driven across the BERT family.
/// </summary>
static class BertGpuKeys
{
    public static string Embeddings(BertGpuNaming naming)
        => naming == BertGpuNaming.Bert ? "embeddings" : "distilbert.embeddings";

    public static string Layer(BertGpuNaming naming, int index)
        => naming == BertGpuNaming.Bert
            ? $"encoder.layer.{index}"
            : $"distilbert.transformer.layer.{index}";

    public static string Attention(BertGpuNaming naming, int index, char role) => naming switch
    {
        BertGpuNaming.Bert => role switch
        {
            'q' => $"{Layer(naming, index)}.attention.self.query",
            'k' => $"{Layer(naming, index)}.attention.self.key",
            'v' => $"{Layer(naming, index)}.attention.self.value",
            'o' => $"{Layer(naming, index)}.attention.output.dense",
            _ => throw new ArgumentOutOfRangeException(nameof(role), role, "Expected q/k/v/o.")
        },
        _ => role switch
        {
            'q' => $"{Layer(naming, index)}.attention.q_lin",
            'k' => $"{Layer(naming, index)}.attention.k_lin",
            'v' => $"{Layer(naming, index)}.attention.v_lin",
            'o' => $"{Layer(naming, index)}.attention.out_lin",
            _ => throw new ArgumentOutOfRangeException(nameof(role), role, "Expected q/k/v/o.")
        }
    };

    public static string Ffn(BertGpuNaming naming, int index, int which) => naming switch
    {
        BertGpuNaming.Bert => which == 1
            ? $"{Layer(naming, index)}.intermediate.dense"
            : $"{Layer(naming, index)}.output.dense",
        _ => which == 1
            ? $"{Layer(naming, index)}.ffn.lin1"
            : $"{Layer(naming, index)}.ffn.lin2"
    };

    public static string LayerNorm(BertGpuNaming naming, int index, int which) => naming switch
    {
        BertGpuNaming.Bert => which == 1
            ? $"{Layer(naming, index)}.attention.output.LayerNorm"
            : $"{Layer(naming, index)}.output.LayerNorm",
        _ => which == 1
            ? $"{Layer(naming, index)}.sa_layer_norm"
            : $"{Layer(naming, index)}.output_layer_norm"
    };
}

/// <summary>
/// Sample-scoped BERT-family encoder GPU forward (docs/BERT-GPU.md §3, §6): uploads the
/// loader weight key set once (GEMM weights pre-transposed [out,in] → Bt [in,out] so C = A·Bt is
/// plain row-major, per the assessment layout note), then runs the batch forward with every
/// intermediate resident on the GPU and only the final hidden state / logits read back. Mirrors
/// the CPU AutoDiff forward exactly: embedding gather (word + position, plus the token-type
/// row 0 when the model ships token-type embeddings — BERT-naming — mirroring the CPU
/// includeTokenTypeEmbedding default), embed
/// LayerNorm, then per layer q/k/v projections → fused attention (scale, +-inf padding mask, row
/// softmax) → o-projection → residual + LN1 → fc1 → exact GELU → fc2 → residual + LN2; optional
/// pre_classifier → ReLU → classifier. Every projection GEMM runs a fused epilogue kernel
/// (bias added in the register accumulators; GELU/ReLU folded into the fc1 / pre-classifier
/// launches via the lean <see cref="GemmKernels.TiledGemmKernelRow4Bias"/> /
/// <see cref="GemmKernels.TiledGemmKernelRow4Gelu"/> / <see cref="GemmKernels.TiledGemmKernelRow4Relu"/>
/// siblings), so bias and activation no longer dispatch separate kernels (issue #437). Config-
/// and naming-driven so DistilBERT and MiniLM (BERT-style keys) share the runner.
/// </summary>
public sealed class BertEncoderGpuRunner : IDisposable
{
    const string PreWKey = "pre_classifier.weight";
    const string PreBKey = "pre_classifier.bias";
    const string ClsWKey = "classifier.weight";
    const string ClsBKey = "classifier.bias";

    readonly IlgpuRuntime runtime;
    readonly int hiddenDim;
    readonly int intermediateDim;
    readonly int numHeads;
    readonly int headDim;
    readonly int numLayers;
    readonly float eps;
    readonly float scale;
    readonly bool hasHead;

    readonly MemoryBuffer1D<float, Stride1D.Dense> wordEmb;
    readonly MemoryBuffer1D<float, Stride1D.Dense> posEmb;
    readonly MemoryBuffer1D<float, Stride1D.Dense> embLnW;
    readonly MemoryBuffer1D<float, Stride1D.Dense> embLnB;
    readonly MemoryBuffer1D<float, Stride1D.Dense> tokenTypeEmb;
    readonly bool hasTokenType;
    readonly LayerGpuWeights[] layers;
    readonly MemoryBuffer1D<float, Stride1D.Dense>? preW;
    readonly MemoryBuffer1D<float, Stride1D.Dense>? preB;
    readonly MemoryBuffer1D<float, Stride1D.Dense>? clsW;
    readonly MemoryBuffer1D<float, Stride1D.Dense>? clsB;

    MemoryBuffer1D<float, Stride1D.Dense> x = null!;
    MemoryBuffer1D<float, Stride1D.Dense> qkv = null!;
    MemoryBuffer1D<float, Stride1D.Dense> attn = null!;
    MemoryBuffer1D<float, Stride1D.Dense> h = null!;
    MemoryBuffer1D<float, Stride1D.Dense> f1 = null!;
    MemoryBuffer1D<float, Stride1D.Dense> clsOut = null!;
    MemoryBuffer1D<float, Stride1D.Dense> logits = null!;
    MemoryBuffer1D<float, Stride1D.Dense> maskBuf = null!;
    MemoryBuffer1D<int, Stride1D.Dense> idsBuf = null!;
    MemoryBuffer1D<int, Stride1D.Dense> posIdsBuf = null!;
    MemoryBuffer1D<int, Stride1D.Dense> clsIdsBuf = null!;
    int cachedPosSeqLen = -1;

    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<int>, ArrayView<float>, int> gather;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> embeddingSum;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, float> layerNorm;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, float> layerNormResidual;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, float> attention;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int> gemmBias;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int> gemmGelu;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int> gemmRelu;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int> gemmQkv;

    /// <summary>
    /// Creates the runner, uploads every weight from the loader tensors (GEMM weights
    /// transposed once), and JIT-compiles the kernel set. Throws if required encoder
    /// keys are missing; the head weights are optional.
    /// </summary>
    public BertEncoderGpuRunner(
        IlgpuRuntime runtime,
        BertConfig config,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        BertGpuNaming naming)
    {
        this.runtime = runtime;
        hiddenDim = config.HiddenSize;
        intermediateDim = config.IntermediateSize;
        numHeads = config.NumAttentionHeads;
        headDim = hiddenDim / numHeads;
        numLayers = config.NumHiddenLayers;
        eps = config.LayerNormEps;
        scale = (float)(1.0 / Math.Sqrt(headDim));

        var acc = runtime.Accelerator;
        GpuBuffers.ValidateAttentionLocalMemory(acc, headDim);
        var emb = BertGpuKeys.Embeddings(naming);

        var wordEmbT = GpuBuffers.Req(tensors, $"{emb}.word_embeddings.weight");
        wordEmb = GpuBuffers.Alloc(acc, wordEmbT.Data.Length);
        wordEmb.CopyFromCPU(wordEmbT.Data);
        var posEmbT = GpuBuffers.Req(tensors, $"{emb}.position_embeddings.weight");
        posEmb = GpuBuffers.Alloc(acc, posEmbT.Data.Length);
        posEmb.CopyFromCPU(posEmbT.Data);
        embLnW = GpuBuffers.Alloc(acc, hiddenDim);
        embLnW.CopyFromCPU(GpuBuffers.Req(tensors, $"{emb}.LayerNorm.weight").Data);
        embLnB = GpuBuffers.Alloc(acc, hiddenDim);
        embLnB.CopyFromCPU(GpuBuffers.Req(tensors, $"{emb}.LayerNorm.bias").Data);

        hasTokenType = tensors.ContainsKey($"{emb}.token_type_embeddings.weight");
        tokenTypeEmb = GpuBuffers.Alloc(acc, 2 * hiddenDim);
        if (hasTokenType)
            tokenTypeEmb.CopyFromCPU(GpuBuffers.Req(tensors, $"{emb}.token_type_embeddings.weight").Data);

        layers = new LayerGpuWeights[numLayers];
        for (int i = 0; i < numLayers; i++)
            layers[i] = new LayerGpuWeights(acc, tensors, naming, i);

        hasHead = tensors.ContainsKey(PreWKey) && tensors.ContainsKey(ClsWKey);
        if (hasHead)
        {
            preW = GpuBuffers.UploadTransposed(acc, tensors, PreWKey);
            preB = GpuBuffers.UploadPlain(acc, tensors, PreBKey);
            clsW = GpuBuffers.UploadTransposed(acc, tensors, ClsWKey);
            clsB = GpuBuffers.UploadPlain(acc, tensors, ClsBKey);
        }

        // Persistent activation workspace (caps: batch <= 8, seqLen <= 128 — the
        // scenario's actual maxima; per-call payload buffers grow on demand).
        int rowsCap = 8 * 128;
        x = GpuBuffers.Alloc(acc, rowsCap * hiddenDim);
        qkv = GpuBuffers.Alloc(acc, rowsCap * 3 * hiddenDim);
        attn = GpuBuffers.Alloc(acc, rowsCap * hiddenDim);
        h = GpuBuffers.Alloc(acc, rowsCap * hiddenDim);
        f1 = GpuBuffers.Alloc(acc, rowsCap * intermediateDim);
        clsOut = GpuBuffers.Alloc(acc, 8 * hiddenDim);
        logits = GpuBuffers.Alloc(acc, 8 * 2);
        maskBuf = GpuBuffers.Alloc(acc, rowsCap);
        idsBuf = GpuBuffers.AllocInt(acc, rowsCap);
        posIdsBuf = GpuBuffers.AllocInt(acc, rowsCap);
        clsIdsBuf = GpuBuffers.AllocInt(acc, 8);

        gather = acc.LoadKernel<ArrayView<float>, ArrayView<int>, ArrayView<float>, int>(ElementwiseKernels.Gather);
        embeddingSum = acc.LoadKernel<ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(
            ElementwiseKernels.EmbeddingSum);
        layerNorm = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, float>(ElementwiseKernels.LayerNorm1D);
        layerNormResidual = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, float>(
            ElementwiseKernels.LayerNormResidual1D);
        attention = acc.LoadKernel<
            ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, int, int, int, int, int, float>(
            AttentionKernels.BatchedAttention);
        gemmBias = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
            GemmKernels.TiledGemmKernelRow4Bias);
        gemmGelu = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
            GemmKernels.TiledGemmKernelRow4Gelu);
        gemmRelu = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
            GemmKernels.TiledGemmKernelRow4Relu);
        gemmQkv = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int>(
            GemmKernels.TiledGemmKernelRow4Qkv);

        // Weight uploads ran on the default stream; sync the device so they are
        // visible to the kernel launches (running on runtime.Stream) in Forward.
        runtime.Synchronize();
    }

    public string DeviceName => runtime.DeviceName;

    /// <summary>
    /// Runs the BERT-family encoder (and, when the head weights are present, the
    /// pre_classifier/ReLU/classifier head) for a padded batch of token IDs.
    /// Token IDs and the attention mask must be padded exactly like the CPU path
    /// (MiniLMTokenizer.Encode → mask 0 on padding positions; positional embedding ids
    /// are 0..seqLen-1 repeated per batch row). Returns the read-back hidden state
    /// [batch*seqLen, hiddenDim] and, if a head exists, the [batch, numClasses] logits.
    /// </summary>
    public BertEncoderGpuResult Forward(int[] tokenIds, float[] attentionMask, int batch, int seqLen)
    {
        int rows = batch * seqLen;
        if (tokenIds.Length != rows)
            throw new ArgumentException($"tokenIds.Length ({tokenIds.Length}) must equal batch*seqLen ({rows}).", nameof(tokenIds));
        if (attentionMask.Length != rows)
            throw new ArgumentException($"attentionMask.Length ({attentionMask.Length}) must equal batch*seqLen ({rows}).", nameof(attentionMask));
        if (batch > 8 || seqLen > 128)
            throw new ArgumentOutOfRangeException(nameof(batch),
                $"The BertEncoderGpuRunner workspace caps at batch<=8, seqLen<=128 (scenario's actual maxima: batch 1, seqLen 128); got batch={batch}, seqLen={seqLen}.");

        GpuBuffers.Ensure(ref maskBuf, rows, runtime.Accelerator);
        maskBuf.View.SubView(0, rows).CopyFromCPU(runtime.Stream, attentionMask);
        GpuBuffers.Ensure(ref idsBuf, rows, runtime.Accelerator);
        idsBuf.View.SubView(0, rows).CopyFromCPU(runtime.Stream, tokenIds);

        // posIds are seqLen-deterministic (row % seqLen): rebuild + re-upload only when
        // seqLen changes (or the workspace grows), so the repeat-forward benchmark path
        // skips the allocation and host->device copy (M2, issue #437).
        int posRows = batch * seqLen;
        if (cachedPosSeqLen != seqLen || posIdsBuf.Length < posRows)
        {
            GpuBuffers.Ensure(ref posIdsBuf, posRows, runtime.Accelerator);
            var posIds = new int[posRows];
            for (int i = 0; i < posRows; i++)
                posIds[i] = i % seqLen;
            posIdsBuf.View.SubView(0, posRows).CopyFromCPU(runtime.Stream, posIds);
            cachedPosSeqLen = seqLen;
        }

        var stream = runtime.Stream;

        // embeddings: x = word_emb(ids) + pos_emb(0..seqLen-1 per row) + token-type row 0 for
        // models that ship one (BERT-naming; mirrors the CPU includeTokenTypeEmbedding default —
        // all-zero token-type ids). Then embed LN. One fused launch (word/pos gathers + sum +
        // optional token-type row).
        EmbeddingSum1D(idsBuf.View, posIdsBuf.View, wordEmb.View, posEmb.View, tokenTypeEmb.View, x.View, hiddenDim, rows * hiddenDim, hasTokenType ? 1 : 0);
        LayerNorm1D(x.View, embLnW.View, embLnB.View, x.View, rows, hiddenDim);

        for (int i = 0; i < numLayers; i++)
        {
            var w = layers[i];

            GemmQkv(x.View, w.Wqkv.View, qkv.View, w.Bqkv.View, rows, hiddenDim, 3 * hiddenDim, hiddenDim);
            var qView = qkv.View.SubView(0, rows * (long)hiddenDim);
            var kView = qkv.View.SubView(rows * (long)hiddenDim, rows * (long)hiddenDim);
            var vView = qkv.View.SubView(2 * rows * (long)hiddenDim, rows * (long)hiddenDim);

            Attention1D(qView, kView, vView, maskBuf.View, attn.View, batch, seqLen);
            GemmBias(attn.View, w.O.View, h.View, w.Bo.View, rows, hiddenDim, hiddenDim);
            LayerNormResidual1D(h.View, x.View, w.Ln1W.View, w.Ln1B.View, h.View, rows, hiddenDim);

            GemmBias(h.View, w.W1.View, f1.View, w.B1.View, rows, hiddenDim, intermediateDim, activation: 1);

            GemmBias(f1.View, w.W2.View, attn.View, w.B2.View, rows, intermediateDim, hiddenDim);
            LayerNormResidual1D(attn.View, h.View, w.Ln2W.View, w.Ln2B.View, x.View, rows, hiddenDim);
        }

        float[]? logitsArr = null;
        if (hasHead)
        {
            var clsIds = new int[batch];
            for (int b = 0; b < batch; b++)
                clsIds[b] = b * seqLen;
            GpuBuffers.Ensure(ref clsIdsBuf, batch, runtime.Accelerator);
            clsIdsBuf.View.SubView(0, batch).CopyFromCPU(runtime.Stream, clsIds);

            Gather1D(x.View, clsIdsBuf.View, clsOut.View, hiddenDim, batch * hiddenDim);
            GemmBias(clsOut.View, preW!.View, h.View, preB!.View, batch, hiddenDim, hiddenDim, activation: 2);
            GemmBias(h.View, clsW!.View, logits.View, clsB!.View, batch, hiddenDim, 2);

            runtime.Synchronize();
            logitsArr = GpuBuffers.Readback(logits, batch * 2);
        }

        runtime.Synchronize();
        var hidden = GpuBuffers.Readback(x, rows * hiddenDim);
        return new BertEncoderGpuResult(hidden, logitsArr);
    }

    public void Dispose()
    {
        wordEmb.Dispose();
        posEmb.Dispose();
        embLnW.Dispose();
        embLnB.Dispose();
        tokenTypeEmb.Dispose();
        foreach (var layer in layers) layer.Dispose();
        preW?.Dispose();
        preB?.Dispose();
        clsW?.Dispose();
        clsB?.Dispose();
        x.Dispose(); qkv.Dispose();
        attn.Dispose(); h.Dispose(); f1.Dispose(); clsOut.Dispose(); logits.Dispose();
        maskBuf.Dispose(); idsBuf.Dispose(); posIdsBuf.Dispose(); clsIdsBuf.Dispose();
    }

    // ── launch helpers ────────────────────────────────────────────

    void Gather1D(ArrayView<float> table, ArrayView<int> ids, ArrayView<float> output, int hidden, int total)
        => gather(runtime.Stream, GpuBuffers.Cfg1D(total), table, ids, output, hidden);

    void EmbeddingSum1D(
        ArrayView<int> ids, ArrayView<int> posIds,
        ArrayView<float> wordEmb, ArrayView<float> posEmb, ArrayView<float> tokenType, ArrayView<float> y,
        int hidden, int total, int includeTt)
        => embeddingSum(runtime.Stream, GpuBuffers.Cfg1D(total), ids, posIds, wordEmb, posEmb, tokenType, y, hidden, includeTt);

    void LayerNormResidual1D(ArrayView<float> a, ArrayView<float> b, ArrayView<float> gamma, ArrayView<float> beta, ArrayView<float> y, int rows, int cols)
        => layerNormResidual(runtime.Stream, GpuBuffers.Cfg1D(rows), a, b, gamma, beta, y, rows, cols, eps);

    void LayerNorm1D(ArrayView<float> x, ArrayView<float> gamma, ArrayView<float> beta, ArrayView<float> y, int rows, int cols)
        => layerNorm(runtime.Stream, GpuBuffers.Cfg1D(rows), x, gamma, beta, y, rows, cols, eps);

    void Attention1D(
        ArrayView<float> q, ArrayView<float> k, ArrayView<float> v,
        ArrayView<float> mask, ArrayView<float> attnOut,
        int batch, int seqLen)
        => attention(runtime.Stream, GpuBuffers.Cfg1D(batch * numHeads * seqLen, GpuBuffers.AttentionGroupSize),
            q, k, v, mask, attnOut, batch, seqLen, numHeads, headDim, GpuBuffers.GlobalAttentionBand, scale);

    void GemmQkv(ArrayView<float> a, ArrayView<float> bt, ArrayView<float> c, ArrayView<float> bias, int aRows, int aCols, int bCols, int blockWidth)
    {
        gemmQkv(runtime.Stream, GpuBuffers.GemmCfg(aRows, bCols), a, bt, c, bias, aRows, aCols, bCols, blockWidth);
    }

    void GemmBias(ArrayView<float> a, ArrayView<float> bt, ArrayView<float> c, ArrayView<float> bias, int aRows, int aCols, int bCols, int activation = 0)
    {
        var cfg = GpuBuffers.GemmCfg(aRows, bCols);
        switch (activation)
        {
            // Each epilogue form is its own lean kernel (no dead-path code on hot launches).
            case 1: gemmGelu(runtime.Stream, cfg, a, bt, c, bias, aRows, aCols, bCols); break;
            case 2: gemmRelu(runtime.Stream, cfg, a, bt, c, bias, aRows, aCols, bCols); break;
            default: gemmBias(runtime.Stream, cfg, a, bt, c, bias, aRows, aCols, bCols); break;
        }
    }

    // ── weight upload helpers ──────────────────────────────────────

    /// <summary>
    /// Builds the packed q/k/v projection: one pre-transposed [in × 3·out] Bt buffer
    /// [Bt_q | Bt_k | Bt_v] and one [3·out] bias [Bq | Bk | Bv], so the per-layer q/k/v
    /// GEMMs collapse into a single launch producing [rows × 3·out]; q/k/v are then
    /// contiguous SubViews of the result (M2, issue #437).
    /// </summary>
    static (MemoryBuffer1D<float, Stride1D.Dense> W, MemoryBuffer1D<float, Stride1D.Dense> Bias) UploadQkvConcat(
        ILGPU.Runtime.Accelerator acc,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        string qKey,
        string kKey,
        string vKey)
    {
        var weights = new (float[] Data, int[] Shape)[] { GpuBuffers.Req(tensors, $"{qKey}.weight"), GpuBuffers.Req(tensors, $"{kKey}.weight"), GpuBuffers.Req(tensors, $"{vKey}.weight") };
        int outDim = weights[0].Shape[0];
        int inDim = weights[0].Shape[1];
        int concat = 3 * outDim;
        var bt = new float[inDim * concat];
        for (int block = 0; block < 3; block++)
            for (int r = 0; r < outDim; r++)
                for (int c = 0; c < inDim; c++)
                    bt[c * concat + block * outDim + r] = weights[block].Data[r * inDim + c];
        var wBuf = GpuBuffers.Alloc(acc, bt.Length);
        wBuf.CopyFromCPU(bt);

        var bias = new float[concat];
        Array.Copy(GpuBuffers.Req(tensors, $"{qKey}.bias").Data, bias, outDim);
        Array.Copy(GpuBuffers.Req(tensors, $"{kKey}.bias").Data, 0, bias, outDim, outDim);
        Array.Copy(GpuBuffers.Req(tensors, $"{vKey}.bias").Data, 0, bias, 2 * outDim, outDim);
        var bBuf = GpuBuffers.Alloc(acc, bias.Length);
        bBuf.CopyFromCPU(bias);
        return (wBuf, bBuf);
    }

    sealed class LayerGpuWeights : IDisposable
    {
        public readonly MemoryBuffer1D<float, Stride1D.Dense> Wqkv;
        public readonly MemoryBuffer1D<float, Stride1D.Dense> Bqkv;
        public readonly MemoryBuffer1D<float, Stride1D.Dense> O, W1, W2;
        public readonly MemoryBuffer1D<float, Stride1D.Dense> Bo, B1, B2;
        public readonly MemoryBuffer1D<float, Stride1D.Dense> Ln1W, Ln1B, Ln2W, Ln2B;

        public LayerGpuWeights(
            ILGPU.Runtime.Accelerator acc,
            Dictionary<string, (float[] Data, int[] Shape)> tensors,
            BertGpuNaming naming,
            int index)
        {
            var (wqkv, bqkv) = UploadQkvConcat(acc, tensors,
                BertGpuKeys.Attention(naming, index, 'q'),
                BertGpuKeys.Attention(naming, index, 'k'),
                BertGpuKeys.Attention(naming, index, 'v'));
            Wqkv = wqkv;
            Bqkv = bqkv;
            O = GpuBuffers.UploadTransposed(acc, tensors, $"{BertGpuKeys.Attention(naming, index, 'o')}.weight");
            W1 = GpuBuffers.UploadTransposed(acc, tensors, $"{BertGpuKeys.Ffn(naming, index, 1)}.weight");
            W2 = GpuBuffers.UploadTransposed(acc, tensors, $"{BertGpuKeys.Ffn(naming, index, 2)}.weight");

            Bo = GpuBuffers.UploadPlain(acc, tensors, $"{BertGpuKeys.Attention(naming, index, 'o')}.bias");
            B1 = GpuBuffers.UploadPlain(acc, tensors, $"{BertGpuKeys.Ffn(naming, index, 1)}.bias");
            B2 = GpuBuffers.UploadPlain(acc, tensors, $"{BertGpuKeys.Ffn(naming, index, 2)}.bias");

            Ln1W = GpuBuffers.UploadPlain(acc, tensors, $"{BertGpuKeys.LayerNorm(naming, index, 1)}.weight");
            Ln1B = GpuBuffers.UploadPlain(acc, tensors, $"{BertGpuKeys.LayerNorm(naming, index, 1)}.bias");
            Ln2W = GpuBuffers.UploadPlain(acc, tensors, $"{BertGpuKeys.LayerNorm(naming, index, 2)}.weight");
            Ln2B = GpuBuffers.UploadPlain(acc, tensors, $"{BertGpuKeys.LayerNorm(naming, index, 2)}.bias");
        }

        public void Dispose()
        {
            Wqkv.Dispose(); Bqkv.Dispose();
            O.Dispose(); W1.Dispose(); W2.Dispose();
            Bo.Dispose(); B1.Dispose(); B2.Dispose();
            Ln1W.Dispose(); Ln1B.Dispose(); Ln2W.Dispose(); Ln2B.Dispose();
        }
    }
}