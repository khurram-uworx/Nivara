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
    MemoryBuffer1D<float, Stride1D.Dense> scores = null!;
    MemoryBuffer1D<float, Stride1D.Dense> maskBuf = null!;
    MemoryBuffer1D<int, Stride1D.Dense> idsBuf = null!;
    MemoryBuffer1D<int, Stride1D.Dense> posIdsBuf = null!;
    MemoryBuffer1D<int, Stride1D.Dense> clsIdsBuf = null!;

    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<int>, ArrayView<float>, int> gather;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> embeddingSum;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, float> layerNorm;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, float> layerNormResidual;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, float> attention;
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
        var emb = BertGpuKeys.Embeddings(naming);

        var wordEmbT = Req(tensors, $"{emb}.word_embeddings.weight");
        wordEmb = Alloc(acc, wordEmbT.Data.Length);
        wordEmb.CopyFromCPU(wordEmbT.Data);
        var posEmbT = Req(tensors, $"{emb}.position_embeddings.weight");
        posEmb = Alloc(acc, posEmbT.Data.Length);
        posEmb.CopyFromCPU(posEmbT.Data);
        embLnW = Alloc(acc, hiddenDim);
        embLnW.CopyFromCPU(Req(tensors, $"{emb}.LayerNorm.weight").Data);
        embLnB = Alloc(acc, hiddenDim);
        embLnB.CopyFromCPU(Req(tensors, $"{emb}.LayerNorm.bias").Data);

        hasTokenType = tensors.ContainsKey($"{emb}.token_type_embeddings.weight");
        tokenTypeEmb = Alloc(acc, 2 * hiddenDim);
        if (hasTokenType)
            tokenTypeEmb.CopyFromCPU(Req(tensors, $"{emb}.token_type_embeddings.weight").Data);

        layers = new LayerGpuWeights[numLayers];
        for (int i = 0; i < numLayers; i++)
            layers[i] = new LayerGpuWeights(acc, tensors, naming, i);

        hasHead = tensors.ContainsKey(PreWKey) && tensors.ContainsKey(ClsWKey);
        if (hasHead)
        {
            preW = UploadTransposed(acc, tensors, PreWKey);
            preB = UploadPlain(acc, tensors, PreBKey);
            clsW = UploadTransposed(acc, tensors, ClsWKey);
            clsB = UploadPlain(acc, tensors, ClsBKey);
        }

        // Persistent activation workspace (caps: batch <= 8, seqLen <= 128 — the
        // scenario's actual maxima; per-call payload buffers grow on demand).
        int rowsCap = 8 * 128;
        x = Alloc(acc, rowsCap * hiddenDim);
        qkv = Alloc(acc, rowsCap * 3 * hiddenDim);
        attn = Alloc(acc, rowsCap * hiddenDim);
        h = Alloc(acc, rowsCap * hiddenDim);
        f1 = Alloc(acc, rowsCap * intermediateDim);
        clsOut = Alloc(acc, 8 * hiddenDim);
        logits = Alloc(acc, 8 * 2);
        scores = Alloc(acc, 8 * numHeads * 128 * 128);
        maskBuf = Alloc(acc, rowsCap);
        idsBuf = AllocInt(acc, rowsCap);
        posIdsBuf = AllocInt(acc, rowsCap);
        clsIdsBuf = AllocInt(acc, 8);

        gather = acc.LoadKernel<ArrayView<float>, ArrayView<int>, ArrayView<float>, int>(ElementwiseKernels.Gather);
        embeddingSum = acc.LoadKernel<ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(
            ElementwiseKernels.EmbeddingSum);
        layerNorm = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, float>(ElementwiseKernels.LayerNorm1D);
        layerNormResidual = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, float>(
            ElementwiseKernels.LayerNormResidual1D);
        attention = acc.LoadKernel<
            ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, ArrayView<float>, int, int, int, int, float>(
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

        Ensure(ref maskBuf, rows);
        maskBuf.View.SubView(0, rows).CopyFromCPU(runtime.Stream, attentionMask);
        Ensure(ref idsBuf, rows);
        idsBuf.View.SubView(0, rows).CopyFromCPU(runtime.Stream, tokenIds);

        var posIds = new int[rows];
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < seqLen; i++)
                posIds[b * seqLen + i] = i;
        Ensure(ref posIdsBuf, rows);
        posIdsBuf.View.SubView(0, rows).CopyFromCPU(runtime.Stream, posIds);

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

            Attention1D(qView, kView, vView, maskBuf.View, attn.View, scores.View, batch, seqLen);
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
            Ensure(ref clsIdsBuf, batch);
            clsIdsBuf.View.SubView(0, batch).CopyFromCPU(runtime.Stream, clsIds);

            Gather1D(x.View, clsIdsBuf.View, clsOut.View, hiddenDim, batch * hiddenDim);
            GemmBias(clsOut.View, preW!.View, h.View, preB!.View, batch, hiddenDim, hiddenDim, activation: 2);
            GemmBias(h.View, clsW!.View, logits.View, clsB!.View, batch, hiddenDim, 2);

            runtime.Synchronize();
            logitsArr = Readback(logits, batch * 2);
        }

        runtime.Synchronize();
        var hidden = Readback(x, rows * hiddenDim);
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
        scores.Dispose(); maskBuf.Dispose(); idsBuf.Dispose(); posIdsBuf.Dispose(); clsIdsBuf.Dispose();
    }

    // ── launch helpers ────────────────────────────────────────────

    void Gather1D(ArrayView<float> table, ArrayView<int> ids, ArrayView<float> output, int hidden, int total)
        => gather(runtime.Stream, (Cfg(total), 256), table, ids, output, hidden);

    void EmbeddingSum1D(
        ArrayView<int> ids, ArrayView<int> posIds,
        ArrayView<float> wordEmb, ArrayView<float> posEmb, ArrayView<float> tokenType, ArrayView<float> y,
        int hidden, int total, int includeTt)
        => embeddingSum(runtime.Stream, (Cfg(total), 256), ids, posIds, wordEmb, posEmb, tokenType, y, hidden, includeTt);

    void LayerNormResidual1D(ArrayView<float> a, ArrayView<float> b, ArrayView<float> gamma, ArrayView<float> beta, ArrayView<float> y, int rows, int cols)
        => layerNormResidual(runtime.Stream, (Cfg(rows), 256), a, b, gamma, beta, y, rows, cols, eps);

    void LayerNorm1D(ArrayView<float> x, ArrayView<float> gamma, ArrayView<float> beta, ArrayView<float> y, int rows, int cols)
        => layerNorm(runtime.Stream, (Cfg(rows), 256), x, gamma, beta, y, rows, cols, eps);

    void Attention1D(
        ArrayView<float> q, ArrayView<float> k, ArrayView<float> v,
        ArrayView<float> mask, ArrayView<float> attnOut, ArrayView<float> scores,
        int batch, int seqLen)
        => attention(runtime.Stream, (Cfg(batch * numHeads * seqLen), 256),
            q, k, v, mask, attnOut, scores, batch, seqLen, numHeads, headDim, scale);

    void GemmQkv(ArrayView<float> a, ArrayView<float> bt, ArrayView<float> c, ArrayView<float> bias, int aRows, int aCols, int bCols, int blockWidth)
    {
        int blockCols = GemmKernels.TileSize * GemmKernels.BlockCols;
        var numGroups = new Index2D((aRows + GemmKernels.TileSize - 1) / GemmKernels.TileSize, (bCols + blockCols - 1) / blockCols);
        var groupSize = new Index2D(GemmKernels.TileSize, GemmKernels.TileSize);
        gemmQkv(runtime.Stream, (numGroups, groupSize), a, bt, c, bias, aRows, aCols, bCols, blockWidth);
    }

    void GemmBias(ArrayView<float> a, ArrayView<float> bt, ArrayView<float> c, ArrayView<float> bias, int aRows, int aCols, int bCols, int activation = 0)
    {
        int blockCols = GemmKernels.TileSize * GemmKernels.BlockCols;
        var numGroups = new Index2D((aRows + GemmKernels.TileSize - 1) / GemmKernels.TileSize, (bCols + blockCols - 1) / blockCols);
        var groupSize = new Index2D(GemmKernels.TileSize, GemmKernels.TileSize);
        switch (activation)
        {
            // Each epilogue form is its own lean kernel (no dead-path code on hot launches).
            case 1: gemmGelu(runtime.Stream, (numGroups, groupSize), a, bt, c, bias, aRows, aCols, bCols); break;
            case 2: gemmRelu(runtime.Stream, (numGroups, groupSize), a, bt, c, bias, aRows, aCols, bCols); break;
            default: gemmBias(runtime.Stream, (numGroups, groupSize), a, bt, c, bias, aRows, aCols, bCols); break;
        }
    }

    static int Cfg(int total) => total <= 0 ? 1 : (total + 255) / 256;

    static float[] Readback(MemoryBuffer1D<float, Stride1D.Dense> buffer, int length)
    {
        var arr = buffer.AsContiguous().GetAsArray();
        if (arr.Length < length)
            throw new InvalidOperationException("GPU readback returned a shorter buffer than requested.");
        var result = new float[length];
        Array.Copy(arr, result, length);
        return result;
    }

    void Ensure(ref MemoryBuffer1D<float, Stride1D.Dense> buffer, int length)
    {
        if (buffer.Length < length)
        {
            var acc = runtime.Accelerator;
            buffer.Dispose();
            buffer = acc.Allocate1D<float>(length);
        }
    }

    void Ensure(ref MemoryBuffer1D<int, Stride1D.Dense> buffer, int length)
    {
        if (buffer.Length < length)
        {
            var acc = runtime.Accelerator;
            buffer.Dispose();
            buffer = acc.Allocate1D<int>(length);
        }
    }

    // ── weight upload helpers ──────────────────────────────────────

    static MemoryBuffer1D<float, Stride1D.Dense> UploadTransposed(
        ILGPU.Runtime.Accelerator acc,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        string key)
    {
        var t = Req(tensors, key);
        int outDim = t.Shape[0];
        int inDim = t.Shape[1];
        var bt = new float[outDim * inDim];
        for (int r = 0; r < outDim; r++)
            for (int c = 0; c < inDim; c++)
                bt[c * outDim + r] = t.Data[r * inDim + c];
        var buf = Alloc(acc, bt.Length);
        buf.CopyFromCPU(bt);
        return buf;
    }

    static MemoryBuffer1D<float, Stride1D.Dense> UploadPlain(
        ILGPU.Runtime.Accelerator acc,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        string key)
    {
        var buf = Alloc(acc, Req(tensors, key).Data.Length);
        buf.CopyFromCPU(Req(tensors, key).Data);
        return buf;
    }

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
        var weights = new (float[] Data, int[] Shape)[] { Req(tensors, $"{qKey}.weight"), Req(tensors, $"{kKey}.weight"), Req(tensors, $"{vKey}.weight") };
        int outDim = weights[0].Shape[0];
        int inDim = weights[0].Shape[1];
        int concat = 3 * outDim;
        var bt = new float[inDim * concat];
        for (int block = 0; block < 3; block++)
            for (int r = 0; r < outDim; r++)
                for (int c = 0; c < inDim; c++)
                    bt[c * concat + block * outDim + r] = weights[block].Data[r * inDim + c];
        var wBuf = Alloc(acc, bt.Length);
        wBuf.CopyFromCPU(bt);

        var bias = new float[concat];
        Array.Copy(Req(tensors, $"{qKey}.bias").Data, bias, outDim);
        Array.Copy(Req(tensors, $"{kKey}.bias").Data, 0, bias, outDim, outDim);
        Array.Copy(Req(tensors, $"{vKey}.bias").Data, 0, bias, 2 * outDim, outDim);
        var bBuf = Alloc(acc, bias.Length);
        bBuf.CopyFromCPU(bias);
        return (wBuf, bBuf);
    }

    static (float[] Data, int[] Shape) Req(
        Dictionary<string, (float[] Data, int[] Shape)> tensors, string key)
        => tensors.TryGetValue(key, out var t)
            ? t
            : throw new InvalidOperationException($"Missing weight key '{key}' for the --gpu forward.");

    static MemoryBuffer1D<float, Stride1D.Dense> Alloc(ILGPU.Runtime.Accelerator acc, int length)
        => acc.Allocate1D<float>(length);

    static MemoryBuffer1D<int, Stride1D.Dense> AllocInt(ILGPU.Runtime.Accelerator acc, int length)
        => acc.Allocate1D<int>(length);

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
            O = UploadTransposed(acc, tensors, $"{BertGpuKeys.Attention(naming, index, 'o')}.weight");
            W1 = UploadTransposed(acc, tensors, $"{BertGpuKeys.Ffn(naming, index, 1)}.weight");
            W2 = UploadTransposed(acc, tensors, $"{BertGpuKeys.Ffn(naming, index, 2)}.weight");

            Bo = UploadPlain(acc, tensors, $"{BertGpuKeys.Attention(naming, index, 'o')}.bias");
            B1 = UploadPlain(acc, tensors, $"{BertGpuKeys.Ffn(naming, index, 1)}.bias");
            B2 = UploadPlain(acc, tensors, $"{BertGpuKeys.Ffn(naming, index, 2)}.bias");

            Ln1W = UploadPlain(acc, tensors, $"{BertGpuKeys.LayerNorm(naming, index, 1)}.weight");
            Ln1B = UploadPlain(acc, tensors, $"{BertGpuKeys.LayerNorm(naming, index, 1)}.bias");
            Ln2W = UploadPlain(acc, tensors, $"{BertGpuKeys.LayerNorm(naming, index, 2)}.weight");
            Ln2B = UploadPlain(acc, tensors, $"{BertGpuKeys.LayerNorm(naming, index, 2)}.bias");
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