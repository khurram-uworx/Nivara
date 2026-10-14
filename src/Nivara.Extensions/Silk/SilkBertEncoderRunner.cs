using Nivara.AI;

namespace Nivara.Silk;

/// <summary>
/// BERT-family encoder forward on the Silk.NET OpenCL backend (<see cref="SilkKernels"/>): the
/// Silk counterpart of <c>BertEncoderGpuRunner</c>. Same weight layout, same operation order and
/// the same fused epilogues, so the two backends are elementwise comparable — the kernel gate
/// holds their GEMMs to byte-identity and the remaining kernels to the model's numeric bound.
///
/// <para>
/// <b>Scope.</b> The BERT-family encoders the ILGPU runner covers — MiniLM and DistilBERT — selected
/// by the <see cref="BertGpuNaming"/> passed to the constructor, which resolves the checkpoint's key
/// style (<see cref="BertGpuKeys"/>). A checkpoint that ships the classifier head
/// (<c>pre_classifier</c>/<c>classifier</c> keys) runs it too, through the <c>gather_rows</c> and
/// <c>gemm_row4_relu</c> kernels, and returns the <c>[batch, numClasses]</c> logits.
/// </para>
///
/// <para>
/// The workspace is cap-sized (batch ≤ 8, seqLen ≤ 128 — the scenario's actual maxima) and every
/// layer intermediate stays on the device, mirroring <c>BertEncoderGpuRunner</c>; only the
/// final hidden state is read back. Every dispatch finishes the queue, so the readback needs no
/// further sync.
/// </para>
/// </summary>
public sealed class SilkBertEncoderRunner : IBertEncoderRunner
{
    /// <summary>Workspace cap, matching the ILGPU runner's (the scenario's actual maxima).</summary>
    const int MaxBatch = 8;
    const int MaxSeqLen = 128;
    const int NumClasses = 2;
    const string Backend = "Silk";

    const string PreWKey = "pre_classifier.weight";
    const string PreBKey = "pre_classifier.bias";
    const string ClsWKey = "classifier.weight";
    const string ClsBKey = "classifier.bias";

    readonly SilkRuntime runtime;
    readonly int hiddenDim;
    readonly int intermediateDim;
    readonly int numHeads;
    readonly int headDim;
    readonly int numLayers;
    readonly float eps;
    readonly float scale;
    readonly bool hasTokenType;
    readonly BertGpuNaming naming;

    readonly nint wordEmb;
    readonly nint posEmb;
    readonly nint embLnW;
    readonly nint embLnB;
    readonly nint tokenTypeEmb;
    readonly bool hasHead;
    readonly nint preW;
    readonly nint preB;
    readonly nint clsW;
    readonly nint clsB;
    readonly LayerSilkWeights[] layers;

    readonly nint x;
    readonly nint qkv;
    readonly nint attn;
    readonly nint h;
    readonly nint f1;
    readonly nint zeroResidual;
    readonly nint clsOut;
    readonly nint logits;
    readonly nint maskBuf;
    readonly nint idsBuf;
    readonly nint posIdsBuf;
    readonly nint clsIdsBuf;

    readonly List<nint> owned = [];
    int cachedPosSeqLen = -1;

    /// <summary>
    /// Uploads every weight — GEMM weights transposed once and q/k/v packed through
    /// <see cref="WeightLayout"/>, the same arithmetic the ILGPU runner uses, so the two backends
    /// cannot drift — and sizes the cap workspace. <paramref name="naming"/> selects the checkpoint's
    /// weight-key style. Throws if a required encoder key is missing or if the device's local memory
    /// cannot hold the attention tile. The classifier-head weights are optional: when the checkpoint
    /// ships <c>pre_classifier</c>/<c>classifier</c> keys they are uploaded and the head runs in
    /// <see cref="Forward"/>.
    /// </summary>
    public SilkBertEncoderRunner(
        SilkRuntime runtime,
        BertConfig config,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        BertGpuNaming naming)
    {
        this.runtime = runtime;
        this.naming = naming;
        hiddenDim = config.HiddenSize;
        intermediateDim = config.IntermediateSize;
        numHeads = config.NumAttentionHeads;
        headDim = hiddenDim / numHeads;
        numLayers = config.NumHiddenLayers;
        eps = config.LayerNormEps;
        scale = (float)(1.0 / Math.Sqrt(headDim));

        runtime.EnsureAttentionLocalMemory(headDim);

        hasHead = tensors.ContainsKey(PreWKey) && tensors.ContainsKey(ClsWKey);

        string emb = BertGpuKeys.Embeddings(naming);
        wordEmb = UploadPlain(tensors, $"{emb}.word_embeddings.weight");
        posEmb = UploadPlain(tensors, $"{emb}.position_embeddings.weight");
        embLnW = UploadPlain(tensors, $"{emb}.LayerNorm.weight");
        embLnB = UploadPlain(tensors, $"{emb}.LayerNorm.bias");

        // Only row 0 of the token-type table is read (all-zero token-type ids, mirroring the CPU
        // includeTokenTypeEmbedding default), so this is one row rather than the checkpoint's two.
        hasTokenType = tensors.ContainsKey($"{emb}.token_type_embeddings.weight");
        tokenTypeEmb = Register(runtime.Allocate(hiddenDim));
        if (hasTokenType)
            runtime.Upload(tokenTypeEmb,
                WeightLayout.Req(tensors, $"{emb}.token_type_embeddings.weight", Backend).Data.AsSpan(0, hiddenDim));

        layers = new LayerSilkWeights[numLayers];
        for (int i = 0; i < numLayers; i++)
            layers[i] = UploadLayer(tensors, i);

        // Classifier head: pre_classifier (hidden→hidden, ReLU) then classifier (hidden→numClasses).
        // Allocated even when the checkpoint has no head so Dispose is unconditional; uploaded only
        // when hasHead.
        preW = Register(runtime.Allocate(hiddenDim * hiddenDim));
        preB = Register(runtime.Allocate(hiddenDim));
        clsW = Register(runtime.Allocate(NumClasses * hiddenDim));
        clsB = Register(runtime.Allocate(NumClasses));
        if (hasHead)
        {
            runtime.Upload(preW, WeightLayout.Transposed(tensors, PreWKey, Backend));
            runtime.Upload(preB, WeightLayout.Req(tensors, PreBKey, Backend).Data);
            runtime.Upload(clsW, WeightLayout.Transposed(tensors, ClsWKey, Backend));
            runtime.Upload(clsB, WeightLayout.Req(tensors, ClsBKey, Backend).Data);
        }

        int rowsCap = MaxBatch * MaxSeqLen;
        x = Register(runtime.Allocate(rowsCap * hiddenDim));
        qkv = Register(runtime.Allocate(rowsCap * 3 * hiddenDim));
        attn = Register(runtime.Allocate(rowsCap * hiddenDim));
        h = Register(runtime.Allocate(rowsCap * hiddenDim));
        f1 = Register(runtime.Allocate(rowsCap * intermediateDim));
        zeroResidual = Register(runtime.Allocate(rowsCap * hiddenDim));
        clsOut = Register(runtime.Allocate(MaxBatch * hiddenDim));
        logits = Register(runtime.Allocate(MaxBatch * NumClasses));
        maskBuf = Register(runtime.Allocate(rowsCap));
        idsBuf = Register(runtime.Allocate(rowsCap));
        posIdsBuf = Register(runtime.Allocate(rowsCap));
        clsIdsBuf = Register(runtime.Allocate(MaxBatch));

        // The embedding LayerNorm is the encoder's one non-residual layer norm, and
        // layer_norm_residual reproduces it exactly with a zero residual (a + 0.0f == a), which is
        // why the kernel set has no dedicated embedding-norm kernel.
        runtime.Upload(zeroResidual, new float[rowsCap * hiddenDim]);
    }

    public string DeviceName => runtime.DeviceName;

    /// <inheritdoc/>
    public BertEncoderGpuResult Forward(int[] tokenIds, float[] attentionMask, int batch, int seqLen)
    {
        int rows = batch * seqLen;
        if (tokenIds.Length != rows)
            throw new ArgumentException($"tokenIds.Length ({tokenIds.Length}) must equal batch*seqLen ({rows}).", nameof(tokenIds));
        if (attentionMask.Length != rows)
            throw new ArgumentException($"attentionMask.Length ({attentionMask.Length}) must equal batch*seqLen ({rows}).", nameof(attentionMask));
        if (batch > MaxBatch || seqLen > MaxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(batch),
                $"The Silk encoder workspace caps at batch<=8, seqLen<=128 (the scenario's actual maxima: batch 1, seqLen 128); got batch={batch}, seqLen={seqLen}.");

        runtime.Upload(maskBuf, attentionMask.AsSpan(0, rows));
        runtime.Upload(idsBuf, tokenIds.AsSpan(0, rows));

        // posIds are seqLen-deterministic (row % seqLen), so the host->device copy is skipped when
        // seqLen has not changed — the repeat-forward benchmark path reuses one shape.
        if (cachedPosSeqLen != seqLen)
        {
            var posIds = new int[rows];
            for (int i = 0; i < rows; i++)
                posIds[i] = i % seqLen;
            runtime.Upload(posIdsBuf, posIds);
            cachedPosSeqLen = seqLen;
        }

        // embeddings: one fused word/pos (+ token-type row 0) gather/sum, then the embedding
        // LayerNorm as a zero-residual layer norm.
        LaunchEmbedSum(rows * hiddenDim, hasTokenType ? 1 : 0);
        LaunchLayerNormResidual(x, zeroResidual, embLnW, embLnB, x, rows, hiddenDim);

        // q/k/v are the three contiguous blocks of the packed qkv result; attention reads them as
        // read-only sub-buffer views (the OpenCL analogue of the ILGPU runner's qkv SubViews).
        // They depend only on `rows`, so one set covers the whole layer stack.
        nint q = runtime.CreateSubview(qkv, 0, rows * hiddenDim);
        nint k = runtime.CreateSubview(qkv, rows * hiddenDim, rows * hiddenDim);
        nint v = runtime.CreateSubview(qkv, 2 * rows * hiddenDim, rows * hiddenDim);
        try
        {
            for (int i = 0; i < numLayers; i++)
            {
                var w = layers[i];

                LaunchGemm(SilkKernels.GemmQkv, x, w.Wqkv, qkv, w.Bqkv, rows, hiddenDim, 3 * hiddenDim, hiddenDim);
                LaunchAttention(q, k, v, maskBuf, attn, batch, seqLen);

                LaunchGemm(SilkKernels.GemmBias, attn, w.O, h, w.Bo, rows, hiddenDim, hiddenDim);
                LaunchLayerNormResidual(h, x, w.Ln1W, w.Ln1B, h, rows, hiddenDim);
                LaunchGemm(SilkKernels.GemmGelu, h, w.W1, f1, w.B1, rows, hiddenDim, intermediateDim);
                LaunchGemm(SilkKernels.GemmBias, f1, w.W2, attn, w.B2, rows, intermediateDim, hiddenDim);
                LaunchLayerNormResidual(attn, h, w.Ln2W, w.Ln2B, x, rows, hiddenDim);
            }
        }
        finally
        {
            runtime.Release(q);
            runtime.Release(k);
            runtime.Release(v);
        }

        float[]? logitsArr = null;
        if (hasHead)
        {
            // The CLS row of each batch row is row b*seqLen (position 0), matching the CPU/ILGPU
            // head: gather it, then pre_classifier (ReLU) → classifier (bias only).
            var clsIds = new int[batch];
            for (int b = 0; b < batch; b++)
                clsIds[b] = b * seqLen;

            runtime.Upload(clsIdsBuf, clsIds.AsSpan(0, batch));
            LaunchGather(clsOut, x, clsIdsBuf, hiddenDim, batch * hiddenDim);
            LaunchGemm(SilkKernels.GemmRelu, clsOut, preW, h, preB, batch, hiddenDim, hiddenDim);
            LaunchGemm(SilkKernels.GemmBias, h, clsW, logits, clsB, batch, hiddenDim, NumClasses);
            logitsArr = runtime.Download(logits, batch * NumClasses);
        }

        return new BertEncoderGpuResult(runtime.Download(x, rows * hiddenDim), logitsArr);
    }

    public void Dispose()
    {
        foreach (nint buffer in owned)
            runtime.Release(buffer);
        owned.Clear();
    }

    // ── launch helpers ────────────────────────────────────────────

    void LaunchEmbedSum(int total, int includeTokenType)
        => SilkLauncher.EmbedSum(
            runtime, idsBuf, posIdsBuf, wordEmb, posEmb, tokenTypeEmb, x, total, hiddenDim, includeTokenType);

    void LaunchLayerNormResidual(nint a, nint b, nint gamma, nint beta, nint y, int rows, int cols)
        => SilkLauncher.LayerNormResidual(runtime, a, b, gamma, beta, y, rows, cols, eps);

    /// <summary>
    /// Dispatches a Row4 epilogue GEMM over the <c>(ceil(aRows/16), ceil(bCols/64))</c> grid of
    /// 16×16 groups. <paramref name="qkvBlockWidth"/> is <c>gemm_row4_qkv</c>'s extra argument:
    /// it is <c>null</c> for <c>gemm_row4_bias</c> and <c>gemm_row4_gelu</c>, which take six.
    /// </summary>
    void LaunchGemm(string kernel, nint a, nint b, nint c, nint bias, int aRows, int aCols, int bCols, int? qkvBlockWidth = null)
        => SilkLauncher.GemmEpilogue(runtime, kernel, a, b, c, bias, aRows, aCols, bCols, qkvBlockWidth);

    void LaunchAttention(nint q, nint k, nint v, nint mask, nint output, int batch, int seqLen)
        => SilkLauncher.Attention(
            runtime, q, k, v, mask, output, batch, seqLen, numHeads, headDim, SilkKernels.GlobalAttentionBand, scale);

    void LaunchGather(nint output, nint table, nint ids, int hidden, int total)
        => SilkLauncher.Gather(runtime, table, ids, output, total, hidden);

    // ── weight upload ─────────────────────────────────────────────

    LayerSilkWeights UploadLayer(Dictionary<string, (float[] Data, int[] Shape)> tensors, int index)
    {
        string q = BertGpuKeys.Attention(naming, index, 'q');
        string k = BertGpuKeys.Attention(naming, index, 'k');
        string v = BertGpuKeys.Attention(naming, index, 'v');
        string o = BertGpuKeys.Attention(naming, index, 'o');
        string ffn1 = BertGpuKeys.Ffn(naming, index, 1);
        string ffn2 = BertGpuKeys.Ffn(naming, index, 2);
        string ln1 = BertGpuKeys.LayerNorm(naming, index, 1);
        string ln2 = BertGpuKeys.LayerNorm(naming, index, 2);

        var (wqkv, bqkv) = WeightLayout.QkvPacked(tensors, q, k, v, Backend);
        return new LayerSilkWeights
        {
            Wqkv = Upload(wqkv),
            Bqkv = Upload(bqkv),
            O = UploadTransposed(tensors, $"{o}.weight"),
            W1 = UploadTransposed(tensors, $"{ffn1}.weight"),
            W2 = UploadTransposed(tensors, $"{ffn2}.weight"),
            Bo = UploadPlain(tensors, $"{o}.bias"),
            B1 = UploadPlain(tensors, $"{ffn1}.bias"),
            B2 = UploadPlain(tensors, $"{ffn2}.bias"),
            Ln1W = UploadPlain(tensors, $"{ln1}.weight"),
            Ln1B = UploadPlain(tensors, $"{ln1}.bias"),
            Ln2W = UploadPlain(tensors, $"{ln2}.weight"),
            Ln2B = UploadPlain(tensors, $"{ln2}.bias")
        };
    }

    nint UploadPlain(Dictionary<string, (float[] Data, int[] Shape)> tensors, string key)
        => Upload(WeightLayout.Req(tensors, key, Backend).Data);

    nint UploadTransposed(Dictionary<string, (float[] Data, int[] Shape)> tensors, string key)
        => Upload(WeightLayout.Transposed(tensors, key, Backend));

    nint Upload(float[] data)
    {
        nint buffer = Register(runtime.Allocate(data.Length));
        runtime.Upload(buffer, data);
        return buffer;
    }

    /// <summary>Allocates and registers a buffer for release in <see cref="Dispose"/>. The element
    /// count is bytes/4, so it is the same call for float and int payloads.</summary>
    nint Register(nint buffer)
    {
        owned.Add(buffer);
        return buffer;
    }

    /// <summary>One layer's uploaded weights: packed q/k/v projection + bias, the four projections,
    /// their biases, and the two LayerNorm gamma/beta pairs.</summary>
    sealed class LayerSilkWeights
    {
        public nint Wqkv;
        public nint Bqkv;
        public nint O;
        public nint W1;
        public nint W2;
        public nint Bo;
        public nint B1;
        public nint B2;
        public nint Ln1W;
        public nint Ln1B;
        public nint Ln2W;
        public nint Ln2B;
    }
}
