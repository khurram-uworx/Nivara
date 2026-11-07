using Nivara.AI;

namespace Nivara.Silk;

/// <summary>
/// End-to-end fine-tuning of a BERT-family encoder with a sequence-classification head on the
/// Silk/OpenCL backend: forward, backward and the optimizer all run on the device, and the trained
/// weights materialize back into the checkpoint's <c>[out, in]</c> layout for
/// <c>DistilBertForSequenceClassification.LoadWeights</c>.
///
/// <para>
/// <b>Why a second runner.</b> <see cref="SilkBertEncoderRunner"/> reuses one set of activation
/// buffers across the layer stack, which is all inference needs and exactly what a backward pass
/// cannot do: every layer's pre-norm input, attention context, pre-activation and post-activation
/// is needed again in reverse. The two types therefore share the weights
/// (<see cref="SilkBertWeights"/>) and the kernel set, not the workspace. The training forward runs
/// <c>gemm_row4_bias</c> + <c>gelu</c> and <c>layer_norm</c> over a materialized pre-norm sum instead
/// of the fused epilogues, because <c>gelu_backward</c> and <c>layer_norm_backward</c> need those
/// intermediates; both split forms are bit-identical to the fused ones, so the logits are too.
/// </para>
///
/// <para>
/// <b>Layout.</b> Activations are row-major <c>[rows, cols]</c> with <c>rows = batch·seqLen</c>.
/// Weight gradients are written in the device weight layout (<c>[in, out]</c>, q/k/v packed), so no
/// pass unpacks or re-packs anything until <see cref="ExportCheckpoint"/>. The attention backward
/// writes its gradient row-major <c>[rows, 3·hidden]</c> in q|k|v column order even though the
/// forward's packed q/k/v output is block-major, so <c>dW</c>/<c>dB</c>/<c>dX</c> reuse the plain
/// <c>gemm_row4_tn</c>/<c>gemm_row4_nt</c>/<c>col_sum</c> path.
/// </para>
///
/// <para>
/// <b>Determinism.</b> No kernel in the chain uses atomics; every reduction is an ascending loop
/// inside one work item, so a repeat run on the same device is bit-reproducible.
/// </para>
///
/// <para>
/// <b>Memory.</b> The workspace is sized to the caller's caps, not to one batch: activations, their
/// gradients and AdamW's two moment estimates for every parameter. At the DistilBERT-base shapes and
/// an 8×128 cap that is roughly 0.3 GB of activations on top of ~1.06 GB of weight/gradient/moment
/// state.
/// </para>
/// </summary>
public sealed class SilkDistilBertTrainer : IDisposable
{
    const float Beta1 = 0.9f;
    const float Beta2 = 0.999f;
    const float AdamEpsilon = 1e-8f;

    readonly SilkRuntime runtime;
    readonly BertConfig config;
    readonly int hiddenDim;
    readonly int intermediateDim;
    readonly int numHeads;
    readonly int headDim;
    readonly int numLayers;
    readonly float eps;
    readonly float scale;
    readonly int maxBatch;
    readonly int maxSeqLen;
    readonly int rowsCap;
    readonly int numClasses;

    readonly SilkBertWeights weights;
    readonly LayerActivations[] activations;
    readonly List<nint> owned = [];

    readonly nint idsBuf;
    readonly nint posIdsBuf;
    readonly nint maskBuf;
    readonly nint clsIdsBuf;

    readonly nint[] hidden;      // [numLayers + 1] layer inputs and the final hidden state
    readonly nint embPreNorm;    // embedding LayerNorm input

    readonly nint dCur;
    readonly nint dTmp;
    readonly nint dTmp2;
    readonly nint dInterm;
    readonly nint dQkv;
    readonly nint dAttn;
    readonly nint dLnA;
    readonly nint dLnB;
    readonly nint probs;
    readonly nint delta;

    readonly nint clsIn;
    readonly nint reluH;
    readonly nint logits;
    readonly nint dLogits;
    readonly nint dClsIn;

    bool disposed;
    int adamStep;
    int posSeqLen = -1;
    readonly int[] posScratch;

    /// <summary>
    /// Uploads the checkpoint twice — once as the live weights, once for the gradient and the two
    /// AdamW moment buffers of every parameter — and sizes the activation workspace to
    /// <paramref name="maxBatch"/> × <paramref name="maxSeqLen"/>. Throws when the checkpoint has no
    /// classifier head (there is nothing to train) or when the device cannot hold the attention tile.
    /// </summary>
    public SilkDistilBertTrainer(
        SilkRuntime runtime,
        BertConfig config,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        BertGpuNaming naming,
        int maxBatch,
        int maxSeqLen)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(tensors);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxBatch);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSeqLen);

        this.runtime = runtime;
        this.config = config;
        hiddenDim = config.HiddenSize;
        intermediateDim = config.IntermediateSize;
        numHeads = config.NumAttentionHeads;
        headDim = hiddenDim / numHeads;
        numLayers = config.NumHiddenLayers;
        eps = config.LayerNormEps;
        scale = (float)(1.0 / Math.Sqrt(headDim));

        runtime.EnsureAttentionLocalMemory(headDim);

        weights = new SilkBertWeights(runtime, config, tensors, naming, trainable: true);
        if (!weights.HasHead)
            throw new InvalidOperationException(
                "SilkDistilBertTrainer needs the classifier head keys 'pre_classifier.*' and 'classifier.*' in the checkpoint.");
        numClasses = weights.NumClasses;

        this.maxBatch = maxBatch;
        this.maxSeqLen = maxSeqLen;
        rowsCap = maxBatch * maxSeqLen;

        idsBuf = Alloc(rowsCap);
        posIdsBuf = Alloc(rowsCap);
        maskBuf = Alloc(rowsCap);
        clsIdsBuf = Alloc(maxBatch);
        posScratch = new int[rowsCap];

        int hiddenCap = rowsCap * hiddenDim;
        int intermCap = rowsCap * intermediateDim;
        int qkvCap = rowsCap * 3 * hiddenDim;

        hidden = new nint[numLayers + 1];
        for (int i = 0; i < hidden.Length; i++)
            hidden[i] = Alloc(hiddenCap);
        embPreNorm = Alloc(hiddenCap);

        activations = new LayerActivations[numLayers];
        for (int i = 0; i < numLayers; i++)
            activations[i] = new LayerActivations
            {
                Qkv = Alloc(qkvCap),
                Attn = Alloc(hiddenCap),
                PreNorm1 = Alloc(hiddenCap),
                H1 = Alloc(hiddenCap),
                F1Pre = Alloc(intermCap),
                F1Post = Alloc(intermCap),
                PreNorm2 = Alloc(hiddenCap),
            };

        dCur = Alloc(hiddenCap);
        dTmp = Alloc(hiddenCap);
        dTmp2 = Alloc(hiddenCap);
        dInterm = Alloc(intermCap);
        dQkv = Alloc(qkvCap);
        dAttn = Alloc(hiddenCap);
        dLnA = Alloc(hiddenCap);
        dLnB = Alloc(hiddenCap);
        probs = Alloc(maxBatch * numHeads * maxSeqLen * maxSeqLen);
        delta = Alloc(maxBatch * numHeads * maxSeqLen);

        int headCap = maxBatch * hiddenDim;
        clsIn = Alloc(headCap);
        reluH = Alloc(headCap);
        logits = Alloc(maxBatch * numClasses);
        dLogits = Alloc(maxBatch * numClasses);
        dClsIn = Alloc(headCap);
    }

    /// <summary>The OpenCL device the whole training loop runs on.</summary>
    public string DeviceName => runtime.DeviceName;

    /// <summary>The classifier head's output width, taken from <c>classifier.weight</c>'s first dimension.</summary>
    public int NumClasses => numClasses;

    /// <summary>
    /// Builds an inference-only <see cref="SilkBertEncoderRunner"/> that shares this trainer's
    /// live device weights (no copy), so dev-set evaluation can run on the device without an
    /// <c>ExportCheckpoint</c>. The runner borrows the weights; this trainer remains their owner.
    /// </summary>
    public SilkBertEncoderRunner CreateInferenceRunner()
    {
        ThrowIfDisposed();
        return new SilkBertEncoderRunner(runtime, config, weights, ownsWeights: false);
    }

    /// <summary>
    /// Runs the training forward on a batch and returns the <c>[batch, numClasses]</c> logits. The
    /// per-layer intermediates stay resident for the matching <see cref="Backward"/> call, so this is
    /// the first half of one training step rather than a pure inference entry point — use
    /// <see cref="SilkBertEncoderRunner"/> for inference.
    /// </summary>
    public float[] Forward(int[] tokenIds, float[] attentionMask, int batch, int seqLen)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(tokenIds);
        ArgumentNullException.ThrowIfNull(attentionMask);
        UploadBatch(tokenIds, attentionMask, batch, seqLen);
        ForwardCore(batch, seqLen);
        return runtime.Download(logits, batch * numClasses);
    }

    /// <summary>
    /// Trains for <paramref name="epochs"/> epochs of shuffled mini-batches, one device AdamW step
    /// per batch, and returns the mean cross-entropy of every epoch. <paramref name="tokenIds"/> and
    /// <paramref name="attentionMask"/> are flat <c>[samples · seqLen]</c>; <paramref name="labels"/>
    /// is <c>[samples]</c>. Weight decay is applied to every parameter (biases and LayerNorm γ/β
    /// included), matching the CPU path's single parameter group.
    /// </summary>
    public IReadOnlyList<SilkTrainEpoch> Train(
        int[] tokenIds, float[] attentionMask, int[] labels,
        int batchSize, int seqLen, int epochs, float learningRate, float weightDecay, int seed,
        Action<int, int, int, float, TimeSpan>? onBatchCompleted = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(tokenIds);
        ArgumentNullException.ThrowIfNull(attentionMask);
        ArgumentNullException.ThrowIfNull(labels);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(seqLen);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(epochs);
        if (batchSize > maxBatch)
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, $"This trainer caps at batch {maxBatch}.");
        if (seqLen > maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(seqLen), seqLen, $"This trainer caps at seqLen {maxSeqLen}.");
        if (tokenIds.Length != attentionMask.Length)
            throw new ArgumentException("tokenIds and attentionMask must have the same length.", nameof(attentionMask));
        if (tokenIds.Length % seqLen != 0)
            throw new ArgumentException("tokenIds length must be a multiple of seqLen.", nameof(tokenIds));

        int sampleCount = tokenIds.Length / seqLen;
        if (labels.Length != sampleCount)
            throw new ArgumentException($"labels.Length ({labels.Length}) must equal samples ({sampleCount}).", nameof(labels));

        adamStep = 0;
        ZeroMoments();

        var batchTokens = new int[batchSize * seqLen];
        var batchMask = new float[batchSize * seqLen];
        var batchLabels = new int[batchSize];
        var batchGradient = new float[batchSize * numClasses];

        var results = new List<SilkTrainEpoch>(epochs);
        int batchesPerEpoch = (sampleCount + batchSize - 1) / batchSize;
        var shuffle = new Random(seed);
        int[] order = new int[sampleCount];

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < order.Length; i++) order[i] = i;
            for (int i = order.Length - 1; i > 0; i--)
            {
                int j = shuffle.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }

            float lossSum = 0f;
            int lossBatches = 0;
            for (int b = 0; b < batchesPerEpoch; b++)
            {
                int start = b * batchSize;
                int n = Math.Min(start + batchSize, sampleCount) - start;
                if (n <= 0) continue;

                var batchSw = System.Diagnostics.Stopwatch.StartNew();
                for (int i = 0; i < n; i++)
                {
                    int source = order[start + i];
                    Array.Copy(tokenIds, source * seqLen, batchTokens, i * seqLen, seqLen);
                    Array.Copy(attentionMask, source * seqLen, batchMask, i * seqLen, seqLen);
                    batchLabels[i] = labels[source];
                }

                UploadBatch(batchTokens.AsSpan(0, n * seqLen), batchMask.AsSpan(0, n * seqLen), n, seqLen);
                ForwardCore(n, seqLen);
                var batchLogits = runtime.Download(logits, n * numClasses);
                float batchLoss = SilkLoss.CrossEntropy(batchLogits, batchLabels, n, numClasses);
                lossSum += batchLoss;

                SilkLoss.InputGradient(batchLogits, batchLabels, n, numClasses, 1f / n, batchGradient);
                Backward(n, seqLen, batchGradient.AsSpan(0, n * numClasses));
                AdamUpdateAll(learningRate, weightDecay);
                lossBatches++;
                batchSw.Stop();
                onBatchCompleted?.Invoke(epoch + 1, b + 1, batchesPerEpoch, batchLoss, batchSw.Elapsed);
            }

            stopwatch.Stop();
            results.Add(new SilkTrainEpoch(epoch, lossBatches > 0 ? lossSum / lossBatches : 0f, stopwatch.Elapsed));
        }

        return results;
    }

    /// <summary>
    /// Runs the reverse chain over the activations the last <see cref="Forward"/> left resident and
    /// leaves every parameter gradient in its device buffer. <paramref name="dLogits"/> is the
    /// already batch-averaged <c>[batch, numClasses]</c> softmax-cross-entropy gradient, so no pass
    /// here scales by the batch size. Exposed internally for the gradient-parity tests.
    /// </summary>
    internal void Backward(int batch, int seqLen, ReadOnlySpan<float> dLogitsSpan)
    {
        ThrowIfDisposed();
        if (batch <= 0 || batch > maxBatch)
            throw new ArgumentOutOfRangeException(nameof(batch), batch, $"Expected 1..{maxBatch}.");
        if (seqLen <= 0 || seqLen > maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(seqLen), seqLen, $"Expected 1..{maxSeqLen}.");
        if (dLogitsSpan.Length != batch * numClasses)
            throw new ArgumentException($"dLogits.Length ({dLogitsSpan.Length}) must equal batch*numClasses ({batch * numClasses}).", nameof(dLogits));

        int rows = batch * seqLen;
        int hiddenLen = rows * hiddenDim;
        int intermLen = rows * intermediateDim;

        runtime.Upload(dLogits, dLogitsSpan);

        // Classifier head: classifier → ReLU → pre_classifier → CLS scatter.
        SilkLauncher.ColSum(runtime, dLogits, weights.ClsB.Gradient.Handle, batch, numClasses, 1f);
        SilkLauncher.GemmTn(runtime, reluH, dLogits, weights.ClsW.Gradient.Handle,
            aRows: hiddenDim, aCols: batch, bCols: numClasses);
        SilkLauncher.GemmNt(runtime, dLogits, weights.ClsW.Handle, dTmp,
            aRows: batch, aCols: numClasses, bCols: hiddenDim);
        SilkLauncher.ReluBackward(runtime, dTmp, reluH, dTmp, batch * hiddenDim);
        SilkLauncher.ColSum(runtime, dTmp, weights.PreB.Gradient.Handle, batch, hiddenDim, 1f);
        SilkLauncher.GemmTn(runtime, clsIn, dTmp, weights.PreW.Gradient.Handle,
            aRows: hiddenDim, aCols: batch, bCols: hiddenDim);
        SilkLauncher.GemmNt(runtime, dTmp, weights.PreW.Handle, dClsIn,
            aRows: batch, aCols: hiddenDim, bCols: hiddenDim);

        // Only the CLS rows consume the final hidden state, so the top-level gradient is those rows
        // and zeros elsewhere.
        SilkLauncher.Fill(runtime, dCur, 0f, hiddenLen);
        SilkLauncher.ScatterRows(runtime, dClsIn, dCur, batch, seqLen, hiddenDim);

        for (int i = numLayers - 1; i >= 0; i--)
        {
            var w = weights.Layers[i];
            var a = activations[i];

            // LN2: the layer output is layer_norm(preNorm2), so dCur is its gradient on entry.
            LayerNormBackward(dCur, a.PreNorm2, w.Ln2W, dTmp, rows);
            SilkLauncher.ColSum(runtime, dLnA, w.Ln2W.Gradient.Handle, rows, hiddenDim, 1f);
            SilkLauncher.ColSum(runtime, dLnB, w.Ln2B.Gradient.Handle, rows, hiddenDim, 1f);

            // FC2: preNorm2 = f1Post + h1, so dTmp is both dF1Post and the residual's dH1.
            SilkLauncher.GemmTn(runtime, a.F1Post, dTmp, w.W2.Gradient.Handle,
                aRows: intermediateDim, aCols: rows, bCols: hiddenDim);
            SilkLauncher.ColSum(runtime, dTmp, w.B2.Gradient.Handle, rows, hiddenDim, 1f);
            SilkLauncher.GemmNt(runtime, dTmp, w.W2.Handle, dInterm,
                aRows: rows, aCols: hiddenDim, bCols: intermediateDim);

            SilkLauncher.GeluBackward(runtime, dInterm, a.F1Pre, dInterm, intermLen);

            SilkLauncher.GemmTn(runtime, a.H1, dInterm, w.W1.Gradient.Handle,
                aRows: hiddenDim, aCols: rows, bCols: intermediateDim);
            SilkLauncher.ColSum(runtime, dInterm, w.B1.Gradient.Handle, rows, intermediateDim, 1f);
            SilkLauncher.GemmNt(runtime, dInterm, w.W1.Handle, dTmp2,
                aRows: rows, aCols: intermediateDim, bCols: hiddenDim);

            // dH1 = d(preNorm2) + d(fc1 path).
            SilkLauncher.Add(runtime, dTmp, dTmp2, dTmp, hiddenLen);

            // LN1: h1 = layer_norm(preNorm1), and dCur now holds dH1.
            LayerNormBackward(dTmp, a.PreNorm1, w.Ln1W, dCur, rows);
            SilkLauncher.ColSum(runtime, dLnA, w.Ln1W.Gradient.Handle, rows, hiddenDim, 1f);
            SilkLauncher.ColSum(runtime, dLnB, w.Ln1B.Gradient.Handle, rows, hiddenDim, 1f);

            // Output projection: dCur is d(preNorm1) == d(attnOut), since preNorm1 = attnOut + xIn.
            SilkLauncher.GemmTn(runtime, a.Attn, dCur, w.O.Gradient.Handle,
                aRows: hiddenDim, aCols: rows, bCols: hiddenDim);
            SilkLauncher.ColSum(runtime, dCur, w.Bo.Gradient.Handle, rows, hiddenDim, 1f);
            SilkLauncher.GemmNt(runtime, dCur, w.O.Handle, dAttn,
                aRows: rows, aCols: hiddenDim, bCols: hiddenDim);

            SilkLauncher.AttentionBackwardPrep(runtime, a.Qkv, maskBuf, dAttn, probs, delta,
                batch, seqLen, numHeads, headDim, SilkKernels.GlobalAttentionBand, scale);
            SilkLauncher.AttentionBackwardDq(runtime, a.Qkv, probs, delta, dAttn, dQkv,
                batch, seqLen, numHeads, headDim, scale);
            SilkLauncher.AttentionBackwardDkv(runtime, a.Qkv, probs, delta, dAttn, dQkv,
                batch, seqLen, numHeads, headDim, scale);

            SilkLauncher.GemmTn(runtime, hidden[i], dQkv, w.Wqkv.Gradient.Handle,
                aRows: hiddenDim, aCols: rows, bCols: 3 * hiddenDim);
            SilkLauncher.ColSum(runtime, dQkv, w.Bqkv.Gradient.Handle, rows, 3 * hiddenDim, 1f);
            SilkLauncher.GemmNt(runtime, dQkv, w.Wqkv.Handle, dTmp2,
                aRows: rows, aCols: 3 * hiddenDim, bCols: hiddenDim);

            // dXIn = d(attnOut + xIn) + d(qkv path).
            SilkLauncher.Add(runtime, dCur, dTmp2, dCur, hiddenLen);
        }

        // Embedding LayerNorm, then the word and position embedding scatters.
        LayerNormBackward(dCur, embPreNorm, weights.EmbLnW, dTmp, rows);
        SilkLauncher.ColSum(runtime, dLnA, weights.EmbLnW.Gradient.Handle, rows, hiddenDim, 1f);
        SilkLauncher.ColSum(runtime, dLnB, weights.EmbLnB.Gradient.Handle, rows, hiddenDim, 1f);

        SilkLauncher.Fill(runtime, weights.WordEmb.Gradient.Handle, 0f, weights.WordEmb.Count);
        SilkLauncher.EmbeddingBackwardRows(runtime, idsBuf, dTmp, weights.WordEmb.Gradient.Handle, rows, hiddenDim);
        SilkLauncher.Fill(runtime, weights.PosEmb.Gradient.Handle, 0f, weights.PosEmb.Count);
        SilkLauncher.EmbeddingBackwardRows(runtime, posIdsBuf, dTmp, weights.PosEmb.Gradient.Handle, rows, hiddenDim);
    }

    /// <summary>One AdamW step over every parameter, with the caller's weight decay and the shared
    /// step count's bias correction. Internal for the parity tests.</summary>
    internal void AdamUpdateAll(float learningRate, float weightDecay)
    {
        ThrowIfDisposed();
        adamStep++;
        float correction1 = 1f - MathF.Pow(Beta1, adamStep);
        float correction2 = 1f - MathF.Pow(Beta2, adamStep);

        foreach (var p in weights.Parameters)
        {
            SilkLauncher.AdamUpdate(runtime, p.Weight.Handle, p.Gradient.Handle,
                p.FirstMoment.Handle, p.SecondMoment.Handle,
                learningRate, correction1, correction2, AdamEpsilon, weightDecay, p.Count);
        }
    }

    /// <summary>Downloads every parameter gradient in the checkpoint's own key and shape, matching
    /// what <see cref="ExportCheckpoint"/> produces for the weights.</summary>
    internal Dictionary<string, (float[] Data, int[] Shape)> ExportGradients() => weights.ExportGradients();

    /// <summary>
    /// Materializes the trained weights into the checkpoint layout
    /// <c>DistilBertForSequenceClassification.LoadWeights</c> consumes, so the caller can save them
    /// with <c>ModelSerializer.Save</c> exactly as the CPU path does.
    /// </summary>
    public Dictionary<string, (float[] Data, int[] Shape)> ExportCheckpoint() => weights.Export();

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;

        foreach (var layer in activations)
            layer.ReleaseSubviews(runtime);
        foreach (nint buffer in owned)
            runtime.Release(buffer);
        owned.Clear();
        weights.Dispose();
    }

    void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);

    void UploadBatch(ReadOnlySpan<int> tokenIds, ReadOnlySpan<float> attentionMask, int batch, int seqLen)
    {
        if (batch <= 0 || batch > maxBatch)
            throw new ArgumentOutOfRangeException(nameof(batch), batch, $"Expected 1..{maxBatch}.");
        if (seqLen <= 0 || seqLen > maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(seqLen), seqLen, $"Expected 1..{maxSeqLen}.");

        int rows = batch * seqLen;
        if (tokenIds.Length != rows)
            throw new ArgumentException($"tokenIds.Length ({tokenIds.Length}) must equal batch*seqLen ({rows}).", nameof(tokenIds));
        if (attentionMask.Length != rows)
            throw new ArgumentException($"attentionMask.Length ({attentionMask.Length}) must equal batch*seqLen ({rows}).", nameof(attentionMask));

        runtime.Upload(idsBuf, tokenIds);
        runtime.Upload(maskBuf, attentionMask);

        // Positions are row % seqLen, so they only change with seqLen. The whole cap is written so a
        // later, larger batch does not read rows this upload never covered.
        if (posSeqLen != seqLen)
        {
            for (int i = 0; i < posScratch.Length; i++) posScratch[i] = i % seqLen;
            runtime.Upload(posIdsBuf, posScratch);
            posSeqLen = seqLen;
        }
    }

    void ForwardCore(int batch, int seqLen)
    {
        int rows = batch * seqLen;
        int hiddenLen = rows * hiddenDim;
        int intermLen = rows * intermediateDim;

        SilkLauncher.EmbedSum(runtime, idsBuf, posIdsBuf,
            weights.WordEmb.Handle, weights.PosEmb.Handle, weights.TokenTypeEmb.Handle,
            embPreNorm, hiddenLen, hiddenDim, weights.HasTokenType ? 1 : 0);
        SilkLauncher.LayerNorm(runtime, embPreNorm, weights.EmbLnW.Handle, weights.EmbLnB.Handle,
            hidden[0], rows, hiddenDim, eps);

        for (int i = 0; i < numLayers; i++)
        {
            var w = weights.Layers[i];
            var a = activations[i];

            SilkLauncher.GemmEpilogue(runtime, SilkKernels.GemmQkv, hidden[i], w.Wqkv.Handle, a.Qkv,
                w.Bqkv.Handle, rows, hiddenDim, 3 * hiddenDim, hiddenDim);
            a.PrepareSubviews(runtime, rows, hiddenDim);
            SilkLauncher.Attention(runtime, a.Q, a.K, a.V, maskBuf, a.Attn,
                batch, seqLen, numHeads, headDim, SilkKernels.GlobalAttentionBand, scale);

            SilkLauncher.GemmEpilogue(runtime, SilkKernels.GemmBias, a.Attn, w.O.Handle, a.PreNorm1,
                w.Bo.Handle, rows, hiddenDim, hiddenDim);
            SilkLauncher.Add(runtime, a.PreNorm1, hidden[i], a.PreNorm1, hiddenLen);
            SilkLauncher.LayerNorm(runtime, a.PreNorm1, w.Ln1W.Handle, w.Ln1B.Handle,
                a.H1, rows, hiddenDim, eps);

            SilkLauncher.GemmEpilogue(runtime, SilkKernels.GemmBias, a.H1, w.W1.Handle, a.F1Pre,
                w.B1.Handle, rows, hiddenDim, intermediateDim);
            SilkLauncher.Gelu(runtime, a.F1Pre, a.F1Post, intermLen);

            SilkLauncher.GemmEpilogue(runtime, SilkKernels.GemmBias, a.F1Post, w.W2.Handle, a.PreNorm2,
                w.B2.Handle, rows, intermediateDim, hiddenDim);
            SilkLauncher.Add(runtime, a.PreNorm2, a.H1, a.PreNorm2, hiddenLen);
            SilkLauncher.LayerNorm(runtime, a.PreNorm2, w.Ln2W.Handle, w.Ln2B.Handle,
                hidden[i + 1], rows, hiddenDim, eps);
        }

        var clsIds = new int[batch];
        for (int b = 0; b < batch; b++) clsIds[b] = b * seqLen;
        runtime.Upload(clsIdsBuf, clsIds.AsSpan(0, batch));

        SilkLauncher.Gather(runtime, hidden[numLayers], clsIdsBuf, clsIn, batch * hiddenDim, hiddenDim);
        SilkLauncher.GemmEpilogue(runtime, SilkKernels.GemmRelu, clsIn, weights.PreW.Handle, reluH,
            weights.PreB.Handle, batch, hiddenDim, hiddenDim);
        SilkLauncher.GemmEpilogue(runtime, SilkKernels.GemmBias, reluH, weights.ClsW.Handle, logits,
            weights.ClsB.Handle, batch, hiddenDim, numClasses);
    }

    void LayerNormBackward(nint d, nint x, SilkParam gamma, nint dx, int rows)
        => SilkLauncher.LayerNormBackward(runtime, d, x, gamma.Handle, dx, dLnA, dLnB, rows, hiddenDim, eps);

    void ZeroMoments()
    {
        foreach (var p in weights.Parameters)
        {
            SilkLauncher.Fill(runtime, p.FirstMoment.Handle, 0f, p.Count);
            SilkLauncher.Fill(runtime, p.SecondMoment.Handle, 0f, p.Count);
        }
    }

    nint Alloc(int floatCount)
    {
        nint buffer = runtime.Allocate(floatCount);
        owned.Add(buffer);
        return buffer;
    }

    /// <summary>
    /// One layer's resident activations. <see cref="Q"/>/<see cref="K"/>/<see cref="V"/> are the
    /// read-only views attention reads out of the block-major packed <see cref="Qkv"/>; their offsets
    /// depend on the batch's <c>rows</c>, so they are rebuilt when it changes.
    /// </summary>
    sealed class LayerActivations
    {
        public required nint Qkv;
        public required nint Attn;
        public required nint PreNorm1;
        public required nint H1;
        public required nint F1Pre;
        public required nint F1Post;
        public required nint PreNorm2;

        public nint Q;
        public nint K;
        public nint V;
        public int SubviewRows = -1;

        public void PrepareSubviews(SilkRuntime runtime, int rows, int hiddenDim)
        {
            if (SubviewRows == rows) return;
            ReleaseSubviews(runtime);
            Q = runtime.CreateSubview(Qkv, 0, rows * hiddenDim);
            K = runtime.CreateSubview(Qkv, rows * hiddenDim, rows * hiddenDim);
            V = runtime.CreateSubview(Qkv, 2 * rows * hiddenDim, rows * hiddenDim);
            SubviewRows = rows;
        }

        public void ReleaseSubviews(SilkRuntime runtime)
        {
            if (Q != 0) runtime.Release(Q);
            if (K != 0) runtime.Release(K);
            if (V != 0) runtime.Release(V);
            Q = 0;
            K = 0;
            V = 0;
            SubviewRows = -1;
        }
    }
}
