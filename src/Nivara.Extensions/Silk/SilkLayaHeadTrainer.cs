using Nivara.AI;

namespace Nivara.Silk;

public sealed partial class SilkLayaHeadTrainer : IDisposable
{
    const float Beta1 = 0.9f;
    const float Beta2 = 0.999f;
    const float AdamEpsilon = 1e-8f;

    readonly SilkRuntime runtime;
    readonly SilkLayaHeadWeights weights;
    readonly int headLayerCount;
    readonly int actClassCount;
    readonly int maxSeqLen;
    readonly int maxMarkers;
    readonly bool narrowBf16;
    readonly bool narrowFp16;

    readonly int hiddenDim;
    readonly int intermediateDim;
    readonly int numHeads;
    readonly int headDim;
    readonly int actHidden;
    readonly float eps = 1e-5f;
    readonly float scale;

    bool disposed;
    int adamStep;

    // activation buffers (current capacity maxSeqLen)
    readonly nint hiddenBuf;
    readonly nint hiddenIn;
    readonly nint[] layerInput;
    readonly nint[] normed1;
    readonly nint[] qkv;
    readonly nint[] attn;
    readonly nint[] postAttnHidden;
    readonly nint[] normed2;
    readonly nint[] f1Pre;
    readonly nint[] f1PreAct; // pre-gelu for scorer hidden? reuse naming
    readonly nint[] f1Post;
    readonly nint[] postFfnHidden;
    readonly nint projected;
    readonly nint maskBuf;
    readonly nint markerIdsBuf;
    readonly nint markerRows;
    readonly nint markerScored;
    readonly nint markerScoredPre; // pre-gelu
    readonly nint markerScoredOut;
    readonly nint scorerLogitsBuf;
    readonly nint actRow;
    readonly nint actHiddenPre;
    readonly nint actHiddenBuf;
    readonly nint actLogitsBuf;
    readonly nint probsBuf;
    readonly nint deltaBuf;

    // gradient buffers
    readonly nint dCur;
    readonly nint dTmp;
    readonly nint dTmp2;
    readonly nint dInterm;
    readonly nint dQkv;
    readonly nint dAttn;
    readonly nint dLnA;
    readonly nint dLnB;
    readonly nint dMarkerHidden;
    readonly nint dScorerHidden;
    readonly nint dActTmp;

    // saved metadata
    int currentSeqLen;
    int currentMarkerCount;
    int currentValidLength;
    LayaQuestionType currentType;
    int[] currentMarkers = [];

    readonly List<nint> owned = [];

    public SilkLayaHeadTrainer(
        SilkRuntime runtime,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        int headLayerCount = 2,
        int actClassCount = 2,
        int maxSeqLen = 64)
        : this(runtime, new SilkLayaHeadWeights(runtime, tensors, headLayerCount, actClassCount, trainable: true), false, false, maxSeqLen)
    {
    }

    internal SilkLayaHeadTrainer(
        SilkRuntime runtime,
        SilkLayaHeadWeights weights,
        bool narrowBf16,
        bool narrowFp16,
        int maxSeqLen)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(weights);
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));

        this.runtime = runtime;
        this.weights = weights;
        this.narrowBf16 = narrowBf16;
        this.narrowFp16 = narrowFp16;
        this.headLayerCount = weights.HeadLayerCount;
        this.actClassCount = weights.ActClassCount;
        this.maxSeqLen = maxSeqLen;
        this.maxMarkers = maxSeqLen;

        hiddenDim = weights.HiddenSize;
        intermediateDim = 4 * hiddenDim;
        numHeads = Math.Max(1, hiddenDim / LayaDecisionHead<float>.HeadDimTarget);
        headDim = hiddenDim / numHeads;
        actHidden = LayaDecisionHead<float>.ActHeadHidden;
        scale = 1f / MathF.Sqrt(headDim);

        runtime.EnsureAttentionLocalMemory(headDim);

        int hiddenCap = maxSeqLen * hiddenDim;
        int intermCap = maxSeqLen * intermediateDim;
        int qkvCap = maxSeqLen * 3 * hiddenDim;
        int markerCap = maxMarkers * hiddenDim;

        hiddenBuf = Alloc(hiddenCap);
        hiddenIn = Alloc(hiddenCap);
        layerInput = new nint[headLayerCount];
        normed1 = new nint[headLayerCount];
        qkv = new nint[headLayerCount];
        attn = new nint[headLayerCount];
        postAttnHidden = new nint[headLayerCount];
        normed2 = new nint[headLayerCount];
        f1Pre = new nint[headLayerCount];
        f1PreAct = new nint[headLayerCount];
        f1Post = new nint[headLayerCount];
        postFfnHidden = new nint[headLayerCount];
        for (int i = 0; i < headLayerCount; i++)
        {
            layerInput[i] = Alloc(hiddenCap);
            normed1[i] = Alloc(hiddenCap);
            qkv[i] = Alloc(qkvCap);
            attn[i] = Alloc(hiddenCap);
            postAttnHidden[i] = Alloc(hiddenCap);
            normed2[i] = Alloc(hiddenCap);
            f1Pre[i] = Alloc(intermCap);
            f1PreAct[i] = Alloc(intermCap);
            f1Post[i] = Alloc(intermCap);
            postFfnHidden[i] = Alloc(hiddenCap);
        }
        projected = Alloc(Math.Max(hiddenCap, intermCap));
        maskBuf = Alloc(maxSeqLen);
        markerIdsBuf = Alloc(maxMarkers);
        markerRows = Alloc(markerCap);
        markerScored = Alloc(markerCap);
        markerScoredPre = Alloc(markerCap);
        markerScoredOut = Alloc(markerCap);
        scorerLogitsBuf = Alloc(maxMarkers);
        actRow = Alloc(hiddenDim + LayaHeadOutput.FeatureCount);
        actHiddenPre = Alloc(actHidden);
        actHiddenBuf = Alloc(actHidden);
        actLogitsBuf = Alloc(actClassCount);
        probsBuf = Alloc(numHeads * maxSeqLen * maxSeqLen);
        deltaBuf = Alloc(numHeads * maxSeqLen);

        dCur = Alloc(hiddenCap);
        dTmp = Alloc(hiddenCap);
        dTmp2 = Alloc(hiddenCap);
        dInterm = Alloc(intermCap);
        dQkv = Alloc(qkvCap);
        dAttn = Alloc(hiddenCap);
        dLnA = Alloc(hiddenCap);
        dLnB = Alloc(hiddenCap);
        dMarkerHidden = Alloc(markerCap);
        dScorerHidden = Alloc(markerCap);
        dActTmp = Alloc(actHidden);
    }

    public string DeviceName => runtime.DeviceName;

    public LayaHeadOutput Forward(float[] hidden, LayaQuestionType type, IReadOnlyList<int> markerPositions, int validLength)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(hidden);
        ArgumentNullException.ThrowIfNull(markerPositions);
        if (markerPositions.Count == 0) throw new ArgumentException("markerPositions must not be empty.");
        int seqLen = hidden.Length / hiddenDim;
        if (seqLen <= 0 || seqLen > maxSeqLen) throw new ArgumentOutOfRangeException(nameof(hidden));
        if (validLength <= 0 || validLength > seqLen) throw new ArgumentOutOfRangeException(nameof(validLength));

        currentSeqLen = seqLen;
        currentMarkerCount = markerPositions.Count;
        currentValidLength = validLength;
        currentType = type;
        currentMarkers = markerPositions.ToArray();

        int hiddenLen = seqLen * hiddenDim;

        runtime.Upload(hiddenBuf, hidden.AsSpan(0, hiddenLen));

        int typeOffset = (int)type * hiddenDim;
        nint typeRow = runtime.CreateSubview(weights.TypeEmb.Handle, typeOffset, hiddenDim);
        SilkLauncher.AddBias(runtime, hiddenBuf, typeRow, hiddenBuf, seqLen, hiddenDim);
        runtime.Release(typeRow);
        runtime.CopyBuffer(hiddenBuf, hiddenIn, 0, 0, hiddenLen);

        // mask
        runtime.Upload(maskBuf, LayaHeadScoring.PrefixMask(seqLen, validLength));

        for (int i = 0; i < headLayerCount; i++)
        {
            var w = weights.Layers[i];
            runtime.CopyBuffer(hiddenBuf, layerInput[i], 0, 0, hiddenLen);
            SilkLauncher.LayerNorm(runtime, hiddenBuf, w.Norm1W.Handle, w.Norm1B.Handle, normed1[i], seqLen, hiddenDim, eps);
            SilkLauncher.GemmEpilogue(runtime, K(SilkKernels.GemmQkv), normed1[i], w.Wqkv.Handle, qkv[i], w.Bqkv.Handle, seqLen, hiddenDim, 3 * hiddenDim, hiddenDim);
            // attention
            nint q = runtime.CreateSubview(qkv[i], 0, hiddenLen);
            nint k = runtime.CreateSubview(qkv[i], hiddenLen, hiddenLen);
            nint v = runtime.CreateSubview(qkv[i], 2 * hiddenLen, hiddenLen);
            SilkLauncher.Attention(runtime, q, k, v, maskBuf, attn[i], 1, seqLen, numHeads, headDim, SilkKernels.GlobalAttentionBand, scale);
            runtime.Release(q); runtime.Release(k); runtime.Release(v);

            SilkLauncher.GemmEpilogue(runtime, K(SilkKernels.GemmBias), attn[i], w.Wo.Handle, projected, w.Bo.Handle, seqLen, hiddenDim, hiddenDim);
            SilkLauncher.Add(runtime, hiddenBuf, projected, postAttnHidden[i], hiddenLen);
            runtime.CopyBuffer(postAttnHidden[i], hiddenBuf, 0, 0, hiddenLen);

            SilkLauncher.LayerNorm(runtime, hiddenBuf, w.Norm2W.Handle, w.Norm2B.Handle, normed2[i], seqLen, hiddenDim, eps);
            SilkLauncher.GemmEpilogue(runtime, K(SilkKernels.GemmBias), normed2[i], w.W1.Handle, f1Pre[i], w.B1.Handle, seqLen, hiddenDim, intermediateDim);
            SilkLauncher.Relu(runtime, f1Pre[i], f1Post[i], seqLen * intermediateDim);
            SilkLauncher.GemmEpilogue(runtime, K(SilkKernels.GemmBias), f1Post[i], w.W2.Handle, projected, w.B2.Handle, seqLen, intermediateDim, hiddenDim);
            SilkLauncher.Add(runtime, hiddenBuf, projected, postFfnHidden[i], seqLen * hiddenDim);
            runtime.CopyBuffer(postFfnHidden[i], hiddenBuf, 0, 0, seqLen * hiddenDim);
        }

        var markerIdHost = new int[currentMarkerCount];
        for (int i = 0; i < currentMarkerCount; i++) markerIdHost[i] = markerPositions[i];
        runtime.Upload(markerIdsBuf, markerIdHost.AsSpan(0, currentMarkerCount));

        int kCount = currentMarkerCount;
        int markerHiddenLen = kCount * hiddenDim;
        SilkLauncher.Gather(runtime, hiddenBuf, markerIdsBuf, markerRows, markerHiddenLen, hiddenDim);

        SilkLauncher.LayerNorm(runtime, markerRows, weights.ScorerNormW.Handle, weights.ScorerNormB.Handle, markerScored, kCount, hiddenDim, eps);
        SilkLauncher.GemmEpilogue(runtime, K(SilkKernels.GemmBias), markerScored, weights.ScorerHiddenW.Handle, markerScoredPre, weights.ScorerHiddenB.Handle, kCount, hiddenDim, hiddenDim);
        SilkLauncher.Gelu(runtime, markerScoredPre, markerScoredOut, kCount * hiddenDim);
        SilkLauncher.GemmEpilogue(runtime, K(SilkKernels.GemmBias), markerScoredOut, weights.ScorerOutW.Handle, scorerLogitsBuf, weights.ScorerOutB.Handle, kCount, hiddenDim, 1);

        // act pooled row
        var pooledIds = new int[] { 0 };
        runtime.Upload(markerIdsBuf, pooledIds.AsSpan());
        SilkLauncher.Gather(runtime, hiddenBuf, markerIdsBuf, actRow, hiddenDim, hiddenDim);

        // download logits for output
        float[] logitsHost = runtime.Download(scorerLogitsBuf, kCount);

        // softmax etc
        var probabilities = LayaHeadScoring.Softmax(logitsHost);
        var features = LayaHeadScoring.BuildFeatures(probabilities);

        // fill features into actRow
        float[] actRowHost = runtime.Download(actRow, hiddenDim + LayaHeadOutput.FeatureCount);
        for (int i = 0; i < features.Length; i++) actRowHost[hiddenDim + i] = (float)features[i];
        runtime.Upload(actRow, actRowHost);

        SilkLauncher.GemmEpilogue(runtime, K(SilkKernels.GemmBias), actRow, weights.ActInW.Handle, actHiddenPre, weights.ActInB.Handle, 1, hiddenDim + LayaHeadOutput.FeatureCount, actHidden);
        SilkLauncher.Gelu(runtime, actHiddenPre, actHiddenBuf, actHidden);
        SilkLauncher.GemmEpilogue(runtime, K(SilkKernels.GemmBias), actHiddenBuf, weights.ActOutW.Handle, actLogitsBuf, weights.ActOutB.Handle, 1, actHidden, actClassCount);

        float[] actHost = runtime.Download(actLogitsBuf, actClassCount);
        var actionProbabilities = LayaHeadScoring.Softmax(actHost);

        return new LayaHeadOutput
        {
            Logits = logitsHost,
            MarkerProbabilities = probabilities,
            Features = features,
            ActionProbability = actionProbabilities[0],
            ActionClassCount = actionProbabilities.Length
        };
    }

    public void Backward(ReadOnlySpan<float> dScorerLogits, ReadOnlySpan<float> dActLogits)
    {
        ThrowIfDisposed();
        if (dScorerLogits.Length != currentMarkerCount) throw new ArgumentException("dScorerLogits length mismatch.");
        if (dActLogits.Length != actClassCount) throw new ArgumentException("dActLogits length mismatch.");

        int seqLen = currentSeqLen;
        int k = currentMarkerCount;
        int hiddenLen = seqLen * hiddenDim;
        int intermLen = seqLen * intermediateDim;
        int markerHiddenLen = k * hiddenDim;

        runtime.Upload(scorerLogitsBuf, dScorerLogits);
        runtime.Upload(actLogitsBuf, dActLogits);

        int actInDim = hiddenDim + LayaHeadOutput.FeatureCount;
        SilkLauncher.GemmTn(runtime, actHiddenBuf, actLogitsBuf, weights.ActOutW.Gradient.Handle, actHidden, 1, actClassCount);
        SilkLauncher.ColSum(runtime, actLogitsBuf, weights.ActOutB.Gradient.Handle, 1, actClassCount, 1f);
        GemmNtW(actLogitsBuf, weights.ActOutW.Handle, dActTmp, 1, actClassCount, actHidden);
        SilkLauncher.GeluBackward(runtime, dActTmp, actHiddenPre, dActTmp, actHidden);
        SilkLauncher.GemmTn(runtime, actRow, dActTmp, weights.ActInW.Gradient.Handle, actInDim, 1, actHidden);
        SilkLauncher.ColSum(runtime, dActTmp, weights.ActInB.Gradient.Handle, 1, actHidden, 1f);

        SilkLauncher.GemmTn(runtime, markerScoredOut, scorerLogitsBuf, weights.ScorerOutW.Gradient.Handle, hiddenDim, k, 1);
        SilkLauncher.ColSum(runtime, scorerLogitsBuf, weights.ScorerOutB.Gradient.Handle, k, 1, 1f);
        GemmNtW(scorerLogitsBuf, weights.ScorerOutW.Handle, dScorerHidden, k, 1, hiddenDim);
        SilkLauncher.GeluBackward(runtime, dScorerHidden, markerScoredPre, dScorerHidden, k * hiddenDim);

        SilkLauncher.GemmTn(runtime, markerScored, dScorerHidden, weights.ScorerHiddenW.Gradient.Handle, hiddenDim, k, hiddenDim);
        SilkLauncher.ColSum(runtime, dScorerHidden, weights.ScorerHiddenB.Gradient.Handle, k, hiddenDim, 1f);
        GemmNtW(dScorerHidden, weights.ScorerHiddenW.Handle, dMarkerHidden, k, hiddenDim, hiddenDim);

        SilkLauncher.LayerNormBackward(runtime, dMarkerHidden, markerRows, weights.ScorerNormW.Handle, dMarkerHidden, dLnA, dLnB, k, hiddenDim, eps);
        SilkLauncher.ColSum(runtime, dLnA, weights.ScorerNormW.Gradient.Handle, k, hiddenDim, 1f);
        SilkLauncher.ColSum(runtime, dLnB, weights.ScorerNormB.Gradient.Handle, k, hiddenDim, 1f);

        SilkLauncher.Fill(runtime, dCur, 0f, hiddenLen);
        {
            float[] dh = runtime.Download(dMarkerHidden, markerHiddenLen);
            float[] cur = runtime.Download(dCur, hiddenLen);
            for (int i = 0; i < k; i++)
            {
                int pos = currentMarkers[i];
                for (int h = 0; h < hiddenDim; h++)
                    cur[pos * hiddenDim + h] += dh[i * hiddenDim + h];
            }
            runtime.Upload(dCur, cur);
        }

        {
            float[] cur = runtime.Download(dCur, hiddenLen);
            var typeGrad = new float[hiddenDim];
            for (int r = 0; r < seqLen; r++)
                for (int h = 0; h < hiddenDim; h++)
                    typeGrad[h] += cur[r * hiddenDim + h];
            float[] full = runtime.Download(weights.TypeEmb.Gradient.Handle, 3 * hiddenDim);
            int off = (int)currentType * hiddenDim;
            for (int h = 0; h < hiddenDim; h++) full[off + h] += typeGrad[h];
            runtime.Upload(weights.TypeEmb.Gradient.Handle, full);
        }

        for (int li = headLayerCount - 1; li >= 0; li--)
        {
            var w = weights.Layers[li];

            SilkLauncher.GemmTn(runtime, f1Post[li], dCur, w.W2.Gradient.Handle, intermediateDim, seqLen, hiddenDim);
            SilkLauncher.ColSum(runtime, dCur, w.B2.Gradient.Handle, seqLen, hiddenDim, 1f);
            GemmNtW(dCur, w.W2.Handle, dInterm, seqLen, hiddenDim, intermediateDim);
            SilkLauncher.ReluBackward(runtime, dInterm, f1Pre[li], dInterm, intermLen);

            SilkLauncher.GemmTn(runtime, normed2[li], dInterm, w.W1.Gradient.Handle, hiddenDim, seqLen, intermediateDim);
            SilkLauncher.ColSum(runtime, dInterm, w.B1.Gradient.Handle, seqLen, intermediateDim, 1f);
            GemmNtW(dInterm, w.W1.Handle, dTmp, seqLen, intermediateDim, hiddenDim);

            SilkLauncher.LayerNormBackward(runtime, dTmp, postAttnHidden[li], w.Norm2W.Handle, dTmp2, dLnA, dLnB, seqLen, hiddenDim, eps);
            SilkLauncher.ColSum(runtime, dLnA, w.Norm2W.Gradient.Handle, seqLen, hiddenDim, 1f);
            SilkLauncher.ColSum(runtime, dLnB, w.Norm2B.Gradient.Handle, seqLen, hiddenDim, 1f);

            SilkLauncher.Add(runtime, dTmp2, dCur, dTmp2, hiddenLen);

            SilkLauncher.GemmTn(runtime, attn[li], dTmp2, w.Wo.Gradient.Handle, hiddenDim, seqLen, hiddenDim);
            SilkLauncher.ColSum(runtime, dTmp2, w.Bo.Gradient.Handle, seqLen, hiddenDim, 1f);
            GemmNtW(dTmp2, w.Wo.Handle, dAttn, seqLen, hiddenDim, hiddenDim);

            SilkLauncher.AttentionBackwardPrep(runtime, qkv[li], maskBuf, dAttn, probsBuf, deltaBuf, 1, seqLen, numHeads, headDim, SilkKernels.GlobalAttentionBand, scale);
            SilkLauncher.AttentionBackwardDq(runtime, qkv[li], probsBuf, deltaBuf, dAttn, dQkv, 1, seqLen, numHeads, headDim, scale);
            SilkLauncher.AttentionBackwardDkv(runtime, qkv[li], probsBuf, deltaBuf, dAttn, dQkv, 1, seqLen, numHeads, headDim, scale);

            SilkLauncher.GemmTn(runtime, normed1[li], dQkv, w.Wqkv.Gradient.Handle, hiddenDim, seqLen, 3 * hiddenDim);
            SilkLauncher.ColSum(runtime, dQkv, w.Bqkv.Gradient.Handle, seqLen, 3 * hiddenDim, 1f);
            GemmNtW(dQkv, w.Wqkv.Handle, dTmp, seqLen, 3 * hiddenDim, hiddenDim);

            SilkLauncher.LayerNormBackward(runtime, dTmp, layerInput[li], w.Norm1W.Handle, dTmp2, dLnA, dLnB, seqLen, hiddenDim, eps);
            SilkLauncher.ColSum(runtime, dLnA, w.Norm1W.Gradient.Handle, seqLen, hiddenDim, 1f);
            SilkLauncher.ColSum(runtime, dLnB, w.Norm1B.Gradient.Handle, seqLen, hiddenDim, 1f);

            SilkLauncher.Add(runtime, dTmp2, dCur, dCur, hiddenLen);
        }
    }

    public void AdamUpdateAll(float learningRate, float weightDecay)
    {
        ThrowIfDisposed();
        adamStep++;
        float c1 = 1f - MathF.Pow(Beta1, adamStep);
        float c2 = 1f - MathF.Pow(Beta2, adamStep);
        foreach (var p in weights.Parameters.Where(p => p.Trainable))
        {
            if (p.Slot.NarrowMatrix && narrowBf16)
                SilkLauncher.AdamUpdateBf16(runtime, p.Weight.Handle, p.Gradient.Handle, p.FirstMoment.Handle, p.SecondMoment.Handle, learningRate, c1, c2, AdamEpsilon, weightDecay, p.Count);
            else
                SilkLauncher.AdamUpdate(runtime, p.Weight.Handle, p.Gradient.Handle, p.FirstMoment.Handle, p.SecondMoment.Handle, learningRate, c1, c2, AdamEpsilon, weightDecay, p.Count);
        }
    }

    public Dictionary<string, (float[] Data, int[] Shape)> ExportCheckpoint() => weights.Export();

    public Dictionary<string, (float[] Data, int[] Shape)> ExportGradients() => weights.ExportGradients();

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        foreach (nint b in owned) runtime.Release(b);
        owned.Clear();
        weights.Dispose();
    }

    void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);

    string K(string kernel) => kernel switch
    {
        SilkKernels.GemmQkv => narrowBf16 ? SilkKernels.GemmQkvBf16 : narrowFp16 ? SilkKernels.GemmQkvF16 : kernel,
        SilkKernels.GemmBias => narrowBf16 ? SilkKernels.GemmBiasBf16 : narrowFp16 ? SilkKernels.GemmBiasF16 : kernel,
        SilkKernels.GemmGelu => narrowBf16 ? SilkKernels.GemmGeluBf16 : narrowFp16 ? SilkKernels.GemmGeluF16 : kernel,
        SilkKernels.GemmRelu => narrowBf16 ? SilkKernels.GemmReluBf16 : narrowFp16 ? SilkKernels.GemmReluF16 : kernel,
        _ => kernel,
    };

    void GemmNtW(nint a, nint b, nint c, int aRows, int aCols, int bCols)
    {
        if (narrowBf16) SilkLauncher.GemmNtBf16(runtime, a, b, c, aRows, aCols, bCols);
        else SilkLauncher.GemmNt(runtime, a, b, c, aRows, aCols, bCols);
    }

    void GemmTnW(nint a, nint b, nint c, int aRows, int aCols, int bCols)
    {
        // dW accumulators are F32 regardless of weight precision
        SilkLauncher.GemmTn(runtime, a, b, c, aRows, aCols, bCols);
    }

    nint Alloc(int count)
    {
        nint b = runtime.Allocate(count);
        owned.Add(b);
        return b;
    }
}
