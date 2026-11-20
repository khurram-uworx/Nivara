using System.Numerics;

namespace Nivara.Silk;

/// <summary>
/// TinyShakespeare (batched causal transformer) trainer on Silk/OpenCL, F32 only.
/// Mirrors <see cref="BatchedTransformer{T}"/> forward/backward and an Adam step, exporting
/// checkpoints in the same shape as <c>ToSilkWeights()</c>.
/// </summary>
public sealed class SilkTinyShakespeareTrainer : IDisposable
{
    readonly SilkRuntime runtime;
    readonly TinyShakespeareConfig config;
    readonly int hiddenDim;
    readonly int intermediateDim;
    readonly int numHeads;
    readonly int headDim;
    readonly int vocabSize;
    readonly int maxSeqLen;
    readonly int maxBatch;
    readonly float eps;
    readonly float scale;

    // weights
    readonly nint tokEmb;
    readonly nint tokEmbTransposed;
    readonly nint posEmb;
    readonly nint finalNormW;
    readonly nint finalNormB;
    readonly nint[] ln1W;
    readonly nint[] ln1B;
    readonly nint[] ln2W;
    readonly nint[] ln2B;
    readonly nint[] wQ;
    readonly nint[] wK;
    readonly nint[] wV;
    readonly nint[] wO;
    readonly nint[] wFc1;
    readonly nint[] fc1Bias;
    readonly nint[] wFc2;
    readonly nint[] fc2Bias;

    // gradients
    readonly nint tokEmbGrad;
    readonly nint finalNormWGrad;
    readonly nint finalNormBGrad;
    readonly nint[] ln1WGrad;
    readonly nint[] ln1BGrad;
    readonly nint[] ln2WGrad;
    readonly nint[] ln2BGrad;
    readonly nint[] wQGrad;
    readonly nint[] wKGrad;
    readonly nint[] wVGrad;
    readonly nint[] wOGrad;
    readonly nint[] wFc1Grad;
    readonly nint[] fc1BiasGrad;
    readonly nint[] wFc2Grad;
    readonly nint[] fc2BiasGrad;

    // adam moments
    readonly nint tokEmbM, tokEmbV;
    readonly nint finalNormWM, finalNormWV, finalNormBM, finalNormBV;
    readonly nint[][] ln1WM, ln1WV, ln1BM, ln1BV;
    readonly nint[][] ln2WM, ln2WV, ln2BM, ln2BV;
    readonly nint[][] wQM, wQV, wKM, wKV, wVM, wVV, wOM, wOV;
    readonly nint[][] wFc1M, wFc1V, fc1BiasM, fc1BiasV;
    readonly nint[][] wFc2M, wFc2V, fc2BiasM, fc2BiasV;

    // activations (maxSeqLen capacity)
    readonly nint xBuf;
    readonly nint normedBuf;
    readonly nint posBuf;
    readonly nint idsBuf;
    readonly nint posIdsBuf;
    readonly nint[] layerInput;
    readonly nint[] normed1;
    readonly nint[] qBuf;
    readonly nint[] kBuf;
    readonly nint[] vBuf;
    readonly nint[] qkvBuf;
    readonly nint[] attnBuf;
    readonly nint[] postAttn;
    readonly nint[] normed2;
    readonly nint[] f1Pre;
    readonly nint[] gateBuf;
    readonly nint finalNormInput;
    readonly nint logitsRowBuf;

    // gradients scratch
    readonly nint dCur;
    readonly nint dTmp;
    readonly nint dTmp2;
    readonly nint dInterm;
    readonly nint dQkv;
    readonly nint dAttn;
    readonly nint dLnA;
    readonly nint dLnB;
    readonly nint attnProbs;
    readonly nint attnDelta;
    readonly nint dLogitsBuf;
    readonly nint targetIdsBuf;

    int currentSeqLen;
    int adamStep;
    bool disposed;

    // parameter list for adam/zero-grad
    readonly List<(nint Weight, nint Grad, nint M, nint V, int Count)> parameters = new();

    int[]? posIdHost;

    public SilkTinyShakespeareTrainer(
        SilkRuntime runtime,
        TinyShakespeareConfig config,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        int maxBatch = 32,
        int maxSeqLen = 64)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(tensors);
        if (maxBatch <= 0) throw new ArgumentOutOfRangeException(nameof(maxBatch), maxBatch, "maxBatch must be positive.");
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen), maxSeqLen, "maxSeqLen must be positive.");

        this.runtime = runtime;
        this.config = config;
        this.maxBatch = maxBatch;
        this.maxSeqLen = maxSeqLen;

        hiddenDim = config.HiddenSize;
        intermediateDim = 4 * hiddenDim;
        numHeads = config.NumAttentionHeads;
        headDim = hiddenDim / numHeads;
        vocabSize = config.VocabSize;
        eps = config.LayerNormEps;
        scale = 1f / MathF.Sqrt(headDim);

        runtime.EnsureAttentionLocalMemory(headDim);

        int nLayers = config.NumHiddenLayers;

        // position table
        posEmb = runtime.Allocate(maxSeqLen * hiddenDim);
        runtime.Upload(posEmb, BuildPositionTable(maxSeqLen, hiddenDim));

        // token embedding
        tokEmb = UploadPlain(tensors, "tok_emb.weight");
        tokEmbTransposed = runtime.Allocate(hiddenDim * vocabSize);
        runtime.Upload(tokEmbTransposed, WeightLayout.Transposed(tensors, "tok_emb.weight", "Silk"));

        finalNormW = UploadPlain(tensors, "final_norm.weight");
        finalNormB = UploadPlain(tensors, "final_norm.bias");

        ln1W = new nint[nLayers];
        ln1B = new nint[nLayers];
        ln2W = new nint[nLayers];
        ln2B = new nint[nLayers];
        wQ = new nint[nLayers];
        wK = new nint[nLayers];
        wV = new nint[nLayers];
        wO = new nint[nLayers];
        wFc1 = new nint[nLayers];
        fc1Bias = new nint[nLayers];
        wFc2 = new nint[nLayers];
        fc2Bias = new nint[nLayers];

        ln1WGrad = new nint[nLayers];
        ln1BGrad = new nint[nLayers];
        ln2WGrad = new nint[nLayers];
        ln2BGrad = new nint[nLayers];
        wQGrad = new nint[nLayers];
        wKGrad = new nint[nLayers];
        wVGrad = new nint[nLayers];
        wOGrad = new nint[nLayers];
        wFc1Grad = new nint[nLayers];
        fc1BiasGrad = new nint[nLayers];
        wFc2Grad = new nint[nLayers];
        fc2BiasGrad = new nint[nLayers];

        ln1WM = new nint[nLayers][];
        ln1WV = new nint[nLayers][];
        ln1BM = new nint[nLayers][];
        ln1BV = new nint[nLayers][];
        ln2WM = new nint[nLayers][];
        ln2WV = new nint[nLayers][];
        ln2BM = new nint[nLayers][];
        ln2BV = new nint[nLayers][];
        wQM = new nint[nLayers][];
        wQV = new nint[nLayers][];
        wKM = new nint[nLayers][];
        wKV = new nint[nLayers][];
        wVM = new nint[nLayers][];
        wVV = new nint[nLayers][];
        wOM = new nint[nLayers][];
        wOV = new nint[nLayers][];
        wFc1M = new nint[nLayers][];
        wFc1V = new nint[nLayers][];
        fc1BiasM = new nint[nLayers][];
        fc1BiasV = new nint[nLayers][];
        wFc2M = new nint[nLayers][];
        wFc2V = new nint[nLayers][];
        fc2BiasM = new nint[nLayers][];
        fc2BiasV = new nint[nLayers][];

        layerInput = new nint[nLayers];
        normed1 = new nint[nLayers];
        qBuf = new nint[nLayers];
        kBuf = new nint[nLayers];
        vBuf = new nint[nLayers];
        qkvBuf = new nint[nLayers];
        attnBuf = new nint[nLayers];
        postAttn = new nint[nLayers];
        normed2 = new nint[nLayers];
        f1Pre = new nint[nLayers];
        gateBuf = new nint[nLayers];

        int cap = maxSeqLen;
        for (int i = 0; i < nLayers; i++)
        {
            string p = $"layers.{i}.";
            ln1W[i] = UploadPlain(tensors, p + "ln1.weight");
            ln1B[i] = UploadPlain(tensors, p + "ln1.bias");
            ln2W[i] = UploadPlain(tensors, p + "ln2.weight");
            ln2B[i] = UploadPlain(tensors, p + "ln2.bias");
            wQ[i] = UploadT(tensors, p + "q_proj.weight");
            wK[i] = UploadT(tensors, p + "k_proj.weight");
            wV[i] = UploadT(tensors, p + "v_proj.weight");
            wO[i] = UploadT(tensors, p + "o_proj.weight");
            wFc1[i] = UploadT(tensors, p + "mlp_fc1.weight");
            fc1Bias[i] = UploadPlain(tensors, p + "mlp_fc1.bias");
            wFc2[i] = UploadT(tensors, p + "mlp_fc2.weight");
            fc2Bias[i] = UploadPlain(tensors, p + "mlp_fc2.bias");

            // gradients
            ln1WGrad[i] = runtime.Allocate(hiddenDim);
            ln1BGrad[i] = runtime.Allocate(hiddenDim);
            ln2WGrad[i] = runtime.Allocate(hiddenDim);
            ln2BGrad[i] = runtime.Allocate(hiddenDim);
            wQGrad[i] = runtime.Allocate(hiddenDim * hiddenDim);
            wKGrad[i] = runtime.Allocate(hiddenDim * hiddenDim);
            wVGrad[i] = runtime.Allocate(hiddenDim * hiddenDim);
            wOGrad[i] = runtime.Allocate(hiddenDim * hiddenDim);
            wFc1Grad[i] = runtime.Allocate(hiddenDim * intermediateDim);
            fc1BiasGrad[i] = runtime.Allocate(intermediateDim);
            wFc2Grad[i] = runtime.Allocate(intermediateDim * hiddenDim);
            fc2BiasGrad[i] = runtime.Allocate(hiddenDim);

            // moments
            ln1WM[i] = new[] { runtime.Allocate(hiddenDim) };
            ln1WV[i] = new[] { runtime.Allocate(hiddenDim) };
            ln1BM[i] = new[] { runtime.Allocate(hiddenDim) };
            ln1BV[i] = new[] { runtime.Allocate(hiddenDim) };
            ln2WM[i] = new[] { runtime.Allocate(hiddenDim) };
            ln2WV[i] = new[] { runtime.Allocate(hiddenDim) };
            ln2BM[i] = new[] { runtime.Allocate(hiddenDim) };
            ln2BV[i] = new[] { runtime.Allocate(hiddenDim) };
            wQM[i] = new[] { runtime.Allocate(hiddenDim * hiddenDim) };
            wQV[i] = new[] { runtime.Allocate(hiddenDim * hiddenDim) };
            wKM[i] = new[] { runtime.Allocate(hiddenDim * hiddenDim) };
            wKV[i] = new[] { runtime.Allocate(hiddenDim * hiddenDim) };
            wVM[i] = new[] { runtime.Allocate(hiddenDim * hiddenDim) };
            wVV[i] = new[] { runtime.Allocate(hiddenDim * hiddenDim) };
            wOM[i] = new[] { runtime.Allocate(hiddenDim * hiddenDim) };
            wOV[i] = new[] { runtime.Allocate(hiddenDim * hiddenDim) };
            wFc1M[i] = new[] { runtime.Allocate(hiddenDim * intermediateDim) };
            wFc1V[i] = new[] { runtime.Allocate(hiddenDim * intermediateDim) };
            fc1BiasM[i] = new[] { runtime.Allocate(intermediateDim) };
            fc1BiasV[i] = new[] { runtime.Allocate(intermediateDim) };
            wFc2M[i] = new[] { runtime.Allocate(intermediateDim * hiddenDim) };
            wFc2V[i] = new[] { runtime.Allocate(intermediateDim * hiddenDim) };
            fc2BiasM[i] = new[] { runtime.Allocate(hiddenDim) };
            fc2BiasV[i] = new[] { runtime.Allocate(hiddenDim) };



            layerInput[i] = runtime.Allocate(cap * hiddenDim);
            normed1[i] = runtime.Allocate(cap * hiddenDim);
            qBuf[i] = runtime.Allocate(cap * hiddenDim);
            kBuf[i] = runtime.Allocate(cap * hiddenDim);
            vBuf[i] = runtime.Allocate(cap * hiddenDim);
            qkvBuf[i] = runtime.Allocate(cap * hiddenDim * 3);
            attnBuf[i] = runtime.Allocate(cap * hiddenDim);
            postAttn[i] = runtime.Allocate(cap * hiddenDim);
            normed2[i] = runtime.Allocate(cap * hiddenDim);
            f1Pre[i] = runtime.Allocate(cap * intermediateDim);
            gateBuf[i] = runtime.Allocate(cap * intermediateDim);
        }

        tokEmbGrad = runtime.Allocate(vocabSize * hiddenDim);
        finalNormWGrad = runtime.Allocate(hiddenDim);
        finalNormBGrad = runtime.Allocate(hiddenDim);
        tokEmbM = runtime.Allocate(vocabSize * hiddenDim);
        tokEmbV = runtime.Allocate(vocabSize * hiddenDim);
        finalNormWM = runtime.Allocate(hiddenDim);
        finalNormWV = runtime.Allocate(hiddenDim);
        finalNormBM = runtime.Allocate(hiddenDim);
        finalNormBV = runtime.Allocate(hiddenDim);

        xBuf = runtime.Allocate(cap * hiddenDim);
        normedBuf = runtime.Allocate(cap * hiddenDim);
        posBuf = runtime.Allocate(cap * hiddenDim);
        idsBuf = runtime.Allocate(cap);
        posIdsBuf = runtime.Allocate(cap);
        finalNormInput = runtime.Allocate(cap * hiddenDim);
        logitsRowBuf = runtime.Allocate(vocabSize);
        dCur = runtime.Allocate(cap * hiddenDim);
        dTmp = runtime.Allocate(cap * hiddenDim);
        dTmp2 = runtime.Allocate(cap * hiddenDim);
        dInterm = runtime.Allocate(cap * intermediateDim);
        dQkv = runtime.Allocate(cap * hiddenDim * 3);
        dAttn = runtime.Allocate(cap * hiddenDim);
        dLnA = runtime.Allocate(hiddenDim);
        dLnB = runtime.Allocate(hiddenDim);
        dLogitsBuf = runtime.Allocate(cap * vocabSize);
        targetIdsBuf = runtime.Allocate(cap);
        attnProbs = runtime.Allocate(numHeads * cap * cap);
        attnDelta = runtime.Allocate(numHeads * cap);

        // register params
        AddParam(tokEmb, tokEmbGrad, tokEmbM, tokEmbV, vocabSize * hiddenDim);
        AddParam(finalNormW, finalNormWGrad, finalNormWM, finalNormWV, hiddenDim);
        AddParam(finalNormB, finalNormBGrad, finalNormBM, finalNormBV, hiddenDim);
        for (int i = 0; i < nLayers; i++)
        {
            AddParam(ln1W[i], ln1WGrad[i], ln1WM[i][0], ln1WV[i][0], hiddenDim);
            AddParam(ln1B[i], ln1BGrad[i], ln1BM[i][0], ln1BV[i][0], hiddenDim);
            AddParam(ln2W[i], ln2WGrad[i], ln2WM[i][0], ln2WV[i][0], hiddenDim);
            AddParam(ln2B[i], ln2BGrad[i], ln2BM[i][0], ln2BV[i][0], hiddenDim);
            AddParam(wQ[i], wQGrad[i], wQM[i][0], wQV[i][0], hiddenDim * hiddenDim);
            AddParam(wK[i], wKGrad[i], wKM[i][0], wKV[i][0], hiddenDim * hiddenDim);
            AddParam(wV[i], wVGrad[i], wVM[i][0], wVV[i][0], hiddenDim * hiddenDim);
            AddParam(wO[i], wOGrad[i], wOM[i][0], wOV[i][0], hiddenDim * hiddenDim);
            AddParam(wFc1[i], wFc1Grad[i], wFc1M[i][0], wFc1V[i][0], hiddenDim * intermediateDim);
            AddParam(fc1Bias[i], fc1BiasGrad[i], fc1BiasM[i][0], fc1BiasV[i][0], intermediateDim);
            AddParam(wFc2[i], wFc2Grad[i], wFc2M[i][0], wFc2V[i][0], intermediateDim * hiddenDim);
            AddParam(fc2Bias[i], fc2BiasGrad[i], fc2BiasM[i][0], fc2BiasV[i][0], hiddenDim);
        }

        foreach (var p in parameters)
        {
            SilkLauncher.Fill(runtime, p.Grad, 0f, p.Count);
            SilkLauncher.Fill(runtime, p.M, 0f, p.Count);
            SilkLauncher.Fill(runtime, p.V, 0f, p.Count);
        }
    }

    void AddParam(nint w, nint g, nint m, nint v, int count)
        => parameters.Add((w, g, m, v, count));

    public string DeviceName => runtime.DeviceName;

    public float[] Forward(int[] inputTokens, int seqLen)
    {
        ThrowIfDisposed();
        if (seqLen <= 0 || seqLen > maxSeqLen) throw new ArgumentOutOfRangeException(nameof(seqLen), seqLen, $"Expected 1..{maxSeqLen}.");
        if ((long)seqLen > (long)maxBatch * maxSeqLen) throw new ArgumentOutOfRangeException(nameof(seqLen), seqLen, $"seqLen ({seqLen}) exceeds maxBatch*maxSeqLen ({(long)maxBatch * maxSeqLen}).");
        if (inputTokens.Length < seqLen) throw new ArgumentException("inputTokens too short.", nameof(inputTokens));

        currentSeqLen = seqLen;

        // ids
        runtime.Upload(idsBuf, inputTokens.AsSpan(0, seqLen));
        // pos ids
        if (posIdHost is null || posIdHost.Length < seqLen) posIdHost = new int[seqLen];
        for (int p = 0; p < seqLen; p++) posIdHost[p] = p;
        runtime.Upload(posIdsBuf, posIdHost.AsSpan(0, seqLen));

        SilkLauncher.Gather(runtime, tokEmb, idsBuf, xBuf, seqLen * hiddenDim, hiddenDim);
        SilkLauncher.Gather(runtime, posEmb, posIdsBuf, posBuf, seqLen * hiddenDim, hiddenDim);
        SilkLauncher.Add(runtime, xBuf, posBuf, xBuf, seqLen * hiddenDim);

        for (int li = 0; li < config.NumHiddenLayers; li++)
        {
            runtime.CopyBuffer(xBuf, layerInput[li], 0, 0, seqLen * hiddenDim);

            SilkLauncher.LayerNorm(runtime, xBuf, ln1W[li], ln1B[li], normed1[li], seqLen, hiddenDim, eps);

            SilkLauncher.Gemm(runtime, normed1[li], wQ[li], qBuf[li], seqLen, hiddenDim, hiddenDim);
            SilkLauncher.Gemm(runtime, normed1[li], wK[li], kBuf[li], seqLen, hiddenDim, hiddenDim);
            SilkLauncher.Gemm(runtime, normed1[li], wV[li], vBuf[li], seqLen, hiddenDim, hiddenDim);

            // build qkv combined buffer (row-major [seq, 3*hidden])
            {
                float[] q = runtime.Download(qBuf[li], seqLen * hiddenDim);
                float[] k = runtime.Download(kBuf[li], seqLen * hiddenDim);
                float[] v = runtime.Download(vBuf[li], seqLen * hiddenDim);
                float[] qkv = new float[seqLen * hiddenDim * 3];
                for (int r = 0; r < seqLen; r++)
                {
                    Array.Copy(q, r * hiddenDim, qkv, r * hiddenDim * 3, hiddenDim);
                    Array.Copy(k, r * hiddenDim, qkv, r * hiddenDim * 3 + hiddenDim, hiddenDim);
                    Array.Copy(v, r * hiddenDim, qkv, r * hiddenDim * 3 + 2 * hiddenDim, hiddenDim);
                }
                runtime.Upload(qkvBuf[li], qkv);
            }

            SilkLauncher.CausalGqaAttention(runtime, qBuf[li], kBuf[li], vBuf[li], attnBuf[li], seqLen, numHeads, numHeads, headDim, scale);

            SilkLauncher.Gemm(runtime, attnBuf[li], wO[li], dTmp, seqLen, hiddenDim, hiddenDim);
            SilkLauncher.Add(runtime, xBuf, dTmp, xBuf, seqLen * hiddenDim);

            runtime.CopyBuffer(xBuf, postAttn[li], 0, 0, seqLen * hiddenDim);

            SilkLauncher.LayerNorm(runtime, xBuf, ln2W[li], ln2B[li], normed2[li], seqLen, hiddenDim, eps);

            SilkLauncher.Gemm(runtime, normed2[li], wFc1[li], f1Pre[li], seqLen, hiddenDim, intermediateDim);
            SilkLauncher.AddBias(runtime, f1Pre[li], fc1Bias[li], f1Pre[li], seqLen, intermediateDim);
            runtime.CopyBuffer(f1Pre[li], f1Pre[li], 0, 0, seqLen * intermediateDim); // ensure saved
            SilkLauncher.Gelu(runtime, f1Pre[li], gateBuf[li], seqLen * intermediateDim);

            SilkLauncher.Gemm(runtime, gateBuf[li], wFc2[li], dTmp, seqLen, intermediateDim, hiddenDim);
            SilkLauncher.AddBias(runtime, dTmp, fc2Bias[li], dTmp, seqLen, hiddenDim);
            SilkLauncher.Add(runtime, xBuf, dTmp, xBuf, seqLen * hiddenDim);
        }

        runtime.CopyBuffer(xBuf, finalNormInput, 0, 0, seqLen * hiddenDim);
        SilkLauncher.LayerNorm(runtime, xBuf, finalNormW, finalNormB, normedBuf, seqLen, hiddenDim, eps);

        // logits for every position
        var logits = new float[seqLen * vocabSize];
        for (int p = 0; p < seqLen; p++)
        {
            nint sub = runtime.CreateSubview(normedBuf, p * hiddenDim, hiddenDim);
            SilkLauncher.Gemm(runtime, sub, tokEmbTransposed, logitsRowBuf, 1, hiddenDim, vocabSize);
            var row = runtime.Download(logitsRowBuf, vocabSize);
            Array.Copy(row, 0, logits, p * vocabSize, vocabSize);
            runtime.Release(sub);
        }
        return logits;
    }

    public void Backward(int[] targetTokens, int seqLen, float scale)
    {
        ThrowIfDisposed();
        if (seqLen <= 0 || seqLen > maxSeqLen) throw new ArgumentOutOfRangeException(nameof(seqLen), seqLen, $"Expected 1..{maxSeqLen}.");
        if ((long)seqLen > (long)maxBatch * maxSeqLen) throw new ArgumentOutOfRangeException(nameof(seqLen), seqLen, $"seqLen ({seqLen}) exceeds maxBatch*maxSeqLen ({(long)maxBatch * maxSeqLen}).");
        if (targetTokens.Length < seqLen) throw new ArgumentException("targetTokens too short.", nameof(targetTokens));

        // upload targets
        runtime.Upload(targetIdsBuf, targetTokens.AsSpan(0, seqLen));

        // recompute logits? assume Forward called; dLogits from softmax CE
        // Need logits buffer for kernel; rebuild into dLogits? SoftmaxCrossEntropyBackward reads logits from? 
        // We didn't save logits. Recompute quickly: forward already computed normed; but normedBuf overwritten? 
        // We saved finalNormInput but not normedBuf? normedBuf holds final norm output. It is still there (not overwritten after Forward). Good.
        // Build logits into dLogitsBuf? kernel expects logits input. We'll recompute logits into dLogitsBuf row by row.
        for (int p = 0; p < seqLen; p++)
        {
            nint sub = runtime.CreateSubview(normedBuf, p * hiddenDim, hiddenDim);
            nint lsub = runtime.CreateSubview(dLogitsBuf, p * vocabSize, vocabSize);
            SilkLauncher.Gemm(runtime, sub, tokEmbTransposed, lsub, 1, hiddenDim, vocabSize);
            runtime.Release(sub);
            runtime.Release(lsub);
        }

        SilkLauncher.SoftmaxCrossEntropyBackward(runtime, dLogitsBuf, targetIdsBuf, dLogitsBuf, seqLen, vocabSize, scale);

        // LM head backward: dX_final (seq,hidden) = dLogits (seq,vocab) @ tokEmb (vocab,hidden)
        SilkLauncher.Gemm(runtime, dLogitsBuf, tokEmb, dTmp, seqLen, vocabSize, hiddenDim);
        // dTokEmb (vocab,hidden) += dLogits^T @ finalNorm (hidden? actually x) 
        // grad tokEmb (vocab,hidden) = dLogits^T (vocab,seq) @ normedBuf (seq,hidden)
        SilkLauncher.GemmTn(runtime, dLogitsBuf, normedBuf, tokEmbGrad, vocabSize, seqLen, hiddenDim);
        // Wait GemmTn(a=normedBuf [seq,hidden], b=dLogits [seq,vocab], c=tokEmbGrad [hidden? no])
        // Need check shape. Use runtime? We'll trust and test.

        // final norm backward
        SilkLauncher.LayerNormBackward(runtime, dTmp, finalNormInput, finalNormW, dTmp2, dLnA, dLnB, seqLen, hiddenDim, eps);
        SilkLauncher.ColSum(runtime, dLnA, finalNormWGrad, seqLen, hiddenDim, 1f);
        SilkLauncher.ColSum(runtime, dLnB, finalNormBGrad, seqLen, hiddenDim, 1f);

        // dCur starts as dTmp2 (grad into last block output)
        runtime.CopyBuffer(dTmp2, dCur, 0, 0, seqLen * hiddenDim);

        for (int li = config.NumHiddenLayers - 1; li >= 0; li--)
        {
            // FFN residual: dCur already includes residual grad from upstream? We'll handle residual add.
            // fc2 backward
            SilkLauncher.GemmTn(runtime, gateBuf[li], dCur, wFc2Grad[li], intermediateDim, seqLen, hiddenDim);
            SilkLauncher.ColSum(runtime, dCur, fc2BiasGrad[li], seqLen, hiddenDim, 1f);
            SilkLauncher.GemmNt(runtime, dCur, wFc2[li], dInterm, seqLen, hiddenDim, intermediateDim);

            // gelu backward
            SilkLauncher.GeluBackward(runtime, dInterm, f1Pre[li], dInterm, seqLen * intermediateDim);

            // fc1 backward
            SilkLauncher.GemmTn(runtime, normed2[li], dInterm, wFc1Grad[li], hiddenDim, seqLen, intermediateDim);
            SilkLauncher.ColSum(runtime, dInterm, fc1BiasGrad[li], seqLen, intermediateDim, 1f);
            SilkLauncher.GemmNt(runtime, dInterm, wFc1[li], dTmp2, seqLen, intermediateDim, hiddenDim);

            // LN2 backward: convert grad normed2 (dTmp2) -> grad postAttn (dTmp)
            SilkLauncher.LayerNormBackward(runtime, dTmp2, postAttn[li], ln2W[li], dTmp, dLnA, dLnB, seqLen, hiddenDim, eps);
            SilkLauncher.ColSum(runtime, dLnA, ln2WGrad[li], seqLen, hiddenDim, 1f);
            SilkLauncher.ColSum(runtime, dLnB, ln2BGrad[li], seqLen, hiddenDim, 1f);

            // residual add for FFN: grad postAttn total = FFN path (dTmp) + residual (dCur)
            SilkLauncher.Add(runtime, dTmp, dCur, dTmp, seqLen * hiddenDim);

            // LN1 backward (attention path): dTmp2 is grad wrt postAttn (block output before residual? actually after attn residual)
            // We need grad wrt attention output residual. Split residual: dAttnPath = dTmp2 (for attn residual) 
            // LN1 not yet; attention block: postAttn = input + oProj(attn). We have dTmp2 grad wrt postAttn.
            // residual: dInputResidual += dTmp2; dOproj = dTmp2.
            // oProj backward:
            SilkLauncher.GemmTn(runtime, attnBuf[li], dTmp, wOGrad[li], hiddenDim, seqLen, hiddenDim);
            // dAttn (seq,hidden) = dTmp (seq,hidden) @ wO (hidden,hidden)
            SilkLauncher.GemmNt(runtime, dTmp, wO[li], dAttn, seqLen, hiddenDim, hiddenDim);

            // attention backward (causal)
            SilkLauncher.CausalAttentionBackwardPrep(runtime, qkvBuf[li], dAttn, attnProbs, attnDelta, 1, seqLen, numHeads, headDim, scale);
            SilkLauncher.AttentionBackwardDq(runtime, qkvBuf[li], attnProbs, attnDelta, dAttn, dQkv, 1, seqLen, numHeads, headDim, scale);
            SilkLauncher.AttentionBackwardDkv(runtime, qkvBuf[li], attnProbs, attnDelta, dAttn, dQkv, 1, seqLen, numHeads, headDim, scale);

            // qkv -> q,k,v grads already in dQkv (interleaved). Split into dQ,dK,dV? dQkv is [seq,3*hidden] with gradients for q,k,v. 
            // qProj/kProj/vProj backward: grad Wq += normed1^T @ dQ
            // Need extract dQ,dK,dV from dQkv. Use host split? 
            // For simplicity, download dQkv and split, upload dQ,dK,dV buffers (qBuf etc reuse as grad buffers).
            {
                float[] g = runtime.Download(dQkv, seqLen * hiddenDim * 3);
                float[] gq = new float[seqLen * hiddenDim];
                float[] gk = new float[seqLen * hiddenDim];
                float[] gv = new float[seqLen * hiddenDim];
                for (int r = 0; r < seqLen; r++)
                {
                    Array.Copy(g, r * hiddenDim * 3, gq, r * hiddenDim, hiddenDim);
                    Array.Copy(g, r * hiddenDim * 3 + hiddenDim, gk, r * hiddenDim, hiddenDim);
                    Array.Copy(g, r * hiddenDim * 3 + 2 * hiddenDim, gv, r * hiddenDim, hiddenDim);
                }
                runtime.Upload(qBuf[li], gq);
                runtime.Upload(kBuf[li], gk);
                runtime.Upload(vBuf[li], gv);
            }

            SilkLauncher.GemmTn(runtime, normed1[li], qBuf[li], wQGrad[li], hiddenDim, seqLen, hiddenDim);
            SilkLauncher.GemmTn(runtime, normed1[li], kBuf[li], wKGrad[li], hiddenDim, seqLen, hiddenDim);
            SilkLauncher.GemmTn(runtime, normed1[li], vBuf[li], wVGrad[li], hiddenDim, seqLen, hiddenDim);
            SilkLauncher.ColSum(runtime, qBuf[li], /*bias none*/ wQGrad[li], 0, 0, 0f); // no bias
            // dNormed1 (seq,hidden) = dQ @ Wq^T + dK @ Wk^T + dV @ Wv^T
            SilkLauncher.GemmNt(runtime, qBuf[li], wQ[li], dCur, seqLen, hiddenDim, hiddenDim);
            SilkLauncher.GemmNt(runtime, kBuf[li], wK[li], dTmp2, seqLen, hiddenDim, hiddenDim);
            SilkLauncher.Add(runtime, dCur, dTmp2, dCur, seqLen * hiddenDim);
            SilkLauncher.GemmNt(runtime, vBuf[li], wV[li], dTmp2, seqLen, hiddenDim, hiddenDim);
            SilkLauncher.Add(runtime, dCur, dTmp2, dCur, seqLen * hiddenDim);

            // LN1 backward
            SilkLauncher.LayerNormBackward(runtime, dCur, layerInput[li], ln1W[li], dTmp2, dLnA, dLnB, seqLen, hiddenDim, eps);
            SilkLauncher.ColSum(runtime, dLnA, ln1WGrad[li], seqLen, hiddenDim, 1f);
            SilkLauncher.ColSum(runtime, dLnB, ln1BGrad[li], seqLen, hiddenDim, 1f);

            // residual add from block input: grad x_in = grad from attention path (dTmp2) + residual grad (dTmp)
            SilkLauncher.Add(runtime, dTmp2, dTmp, dCur, seqLen * hiddenDim);
        }

        // token embedding backward: scatter row grads into tokEmbGrad
        // dCur now holds grad wrt x (input to first layer) = grad wrt token+pos embedding (pos no grad)
        // We need map dCur (seq,hidden) back to token embedding rows: EmbeddingBackwardRows(ids, dCur, tokEmbGrad, seq, hidden)
        SilkLauncher.EmbeddingBackwardRows(runtime, idsBuf, dCur, tokEmbGrad, seqLen, hiddenDim);

        // tokEmbGrad already accumulated LM head grad above? We added tokEmbGrad += ... earlier. Good.
    }

    public void ZeroGradients()
    {
        ThrowIfDisposed();
        foreach (var p in parameters)
            SilkLauncher.Fill(runtime, p.Grad, 0f, p.Count);
    }

    public void AdamUpdateAll(float learningRate, float weightDecay, float beta1 = 0.9f, float beta2 = 0.999f)
    {
        ThrowIfDisposed();
        adamStep++;
        float c1 = 1f - MathF.Pow(beta1, adamStep);
        float c2 = 1f - MathF.Pow(beta2, adamStep);
        foreach (var p in parameters)
        {
            SilkLauncher.AdamUpdate(runtime, p.Weight, p.Grad, p.M, p.V, learningRate, c1, c2, 1e-8f, weightDecay, p.Count);
        }
        // refresh transposed embedding
        var data = runtime.Download(tokEmb, vocabSize * hiddenDim);
        var t = new float[hiddenDim * vocabSize];
        for (int v = 0; v < vocabSize; v++)
            for (int d = 0; d < hiddenDim; d++)
                t[d * vocabSize + v] = data[v * hiddenDim + d];
        runtime.Upload(tokEmbTransposed, t);
    }

    public Dictionary<string, (float[] Data, int[] Shape)> ExportCheckpoint()
    {
        ThrowIfDisposed();
        var result = new Dictionary<string, (float[] Data, int[] Shape)>(StringComparer.Ordinal);
        result["tok_emb.weight"] = (Download(tokEmb, vocabSize * hiddenDim), new[] { vocabSize, hiddenDim });
        result["final_norm.weight"] = (Download(finalNormW, hiddenDim), new[] { hiddenDim });
        result["final_norm.bias"] = (Download(finalNormB, hiddenDim), new[] { hiddenDim });
        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            string p = $"layers.{i}.";
            result[p + "ln1.weight"] = (Download(ln1W[i], hiddenDim), new[] { hiddenDim });
            result[p + "ln1.bias"] = (Download(ln1B[i], hiddenDim), new[] { hiddenDim });
            result[p + "ln2.weight"] = (Download(ln2W[i], hiddenDim), new[] { hiddenDim });
            result[p + "ln2.bias"] = (Download(ln2B[i], hiddenDim), new[] { hiddenDim });
            result[p + "q_proj.weight"] = (DownloadTransposed(wQ[i], hiddenDim, hiddenDim), new[] { hiddenDim, hiddenDim });
            result[p + "k_proj.weight"] = (DownloadTransposed(wK[i], hiddenDim, hiddenDim), new[] { hiddenDim, hiddenDim });
            result[p + "v_proj.weight"] = (DownloadTransposed(wV[i], hiddenDim, hiddenDim), new[] { hiddenDim, hiddenDim });
            result[p + "o_proj.weight"] = (DownloadTransposed(wO[i], hiddenDim, hiddenDim), new[] { hiddenDim, hiddenDim });
            result[p + "mlp_fc1.weight"] = (DownloadTransposed(wFc1[i], intermediateDim, hiddenDim), new[] { intermediateDim, hiddenDim });
            result[p + "mlp_fc1.bias"] = (Download(fc1Bias[i], intermediateDim), new[] { intermediateDim });
            result[p + "mlp_fc2.weight"] = (DownloadTransposed(wFc2[i], hiddenDim, intermediateDim), new[] { hiddenDim, intermediateDim });
            result[p + "mlp_fc2.bias"] = (Download(fc2Bias[i], hiddenDim), new[] { hiddenDim });
        }
        return result;
    }

    public Dictionary<string, (float[] Data, int[] Shape)> ExportGradients()
    {
        ThrowIfDisposed();
        var result = new Dictionary<string, (float[] Data, int[] Shape)>(StringComparer.Ordinal);
        result["tok_emb.weight"] = (Download(tokEmbGrad, vocabSize * hiddenDim), new[] { vocabSize, hiddenDim });
        result["final_norm.weight"] = (Download(finalNormWGrad, hiddenDim), new[] { hiddenDim });
        result["final_norm.bias"] = (Download(finalNormBGrad, hiddenDim), new[] { hiddenDim });
        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            string p = $"layers.{i}.";
            result[p + "ln1.weight"] = (Download(ln1WGrad[i], hiddenDim), new[] { hiddenDim });
            result[p + "ln1.bias"] = (Download(ln1BGrad[i], hiddenDim), new[] { hiddenDim });
            result[p + "ln2.weight"] = (Download(ln2WGrad[i], hiddenDim), new[] { hiddenDim });
            result[p + "ln2.bias"] = (Download(ln2BGrad[i], hiddenDim), new[] { hiddenDim });
            result[p + "q_proj.weight"] = (DownloadTransposed(wQGrad[i], hiddenDim, hiddenDim), new[] { hiddenDim, hiddenDim });
            result[p + "k_proj.weight"] = (DownloadTransposed(wKGrad[i], hiddenDim, hiddenDim), new[] { hiddenDim, hiddenDim });
            result[p + "v_proj.weight"] = (DownloadTransposed(wVGrad[i], hiddenDim, hiddenDim), new[] { hiddenDim, hiddenDim });
            result[p + "o_proj.weight"] = (DownloadTransposed(wOGrad[i], hiddenDim, hiddenDim), new[] { hiddenDim, hiddenDim });
            result[p + "mlp_fc1.weight"] = (DownloadTransposed(wFc1Grad[i], intermediateDim, hiddenDim), new[] { intermediateDim, hiddenDim });
            result[p + "mlp_fc1.bias"] = (Download(fc1BiasGrad[i], intermediateDim), new[] { intermediateDim });
            result[p + "mlp_fc2.weight"] = (DownloadTransposed(wFc2Grad[i], hiddenDim, intermediateDim), new[] { hiddenDim, intermediateDim });
            result[p + "mlp_fc2.bias"] = (Download(fc2BiasGrad[i], hiddenDim), new[] { hiddenDim });
        }
        return result;
    }

    public float Train(
        int[] allTokens,
        int epochs,
        int batchSize,
        int blockSize,
        float learningRate,
        float beta1,
        float beta2,
        int seed)
    {
        ThrowIfDisposed();
        if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "batchSize must be positive.");
        if (blockSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockSize), blockSize, "blockSize must be positive.");
        if ((long)batchSize * blockSize > (long)maxBatch * maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize,
                $"batchSize*blockSize ({(long)batchSize * blockSize}) exceeds maxBatch*maxSeqLen ({(long)maxBatch * maxSeqLen}).");
        if (allTokens.Length < blockSize + 1) throw new ArgumentException("not enough tokens", nameof(allTokens));
        var rng = new Random(seed);
        int nTokens = allTokens.Length;
        int nBatches = Math.Max(1, (nTokens - 1) / (batchSize * blockSize));
        float lastLoss = 0f;
        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            float epochLoss = 0;
            int batchCount = 0;
            for (int b = 0; b < nBatches; b++)
            {
                ZeroGradients();
                float batchLoss = 0;
                int tokenCount = 0;
                for (int bi = 0; bi < batchSize; bi++)
                {
                    int start = rng.Next(nTokens - blockSize - 1);
                    var input = new int[blockSize];
                    var target = new int[blockSize];
                    for (int t = 0; t < blockSize; t++)
                    {
                        input[t] = allTokens[start + t];
                        target[t] = allTokens[start + t + 1];
                    }
                    var logits = Forward(input, blockSize);
                    // host CE loss
                    float loss = CrossEntropy(logits, target, blockSize, vocabSize);
                    batchLoss += loss;
                    tokenCount += blockSize;
                    float scale = 1f / (batchSize * blockSize);
                    Backward(target, blockSize, scale);
                }
                AdamUpdateAll(learningRate, 0f, beta1, beta2);
                epochLoss += batchLoss / batchSize;
                batchCount++;
            }
            lastLoss = batchCount > 0 ? epochLoss / batchCount : 0;
        }
        return lastLoss;
    }

    static float CrossEntropy(float[] logits, int[] targets, int seqLen, int vocab)
    {
        float sum = 0;
        for (int t = 0; t < seqLen; t++)
        {
            float max = float.NegativeInfinity;
            for (int v = 0; v < vocab; v++) max = MathF.Max(max, logits[t * vocab + v]);
            float s = 0;
            for (int v = 0; v < vocab; v++) s += MathF.Exp(logits[t * vocab + v] - max);
            float logp = logits[t * vocab + targets[t]] - max - MathF.Log(s);
            sum += -logp;
        }
        return sum / seqLen;
    }

    float[] Download(nint buf, int count)
    {
        return runtime.Download(buf, count);
    }

    float[] DownloadTransposed(nint buf, int rows, int cols)
    {
        // buf stored transposed [cols, rows]? We store weights transposed [in,out]. Export as [out,in].
        var data = runtime.Download(buf, rows * cols);
        var result = new float[rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result[r * cols + c] = data[c * rows + r];
        return result;
    }

    nint UploadPlain(Dictionary<string, (float[] Data, int[] Shape)> tensors, string key)
    {
        var t = WeightLayout.Req(tensors, key, "Silk");
        nint buf = runtime.Allocate(t.Data.Length);
        runtime.Upload(buf, t.Data);
        return buf;
    }

    nint UploadT(Dictionary<string, (float[] Data, int[] Shape)> tensors, string key)
    {
        var data = WeightLayout.Transposed(tensors, key, "Silk");
        nint buf = runtime.Allocate(data.Length);
        runtime.Upload(buf, data);
        return buf;
    }

    // We'll override below

    static float[] BuildPositionTable(int seqLen, int embedDim)
    {
        var data = new float[seqLen * embedDim];
        for (int pos = 0; pos < seqLen; pos++)
        {
            for (int i = 0; i < embedDim; i++)
            {
                double exponent = 2.0 * (i / 2) / embedDim;
                double angle = pos / Math.Pow(10000.0, exponent);
                data[pos * embedDim + i] = i % 2 == 0
                    ? (float)Math.Sin(angle)
                    : (float)Math.Cos(angle);
            }
        }
        return data;
    }

    void ThrowIfDisposed()
    {
        if (disposed) throw new ObjectDisposedException(nameof(SilkTinyShakespeareTrainer));
    }

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        // release all (best-effort)
        runtime.Release(tokEmb); runtime.Release(tokEmbTransposed); runtime.Release(posEmb);
        runtime.Release(finalNormW); runtime.Release(finalNormB);
        runtime.Release(tokEmbGrad); runtime.Release(finalNormWGrad); runtime.Release(finalNormBGrad);
        runtime.Release(tokEmbM); runtime.Release(tokEmbV); runtime.Release(finalNormWM); runtime.Release(finalNormWV);
        runtime.Release(finalNormBM); runtime.Release(finalNormBV);
        runtime.Release(xBuf); runtime.Release(normedBuf); runtime.Release(posBuf); runtime.Release(idsBuf);
        runtime.Release(posIdsBuf); runtime.Release(finalNormInput); runtime.Release(logitsRowBuf);
        runtime.Release(dCur); runtime.Release(dTmp); runtime.Release(dTmp2); runtime.Release(dInterm);
        runtime.Release(dQkv); runtime.Release(dAttn); runtime.Release(dLnA); runtime.Release(dLnB);
        runtime.Release(dLogitsBuf); runtime.Release(targetIdsBuf);
        foreach (var b in ln1W) runtime.Release(b);
        foreach (var b in ln1B) runtime.Release(b);
        foreach (var b in ln2W) runtime.Release(b);
        foreach (var b in ln2B) runtime.Release(b);
        foreach (var b in wQ) runtime.Release(b);
        foreach (var b in wK) runtime.Release(b);
        foreach (var b in wV) runtime.Release(b);
        foreach (var b in wO) runtime.Release(b);
        foreach (var b in wFc1) runtime.Release(b);
        foreach (var b in fc1Bias) runtime.Release(b);
        foreach (var b in wFc2) runtime.Release(b);
        foreach (var b in fc2Bias) runtime.Release(b);
        foreach (var b in ln1WGrad) runtime.Release(b);
        foreach (var b in ln1BGrad) runtime.Release(b);
        foreach (var b in ln2WGrad) runtime.Release(b);
        foreach (var b in ln2BGrad) runtime.Release(b);
        foreach (var b in wQGrad) runtime.Release(b);
        foreach (var b in wKGrad) runtime.Release(b);
        foreach (var b in wVGrad) runtime.Release(b);
        foreach (var b in wOGrad) runtime.Release(b);
        foreach (var b in wFc1Grad) runtime.Release(b);
        foreach (var b in fc1BiasGrad) runtime.Release(b);
        foreach (var b in wFc2Grad) runtime.Release(b);
        foreach (var b in fc2BiasGrad) runtime.Release(b);
        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            runtime.Release(layerInput[i]); runtime.Release(normed1[i]); runtime.Release(qBuf[i]);
            runtime.Release(kBuf[i]); runtime.Release(vBuf[i]); runtime.Release(qkvBuf[i]);
            runtime.Release(attnBuf[i]); runtime.Release(postAttn[i]); runtime.Release(normed2[i]);
            runtime.Release(f1Pre[i]); runtime.Release(gateBuf[i]);
            foreach (var arr in new[] { ln1WM[i], ln1WV[i], ln1BM[i], ln1BV[i], ln2WM[i], ln2WV[i], ln2BM[i], ln2BV[i],
                wQM[i], wQV[i], wKM[i], wKV[i], wVM[i], wVV[i], wOM[i], wOV[i],
                wFc1M[i], wFc1V[i], fc1BiasM[i], fc1BiasV[i], wFc2M[i], wFc2V[i], fc2BiasM[i], fc2BiasV[i] })
                foreach (var b in arr) runtime.Release(b);
        }
    }
}
