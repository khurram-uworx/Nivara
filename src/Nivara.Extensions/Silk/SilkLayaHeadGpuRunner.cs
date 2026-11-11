using Nivara.AI;
using System.Diagnostics;

namespace Nivara.Silk;

/// <summary>
/// Laya's decision head on the Silk runtime, operation for operation with
/// <c>LayaHeadGpuRunner</c>. Norms are biased and use eps 1e-5, not the encoder's.
/// The four act-head features stay on <see cref="LayaHeadScoring"/>.
/// </summary>
public sealed partial class SilkLayaHeadGpuRunner : IDisposable
{
    readonly SilkRuntime runtime;
    readonly int hiddenDim;
    readonly int numHeads;
    readonly int headDim;
    readonly int intermediateDim;
    readonly int actClassCount;
    readonly int actHidden;
    readonly float eps;
    readonly float scale;

    readonly bool narrowBf16;
    readonly bool narrowF16;
    readonly bool activationBf16;
    readonly bool activationF16;

    readonly nint typeEmb;
    readonly nint scorerNormW;
    readonly nint scorerNormB;
    readonly nint scorerHiddenW;
    readonly nint scorerHiddenB;
    readonly nint scorerOutW;
    readonly nint scorerOutB;
    readonly nint actInW;
    readonly nint actInB;
    readonly nint actOutW;
    readonly nint actOutB;
    readonly LayerWeights[] layers;
    readonly nint pooledIds;

    SilkRuntime.SilkBuffer normed;
    SilkRuntime.SilkBuffer qkv;
    SilkRuntime.SilkBuffer context;
    SilkRuntime.SilkBuffer projected;
    SilkRuntime.SilkBuffer ffnIn;
    SilkRuntime.SilkBuffer ffnOut;
    SilkRuntime.SilkBuffer markerRows;
    SilkRuntime.SilkBuffer markerScored;
    SilkRuntime.SilkBuffer markerScoredOut;
    SilkRuntime.SilkBuffer logits;
    SilkRuntime.SilkBuffer actRow;
    SilkRuntime.SilkBuffer actHiddenBuf;
    SilkRuntime.SilkBuffer actLogits;
    SilkRuntime.SilkBuffer maskBuf;
    SilkRuntime.SilkBuffer markerIds;

    int[] markerIdHost = [];

    public bool ProfilingEnabled { get; set; }
    public readonly Dictionary<string, long> LastPhaseMs = new();

    public SilkLayaHeadGpuRunner(
        SilkRuntime runtime,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        int headLayerCount = 2,
        int actClassCount = 2)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(tensors);
        if (headLayerCount < 0)
            throw new ArgumentOutOfRangeException(nameof(headLayerCount), headLayerCount, "headLayerCount cannot be negative");
        if (actClassCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(actClassCount), actClassCount, "actClassCount must be positive");

        this.runtime = runtime;
        this.actClassCount = actClassCount;

        if (!tensors.TryGetValue("type_emb.weight", out var typeWeight))
            throw new KeyNotFoundException(
                "Missing tensor: type_emb.weight. A stock encoder checkpoint is not a Laya decision " +
                "model; the head is what makes it one.");
        hiddenDim = typeWeight.Shape[1];
        if (typeWeight.Shape[0] != LayaDecisionHead<float>.QuestionTypeCount)
            throw new ArgumentException(
                $"type_emb.weight has {typeWeight.Shape[0]} rows, expected " +
                $"{LayaDecisionHead<float>.QuestionTypeCount} question types.", nameof(tensors));

        numHeads = Math.Max(1, hiddenDim / LayaDecisionHead<float>.HeadDimTarget);
        if (hiddenDim % numHeads != 0)
            throw new ArgumentException(
                $"hiddenSize {hiddenDim} does not divide into {numHeads} heads at " +
                $"{LayaDecisionHead<float>.HeadDimTarget} wide.", nameof(tensors));
        headDim = hiddenDim / numHeads;
        intermediateDim = 4 * hiddenDim;
        actHidden = LayaDecisionHead<float>.ActHeadHidden;
        eps = 1e-5f;
        scale = 1f / MathF.Sqrt(headDim);
        this.narrowBf16 = false;
        this.narrowF16 = false;
        runtime.EnsureAttentionLocalMemory(headDim);

        try
        {
            typeEmb = UploadPlain(tensors, "type_emb.weight");
            scorerNormW = UploadPlain(tensors, "scorer.0.weight");
            scorerNormB = UploadPlain(tensors, "scorer.0.bias");
            scorerHiddenW = UploadTransposed(tensors, "scorer.1.weight");
            scorerHiddenB = UploadPlain(tensors, "scorer.1.bias");
            scorerOutW = UploadTransposed(tensors, "scorer.3.weight");
            scorerOutB = UploadPlain(tensors, "scorer.3.bias");
            actInW = UploadTransposed(tensors, "act_head.0.weight");
            actInB = UploadPlain(tensors, "act_head.0.bias");
            actOutW = UploadTransposed(tensors, "act_head.2.weight");
            actOutB = UploadPlain(tensors, "act_head.2.bias");

            layers = new LayerWeights[headLayerCount];
            for (int i = 0; i < layers.Length; i++)
                layers[i] = new LayerWeights(this, tensors, i);

            pooledIds = runtime.Allocate(1);
            runtime.Upload(pooledIds, new[] { 0 });
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public SilkLayaHeadGpuRunner(
        SilkRuntime runtime,
        Dictionary<string, (ushort[] Data, int[] Shape)> tensors,
        int headLayerCount = 2,
        int actClassCount = 2,
        bool isBf16 = false,
        bool isFp16 = false,
        bool activationBf16 = false,
        bool activationF16 = false)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(tensors);
        if (headLayerCount < 0)
            throw new ArgumentOutOfRangeException(nameof(headLayerCount), headLayerCount, "headLayerCount cannot be negative");
        if (actClassCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(actClassCount), actClassCount, "actClassCount must be positive");

        this.runtime = runtime;
        this.actClassCount = actClassCount;
        this.narrowBf16 = isBf16;
        this.narrowF16 = isFp16;
        this.activationBf16 = activationBf16;
        this.activationF16 = activationF16;

        if (!tensors.TryGetValue("type_emb.weight", out var typeWeight))
            throw new KeyNotFoundException(
                "Missing tensor: type_emb.weight. A stock encoder checkpoint is not a Laya decision " +
                "model; the head is what makes it one.");
        hiddenDim = typeWeight.Shape[1];
        if (typeWeight.Shape[0] != LayaDecisionHead<float>.QuestionTypeCount)
            throw new ArgumentException(
                $"type_emb.weight has {typeWeight.Shape[0]} rows, expected " +
                $"{LayaDecisionHead<float>.QuestionTypeCount} question types.", nameof(tensors));

        numHeads = Math.Max(1, hiddenDim / LayaDecisionHead<float>.HeadDimTarget);
        if (hiddenDim % numHeads != 0)
            throw new ArgumentException(
                $"hiddenSize {hiddenDim} does not divide into {numHeads} heads at " +
                $"{LayaDecisionHead<float>.HeadDimTarget} wide.", nameof(tensors));
        headDim = hiddenDim / numHeads;
        intermediateDim = 4 * hiddenDim;
        actHidden = LayaDecisionHead<float>.ActHeadHidden;
        eps = 1e-5f;
        scale = 1f / MathF.Sqrt(headDim);
        runtime.EnsureAttentionLocalMemory(headDim);

        Func<ushort, float> widen = isBf16
            ? new Func<ushort, float>(u => (float)BitConverter.UInt16BitsToBFloat16(u))
            : new Func<ushort, float>(u => (float)BitConverter.Int16BitsToHalf((short)u));

        try
        {
            typeEmb = UploadUshortAsFloat(tensors, "type_emb.weight", widen);
            scorerNormW = UploadUshortAsFloat(tensors, "scorer.0.weight", widen);
            scorerNormB = UploadUshortAsFloat(tensors, "scorer.0.bias", widen);
            scorerHiddenW = UploadTransposedUshort(tensors, "scorer.1.weight");
            scorerHiddenB = UploadUshortAsFloat(tensors, "scorer.1.bias", widen);
            scorerOutW = UploadTransposedUshort(tensors, "scorer.3.weight");
            scorerOutB = UploadUshortAsFloat(tensors, "scorer.3.bias", widen);
            actInW = UploadTransposedUshort(tensors, "act_head.0.weight");
            actInB = UploadUshortAsFloat(tensors, "act_head.0.bias", widen);
            actOutW = UploadTransposedUshort(tensors, "act_head.2.weight");
            actOutB = UploadUshortAsFloat(tensors, "act_head.2.bias", widen);

            layers = new LayerWeights[headLayerCount];
            for (int i = 0; i < layers.Length; i++)
                layers[i] = new LayerWeights(this, tensors, i, isBf16, isFp16);

            pooledIds = runtime.Allocate(1);
            runtime.Upload(pooledIds, new[] { 0 });
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public int HiddenSize => hiddenDim;

    public LayaHeadOutput Forward(
        nint hidden,
        int rows,
        LayaQuestionType type,
        IReadOnlyList<int> markerPositions,
        int validLength)
    {
        ArgumentNullException.ThrowIfNull(markerPositions);
        if (rows <= 0)
            throw new ArgumentOutOfRangeException(nameof(rows), $"rows must be non-empty, got {rows}.");
        if (validLength <= 0 || validLength > rows)
            throw new ArgumentOutOfRangeException(nameof(validLength), $"validLength must be within [1, {rows}], got {validLength}.");
        if (markerPositions.Count == 0)
            throw new ArgumentException(
                "No marker positions survived. Every option's [MASK] was pushed out by the length " +
                "budget, so the question does not fit head_max_len.", nameof(markerPositions));
        if (hidden == 0)
            throw new ArgumentException("The encoder buffer is empty. Forward the encoder before the head.", nameof(hidden));

        int k = markerPositions.Count;
        EnsureWorkspace(rows, k);
        if (activationBf16)
        {
            float[] __h = runtime.Download(hidden, rows * hiddenDim);
            var __u = new ushort[__h.Length];
            for (int __i = 0; __i < __h.Length; __i++) __u[__i] = (ushort)(BitConverter.SingleToInt32Bits(__h[__i]) >> 16);
            hidden = runtime.AllocateUshort(__u.Length);
            runtime.Upload(hidden, __u);
        }
        if (ProfilingEnabled) { var _es = Stopwatch.StartNew(); runtime.Finish(); LastPhaseMs["EncodeDrain"] = _es.ElapsedMilliseconds; }

        int typeOffset = (int)type * hiddenDim;
        nint typeRow = runtime.CreateSubview(typeEmb, typeOffset, hiddenDim);
        try
        {
            var _sw = ProfilingEnabled ? Stopwatch.StartNew() : null;
            LaunchAddBias(hidden, typeRow, hidden, rows, hiddenDim);
            if (ProfilingEnabled) { runtime.Finish(); LastPhaseMs["AddBias"] = _sw!.ElapsedMilliseconds; }
        }
        finally
        {
            runtime.Release(typeRow);
        }

        if (layers.Length > 0)
        {
            runtime.Upload(maskBuf.Handle, LayaHeadScoring.PrefixMask(rows, validLength));
            nint q, kv, v;
            if (activationBf16)
            {
                q = runtime.CreateSubviewUshort(qkv.Handle, 0, rows * hiddenDim);
                kv = runtime.CreateSubviewUshort(qkv.Handle, rows * hiddenDim, rows * hiddenDim);
                v = runtime.CreateSubviewUshort(qkv.Handle, 2 * rows * hiddenDim, rows * hiddenDim);
            }
            else
            {
                q = runtime.CreateSubview(qkv.Handle, 0, rows * hiddenDim);
                kv = runtime.CreateSubview(qkv.Handle, rows * hiddenDim, rows * hiddenDim);
                v = runtime.CreateSubview(qkv.Handle, 2 * rows * hiddenDim, rows * hiddenDim);
            }
            try
            {
                long _norm = 0, _gemm = 0, _attn = 0, _add = 0;
                var _layerSw = ProfilingEnabled ? Stopwatch.StartNew() : null;
                for (int i = 0; i < layers.Length; i++)
                {
                    LayerWeights w = layers[i];
                    if (ProfilingEnabled) { var s = Stopwatch.StartNew(); LaunchLayerNorm(hidden, w.Norm1W, w.Norm1B, normed.Handle, rows); runtime.Finish(); _norm += s.ElapsedMilliseconds; }
                    else LaunchLayerNorm(hidden, w.Norm1W, w.Norm1B, normed.Handle, rows);
                    if (ProfilingEnabled) { var s = Stopwatch.StartNew(); LaunchGemm(SilkKernels.GemmQkv, normed.Handle, w.Wqkv, qkv.Handle, w.Bqkv, rows, hiddenDim, 3 * hiddenDim, hiddenDim); runtime.Finish(); _gemm += s.ElapsedMilliseconds; }
                    else LaunchGemm(SilkKernels.GemmQkv, normed.Handle, w.Wqkv, qkv.Handle, w.Bqkv, rows, hiddenDim, 3 * hiddenDim, hiddenDim);
                    if (ProfilingEnabled) { var s = Stopwatch.StartNew(); LaunchAttention(q, kv, v, maskBuf.Handle, context.Handle, rows); runtime.Finish(); _attn += s.ElapsedMilliseconds; }
                    else LaunchAttention(q, kv, v, maskBuf.Handle, context.Handle, rows);
                    if (ProfilingEnabled) { var s = Stopwatch.StartNew(); LaunchGemm(SilkKernels.GemmBias, context.Handle, w.Wo, projected.Handle, w.Bo, rows, hiddenDim, hiddenDim); runtime.Finish(); _gemm += s.ElapsedMilliseconds; }
                    else LaunchGemm(SilkKernels.GemmBias, context.Handle, w.Wo, projected.Handle, w.Bo, rows, hiddenDim, hiddenDim);
                    if (ProfilingEnabled) { var s = Stopwatch.StartNew(); LaunchAdd(hidden, projected.Handle, hidden, rows * hiddenDim); runtime.Finish(); _add += s.ElapsedMilliseconds; }
                    else LaunchAdd(hidden, projected.Handle, hidden, rows * hiddenDim);
                    if (ProfilingEnabled) { var s = Stopwatch.StartNew(); LaunchLayerNorm(hidden, w.Norm2W, w.Norm2B, normed.Handle, rows); runtime.Finish(); _norm += s.ElapsedMilliseconds; }
                    else LaunchLayerNorm(hidden, w.Norm2W, w.Norm2B, normed.Handle, rows);
                    if (ProfilingEnabled) { var s = Stopwatch.StartNew(); LaunchGemm(SilkKernels.GemmRelu, normed.Handle, w.W1, ffnIn.Handle, w.B1, rows, hiddenDim, intermediateDim); runtime.Finish(); _gemm += s.ElapsedMilliseconds; }
                    else LaunchGemm(SilkKernels.GemmRelu, normed.Handle, w.W1, ffnIn.Handle, w.B1, rows, hiddenDim, intermediateDim);
                    if (ProfilingEnabled) { var s = Stopwatch.StartNew(); LaunchGemm(SilkKernels.GemmBias, ffnIn.Handle, w.W2, ffnOut.Handle, w.B2, rows, intermediateDim, hiddenDim); runtime.Finish(); _gemm += s.ElapsedMilliseconds; }
                    else LaunchGemm(SilkKernels.GemmBias, ffnIn.Handle, w.W2, ffnOut.Handle, w.B2, rows, intermediateDim, hiddenDim);
                    if (ProfilingEnabled) { var s = Stopwatch.StartNew(); LaunchAdd(hidden, ffnOut.Handle, hidden, rows * hiddenDim); runtime.Finish(); _add += s.ElapsedMilliseconds; }
                    else LaunchAdd(hidden, ffnOut.Handle, hidden, rows * hiddenDim);
                }
                if (ProfilingEnabled)
                {
                    LastPhaseMs["Norm"] = _norm;
                    LastPhaseMs["Gemm"] = _gemm;
                    LastPhaseMs["Attention"] = _attn;
                    LastPhaseMs["Add"] = _add;
                    LastPhaseMs["LayerLoop"] = _layerSw!.ElapsedMilliseconds;
                }
            }
            finally
            {
                runtime.Release(q);
                runtime.Release(kv);
                runtime.Release(v);
            }
        }

        if (markerIdHost.Length < k)
            markerIdHost = new int[k];
        for (int i = 0; i < k; i++)
            markerIdHost[i] = markerPositions[i];
        runtime.Upload(markerIds.Handle, markerIdHost.AsSpan(0, k));
        long _markerMs = 0;
        var _mSw = ProfilingEnabled ? Stopwatch.StartNew() : null;
        if (ProfilingEnabled) { var s = Stopwatch.StartNew(); LaunchGather(hidden, markerIds.Handle, markerRows.Handle, k * hiddenDim); runtime.Finish(); _markerMs += s.ElapsedMilliseconds; }
        else LaunchGather(hidden, markerIds.Handle, markerRows.Handle, k * hiddenDim);
        if (ProfilingEnabled) { var s = Stopwatch.StartNew(); LaunchLayerNorm(markerRows.Handle, scorerNormW, scorerNormB, markerScored.Handle, k); runtime.Finish(); _markerMs += s.ElapsedMilliseconds; }
        else LaunchLayerNorm(markerRows.Handle, scorerNormW, scorerNormB, markerScored.Handle, k);
        if (ProfilingEnabled) { var s = Stopwatch.StartNew(); LaunchGemm(SilkKernels.GemmGelu, markerScored.Handle, scorerHiddenW, markerScoredOut.Handle, scorerHiddenB, k, hiddenDim, hiddenDim); runtime.Finish(); _markerMs += s.ElapsedMilliseconds; }
        else LaunchGemm(SilkKernels.GemmGelu, markerScored.Handle, scorerHiddenW, markerScoredOut.Handle, scorerHiddenB, k, hiddenDim, hiddenDim);
        if (ProfilingEnabled) { var s = Stopwatch.StartNew(); LaunchGemm(SilkKernels.GemmBias, markerScoredOut.Handle, scorerOutW, logits.Handle, scorerOutB, k, hiddenDim, 1); runtime.Finish(); _markerMs += s.ElapsedMilliseconds; }
        else LaunchGemm(SilkKernels.GemmBias, markerScoredOut.Handle, scorerOutW, logits.Handle, scorerOutB, k, hiddenDim, 1);
        float[] logitsHost;
        if (activationBf16)
        {
            ushort[] __l = runtime.DownloadUshort(logits.Handle, k);
            logitsHost = new float[__l.Length];
            for (int __i = 0; __i < __l.Length; __i++) logitsHost[__i] = (float)BitConverter.UInt16BitsToBFloat16(__l[__i]);
        }
        else logitsHost = runtime.Download(logits.Handle, k);
        if (ProfilingEnabled) LastPhaseMs["MarkerHead"] = _markerMs;

        double[] probabilities = LayaHeadScoring.Softmax(logitsHost);
        double[] features = LayaHeadScoring.BuildFeatures(probabilities);

        long _actMs = 0;
        if (ProfilingEnabled) { var s = Stopwatch.StartNew(); LaunchGather(hidden, pooledIds, actRow.Handle, hiddenDim); runtime.Finish(); _actMs += s.ElapsedMilliseconds; }
        else LaunchGather(hidden, pooledIds, actRow.Handle, hiddenDim);
        float[] actRowHost;
        if (activationBf16)
        {
            ushort[] __a = runtime.DownloadUshort(actRow.Handle, hiddenDim + LayaHeadOutput.FeatureCount);
            actRowHost = new float[__a.Length];
            for (int __i = 0; __i < __a.Length; __i++) actRowHost[__i] = (float)BitConverter.UInt16BitsToBFloat16(__a[__i]);
        }
        else actRowHost = runtime.Download(actRow.Handle, hiddenDim + LayaHeadOutput.FeatureCount);
        for (int i = 0; i < features.Length; i++)
            actRowHost[hiddenDim + i] = (float)features[i];
        if (activationBf16)
        {
            var __u = new ushort[actRowHost.Length];
            for (int __i = 0; __i < actRowHost.Length; __i++) __u[__i] = (ushort)(BitConverter.SingleToInt32Bits(actRowHost[__i]) >> 16);
            runtime.Upload(actRow.Handle, __u);
        }
        else runtime.Upload(actRow.Handle, actRowHost);

        if (ProfilingEnabled) { var s = Stopwatch.StartNew(); LaunchGemm(SilkKernels.GemmGelu, actRow.Handle, actInW, actHiddenBuf.Handle, actInB, 1, hiddenDim + LayaHeadOutput.FeatureCount, actHidden); runtime.Finish(); _actMs += s.ElapsedMilliseconds; }
        else LaunchGemm(SilkKernels.GemmGelu, actRow.Handle, actInW, actHiddenBuf.Handle, actInB, 1, hiddenDim + LayaHeadOutput.FeatureCount, actHidden);
        if (ProfilingEnabled) { var s = Stopwatch.StartNew(); LaunchGemm(SilkKernels.GemmBias, actHiddenBuf.Handle, actOutW, actLogits.Handle, actOutB, 1, actHidden, actClassCount); runtime.Finish(); _actMs += s.ElapsedMilliseconds; }
        else LaunchGemm(SilkKernels.GemmBias, actHiddenBuf.Handle, actOutW, actLogits.Handle, actOutB, 1, actHidden, actClassCount);
        float[] actHost;
        if (activationBf16)
        {
            ushort[] __a = runtime.DownloadUshort(actLogits.Handle, actClassCount);
            actHost = new float[__a.Length];
            for (int __i = 0; __i < __a.Length; __i++) actHost[__i] = (float)BitConverter.UInt16BitsToBFloat16(__a[__i]);
        }
        else actHost = runtime.Download(actLogits.Handle, actClassCount);
        if (ProfilingEnabled) LastPhaseMs["ActHead"] = _actMs;

        if (activationBf16) runtime.Release(hidden);

        if (activationBf16) runtime.Release(hidden);

        return new LayaHeadOutput
        {
            Logits = logitsHost,
            MarkerProbabilities = probabilities,
            Features = features,
            ActionProbability = LayaHeadScoring.Softmax(actHost)[0],
            ActionClassCount = actClassCount
        };
    }

    void EnsureWorkspace(int rows, int markers)
    {
        if (activationBf16)
        {
            normed = runtime.EnsureUshort(normed, rows * hiddenDim);
            qkv = runtime.EnsureUshort(qkv, rows * 3 * hiddenDim);
            context = runtime.EnsureUshort(context, rows * hiddenDim);
            projected = runtime.EnsureUshort(projected, rows * hiddenDim);
            ffnIn = runtime.EnsureUshort(ffnIn, rows * intermediateDim);
            ffnOut = runtime.EnsureUshort(ffnOut, rows * hiddenDim);
            markerRows = runtime.EnsureUshort(markerRows, markers * hiddenDim);
            markerScored = runtime.EnsureUshort(markerScored, markers * hiddenDim);
            markerScoredOut = runtime.EnsureUshort(markerScoredOut, markers * hiddenDim);
            logits = runtime.EnsureUshort(logits, markers);
            actRow = runtime.EnsureUshort(actRow, hiddenDim + LayaHeadOutput.FeatureCount);
            actHiddenBuf = runtime.EnsureUshort(actHiddenBuf, actHidden);
            actLogits = runtime.EnsureUshort(actLogits, actClassCount);
        }
        else
        {
            normed = runtime.Ensure(normed, rows * hiddenDim);
            qkv = runtime.Ensure(qkv, rows * 3 * hiddenDim);
            context = runtime.Ensure(context, rows * hiddenDim);
            projected = runtime.Ensure(projected, rows * hiddenDim);
            ffnIn = runtime.Ensure(ffnIn, rows * intermediateDim);
            ffnOut = runtime.Ensure(ffnOut, rows * hiddenDim);
            markerRows = runtime.Ensure(markerRows, markers * hiddenDim);
            markerScored = runtime.Ensure(markerScored, markers * hiddenDim);
            markerScoredOut = runtime.Ensure(markerScoredOut, markers * hiddenDim);
            logits = runtime.Ensure(logits, markers);
            actRow = runtime.Ensure(actRow, hiddenDim + LayaHeadOutput.FeatureCount);
            actHiddenBuf = runtime.Ensure(actHiddenBuf, actHidden);
            actLogits = runtime.Ensure(actLogits, actClassCount);
        }
        maskBuf = runtime.Ensure(maskBuf, rows);
        markerIds = runtime.Ensure(markerIds, markers);
    }
    void LaunchAddBias(nint x, nint bias, nint y, int rows, int cols)
    {
        if (activationBf16)
            SilkLauncher.AddBiasBf16Act(runtime, x, bias, y, rows, cols);
        else
            SilkLauncher.AddBias(runtime, x, bias, y, rows, cols);
    }

    void LaunchLayerNorm(nint src, nint gamma, nint beta, nint y, int rows)
    {
        if (activationBf16)
            SilkLauncher.LayerNormBf16Act(runtime, src, gamma, beta, y, rows, hiddenDim, eps);
        else
            SilkLauncher.LayerNorm(runtime, src, gamma, beta, y, rows, hiddenDim, eps);
    }

    void LaunchGemm(string kernel, nint a, nint b, nint c, nint bias, int aRows, int aCols, int bCols, int? blockWidth = null)
    {
        string mapped = ChooseKernel(kernel, activationBf16, narrowBf16, narrowF16);
        SilkLauncher.GemmEpilogue(runtime, mapped, a, b, c, bias, aRows, aCols, bCols, blockWidth);
    }

    internal static string ChooseKernel(string kernel, bool activationBf16 = false, bool narrowBf16 = false, bool narrowF16 = false)
        => ChooseKernelImpl(kernel, activationBf16, narrowBf16, narrowF16);

    internal static string MapKernel(string kernel, bool activationBf16 = false, bool narrowBf16 = false, bool narrowF16 = false)
        => ChooseKernelImpl(kernel, activationBf16, narrowBf16, narrowF16);

    internal static string MapGemmKernel(string kernel, bool activationBf16 = false, bool narrowBf16 = false, bool narrowF16 = false)
        => ChooseKernelImpl(kernel, activationBf16, narrowBf16, narrowF16);

    internal static string MapHeadGemmKernel(string kernel, bool activationBf16 = false, bool narrowBf16 = false, bool narrowF16 = false)
        => ChooseKernelImpl(kernel, activationBf16, narrowBf16, narrowF16);

    internal static string DispatchKernel(string kernel, bool activationBf16 = false, bool narrowBf16 = false, bool narrowF16 = false)
        => ChooseKernelImpl(kernel, activationBf16, narrowBf16, narrowF16);

    static string ChooseKernelImpl(string kernel, bool activationBf16, bool narrowBf16, bool narrowF16)
    {
        if (activationBf16)
            return kernel switch
            {
                SilkKernels.GemmQkv => SilkKernels.GemmQkvBf16Act,
                SilkKernels.GemmGelu => SilkKernels.GemmGeluBf16Act,
                SilkKernels.GemmRelu => SilkKernels.GemmReluBf16Act,
                SilkKernels.GemmBias => SilkKernels.GemmBiasBf16Act,
                _ => kernel
            };
        if (narrowBf16)
            return kernel switch
            {
                SilkKernels.GemmQkv => SilkKernels.GemmQkvBf16,
                SilkKernels.GemmGelu => SilkKernels.GemmGeluBf16,
                SilkKernels.GemmRelu => SilkKernels.GemmReluBf16,
                SilkKernels.GemmBias => SilkKernels.GemmBiasBf16,
                _ => kernel
            };
        if (narrowF16)
            return kernel switch
            {
                SilkKernels.GemmQkv => SilkKernels.GemmQkvF16,
                SilkKernels.GemmGelu => SilkKernels.GemmGeluF16,
                SilkKernels.GemmRelu => SilkKernels.GemmReluF16,
                SilkKernels.GemmBias => SilkKernels.GemmBiasF16,
                _ => kernel
            };
        return kernel;
    }

    void LaunchAttention(nint q, nint k, nint v, nint mask, nint output, int seqLen)
    {
        if (activationBf16)
            SilkLauncher.AttentionBf16Act(runtime, q, k, v, mask, output, 1, seqLen, numHeads, headDim, scale);
        else
            SilkLauncher.Attention(runtime, q, k, v, mask, output, 1, seqLen, numHeads, headDim, SilkKernels.GlobalAttentionBand, scale);
    }

    void LaunchAdd(nint a, nint b, nint y, int n)
    {
        if (activationBf16)
            SilkLauncher.AddBf16Act(runtime, a, b, y, n);
        else
            SilkLauncher.Add(runtime, a, b, y, n);
    }

    void LaunchGather(nint table, nint ids, nint y, int n)
    {
        if (activationBf16)
            SilkLauncher.GatherBf16Act(runtime, table, ids, y, n, hiddenDim);
        else
            SilkLauncher.Gather(runtime, table, ids, y, n, hiddenDim);
    }

    nint UploadTransposedUshort(Dictionary<string, (ushort[] Data, int[] Shape)> tensors, string key)
    {
        var data = WeightLayout.Transposed(tensors, key, "Silk");
        nint buf = runtime.AllocateBytes((ulong)data.Length * sizeof(ushort));
        runtime.Upload(buf, data.AsSpan());
        return buf;
    }

    nint UploadUshortAsFloat(Dictionary<string, (ushort[] Data, int[] Shape)> tensors, string key, Func<ushort, float> widen)
    {
        var t = WeightLayout.Req(tensors, key, "Silk");
        var f = new float[t.Data.Length];
        for (int i = 0; i < f.Length; i++) f[i] = widen(t.Data[i]);
        nint buf = runtime.Allocate(f.Length);
        runtime.Upload(buf, f);
        return buf;
    }

    nint UploadPlain(Dictionary<string, (float[] Data, int[] Shape)> tensors, string key)
        => Upload(WeightLayout.Req(tensors, key, "Silk").Data);

    nint UploadTransposed(Dictionary<string, (float[] Data, int[] Shape)> tensors, string key)
        => Upload(WeightLayout.Transposed(tensors, key, "Silk"));

    nint Upload(float[] data) => SilkLauncher.Upload(runtime, data);

    public void Dispose()
    {
        Release(typeEmb);
        Release(scorerNormW);
        Release(scorerNormB);
        Release(scorerHiddenW);
        Release(scorerHiddenB);
        Release(scorerOutW);
        Release(scorerOutB);
        Release(actInW);
        Release(actInB);
        Release(actOutW);
        Release(actOutB);
        Release(pooledIds);
        if (layers is not null)
        {
            foreach (LayerWeights w in layers)
                w?.Dispose();
        }

        Release(normed);
        Release(qkv);
        Release(context);
        Release(projected);
        Release(ffnIn);
        Release(ffnOut);
        Release(markerRows);
        Release(markerScored);
        Release(markerScoredOut);
        Release(logits);
        Release(actRow);
        Release(actHiddenBuf);
        Release(actLogits);
        Release(maskBuf);
        Release(markerIds);
    }

    void Release(nint buffer) => runtime.Release(buffer);

    void Release(SilkRuntime.SilkBuffer buffer) => runtime.Release(buffer.Handle);

    sealed class LayerWeights : IDisposable
    {
        readonly SilkRuntime runtime;
        public readonly nint Norm1W;
        public readonly nint Norm1B;
        public readonly nint Wqkv;
        public readonly nint Bqkv;
        public readonly nint Wo;
        public readonly nint Bo;
        public readonly nint Norm2W;
        public readonly nint Norm2B;
        public readonly nint W1;
        public readonly nint B1;
        public readonly nint W2;
        public readonly nint B2;

        public LayerWeights(SilkLayaHeadGpuRunner owner, Dictionary<string, (float[] Data, int[] Shape)> tensors, int index)
        {
            runtime = owner.runtime;
            try
            {
                string prefix = $"head.layers.{index}";
                string attn = $"{prefix}.self_attn";
                Norm1W = owner.UploadPlain(tensors, $"{prefix}.norm1.weight");
                Norm1B = owner.UploadPlain(tensors, $"{prefix}.norm1.bias");
                Wqkv = owner.UploadTransposed(tensors, $"{attn}.in_proj_weight");
                Bqkv = owner.UploadPlain(tensors, $"{attn}.in_proj_bias");
                Wo = owner.UploadTransposed(tensors, $"{attn}.out_proj.weight");
                Bo = owner.UploadPlain(tensors, $"{attn}.out_proj.bias");
                Norm2W = owner.UploadPlain(tensors, $"{prefix}.norm2.weight");
                Norm2B = owner.UploadPlain(tensors, $"{prefix}.norm2.bias");
                W1 = owner.UploadTransposed(tensors, $"{prefix}.linear1.weight");
                B1 = owner.UploadPlain(tensors, $"{prefix}.linear1.bias");
                W2 = owner.UploadTransposed(tensors, $"{prefix}.linear2.weight");
                B2 = owner.UploadPlain(tensors, $"{prefix}.linear2.bias");
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        public LayerWeights(
            SilkLayaHeadGpuRunner owner,
            Dictionary<string, (ushort[] Data, int[] Shape)> tensors,
            int index,
            bool isBf16,
            bool isFp16)
        {
            runtime = owner.runtime;
            Func<ushort, float> widen = isBf16
                ? new Func<ushort, float>(u => BitConverter.UInt32BitsToSingle((uint)u << 16))
                : new Func<ushort, float>(u => (float)BitConverter.Int16BitsToHalf((short)u));
            try
            {
                string prefix = $"head.layers.{index}";
                string attn = $"{prefix}.self_attn";
                Norm1W = owner.UploadUshortAsFloat(tensors, $"{prefix}.norm1.weight", widen);
                Norm1B = owner.UploadUshortAsFloat(tensors, $"{prefix}.norm1.bias", widen);
                Wqkv = owner.UploadTransposedUshort(tensors, $"{attn}.in_proj_weight");
                Bqkv = owner.UploadUshortAsFloat(tensors, $"{attn}.in_proj_bias", widen);
                Wo = owner.UploadTransposedUshort(tensors, $"{attn}.out_proj.weight");
                Bo = owner.UploadUshortAsFloat(tensors, $"{attn}.out_proj.bias", widen);
                Norm2W = owner.UploadUshortAsFloat(tensors, $"{prefix}.norm2.weight", widen);
                Norm2B = owner.UploadUshortAsFloat(tensors, $"{prefix}.norm2.bias", widen);
                W1 = owner.UploadTransposedUshort(tensors, $"{prefix}.linear1.weight");
                B1 = owner.UploadUshortAsFloat(tensors, $"{prefix}.linear1.bias", widen);
                W2 = owner.UploadTransposedUshort(tensors, $"{prefix}.linear2.weight");
                B2 = owner.UploadUshortAsFloat(tensors, $"{prefix}.linear2.bias", widen);
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        public void Dispose()
        {
            runtime.Release(Norm1W);
            runtime.Release(Norm1B);
            runtime.Release(Wqkv);
            runtime.Release(Bqkv);
            runtime.Release(Wo);
            runtime.Release(Bo);
            runtime.Release(Norm2W);
            runtime.Release(Norm2B);
            runtime.Release(W1);
            runtime.Release(B1);
            runtime.Release(W2);
            runtime.Release(B2);
        }
    }
}
