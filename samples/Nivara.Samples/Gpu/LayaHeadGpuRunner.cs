using ILGPU;
using ILGPU.Runtime;
using Nivara.AutoDiff.Nn;

namespace Nivara.Samples.Gpu;

/// <summary>
/// Laya's typed decision head on an OpenCL accelerator, mirroring
/// <c>LayaDecisionHead&lt;T&gt;.Forward</c> operation for operation (docs/LAYA.md). Reads the
/// encoder's final hidden state where <see cref="ModernBertGpuRunner"/> left it on device, so
/// nothing but the marker logits and the act logits ever crosses to the host.
/// </summary>
/// <remarks>
/// <para>
/// Per layer, matching <c>LayaHeadLayer&lt;T&gt;.Forward</c>:
/// <code>
/// context = attention(norm1(x), paddingMask) · Wo^T
/// x       = x + context
/// x       = x + linear2(relu(linear1(norm2(x))))
/// </code>
/// This is a pre-norm stack with <em>biased</em> LayerNorms, a ReLU feed-forward, a fused QKV
/// projection and no positional encoding, no RoPE and no sliding-window band — every one of those
/// is a <c>nn.TransformerEncoderLayer</c> default, not a choice, so the whole head reuses the
/// existing kernels and adds none.
/// </para>
/// <para>
/// <b>Two details are easy to get backwards and still produce plausible output.</b>
/// <list type="number">
/// <item>The residual lives in the head layer, not inside the attention: <c>out_proj</c>'s output
/// is added by the caller. Adding it inside the attention would double-count and the result would
/// still be finite.</item>
/// <item>Unlike the encoder, every norm here is <em>biased</em> (nn.TransformerEncoderLayer's
/// <c>norm_eps</c> pair keeps a beta), so each LayerNorm gets a real uploaded beta. Passing the
/// encoder's shared zero beta would be a silent wrong-answer, not a crash.</item>
/// </list>
/// </para>
/// <para>
/// <b>Where "zero new kernels" has to bend.</b> The act head's four features — top-1, the top-1/top-2
/// gap, normalized entropy and the option count — come from a softmax over the marker logits.
/// Computing them on device would need a softmax, a top-k and an entropy kernel, so they are
/// computed on the host by <see cref="LayaHeadScoring"/>, exactly as the CPU head does, and the
/// assembled <c>[hidden + 4]</c> row is uploaded. Only two scalars' worth of round trip separates
/// them from the GEMM they feed, and sharing the helper means both backends produce bit-identical
/// features rather than two implementations that agree today.
/// </para>
/// </remarks>
public sealed class LayaHeadGpuRunner : IDisposable
{
    readonly IlgpuRuntime runtime;
    readonly int hiddenDim;
    readonly int numHeads;
    readonly int headDim;
    readonly int intermediateDim;
    readonly int actClassCount;
    readonly int actHidden;
    readonly float eps;
    readonly float scale;

    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<int>, ArrayView<float>, int> gather;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, float> layerNorm;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>> add;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> addBias;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int> gemmBias;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int> gemmGelu;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int> gemmRelu;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int> gemmQkv;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, float> attention;

    readonly MemoryBuffer1D<float, Stride1D.Dense> typeEmb;
    readonly MemoryBuffer1D<float, Stride1D.Dense> scorerNormW;
    readonly MemoryBuffer1D<float, Stride1D.Dense> scorerNormB;
    readonly MemoryBuffer1D<float, Stride1D.Dense> scorerHiddenW;
    readonly MemoryBuffer1D<float, Stride1D.Dense> scorerHiddenB;
    readonly MemoryBuffer1D<float, Stride1D.Dense> scorerOutW;
    readonly MemoryBuffer1D<float, Stride1D.Dense> scorerOutB;
    readonly MemoryBuffer1D<float, Stride1D.Dense> actInW;
    readonly MemoryBuffer1D<float, Stride1D.Dense> actInB;
    readonly MemoryBuffer1D<float, Stride1D.Dense> actOutW;
    readonly MemoryBuffer1D<float, Stride1D.Dense> actOutB;
    readonly LayerWeights[] layers;

    // The residual stream is the encoder's own buffer, handed in per call rather than owned here.
    // The head rewrites it in place, so the encoder must not forward again before this returns.
    MemoryBuffer1D<float, Stride1D.Dense> normed = null!;
    MemoryBuffer1D<float, Stride1D.Dense> qkv = null!;
    MemoryBuffer1D<float, Stride1D.Dense> context = null!;
    MemoryBuffer1D<float, Stride1D.Dense> projected = null!;
    MemoryBuffer1D<float, Stride1D.Dense> ffnIn = null!;
    MemoryBuffer1D<float, Stride1D.Dense> ffnOut = null!;
    MemoryBuffer1D<float, Stride1D.Dense> markerRows = null!;
    MemoryBuffer1D<float, Stride1D.Dense> markerScored = null!;
    MemoryBuffer1D<float, Stride1D.Dense> logits = null!;
    MemoryBuffer1D<float, Stride1D.Dense> actRow = null!;
    MemoryBuffer1D<float, Stride1D.Dense> actHiddenBuf = null!;
    MemoryBuffer1D<float, Stride1D.Dense> actLogits = null!;
    MemoryBuffer1D<float, Stride1D.Dense> maskBuf = null!;
    MemoryBuffer1D<int, Stride1D.Dense> markerIds = null!;
    readonly MemoryBuffer1D<int, Stride1D.Dense> pooledIds;

    int[] markerIdHost = [];

    /// <summary>
    /// Uploads the decision head and its kernels to <paramref name="runtime"/>. The encoder is not
    /// touched; pass the same dictionary to <see cref="ModernBertGpuRunner"/> with prefix
    /// <c>"encoder"</c>.
    /// </summary>
    /// <param name="headLayerCount">Number of pre-norm layers; 0, 1 or 2, from the agent config.</param>
    /// <param name="actClassCount">Width of the act head's output; 2 on the shipped checkpoint.</param>
    public LayaHeadGpuRunner(
        IlgpuRuntime runtime,
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

        // max(1, d // 64), the reference's own expression, and the head's eps default rather than
        // the encoder's config.NormEps — reusing the encoder's eps would be a plausible-looking
        // value that is not the checkpoint's.
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

        var acc = runtime.Accelerator;
        GpuBuffers.ValidateAttentionLocalMemory(acc, headDim);

        gather = acc.LoadKernel<ArrayView<float>, ArrayView<int>, ArrayView<float>, int>(ElementwiseKernels.Gather);
        layerNorm = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, float>(ElementwiseKernels.LayerNorm1D);
        add = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>>(ElementwiseKernels.Add);
        addBias = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(ElementwiseKernels.AddBias);
        gemmBias = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(GemmKernels.TiledGemmKernelRow4Bias);
        gemmGelu = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(GemmKernels.TiledGemmKernelRow4Gelu);
        gemmRelu = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(GemmKernels.TiledGemmKernelRow4Relu);
        gemmQkv = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int>(GemmKernels.TiledGemmKernelRow4Qkv);
        attention = acc.LoadKernel<
            ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, int, int, int, int, int, float>(AttentionKernels.BatchedAttention);

        typeEmb = GpuBuffers.UploadPlain(acc, tensors, "type_emb.weight");

        scorerNormW = GpuBuffers.UploadPlain(acc, tensors, "scorer.0.weight");
        scorerNormB = GpuBuffers.UploadPlain(acc, tensors, "scorer.0.bias");
        scorerHiddenW = GpuBuffers.UploadTransposed(acc, tensors, "scorer.1.weight");
        scorerHiddenB = GpuBuffers.UploadPlain(acc, tensors, "scorer.1.bias");
        scorerOutW = GpuBuffers.UploadTransposed(acc, tensors, "scorer.3.weight");
        scorerOutB = GpuBuffers.UploadPlain(acc, tensors, "scorer.3.bias");

        actInW = GpuBuffers.UploadTransposed(acc, tensors, "act_head.0.weight");
        actInB = GpuBuffers.UploadPlain(acc, tensors, "act_head.0.bias");
        actOutW = GpuBuffers.UploadTransposed(acc, tensors, "act_head.2.weight");
        actOutB = GpuBuffers.UploadPlain(acc, tensors, "act_head.2.bias");

        layers = new LayerWeights[headLayerCount];
        for (int i = 0; i < layers.Length; i++)
            layers[i] = new LayerWeights(acc, tensors, i);

        // pooled = h[0] — the post-head row 0, i.e. the [CLS] position. Always position 0, so the
        // index is uploaded once here rather than per call.
        pooledIds = acc.Allocate1D<int>(1);
        pooledIds.CopyFromCPU(new[] { 0 });
    }

    /// <summary>Hidden width this head was uploaded for.</summary>
    public int HiddenSize => hiddenDim;

    /// <summary>
    /// Runs the head over the encoder output left in <paramref name="hidden"/>, and returns the
    /// same <see cref="LayaHeadOutput"/> the CPU head returns, so the two are directly comparable
    /// and the gate compares them field for field.
    /// </summary>
    /// <param name="hidden">
    /// The encoder's last hidden state, <c>[rows, hiddenDim]</c> row-major, still on device. Taken
    /// from <see cref="ModernBertGpuRunner.HiddenOnDevice"/>; the buffer is reused as the head's
    /// residual stream, so the encoder must not forward again before this returns.
    /// </param>
    /// <param name="rows">Row count of <paramref name="hidden"/>; the padded sequence length.</param>
    /// <param name="type">The question type, selecting the type-embedding row.</param>
    /// <param name="markerPositions">Indices of the <c>[MASK]</c> tokens, one per option.</param>
    /// <param name="validLength">The number of leading positions that are real tokens.</param>
    public LayaHeadOutput Forward(
        MemoryBuffer1D<float, Stride1D.Dense> hidden,
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
        if (hidden.Length < rows * hiddenDim)
            throw new ArgumentException(
                $"The encoder buffer holds {hidden.Length} floats but {rows} rows of {hiddenDim} were " +
                "expected. Pass ModernBertGpuRunner.HiddenOnDevice with the row count it reports.",
                nameof(hidden));

        var acc = runtime.Accelerator;
        int k = markerPositions.Count;
        EnsureWorkspace(acc, rows, k);

        // Only the first `rows * hiddenDim` elements are this sequence's: the encoder's workspace
        // is grown to the longest forward it has seen and is never shrunk.
        var xView = hidden.View.SubView(0, rows * hiddenDim);
        var normedView = normed.View;
        var qkvView = qkv.View;
        var contextView = context.View;
        var projectedView = projected.View;
        var ffnInView = ffnIn.View;
        var ffnOutView = ffnOut.View;
        var maskView = maskBuf.View;

        // The encoder left the final-normed hidden state in `x`; AddBias adds the type row over it
        // in place. The residual stream is never aliased by a norm output or a projection result,
        // so the two in-place residual adds below always read a value another kernel produced.
        var typeRow = typeEmb.View.SubView((int)type * hiddenDim, hiddenDim);
        addBias(runtime.Stream, GpuBuffers.Cfg1D(rows * hiddenDim), xView, typeRow, xView, rows, hiddenDim);

        if (layers.Length > 0)
        {
            maskBuf.View.SubView(0, rows).CopyFromCPU(runtime.Stream, LayaHeadScoring.PrefixMask(rows, validLength));

            for (int i = 0; i < layers.Length; i++)
            {
                var w = layers[i];

                layerNorm(runtime.Stream, GpuBuffers.Cfg1D(rows), xView, w.Norm1W.View, w.Norm1B.View, normedView, rows, hiddenDim, eps);

                // nn.MultiheadAttention ships one [3d, d] weight and one [3d] bias, in q, k, v
                // order. The fused GEMM writes q, k and v as three contiguous [rows, d] blocks, so
                // unlike the encoder's row-interleaved Wqkv they need no split.
                gemmQkv(runtime.Stream, GpuBuffers.GemmCfg(rows, 3 * hiddenDim),
                    normedView, w.Wqkv.View, qkvView, w.Bqkv.View, rows, hiddenDim, 3 * hiddenDim, hiddenDim);

                var qView = qkvView.SubView(0, rows * hiddenDim);
                var kView = qkvView.SubView(rows * hiddenDim, rows * hiddenDim);
                var vView = qkvView.SubView(2 * rows * hiddenDim, rows * hiddenDim);

                attention(runtime.Stream, GpuBuffers.Cfg1D(rows * numHeads, GpuBuffers.AttentionGroupSize),
                    qView, kView, vView, maskView, contextView, 1, rows, numHeads, headDim,
                    GpuBuffers.GlobalAttentionBand, scale);

                gemmBias(runtime.Stream, GpuBuffers.GemmCfg(rows, hiddenDim),
                    contextView, w.Wo.View, projectedView, w.Bo.View, rows, hiddenDim, hiddenDim);
                add(runtime.Stream, GpuBuffers.Cfg1D(rows * hiddenDim), xView, projectedView, xView);

                layerNorm(runtime.Stream, GpuBuffers.Cfg1D(rows), xView, w.Norm2W.View, w.Norm2B.View, normedView, rows, hiddenDim, eps);
                gemmRelu(runtime.Stream, GpuBuffers.GemmCfg(rows, intermediateDim),
                    normedView, w.W1.View, ffnInView, w.B1.View, rows, hiddenDim, intermediateDim);
                gemmBias(runtime.Stream, GpuBuffers.GemmCfg(rows, hiddenDim),
                    ffnInView, w.W2.View, ffnOutView, w.B2.View, rows, intermediateDim, hiddenDim);
                add(runtime.Stream, GpuBuffers.Cfg1D(rows * hiddenDim), xView, ffnOutView, xView);
            }
        }

        // scorer = Sequential(LayerNorm, Linear, GELU, Linear) over the gathered marker rows.
        if (markerIdHost.Length < k)
            markerIdHost = new int[k];
        for (int i = 0; i < k; i++)
            markerIdHost[i] = markerPositions[i];
        markerIds.View.SubView(0, k).CopyFromCPU(runtime.Stream, markerIdHost.AsSpan(0, k));
        gather(runtime.Stream, GpuBuffers.Cfg1D(k * hiddenDim), xView, markerIds.View, markerRows.View, hiddenDim);
        layerNorm(runtime.Stream, GpuBuffers.Cfg1D(k), markerRows.View, scorerNormW.View, scorerNormB.View, markerScored.View, k, hiddenDim, eps);
        gemmGelu(runtime.Stream, GpuBuffers.GemmCfg(k, hiddenDim),
            markerScored.View, scorerHiddenW.View, markerScored.View, scorerHiddenB.View, k, hiddenDim, hiddenDim);
        gemmBias(runtime.Stream, GpuBuffers.GemmCfg(k, 1),
            markerScored.View, scorerOutW.View, logits.View, scorerOutB.View, k, hiddenDim, 1);

        runtime.Synchronize();
        var logitsHost = GpuBuffers.Readback(logits, k);

        // p = softmax(logits) and the four features are host arithmetic in the reference too, and
        // share the CPU head's implementation so they cannot drift between the two backends.
        var probabilities = LayaHeadScoring.Softmax(logitsHost);
        var features = LayaHeadScoring.BuildFeatures(probabilities);

        // act_head = Sequential(Linear, GELU, Linear) over cat([h[0], feats]). The pooled row is
        // gathered on device, but the features are host values, so the assembled row is uploaded.
        gather(runtime.Stream, GpuBuffers.Cfg1D(hiddenDim), xView, pooledIds.View, actRow.View, hiddenDim);
        runtime.Synchronize();
        var pooledHost = GpuBuffers.Readback(actRow, hiddenDim);
        for (int i = 0; i < features.Length; i++)
            pooledHost[hiddenDim + i] = (float)features[i];
        actRow.View.CopyFromCPU(runtime.Stream, pooledHost);

        gemmGelu(runtime.Stream, GpuBuffers.GemmCfg(1, actHidden),
            actRow.View, actInW.View, actHiddenBuf.View, actInB.View, 1, hiddenDim + LayaHeadOutput.FeatureCount, actHidden);
        gemmBias(runtime.Stream, GpuBuffers.GemmCfg(1, actClassCount),
            actHiddenBuf.View, actOutW.View, actLogits.View, actOutB.View, 1, actHidden, actClassCount);

        runtime.Synchronize();
        var actHost = GpuBuffers.Readback(actLogits, actClassCount);

        return new LayaHeadOutput
        {
            Logits = logitsHost,
            MarkerProbabilities = probabilities,
            Features = features,
            ActionProbability = LayaHeadScoring.Softmax(actHost)[0],
            ActionClassCount = actClassCount
        };
    }

    void EnsureWorkspace(Accelerator acc, int rows, int markers)
    {
        GpuBuffers.Ensure(ref normed, rows * hiddenDim, acc);
        GpuBuffers.Ensure(ref qkv, rows * 3 * hiddenDim, acc);
        GpuBuffers.Ensure(ref context, rows * hiddenDim, acc);
        GpuBuffers.Ensure(ref projected, rows * hiddenDim, acc);
        GpuBuffers.Ensure(ref ffnIn, rows * intermediateDim, acc);
        GpuBuffers.Ensure(ref ffnOut, rows * hiddenDim, acc);
        GpuBuffers.Ensure(ref markerRows, markers * hiddenDim, acc);
        GpuBuffers.Ensure(ref markerScored, markers * hiddenDim, acc);
        GpuBuffers.Ensure(ref logits, markers, acc);
        GpuBuffers.Ensure(ref actRow, hiddenDim + LayaHeadOutput.FeatureCount, acc);
        GpuBuffers.Ensure(ref actHiddenBuf, actHidden, acc);
        GpuBuffers.Ensure(ref actLogits, actClassCount, acc);
        GpuBuffers.Ensure(ref maskBuf, rows, acc);
        GpuBuffers.Ensure(ref markerIds, markers, acc);
    }

    public void Dispose()
    {
        typeEmb.Dispose();
        scorerNormW.Dispose();
        scorerNormB.Dispose();
        scorerHiddenW.Dispose();
        scorerHiddenB.Dispose();
        scorerOutW.Dispose();
        scorerOutB.Dispose();
        actInW.Dispose();
        actInB.Dispose();
        actOutW.Dispose();
        actOutB.Dispose();
        foreach (var w in layers)
            w.Dispose();

        normed?.Dispose();
        qkv?.Dispose();
        context?.Dispose();
        projected?.Dispose();
        ffnIn?.Dispose();
        ffnOut?.Dispose();
        markerRows?.Dispose();
        markerScored?.Dispose();
        logits?.Dispose();
        actRow?.Dispose();
        actHiddenBuf?.Dispose();
        actLogits?.Dispose();
        maskBuf?.Dispose();
        markerIds?.Dispose();
        pooledIds?.Dispose();
    }

    /// <summary>One head layer's uploaded weights, including the biases the encoder's norms lack.</summary>
    sealed class LayerWeights : IDisposable
    {
        public readonly MemoryBuffer1D<float, Stride1D.Dense> Norm1W;
        public readonly MemoryBuffer1D<float, Stride1D.Dense> Norm1B;
        public readonly MemoryBuffer1D<float, Stride1D.Dense> Wqkv;
        public readonly MemoryBuffer1D<float, Stride1D.Dense> Bqkv;
        public readonly MemoryBuffer1D<float, Stride1D.Dense> Wo;
        public readonly MemoryBuffer1D<float, Stride1D.Dense> Bo;
        public readonly MemoryBuffer1D<float, Stride1D.Dense> Norm2W;
        public readonly MemoryBuffer1D<float, Stride1D.Dense> Norm2B;
        public readonly MemoryBuffer1D<float, Stride1D.Dense> W1;
        public readonly MemoryBuffer1D<float, Stride1D.Dense> B1;
        public readonly MemoryBuffer1D<float, Stride1D.Dense> W2;
        public readonly MemoryBuffer1D<float, Stride1D.Dense> B2;

        public LayerWeights(Accelerator acc, Dictionary<string, (float[] Data, int[] Shape)> tensors, int index)
        {
            string prefix = $"head.layers.{index}";
            string attn = $"{prefix}.self_attn";
            Norm1W = GpuBuffers.UploadPlain(acc, tensors, $"{prefix}.norm1.weight");
            Norm1B = GpuBuffers.UploadPlain(acc, tensors, $"{prefix}.norm1.bias");
            Wqkv = GpuBuffers.UploadTransposed(acc, tensors, $"{attn}.in_proj_weight");
            Bqkv = GpuBuffers.UploadPlain(acc, tensors, $"{attn}.in_proj_bias");
            Wo = GpuBuffers.UploadTransposed(acc, tensors, $"{attn}.out_proj.weight");
            Bo = GpuBuffers.UploadPlain(acc, tensors, $"{attn}.out_proj.bias");
            Norm2W = GpuBuffers.UploadPlain(acc, tensors, $"{prefix}.norm2.weight");
            Norm2B = GpuBuffers.UploadPlain(acc, tensors, $"{prefix}.norm2.bias");
            W1 = GpuBuffers.UploadTransposed(acc, tensors, $"{prefix}.linear1.weight");
            B1 = GpuBuffers.UploadPlain(acc, tensors, $"{prefix}.linear1.bias");
            W2 = GpuBuffers.UploadTransposed(acc, tensors, $"{prefix}.linear2.weight");
            B2 = GpuBuffers.UploadPlain(acc, tensors, $"{prefix}.linear2.bias");
        }

        public void Dispose()
        {
            Norm1W.Dispose();
            Norm1B.Dispose();
            Wqkv.Dispose();
            Bqkv.Dispose();
            Wo.Dispose();
            Bo.Dispose();
            Norm2W.Dispose();
            Norm2B.Dispose();
            W1.Dispose();
            B1.Dispose();
            W2.Dispose();
            B2.Dispose();
        }
    }
}
