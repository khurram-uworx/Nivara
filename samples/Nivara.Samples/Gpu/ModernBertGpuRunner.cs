using ILGPU;
using ILGPU.Runtime;
using Nivara.AutoDiff.Nn;

namespace Nivara.Samples.Gpu;

/// <summary>
/// ModernBERT encoder forward on an OpenCL accelerator, mirroring
/// <see cref="ModernBertEncoder{T}.Forward(int[], int)"/> operation for operation
/// (docs/BERT-GPU.md §3). This is the GPU half of issue #449; the CPU encoder it is gated
/// against is the Phase 1 implementation, so the fixture cost is one, not two.
///
/// Per layer, matching <c>ModernBertLayer&lt;T&gt;.Forward</c>:
///   normed   = attnNorm(input)          (an identity on layer 0, which has no attn_norm weights)
///   q, k, v  = [q | k | v] = normed · Wqkv^T      — one fused, bias-free GEMM
///   q, k     = rope(q), rope(k)          — in place; v is NOT rotated
///   context  = attention(q, k, v)        — bidirectional, banded for sliding layers
///   hidden   = input + context · Wo^T
///   hidden   = hidden + (geglu(hidden · Wi^T)) · WoMlp^T
///
/// Two details are easy to get backwards and still produce plausible output, so they are
/// spelled out rather than left implicit:
/// <list type="bullet">
/// <item>GeGLU activates the <em>first</em> half of the fused <c>Wi</c> projection
/// (<c>gelu(Wi[0:I]) * Wi[I:2I]</c>), per HF's <c>act(input) * gate</c> with
/// <c>input, gate = Wi(x).chunk(2, -1)</c>.</item>
/// <item>RoPE tables come from the CPU <see cref="RotaryEmbedding{T}"/> rather than a device
/// math library, so cos/sin are bit-identical instead of one libm-vs-<c>XMath.Cos</c> ulp
/// apart — which would otherwise become the largest single error term in the gate.</item>
/// </list>
///
/// The bias-free norms take a shared zero-filled Beta. The CPU encoder now constructs
/// <c>LayerNorm&lt;T&gt;</c> with <c>bias: false</c> and has no beta parameter; an ILGPU view
/// cannot be null, so this runner still passes a zero buffer instead of skipping the term.
/// </summary>
public sealed class ModernBertGpuRunner : IDisposable
{
    readonly IlgpuRuntime runtime;
    readonly int hiddenDim;
    readonly int intermediateDim;
    readonly int numHeads;
    readonly int headDim;
    readonly int halfDim;
    readonly float eps;
    readonly float scale;
    readonly int seqCap;

    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<int>, ArrayView<float>, int> gather;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, float> layerNorm;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>> add;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, int, int, int, int> splitColumns;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int> gemm;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<int>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int> rotary;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, int, int> geGlu;
    readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, float> attention;

    readonly MemoryBuffer1D<float, Stride1D.Dense> tokEmb;
    readonly MemoryBuffer1D<float, Stride1D.Dense> embedNormW;
    readonly MemoryBuffer1D<float, Stride1D.Dense> finalNormW;
    readonly MemoryBuffer1D<float, Stride1D.Dense> zeroBeta;
    readonly LayerWeights[] layers;

    readonly RotaryEmbedding<float> ropeFull;
    readonly RotaryEmbedding<float> ropeSliding;
    MemoryBuffer1D<float, Stride1D.Dense> cosFull = null!;
    MemoryBuffer1D<float, Stride1D.Dense> sinFull = null!;
    MemoryBuffer1D<float, Stride1D.Dense> cosSliding = null!;
    MemoryBuffer1D<float, Stride1D.Dense> sinSliding = null!;
    int ropeTableRows;

    int[]? posIdHost;
    float[]? maskHost;
    int hiddenRows;

    MemoryBuffer1D<float, Stride1D.Dense> x = null!;
    MemoryBuffer1D<float, Stride1D.Dense> normed = null!;
    MemoryBuffer1D<float, Stride1D.Dense> qkv = null!;
    MemoryBuffer1D<float, Stride1D.Dense> qkvSplit = null!;
    MemoryBuffer1D<float, Stride1D.Dense> context = null!;
    MemoryBuffer1D<float, Stride1D.Dense> projected = null!;
    MemoryBuffer1D<float, Stride1D.Dense> mlpIn = null!;
    MemoryBuffer1D<float, Stride1D.Dense> gated = null!;
    MemoryBuffer1D<float, Stride1D.Dense> mlpOut = null!;
    MemoryBuffer1D<float, Stride1D.Dense> maskBuf = null!;
    MemoryBuffer1D<int, Stride1D.Dense> idsBuf = null!;
    MemoryBuffer1D<int, Stride1D.Dense> posIdsBuf = null!;

    /// <summary>
    /// Uploads a ModernBERT encoder and its kernels to <paramref name="runtime"/>.
    /// <paramref name="prefix"/> selects the checkpoint layout, matching
    /// <c>ModernBertEncoder&lt;T&gt;.LoadWeights</c>: <c>"model"</c> for the stock HuggingFace
    /// backbone, <c>"encoder"</c> for Laya, which nests the same architecture under that name.
    /// </summary>
    public ModernBertGpuRunner(
        IlgpuRuntime runtime,
        ModernBertConfig config,
        Dictionary<string, (float[] Data, int[] Shape)> tensors,
        string prefix = "model")
    {
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(tensors);

        this.runtime = runtime;
        hiddenDim = config.HiddenSize;
        intermediateDim = config.IntermediateSize;
        numHeads = config.NumAttentionHeads;
        headDim = config.HeadDim;
        halfDim = headDim / 2;
        eps = config.NormEps;
        scale = 1f / MathF.Sqrt(headDim);
        seqCap = config.MaxPositionEmbeddings;

        var acc = runtime.Accelerator;
        GpuBuffers.ValidateAttentionLocalMemory(acc, headDim);

        gather = acc.LoadKernel<ArrayView<float>, ArrayView<int>, ArrayView<float>, int>(ElementwiseKernels.Gather);
        layerNorm = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, float>(ElementwiseKernels.LayerNorm1D);
        add = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>>(ElementwiseKernels.Add);
        splitColumns = acc.LoadKernel<ArrayView<float>, ArrayView<float>, int, int, int, int>(ElementwiseKernels.SplitColumns);
        gemm = acc.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(GemmKernels.TiledGemmKernelRow4);
        rotary = acc.LoadKernel<ArrayView<float>, ArrayView<int>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(ElementwiseKernels.Rotary);
        geGlu = acc.LoadKernel<ArrayView<float>, ArrayView<float>, int, int>(ElementwiseKernels.GeGlu);
        attention = acc.LoadKernel<
            ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, int, int, int, int, int, float>(AttentionKernels.BatchedAttention);

        tokEmb = GpuBuffers.UploadPlain(acc, tensors, $"{prefix}.embeddings.tok_embeddings.weight");
        embedNormW = GpuBuffers.UploadPlain(acc, tensors, $"{prefix}.embeddings.norm.weight");
        finalNormW = GpuBuffers.UploadPlain(acc, tensors, $"{prefix}.final_norm.weight");

        // Every norm in this architecture is bias-free (HF's norm_bias: false). The CPU encoder
        // constructs LayerNorm with bias: false and has no beta parameter. ILGPU views cannot
        // be null, so the same zero value is passed as a shared read-only buffer rather than skipped.
        zeroBeta = GpuBuffers.Alloc(acc, hiddenDim);
        zeroBeta.CopyFromCPU(new float[hiddenDim]);

        layers = new LayerWeights[config.NumHiddenLayers];
        for (int i = 0; i < layers.Length; i++)
            layers[i] = new LayerWeights(acc, tensors, config, prefix, i);

        ropeFull = new RotaryEmbedding<float>(headDim, config.MaxPositionEmbeddings, config.RopeThetaFull);
        ropeSliding = new RotaryEmbedding<float>(headDim, config.MaxPositionEmbeddings, config.RopeThetaSliding);
    }

    /// <summary>
    /// The final-normed hidden state of the most recent forward, still on device as
    /// <c>[HiddenRows, hidden]</c> row-major. Lets a caller hang its own head off the trunk
    /// without a host round trip — see <see cref="LayaHeadGpuRunner"/>, which is gated against
    /// the CPU head this runner is itself gated against.
    /// </summary>
    /// <remarks>
    /// The buffer is the runner's, not a copy: a head is expected to run in place, so
    /// <c>Forward</c> must not be called again until the head has returned. It is also grown, never
    /// shrunk, so a later shorter sequence leaves a longer buffer whose first
    /// <c>HiddenRows * hidden</c> elements are the current one.
    /// </remarks>
    public MemoryBuffer1D<float, Stride1D.Dense> HiddenOnDevice => projected;

    /// <summary>Row count of <see cref="HiddenOnDevice"/>, set by the last forward.</summary>
    public int HiddenRows => hiddenRows;

    /// <summary>
    /// Encodes <paramref name="tokenIds"/> and returns the last hidden state as
    /// <c>[seqLen, hidden]</c> row-major. <paramref name="validLength"/> is the number of leading
    /// positions that are real tokens; the rest are padding, masked out of every attention. Same
    /// contract as <c>ModernBertEncoder&lt;T&gt;.Forward(int[], int)</c>, so the two are directly
    /// comparable and the gate compares them position for position.
    /// </summary>
    public float[] Forward(int[] tokenIds, int validLength)
    {
        ForwardOnDevice(tokenIds, validLength);
        runtime.Synchronize();
        return GpuBuffers.Readback(projected, hiddenRows * hiddenDim);
    }

    /// <summary>
    /// As <see cref="Forward(int[], int)"/>, but leaves the final hidden state on the device for
    /// <see cref="HiddenOnDevice"/> instead of reading it back. Both entry points launch on
    /// <see cref="IlgpuRuntime.Stream"/>, so a head that also uses that stream observes the
    /// encoder's writes in launch order without an explicit synchronise of its own.
    /// </summary>
    public void ForwardOnDevice(int[] tokenIds, int validLength)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);

        int seqLen = tokenIds.Length;
        if (seqLen <= 0)
            throw new ArgumentOutOfRangeException(nameof(tokenIds), $"tokenIds must be non-empty, got {seqLen}.");
        if (validLength <= 0 || validLength > seqLen)
            throw new ArgumentOutOfRangeException(nameof(validLength), $"validLength must be within [1, {seqLen}], got {validLength}.");
        if (seqLen > seqCap)
            throw new ArgumentOutOfRangeException(nameof(tokenIds),
                $"seqLen {seqLen} exceeds max_position_embeddings {seqCap}.");

        var acc = runtime.Accelerator;
        EnsureWorkspace(acc, seqLen);
        EnsureRopeTables(acc, seqLen);

        // Device views cannot be written from the host, so the per-row position ids and the
        // padding mask go up as ordinary arrays. Positions are the row index here because this
        // API encodes one sequence from position 0 with no offset.
        if (posIdHost is null || posIdHost.Length < seqLen)
            posIdHost = new int[seqLen];
        for (int p = 0; p < seqLen; p++)
            posIdHost[p] = p;

        if (maskHost is null || maskHost.Length < seqLen)
            maskHost = new float[seqLen];
        for (int p = 0; p < seqLen; p++)
            maskHost[p] = p < validLength ? 1f : 0f;

        idsBuf.View.SubView(0, seqLen).CopyFromCPU(runtime.Stream, tokenIds);
        posIdsBuf.View.SubView(0, seqLen).CopyFromCPU(runtime.Stream, posIdHost.AsSpan(0, seqLen));
        maskBuf.View.SubView(0, seqLen).CopyFromCPU(runtime.Stream, maskHost.AsSpan(0, seqLen));

        int rows = seqLen;
        var xView = x.View;
        var normedView = normed.View;
        var qkvView = qkv.View;
        var qkvSplitView = qkvSplit.View;
        var contextView = context.View;
        var projectedView = projected.View;
        var mlpInView = mlpIn.View;
        var gatedView = gated.View;
        var mlpOutView = mlpOut.View;
        var maskView = maskBuf.View;
        var posView = posIdsBuf.View;
        var idsView = idsBuf.View;

        gather(runtime.Stream, GpuBuffers.Cfg1D(rows * hiddenDim), tokEmb.View, idsView, xView, hiddenDim);

        // In place: the embedding norm's output *is* the residual stream that layer 0 reads, and
        // LayerNorm1D owns a whole row across its two reduction passes before it writes any of it,
        // so a work item never reads an element another work item has already overwritten.
        layerNorm(runtime.Stream, GpuBuffers.Cfg1D(rows), xView, embedNormW.View, zeroBeta.View, xView, rows, hiddenDim, eps);

        for (int i = 0; i < layers.Length; i++)
        {
            var w = layers[i];

            // Layer 0 has no attn_norm in the checkpoint and the CPU makes it an identity, so
            // the residual stream feeds the fused QKV projection directly. Every other layer
            // pre-norms into `normed`; that buffer is never a candidate for the aliasing
            // write-back in `add`, which always reads and writes xView.
            ArrayView<float> qkvIn = w.AttnNormW == null
                ? xView
                : LayerNormInto(runtime.Stream, GpuBuffers.Cfg1D(rows), xView, w.AttnNormW.View, normedView, rows);

            gemm(runtime.Stream, GpuBuffers.GemmCfg(rows, 3 * hiddenDim),
                qkvIn, w.Wqkv.View, qkvView, rows, hiddenDim, 3 * hiddenDim);

            var qView = qkvSplitView.SubView(0, rows * hiddenDim);
            var kView = qkvSplitView.SubView(rows * hiddenDim, rows * hiddenDim);
            var vView = qkvSplitView.SubView(2 * rows * hiddenDim, rows * hiddenDim);

            // The fused projection is row-major, so [q | k | v] are interleaved within each row
            // rather than concatenated; split per row before anything reads them as dense
            // [rows, heads*headDim] blocks.
            for (int part = 0; part < 3; part++)
                splitColumns(runtime.Stream, GpuBuffers.Cfg1D(rows * hiddenDim),
                    qkvView, qkvSplitView.SubView(part * rows * hiddenDim, rows * hiddenDim),
                    rows, hiddenDim, 3, part);

            // RoPE in place: a work item reads the pair it owns and writes exactly that pair
            // back, so no other work item's inputs are disturbed and the round trip is safe.
            // Only q and k are rotated; v is not.
            var cos = w.Full ? cosFull.View : cosSliding.View;
            var sin = w.Full ? sinFull.View : sinSliding.View;
            rotary(runtime.Stream, GpuBuffers.Cfg1D(rows * numHeads * halfDim),
                qView, posView, cos, sin, qView, rows, numHeads, headDim);
            rotary(runtime.Stream, GpuBuffers.Cfg1D(rows * numHeads * halfDim),
                kView, posView, cos, sin, kView, rows, numHeads, headDim);

            attention(runtime.Stream, GpuBuffers.Cfg1D(rows * numHeads, GpuBuffers.AttentionGroupSize),
                qView, kView, vView, maskView, contextView, 1, seqLen, numHeads, headDim, w.Band, scale);

            gemm(runtime.Stream, GpuBuffers.GemmCfg(rows, hiddenDim),
                contextView, w.Wo.View, projectedView, rows, hiddenDim, hiddenDim);
            add(runtime.Stream, GpuBuffers.Cfg1D(rows * hiddenDim), xView, projectedView, xView);

            LayerNormInto(runtime.Stream, GpuBuffers.Cfg1D(rows), xView, w.MlpNormW.View, normedView, rows);
            gemm(runtime.Stream, GpuBuffers.GemmCfg(rows, 2 * intermediateDim),
                normedView, w.Wi.View, mlpInView, rows, hiddenDim, 2 * intermediateDim);
            geGlu(runtime.Stream, GpuBuffers.Cfg1D(rows * intermediateDim), mlpInView, gatedView, rows, intermediateDim);
            gemm(runtime.Stream, GpuBuffers.GemmCfg(rows, hiddenDim),
                gatedView, w.WoMlp.View, mlpOutView, rows, intermediateDim, hiddenDim);
            add(runtime.Stream, GpuBuffers.Cfg1D(rows * hiddenDim), xView, mlpOutView, xView);
        }

        layerNorm(runtime.Stream, GpuBuffers.Cfg1D(rows), xView, finalNormW.View, zeroBeta.View, projectedView, rows, hiddenDim, eps);
        hiddenRows = rows;
    }

    /// <summary>
    /// Pre-norms the residual stream into the layer's scratch buffer and returns the view to feed
    /// the following projection. This architecture's norms are all bias-free, so the shared
    /// zero Beta stands in; the method exists so the "which buffer is the norm's output"
    /// question is answered in one place, since the QKV GEMM consumes it directly.
    /// </summary>
    ArrayView<float> LayerNormInto(AcceleratorStream stream, KernelConfig cfg, ArrayView<float> src, ArrayView<float> gamma, ArrayView<float> dst, int rows)
    {
        layerNorm(stream, cfg, src, gamma, zeroBeta.View, dst, rows, hiddenDim, eps);
        return dst;
    }

    void EnsureWorkspace(Accelerator acc, int seqLen)
    {
        int rows = seqLen;
        GpuBuffers.Ensure(ref x, rows * hiddenDim, acc);
        GpuBuffers.Ensure(ref normed, rows * hiddenDim, acc);
        GpuBuffers.Ensure(ref qkv, rows * 3 * hiddenDim, acc);
        GpuBuffers.Ensure(ref qkvSplit, rows * 3 * hiddenDim, acc);
        GpuBuffers.Ensure(ref context, rows * hiddenDim, acc);
        GpuBuffers.Ensure(ref projected, rows * hiddenDim, acc);
        GpuBuffers.Ensure(ref mlpIn, rows * 2 * intermediateDim, acc);
        GpuBuffers.Ensure(ref gated, rows * intermediateDim, acc);
        GpuBuffers.Ensure(ref mlpOut, rows * hiddenDim, acc);
        GpuBuffers.Ensure(ref maskBuf, rows, acc);
        GpuBuffers.Ensure(ref idsBuf, rows, acc);
        GpuBuffers.Ensure(ref posIdsBuf, rows, acc);
    }

    /// <summary>
    /// Uploads the cos/sin tables for the two rope thetas, grown on a longer sequence. The
    /// tables are taken from the CPU <see cref="RotaryEmbedding{T}"/> and copied out, because the
    /// spans it returns alias its own cache.
    /// </summary>
    void EnsureRopeTables(Accelerator acc, int seqLen)
    {
        if (ropeTableRows >= seqLen)
            return;

        GpuBuffers.Ensure(ref cosFull, seqLen * halfDim, acc);
        GpuBuffers.Ensure(ref sinFull, seqLen * halfDim, acc);
        GpuBuffers.Ensure(ref cosSliding, seqLen * halfDim, acc);
        GpuBuffers.Ensure(ref sinSliding, seqLen * halfDim, acc);

        ropeFull.GetPositionTables(0, seqLen, out var fullCos, out var fullSin);
        ropeSliding.GetPositionTables(0, seqLen, out var slideCos, out var slideSin);
        cosFull.CopyFromCPU(fullCos.ToArray());
        sinFull.CopyFromCPU(fullSin.ToArray());
        cosSliding.CopyFromCPU(slideCos.ToArray());
        sinSliding.CopyFromCPU(slideSin.ToArray());
        ropeTableRows = seqLen;
    }

    public void Dispose()
    {
        tokEmb.Dispose();
        embedNormW.Dispose();
        finalNormW.Dispose();
        zeroBeta.Dispose();
        foreach (var w in layers)
            w.Dispose();

        cosFull?.Dispose();
        sinFull?.Dispose();
        cosSliding?.Dispose();
        sinSliding?.Dispose();
        x?.Dispose();
        normed?.Dispose();
        qkv?.Dispose();
        qkvSplit?.Dispose();
        context?.Dispose();
        projected?.Dispose();
        mlpIn?.Dispose();
        gated?.Dispose();
        mlpOut?.Dispose();
        maskBuf?.Dispose();
        idsBuf?.Dispose();
        posIdsBuf?.Dispose();
    }

    /// <summary>One layer's uploaded weights plus the two per-layer attention parameters.</summary>
    sealed class LayerWeights : IDisposable
    {
        public readonly MemoryBuffer1D<float, Stride1D.Dense> Wqkv;
        public readonly MemoryBuffer1D<float, Stride1D.Dense> Wo;
        public readonly MemoryBuffer1D<float, Stride1D.Dense> Wi;
        public readonly MemoryBuffer1D<float, Stride1D.Dense> WoMlp;
        public readonly MemoryBuffer1D<float, Stride1D.Dense> MlpNormW;

        /// <summary>Null on layer 0, which the CPU treats as an identity attention norm.</summary>
        public readonly MemoryBuffer1D<float, Stride1D.Dense>? AttnNormW;

        /// <summary>Inclusive <c>abs(q - j)</c> limit, or -1 for a globally-attending layer.</summary>
        public readonly int Band;

        /// <summary>Whether this layer uses the full-attention rope theta (and no band).</summary>
        public readonly bool Full;

        public LayerWeights(
            Accelerator acc,
            Dictionary<string, (float[] Data, int[] Shape)> tensors,
            ModernBertConfig config,
            string prefix,
            int index)
        {
            string layerPrefix = $"{prefix}.layers.{index}";
            Wqkv = GpuBuffers.UploadTransposed(acc, tensors, $"{layerPrefix}.attn.Wqkv.weight");
            Wo = GpuBuffers.UploadTransposed(acc, tensors, $"{layerPrefix}.attn.Wo.weight");
            Wi = GpuBuffers.UploadTransposed(acc, tensors, $"{layerPrefix}.mlp.Wi.weight");
            WoMlp = GpuBuffers.UploadTransposed(acc, tensors, $"{layerPrefix}.mlp.Wo.weight");
            MlpNormW = GpuBuffers.UploadPlain(acc, tensors, $"{layerPrefix}.mlp_norm.weight");
            if (index != 0)
                AttnNormW = GpuBuffers.UploadPlain(acc, tensors, $"{layerPrefix}.attn_norm.weight");
            Band = config.Band(index);
            Full = config.IsFullAttention(index);
        }

        public void Dispose()
        {
            Wqkv.Dispose();
            Wo.Dispose();
            Wi.Dispose();
            WoMlp.Dispose();
            MlpNormW.Dispose();
            AttnNormW?.Dispose();
        }
    }
}
