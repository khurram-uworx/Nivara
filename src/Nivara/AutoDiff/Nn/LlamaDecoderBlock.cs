using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// A single Llama-family transformer decoder block: pre-norm self-attention with a residual
/// add, followed by a pre-norm gated SiLU feed-forward network with a second residual add.
/// This mirrors the <c>LlamaDecoderLayer</c> used by SmolLM and the Llama family. Output
/// shape equals the input shape <c>[L, hiddenSize]</c>.
/// </summary>
public sealed class LlamaDecoderBlock<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int hiddenSize;
    readonly int intermediateSize;

    // Fused-path scratch (inference-only). Plain fields not registered with the module system,
    // so they stay invisible to StateDict/serialization. Lazily sized on first fused use.
    T[]? norm;
    T[]? q;
    T[]? k;
    T[]? v;
    T[]? attn;
    T[]? h;
    T[]? residual;
    T[]? gate;
    T[]? gateOut;
    T[]? up;

    /// <summary>Gets the pre-attention RMS norm.</summary>
    public RMSNorm<T> InputNorm { get; }
    /// <summary>Gets the attention module.</summary>
    public LlamaCausalAttention<T> Attention { get; }
    /// <summary>Gets the post-attention RMS norm.</summary>
    public RMSNorm<T> PostNorm { get; }
    /// <summary>Gets the SiLU gated-projection linear.</summary>
    public Linear<T> GateProj { get; }
    /// <summary>Gets the up-projection linear.</summary>
    public Linear<T> UpProj { get; }
    /// <summary>Gets the down-projection linear.</summary>
    public Linear<T> DownProj { get; }

    /// <summary>Gets the hidden (embedding) dimension.</summary>
    public int HiddenSize => hiddenSize;

    /// <summary>
    /// Creates a Llama decoder block.
    /// </summary>
    /// <param name="hiddenSize">Hidden (embedding) dimension</param>
    /// <param name="numHeads">Number of query attention heads</param>
    /// <param name="numKeyValueHeads">Number of key/value attention heads</param>
    /// <param name="intermediateSize">Feed-forward hidden size</param>
    /// <param name="rmsNormEps">RMS normalization stability term</param>
    /// <param name="maxPositionEmbeddings">Maximum position for RoPE tables</param>
    /// <param name="ropeTheta">RoPE inverse-frequency base</param>
    /// <param name="qkvBias">Whether the self-attention query/key/value projections carry a bias
    /// (Qwen2-style models; backlog #384).</param>
    public LlamaDecoderBlock(
        int hiddenSize,
        int numHeads,
        int numKeyValueHeads,
        int intermediateSize,
        float rmsNormEps = 1e-5f,
        int maxPositionEmbeddings = 2048,
        float ropeTheta = 10000f,
        bool qkvBias = false)
    {
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (intermediateSize <= 0) throw new ArgumentOutOfRangeException(nameof(intermediateSize));

        this.hiddenSize = hiddenSize;
        this.intermediateSize = intermediateSize;

        InputNorm = new RMSNorm<T>(hiddenSize, rmsNormEps);
        Attention = new LlamaCausalAttention<T>(hiddenSize, numHeads, numKeyValueHeads, maxPositionEmbeddings, ropeTheta, qkvBias);
        PostNorm = new RMSNorm<T>(hiddenSize, rmsNormEps);
        GateProj = new Linear<T>(hiddenSize, intermediateSize, bias: false);
        UpProj = new Linear<T>(hiddenSize, intermediateSize, bias: false);
        DownProj = new Linear<T>(intermediateSize, hiddenSize, bias: false);

        RegisterModules(InputNorm, Attention, PostNorm, GateProj, UpProj, DownProj);
    }

    /// <summary>
    /// Runs one Llama decoder block over a <c>[L, hiddenSize]</c> input.
    /// </summary>
    /// <param name="input">The input tensor (rank 2)</param>
    /// <returns>The block output with shape <c>[L, hiddenSize]</c></returns>
    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 2) throw new ArgumentException($"LlamaDecoderBlock expects 2D input [L, D], got {input.Rank}D");
        if (input.shape[1] != hiddenSize)
            throw new ArgumentException($"Expected input width {hiddenSize}, got {input.shape[1]}.");

        // Pre-norm self-attention with residual add.
        var attnOut = Attention.Forward(InputNorm.Forward(input));
        var h = ReverseGradOperations.Add(input, attnOut);

        // Pre-norm gated SiLU feed-forward with residual add.
        var ffnIn = PostNorm.Forward(h);
        var gate = Activation.Silu(GateProj.Forward(ffnIn));
        var up = UpProj.Forward(ffnIn);
        var gated = ReverseGradOperations.Multiply(gate, up);
        var mlpOut = DownProj.Forward(gated);
        return ReverseGradOperations.Add(h, mlpOut);
    }

    /// <summary>
    /// Runs one Llama decoder block over a single new token during cached inference. Mirrors
    /// <see cref="Forward(ReverseGradTensor{T})"/> (same residual/FFN structure) but routes the
    /// attention through <see cref="LlamaCausalAttention{T}.ForwardCached"/> so projections are
    /// only computed for the one position.
    /// </summary>
    /// <param name="input">The new-token hidden state <c>[1, hiddenSize]</c></param>
    /// <param name="positionOffset">Absolute position of the new token</param>
    /// <param name="kCache">Per-layer RoPE'd key cache (row-major per-KV-head)</param>
    /// <param name="vCache">Per-layer value cache (row-major per-KV-head)</param>
    /// <param name="cacheLen">Number of tokens already cached before this call</param>
    /// <returns>The block output <c>[1, hiddenSize]</c></returns>
    public ReverseGradTensor<T> ForwardCached(
        ReverseGradTensor<T> input,
        int positionOffset,
        T[] kCache,
        T[] vCache,
        int cacheLen)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        var attnOut = Attention.ForwardCached(InputNorm.Forward(input), positionOffset, kCache, vCache, cacheLen);
        var h = ReverseGradOperations.Add(input, attnOut);

        var ffnIn = PostNorm.Forward(h);
        var gate = Activation.Silu(GateProj.Forward(ffnIn));
        var up = UpProj.Forward(ffnIn);
        var gated = ReverseGradOperations.Multiply(gate, up);
        var mlpOut = DownProj.Forward(gated);
        return ReverseGradOperations.Add(h, mlpOut);
    }

    /// <summary>
    /// Runs one Llama decoder block over a batched <c>[L, hiddenSize]</c> prompt during cached
    /// inference, routing attention through <see cref="LlamaCausalAttention{T}.ForwardPrefill"/>
    /// so the per-KV-head key/value rows are captured into the caches in a single pass. Mirrors
    /// <see cref="Forward(ReverseGradTensor{T})"/> (same residual/FFN structure).
    /// </summary>
    /// <param name="input">The prompt hidden states <c>[L, hiddenSize]</c></param>
    /// <param name="positionOffset">Absolute position of the first prompt row</param>
    /// <param name="kCache">Per-layer RoPE'd key cache (row-major per-KV-head)</param>
    /// <param name="vCache">Per-layer value cache (row-major per-KV-head)</param>
    /// <returns>The block output with shape <c>[L, hiddenSize]</c></returns>
    public ReverseGradTensor<T> ForwardPrefill(
        ReverseGradTensor<T> input,
        int positionOffset,
        T[] kCache,
        T[] vCache)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        var attnOut = Attention.ForwardPrefill(InputNorm.Forward(input), positionOffset, kCache, vCache);
        var h = ReverseGradOperations.Add(input, attnOut);

        var ffnIn = PostNorm.Forward(h);
        var gate = Activation.Silu(GateProj.Forward(ffnIn));
        var up = UpProj.Forward(ffnIn);
        var gated = ReverseGradOperations.Multiply(gate, up);
        var mlpOut = DownProj.Forward(gated);
        return ReverseGradOperations.Add(h, mlpOut);
    }

    /// <summary>
    /// Runs the fused single-token decoder block for cached inference with zero steady-state
    /// heap allocations (scratch arrays are reused across calls). Replicates
    /// <see cref="ForwardCached(ReverseGradTensor{T}, int, T[], T[], int)"/> using the shared
    /// span kernels directly — GEMV fast-path matmuls (<see cref="GradKernels.MatMulTransposedB{T}"/>),
    /// <see cref="AttentionKernels{T}.DecodeAttention"/>, the per-row RMS kernel, RoPE in place,
    /// and elementwise tensor primitives — so the values match the per-op chain bit-for-bit.
    /// The block output is written to <paramref name="output"/> (length ≥ <c>hiddenSize</c>) and
    /// the new RoPE'd pre-repeat key/value rows are appended to the caches at
    /// <paramref name="cacheLen"/>. Inference-only: throws inside a
    /// <see cref="GradientUtils.Grad()"/> scope.
    /// </summary>
    /// <param name="input">The new-token hidden state, <c>[hiddenSize]</c></param>
    /// <param name="output">The block output, length at least <c>hiddenSize</c></param>
    /// <param name="positionOffset">Absolute position of the new token</param>
    /// <param name="kCache">Per-layer RoPE'd key cache (row-major per-KV-head)</param>
    /// <param name="vCache">Per-layer value cache (row-major per-KV-head)</param>
    /// <param name="cacheLen">Number of tokens already cached before this call</param>
    public void ForwardCachedFused(
        ReadOnlySpan<T> input,
        Span<T> output,
        int positionOffset,
        T[] kCache,
        T[] vCache,
        int cacheLen)
    {
        if (input.Length != hiddenSize)
            throw new ArgumentException($"Expected a single-token input of length {hiddenSize}, got {input.Length}.");
        if (output.Length < hiddenSize)
            throw new ArgumentException($"Output span (length {output.Length}) must hold {hiddenSize} elements.");
        if (positionOffset < 0) throw new ArgumentOutOfRangeException(nameof(positionOffset));
        if (cacheLen < 0) throw new ArgumentOutOfRangeException(nameof(cacheLen));
        if (GradientUtils.IsGradEnabled)
            throw new InvalidOperationException("ForwardCachedFused is inference-only; do not call it inside GradientUtils.Grad().");

        int headDim = Attention.HeadDim;
        int numHeads = Attention.NumHeads;
        int numKvHeads = Attention.NumKeyValueHeads;
        int qWidth = numHeads * headDim;
        int kvWidth = numKvHeads * headDim;
        int newLen = cacheLen + 1;
        int needed = newLen * kvWidth;
        if (kCache.Length < needed || vCache.Length < needed)
            throw new ArgumentException("Cache buffers must have capacity for the new token row.");
        T scale = T.CreateChecked(1.0 / Math.Sqrt(headDim));

        if (norm is null) EnsureFusedScratch(qWidth, kvWidth);
        var normBuf = norm!;
        var qBuf = q!;
        var kBuf = k!;
        var vBuf = v!;
        var attnBuf = attn!;
        var hBuf = h!;
        var gateBuf = gate!;
        var gateOutBuf = gateOut!;
        var upBuf = up!;

        // InputNorm (in-place kernel then gamma multiply, same math as RMSNorm.ForwardInference).
        input.CopyTo(normBuf);
        RMSNormKernel<T>.PerRowRMSNormForwardKernel(normBuf, normBuf, 1, hiddenSize, double.CreateChecked(InputNorm.Eps));
        var iGamma = WeightSpan(InputNorm.Weight);
        TensorPrimitives.Multiply(normBuf.AsSpan(0, hiddenSize), iGamma, normBuf.AsSpan(0, hiddenSize));

        // Q/K/V projections (B = W^T for a 1-row a is the GEMV fast path) plus bias.
        GradKernels.MatMulTransposedB(normBuf.AsSpan(0, hiddenSize), WeightSpan(Attention.QProj.Weight), qBuf, 1, hiddenSize, qWidth);
        GradKernels.MatMulTransposedB(normBuf.AsSpan(0, hiddenSize), WeightSpan(Attention.KProj.Weight), kBuf, 1, hiddenSize, kvWidth);
        GradKernels.MatMulTransposedB(normBuf.AsSpan(0, hiddenSize), WeightSpan(Attention.VProj.Weight), vBuf, 1, hiddenSize, kvWidth);
        if (Attention.QProj.Bias is { } qBiasParam)
        {
            var qBias = WeightSpan(qBiasParam);
            var kBias = WeightSpan(Attention.KProj.Bias);
            var vBias = WeightSpan(Attention.VProj.Bias);
            TensorPrimitives.Add(qBuf.AsSpan(0, qWidth), qBias, qBuf.AsSpan(0, qWidth));
            TensorPrimitives.Add(kBuf.AsSpan(0, kvWidth), kBias, kBuf.AsSpan(0, kvWidth));
            TensorPrimitives.Add(vBuf.AsSpan(0, kvWidth), vBias, vBuf.AsSpan(0, kvWidth));
        }

        // RoPE in place (RotaryForward reads both halves before writing, so the same head block
        // as both source and destination is safe), then cache the pre-repeat K/V rows.
        Attention.Rotary.GetPositionTables(positionOffset, 1, out var cos, out var sin);
        int half = headDim / 2;
        for (int b = 0; b < numHeads; b++)
            GradKernels.RotaryForward(qBuf.AsSpan(b * headDim, headDim), cos, sin, qBuf.AsSpan(b * headDim, headDim));
        for (int b = 0; b < numKvHeads; b++)
            GradKernels.RotaryForward(kBuf.AsSpan(b * headDim, headDim), cos, sin, kBuf.AsSpan(b * headDim, headDim));
        kBuf.AsSpan(0, kvWidth).CopyTo(kCache.AsSpan(cacheLen * kvWidth, kvWidth));
        vBuf.AsSpan(0, kvWidth).CopyTo(vCache.AsSpan(cacheLen * kvWidth, kvWidth));

        // GQA decode attention over the full cached prefix (inclusive of the new row).
        AttentionKernels<T>.DecodeAttention(
            qBuf.AsSpan(0, qWidth), kCache.AsSpan(0, needed), vCache.AsSpan(0, needed),
            attnBuf.AsSpan(0, qWidth), newLen, numHeads, numKvHeads, headDim, scale);

        // Output projection into h, then the attention residual in place; the residual is also
        // preserved for the final output add because PostNorm below writes h in place.
        GradKernels.MatMulTransposedB(attnBuf.AsSpan(0, qWidth), WeightSpan(Attention.OProj.Weight), hBuf, 1, qWidth, hiddenSize);
        TensorPrimitives.Add(input, hBuf.AsSpan(0, hiddenSize), hBuf.AsSpan(0, hiddenSize));
        if (residual is not null) hBuf.AsSpan(0, hiddenSize).CopyTo(residual);

        // PostNorm (in place), then the gated SiLU feed-forward.
        RMSNormKernel<T>.PerRowRMSNormForwardKernel(hBuf, hBuf, 1, hiddenSize, double.CreateChecked(PostNorm.Eps));
        var pGamma = WeightSpan(PostNorm.Weight);
        TensorPrimitives.Multiply(hBuf.AsSpan(0, hiddenSize), pGamma, hBuf.AsSpan(0, hiddenSize));

        GradKernels.MatMulTransposedB(hBuf.AsSpan(0, hiddenSize), WeightSpan(GateProj.Weight), gateBuf, 1, hiddenSize, intermediateSize);
        GradKernels.Silu(gateBuf.AsSpan(0, intermediateSize), gateOutBuf.AsSpan(0, intermediateSize));
        GradKernels.MatMulTransposedB(hBuf.AsSpan(0, hiddenSize), WeightSpan(UpProj.Weight), upBuf, 1, hiddenSize, intermediateSize);
        TensorPrimitives.Multiply(gateOutBuf.AsSpan(0, intermediateSize), upBuf.AsSpan(0, intermediateSize), upBuf.AsSpan(0, intermediateSize));

        // Down projection (into the spent attn buffer) + residual from the preserved buffer into
        // the caller's output.
        GradKernels.MatMulTransposedB(upBuf.AsSpan(0, intermediateSize), WeightSpan(DownProj.Weight), attnBuf, 1, intermediateSize, hiddenSize);
        TensorPrimitives.Add(residual!.AsSpan(0, hiddenSize), attnBuf.AsSpan(0, hiddenSize), output);
    }

    /// <summary>
    /// Runs the fused batched decoder block for cached-inference prefill, replicating
    /// <see cref="ForwardPrefill(ReverseGradTensor{T}, int, T[], T[])"/> with the same span
    /// kernels as <see cref="ForwardCachedFused"/> over <c>[qLen, hiddenSize]</c> rows. The
    /// per-layer workspace is ArrayPool-rented for the call (zero steady-state allocations past
    /// pool warm-up), so an L-token prefill builds no per-op tensors on the row chain. The
    /// block output is written to <paramref name="output"/> (length ≥ <c>qLen * hiddenSize</c>)
    /// and the RoPE'd pre-repeat key/value rows are captured into cache rows
    /// <c>[positionOffset, positionOffset + qLen)</c>. Inference-only: throws inside a
    /// <see cref="GradientUtils.Grad()"/> scope.
    /// </summary>
    /// <param name="input">The prompt hidden states, <c>[qLen * hiddenSize]</c></param>
    /// <param name="output">The block output, length at least <c>qLen * hiddenSize</c></param>
    /// <param name="positionOffset">Absolute position of the first prompt row</param>
    /// <param name="kCache">Per-layer RoPE'd key cache (row-major per-KV-head)</param>
    /// <param name="vCache">Per-layer value cache (row-major per-KV-head)</param>
    public void ForwardPrefillFused(
        ReadOnlySpan<T> input,
        Span<T> output,
        int positionOffset,
        T[] kCache,
        T[] vCache)
    {
        if (input.Length == 0 || input.Length % hiddenSize != 0)
            throw new ArgumentException($"Input length {input.Length} must be a positive multiple of the hidden size {hiddenSize}.");
        int qLen = input.Length / hiddenSize;
        if (output.Length < qLen * hiddenSize)
            throw new ArgumentException($"Output span (length {output.Length}) must hold {qLen * hiddenSize} elements.");
        if (positionOffset < 0) throw new ArgumentOutOfRangeException(nameof(positionOffset));
        if (GradientUtils.IsGradEnabled)
            throw new InvalidOperationException("ForwardPrefillFused is inference-only; do not call it inside GradientUtils.Grad().");

        int headDim = Attention.HeadDim;
        int numHeads = Attention.NumHeads;
        int numKvHeads = Attention.NumKeyValueHeads;
        int qWidth = numHeads * headDim;
        int kvWidth = numKvHeads * headDim;
        int needed = (positionOffset + qLen) * kvWidth;
        if (kCache.Length < needed || vCache.Length < needed)
            throw new ArgumentException("Cache buffers must have capacity for the prefill rows.");
        T scale = T.CreateChecked(1.0 / Math.Sqrt(headDim));

        var normArr = ArrayPool<T>.Shared.Rent(Math.Max(qLen * hiddenSize, 1));
        var qArr = ArrayPool<T>.Shared.Rent(Math.Max(qLen * qWidth, 1));
        var kArr = ArrayPool<T>.Shared.Rent(Math.Max(qLen * kvWidth, 1));
        var vArr = ArrayPool<T>.Shared.Rent(Math.Max(qLen * kvWidth, 1));
        var attnArr = ArrayPool<T>.Shared.Rent(Math.Max(qLen * qWidth, 1));
        var hArr = ArrayPool<T>.Shared.Rent(Math.Max(qLen * hiddenSize, 1));
        var residualArr = ArrayPool<T>.Shared.Rent(Math.Max(qLen * hiddenSize, 1));
        var gateArr = ArrayPool<T>.Shared.Rent(Math.Max(qLen * intermediateSize, 1));
        var gateOutArr = ArrayPool<T>.Shared.Rent(Math.Max(qLen * intermediateSize, 1));
        var upArr = ArrayPool<T>.Shared.Rent(Math.Max(qLen * intermediateSize, 1));
        var mlpArr = ArrayPool<T>.Shared.Rent(Math.Max(qLen * hiddenSize, 1));
        try
        {
            // InputNorm over the whole chunk.
            input.CopyTo(normArr);
            RMSNormKernel<T>.PerRowRMSNormForwardKernel(normArr, normArr, qLen, hiddenSize, double.CreateChecked(InputNorm.Eps));
            var iGamma = WeightSpan(InputNorm.Weight);
            for (int r = 0; r < qLen; r++)
                TensorPrimitives.Multiply(normArr.AsSpan(r * hiddenSize, hiddenSize), iGamma, normArr.AsSpan(r * hiddenSize, hiddenSize));

            // Q/K/V projections + per-row bias broadcast.
            var rowSpan = normArr.AsSpan(0, qLen * hiddenSize);
            GradKernels.MatMulTransposedB(rowSpan, WeightSpan(Attention.QProj.Weight), qArr, qLen, hiddenSize, qWidth);
            GradKernels.MatMulTransposedB(rowSpan, WeightSpan(Attention.KProj.Weight), kArr, qLen, hiddenSize, kvWidth);
            GradKernels.MatMulTransposedB(rowSpan, WeightSpan(Attention.VProj.Weight), vArr, qLen, hiddenSize, kvWidth);
            if (Attention.QProj.Bias is { } qBiasParam)
            {
                var qBias = WeightSpan(qBiasParam);
                var kBias = WeightSpan(Attention.KProj.Bias);
                var vBias = WeightSpan(Attention.VProj.Bias);
                for (int r = 0; r < qLen; r++)
                {
                    TensorPrimitives.Add(qArr.AsSpan(r * qWidth, qWidth), qBias, qArr.AsSpan(r * qWidth, qWidth));
                    TensorPrimitives.Add(kArr.AsSpan(r * kvWidth, kvWidth), kBias, kArr.AsSpan(r * kvWidth, kvWidth));
                    TensorPrimitives.Add(vArr.AsSpan(r * kvWidth, kvWidth), vBias, vArr.AsSpan(r * kvWidth, kvWidth));
                }
            }

            // RoPE at absolute positions [positionOffset, positionOffset + qLen), then cache.
            Attention.Rotary.GetPositionTables(positionOffset, qLen, out var cos, out var sin);
            int half = headDim / 2;
            for (int r = 0; r < qLen; r++)
            {
                var rowC = cos.Slice(r * half, half);
                var rowS = sin.Slice(r * half, half);
                for (int b = 0; b < numHeads; b++)
                    GradKernels.RotaryForward(qArr.AsSpan(r * qWidth + b * headDim, headDim), rowC, rowS, qArr.AsSpan(r * qWidth + b * headDim, headDim));
                for (int b = 0; b < numKvHeads; b++)
                    GradKernels.RotaryForward(kArr.AsSpan(r * kvWidth + b * headDim, headDim), rowC, rowS, kArr.AsSpan(r * kvWidth + b * headDim, headDim));
            }
            int rowBytes = qLen * kvWidth;
            kArr.AsSpan(0, rowBytes).CopyTo(kCache.AsSpan(positionOffset * kvWidth, rowBytes));
            vArr.AsSpan(0, rowBytes).CopyTo(vCache.AsSpan(positionOffset * kvWidth, rowBytes));

            // Batched GQA attention over this chunk, then OProj + residual in place.
            AttentionKernels<T>.BatchedAttention(
                qArr.AsSpan(0, qLen * qWidth),
                kCache.AsSpan(positionOffset * kvWidth, rowBytes),
                vCache.AsSpan(positionOffset * kvWidth, rowBytes),
                attnArr.AsSpan(0, qLen * qWidth), qLen, numHeads, numKvHeads, headDim, scale);
            GradKernels.MatMulTransposedB(attnArr.AsSpan(0, qLen * qWidth), WeightSpan(Attention.OProj.Weight), hArr, qLen, qWidth, hiddenSize);
            TensorPrimitives.Add(input, hArr.AsSpan(0, qLen * hiddenSize), hArr.AsSpan(0, qLen * hiddenSize));
            // Preserve the residual for the final output add (PostNorm below writes hArr in place).
            hArr.AsSpan(0, qLen * hiddenSize).CopyTo(residualArr);

            // PostNorm, then the gated SiLU feed-forward.
            RMSNormKernel<T>.PerRowRMSNormForwardKernel(hArr, hArr, qLen, hiddenSize, double.CreateChecked(PostNorm.Eps));
            var pGamma = WeightSpan(PostNorm.Weight);
            for (int r = 0; r < qLen; r++)
                TensorPrimitives.Multiply(hArr.AsSpan(r * hiddenSize, hiddenSize), pGamma, hArr.AsSpan(r * hiddenSize, hiddenSize));

            var hSpan = hArr.AsSpan(0, qLen * hiddenSize);
            GradKernels.MatMulTransposedB(hSpan, WeightSpan(GateProj.Weight), gateArr, qLen, hiddenSize, intermediateSize);
            GradKernels.Silu(gateArr.AsSpan(0, qLen * intermediateSize), gateOutArr.AsSpan(0, qLen * intermediateSize));
            GradKernels.MatMulTransposedB(hSpan, WeightSpan(UpProj.Weight), upArr, qLen, hiddenSize, intermediateSize);
            TensorPrimitives.Multiply(gateOutArr.AsSpan(0, qLen * intermediateSize), upArr.AsSpan(0, qLen * intermediateSize), upArr.AsSpan(0, qLen * intermediateSize));

            // Down projection into the spent attn buffer + residual from the preserved buffer into
            // the caller's output.
            GradKernels.MatMulTransposedB(upArr.AsSpan(0, qLen * intermediateSize), WeightSpan(DownProj.Weight), mlpArr, qLen, intermediateSize, hiddenSize);
            TensorPrimitives.Add(residualArr.AsSpan(0, qLen * hiddenSize), mlpArr.AsSpan(0, qLen * hiddenSize), output);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(normArr);
            ArrayPool<T>.Shared.Return(qArr);
            ArrayPool<T>.Shared.Return(kArr);
            ArrayPool<T>.Shared.Return(vArr);
            ArrayPool<T>.Shared.Return(attnArr);
            ArrayPool<T>.Shared.Return(hArr);
            ArrayPool<T>.Shared.Return(residualArr);
            ArrayPool<T>.Shared.Return(gateArr);
            ArrayPool<T>.Shared.Return(gateOutArr);
            ArrayPool<T>.Shared.Return(upArr);
            ArrayPool<T>.Shared.Return(mlpArr);
        }
    }

    void EnsureFusedScratch(int qWidth, int kvWidth)
    {
        norm = new T[hiddenSize];
        q = new T[qWidth];
        k = new T[kvWidth];
        v = new T[kvWidth];
        attn = new T[qWidth];
        h = new T[hiddenSize];
        residual = new T[hiddenSize];
        gate = new T[intermediateSize];
        gateOut = new T[intermediateSize];
        up = new T[intermediateSize];
    }

    static ReadOnlySpan<T> WeightSpan(Parameter<T>? weight)
    {
        if (weight is null) throw new InvalidOperationException("Expected a non-null parameter.");
        if (!weight.Tensor.Data.TryGetSpan(out var span))
            throw new InvalidOperationException("Parameter data must be null-free.");
        return span;
    }
}
