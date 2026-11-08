namespace Nivara.Silk;

/// <summary>
/// The OpenCL C kernels the BERT-family encoder needs, authored once and shared by the Silk
/// runners and the performance gate that compares them to ILGPU. The host side lives
/// in <see cref="SilkRuntime"/>; this type is pure data and arithmetic-free geometry so it can
/// be unit-checked without a device.
///
/// <para>
/// Each kernel is a port of the ILGPU kernel of the same shape
/// (<c>GemmKernels</c>, <c>AttentionKernels</c>, <c>ElementwiseKernels</c>)
/// with the operation order preserved: same K-reduction order (strictly ascending), same tile
/// geometry, same fused epilogues, so the gate can compare the two backends elementwise.
/// </para>
///
/// <para>
/// <b>Bit-identity.</b> The Row4 GEMM family shares one accumulation body — the
/// <c>GEMM_ROW4_ACCUMULATE</c> macro — so their f32 accumulation text cannot diverge.
/// <c>gemm_row4</c> stores that accumulator and adds no bias; the epilogues add theirs
/// after the macro. A zero bias passed to an epilogue would be <c>acc + 0.0f</c>, which
/// is not the plain store. Its inner
/// product is a plain <c>acc += a * b</c>, and <c>#pragma OPENCL FP_CONTRACT OFF</c> (uppercase;
/// the lowercase spelling the spec also admits is not honoured here) keeps it from being fused,
/// because that is what ILGPU produces: the gate replays each dot product on the host as
/// <c>mul+add</c>, <c>fma</c> and <c>f64</c> in the same ascending-K order, and ILGPU matches
/// <c>mul+add</c> on 100% of elements and <c>fma</c> on ~19%. The OpenCL driver contracts by
/// default, so without the pragma this side computes the fma model and diverges from ILGPU on
/// ~81% of outputs at rounding scale — systematically wrong, and invisible to any tolerance.
/// The pragma also makes the softmax and layer-norm reductions bit-identical. Transcendentals
/// (<c>exp</c> in the softmax and the GELU erf) are not assumed identical to ILGPU's
/// <c>XMath.Exp</c>; those kernels gate on the model's numeric bound instead.
/// </para>
/// </summary>
internal static class SilkKernels
{
    /// <summary>Output tile edge (one thread covers one row of a 16×64 tile).</summary>
    public const int GemmTileSize = 16;

    /// <summary>Output columns per thread in the register-blocked GEMM.</summary>
    public const int GemmBlockCols = 4;

    /// <summary>Output columns per group (<c>TileSize * BlockCols</c>).</summary>
    public const int GemmTileCols = GemmTileSize * GemmBlockCols;

    /// <summary>Column extent of the shared GEMM tile (A in [0, GemmTileSize), B in
    /// [GemmTileSize, GemmRowStride)). The tile is <c>GemmTileSize * GemmRowStride</c> floats; the
    /// kernel indexes it <c>i + j * GemmTileSize</c>.</summary>
    public const int GemmRowStride = GemmTileSize + GemmTileCols;

    /// <summary>Group size for the flat elementwise / row-reduce launches.</summary>
    public const int LinearGroupSize = 256;

    /// <summary>Group size for the fused attention launch (one work item per (b, h, q)).</summary>
    public const int AttentionGroupSize = 64;

    /// <summary>Group size for the fused decode GEMV (one work item per <see cref="GemvBlockCols"/>
    /// consecutive output columns, one row).</summary>
    public const int GemvGroupSize = 32;

    /// <summary>Output columns per thread in the fused decode GEMV. The same value the tile kernels
    /// use (<see cref="GemmBlockCols"/>, the single <c>#define BLOCK</c> in the source); declared
    /// separately because the two geometries are independent even though they share the macro.</summary>
    public const int GemvBlockCols = 4;

    /// <summary>Output columns covered by one decode work-group
    /// (<c>GemvGroupSize * GemvBlockCols</c>). Every modelled width — 896, 128, 4864, 151936 —
    /// is a whole number of these, so the decode shape launches no idle lane there; SmolLM's 576
    /// and 192 are not, and that tail is what the kernel's <c>col &gt;= bCols</c> guard is for.</summary>
    public const int GemvGroupCols = GemvGroupSize * GemvBlockCols;

    /// <summary>Compile-time bound on the attention kernel's per-thread accumulator tile.</summary>
    public const int MaxHeadDim = 64;

    /// <summary>Band sentinel meaning "attend to every non-suppressed key" (global attention).</summary>
    public const int GlobalAttentionBand = -1;

    public const int SlidingAttentionBandMax = 128;

    public const int SlidingAttentionCacheSize = 2 * SlidingAttentionBandMax + 1;

    public const string EmbedSum = "embed_sum";
    public const string GatherRows = "gather_rows";
    public const string GatherRowsBf16 = "gather_rows_bf16";
    public const string GatherRowsF16 = "gather_rows_f16";
    public const string EmbeddingGatherPool = "embedding_gather_pool";
    public const string EmbeddingGatherInput = "embedding_gather_input";
    public const string EmbeddingBackwardPool = "embedding_backward_pool";
    public const string EmbeddingBackwardInput = "embedding_backward_input";
    public const string LayerNormResidual = "layer_norm_residual";
    public const string LayerNorm = "layer_norm";
    public const string Add = "add";
    public const string AddBias = "add_bias";
    public const string SplitColumns = "split_columns";
    public const string Rotary = "rotary";
    public const string GeGlu = "geglu";
    public const string Gemm = "gemm_row4"; public const string GemmBf16 = "gemm_row4_bf16"; public const string GemmF16 = "gemm_row4_f16";
    public const string GemmBias = "gemm_row4_bias";
    public const string GemmRelu = "gemm_row4_relu";
    public const string GemmGelu = "gemm_row4_gelu";
    public const string GemmQkv = "gemm_row4_qkv";
    public const string GemmBiasBf16 = "gemm_row4_bias_bf16";
    public const string GemmBiasF16 = "gemm_row4_bias_f16";
    public const string GemmReluBf16 = "gemm_row4_relu_bf16";
    public const string GemmReluF16 = "gemm_row4_relu_f16";
    public const string GemmGeluBf16 = "gemm_row4_gelu_bf16";
    public const string GemmGeluF16 = "gemm_row4_gelu_f16";
    public const string GemmQkvBf16 = "gemm_row4_qkv_bf16";
    public const string GemmQkvF16 = "gemm_row4_qkv_f16";

    // Measured candidate geometries (issue #784) — not routed by any runner yet.
    public const string GemmReg2x4 = "gemm_reg2x4";
    public const string GemmReg4x4 = "gemm_reg4x4";
    public const string Attention = "batched_attention";
    public const string AttentionSliding = "batched_attention_sliding";

    // SmolLM (Llama-family causal LM) kernels.
    public const string RmsNorm = "rms_norm";
    public const string RmsNormLmHead = "rms_norm_lm_head";
    public const string RmsNormLmHeadBf16 = "rms_norm_lm_head_bf16";
    public const string RmsNormLmHeadF16 = "rms_norm_lm_head_f16";
    public const string RmsNormGemv = "rms_norm_gemv";
    public const string RmsNormGemvBf16 = "rms_norm_gemv_bf16";
    public const string Silu = "silu";
    public const string SwiGlu = "swiglu";
    public const string CausalGqaAttention = "causal_gqa_attention";
    public const string DecodeGqaAttention = "decode_gqa_attention";

    // Fused multi-op helpers (#657).
    public const string GemmAdd = "gemm_row4_add";

    // Flat single-row GEMV + residual for cached decode O-projection (#672, Stage A1).
    public const string GemmGemvAdd = "gemm_gemv_add";
    public const string GemmGemvAddBf16 = "gemm_gemv_add_bf16";

    // True single-work-group fused decode attention + O-projection (#691). Only usable
    // when the device exposes at least numHeads * AttentionGroupSize work-items per group.
    public const string DecodeGqaAttentionOprojFused = "decode_gqa_attention_oproj_fused";
    // Fused single-token MLP block for cached decode (#672, Stage B): norm + gate/up + SwiGLU + down + residual.
    public const string MlpBlock = "mlp_block";
    public const string MlpBlockBf16 = "mlp_block_bf16";

    // Device-side reduction for causal-LM greedy decode (#663).
    public const string ArgMax = "argmax";

    // Analytics (incident) kernels.
    public const string GroupedHashReducePartial = "grouped_hash_reduce_partial";
    public const string GroupedMultiStatReducePartial = "grouped_multistat_reduce_partial";

    // Analytics (incident) kernels, phase B.
    public const string RollingMean = "rolling_mean";

    // Analytics (incident) kernels, phase C.
    public const string RollingMax = "rolling_max";
    public const string RollingMin = "rolling_min";

    // Vision (conv) kernels.
    public const string Conv2d = "conv2d";
    public const string BatchNormEval = "batchnorm_eval";
    public const string Relu = "relu";
    public const string Relu6 = "relu6";
    public const string MaxPool2d = "maxpool2d";
    public const string AdaptiveAvgPool2d = "adaptive_avgpool2d";

    // Text-classifier backward kernels (#900): transposed-operand GEMMs, elementwise backward,
    // reductions, and the fused Adam step.
    public const string GemmNt = "gemm_row4_nt";
    public const string GemmNtResidual = "gemm_row4_nt_residual";
    public const string GemmTn = "gemm_row4_tn";
    public const string ReluBackward = "relu_backward";
    public const string GeluBackward = "gelu_backward";
    public const string LayerNormBackward = "layer_norm_backward";
    public const string ColSum = "col_sum";
    public const string Fill = "fill";
    public const string Conv2dBackwardWeight = "conv2d_backward_weight";
    public const string Conv2dBackwardInput = "conv2d_backward_input";
    public const string Conv2dBackwardBias = "conv2d_backward_bias";
    public const string AdamUpdate = "adam_update";
    public const string SoftmaxCrossEntropyBackward = "softmax_cross_entropy_backward";

    // Rag-learn fused late-fusion helpers (#915 follow-up).
    public const string FusionForward = "fusion_forward";
    public const string FusionWgrad = "fusion_wgrad";

    // Transformer-training backward kernels (#956): the multi-head attention gradient, split into
    // three deterministic launches so each output element has one writer and an ascending loop.
    public const string AttentionBackwardPrep = "attention_backward_prep";
    public const string AttentionBackwardDq = "attention_backward_dq";
    public const string AttentionBackwardDkv = "attention_backward_dkv";

    // Training glue (#956): the split GELU (keeps the pre-activation), the row-major embedding
    // scatter, and the gather_rows backward.
    public const string Gelu = "gelu";
    public const string EmbeddingBackwardRows = "embedding_backward_rows";
    public const string ScatterRows = "scatter_rows";

    /// <summary>Every kernel the translation unit defines, in build order.</summary>
    public static readonly string[] KernelNames =
    [
        EmbedSum, GatherRows, GatherRowsBf16, GatherRowsF16, LayerNormResidual, LayerNorm, Add, AddBias, SplitColumns, Rotary, GeGlu,        Gemm, GemmF16, GemmBias, GemmBiasBf16, GemmBiasF16, GemmRelu, GemmReluBf16, GemmReluF16, GemmGelu, GemmGeluBf16, GemmGeluF16, GemmQkv, GemmQkvBf16, GemmQkvF16, GemmReg2x4, GemmReg4x4, Attention, AttentionSliding,        RmsNorm, RmsNormLmHead, RmsNormLmHeadBf16, RmsNormLmHeadF16, RmsNormGemv, Silu, SwiGlu, CausalGqaAttention, DecodeGqaAttention, ArgMax,        GemmAdd, GemmGemvAdd, GemmGemvAddBf16, MlpBlock, MlpBlockBf16, RmsNormGemvBf16, GemmBf16,
        DecodeGqaAttentionOprojFused,
        Conv2d, BatchNormEval, Relu, Relu6, MaxPool2d, AdaptiveAvgPool2d,
        GroupedHashReducePartial, GroupedMultiStatReducePartial, RollingMean, RollingMax, RollingMin,
        GemmNt, GemmNtResidual, GemmTn, ReluBackward, GeluBackward, LayerNormBackward, ColSum, Fill,
        SoftmaxCrossEntropyBackward,
        Conv2dBackwardWeight, Conv2dBackwardInput, Conv2dBackwardBias, AdamUpdate,
        FusionForward, FusionWgrad,
        AttentionBackwardPrep, AttentionBackwardDq, AttentionBackwardDkv,
        Gelu, EmbeddingBackwardRows, ScatterRows,
        EmbeddingGatherPool, EmbeddingGatherInput, EmbeddingBackwardPool, EmbeddingBackwardInput,
    ];

    /// <summary>Global work size (dim 0) for a GEMM producing <paramref name="aRows"/> rows.</summary>
    public static int GemmGlobalRows(int aRows) => RoundUp(aRows, GemmTileSize);

    /// <summary>Global work size (dim 0) for the measured 2×4 geometry (32-row tile, 16 threads).</summary>
    public static int GemmGlobalRowsReg2x4(int aRows) => RoundUp(aRows, GemmTileSize * 2) / 2;

    /// <summary>Global work size (dim 0) for the measured 4×4 geometry (64-row tile, 16 threads).</summary>
    public static int GemmGlobalRowsReg4x4(int aRows) => RoundUp(aRows, GemmTileSize * 4) / 4;

    /// <summary>Global work size (dim 1) for a GEMM producing <paramref name="bCols"/> columns.
    /// A work-group is 16 threads wide here but covers <see cref="GemmTileCols"/> output columns
    /// (each thread takes <see cref="GemmBlockCols"/> of them), so the group count is
    /// <c>ceil(bCols / GemmTileCols)</c> and the thread count is that times
    /// <see cref="GemmTileSize"/>. Returning <c>RoundUp(bCols, GemmTileCols)</c> instead
    /// over-launches by <see cref="GemmBlockCols"/>×: the surplus groups run the whole K-loop and
    /// then discard every result (the <c>outCol &lt; bCols</c> guard), which is pure waste.</summary>
    public static int GemmGlobalCols(int bCols) => RoundUp(bCols, GemmTileCols) / GemmBlockCols;

    /// <summary>Global work size (dim 0) for the fused decode GEMV over <paramref name="cols"/>
    /// output columns. The launch is flat — one row, so there is no row dimension — and the group
    /// count is <c>ceil(cols / GemvGroupCols)</c>, giving
    /// <c>RoundUp(cols, GemvGroupCols) / GemvBlockCols</c> threads. A zero width contributes no
    /// group, which is how a trailing slot is omitted; the group count the kernel recomputes from
    /// the slot column counts is the same arithmetic.</summary>
    public static int GemvGlobalCols(int cols) => RoundUp(cols, GemvGroupCols) / GemvBlockCols;

    /// <summary>Global work size for a flat 1-D launch guarded by an <c>idx &gt;= n</c> early return.</summary>
    public static int LinearGlobal(int total) => RoundUp(Math.Max(1, total), LinearGroupSize);

    /// <summary>Global work size for the attention launch (rounded to whole work-groups).</summary>
    public static int AttentionGlobal(int total) => RoundUp(Math.Max(1, total), AttentionGroupSize);

    static int RoundUp(int value, int multiple) => ((value + multiple - 1) / multiple) * multiple;

    public const string Source = """
        #pragma OPENCL FP_CONTRACT OFF
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable

        #define TILE 16
        #define BLOCK 4
        #define TILECOLS 64
        #define ROWSTRIDE 80
        #define MAXHEADDIM 64
        #define ATTNGROUP 64
        #define GEMVLANES 32
        #define GEMVCOLS 128
        #define BF16_TO_F32(x) as_float(((uint)(x)) << 16)
        #define F16_TO_F32(x) (float)as_half((ushort)(x))

        inline float gelu_exact(float v)
        {
            float z = v * 0.7071067811865475f;
            float az = fabs(z);
            float t = 1.0f / (1.0f + 0.3275911f * az);
            float p = 1.061405429f * t - 1.453152027f;
            p = p * t + 1.421413741f;
            p = p * t - 0.284496736f;
            p = p * t + 0.254829592f;
            float e = 1.0f - p * t * exp(-az * az);
            if (z < 0.0f)
                e = -e;
            return 0.5f * v * (1.0f + e);
        }

        __kernel void embed_sum(
            __global const int* ids,
            __global const int* posIds,
            __global const float* wordEmb,
            __global const float* posEmb,
            __global const float* tokenType,
            __global float* y,
            const uint n,
            const uint hidden,
            const uint includeTt)
        {
            uint idx = get_global_id(0);
            if (idx >= n) return;
            uint row = idx / hidden;
            uint col = idx - row * hidden;
            float v = wordEmb[(uint)ids[row] * hidden + col] + posEmb[(uint)posIds[row] * hidden + col];
            if (includeTt != 0u)
                v += tokenType[col];
            y[idx] = v;
        }

        __kernel void gather_rows(
            __global const float* table,
            __global const int* ids,
            __global float* y,
            const uint n,
            const uint hidden)
        {
            uint idx = get_global_id(0);
            if (idx >= n) return;
            uint row = idx / hidden;
            uint col = idx - row * hidden;
            y[idx] = table[(uint)ids[row] * hidden + col];
        }

        __kernel void gather_rows_bf16(
            __global const ushort* table,
            __global const int* ids,
            __global float* y,
            const uint n,
            const uint hidden)
        {
            uint idx = get_global_id(0);
            if (idx >= n) return;
            uint row = idx / hidden;
            uint col = idx - row * hidden;
            y[idx] = BF16_TO_F32(table[(uint)ids[row] * hidden + col]);
        }

        __kernel void gather_rows_f16(
            __global const ushort* table,
            __global const int* ids,
            __global float* y,
            const uint n,
            const uint hidden)
        {
            uint idx = get_global_id(0);
            if (idx >= n) return;
            uint row = idx / hidden;
            uint col = idx - row * hidden;
            y[idx] = F16_TO_F32(table[(uint)ids[row] * hidden + col]);
        }

        __kernel void layer_norm_residual(
            __global const float* a,
            __global const float* b,
            __global const float* gamma,
            __global const float* beta,
            __global float* y,
            const uint rows,
            const uint cols,
            const float eps)
        {
            uint r = get_global_id(0);
            if (r >= rows) return;
            uint off = r * cols;
            float sum = 0.0f;
            for (uint c = 0; c < cols; ++c)
                sum += a[off + c] + b[off + c];
            float mean = sum / (float)cols;
            float sumSq = 0.0f;
            for (uint c = 0; c < cols; ++c)
            {
                float d = a[off + c] + b[off + c] - mean;
                sumSq += d * d;
            }
            float invStd = 1.0f / sqrt(sumSq / (float)cols + eps);
            for (uint c = 0; c < cols; ++c)
                y[off + c] = (a[off + c] + b[off + c] - mean) * invStd * gamma[c] + beta[c];
        }

        /* LayerNorm over x, not over a+b. A zero partner in layer_norm_residual is a different
           reduction (a[c]+b[c]) and is not this kernel. y may alias x: the owning work item
           finishes both reductions before it writes the row. */
        __kernel void layer_norm(
            __global const float* x,
            __global const float* gamma,
            __global const float* beta,
            __global float* y,
            const uint rows,
            const uint cols,
            const float eps)
        {
            uint r = get_global_id(0);
            if (r >= rows) return;
            uint off = r * cols;
            float sum = 0.0f;
            for (uint c = 0; c < cols; ++c)
                sum += x[off + c];
            float mean = sum / (float)cols;
            float sumSq = 0.0f;
            for (uint c = 0; c < cols; ++c)
            {
                float d = x[off + c] - mean;
                sumSq += d * d;
            }
            float invStd = 1.0f / sqrt(sumSq / (float)cols + eps);
            for (uint c = 0; c < cols; ++c)
                y[off + c] = (x[off + c] - mean) * invStd * gamma[c] + beta[c];
        }

        /* y may alias a. Each work item reads both inputs before it writes its one element. */
        __kernel void add(
            __global const float* a,
            __global const float* b,
            __global float* y,
            const uint n)
        {
            uint idx = get_global_id(0);
            if (idx >= n) return;
            y[idx] = a[idx] + b[idx];
        }

        __kernel void add_bias(
            __global const float* x,
            __global const float* bias,
            __global float* y,
            const uint rows,
            const uint cols)
        {
            uint idx = get_global_id(0);
            if (idx >= rows * cols) return;
            y[idx] = x[idx] + bias[idx % cols];
        }

        /* Row-interleaved [parts] blocks to a dense block. A contiguous sub-view of the source
           is right for row 0 and wrong for every row after it. */
        __kernel void split_columns(
            __global const float* src,
            __global float* dst,
            const uint rows,
            const uint blockCols,
            const uint parts,
            const uint part)
        {
            uint idx = get_global_id(0);
            if (idx >= rows * blockCols) return;
            uint row = idx / blockCols;
            uint col = idx - row * blockCols;
            dst[idx] = src[row * blockCols * parts + part * blockCols + col];
        }

        /* Half-split RoPE. One work item owns both halves of one pair, so y may alias x.
           cosTab/sinTab are not named cos/sin, and the pair width is not named half:
           those are OpenCL builtins or type names, and this driver rejects the program. */
        __kernel void rotary(
            __global const float* x,
            __global const int* posIds,
            __global const float* cosTab,
            __global const float* sinTab,
            __global float* y,
            const uint rows,
            const uint numHeads,
            const uint headDim)
        {
            uint pair = headDim / 2u;
            uint idx = get_global_id(0);
            if (idx >= rows * numHeads * pair) return;
            uint i = idx % pair;
            uint rem = idx / pair;
            uint h = rem % numHeads;
            uint row = rem / numHeads;
            uint t = (uint)posIds[row] * pair + i;
            float c = cosTab[t];
            float s = sinTab[t];
            uint b = row * (numHeads * headDim) + h * headDim;
            float x0 = x[b + i];
            float x1 = x[b + i + pair];
            y[b + i] = x0 * c - x1 * s;
            y[b + i + pair] = x0 * s + x1 * c;
        }

        /* First half is the GELU, second half is the gate. Reversing them still looks plausible. */
        __kernel void geglu(
            __global const float* x,
            __global float* y,
            const uint rows,
            const uint intermediate)
        {
            uint idx = get_global_id(0);
            if (idx >= rows * intermediate) return;
            uint col = idx % intermediate;
            uint row = idx / intermediate;
            uint xBase = row * 2u * intermediate;
            y[idx] = gelu_exact(x[xBase + col]) * x[xBase + intermediate + col];
        }

        /* The accumulation core every f32 Row4 GEMM variant shares: the K-loop's barrier, the
           unrolled ascending-K mul+add body and the c0..c3 column bases. Deliberately one
           macro — the byte-identity claim rests on every variant adding the same f32 values
           to the same accumulator in the same order, so this text lives in exactly one place
           and cannot drift between the forward, BF16 and NT/TN backward variants. The tile
           loads before it and the store after it are the only per-variant differences. */
        #define GEMM_ACCUMULATE_CORE \
            barrier(CLK_LOCAL_MEM_FENCE); \
            _Pragma("unroll 16") for (uint k = 0; k < TILE; ++k) { \
                float av = smem[x + k * TILE]; \
                uint bBase = TILE + y * BLOCK; \
                a0 += av * smem[k + (bBase + 0) * TILE]; \
                a1 += av * smem[k + (bBase + 1) * TILE]; \
                a2 += av * smem[k + (bBase + 2) * TILE]; \
                a3 += av * smem[k + (bBase + 3) * TILE]; \
            } \
            barrier(CLK_LOCAL_MEM_FENCE); \
            } \
            uint c0 = colBase + y * BLOCK + 0; \
            uint c1 = colBase + y * BLOCK + 1; \
            uint c2 = colBase + y * BLOCK + 2; \
            uint c3 = colBase + y * BLOCK + 3;

        /* Shared tile loads and accumulation for the forward Row4 GEMM family, including the
           bias-free store. The accumulation body is `GEMM_ACCUMULATE_CORE`; the loads above
           it and the store after the macro are the only per-variant differences.

           The scratch tile is a private layout: element [i, j] lives at i + j*TILE, so the
           first index is the contiguous one. Only the store/load pairs have to agree on it, and
           the values are identical whichever way it is laid out — this way is simply the one
           that makes a warp's A-tile read (`smem[x + k*TILE]` is x + 16k) land on 16 consecutive
           banks rather than eight lanes per bank. It still fits the TILE*ROWSTRIDE floats
           exactly: the largest index is 15 + 79*16 = 1279.

           The inner product is a plain mul-then-add, deliberately not fma(). ILGPU does not fuse
           `acc += a * b`; the gate replays each dot product on the host as mul+add, fma and f64
           in the same ascending-K order and ILGPU matches mul+add on 100% of elements and fma on
           only ~19% (the ones where fusing happens not to change the rounding). The OpenCL driver
           fuses by default, so `#pragma OPENCL FP_CONTRACT OFF` at the top of the translation
           unit is load-bearing — and an explicit fma() here would defeat it, which is exactly
           what a fused inner product does: it diverges from ILGPU on ~81% of outputs at rounding
           scale, systematically wrong and invisible to any tolerance.

           Leaves a0..a3, outRow, colBase, y and c0..c3 in scope. */
        #define GEMM_ROW4_ACCUMULATE \
            __local float smem[TILE * ROWSTRIDE]; \
            uint x = get_local_id(0); \
            uint y = get_local_id(1); \
            uint outRow = get_global_id(0); \
            uint colBase = get_group_id(1) * TILECOLS; \
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f; \
            for (uint k0 = 0; k0 < aCols; k0 += TILE) { \
                if (y < 4u) { \
                    uint aColBase = k0 + y * 4u; \
                    uint aBase = outRow * aCols + aColBase; \
                    if (outRow < aRows && (aColBase + 3u) < aCols) { \
                        float4 av = vload4(0, a + aBase); \
                        smem[x + (y * 4u + 0u) * TILE] = av.s0; \
                        smem[x + (y * 4u + 1u) * TILE] = av.s1; \
                        smem[x + (y * 4u + 2u) * TILE] = av.s2; \
                        smem[x + (y * 4u + 3u) * TILE] = av.s3; \
                    } else { \
                        for (uint aw = 0u; aw < 4u; ++aw) { \
                            uint aCol = aColBase + aw; \
                            smem[x + (y * 4u + aw) * TILE] = (outRow < aRows && aCol < aCols) ? a[outRow * aCols + aCol] : 0.0f; \
                        } \
                    } \
                } \
                uint bRow = k0 + x; \
                { \
                    uint outCol0 = colBase + y * BLOCK; \
                    uint bBase = bRow * bCols + outCol0; \
                    if (bRow < aCols && (outCol0 + 3u) < bCols) { \
                        float4 bv = vload4(0, b + bBase); \
                        smem[x + (TILE + y * BLOCK + 0) * TILE] = bv.s0; \
                        smem[x + (TILE + y * BLOCK + 1) * TILE] = bv.s1; \
                        smem[x + (TILE + y * BLOCK + 2) * TILE] = bv.s2; \
                        smem[x + (TILE + y * BLOCK + 3) * TILE] = bv.s3; \
                    } else { \
                        for (uint w = 0; w < BLOCK; ++w) { \
                            uint outCol = outCol0 + w; \
                            smem[x + (TILE + y * BLOCK + w) * TILE] = \
                                (bRow < aCols && outCol < bCols) ? b[bRow * bCols + outCol] : 0.0f; \
                        } \
                    } \
                } \
            GEMM_ACCUMULATE_CORE



        #define GEMM_ROW4_ACCUMULATE_BF16 \
            __local float smem[TILE * ROWSTRIDE]; \
            uint x = get_local_id(0); \
            uint y = get_local_id(1); \
            uint outRow = get_global_id(0); \
            uint colBase = get_group_id(1) * TILECOLS; \
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f; \
            for (uint k0 = 0; k0 < aCols; k0 += TILE) { \
                if (y < 4u) { \
                    uint aColBase = k0 + y * 4u; \
                    uint aBase = outRow * aCols + aColBase; \
                    if (outRow < aRows && (aColBase + 3u) < aCols) { \
                        float4 av = vload4(0, a + aBase); \
                        smem[x + (y * 4u + 0u) * TILE] = av.s0; \
                        smem[x + (y * 4u + 1u) * TILE] = av.s1; \
                        smem[x + (y * 4u + 2u) * TILE] = av.s2; \
                        smem[x + (y * 4u + 3u) * TILE] = av.s3; \
                    } else { \
                        for (uint aw = 0u; aw < 4u; ++aw) { \
                            uint aCol = aColBase + aw; \
                            smem[x + (y * 4u + aw) * TILE] = (outRow < aRows && aCol < aCols) ? a[outRow * aCols + aCol] : 0.0f; \
                        } \
                    } \
                } \
                uint bRow = k0 + x; \
                { \
                    uint outCol0 = colBase + y * BLOCK; \
                    uint bBase = bRow * bCols + outCol0; \
                    if (bRow < aCols && (outCol0 + 3u) < bCols) { \
                        ushort4 bv = vload4(0, b + bBase); \
                        smem[x + (TILE + y * BLOCK + 0) * TILE] = BF16_TO_F32(bv.s0); \
                        smem[x + (TILE + y * BLOCK + 1) * TILE] = BF16_TO_F32(bv.s1); \
                        smem[x + (TILE + y * BLOCK + 2) * TILE] = BF16_TO_F32(bv.s2); \
                        smem[x + (TILE + y * BLOCK + 3) * TILE] = BF16_TO_F32(bv.s3); \
                    } else { \
                        for (uint w = 0; w < BLOCK; ++w) { \
                            uint outCol = outCol0 + w; \
                            smem[x + (TILE + y * BLOCK + w) * TILE] = \
                                (bRow < aCols && outCol < bCols) ? BF16_TO_F32(b[bRow * bCols + outCol]) : 0.0f; \
                        } \
                    } \
                } \
            GEMM_ACCUMULATE_CORE

        #define GEMM_ROW4_ACCUMULATE_F16 \
            __local float smem[TILE * ROWSTRIDE]; \
            uint x = get_local_id(0); \
            uint y = get_local_id(1); \
            uint outRow = get_global_id(0); \
            uint colBase = get_group_id(1) * TILECOLS; \
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f; \
            for (uint k0 = 0; k0 < aCols; k0 += TILE) { \
                if (y < 4u) { \
                    uint aColBase = k0 + y * 4u; \
                    uint aBase = outRow * aCols + aColBase; \
                    if (outRow < aRows && (aColBase + 3u) < aCols) { \
                        float4 av = vload4(0, a + aBase); \
                        smem[x + (y * 4u + 0u) * TILE] = av.s0; \
                        smem[x + (y * 4u + 1u) * TILE] = av.s1; \
                        smem[x + (y * 4u + 2u) * TILE] = av.s2; \
                        smem[x + (y * 4u + 3u) * TILE] = av.s3; \
                    } else { \
                        for (uint aw = 0u; aw < 4u; ++aw) { \
                            uint aCol = aColBase + aw; \
                            smem[x + (y * 4u + aw) * TILE] = (outRow < aRows && aCol < aCols) ? a[outRow * aCols + aCol] : 0.0f; \
                        } \
                    } \
                } \
                uint bRow = k0 + x; \
                { \
                    uint outCol0 = colBase + y * BLOCK; \
                    uint bBase = bRow * bCols + outCol0; \
                    if (bRow < aCols && (outCol0 + 3u) < bCols) { \
                        ushort4 bv = vload4(0, b + bBase); \
                        smem[x + (TILE + y * BLOCK + 0) * TILE] = F16_TO_F32(bv.s0); \
                        smem[x + (TILE + y * BLOCK + 1) * TILE] = F16_TO_F32(bv.s1); \
                        smem[x + (TILE + y * BLOCK + 2) * TILE] = F16_TO_F32(bv.s2); \
                        smem[x + (TILE + y * BLOCK + 3) * TILE] = F16_TO_F32(bv.s3); \
                    } else { \
                        for (uint w = 0; w < BLOCK; ++w) { \
                            uint outCol = outCol0 + w; \
                            smem[x + (TILE + y * BLOCK + w) * TILE] = \
                                (bRow < aCols && outCol < bCols) ? F16_TO_F32(b[bRow * bCols + outCol]) : 0.0f; \
                        } \
                    } \
                } \
            GEMM_ACCUMULATE_CORE

        /* Backward GEMM variants. Same `GEMM_ACCUMULATE_CORE` accumulation as the forward
           Row4 GEMMs (ascending-K mul + add, no contraction), so a host replay of the same
           order is byte-identical; only the tile LOADS change. gemm_row4_nt takes B stored row-major [bCols, aCols] and
           transposes it in the load (dH = dZ·W2ᵀ, dPooled = dH·W1ᵀ, dA = dH·W1ᵀ(branch));
           gemm_row4_tn takes A stored row-major [aCols, aRows] and transposes it in the load
           (dW2 = hᵀ·dZ, dW1 = pooledᵀ·dH). The transposed loads are scalar — the vload4 path
           needs elements contiguous along the load direction, which a transposed operand
           destroys — but the load is a small fraction of the MAC loop. */
        #define GEMM_NT_ACCUMULATE \
            __local float smem[TILE * ROWSTRIDE]; \
            uint x = get_local_id(0); \
            uint y = get_local_id(1); \
            uint outRow = get_global_id(0); \
            uint colBase = get_group_id(1) * TILECOLS; \
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f; \
            for (uint k0 = 0; k0 < aCols; k0 += TILE) { \
                if (y < 4u) { \
                    uint aColBase = k0 + y * 4u; \
                    uint aBase = outRow * aCols + aColBase; \
                    if (outRow < aRows && (aColBase + 3u) < aCols) { \
                        float4 av = vload4(0, a + aBase); \
                        smem[x + (y * 4u + 0u) * TILE] = av.s0; \
                        smem[x + (y * 4u + 1u) * TILE] = av.s1; \
                        smem[x + (y * 4u + 2u) * TILE] = av.s2; \
                        smem[x + (y * 4u + 3u) * TILE] = av.s3; \
                    } else { \
                        for (uint aw = 0u; aw < 4u; ++aw) { \
                            uint aCol = aColBase + aw; \
                            smem[x + (y * 4u + aw) * TILE] = (outRow < aRows && aCol < aCols) ? a[outRow * aCols + aCol] : 0.0f; \
                        } \
                    } \
                } \
                uint bRow = k0 + x; \
                { \
                    uint outCol0 = colBase + y * BLOCK; \
                    for (uint w = 0; w < BLOCK; ++w) { \
                        uint outCol = outCol0 + w; \
                        smem[x + (TILE + y * BLOCK + w) * TILE] = \
                            (bRow < aCols && outCol < bCols) ? b[outCol * aCols + bRow] : 0.0f; \
                    } \
                } \
            GEMM_ACCUMULATE_CORE

        #define GEMM_TN_ACCUMULATE \
            __local float smem[TILE * ROWSTRIDE]; \
            uint x = get_local_id(0); \
            uint y = get_local_id(1); \
            uint outRow = get_global_id(0); \
            uint colBase = get_group_id(1) * TILECOLS; \
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f; \
            for (uint k0 = 0; k0 < aCols; k0 += TILE) { \
                if (y < 4u) { \
                    uint aColBase = k0 + y * 4u; \
                    for (uint aw = 0u; aw < 4u; ++aw) { \
                        uint aCol = aColBase + aw; \
                        smem[x + (y * 4u + aw) * TILE] = \
                            (outRow < aRows && aCol < aCols) ? a[aCol * aRows + outRow] : 0.0f; \
                    } \
                } \
                uint bRow = k0 + x; \
                { \
                    uint outCol0 = colBase + y * BLOCK; \
                    uint bBase = bRow * bCols + outCol0; \
                    if (bRow < aCols && (outCol0 + 3u) < bCols) { \
                        float4 bv = vload4(0, b + bBase); \
                        smem[x + (TILE + y * BLOCK + 0) * TILE] = bv.s0; \
                        smem[x + (TILE + y * BLOCK + 1) * TILE] = bv.s1; \
                        smem[x + (TILE + y * BLOCK + 2) * TILE] = bv.s2; \
                        smem[x + (TILE + y * BLOCK + 3) * TILE] = bv.s3; \
                    } else { \
                        for (uint w = 0; w < BLOCK; ++w) { \
                            uint outCol = outCol0 + w; \
                            smem[x + (TILE + y * BLOCK + w) * TILE] = \
                                (bRow < aCols && outCol < bCols) ? b[bRow * bCols + outCol] : 0.0f; \
                        } \
                    } \
                } \
            GEMM_ACCUMULATE_CORE

        inline uint qkv_dest(uint j, uint outRow, uint aRows, uint blockWidth)
        {
            uint block = j / blockWidth;
            uint colInBlock = j - block * blockWidth;
            return block * aRows * blockWidth + outRow * blockWidth + colInBlock;
        }

        /* Bias-free store. Not gemm_row4_bias with a zero bias: acc + 0.0f canonicalizes -0. */
        __kernel void gemm_row4(
            __global const float* a,
            __global const float* b,
            __global float* c,
            const uint aRows,
            const uint aCols,
            const uint bCols)
        {
            GEMM_ROW4_ACCUMULATE
            if (outRow < aRows)
            {
                uint rowBase = outRow * bCols + colBase + y * BLOCK;
                if (c0 < bCols) c[rowBase + 0] = a0;
                if (c1 < bCols) c[rowBase + 1] = a1;
                if (c2 < bCols) c[rowBase + 2] = a2;
                if (c3 < bCols) c[rowBase + 3] = a3;
            }
        }

        __kernel void gemm_row4_bias(
            __global const float* a,
            __global const float* b,
            __global float* c,
            __global const float* bias,
            const uint aRows,
            const uint aCols,
            const uint bCols)
        {
            GEMM_ROW4_ACCUMULATE
            if (outRow < aRows)
            {
                float b0 = c0 < bCols ? bias[c0] : 0.0f;
                float b1 = c1 < bCols ? bias[c1] : 0.0f;
                float b2 = c2 < bCols ? bias[c2] : 0.0f;
                float b3 = c3 < bCols ? bias[c3] : 0.0f;
                uint rowBase = outRow * bCols + colBase + y * BLOCK;
                if (c0 < bCols) c[rowBase + 0] = a0 + b0;
                if (c1 < bCols) c[rowBase + 1] = a1 + b1;
                if (c2 < bCols) c[rowBase + 2] = a2 + b2;
                if (c3 < bCols) c[rowBase + 3] = a3 + b3;
            }
        }

        __kernel void gemm_row4_bias_f16(
            __global const float* a,
            __global const ushort* b,
            __global float* c,
            __global const float* bias,
            const uint aRows,
            const uint aCols,
            const uint bCols)
        {
            GEMM_ROW4_ACCUMULATE_F16
            if (outRow < aRows)
            {
                float b0 = c0 < bCols ? bias[c0] : 0.0f;
                float b1 = c1 < bCols ? bias[c1] : 0.0f;
                float b2 = c2 < bCols ? bias[c2] : 0.0f;
                float b3 = c3 < bCols ? bias[c3] : 0.0f;
                uint rowBase = outRow * bCols + colBase + y * BLOCK;
                if (c0 < bCols) c[rowBase + 0] = a0 + b0;
                if (c1 < bCols) c[rowBase + 1] = a1 + b1;
                if (c2 < bCols) c[rowBase + 2] = a2 + b2;
                if (c3 < bCols) c[rowBase + 3] = a3 + b3;
            }
        }


        __kernel void gemm_row4_bias_bf16(
            __global const float* a,
            __global const ushort* b,
            __global float* c,
            __global const float* bias,
            const uint aRows,
            const uint aCols,
            const uint bCols)
        {
            GEMM_ROW4_ACCUMULATE_BF16
            if (outRow < aRows)
            {
                float b0 = c0 < bCols ? bias[c0] : 0.0f;
                float b1 = c1 < bCols ? bias[c1] : 0.0f;
                float b2 = c2 < bCols ? bias[c2] : 0.0f;
                float b3 = c3 < bCols ? bias[c3] : 0.0f;
                uint rowBase = outRow * bCols + colBase + y * BLOCK;
                if (c0 < bCols) c[rowBase + 0] = a0 + b0;
                if (c1 < bCols) c[rowBase + 1] = a1 + b1;
                if (c2 < bCols) c[rowBase + 2] = a2 + b2;
                if (c3 < bCols) c[rowBase + 3] = a3 + b3;
            }
        }


        __kernel void gemm_row4_gelu(
            __global const float* a,
            __global const float* b,
            __global float* c,
            __global const float* bias,
            const uint aRows,
            const uint aCols,
            const uint bCols)
        {
            GEMM_ROW4_ACCUMULATE
            if (outRow < aRows)
            {
                float b0 = c0 < bCols ? bias[c0] : 0.0f;
                float b1 = c1 < bCols ? bias[c1] : 0.0f;
                float b2 = c2 < bCols ? bias[c2] : 0.0f;
                float b3 = c3 < bCols ? bias[c3] : 0.0f;
                uint rowBase = outRow * bCols + colBase + y * BLOCK;
                if (c0 < bCols) c[rowBase + 0] = gelu_exact(a0 + b0);
                if (c1 < bCols) c[rowBase + 1] = gelu_exact(a1 + b1);
                if (c2 < bCols) c[rowBase + 2] = gelu_exact(a2 + b2);
                if (c3 < bCols) c[rowBase + 3] = gelu_exact(a3 + b3);
            }
        }

        __kernel void gemm_row4_gelu_f16(
            __global const float* a,
            __global const ushort* b,
            __global float* c,
            __global const float* bias,
            const uint aRows,
            const uint aCols,
            const uint bCols)
        {
            GEMM_ROW4_ACCUMULATE_F16
            if (outRow < aRows)
            {
                float b0 = c0 < bCols ? bias[c0] : 0.0f;
                float b1 = c1 < bCols ? bias[c1] : 0.0f;
                float b2 = c2 < bCols ? bias[c2] : 0.0f;
                float b3 = c3 < bCols ? bias[c3] : 0.0f;
                uint rowBase = outRow * bCols + colBase + y * BLOCK;
                if (c0 < bCols) c[rowBase + 0] = gelu_exact(a0 + b0);
                if (c1 < bCols) c[rowBase + 1] = gelu_exact(a1 + b1);
                if (c2 < bCols) c[rowBase + 2] = gelu_exact(a2 + b2);
                if (c3 < bCols) c[rowBase + 3] = gelu_exact(a3 + b3);
            }
        }


        __kernel void gemm_row4_gelu_bf16(
            __global const float* a,
            __global const ushort* b,
            __global float* c,
            __global const float* bias,
            const uint aRows,
            const uint aCols,
            const uint bCols)
        {
            GEMM_ROW4_ACCUMULATE_BF16
            if (outRow < aRows)
            {
                float b0 = c0 < bCols ? bias[c0] : 0.0f;
                float b1 = c1 < bCols ? bias[c1] : 0.0f;
                float b2 = c2 < bCols ? bias[c2] : 0.0f;
                float b3 = c3 < bCols ? bias[c3] : 0.0f;
                uint rowBase = outRow * bCols + colBase + y * BLOCK;
                if (c0 < bCols) c[rowBase + 0] = gelu_exact(a0 + b0);
                if (c1 < bCols) c[rowBase + 1] = gelu_exact(a1 + b1);
                if (c2 < bCols) c[rowBase + 2] = gelu_exact(a2 + b2);
                if (c3 < bCols) c[rowBase + 3] = gelu_exact(a3 + b3);
            }
        }


        /* As gemm_row4_bias with a fused ReLU epilogue — the DistilBERT classifier head's
           pre_classifier projection. The clamp is written as the same conditional ILGPU's Row4
           ReLU kernel uses (r < 0 ? 0 : r), not fmax(), so a NaN-bearing input would round the
           same way on both backends; the epilogue is otherwise a pure add, so this stays
           byte-identical to ILGPU. */
        __kernel void gemm_row4_relu(
            __global const float* a,
            __global const float* b,
            __global float* c,
            __global const float* bias,
            const uint aRows,
            const uint aCols,
            const uint bCols)
        {
            GEMM_ROW4_ACCUMULATE
            if (outRow < aRows)
            {
                float b0 = c0 < bCols ? bias[c0] : 0.0f;
                float b1 = c1 < bCols ? bias[c1] : 0.0f;
                float b2 = c2 < bCols ? bias[c2] : 0.0f;
                float b3 = c3 < bCols ? bias[c3] : 0.0f;
                float r0 = a0 + b0; if (r0 < 0.0f) r0 = 0.0f;
                float r1 = a1 + b1; if (r1 < 0.0f) r1 = 0.0f;
                float r2 = a2 + b2; if (r2 < 0.0f) r2 = 0.0f;
                float r3 = a3 + b3; if (r3 < 0.0f) r3 = 0.0f;
                uint rowBase = outRow * bCols + colBase + y * BLOCK;
                if (c0 < bCols) c[rowBase + 0] = r0;
                if (c1 < bCols) c[rowBase + 1] = r1;
                if (c2 < bCols) c[rowBase + 2] = r2;
                if (c3 < bCols) c[rowBase + 3] = r3;
            }
        }

        __kernel void gemm_row4_relu_f16(
            __global const float* a,
            __global const ushort* b,
            __global float* c,
            __global const float* bias,
            const uint aRows,
            const uint aCols,
            const uint bCols)
        {
            GEMM_ROW4_ACCUMULATE_F16
            if (outRow < aRows)
            {
                float b0 = c0 < bCols ? bias[c0] : 0.0f;
                float b1 = c1 < bCols ? bias[c1] : 0.0f;
                float b2 = c2 < bCols ? bias[c2] : 0.0f;
                float b3 = c3 < bCols ? bias[c3] : 0.0f;
                float r0 = a0 + b0; if (r0 < 0.0f) r0 = 0.0f;
                float r1 = a1 + b1; if (r1 < 0.0f) r1 = 0.0f;
                float r2 = a2 + b2; if (r2 < 0.0f) r2 = 0.0f;
                float r3 = a3 + b3; if (r3 < 0.0f) r3 = 0.0f;
                uint rowBase = outRow * bCols + colBase + y * BLOCK;
                if (c0 < bCols) c[rowBase + 0] = r0;
                if (c1 < bCols) c[rowBase + 1] = r1;
                if (c2 < bCols) c[rowBase + 2] = r2;
                if (c3 < bCols) c[rowBase + 3] = r3;
            }
        }


        __kernel void gemm_row4_relu_bf16(
            __global const float* a,
            __global const ushort* b,
            __global float* c,
            __global const float* bias,
            const uint aRows,
            const uint aCols,
            const uint bCols)
        {
            GEMM_ROW4_ACCUMULATE_BF16
            if (outRow < aRows)
            {
                float b0 = c0 < bCols ? bias[c0] : 0.0f;
                float b1 = c1 < bCols ? bias[c1] : 0.0f;
                float b2 = c2 < bCols ? bias[c2] : 0.0f;
                float b3 = c3 < bCols ? bias[c3] : 0.0f;
                float r0 = a0 + b0; if (r0 < 0.0f) r0 = 0.0f;
                float r1 = a1 + b1; if (r1 < 0.0f) r1 = 0.0f;
                float r2 = a2 + b2; if (r2 < 0.0f) r2 = 0.0f;
                float r3 = a3 + b3; if (r3 < 0.0f) r3 = 0.0f;
                uint rowBase = outRow * bCols + colBase + y * BLOCK;
                if (c0 < bCols) c[rowBase + 0] = r0;
                if (c1 < bCols) c[rowBase + 1] = r1;
                if (c2 < bCols) c[rowBase + 2] = r2;
                if (c3 < bCols) c[rowBase + 3] = r3;
            }
        }


        __kernel void gemm_row4_qkv(
            __global const float* a,
            __global const float* b,
            __global float* c,
            __global const float* bias,
            const uint aRows,
            const uint aCols,
            const uint bCols,
            const uint blockWidth)
        {
            GEMM_ROW4_ACCUMULATE
            if (outRow < aRows)
            {
                float b0 = c0 < bCols ? bias[c0] : 0.0f;
                float b1 = c1 < bCols ? bias[c1] : 0.0f;
                float b2 = c2 < bCols ? bias[c2] : 0.0f;
                float b3 = c3 < bCols ? bias[c3] : 0.0f;
                if (c0 < bCols) c[qkv_dest(c0, outRow, aRows, blockWidth)] = a0 + b0;
                if (c1 < bCols) c[qkv_dest(c1, outRow, aRows, blockWidth)] = a1 + b1;
                if (c2 < bCols) c[qkv_dest(c2, outRow, aRows, blockWidth)] = a2 + b2;
                if (c3 < bCols) c[qkv_dest(c3, outRow, aRows, blockWidth)] = a3 + b3;
            }
        }

        inline bool attn_keep(__global const float* mask, uint maskBase, int qPos, int j, int band)
        {
            return mask[maskBase + (uint)j] >= 0.5f && (band < 0 || abs(qPos - j) <= band);
        }

        inline float attn_score(
            __global const float* q,
            __global const float* k,
            uint qRow,
            uint jRow,
            uint dOffset,
            uint D,
            uint headDim,
            float scale)
        {
            uint qBase = qRow * D + dOffset;
            uint kBase = jRow * D + dOffset;
            float acc = 0.0f;
            for (uint d = 0; d < headDim; ++d)
                acc += q[qBase + d] * k[kBase + d];
            return acc * scale;
        }

        __kernel void gemm_row4_qkv_f16(
            __global const float* a,
            __global const ushort* b,
            __global float* c,
            __global const float* bias,
            const uint aRows,
            const uint aCols,
            const uint bCols,
            const uint blockWidth)
        {
            GEMM_ROW4_ACCUMULATE_F16
            if (outRow < aRows)
            {
                float b0 = c0 < bCols ? bias[c0] : 0.0f;
                float b1 = c1 < bCols ? bias[c1] : 0.0f;
                float b2 = c2 < bCols ? bias[c2] : 0.0f;
                float b3 = c3 < bCols ? bias[c3] : 0.0f;
                if (c0 < bCols) c[qkv_dest(c0, outRow, aRows, blockWidth)] = a0 + b0;
                if (c1 < bCols) c[qkv_dest(c1, outRow, aRows, blockWidth)] = a1 + b1;
                if (c2 < bCols) c[qkv_dest(c2, outRow, aRows, blockWidth)] = a2 + b2;
                if (c3 < bCols) c[qkv_dest(c3, outRow, aRows, blockWidth)] = a3 + b3;
            }
        }

        __kernel void gemm_row4_qkv_bf16(
            __global const float* a,
            __global const ushort* b,
            __global float* c,
            __global const float* bias,
            const uint aRows,
            const uint aCols,
            const uint bCols,
            const uint blockWidth)
        {
            GEMM_ROW4_ACCUMULATE_BF16
            if (outRow < aRows)
            {
                float b0 = c0 < bCols ? bias[c0] : 0.0f;
                float b1 = c1 < bCols ? bias[c1] : 0.0f;
                float b2 = c2 < bCols ? bias[c2] : 0.0f;
                float b3 = c3 < bCols ? bias[c3] : 0.0f;
                if (c0 < bCols) c[qkv_dest(c0, outRow, aRows, blockWidth)] = a0 + b0;
                if (c1 < bCols) c[qkv_dest(c1, outRow, aRows, blockWidth)] = a1 + b1;
                if (c2 < bCols) c[qkv_dest(c2, outRow, aRows, blockWidth)] = a2 + b2;
                if (c3 < bCols) c[qkv_dest(c3, outRow, aRows, blockWidth)] = a3 + b3;
            }
        }


        __kernel void batched_attention(
            __global const float* q,
            __global const float* k,
            __global const float* v,
            __global const float* mask,
            __global float* attnOut,
            const uint batch,
            const uint seqLen,
            const uint numHeads,
            const uint headDim,
            const int band,
            const float scale)
        {
            __local float acc[MAXHEADDIM * ATTNGROUP];
            uint total = batch * numHeads * seqLen;
            uint idx = get_global_id(0);
            if (idx >= total) return;

            uint b = idx / (numHeads * seqLen);
            uint rem = idx - b * (numHeads * seqLen);
            uint h = rem / seqLen;
            uint qPos = rem - h * seqLen;

            uint D = numHeads * headDim;
            uint dOffset = h * headDim;
            uint qRow = b * seqLen + qPos;
            uint maskBase = b * seqLen;
            uint outBase = qRow * D;

            int qp = (int)qPos;
            int sl = (int)seqLen;
            int jFirst = band < 0 ? 0 : max(0, qp - band);
            int jLast = band < 0 ? sl - 1 : min(sl - 1, qp + band);

            // seqLen <= 512 covers the attention shapes this backend is benchmarked on; higher values keep
            // the original recompute path to avoid fixed-size local bounds.
            float scoreCache[512];
            int cacheReady = sl <= 512 ? 1 : 0;
            int range = jLast - jFirst + 1;

            float mx = -INFINITY;
            if (cacheReady)
            {
                for (int j = jFirst; j <= jLast; ++j)
                {
                    float s = attn_score(q, k, qRow, b * seqLen + (uint)j, dOffset, D, headDim, scale);
                    if (!attn_keep(mask, maskBase, qp, j, band)) s = -INFINITY;
                    scoreCache[j - jFirst] = s;
                    if (s > mx) mx = s;
                }
            }
            else
            {
                for (int j = jFirst; j <= jLast; ++j)
                {
                    float s = attn_score(q, k, qRow, b * seqLen + (uint)j, dOffset, D, headDim, scale);
                    if (!attn_keep(mask, maskBase, qp, j, band)) s = -INFINITY;
                    if (s > mx) mx = s;
                }
            }

            if (mx == -INFINITY)
            {
                for (uint d = 0; d < headDim; ++d)
                    attnOut[outBase + dOffset + d] = 0.0f;
                return;
            }

            float sum = 0.0f;
            if (cacheReady)
            {
                for (int j = 0; j < range; ++j)
                    sum += exp(scoreCache[j] - mx);
            }
            else
            {
                for (int j = jFirst; j <= jLast; ++j)
                {
                    float s = attn_score(q, k, qRow, b * seqLen + (uint)j, dOffset, D, headDim, scale);
                    if (!attn_keep(mask, maskBase, qp, j, band)) s = -INFINITY;
                    sum += exp(s - mx);
                }
            }

            uint lane = get_local_id(0);
            for (uint d = 0; d < headDim; ++d)
                acc[d * MAXHEADDIM + lane] = 0.0f;

            if (cacheReady)
            {
                for (int j = 0; j < range; ++j)
                {
                    float s = scoreCache[j];
                    float p = exp(s - mx) / sum;
                    int jGlobal = jFirst + j;
                    uint vBase = (b * seqLen + (uint)jGlobal) * D + dOffset;
                    for (uint d = 0; d < headDim; ++d)
                        acc[d * MAXHEADDIM + lane] += p * v[vBase + d];
                }
            }
            else
            {
                for (int j = jFirst; j <= jLast; ++j)
                {
                    float s = attn_score(q, k, qRow, b * seqLen + (uint)j, dOffset, D, headDim, scale);
                    if (!attn_keep(mask, maskBase, qp, j, band)) s = -INFINITY;
                    float p = exp(s - mx) / sum;
                    uint vBase = (b * seqLen + (uint)j) * D + dOffset;
                    for (uint d = 0; d < headDim; ++d)
                        acc[d * MAXHEADDIM + lane] += p * v[vBase + d];
                }
            }

            for (uint d = 0; d < headDim; ++d)
                attnOut[outBase + dOffset + d] = acc[d * MAXHEADDIM + lane];
        }

        __kernel void batched_attention_sliding(
            __global const float* q,
            __global const float* k,
            __global const float* v,
            __global const float* mask,
            __global float* attnOut,
            const uint batch,
            const uint seqLen,
            const uint numHeads,
            const uint headDim,
            const int band,
            const float scale)
        {
            __local float acc[MAXHEADDIM * ATTNGROUP];
            uint total = batch * numHeads * seqLen;
            uint idx = get_global_id(0);
            if (idx >= total) return;

            uint b = idx / (numHeads * seqLen);
            uint rem = idx - b * (numHeads * seqLen);
            uint h = rem / seqLen;
            uint qPos = rem - h * seqLen;

            uint D = numHeads * headDim;
            uint dOffset = h * headDim;
            uint qRow = b * seqLen + qPos;
            uint maskBase = b * seqLen;
            uint outBase = qRow * D;

            int qp = (int)qPos;
            int sl = (int)seqLen;
            int jFirst = band < 0 ? 0 : max(0, qp - band);
            int jLast = band < 0 ? sl - 1 : min(sl - 1, qp + band);

            // Sliding-window path: the attended span is at most 2*band+1 (<=257), so cache only that.
            // Out-of-range bands fall back to the global kernel, which recomputes above 512 keys.
            float scoreCache[257];
            int range = jLast - jFirst + 1;
            int cacheReady = range <= 257 ? 1 : 0;

            float mx = -INFINITY;
            if (cacheReady)
            {
                for (int j = jFirst; j <= jLast; ++j)
                {
                    float s = attn_score(q, k, qRow, b * seqLen + (uint)j, dOffset, D, headDim, scale);
                    if (!attn_keep(mask, maskBase, qp, j, band)) s = -INFINITY;
                    scoreCache[j - jFirst] = s;
                    if (s > mx) mx = s;
                }
            }
            else
            {
                for (int j = jFirst; j <= jLast; ++j)
                {
                    float s = attn_score(q, k, qRow, b * seqLen + (uint)j, dOffset, D, headDim, scale);
                    if (!attn_keep(mask, maskBase, qp, j, band)) s = -INFINITY;
                    if (s > mx) mx = s;
                }
            }

            if (mx == -INFINITY)
            {
                for (uint d = 0; d < headDim; ++d)
                    attnOut[outBase + dOffset + d] = 0.0f;
                return;
            }

            float sum = 0.0f;
            if (cacheReady)
            {
                for (int j = 0; j < range; ++j)
                    sum += exp(scoreCache[j] - mx);
            }
            else
            {
                for (int j = jFirst; j <= jLast; ++j)
                {
                    float s = attn_score(q, k, qRow, b * seqLen + (uint)j, dOffset, D, headDim, scale);
                    if (!attn_keep(mask, maskBase, qp, j, band)) s = -INFINITY;
                    sum += exp(s - mx);
                }
            }

            uint lane = get_local_id(0);
            for (uint d = 0; d < headDim; ++d)
                acc[d * MAXHEADDIM + lane] = 0.0f;

            if (cacheReady)
            {
                for (int j = 0; j < range; ++j)
                {
                    float s = scoreCache[j];
                    float p = exp(s - mx) / sum;
                    int jGlobal = jFirst + j;
                    uint vBase = (b * seqLen + (uint)jGlobal) * D + dOffset;
                    for (uint d = 0; d < headDim; ++d)
                        acc[d * MAXHEADDIM + lane] += p * v[vBase + d];
                }
            }
            else
            {
                for (int j = jFirst; j <= jLast; ++j)
                {
                    float s = attn_score(q, k, qRow, b * seqLen + (uint)j, dOffset, D, headDim, scale);
                    if (!attn_keep(mask, maskBase, qp, j, band)) s = -INFINITY;
                    float p = exp(s - mx) / sum;
                    uint vBase = (b * seqLen + (uint)j) * D + dOffset;
                    for (uint d = 0; d < headDim; ++d)
                        acc[d * MAXHEADDIM + lane] += p * v[vBase + d];
                }
            }

            for (uint d = 0; d < headDim; ++d)
                attnOut[outBase + dOffset + d] = acc[d * MAXHEADDIM + lane];
        }


        /* RMSNorm: no mean subtraction, affine gamma only. One work item per row. y may alias x. */
        __kernel void rms_norm(
            __global const float* x,
            __global const float* gamma,
            __global float* y,
            const uint rows,
            const uint cols,
            const float eps)
        {
            uint r = get_global_id(0);
            if (r >= rows) return;
            uint off = r * cols;
            float sumSq = 0.0f;
            for (uint c = 0; c < cols; ++c)
            {
                float v = x[off + c];
                sumSq += v * v;
            }
            float rms = sqrt(sumSq / (float)cols + eps);
            float inv = 1.0f / rms;
            for (uint c = 0; c < cols; ++c)
                y[off + c] = x[off + c] * inv * gamma[c];
        }

        /* Elementwise SiLU. */
        __kernel void silu(
            __global const float* x,
            __global float* y,
            const uint n)
        {
            uint idx = get_global_id(0);
            if (idx >= n) return;
            float v = x[idx];
            y[idx] = v / (1.0f + exp(-v));
        }

        /* Fused SwiGLU: y = silu(gate) * up. */
        __kernel void swiglu(
            __global const float* gate,
            __global const float* up,
            __global float* y,
            const uint rows,
            const uint intermediate)
        {
            uint idx = get_global_id(0);
            if (idx >= rows * intermediate) return;
            float g = gate[idx];
            float siluGate = g / (1.0f + exp(-g));
            y[idx] = siluGate * up[idx];
        }

        /* Causal GQA attention. q: [seq, numHeads*headDim], k/v: [seq, numKvHeads*headDim].
           One work item per (qPos, head). Attends j=0..qPos with KV-repeat. */
        __kernel void causal_gqa_attention(
            __global const float* q,
            __global const float* k,
            __global const float* v,
            __global float* out,
            const uint seqLen,
            const uint numHeads,
            const uint numKvHeads,
            const uint headDim,
            const float scale)
        {
            uint idx = get_global_id(0);
            uint total = seqLen * numHeads;
            if (idx >= total) return;

            uint qPos = idx / numHeads;
            uint h = idx % numHeads;
            uint repeat = numHeads / numKvHeads;
            uint kvHead = h / repeat;

            uint Dq = numHeads * headDim;
            uint Dkv = numKvHeads * headDim;
            uint qBase = qPos * Dq + h * headDim;
            uint kBaseHead = kvHead * headDim;
            uint vBaseHead = kvHead * headDim;

            // Fused online-softmax attention: each q*k is computed once; running max
            // (mx) and unnormalised accumulators (sum, accv[]) are rescaled in place.
            float accv[64];
            for (uint d = 0; d < headDim; ++d)
                accv[d] = 0.0f;

            float mx = -INFINITY;
            float sum = 0.0f;
            for (uint j = 0; j <= qPos; ++j)
            {
                float acc = 0.0f;
                uint kBase = j * Dkv + kBaseHead;
                for (uint d = 0; d < headDim; ++d)
                    acc += q[qBase + d] * k[kBase + d];
                float s = acc * scale;

                if (s > mx)
                {
                    if (mx != -INFINITY)
                    {
                        float corr = exp(mx - s);
                        sum *= corr;
                        for (uint d = 0; d < headDim; ++d)
                            accv[d] *= corr;
                    }
                    mx = s;
                }

                float w = exp(s - mx);
                sum += w;
                uint vBase = j * Dkv + vBaseHead;
                for (uint d = 0; d < headDim; ++d)
                    accv[d] += w * v[vBase + d];
            }

            if (mx == -INFINITY)
            {
                for (uint d = 0; d < headDim; ++d)
                    out[qBase + d] = 0.0f;
                return;
            }

            float invSum = 1.0f / sum;
            for (uint d = 0; d < headDim; ++d)
                out[qBase + d] = accv[d] * invSum;
        }

        __kernel void conv2d(
            __global const float* input,
            __global const float* weight,
            __global const float* bias,
            __global float* output,
            const uint N, const uint C, const uint H, const uint W,
            const uint outC, const uint kH, const uint kW,
            const uint strideH, const uint strideW,
            const uint padTop, const uint padBottom, const uint padLeft, const uint padRight,
            const uint groups, const uint hasBias,
            const uint oH, const uint oW)
        {
            uint idx = get_global_id(0);
            uint total = N * outC * oH * oW;
            if (idx >= total) return;
            uint ow = idx % oW;
            uint tmp = idx / oW;
            uint oh = tmp % oH;
            tmp = tmp / oH;
            uint oc = tmp % outC;
            uint n = tmp / outC;

            uint inC = C;
            uint inCperGroup = inC / groups;
            uint outCperGroup = outC / groups;
            uint g = oc / outCperGroup;

            float acc = 0.0f;
            for (uint icg = 0; icg < inCperGroup; ++icg)
            {
                uint ic = g * inCperGroup + icg;
                for (uint kh = 0; kh < kH; ++kh)
                {
                    int ih = (int)(oh * strideH + kh) - (int)padTop;
                    if (ih < 0 || ih >= (int)H) continue;
                    for (uint kw = 0; kw < kW; ++kw)
                    {
                        int iw = (int)(ow * strideW + kw) - (int)padLeft;
                        if (iw < 0 || iw >= (int)W) continue;
                        uint inIdx = ((n * inC + ic) * H + (uint)ih) * W + (uint)iw;
                        uint wIdx = ((oc * inCperGroup + icg) * kH + kh) * kW + kw;
                        acc += input[inIdx] * weight[wIdx];
                    }
                }
            }
            if (hasBias != 0u)
                acc += bias[oc];
            uint outIdx = ((n * outC + oc) * oH + oh) * oW + ow;
            output[outIdx] = acc;
        }

        __kernel void batchnorm_eval(
            __global const float* x,
            __global const float* mean,
            __global const float* var,
            __global const float* gamma,
            __global const float* beta,
            __global float* y,
            const uint N, const uint C, const uint HW,
            const float eps, const uint hasAffine)
        {
            uint idx = get_global_id(0);
            uint total = N * C * HW;
            if (idx >= total) return;
            uint c = (idx / HW) % C;
            float invStd = 1.0f / sqrt(var[c] + eps);
            float g = (hasAffine != 0u) ? gamma[c] : 1.0f;
            float b = (hasAffine != 0u) ? beta[c] : 0.0f;
            y[idx] = (x[idx] - mean[c]) * invStd * g + b;
        }

        __kernel void relu(__global const float* x, __global float* y, const uint n)
        {
            uint idx = get_global_id(0);
            if (idx >= n) return;
            float v = x[idx];
            y[idx] = v > 0.0f ? v : 0.0f;
        }

        __kernel void relu6(__global const float* x, __global float* y, const uint n)
        {
            uint idx = get_global_id(0);
            if (idx >= n) return;
            float v = x[idx];
            if (v < 0.0f) v = 0.0f;
            if (v > 6.0f) v = 6.0f;
            y[idx] = v;
        }

        __kernel void maxpool2d(
            __global const float* input,
            __global float* output,
            const uint N, const uint C, const uint H, const uint W,
            const uint kH, const uint kW,
            const uint strideH, const uint strideW,
            const uint padTop, const uint padBottom, const uint padLeft, const uint padRight,
            const uint oH, const uint oW)
        {
            uint idx = get_global_id(0);
            uint total = N * C * oH * oW;
            if (idx >= total) return;
            uint ow = idx % oW;
            uint tmp = idx / oW;
            uint oh = tmp % oH;
            tmp = tmp / oH;
            uint c = tmp % C;
            uint n = tmp / C;

            float mx = -INFINITY;
            for (uint kh = 0; kh < kH; ++kh)
            {
                int ih = (int)(oh * strideH + kh) - (int)padTop;
                if (ih < 0 || ih >= (int)H) continue;
                for (uint kw = 0; kw < kW; ++kw)
                {
                    int iw = (int)(ow * strideW + kw) - (int)padLeft;
                    if (iw < 0 || iw >= (int)W) continue;
                    uint inIdx = ((n * C + c) * H + (uint)ih) * W + (uint)iw;
                    float v = input[inIdx];
                    if (v > mx) mx = v;
                }
            }
            if (mx == -INFINITY) mx = 0.0f;
            output[idx] = mx;
        }

        __kernel void adaptive_avgpool2d(
            __global const float* input,
            __global float* output,
            const uint N, const uint C, const uint H, const uint W)
        {
            uint idx = get_global_id(0);
            uint total = N * C;
            if (idx >= total) return;
            uint c = idx % C;
            uint n = idx / C;
            float sum = 0.0f;
            for (uint h = 0; h < H; ++h)
                for (uint w = 0; w < W; ++w)
                {
                    uint inIdx = ((n * C + c) * H + h) * W + w;
                    sum += input[inIdx];
                }
            output[idx] = sum / (float)(H * W);
        }

        /* ---- Text-classifier backward (#900). The GEMM-shaped pieces are the NT/TN row4
           variants above; the rest are the small elementwise/reduction kernels the two
           classifier training loops need. All reductions accumulate in one ascending order so
           a host replay of the same order is byte-identical (file-scope FP_CONTRACT OFF). ---- */

        __kernel void gemm_row4_nt(
            __global const float* a,
            __global const float* b,
            __global float* c,
            const uint aRows,
            const uint aCols,
            const uint bCols)
        {
            GEMM_NT_ACCUMULATE
            if (outRow < aRows)
            {
                uint rowBase = outRow * bCols + colBase + y * BLOCK;
                if (c0 < bCols) c[rowBase + 0] = a0;
                if (c1 < bCols) c[rowBase + 1] = a1;
                if (c2 < bCols) c[rowBase + 2] = a2;
                if (c3 < bCols) c[rowBase + 3] = a3;
            }
        }

        __kernel void gemm_row4_tn(
            __global const float* a,
            __global const float* b,
            __global float* c,
            const uint aRows,
            const uint aCols,
            const uint bCols)
        {
            GEMM_TN_ACCUMULATE
            if (outRow < aRows)
            {
                uint rowBase = outRow * bCols + colBase + y * BLOCK;
                if (c0 < bCols) c[rowBase + 0] = a0;
                if (c1 < bCols) c[rowBase + 1] = a1;
                if (c2 < bCols) c[rowBase + 2] = a2;
                if (c3 < bCols) c[rowBase + 3] = a3;
            }
        }

        /* gemm_row4_nt with residual epilogue: like gemm_row4_nt but each output cell adds
           the matching residual value before storing. Exact under FP_CONTRACT OFF. */
        __kernel void gemm_row4_nt_residual(
            __global const float* a,
            __global const float* b,
            __global float* c,
            __global const float* residual,
            const uint aRows,
            const uint aCols,
            const uint bCols)
        {
            GEMM_NT_ACCUMULATE
            if (outRow < aRows)
            {
                uint rowBase = outRow * bCols + colBase + y * BLOCK;
                if (c0 < bCols) { float r = residual[rowBase + 0]; c[rowBase + 0] = a0 + r; }
                if (c1 < bCols) { float r = residual[rowBase + 1]; c[rowBase + 1] = a1 + r; }
                if (c2 < bCols) { float r = residual[rowBase + 2]; c[rowBase + 2] = a2 + r; }
                if (c3 < bCols) { float r = residual[rowBase + 3]; c[rowBase + 3] = a3 + r; }
            }
        }

        /* d flows back only where the forward activation was positive; act is the POST-activation
           buffer (ReLU(x) > 0 iff x > 0), which both classifiers already keep device-resident. */
        __kernel void relu_backward(
            __global const float* d,
            __global const float* act,
            __global float* y,
            const uint n)
        {
            uint idx = get_global_id(0);
            if (idx >= n) return;
            y[idx] = act[idx] > 0.0f ? d[idx] : 0.0f;
        }

        /* ---------------------------------------------------------------------------------------
           batched_attention backward (#956). The forward keeps no softmax, so the backward
           re-derives it in three deterministic launches, each with one work item per output
           element and an ascending inner loop (the property the CPU-replay test pins):

             prep  — per (b, h, query): p_ij and delta_i = sum_j p_ij (dO_i . V_j)
             dq    — per (b, h, query): dQ_i[d] = scale * sum_j p_ij (dotV_ij - delta_i) K_j[d]
             dkv   — per (b, h, key):   dV_j = sum_i p_ij dO_i
                                        dK_j = scale * sum_i p_ij (dotV_ij - delta_i) Q_i

           qkv is gemm_row4_qkv's BLOCK-MAJOR output — [Q (rows x D)][K][V], the same layout the
           forward attention consumes through three sub-buffer views — so a READ base is
           <block offset> + row * D + dOffset. dQkv is a GRADIENT buffer with no reader but the
           qkv backward GEMMs, so it is written row-major [rows, 3*D] in q|k|v column order: the
           layout WeightLayout.QkvPacked gives the weight, which lets dW/dB/dX reuse the plain
           GemmTn/GemmNt/col_sum path with no unpacking pass.

           probs is [batch * heads * seqLen, seqLen] and delta is [batch * heads * seqLen]. A
           fully masked query row writes zeros rather than evaluating exp(-inf - -inf).

           dS_ij is the gradient with respect to the SCALED score (q.k * scale is the forward's
           contraction), so the scale multiplies the K and Q terms below, not p. */

        inline float attn_backward_score(
            __global const float* qkv, uint qBase, uint kBase, uint headDim, float scale)
        {
            float acc = 0.0f;
            for (uint d = 0; d < headDim; ++d)
                acc += qkv[qBase + d] * qkv[kBase + d];
            return acc * scale;
        }

        /* dO_i . V_j — the one contraction every attention-gradient kernel needs and none can
           cache. Kept ahead of its callers because OpenCL C has no hoisted declarations. */
        inline float attn_backward_dotv(
            __global const float* qkv,
            __global const float* dAttn,
            uint vBase,
            uint dAttnBase,
            uint headDim)
        {
            float acc = 0.0f;
            for (uint d = 0; d < headDim; ++d)
                acc += dAttn[dAttnBase + d] * qkv[vBase + d];
            return acc;
        }

        __kernel void attention_backward_prep(
            __global const float* qkv,
            __global const float* mask,
            __global const float* dAttn,
            __global float* probs,
            __global float* delta,
            const uint batch,
            const uint seqLen,
            const uint numHeads,
            const uint headDim,
            const int band,
            const float scale)
        {
            uint total = batch * numHeads * seqLen;
            uint idx = get_global_id(0);
            if (idx >= total) return;

            uint b = idx / (numHeads * seqLen);
            uint rem = idx - b * (numHeads * seqLen);
            uint h = rem / seqLen;
            uint qPos = rem - h * seqLen;

            uint D = numHeads * headDim;
            uint rows = batch * seqLen;
            uint kOff = rows * D;
            uint vOff = 2u * rows * D;
            uint dOffset = h * headDim;
            uint qRow = b * seqLen + qPos;
            uint maskBase = b * seqLen;
            uint probsBase = idx * seqLen;
            uint qBase = qRow * D + dOffset;
            uint dAttnBase = qRow * D + dOffset;

            float qrow[MAXHEADDIM];
            float drow[MAXHEADDIM];
            for (uint d = 0; d < headDim; ++d)
            {
                qrow[d] = qkv[qBase + d];
                drow[d] = dAttn[dAttnBase + d];
            }

            int qp = (int)qPos;
            int sl = (int)seqLen;

            float mx = -INFINITY;
            for (int j = 0; j < sl; ++j)
                if (attn_keep(mask, maskBase, qp, j, band))
                {
                    uint kBase = kOff + (b * seqLen + (uint)j) * D + dOffset;
                    prefetch(qkv + kBase, (size_t)headDim);
                    float score = 0.0f;
                    for (uint d = 0; d < headDim; ++d)
                        score += qrow[d] * qkv[kBase + d];
                    score *= scale;
                    probs[probsBase + (uint)j] = score;
                    mx = fmax(mx, score);
                }
                else
                    probs[probsBase + (uint)j] = -INFINITY;

            if (mx == -INFINITY)
            {
                for (uint j = 0; j < seqLen; ++j) probs[probsBase + j] = 0.0f;
                delta[idx] = 0.0f;
                return;
            }

            float sum = 0.0f;
            for (int j = 0; j < sl; ++j)
                if (probs[probsBase + (uint)j] != -INFINITY)
                    sum += exp(probs[probsBase + (uint)j] - mx);

            float acc = 0.0f;
            for (int j = 0; j < sl; ++j)
            {
                float p;
                if (probs[probsBase + (uint)j] != -INFINITY)
                    p = exp(probs[probsBase + (uint)j] - mx) / sum;
                else
                    p = 0.0f;
                probs[probsBase + (uint)j] = p;
                if (p != 0.0f)
                {
                    uint vBase = vOff + (b * seqLen + (uint)j) * D + dOffset;
                    prefetch(qkv + vBase, (size_t)headDim);
                    float dotv = 0.0f;
                    for (uint d = 0; d < headDim; ++d)
                        dotv += drow[d] * qkv[vBase + d];
                    acc += p * dotv;
                }
            }
            delta[idx] = acc;
        }

        __kernel void attention_backward_dq(
            __global const float* qkv,
            __global const float* probs,
            __global const float* delta,
            __global const float* dAttn,
            __global float* dQkv,
            const uint batch,
            const uint seqLen,
            const uint numHeads,
            const uint headDim,
            const float scale)
        {
            uint total = batch * numHeads * seqLen;
            uint idx = get_global_id(0);
            if (idx >= total) return;

            uint b = idx / (numHeads * seqLen);
            uint rem = idx - b * (numHeads * seqLen);
            uint h = rem / seqLen;
            uint qPos = rem - h * seqLen;

            uint D = numHeads * headDim;
            uint rows = batch * seqLen;
            uint kOff = rows * D;
            uint vOff = 2u * rows * D;
            uint dOffset = h * headDim;
            uint qRow = b * seqLen + qPos;
            uint probsBase = idx * seqLen;
            uint qOutBase = qRow * (3u * D) + dOffset;
            uint dAttnBase = qRow * D + dOffset;

            /* Ascending j per accumulated component: dS_ij is one scalar per (query, key), so it is
               derived once and fanned out to the headDim accumulators rather than recomputing the
               V contraction per component. */
            float dq[MAXHEADDIM];
            for (uint d = 0; d < headDim; ++d) dq[d] = 0.0f;

            float dl = delta[idx];
            for (uint j = 0; j < seqLen; ++j)
            {
                float ds = probs[probsBase + j]
                    * (attn_backward_dotv(qkv, dAttn, vOff + (b * seqLen + j) * D + dOffset, dAttnBase, headDim)
                        - dl);
                if (ds == 0.0f) continue;
                uint kBase = kOff + (b * seqLen + j) * D + dOffset;
                for (uint d = 0; d < headDim; ++d)
                    dq[d] += ds * qkv[kBase + d];
            }

            for (uint d = 0; d < headDim; ++d)
                dQkv[qOutBase + d] = dq[d] * scale;
        }

        __kernel void attention_backward_dkv(
            __global const float* qkv,
            __global const float* probs,
            __global const float* delta,
            __global const float* dAttn,
            __global float* dQkv,
            const uint batch,
            const uint seqLen,
            const uint numHeads,
            const uint headDim,
            const float scale)
        {
            uint total = batch * numHeads * seqLen;
            uint idx = get_global_id(0);
            if (idx >= total) return;

            uint b = idx / (numHeads * seqLen);
            uint rem = idx - b * (numHeads * seqLen);
            uint h = rem / seqLen;
            uint jPos = rem - h * seqLen;

            uint D = numHeads * headDim;
            uint dOffset = h * headDim;
            uint jRow = b * seqLen + jPos;
            uint rows = batch * seqLen;
            uint vOff = 2u * rows * D;
            uint kOutBase = jRow * (3u * D) + D + dOffset;
            uint vOutBase = jRow * (3u * D) + 2u * D + dOffset;

            float dk[MAXHEADDIM];
            float dv[MAXHEADDIM];
            for (uint d = 0; d < headDim; ++d) { dk[d] = 0.0f; dv[d] = 0.0f; }

            for (uint i = 0; i < seqLen; ++i)
            {
                uint rowIdx = (b * numHeads + h) * seqLen + i;
                uint dAttnBase = (b * seqLen + i) * D + dOffset;
                float p = probs[rowIdx * seqLen + jPos];
                if (p == 0.0f) continue;
                float ds = p * (attn_backward_dotv(qkv, dAttn, vOff + jRow * D + dOffset, dAttnBase, headDim) - delta[rowIdx]);
                uint qBase = (b * seqLen + i) * D + dOffset;
                for (uint d = 0; d < headDim; ++d)
                {
                    dv[d] += p * dAttn[dAttnBase + d];
                    dk[d] += ds * qkv[qBase + d];
                }
            }

            for (uint d = 0; d < headDim; ++d)
            {
                dQkv[kOutBase + d] = dk[d] * scale;
                dQkv[vOutBase + d] = dv[d];
            }
        }

        /* gelu_backward: d flows back through the GELU exact derivative. act is the PRE-activation
           buffer (the input to GELU), since the derivative depends on x, not on GELU(x). */
        __kernel void gelu_backward(
            __global const float* d,
            __global const float* act,
            __global float* y,
            const uint n)
        {
            uint idx = get_global_id(0);
            if (idx >= n) return;
            float v = act[idx];
            float invSqrt2 = 0.7071067811865475f;
            float invSqrt2Pi = 0.3989422804014327f;
            float cdf = 0.5f * (1.0f + erf(v * invSqrt2));
            float pdf = exp(-0.5f * v * v) * invSqrt2Pi;
            y[idx] = (cdf + v * pdf) * d[idx];
        }

        /* layer_norm_backward: backward through LayerNorm (no residual). Inputs: d (grad of
           normed output), x (pre-norm input), gamma. Outputs: dx (grad input), dgammaAcc and
           dbetaAcc (per-element products to be reduced with col_sum). eps matches forward. */
        __kernel void layer_norm_backward(
            __global const float* d,
            __global const float* x,
            __global const float* gamma,
            __global float* dx,
            __global float* dgammaAcc,
            __global float* dbetaAcc,
            const uint rows,
            const uint cols,
            const float eps)
        {
            uint idx = get_global_id(0);
            uint total = rows * cols;
            if (idx >= total) return;
            uint row = idx / cols;
            uint col = idx % cols;
            uint base = row * cols;

            float mean = 0.0f;
            for (uint c = 0; c < cols; ++c) mean += x[base + c];
            mean /= (float)cols;
            float var = 0.0f;
            for (uint c = 0; c < cols; ++c) {
                float diff = x[base + c] - mean;
                var += diff * diff;
            }
            var /= (float)cols;
            float invStd = 1.0f / sqrt(var + eps);

            float normalized = (x[idx] - mean) * invStd;
            dgammaAcc[idx] = d[idx] * normalized;
            dbetaAcc[idx] = d[idx];

            float meanD = 0.0f;
            float meanDNorm = 0.0f;
            for (uint c = 0; c < cols; ++c) {
                float nrm = (x[base + c] - mean) * invStd;
                meanD += d[base + c];
                meanDNorm += d[base + c] * nrm;
            }
            meanD /= (float)cols;
            meanDNorm /= (float)cols;

            dx[idx] = (gamma[col] * invStd) * (d[idx] - meanD - normalized * meanDNorm);
        }

        /* y[j] = scale * sum_i x[i*cols + j] — FC bias gradients. The scale is applied once at
           the end, so the host replay is: accumulate ascending, then multiply. */
        __kernel void col_sum(
            __global const float* x,
            __global float* y,
            const uint rows,
            const uint cols,
            const float scale)
        {
            uint col = get_global_id(0);
            if (col >= cols) return;
            float acc = 0.0f;
            for (uint r = 0; r < rows; ++r)
                acc += x[r * cols + col];
            y[col] = acc * scale;
        }

        /* y[i] = value — zeroing the dX workspace before the three branch launches of
           conv2d_backward_input accumulate into it. */
        __kernel void fill(
            __global float* y,
            const float value,
            const uint n)
        {
            uint idx = get_global_id(0);
            if (idx >= n) return;
            y[idx] = value;
        }

        /* Softmax cross-entropy input gradient: dZ[i,c] = (p[i,c] - onehot[i,c]) * scale,
           with the row maximum subtracted before exponentiation (matches SilkLoss.InputGradient
           and CrossEntropyLoss's mean reduction). One work item per sample row. */
        __kernel void softmax_cross_entropy_backward(
            __global const float* logits,
            __global const int* labels,
            __global float* dZ,
            const uint n,
            const uint numClasses,
            const float scale)
        {
            uint i = get_global_id(0);
            if (i >= n) return;
            uint base = i * numClasses;
            float mx = logits[base];
            for (uint c = 1u; c < numClasses; ++c)
                mx = fmax(mx, logits[base + c]);
            float sum = 0.0f;
            for (uint c = 0u; c < numClasses; ++c)
                sum += exp(logits[base + c] - mx);
            int label = labels[i];
            for (uint c = 0u; c < numClasses; ++c)
            {
                float p = exp(logits[base + c] - mx) / sum;
                float onehot = (c == (uint)label) ? 1.0f : 0.0f;
                dZ[base + c] = (p - onehot) * scale;
            }
        }

        /* dW[oc, ic, kh, kw] = sum over (n, oh, ow) of dA[n, oc, oh, ow] * X[n, ic, ih, iw],
           ih = oh*strideH - padTop + kh, iw = ow*strideW - padLeft + kw. Same NCHW layout,
           stride/pad/groups geometry and weight layout ((oc*inCperGroup + icg)*kH*kW + ...) as
           the forward conv2d. Not a GEMM — the reduction domain shifts per kernel position —
           so it is a direct kernel; one work-item per weight element. */
        __kernel void conv2d_backward_weight(
            __global const float* dA,
            __global const float* input,
            __global float* dW,
            const uint N, const uint C, const uint H, const uint W,
            const uint outC, const uint kH, const uint kW,
            const uint strideH, const uint strideW,
            const uint padTop, const uint padLeft,
            const uint groups, const uint oH, const uint oW)
        {
            uint inCperGroup = C / groups;
            uint outCperGroup = outC / groups;
            uint idx = get_global_id(0);
            uint total = outC * inCperGroup * kH * kW;
            if (idx >= total) return;
            uint kw2 = idx % kW;
            uint tmp = idx / kW;
            uint kh2 = tmp % kH;
            tmp = tmp / kH;
            uint icg = tmp % inCperGroup;
            tmp = tmp / inCperGroup;
            uint oc = tmp;
            uint g = oc / outCperGroup;
            uint ic = g * inCperGroup + icg;
            uint wIdx = ((oc * inCperGroup) + icg) * kH * kW + kh2 * kW + kw2;

            float acc = 0.0f;
            for (uint n = 0; n < N; ++n)
            {
                for (uint oh = 0; oh < oH; ++oh)
                {
                    int ih = (int)(oh * strideH) - (int)padTop + (int)kh2;
                    if (ih < 0 || ih >= (int)H) continue;
                    for (uint ow = 0; ow < oW; ++ow)
                    {
                        int iw = (int)(ow * strideW) - (int)padLeft + (int)kw2;
                        if (iw < 0 || iw >= (int)W) continue;
                        uint dIdx = ((n * outC + oc) * oH + oh) * oW + ow;
                        uint iIdx = ((n * C + ic) * H + (uint)ih) * W + (uint)iw;
                        acc += dA[dIdx] * input[iIdx];
                    }
                }
            }
            dW[wIdx] = acc;
        }

        /* dX[n, ic, ih, iw] = sum over (oc of ic's group, kh, kw) of dA[n, oc, oh, ow] *
           W[oc, icg, kh, kw], where oh is the output position that maps back to ih: oh*strideH
           - padTop + kh == ih, i.e. oh = (ih + padTop - kh) / strideH when it divides evenly.
           ACCUMULATES: the text CNN launches it once per branch, so the caller zeroes dX
           first (fill) and the three launches sum. */
        __kernel void conv2d_backward_input(
            __global const float* dA,
            __global const float* weight,
            __global float* dX,
            const uint N, const uint C, const uint H, const uint W,
            const uint outC, const uint kH, const uint kW,
            const uint strideH, const uint strideW,
            const uint padTop, const uint padLeft,
            const uint groups, const uint oH, const uint oW)
        {
            uint idx = get_global_id(0);
            uint total = N * C * H * W;
            if (idx >= total) return;
            uint iw = idx % W;
            uint tmp = idx / W;
            uint ih = tmp % H;
            tmp = tmp / H;
            uint ic = tmp % C;
            tmp = tmp / C;
            uint n = tmp;

            uint inCperGroup = C / groups;
            uint outCperGroup = outC / groups;
            uint g = ic / inCperGroup;
            uint icg = ic - g * inCperGroup;

            float acc = 0.0f;
            for (uint ocg = 0; ocg < outCperGroup; ++ocg)
            {
                uint oc = g * outCperGroup + ocg;
                for (uint kh = 0; kh < kH; ++kh)
                {
                    int ohRaw = (int)ih + (int)padTop - (int)kh;
                    if (ohRaw < 0 || ohRaw % strideH != 0) continue;
                    uint oh = (uint)ohRaw / strideH;
                    if (oh >= oH) continue;
                    for (uint kw = 0; kw < kW; ++kw)
                    {
                        int owRaw = (int)iw + (int)padLeft - (int)kw;
                        if (owRaw < 0 || owRaw % strideW != 0) continue;
                        uint ow = (uint)owRaw / strideW;
                        if (ow >= oW) continue;
                        uint dIdx = ((n * outC + oc) * oH + oh) * oW + ow;
                        uint wIdx = ((oc * inCperGroup) + icg) * kH * kW + kh * kW + kw;
                        acc += dA[dIdx] * weight[wIdx];
                    }
                }
            }
            dX[idx] += acc;
        }

        /* db[oc] = sum over (n, oh, ow) of dA[n, oc, oh, ow]. */
        __kernel void conv2d_backward_bias(
            __global const float* dA,
            __global float* db,
            const uint N, const uint outC, const uint oH, const uint oW)
        {
            uint oc = get_global_id(0);
            if (oc >= outC) return;
            float acc = 0.0f;
            for (uint n = 0; n < N; ++n)
                for (uint i = 0; i < oH * oW; ++i)
                    acc += dA[(n * outC + oc) * oH * oW + i];
            db[oc] = acc;
        }

        /* Fused Adam step (beta1 = 0.9, beta2 = 0.999, eps = 1e-8 — the constants the host
           AdamState uses). bc1/bc2 are the bias corrections 1-beta^t, precomputed host-side so
           the kernel stays elementwise. Gradients arrive already batch-averaged. */
        __kernel void adam_update(
            __global float* w,
            __global const float* g,
            __global float* m,
            __global float* v,
            const float lr,
            const float bc1,
            const float bc2,
            const float eps,
            const float weightDecay,
            const uint n)
        {
            uint i = get_global_id(0);
            if (i >= n) return;
            float grad = g[i];
            float m1 = 0.9f * m[i] + 0.1f * grad;
            float v1 = 0.999f * v[i] + 0.001f * grad * grad;
            m[i] = m1;
            v[i] = v1;
            /* The zero-decay branch repeats the pre-#956 expression verbatim: an unconditional
               `update + 0 * w` would reassociate the lr multiply and move every classifier weight
               by an ulp. */
            if (weightDecay != 0.0f)
                w[i] -= lr * ((m1 / bc1) / (sqrt(v1 / bc2) + eps) + weightDecay * w[i]);
            else
                w[i] -= lr * (m1 / bc1) / (sqrt(v1 / bc2) + eps);
        }

        /* Decode (single-query) GQA attention. q: [numHeads*headDim], kCache/vCache: [kvLen, numKvHeads*headDim].
           One work-group per query head; kvLen is split across the group's lanes and the per-lane
           online-softmax partials (mx, sum, accv[]) are merged with the associative combination, so
           the device is no longer limited to one thread per head. */
        __kernel void decode_gqa_attention(
            __global const float* q,
            __global const float* kCache,
            __global const float* vCache,
            __global float* out,
            const uint kvLen,
            const uint numHeads,
            const uint numKvHeads,
            const uint headDim,
            const float scale)
        {
            uint h = get_group_id(0);
            if (h >= numHeads) return;

            uint lid = get_local_id(0);
            uint gsize = get_local_size(0);

            uint repeat = numHeads / numKvHeads;
            uint kvHead = h / repeat;

            uint Dkv = numKvHeads * headDim;
            uint qBase = h * headDim;
            uint kBaseHead = kvHead * headDim;
            uint vBaseHead = kvHead * headDim;

            __local float lmx[ATTNGROUP];
            __local float lsum[ATTNGROUP];
            __local float laccv[ATTNGROUP * MAXHEADDIM];

            // Per-lane online-softmax partial (same ascending-j / ascending-d order as before).
            float mx = -INFINITY;
            float sum = 0.0f;
            float accv[MAXHEADDIM];
            for (uint d = 0; d < headDim; ++d)
                accv[d] = 0.0f;

            for (uint j = lid; j < kvLen; j += gsize)
            {
                float acc = 0.0f;
                uint kBase = j * Dkv + kBaseHead;
                for (uint d = 0; d < headDim; ++d)
                    acc += q[qBase + d] * kCache[kBase + d];
                float s = acc * scale;

                if (s > mx)
                {
                    if (mx != -INFINITY)
                    {
                        float corr = exp(mx - s);
                        sum *= corr;
                        for (uint d = 0; d < headDim; ++d)
                            accv[d] *= corr;
                    }
                    mx = s;
                }

                float w = exp(s - mx);
                sum += w;
                uint vBase = j * Dkv + vBaseHead;
                for (uint d = 0; d < headDim; ++d)
                    accv[d] += w * vCache[vBase + d];
            }

            lmx[lid] = mx;
            lsum[lid] = sum;
            for (uint d = 0; d < headDim; ++d)
                laccv[lid * MAXHEADDIM + d] = accv[d];
            barrier(CLK_LOCAL_MEM_FENCE);

            // Merge partials: associative online-softmax combination.
            for (uint stride = gsize / 2; stride > 0; stride >>= 1)
            {
                if (lid < stride)
                {
                    uint other = lid + stride;
                    float mx_a = lmx[lid];
                    float mx_b = lmx[other];
                    float mx_new = mx_a > mx_b ? mx_a : mx_b;

                    float corr_a = (mx_a == -INFINITY) ? 0.0f : exp(mx_a - mx_new);
                    float corr_b = (mx_b == -INFINITY) ? 0.0f : exp(mx_b - mx_new);

                    float sum_new = lsum[lid] * corr_a + lsum[other] * corr_b;
                    for (uint d = 0; d < headDim; ++d)
                        laccv[lid * MAXHEADDIM + d] =
                            laccv[lid * MAXHEADDIM + d] * corr_a +
                            laccv[other * MAXHEADDIM + d] * corr_b;

                    lmx[lid] = mx_new;
                    lsum[lid] = sum_new;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid == 0)
            {
                float mx_final = lmx[0];
                float sum_final = lsum[0];
                if (mx_final == -INFINITY || sum_final == 0.0f)
                {
                    for (uint d = 0; d < headDim; ++d)
                        out[qBase + d] = 0.0f;
                }
                else
                {
                    float invSum = 1.0f / sum_final;
                    for (uint d = 0; d < headDim; ++d)
                        out[qBase + d] = laccv[0 * MAXHEADDIM + d] * invSum;
                }
            }
        }

        /* Partial reduction for device-side argmax (#663). Each work-group reduces up to
           256 contiguous elements to one (index, value) pair; the host walks the partials. */
        __kernel void argmax(
            __global const float* input,
            __global int* partialIndex,
            __global float* partialValue,
            const uint n)
        {
            __local float lval[256];
            __local int lidx[256];
            uint lid = get_local_id(0);
            uint gid = get_global_id(0);
            uint lsize = get_local_size(0);

            float bestVal = -INFINITY;
            int bestIdx = -1;
            if (gid < n)
            {
                bestVal = input[gid];
                bestIdx = (int)gid;
            }

            lval[lid] = bestVal;
            lidx[lid] = bestIdx;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = lsize / 2; stride > 0; stride >>= 1)
            {
                if (lid < stride)
                {
                    if (lval[lid + stride] > lval[lid])
                    {
                        lval[lid] = lval[lid + stride];
                        lidx[lid] = lidx[lid + stride];
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid == 0)
            {
                uint gi = get_group_id(0);
                partialIndex[gi] = lidx[0];
                partialValue[gi] = lval[0];
            }
        }

        /* Fused RMSNorm + tied LM head: computes the normalized activation in-place
           into the Row4 GEMV against tokEmbTransposed, using the same RMSNorm
           reduction order as rms_norm and the same Row4 accumulation as gemm_row4. */
                __kernel void rms_norm_lm_head(
            __global const float* x,
            __global const float* gamma,
            __global const float* b,
            __global float* c,
            const uint aRows,
            const uint aCols,
            const uint bCols,
            const float eps)
        {
            uint lid0 = get_local_id(0);
            uint lid1 = get_local_id(1);
            uint outRow = get_global_id(0);
            uint colBase = get_group_id(1) * TILECOLS;

            float normScale = 0.0f;
            if (outRow < aRows)
            {
                uint rbase = outRow * aCols;
                float ssq = 0.0f;
                for (uint k = 0; k < aCols; ++k)
                {
                    float v = x[rbase + k];
                    ssq += v * v;
                }
                normScale = 1.0f / sqrt(ssq / (float)aCols + eps);
            }

            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            __local float smem[TILE * ROWSTRIDE];
            for (uint k0 = 0; k0 < aCols; k0 += TILE)
            {
                uint aCol = k0 + lid1;
                uint bRow = k0 + lid0;
                smem[lid0 + lid1 * TILE] =
                    (outRow < aRows && aCol < aCols)
                    ? (x[outRow * aCols + aCol] * normScale * gamma[aCol])
                    : 0.0f;
                for (uint w = 0; w < BLOCK; ++w)
                {
                    uint outCol = colBase + lid1 * BLOCK + w;
                    smem[lid0 + (TILE + lid1 * BLOCK + w) * TILE] =
                        (bRow < aCols && outCol < bCols)
                        ? b[bRow * bCols + outCol]
                        : 0.0f;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                { float av = smem[lid0 + 0 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[0 + (bBase + 0) * TILE]; a1 += av * smem[0 + (bBase + 1) * TILE]; a2 += av * smem[0 + (bBase + 2) * TILE]; a3 += av * smem[0 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 1 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[1 + (bBase + 0) * TILE]; a1 += av * smem[1 + (bBase + 1) * TILE]; a2 += av * smem[1 + (bBase + 2) * TILE]; a3 += av * smem[1 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 2 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[2 + (bBase + 0) * TILE]; a1 += av * smem[2 + (bBase + 1) * TILE]; a2 += av * smem[2 + (bBase + 2) * TILE]; a3 += av * smem[2 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 3 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[3 + (bBase + 0) * TILE]; a1 += av * smem[3 + (bBase + 1) * TILE]; a2 += av * smem[3 + (bBase + 2) * TILE]; a3 += av * smem[3 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 4 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[4 + (bBase + 0) * TILE]; a1 += av * smem[4 + (bBase + 1) * TILE]; a2 += av * smem[4 + (bBase + 2) * TILE]; a3 += av * smem[4 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 5 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[5 + (bBase + 0) * TILE]; a1 += av * smem[5 + (bBase + 1) * TILE]; a2 += av * smem[5 + (bBase + 2) * TILE]; a3 += av * smem[5 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 6 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[6 + (bBase + 0) * TILE]; a1 += av * smem[6 + (bBase + 1) * TILE]; a2 += av * smem[6 + (bBase + 2) * TILE]; a3 += av * smem[6 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 7 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[7 + (bBase + 0) * TILE]; a1 += av * smem[7 + (bBase + 1) * TILE]; a2 += av * smem[7 + (bBase + 2) * TILE]; a3 += av * smem[7 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 8 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[8 + (bBase + 0) * TILE]; a1 += av * smem[8 + (bBase + 1) * TILE]; a2 += av * smem[8 + (bBase + 2) * TILE]; a3 += av * smem[8 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 9 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[9 + (bBase + 0) * TILE]; a1 += av * smem[9 + (bBase + 1) * TILE]; a2 += av * smem[9 + (bBase + 2) * TILE]; a3 += av * smem[9 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 10 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[10 + (bBase + 0) * TILE]; a1 += av * smem[10 + (bBase + 1) * TILE]; a2 += av * smem[10 + (bBase + 2) * TILE]; a3 += av * smem[10 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 11 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[11 + (bBase + 0) * TILE]; a1 += av * smem[11 + (bBase + 1) * TILE]; a2 += av * smem[11 + (bBase + 2) * TILE]; a3 += av * smem[11 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 12 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[12 + (bBase + 0) * TILE]; a1 += av * smem[12 + (bBase + 1) * TILE]; a2 += av * smem[12 + (bBase + 2) * TILE]; a3 += av * smem[12 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 13 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[13 + (bBase + 0) * TILE]; a1 += av * smem[13 + (bBase + 1) * TILE]; a2 += av * smem[13 + (bBase + 2) * TILE]; a3 += av * smem[13 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 14 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[14 + (bBase + 0) * TILE]; a1 += av * smem[14 + (bBase + 1) * TILE]; a2 += av * smem[14 + (bBase + 2) * TILE]; a3 += av * smem[14 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 15 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[15 + (bBase + 0) * TILE]; a1 += av * smem[15 + (bBase + 1) * TILE]; a2 += av * smem[15 + (bBase + 2) * TILE]; a3 += av * smem[15 + (bBase + 3) * TILE]; }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            uint c0 = colBase + lid1 * BLOCK + 0;
            uint c1 = colBase + lid1 * BLOCK + 1;
            uint c2 = colBase + lid1 * BLOCK + 2;
            uint c3 = colBase + lid1 * BLOCK + 3;
            if (outRow < aRows)
            {
                uint rowBase = outRow * bCols + colBase + lid1 * BLOCK;
                if (c0 < bCols) c[rowBase + 0] = a0;
                if (c1 < bCols) c[rowBase + 1] = a1;
                if (c2 < bCols) c[rowBase + 2] = a2;
                if (c3 < bCols) c[rowBase + 3] = a3;
            }
        }

        /* Fused GEMM + residual add: row4 GEMM store adds the residual row in place. */

        __kernel void rms_norm_lm_head_bf16(
            __global const float* x,
            __global const float* gamma,
            __global const ushort* b,
            __global float* c,
            const uint aRows,
            const uint aCols,
            const uint bCols,
            const float eps)
        {
            uint lid0 = get_local_id(0);
            uint lid1 = get_local_id(1);
            uint outRow = get_global_id(0);
            uint colBase = get_group_id(1) * TILECOLS;

            float normScale = 0.0f;
            if (outRow < aRows)
            {
                uint rbase = outRow * aCols;
                float ssq = 0.0f;
                for (uint k = 0; k < aCols; ++k)
                {
                    float v = x[rbase + k];
                    ssq += v * v;
                }
                normScale = 1.0f / sqrt(ssq / (float)aCols + eps);
            }

            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            __local float smem[TILE * ROWSTRIDE];
            for (uint k0 = 0; k0 < aCols; k0 += TILE)
            {
                uint aCol = k0 + lid1;
                uint bRow = k0 + lid0;
                smem[lid0 + lid1 * TILE] =
                    (outRow < aRows && aCol < aCols)
                    ? (x[outRow * aCols + aCol] * normScale * gamma[aCol])
                    : 0.0f;
                for (uint w = 0; w < BLOCK; ++w)
                {
                    uint outCol = colBase + lid1 * BLOCK + w;
                    smem[lid0 + (TILE + lid1 * BLOCK + w) * TILE] =
                        (bRow < aCols && outCol < bCols)
                        ? BF16_TO_F32(b[bRow * bCols + outCol])
                        : 0.0f;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                { float av = smem[lid0 + 0 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[0 + (bBase + 0) * TILE]; a1 += av * smem[0 + (bBase + 1) * TILE]; a2 += av * smem[0 + (bBase + 2) * TILE]; a3 += av * smem[0 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 1 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[1 + (bBase + 0) * TILE]; a1 += av * smem[1 + (bBase + 1) * TILE]; a2 += av * smem[1 + (bBase + 2) * TILE]; a3 += av * smem[1 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 2 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[2 + (bBase + 0) * TILE]; a1 += av * smem[2 + (bBase + 1) * TILE]; a2 += av * smem[2 + (bBase + 2) * TILE]; a3 += av * smem[2 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 3 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[3 + (bBase + 0) * TILE]; a1 += av * smem[3 + (bBase + 1) * TILE]; a2 += av * smem[3 + (bBase + 2) * TILE]; a3 += av * smem[3 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 4 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[4 + (bBase + 0) * TILE]; a1 += av * smem[4 + (bBase + 1) * TILE]; a2 += av * smem[4 + (bBase + 2) * TILE]; a3 += av * smem[4 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 5 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[5 + (bBase + 0) * TILE]; a1 += av * smem[5 + (bBase + 1) * TILE]; a2 += av * smem[5 + (bBase + 2) * TILE]; a3 += av * smem[5 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 6 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[6 + (bBase + 0) * TILE]; a1 += av * smem[6 + (bBase + 1) * TILE]; a2 += av * smem[6 + (bBase + 2) * TILE]; a3 += av * smem[6 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 7 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[7 + (bBase + 0) * TILE]; a1 += av * smem[7 + (bBase + 1) * TILE]; a2 += av * smem[7 + (bBase + 2) * TILE]; a3 += av * smem[7 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 8 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[8 + (bBase + 0) * TILE]; a1 += av * smem[8 + (bBase + 1) * TILE]; a2 += av * smem[8 + (bBase + 2) * TILE]; a3 += av * smem[8 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 9 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[9 + (bBase + 0) * TILE]; a1 += av * smem[9 + (bBase + 1) * TILE]; a2 += av * smem[9 + (bBase + 2) * TILE]; a3 += av * smem[9 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 10 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[10 + (bBase + 0) * TILE]; a1 += av * smem[10 + (bBase + 1) * TILE]; a2 += av * smem[10 + (bBase + 2) * TILE]; a3 += av * smem[10 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 11 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[11 + (bBase + 0) * TILE]; a1 += av * smem[11 + (bBase + 1) * TILE]; a2 += av * smem[11 + (bBase + 2) * TILE]; a3 += av * smem[11 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 12 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[12 + (bBase + 0) * TILE]; a1 += av * smem[12 + (bBase + 1) * TILE]; a2 += av * smem[12 + (bBase + 2) * TILE]; a3 += av * smem[12 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 13 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[13 + (bBase + 0) * TILE]; a1 += av * smem[13 + (bBase + 1) * TILE]; a2 += av * smem[13 + (bBase + 2) * TILE]; a3 += av * smem[13 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 14 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[14 + (bBase + 0) * TILE]; a1 += av * smem[14 + (bBase + 1) * TILE]; a2 += av * smem[14 + (bBase + 2) * TILE]; a3 += av * smem[14 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 15 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[15 + (bBase + 0) * TILE]; a1 += av * smem[15 + (bBase + 1) * TILE]; a2 += av * smem[15 + (bBase + 2) * TILE]; a3 += av * smem[15 + (bBase + 3) * TILE]; }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            uint c0 = colBase + lid1 * BLOCK + 0;
            uint c1 = colBase + lid1 * BLOCK + 1;
            uint c2 = colBase + lid1 * BLOCK + 2;
            uint c3 = colBase + lid1 * BLOCK + 3;
            if (outRow < aRows)
            {
                uint rowBase = outRow * bCols + colBase + lid1 * BLOCK;
                if (c0 < bCols) c[rowBase + 0] = a0;
                if (c1 < bCols) c[rowBase + 1] = a1;
                if (c2 < bCols) c[rowBase + 2] = a2;
                if (c3 < bCols) c[rowBase + 3] = a3;
            }
        }

        /* Fused GEMM + residual add: row4 GEMM store adds the residual row in place. */

        __kernel void rms_norm_lm_head_f16(
            __global const float* x,
            __global const float* gamma,
            __global const ushort* b,
            __global float* c,
            const uint aRows,
            const uint aCols,
            const uint bCols,
            const float eps)
        {
            uint lid0 = get_local_id(0);
            uint lid1 = get_local_id(1);
            uint outRow = get_global_id(0);
            uint colBase = get_group_id(1) * TILECOLS;

            float normScale = 0.0f;
            if (outRow < aRows)
            {
                uint rbase = outRow * aCols;
                float ssq = 0.0f;
                for (uint k = 0; k < aCols; ++k)
                {
                    float v = x[rbase + k];
                    ssq += v * v;
                }
                normScale = 1.0f / sqrt(ssq / (float)aCols + eps);
            }

            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            __local float smem[TILE * ROWSTRIDE];
            for (uint k0 = 0; k0 < aCols; k0 += TILE)
            {
                uint aCol = k0 + lid1;
                uint bRow = k0 + lid0;
                smem[lid0 + lid1 * TILE] =
                    (outRow < aRows && aCol < aCols)
                    ? (x[outRow * aCols + aCol] * normScale * gamma[aCol])
                    : 0.0f;
                for (uint w = 0; w < BLOCK; ++w)
                {
                    uint outCol = colBase + lid1 * BLOCK + w;
                    smem[lid0 + (TILE + lid1 * BLOCK + w) * TILE] =
                        (bRow < aCols && outCol < bCols)
                        ? F16_TO_F32(b[bRow * bCols + outCol])
                        : 0.0f;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                { float av = smem[lid0 + 0 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[0 + (bBase + 0) * TILE]; a1 += av * smem[0 + (bBase + 1) * TILE]; a2 += av * smem[0 + (bBase + 2) * TILE]; a3 += av * smem[0 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 1 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[1 + (bBase + 0) * TILE]; a1 += av * smem[1 + (bBase + 1) * TILE]; a2 += av * smem[1 + (bBase + 2) * TILE]; a3 += av * smem[1 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 2 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[2 + (bBase + 0) * TILE]; a1 += av * smem[2 + (bBase + 1) * TILE]; a2 += av * smem[2 + (bBase + 2) * TILE]; a3 += av * smem[2 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 3 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[3 + (bBase + 0) * TILE]; a1 += av * smem[3 + (bBase + 1) * TILE]; a2 += av * smem[3 + (bBase + 2) * TILE]; a3 += av * smem[3 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 4 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[4 + (bBase + 0) * TILE]; a1 += av * smem[4 + (bBase + 1) * TILE]; a2 += av * smem[4 + (bBase + 2) * TILE]; a3 += av * smem[4 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 5 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[5 + (bBase + 0) * TILE]; a1 += av * smem[5 + (bBase + 1) * TILE]; a2 += av * smem[5 + (bBase + 2) * TILE]; a3 += av * smem[5 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 6 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[6 + (bBase + 0) * TILE]; a1 += av * smem[6 + (bBase + 1) * TILE]; a2 += av * smem[6 + (bBase + 2) * TILE]; a3 += av * smem[6 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 7 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[7 + (bBase + 0) * TILE]; a1 += av * smem[7 + (bBase + 1) * TILE]; a2 += av * smem[7 + (bBase + 2) * TILE]; a3 += av * smem[7 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 8 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[8 + (bBase + 0) * TILE]; a1 += av * smem[8 + (bBase + 1) * TILE]; a2 += av * smem[8 + (bBase + 2) * TILE]; a3 += av * smem[8 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 9 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[9 + (bBase + 0) * TILE]; a1 += av * smem[9 + (bBase + 1) * TILE]; a2 += av * smem[9 + (bBase + 2) * TILE]; a3 += av * smem[9 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 10 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[10 + (bBase + 0) * TILE]; a1 += av * smem[10 + (bBase + 1) * TILE]; a2 += av * smem[10 + (bBase + 2) * TILE]; a3 += av * smem[10 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 11 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[11 + (bBase + 0) * TILE]; a1 += av * smem[11 + (bBase + 1) * TILE]; a2 += av * smem[11 + (bBase + 2) * TILE]; a3 += av * smem[11 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 12 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[12 + (bBase + 0) * TILE]; a1 += av * smem[12 + (bBase + 1) * TILE]; a2 += av * smem[12 + (bBase + 2) * TILE]; a3 += av * smem[12 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 13 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[13 + (bBase + 0) * TILE]; a1 += av * smem[13 + (bBase + 1) * TILE]; a2 += av * smem[13 + (bBase + 2) * TILE]; a3 += av * smem[13 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 14 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[14 + (bBase + 0) * TILE]; a1 += av * smem[14 + (bBase + 1) * TILE]; a2 += av * smem[14 + (bBase + 2) * TILE]; a3 += av * smem[14 + (bBase + 3) * TILE]; }
                { float av = smem[lid0 + 15 * TILE]; uint bBase = TILE + lid1 * BLOCK; a0 += av * smem[15 + (bBase + 0) * TILE]; a1 += av * smem[15 + (bBase + 1) * TILE]; a2 += av * smem[15 + (bBase + 2) * TILE]; a3 += av * smem[15 + (bBase + 3) * TILE]; }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            uint c0 = colBase + lid1 * BLOCK + 0;
            uint c1 = colBase + lid1 * BLOCK + 1;
            uint c2 = colBase + lid1 * BLOCK + 2;
            uint c3 = colBase + lid1 * BLOCK + 3;
            if (outRow < aRows)
            {
                uint rowBase = outRow * bCols + colBase + lid1 * BLOCK;
                if (c0 < bCols) c[rowBase + 0] = a0;
                if (c1 < bCols) c[rowBase + 1] = a1;
                if (c2 < bCols) c[rowBase + 2] = a2;
                if (c3 < bCols) c[rowBase + 3] = a3;
            }
        }

        /* Fused GEMM + residual add: row4 GEMM store adds the residual row in place. */

        __kernel void gemm_row4_add(
            __global const float* a,
            __global const float* b,
            __global float* c,
            __global const float* residual,
            const uint aRows,
            const uint aCols,
            const uint bCols)
        {
            GEMM_ROW4_ACCUMULATE
            if (outRow < aRows)
            {
                uint rowBase = outRow * bCols + colBase + y * BLOCK;
                if (c0 < bCols) c[rowBase + 0] = a0 + residual[rowBase + 0];
                if (c1 < bCols) c[rowBase + 1] = a1 + residual[rowBase + 1];
                if (c2 < bCols) c[rowBase + 2] = a2 + residual[rowBase + 2];
                if (c3 < bCols) c[rowBase + 3] = a3 + residual[rowBase + 3];
            }
        }

        /* Fused RMSNorm + multi-output GEMV for single-token decode: one row, and up to three
           projections that share its one normalization (cached-decode Q/K/V, gate/up, or the tied
           head). The launch is flat - there is no row dimension to leave idle, which is the whole
           point: the tile form launched TILE rows for a one-row problem and ran 16/256 threads
           useful. A group holds GEMVLANES threads, each owning BLOCK consecutive output columns,
           so the group covers GEMVCOLS columns and consecutive lanes read consecutive spans of b -
           coalesced, with no shared-memory staging needed. Every result column keeps a single
           accumulator summed over k ascending, and the normalized activation is the same
           expression the tile form staged (x * normScale * gamma), so a column's sum is the same
           sequence of float multiplies and adds the tile form performed.

           A slot whose column count is zero contributes no group, so an unused slot needs no
           buffers. hasBias == 0 stores the plain accumulator: a zero bias with hasBias != 0 would
           be acc + 0.0f, which canonicalizes -0.0f and is a different store.

           No barrier appears here, so the early returns below cannot deadlock a group. */
        __kernel void rms_norm_gemv(
            __global const float* x,
            __global const float* gamma,
            __global const float* w0,
            __global const float* bias0,
            __global float* out0,
            __global const float* w1,
            __global const float* bias1,
            __global float* out1,
            __global const float* w2,
            __global const float* bias2,
            __global float* out2,
            const uint aCols,
            const uint cols0,
            const uint cols1,
            const uint cols2,
            const uint hasBias,
            const float eps)
        {
            uint g = get_group_id(0);
            uint groups0 = (cols0 + GEMVCOLS - 1) / GEMVCOLS;
            uint groups1 = (cols1 + GEMVCOLS - 1) / GEMVCOLS;

            __global const float* b;
            __global const float* bias;
            __global float* c;
            uint bCols;
            uint colBase;
            if (g < groups0)
            {
                b = w0;
                bias = bias0;
                c = out0;
                bCols = cols0;
                colBase = g * GEMVCOLS;
            }
            else if (g < groups0 + groups1)
            {
                b = w1;
                bias = bias1;
                c = out1;
                bCols = cols1;
                colBase = (g - groups0) * GEMVCOLS;
            }
            else
            {
                b = w2;
                bias = bias2;
                c = out2;
                bCols = cols2;
                colBase = (g - groups0 - groups1) * GEMVCOLS;
            }

            uint col = colBase + get_local_id(0) * BLOCK;
            if (col >= bCols)
                return;

            float ssq = 0.0f;
            for (uint k = 0; k < aCols; ++k)
            {
                float v = x[k];
                ssq += v * v;
            }
            float normScale = 1.0f / sqrt(ssq / (float)aCols + eps);

            /* A lane's last columns run past bCols when the width is not a whole number of
               BLOCK-wide lanes, which is what bounds every b and c access below to [col, bCols). */
            uint avail = bCols - col;

            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (uint k = 0; k < aCols; ++k)
            {
                float av = x[k] * normScale * gamma[k];
                uint bBase = k * bCols + col;
                a0 += av * b[bBase];
                if (avail > 1u) a1 += av * b[bBase + 1u];
                if (avail > 2u) a2 += av * b[bBase + 2u];
                if (avail > 3u) a3 += av * b[bBase + 3u];
            }

            if (hasBias != 0u)
            {
                c[col] = a0 + bias[col];
                if (avail > 1u) c[col + 1u] = a1 + bias[col + 1u];
                if (avail > 2u) c[col + 2u] = a2 + bias[col + 2u];
                if (avail > 3u) c[col + 3u] = a3 + bias[col + 3u];
            }
            else
            {
                c[col] = a0;
                if (avail > 1u) c[col + 1u] = a1;
                if (avail > 2u) c[col + 2u] = a2;
                if (avail > 3u) c[col + 3u] = a3;
            }
        }

        /* Flat single-row GEMV + residual add for cached-decode O-projection (#672, Stage A1).
           a is [aCols], b is [aCols * bCols] row-major (B[k, outCol]), result c[outCol] =
           sum_k a[k]*b[k*bCols+outCol] + residual[outCol]. One work item per BLOCK output
           columns, single accumulator over k ascending, no shared-memory staging. */
        __kernel void gemm_gemv_add(
            __global const float* a,
            __global const float* b,
            __global float* c,
            __global const float* residual,
            const uint aCols,
            const uint bCols)
        {
            uint col = get_global_id(0) * BLOCK;
            if (col >= bCols) return;

            uint avail = bCols - col;

            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (uint k = 0; k < aCols; ++k)
            {
                float av = a[k];
                uint bBase = k * bCols + col;
                a0 += av * b[bBase];
                if (avail > 1u) a1 += av * b[bBase + 1u];
                if (avail > 2u) a2 += av * b[bBase + 2u];
                if (avail > 3u) a3 += av * b[bBase + 3u];
            }

            c[col] = a0 + residual[col];
            if (avail > 1u) c[col + 1u] = a1 + residual[col + 1u];
            if (avail > 2u) c[col + 2u] = a2 + residual[col + 2u];
            if (avail > 3u) c[col + 3u] = a3 + residual[col + 3u];
        }

        /* Fused single-token MLP block for cached decode (#672, Stage B): RMSNorm(x) feeds
           gate/up, SwiGLU, down-projection, plus the residual add, in one launch. Shared memory
           holds the intermediate swiglu vector (max 5120 columns). One work-group of 256 lanes. */
        #define MLP_MAX_INTERMEDIATE 5120
        __kernel void mlp_block(
            __global const float* x,
            __global const float* gamma,
            __global const float* Wgate,
            __global const float* Wup,
            __global const float* Wdown,
            __global float* out,
            __global const float* residual,
            const uint hidden,
            const uint intermediate,
            const float eps)
        {
            uint lid = get_local_id(0);
            uint gsize = get_local_size(0);

            __local float swiglu[MLP_MAX_INTERMEDIATE];

            // norm scale (each lane computes identically; cheap relative to the GEMVs)
            float ssq = 0.0f;
            for (uint k = 0; k < hidden; ++k)
            {
                float v = x[k];
                ssq += v * v;
            }
            float normScale = 1.0f / sqrt(ssq / (float)hidden + eps);

            // phase 1: gate/up/swiglu into shared memory
            for (uint i = lid; i < intermediate; i += gsize)
            {
                float g = 0.0f, u = 0.0f;
                for (uint k = 0; k < hidden; ++k)
                {
                    float nk = x[k] * normScale * gamma[k];
                    g += nk * Wgate[k * intermediate + i];
                    u += nk * Wup[k * intermediate + i];
                }
                float siluGate = g / (1.0f + exp(-g));
                swiglu[i] = siluGate * u;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // phase 2: down projection + residual
            for (uint j = lid; j < hidden; j += gsize)
            {
                float acc = 0.0f;
                for (uint i = 0; i < intermediate; ++i)
                    acc += swiglu[i] * Wdown[i * hidden + j];
                out[j] = acc + residual[j];
            }
        }
        /* Fused single-work-group decode attention + O-projection (#691). One work-group of
           numHeads * ATTNGROUP threads computes every head's attention context into the shared
           ctx[] array, barriers, then does the O-projection (GEMV + residual) over hiddenDim
           output columns. No global context buffer is written. Only usable when the device
           exposes at least numHeads * ATTNGROUP work-items per group. */
        #define FUSED_MAX_HEADS 64
        __kernel void decode_gqa_attention_oproj_fused(
            __global const float* q,
            __global const float* kCache,
            __global const float* vCache,
            __global const float* Wo,
            __global const float* residual,
            __global float* out,
            const uint kvLen,
            const uint numHeads,
            const uint numKvHeads,
            const uint headDim,
            const uint hiddenDim,
            const float scale)
        {
            uint lid = get_local_id(0);
            uint gsize = get_local_size(0);
            const uint groupSizePerHead = ATTNGROUP;

            uint head = lid / groupSizePerHead;
            uint lane = lid % groupSizePerHead;

            __local float ctx[FUSED_MAX_HEADS * MAXHEADDIM];
            __local float lmx[ATTNGROUP];
            __local float lsum[ATTNGROUP];
            __local float laccv[ATTNGROUP * MAXHEADDIM];

            uint repeat = numHeads / numKvHeads;
            uint qDim = numHeads * headDim;

            for (uint curHead = 0; curHead < numHeads; ++curHead)
            {
                if (head == curHead)
                {
                    uint kvHead = curHead / repeat;
                    uint qBase = curHead * headDim;
                    uint kBaseHead = kvHead * headDim;
                    uint vBaseHead = kvHead * headDim;
                    uint Dkv = numKvHeads * headDim;

                    float mx = -INFINITY;
                    float sum = 0.0f;
                    float accv[MAXHEADDIM];
                    for (uint d = 0; d < headDim; ++d) accv[d] = 0.0f;

                    for (uint j = lane; j < kvLen; j += groupSizePerHead)
                    {
                        float acc = 0.0f;
                        uint kBase = j * Dkv + kBaseHead;
                        for (uint d = 0; d < headDim; ++d)
                            acc += q[qBase + d] * kCache[kBase + d];
                        float s = acc * scale;

                        if (s > mx)
                        {
                            if (mx != -INFINITY)
                            {
                                float corr = exp(mx - s);
                                sum *= corr;
                                for (uint d = 0; d < headDim; ++d)
                                    accv[d] *= corr;
                            }
                            mx = s;
                        }

                        float w = exp(s - mx);
                        sum += w;
                        uint vBase = j * Dkv + vBaseHead;
                        for (uint d = 0; d < headDim; ++d)
                            accv[d] += w * vCache[vBase + d];
                    }

                    lmx[lane] = mx;
                    lsum[lane] = sum;
                    for (uint d = 0; d < headDim; ++d)
                        laccv[lane * MAXHEADDIM + d] = accv[d];
                }

                barrier(CLK_LOCAL_MEM_FENCE);

                // Tree reduction across the head's lanes (matches decode_gqa_attention order).
                if (head == curHead)
                {
                    for (uint stride = groupSizePerHead / 2; stride > 0; stride >>= 1)
                    {
                        if (lane < stride)
                        {
                            uint other = lane + stride;
                            float mx_a = lmx[lane];
                            float mx_b = lmx[other];
                            float mx_new = mx_a > mx_b ? mx_a : mx_b;

                            float corr_a = (mx_a == -INFINITY) ? 0.0f : exp(mx_a - mx_new);
                            float corr_b = (mx_b == -INFINITY) ? 0.0f : exp(mx_b - mx_new);

                            float sum_new = lsum[lane] * corr_a + lsum[other] * corr_b;
                            for (uint d = 0; d < headDim; ++d)
                                laccv[lane * MAXHEADDIM + d] =
                                    laccv[lane * MAXHEADDIM + d] * corr_a +
                                    laccv[other * MAXHEADDIM + d] * corr_b;

                            lmx[lane] = mx_new;
                            lsum[lane] = sum_new;
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                }
                else
                {
                    // Non-participating threads still reach the barriers inside the loop.
                    for (uint stride = groupSizePerHead / 2; stride > 0; stride >>= 1)
                    {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                }

                if (head == curHead && lane == 0)
                {
                    float mx_final = lmx[0];
                    float sum_final = lsum[0];
                    if (mx_final == -INFINITY || sum_final == 0.0f)
                    {
                        for (uint d = 0; d < headDim; ++d)
                            ctx[curHead * headDim + d] = 0.0f;
                    }
                    else
                    {
                        float invSum = 1.0f / sum_final;
                        for (uint d = 0; d < headDim; ++d)
                            ctx[curHead * headDim + d] = laccv[0 * MAXHEADDIM + d] * invSum;
                    }
                }

                barrier(CLK_LOCAL_MEM_FENCE);
            }

            // O-projection pass (all threads).
            barrier(CLK_LOCAL_MEM_FENCE);
            for (uint col = lid; col < hiddenDim; col += gsize)
            {
                float acc = 0.0f;
                for (uint k = 0; k < qDim; ++k)
                    acc += ctx[k] * Wo[k * hiddenDim + col];
                out[col] = acc + residual[col];
            }
        }

        __kernel void grouped_hash_reduce_partial(
            __global const int* serviceIds,
            __global const int* statusCodes,
            __global const float* durations,
            __global int* partialCounts,
            __global int* partialErrors,
            __global float* partialSums,
            const uint n,
            const uint groupCount)
        {
            uint gid = get_global_id(0);
            uint globalSize = get_global_size(0);
            uint wg = get_group_id(0);

            int localCounts[64];
            int localErrors[64];
            float localSums[64];
            for (uint g = 0; g < groupCount; ++g)
            {
                localCounts[g] = 0;
                localErrors[g] = 0;
                localSums[g] = 0.0f;
            }

            for (uint i = gid; i < n; i += globalSize)
            {
                uint g = (uint)serviceIds[i];
                localCounts[g]++;
                if (statusCodes[i] >= 500)
                    localErrors[g]++;
                localSums[g] += durations[i];
            }

            uint baseOffset = wg * groupCount;
            for (uint g = 0; g < groupCount; ++g)
            {
                partialCounts[baseOffset + g] = localCounts[g];
                partialErrors[baseOffset + g] = localErrors[g];
                partialSums[baseOffset + g] = localSums[g];
            }
        }

        __kernel void rolling_min(
            __global const float* input,
            __global float* output,
            __global int* valid,
            const int n,
            const int windowSize,
            const int minPeriods)
        {
            int globalSize = get_global_size(0);
            for (int i = get_global_id(0); i < n; i += globalSize)
            {
                int start = i - windowSize + 1;
                if (start < 0)
                    start = 0;

                float minValue = input[start];
                int count = 0;
                for (int j = start; j <= i; ++j)
                {
                    if (j == start)
                        minValue = input[j];
                    else if (input[j] < minValue)
                        minValue = input[j];
                    ++count;
                }

                if (count >= minPeriods)
                {
                    output[i] = minValue;
                    valid[i] = 1;
                }
                else
                {
                    output[i] = 0.0f;
                    valid[i] = 0;
                }
            }
        }
        __kernel void rolling_max(
            __global const float* input,
            __global float* output,
            __global int* valid,
            const int n,
            const int windowSize,
            const int minPeriods)
        {
            int globalSize = get_global_size(0);
            for (int i = get_global_id(0); i < n; i += globalSize)
            {
                int start = i - windowSize + 1;
                if (start < 0)
                    start = 0;

                float maxValue = input[start];
                int count = 0;
                for (int j = start; j <= i; ++j)
                {
                    if (j == start)
                        maxValue = input[j];
                    else if (input[j] > maxValue)
                        maxValue = input[j];
                    ++count;
                }

                if (count >= minPeriods)
                {
                    output[i] = maxValue;
                    valid[i] = 1;
                }
                else
                {
                    output[i] = 0.0f;
                    valid[i] = 0;
                }
            }
        }

        __kernel void grouped_multistat_reduce_partial(
            __global const int* serviceIds,
            __global const int* statusCodes,
            __global const float* durations,
            __global int* partialCounts,
            __global int* partialErrors,
            __global float* partialSums,
            __global float* partialSumsq,
            __global float* partialMins,
            __global float* partialMaxs,
            const uint n,
            const uint groupCount)
        {
            uint gid = get_global_id(0);
            uint globalSize = get_global_size(0);
            uint wg = get_group_id(0);

            int localCounts[64];
            int localErrors[64];
            float localSums[64];
            float localSumsq[64];
            float localMins[64];
            float localMaxs[64];
            for (uint g = 0; g < groupCount; ++g)
            {
                localCounts[g] = 0;
                localErrors[g] = 0;
                localSums[g] = 0.0f;
                localSumsq[g] = 0.0f;
                localMins[g] = 3.40282346638528860e+38f;
                localMaxs[g] = -3.40282346638528860e+38f;
            }

            for (uint i = gid; i < n; i += globalSize)
            {
                uint g = (uint)serviceIds[i];
                localCounts[g]++;
                if (statusCodes[i] >= 500)
                    localErrors[g]++;
                float d = durations[i];
                localSums[g] += d;
                localSumsq[g] += d * d;
                if (d < localMins[g]) localMins[g] = d;
                if (d > localMaxs[g]) localMaxs[g] = d;
            }

            uint baseOffset = wg * groupCount;
            for (uint g = 0; g < groupCount; ++g)
            {
                partialCounts[baseOffset + g] = localCounts[g];
                partialErrors[baseOffset + g] = localErrors[g];
                partialSums[baseOffset + g] = localSums[g];
                partialSumsq[baseOffset + g] = localSumsq[g];
                partialMins[baseOffset + g] = localMins[g];
                partialMaxs[baseOffset + g] = localMaxs[g];
            }
        }
        __kernel void rolling_mean(
            __global const float* input,
            __global float* output,
            __global int* valid,
            const int n,
            const int windowSize,
            const int minPeriods)
        {
            int globalSize = get_global_size(0);
            for (int i = get_global_id(0); i < n; i += globalSize)
            {
            int start = i - windowSize + 1;
            if (start < 0)
                start = 0;

            float sum = 0.0f;
            int count = 0;
            for (int j = start; j <= i; ++j)
            {
                sum += input[j];
                ++count;
            }

            if (count >= minPeriods)
            {
                output[i] = sum / (float)count;
                valid[i] = 1;
            }
            else
            {
                output[i] = 0.0f;
                valid[i] = 0;
            }
            }
        }
        __kernel void gemm_row4_bf16(
                    __global const float* a,
                    __global const ushort* b,
                    __global float* c,
                    const uint aRows,
                    const uint aCols,
                    const uint bCols)
                {
                    GEMM_ROW4_ACCUMULATE_BF16
                    if (outRow < aRows)
                    {
                        uint rowBase = outRow * bCols + colBase + y * BLOCK;
                        if (c0 < bCols) c[rowBase + 0] = a0;
                        if (c1 < bCols) c[rowBase + 1] = a1;
                        if (c2 < bCols) c[rowBase + 2] = a2;
                        if (c3 < bCols) c[rowBase + 3] = a3;
                    }
                }

        __kernel void gemm_row4_f16(
                    __global const float* a,
                    __global const ushort* b,
                    __global float* c,
                    const uint aRows,
                    const uint aCols,
                    const uint bCols)
                {
                    GEMM_ROW4_ACCUMULATE_F16
                    if (outRow < aRows)
                    {
                        uint rowBase = outRow * bCols + colBase + y * BLOCK;
                        if (c0 < bCols) c[rowBase + 0] = a0;
                        if (c1 < bCols) c[rowBase + 1] = a1;
                        if (c2 < bCols) c[rowBase + 2] = a2;
                        if (c3 < bCols) c[rowBase + 3] = a3;
                    }
                }


        __kernel void gemm_gemv_add_bf16(
                    __global const float* a,
                    __global const ushort* b,
                    __global float* c,
                    __global const float* residual,
                    const uint aCols,
                    const uint bCols)
                {
                    uint col = get_global_id(0) * BLOCK;
                    if (col >= bCols) return;

                    uint avail = bCols - col;

                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                    for (uint k = 0; k < aCols; ++k)
                    {
                        float av = a[k];
                        uint bBase = k * bCols + col;
                        a0 += av * BF16_TO_F32(b[bBase]);
                        if (avail > 1u) a1 += av * BF16_TO_F32(b[bBase + 1u]);
                        if (avail > 2u) a2 += av * BF16_TO_F32(b[bBase + 2u]);
                        if (avail > 3u) a3 += av * BF16_TO_F32(b[bBase + 3u]);
                    }

                    c[col] = a0 + residual[col];
                    if (avail > 1u) c[col + 1u] = a1 + residual[col + 1u];
                    if (avail > 2u) c[col + 2u] = a2 + residual[col + 2u];
                    if (avail > 3u) c[col + 3u] = a3 + residual[col + 3u];
                }

                /* Fused single-token MLP block for cached decode (#672, Stage B): RMSNorm(x) feeds
                   gate/up, SwiGLU, down-projection, plus the residual add, in one launch. Shared memory
                   holds the intermediate swiglu vector (max 5120 columns). One work-group of 256 lanes. */

        __kernel void rms_norm_gemv_bf16(
                    __global const float* x,
                    __global const float* gamma,
                    __global const ushort* w0,
                    __global const float* bias0,
                    __global float* out0,
                    __global const ushort* w1,
                    __global const float* bias1,
                    __global float* out1,
                    __global const ushort* w2,
                    __global const float* bias2,
                    __global float* out2,
                    const uint aCols,
                    const uint cols0,
                    const uint cols1,
                    const uint cols2,
                    const uint hasBias,
                    const float eps)
                {
                    uint g = get_group_id(0);
                    uint groups0 = (cols0 + GEMVCOLS - 1) / GEMVCOLS;
                    uint groups1 = (cols1 + GEMVCOLS - 1) / GEMVCOLS;

                    __global const ushort* b;
                    __global const float* bias;
                    __global float* c;
                    uint bCols;
                    uint colBase;
                    if (g < groups0)
                    {
                        b = w0;
                        bias = bias0;
                        c = out0;
                        bCols = cols0;
                        colBase = g * GEMVCOLS;
                    }
                    else if (g < groups0 + groups1)
                    {
                        b = w1;
                        bias = bias1;
                        c = out1;
                        bCols = cols1;
                        colBase = (g - groups0) * GEMVCOLS;
                    }
                    else
                    {
                        b = w2;
                        bias = bias2;
                        c = out2;
                        bCols = cols2;
                        colBase = (g - groups0 - groups1) * GEMVCOLS;
                    }

                    uint col = colBase + get_local_id(0) * BLOCK;
                    if (col >= bCols)
                        return;

                    float ssq = 0.0f;
                    for (uint k = 0; k < aCols; ++k)
                    {
                        float v = x[k];
                        ssq += v * v;
                    }
                    float normScale = 1.0f / sqrt(ssq / (float)aCols + eps);

                    /* A lane's last columns run past bCols when the width is not a whole number of
                       BLOCK-wide lanes, which is what bounds every b and c access below to [col, bCols). */
                    uint avail = bCols - col;

                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                    for (uint k = 0; k < aCols; ++k)
                    {
                        float av = x[k] * normScale * gamma[k];
                        uint bBase = k * bCols + col;
                        a0 += av * BF16_TO_F32(b[bBase]);
                        if (avail > 1u) a1 += av * BF16_TO_F32(b[bBase + 1u]);
                        if (avail > 2u) a2 += av * BF16_TO_F32(b[bBase + 2u]);
                        if (avail > 3u) a3 += av * BF16_TO_F32(b[bBase + 3u]);
                    }

                    if (hasBias != 0u)
                    {
                        c[col] = a0 + bias[col];
                        if (avail > 1u) c[col + 1u] = a1 + bias[col + 1u];
                        if (avail > 2u) c[col + 2u] = a2 + bias[col + 2u];
                        if (avail > 3u) c[col + 3u] = a3 + bias[col + 3u];
                    }
                    else
                    {
                        c[col] = a0;
                        if (avail > 1u) c[col + 1u] = a1;
                        if (avail > 2u) c[col + 2u] = a2;
                        if (avail > 3u) c[col + 3u] = a3;
                    }
                }

                /* Flat single-row GEMV + residual add for cached-decode O-projection (#672, Stage A1).
                   a is [aCols], b is [aCols * bCols] row-major (B[k, outCol]), result c[outCol] =
                   sum_k a[k]*b[k*bCols+outCol] + residual[outCol]. One work item per BLOCK output
                   columns, single accumulator over k ascending, no shared-memory staging. */

        __kernel void mlp_block_bf16(
                    __global const float* x,
                    __global const float* gamma,
                    __global const ushort* Wgate,
                    __global const ushort* Wup,
                    __global const ushort* Wdown,
                    __global float* out,
                    __global const float* residual,
                    const uint hidden,
                    const uint intermediate,
                    const float eps)
                {
                    uint lid = get_local_id(0);
                    uint gsize = get_local_size(0);

                    __local float swiglu[MLP_MAX_INTERMEDIATE];

                    // norm scale (each lane computes identically; cheap relative to the GEMVs)
                    float ssq = 0.0f;
                    for (uint k = 0; k < hidden; ++k)
                    {
                        float v = x[k];
                        ssq += v * v;
                    }
                    float normScale = 1.0f / sqrt(ssq / (float)hidden + eps);

                    // phase 1: gate/up/swiglu into shared memory
                    for (uint i = lid; i < intermediate; i += gsize)
                    {
                        float g = 0.0f, u = 0.0f;
                        for (uint k = 0; k < hidden; ++k)
                        {
                            float nk = x[k] * normScale * gamma[k];
                            g += nk * BF16_TO_F32(Wgate[k * intermediate + i]);
                            u += nk * BF16_TO_F32(Wup[k * intermediate + i]);
                        }
                        float siluGate = g / (1.0f + exp(-g));
                        swiglu[i] = siluGate * u;
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);

                    // phase 2: down projection + residual
                    for (uint j = lid; j < hidden; j += gsize)
                    {
                        float acc = 0.0f;
                        for (uint i = 0; i < intermediate; ++i)
                            acc += swiglu[i] * BF16_TO_F32(Wdown[i * hidden + j]);
                        out[j] = acc + residual[j];
                    }
                }

        /* Measured candidate geometry: 2x4 @ K16 (issue #784). Plain GEMM, ascending K. */
        __kernel void gemm_reg2x4(
            __global const float* a,
            __global const float* b,
            __global float* c,
            const uint aRows,
            const uint aCols,
            const uint bCols)
        {
            const uint BR = 2, BC = 4, KT = 16, TILE_K = 16;
            const uint TileRows = TILE_K * BR, TileCols = TILE_K * BC;
            uint x = get_local_id(0);
            uint y = get_local_id(1);
            uint outRowBase = get_group_id(0) * TileRows;
            uint colBase = get_group_id(1) * TileCols;
            __local float smem[TILE_K * (KT + TileCols)];
            float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
            float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;

            for (uint k0 = 0; k0 < aCols; k0 += KT)
            {
                uint stripOfThread = (x * BR) / KT;
                for (uint s = 0; s < (TileRows / KT); ++s)
                {
                    if (s == stripOfThread)
                    {
                        if (y < 4u)
                        {
                            for (uint r = 0; r < BR; ++r)
                            {
                                uint srcRow = outRowBase + x * BR + r;
                                uint slot = x * BR + r - s * KT;
                                for (uint kk = 0; kk < KT; kk += TILE_K)
                                {
                                    uint aColBase = k0 + kk + y * 4u;
                                    uint aBase = srcRow * aCols + aColBase;
                                    if (srcRow < aRows && (aColBase + 3u) < aCols)
                                    {
                                        float4 av = vload4(0, a + aBase);
                                        smem[slot + (kk + y * 4u + 0u) * (KT + TileCols)] = av.s0;
                                        smem[slot + (kk + y * 4u + 1u) * (KT + TileCols)] = av.s1;
                                        smem[slot + (kk + y * 4u + 2u) * (KT + TileCols)] = av.s2;
                                        smem[slot + (kk + y * 4u + 3u) * (KT + TileCols)] = av.s3;
                                    }
                                    else
                                    {
                                        for (uint aw = 0u; aw < 4u; ++aw)
                                        {
                                            uint aCol = aColBase + aw;
                                            smem[slot + (kk + y * 4u + aw) * (KT + TileCols)] = (srcRow < aRows && aCol < aCols) ? a[srcRow * aCols + aCol] : 0.0f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    for (uint kk = 0; kk < KT; kk += TILE_K)
                        for (uint cc = 0; cc < TileCols; cc += TILE_K)
                        {
                            uint srcRow = k0 + kk + x;
                            uint srcCol = colBase + cc + y;
                            smem[(kk + x) + (KT + cc + y) * (KT + TileCols)] = (srcRow < aCols && srcCol < bCols) ? b[srcRow * bCols + srcCol] : 0.0f;
                        }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    if (s == stripOfThread)
                    {
                        for (uint k = 0; k < KT; ++k)
                        {
                            float a0 = smem[(x * BR + 0 - s * KT) + k * (KT + TileCols)];
                            float a1 = smem[(x * BR + 1 - s * KT) + k * (KT + TileCols)];
                            uint cb = y * BC;
                            acc00 += a0 * smem[k + (KT + cb + 0) * (KT + TileCols)];
                            acc01 += a0 * smem[k + (KT + cb + 1) * (KT + TileCols)];
                            acc02 += a0 * smem[k + (KT + cb + 2) * (KT + TileCols)];
                            acc03 += a0 * smem[k + (KT + cb + 3) * (KT + TileCols)];
                            acc10 += a1 * smem[k + (KT + cb + 0) * (KT + TileCols)];
                            acc11 += a1 * smem[k + (KT + cb + 1) * (KT + TileCols)];
                            acc12 += a1 * smem[k + (KT + cb + 2) * (KT + TileCols)];
                            acc13 += a1 * smem[k + (KT + cb + 3) * (KT + TileCols)];
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
            }

            uint d = colBase + y * BC;
            {
                uint outRow = outRowBase + x * BR + 0;
                if (outRow < aRows) { uint dst = outRow * bCols + d; if (d + 0 < bCols) c[dst + 0] = acc00; if (d + 1 < bCols) c[dst + 1] = acc01; if (d + 2 < bCols) c[dst + 2] = acc02; if (d + 3 < bCols) c[dst + 3] = acc03; }
            }
            {
                uint outRow = outRowBase + x * BR + 1;
                if (outRow < aRows) { uint dst = outRow * bCols + d; if (d + 0 < bCols) c[dst + 0] = acc10; if (d + 1 < bCols) c[dst + 1] = acc11; if (d + 2 < bCols) c[dst + 2] = acc12; if (d + 3 < bCols) c[dst + 3] = acc13; }
            }
        }

        /* Measured candidate geometry: 4x4 @ K16 (issue #784). Plain GEMM, ascending K. */
        __kernel void gemm_reg4x4(
            __global const float* a,
            __global const float* b,
            __global float* c,
            const uint aRows,
            const uint aCols,
            const uint bCols)
        {
            const uint BR = 4, BC = 4, KT = 16, TILE_K = 16;
            const uint TileRows = TILE_K * BR, TileCols = TILE_K * BC;
            uint x = get_local_id(0);
            uint y = get_local_id(1);
            uint outRowBase = get_group_id(0) * TileRows;
            uint colBase = get_group_id(1) * TileCols;
            __local float smem[TILE_K * (KT + TileCols)];
            float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
            float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
            float acc20 = 0.0f, acc21 = 0.0f, acc22 = 0.0f, acc23 = 0.0f;
            float acc30 = 0.0f, acc31 = 0.0f, acc32 = 0.0f, acc33 = 0.0f;

            for (uint k0 = 0; k0 < aCols; k0 += KT)
            {
                uint stripOfThread = (x * BR) / KT;
                for (uint s = 0; s < (TileRows / KT); ++s)
                {
                    if (s == stripOfThread)
                    {
                        if (y < 4u)
                        {
                            for (uint r = 0; r < BR; ++r)
                            {
                                uint srcRow = outRowBase + x * BR + r;
                                uint slot = x * BR + r - s * KT;
                                for (uint kk = 0; kk < KT; kk += TILE_K)
                                {
                                    uint aColBase = k0 + kk + y * 4u;
                                    uint aBase = srcRow * aCols + aColBase;
                                    if (srcRow < aRows && (aColBase + 3u) < aCols)
                                    {
                                        float4 av = vload4(0, a + aBase);
                                        smem[slot + (kk + y * 4u + 0u) * (KT + TileCols)] = av.s0;
                                        smem[slot + (kk + y * 4u + 1u) * (KT + TileCols)] = av.s1;
                                        smem[slot + (kk + y * 4u + 2u) * (KT + TileCols)] = av.s2;
                                        smem[slot + (kk + y * 4u + 3u) * (KT + TileCols)] = av.s3;
                                    }
                                    else
                                    {
                                        for (uint aw = 0u; aw < 4u; ++aw)
                                        {
                                            uint aCol = aColBase + aw;
                                            smem[slot + (kk + y * 4u + aw) * (KT + TileCols)] = (srcRow < aRows && aCol < aCols) ? a[srcRow * aCols + aCol] : 0.0f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    for (uint kk = 0; kk < KT; kk += TILE_K)
                        for (uint cc = 0; cc < TileCols; cc += TILE_K)
                        {
                            uint srcRow = k0 + kk + x;
                            uint srcCol = colBase + cc + y;
                            smem[(kk + x) + (KT + cc + y) * (KT + TileCols)] = (srcRow < aCols && srcCol < bCols) ? b[srcRow * bCols + srcCol] : 0.0f;
                        }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    if (s == stripOfThread)
                    {
                        for (uint k = 0; k < KT; ++k)
                        {
                            float a0 = smem[(x * BR + 0 - s * KT) + k * (KT + TileCols)];
                            float a1 = smem[(x * BR + 1 - s * KT) + k * (KT + TileCols)];
                            float a2 = smem[(x * BR + 2 - s * KT) + k * (KT + TileCols)];
                            float a3 = smem[(x * BR + 3 - s * KT) + k * (KT + TileCols)];
                            uint cb = y * BC;
                            acc00 += a0 * smem[k + (KT + cb + 0) * (KT + TileCols)];
                            acc01 += a0 * smem[k + (KT + cb + 1) * (KT + TileCols)];
                            acc02 += a0 * smem[k + (KT + cb + 2) * (KT + TileCols)];
                            acc03 += a0 * smem[k + (KT + cb + 3) * (KT + TileCols)];
                            acc10 += a1 * smem[k + (KT + cb + 0) * (KT + TileCols)];
                            acc11 += a1 * smem[k + (KT + cb + 1) * (KT + TileCols)];
                            acc12 += a1 * smem[k + (KT + cb + 2) * (KT + TileCols)];
                            acc13 += a1 * smem[k + (KT + cb + 3) * (KT + TileCols)];
                            acc20 += a2 * smem[k + (KT + cb + 0) * (KT + TileCols)];
                            acc21 += a2 * smem[k + (KT + cb + 1) * (KT + TileCols)];
                            acc22 += a2 * smem[k + (KT + cb + 2) * (KT + TileCols)];
                            acc23 += a2 * smem[k + (KT + cb + 3) * (KT + TileCols)];
                            acc30 += a3 * smem[k + (KT + cb + 0) * (KT + TileCols)];
                            acc31 += a3 * smem[k + (KT + cb + 1) * (KT + TileCols)];
                            acc32 += a3 * smem[k + (KT + cb + 2) * (KT + TileCols)];
                            acc33 += a3 * smem[k + (KT + cb + 3) * (KT + TileCols)];
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
            }

            uint d = colBase + y * BC;
            {
                uint outRow = outRowBase + x * BR + 0;
                if (outRow < aRows) { uint dst = outRow * bCols + d; if (d + 0 < bCols) c[dst + 0] = acc00; if (d + 1 < bCols) c[dst + 1] = acc01; if (d + 2 < bCols) c[dst + 2] = acc02; if (d + 3 < bCols) c[dst + 3] = acc03; }
            }
            {
                uint outRow = outRowBase + x * BR + 1;
                if (outRow < aRows) { uint dst = outRow * bCols + d; if (d + 0 < bCols) c[dst + 0] = acc10; if (d + 1 < bCols) c[dst + 1] = acc11; if (d + 2 < bCols) c[dst + 2] = acc12; if (d + 3 < bCols) c[dst + 3] = acc13; }
            }
            {
                uint outRow = outRowBase + x * BR + 2;
                if (outRow < aRows) { uint dst = outRow * bCols + d; if (d + 0 < bCols) c[dst + 0] = acc20; if (d + 1 < bCols) c[dst + 1] = acc21; if (d + 2 < bCols) c[dst + 2] = acc22; if (d + 3 < bCols) c[dst + 3] = acc23; }
            }
            {
                uint outRow = outRowBase + x * BR + 3;
                if (outRow < aRows) { uint dst = outRow * bCols + d; if (d + 0 < bCols) c[dst + 0] = acc30; if (d + 1 < bCols) c[dst + 1] = acc31; if (d + 2 < bCols) c[dst + 2] = acc32; if (d + 3 < bCols) c[dst + 3] = acc33; }
            }
        }

        __kernel void embedding_gather_pool(
            __global const int* ids, __global const float* table, __global float* out,
            const uint batchSize, const uint seqLen, const uint embedDim)
        {
            uint idx = get_global_id(0);
            uint total = batchSize * embedDim;
            if (idx >= total) return;
            uint b = idx / embedDim;
            uint e = idx % embedDim;
            float acc = 0.0f;
            for (uint s = 0; s < seqLen; ++s) {
                int tid = ids[b * seqLen + s];
                acc += table[(uint)tid * embedDim + e];
            }
            float denom = seqLen > 0u ? (float)seqLen : 1.0f;
            out[idx] = acc / denom;
        }

        __kernel void embedding_gather_input(
            __global const int* ids, __global const float* table, __global float* out,
            const uint batchSize, const uint seqLen, const uint embedDim)
        {
            uint idx = get_global_id(0);
            uint total = batchSize * embedDim * seqLen;
            if (idx >= total) return;
            uint s = idx % seqLen;
            uint tmp = idx / seqLen;
            uint e = tmp % embedDim;
            uint b = tmp / embedDim;
            int tid = ids[b * seqLen + s];
            out[idx] = table[(uint)tid * embedDim + e];
        }

        __kernel void embedding_backward_pool(
            __global const int* ids, __global const float* dPooled, __global float* gradEmb,
            const uint batchSize, const uint seqLen, const uint embedDim,
            const float poolScale)
        {
            uint e = get_global_id(0);
            if (e >= embedDim) return;
            for (uint b = 0; b < batchSize; ++b)
                for (uint s = 0; s < seqLen; ++s)
                {
                    uint tid = (uint)ids[b * seqLen + s];
                    gradEmb[tid * embedDim + e] += poolScale * dPooled[b * embedDim + e];
                }
        }

        __kernel void embedding_backward_input(
            __global const int* ids, __global const float* dX, __global float* gradEmb,
            const uint batchSize, const uint seqLen, const uint embedDim)
        {
            uint e = get_global_id(0);
            if (e >= embedDim) return;
            for (uint b = 0; b < batchSize; ++b)
                for (uint s = 0; s < seqLen; ++s)
                {
                    uint tid = (uint)ids[b * seqLen + s];
                    uint dIdx = b * embedDim * seqLen + e * seqLen + s;
                    gradEmb[tid * embedDim + e] += dX[dIdx];
                }
        }

        /* The split half of gemm_row4_gelu: same gelu_exact() text, so running
           gemm_row4_bias -> gelu reproduces the fused epilogue bit for bit while keeping the
           pre-activation that gelu_backward needs. */
        __kernel void gelu(__global const float* x, __global float* y, const uint n)
        {
            uint idx = get_global_id(0);
            if (idx >= n) return;
            y[idx] = gelu_exact(x[idx]);
        }

        /* Row-major embedding scatter for a [rows, hidden] activation: ids[row] picks the table
           row, one work item per embedding column, rows ascending and deterministic.
           embedding_backward_input is the pooled classifier's column-major sibling and cannot
           address this layout. */
        __kernel void embedding_backward_rows(
            __global const int* ids, __global const float* dX, __global float* gradEmb,
            const uint rows, const uint embedDim)
        {
            uint e = get_global_id(0);
            if (e >= embedDim) return;
            for (uint r = 0; r < rows; ++r)
                gradEmb[(uint)ids[r] * embedDim + e] += dX[r * embedDim + e];
        }

        /* gather_rows backward (the CLS pick): dX is a pre-zeroed [rows, hidden] buffer whose row
           b * seqLen receives dCls[b]. One work item per column, batch ascending. */
        __kernel void scatter_rows(
            __global const float* dCls, __global float* dX,
            const uint batchSize, const uint seqLen, const uint embedDim)
        {
            uint e = get_global_id(0);
            if (e >= embedDim) return;
            for (uint b = 0; b < batchSize; ++b)
                dX[(b * seqLen) * embedDim + e] = dCls[b * embedDim + e];
        }


        __kernel void fusion_forward(
            __global const float* S,
            __global const float* P,
            __global const float* W,
            __global float* F,
            __global float* D,
            const uint rows)
        {
            uint i = get_global_id(0);
            if (i >= rows) return;
            float num = 0.0f;
            float den = 0.0f;
            for (int k = 0; k < 4; k++) {
                float p = P[i * 4 + k];
                if (p == 0.0f) continue;
                float s = S[i * 4 + k];
                float w = W[k];
                num += p * s * w;
                den += p * w;
            }
            if (den == 0.0f) {
                F[i] = 0.0f;
                D[i] = 0.0f;
                return;
            }
            F[i] = num / den;
            D[i] = den;
        }

        __kernel void fusion_wgrad(
            __global const float* S,
            __global const float* P,
            __global const float* G,
            __global const float* F,
            __global const float* D,
            __global float* dW,
            const uint rows)
        {
            uint k = get_global_id(0);
            if (k >= 4) return;
            float acc = 0.0f;
            for (uint i = 0; i < rows; i++) {
                float p = P[i * 4 + k];
                if (p == 0.0f) continue;
                float s = S[i * 4 + k];
                acc += G[i] * p * (s - F[i]) / D[i];
            }
            dW[k] = acc;
        }

        """;
}
