namespace Nivara.Samples.Gpu;

/// <summary>
/// The OpenCL C kernels the BERT-family encoder needs, authored once and shared by the sample
/// runner (<c>--silk</c>) and the performance gate (<c>--silk-kernels</c>). The host side lives
/// in <see cref="SilkRuntime"/>; this type is pure data and arithmetic-free geometry so it can
/// be unit-checked without a device.
///
/// <para>
/// Each kernel is a port of the ILGPU kernel of the same shape
/// (<see cref="GemmKernels"/>, <see cref="AttentionKernels"/>, <see cref="ElementwiseKernels"/>)
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

    /// <summary>Compile-time bound on the attention kernel's per-thread accumulator tile.</summary>
    public const int MaxHeadDim = 64;

    /// <summary>Band sentinel meaning "attend to every non-suppressed key" (global attention).</summary>
    public const int GlobalAttentionBand = -1;

    public const string EmbedSum = "embed_sum";
    public const string GatherRows = "gather_rows";
    public const string LayerNormResidual = "layer_norm_residual";
    public const string LayerNorm = "layer_norm";
    public const string Add = "add";
    public const string AddBias = "add_bias";
    public const string SplitColumns = "split_columns";
    public const string Rotary = "rotary";
    public const string GeGlu = "geglu";
    public const string Gemm = "gemm_row4";
    public const string GemmBias = "gemm_row4_bias";
    public const string GemmRelu = "gemm_row4_relu";
    public const string GemmGelu = "gemm_row4_gelu";
    public const string GemmQkv = "gemm_row4_qkv";
    public const string Attention = "batched_attention";

    /// <summary>Every kernel the translation unit defines, in build order.</summary>
    public static readonly string[] KernelNames =
    [
        EmbedSum, GatherRows, LayerNormResidual, LayerNorm, Add, AddBias, SplitColumns, Rotary, GeGlu,
        Gemm, GemmBias, GemmRelu, GemmGelu, GemmQkv, Attention,
    ];

    /// <summary>Global work size (dim 0) for a GEMM producing <paramref name="aRows"/> rows.</summary>
    public static int GemmGlobalRows(int aRows) => RoundUp(aRows, GemmTileSize);

    /// <summary>Global work size (dim 1) for a GEMM producing <paramref name="bCols"/> columns.
    /// A work-group is 16 threads wide here but covers <see cref="GemmTileCols"/> output columns
    /// (each thread takes <see cref="GemmBlockCols"/> of them), so the group count is
    /// <c>ceil(bCols / GemmTileCols)</c> and the thread count is that times
    /// <see cref="GemmTileSize"/>. Returning <c>RoundUp(bCols, GemmTileCols)</c> instead
    /// over-launches by <see cref="GemmBlockCols"/>×: the surplus groups run the whole K-loop and
    /// then discard every result (the <c>outCol &lt; bCols</c> guard), which is pure waste.</summary>
    public static int GemmGlobalCols(int bCols) => RoundUp(bCols, GemmTileCols) / GemmBlockCols;

    /// <summary>Global work size for a flat 1-D launch guarded by an <c>idx &gt;= n</c> early return.</summary>
    public static int LinearGlobal(int total) => RoundUp(Math.Max(1, total), LinearGroupSize);

    /// <summary>Global work size for the attention launch (rounded to whole work-groups).</summary>
    public static int AttentionGlobal(int total) => RoundUp(Math.Max(1, total), AttentionGroupSize);

    static int RoundUp(int value, int multiple) => ((value + multiple - 1) / multiple) * multiple;

    public const string Source = """
        #pragma OPENCL FP_CONTRACT OFF

        #define TILE 16
        #define BLOCK 4
        #define TILECOLS 64
        #define ROWSTRIDE 80
        #define MAXHEADDIM 64
        #define ATTNGROUP 64

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
           cosTab/sinTab are not named cos/sin: those are OpenCL builtins. */
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
            uint half = headDim / 2u;
            uint idx = get_global_id(0);
            if (idx >= rows * numHeads * half) return;
            uint i = idx % half;
            uint rem = idx / half;
            uint h = rem % numHeads;
            uint row = rem / numHeads;
            uint t = (uint)posIds[row] * half + i;
            float c = cosTab[t];
            float s = sinTab[t];
            uint b = row * (numHeads * headDim) + h * headDim;
            float x0 = x[b + i];
            float x1 = x[b + i + half];
            y[b + i] = x0 * c - x1 * s;
            y[b + i + half] = x0 * s + x1 * c;
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

        /* Shared accumulation for the Row4 GEMM family, including the bias-free store.
           Deliberately one macro: the byte-identity claim rests on every variant adding the
           same f32 values to the same accumulator in the same order, so the text must not
           be able to drift between them. The store after the macro is the only difference.

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
                uint aCol = k0 + y; \
                uint bRow = k0 + x; \
                smem[x + y * TILE] = (outRow < aRows && aCol < aCols) ? a[outRow * aCols + aCol] : 0.0f; \
                for (uint w = 0; w < BLOCK; ++w) { \
                    uint outCol = colBase + y * BLOCK + w; \
                    smem[x + (TILE + y * BLOCK + w) * TILE] = \
                        (bRow < aCols && outCol < bCols) ? b[bRow * bCols + outCol] : 0.0f; \
                } \
                barrier(CLK_LOCAL_MEM_FENCE); \
                for (uint k = 0; k < TILE; ++k) { \
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

            float mx = -INFINITY;
            for (int j = jFirst; j <= jLast; ++j)
            {
                float s = attn_score(q, k, qRow, b * seqLen + (uint)j, dOffset, D, headDim, scale);
                if (!attn_keep(mask, maskBase, qp, j, band)) s = -INFINITY;
                if (s > mx) mx = s;
            }

            if (mx == -INFINITY)
            {
                for (uint d = 0; d < headDim; ++d)
                    attnOut[outBase + dOffset + d] = 0.0f;
                return;
            }

            float sum = 0.0f;
            for (int j = jFirst; j <= jLast; ++j)
            {
                float s = attn_score(q, k, qRow, b * seqLen + (uint)j, dOffset, D, headDim, scale);
                if (!attn_keep(mask, maskBase, qp, j, band)) s = -INFINITY;
                sum += exp(s - mx);
            }

            uint lane = get_local_id(0);
            for (uint d = 0; d < headDim; ++d)
                acc[d * MAXHEADDIM + lane] = 0.0f;

            for (int j = jFirst; j <= jLast; ++j)
            {
                float s = attn_score(q, k, qRow, b * seqLen + (uint)j, dOffset, D, headDim, scale);
                if (!attn_keep(mask, maskBase, qp, j, band)) s = -INFINITY;
                float p = exp(s - mx) / sum;
                uint vBase = (b * seqLen + (uint)j) * D + dOffset;
                for (uint d = 0; d < headDim; ++d)
                    acc[d * MAXHEADDIM + lane] += p * v[vBase + d];
            }

            for (uint d = 0; d < headDim; ++d)
                attnOut[outBase + dOffset + d] = acc[d * MAXHEADDIM + lane];
        }
        """;
}
