using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace Nivara.Samples.Gpu;

/// <summary>
/// Flat f32 elementwise / row-reduce kernels for the DistilBERT GPU forward
/// (docs/BERT-GPU.md §3): embedding gather, bias and residual adds, LayerNorm
/// (per-row mean/variance + affine), exact-GELU via a direct port of the CPU
/// GradKernels.Erf A–S 7.1.26 polynomial (XMath.Exp replaces Math.Exp — the only
/// divergence is the erf evaluation; ILGPU core/Algorithms has no erf), and ReLU.
/// All kernels are implicitly-grouped 1D kernels over flattened buffers.
/// </summary>
internal static class ElementwiseKernels
{
    /// <summary>out[i, :] = table[ids[i], :] for a row-major table [nRows, hidden] and a flat int id per row.</summary>
    public static void Gather(ArrayView<float> table, ArrayView<int> ids, ArrayView<float> output, int hidden)
    {
        int idx = Grid.GlobalIndex.X;
        if (idx >= output.Length) return;
        int row = idx / hidden;
        int col = idx - row * hidden;
        output[idx] = table[ids[row] * hidden + col];
    }

    /// <summary>y[r, c] = x[r, c] + bias[c] on a flattened [rows, cols] buffer.</summary>
    public static void AddBias(ArrayView<float> x, ArrayView<float> bias, ArrayView<float> y, int rows, int cols)
    {
        int idx = Grid.GlobalIndex.X;
        if (idx >= rows * cols) return;
        y[idx] = x[idx] + bias[idx % cols];
    }

    /// <summary>y[i] = a[i] + b[i].</summary>
    public static void Add(ArrayView<float> a, ArrayView<float> b, ArrayView<float> y)
    {
        int idx = Grid.GlobalIndex.X;
        if (idx >= y.Length) return;
        y[idx] = a[idx] + b[idx];
    }

    /// <summary>
    /// y = LayerNorm(a + b) fused into one launch: the LN row-reduction reads both
    /// inputs, so the residual add no longer dispatches a separate kernel. Same numerics
    /// as <see cref="LayerNorm1D"/> over the sum (mean of (a+b), var of (a+b), affine
    /// gamma/beta). y may alias a or b — each output element's inputs are read before its
    /// write, all by the owning thread.
    /// </summary>
    public static void LayerNormResidual1D(
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> gamma,
        ArrayView<float> beta,
        ArrayView<float> y,
        int rows,
        int cols,
        float eps)
    {
        int r = Grid.GlobalIndex.X;
        if (r >= rows) return;
        int off = r * cols;

        float sum = 0f;
        for (int c = 0; c < cols; c++)
            sum += a[off + c] + b[off + c];
        float mean = sum / cols;

        float sumSq = 0f;
        for (int c = 0; c < cols; c++)
        {
            float d = a[off + c] + b[off + c] - mean;
            sumSq += d * d;
        }

        float invStd = 1f / XMath.Sqrt(sumSq / cols + eps);
        for (int c = 0; c < cols; c++)
            y[off + c] = (a[off + c] + b[off + c] - mean) * invStd * gamma[c] + beta[c];
    }

    /// <summary>
    /// y[r, :] = wordEmb[ids[r], :] + posEmb[posIds[r], :] + (includeTt != 0 ? tokenType[:, 0] : 0)
    /// — the embedding gather/sum path fused into one launch (word + position gathers, the
    /// pair-wise add, and the optional token-type row-0 bias; ILGPU views can't be null, so
    /// the tokenType view is always passed and the include flag gates the read).
    /// </summary>
    public static void EmbeddingSum(
        ArrayView<int> ids,
        ArrayView<int> posIds,
        ArrayView<float> wordEmb,
        ArrayView<float> posEmb,
        ArrayView<float> tokenType,
        ArrayView<float> y,
        int hidden,
        int includeTt)
    {
        int idx = Grid.GlobalIndex.X;
        if (idx >= y.Length) return;
        int row = idx / hidden;
        int col = idx - row * hidden;
        float v = wordEmb[ids[row] * hidden + col] + posEmb[posIds[row] * hidden + col];
        if (includeTt != 0) v += tokenType[col];
        y[idx] = v;
    }

    /// <summary>
    /// Per-row layer norm over the trailing dimension: y = (x - mean) / sqrt(var + eps) * gamma + beta.
    /// Matches the CPU LayerNormKernel.ForwardInference numerics (mean = sum/cols,
    /// var = sum((x-mean)^2)/cols, affine gamma/beta) with the config eps.
    /// </summary>
    public static void LayerNorm1D(
        ArrayView<float> x,
        ArrayView<float> gamma,
        ArrayView<float> beta,
        ArrayView<float> y,
        int rows,
        int cols,
        float eps)
    {
        int r = Grid.GlobalIndex.X;
        if (r >= rows) return;
        int off = r * cols;

        float sum = 0f;
        for (int c = 0; c < cols; c++)
            sum += x[off + c];
        float mean = sum / cols;

        float sumSq = 0f;
        for (int c = 0; c < cols; c++)
        {
            float d = x[off + c] - mean;
            sumSq += d * d;
        }

        float invStd = 1f / XMath.Sqrt(sumSq / cols + eps);
        for (int c = 0; c < cols; c++)
            y[off + c] = (x[off + c] - mean) * invStd * gamma[c] + beta[c];
    }

    /// <summary>
    /// Exact GELU via the A–S 7.1.26 erf port: the same polynomial and argument
    /// normalization (x * 1/sqrt(2)) as CPU GradKernels.GeluExact/Erf.
    /// </summary>
    public static void Gelu(ArrayView<float> x, ArrayView<float> y)
    {
        int idx = Grid.GlobalIndex.X;
        if (idx >= y.Length) return;
        float v = x[idx];
        float z = v * 0.7071067811865475f;
        float az = XMath.Abs(z);
        float t = 1f / (1f + 0.3275911f * az);
        float p = 1.061405429f * t - 1.453152027f;
        p = p * t + 1.421413741f;
        p = p * t - 0.284496736f;
        p = p * t + 0.254829592f;
        float erf = 1f - p * t * XMath.Exp(-az * az);
        if (z < 0f) erf = -erf;
        y[idx] = 0.5f * v * (1f + erf);
    }

    /// <summary>y[i] = max(x[i], 0).</summary>
    public static void Relu(ArrayView<float> x, ArrayView<float> y)
    {
        int idx = Grid.GlobalIndex.X;
        if (idx >= y.Length) return;
        float v = x[idx];
        y[idx] = v > 0f ? v : 0f;
    }
}