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