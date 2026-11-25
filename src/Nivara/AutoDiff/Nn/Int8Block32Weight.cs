namespace Nivara.AutoDiff.Nn;

/// <summary>
/// A CPU-side INT8 block-32 quantized weight payload (format <c>SilkInt8Block32 v1</c>):
/// symmetric per-block quantization, no zero point, one F32 scale per 32-element block along
/// the reduction (input) dimension. Weights are stored transposed as <c>[inCols, outCols]</c>
/// (row-major) so a single-row activation GEMV reads contiguous weight blocks.
/// </summary>
/// <remarks>
/// This is an inference-only carrier. It does not participate in the AutoDiff graph and is
/// never a <see cref="ReverseGradTensor{T}"/>. Keep in sync with <c>SilkInt8Block32 v1</c>.
/// </remarks>
public sealed class Int8Block32Weight
{
    /// <summary>Block size along the reduction dimension.</summary>
    public const int BlockSize = 32;

    /// <summary>The quantized weights, length <c>InCols * OutCols</c>, layout <c>[inCols, outCols]</c>.</summary>
    public required sbyte[] Data { get; init; }

    /// <summary>
    /// Per-output-column, per-block scales, length <c>OutCols * NumBlocks</c>, layout
    /// <c>[outCols, numBlocks]</c> so a fixed output column has contiguous scales.
    /// </summary>
    public required float[] Scales { get; init; }

    /// <summary>Reduction (input) dimension.</summary>
    public required int InCols { get; init; }

    /// <summary>Output dimension.</summary>
    public required int OutCols { get; init; }

    /// <summary>Number of blocks along the reduction dimension.</summary>
    public int NumBlocks => (InCols + BlockSize - 1) / BlockSize;

    /// <summary>Validates shape invariants.</summary>
    public void Validate()
    {
        if (InCols <= 0) throw new ArgumentOutOfRangeException(nameof(InCols));
        if (OutCols <= 0) throw new ArgumentOutOfRangeException(nameof(OutCols));
        if (Data.Length != InCols * OutCols)
            throw new ArgumentException($"Data length {Data.Length} != InCols*OutCols {InCols * OutCols}.", nameof(Data));
        if (Scales.Length != OutCols * NumBlocks)
            throw new ArgumentException($"Scales length {Scales.Length} != OutCols*NumBlocks {OutCols * NumBlocks}.", nameof(Scales));
    }
}
