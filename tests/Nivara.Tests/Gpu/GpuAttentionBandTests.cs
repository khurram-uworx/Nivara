using ILGPU;
using ILGPU.Runtime;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using Nivara.Samples;
using Nivara.Samples.Gpu;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Pins the banded GPU attention kernel (#447).
/// </summary>
/// <remarks>
/// <para>
/// Before #447, <c>BatchedAttention</c> swept all <c>seqLen</c> keys in each of its three passes
/// and used <c>band</c> only to overwrite an already-computed score with <c>-inf</c>, so the
/// sliding window bought no arithmetic. The band now narrows the loop bounds.
/// </para>
///
/// <para>
/// <b>Why there is no "banded run == dense run" gate.</b> The obvious regression check - run the
/// kernel with <c>band = -1</c> and with a real band and demand the same bits - asserts something
/// false: <c>band = -1</c> means attend to every key, so those two runs are different problems.
/// It is also not fixable by encoding the band in the mask, because the kernel's mask is indexed
/// <c>mask[b * seqLen + j]</c> with no query dimension, so a per-query band simply cannot be
/// expressed there - which is the whole reason <c>band</c> exists.
/// </para>
///
/// <para>
/// <b>The outputs are invariant under the loop-bound change, by construction.</b> <c>Keep</c>
/// already suppressed every out-of-band key before #447, so narrowing the loops cannot change a
/// single result: an out-of-band key contributed <c>exp(-inf - max) = 0</c> and then
/// <c>0 * v = ±0</c> to an accumulator that starts at <c>+0</c>, and IEEE round-to-nearest leaves
/// <c>+0</c> unchanged under either sign of zero. Verified rather than assumed: reverting the
/// kernel to the pre-#447 dense sweep leaves every test here green, because there is no
/// observable difference to find.
/// </para>
///
/// <para>
/// The consequence sets the shape of this fixture. The change is a pure performance change, so the
/// only way the new bounds can be <em>wrong</em> is by under-reaching - dropping a key that should
/// have been kept - and that is what the gates below attack. Each was checked against a mutation
/// to confirm it can actually fail:
/// </para>
///
/// <list type="number">
///   <item><description>
///     <c>band = -1</c> against a band spanning the sequence, bit for bit. The real regression
///     gate for the bounds computation, and the path DistilBERT, MiniLM, Laya and the
///     <c>--gemm-legs</c> baseline all run via <c>GpuBuffers.GlobalAttentionBand</c>. It cannot
///     fail on an under-reach, by the invariance above, so it is a bound-fidelity check rather
///     than the teeth of this fixture.
///   </description></item>
///   <item><description>
///     A degenerate band has a closed form: at <c>band = 0</c> the only visible key is
///     <c>j = qPos</c>, so the softmax has one term, <c>p = 1</c>, and the output row must be
///     <c>V[b, qPos]</c> exactly. Fails on an under-reach by one.
///   </description></item>
///   <item><description>
///     Fully-masked rows must be zeros, never NaN. Narrowing the band makes those rows more
///     common, so this is the contract #454 is about, and it is the only gate that fails when the
///     <c>max == -inf</c> guard is removed - the zero path and the live path are not otherwise
///     distinguishable.
///   </description></item>
///   <item><description>
///     The production CPU attention with <c>ModernBertMasks.Build</c> is the semantic reference for
///     a real band, at a tolerance, at <c>band = 2</c> and at <c>band = 1</c> where the bounds clip
///     at both sequence ends. Both fail on an under-reach.
///   </description></item>
/// </list>
///
/// <para>
/// Gates 1-3 are exact. Per AGENTS.md, a value that must be reproduced bit-for-bit is asserted
/// bit-for-bit: a tolerance would let a real band defect hide in the same slack. Gate 4 needs one
/// because <c>RowScore</c>'s scalar <c>acc +=</c> is not an FMA while the CPU dot product may be -
/// the same split <c>GpuElementwiseParityTests</c> documents for the two erf ports. Gate 4 is not
/// the only thing standing between a band bug and green; gates 2 and 3 are exact and have been
/// shown to fail.
/// </para>
///
/// <para>
/// <b>Coverage caveat.</b> Every test here needs a real OpenCL GPU, because the kernel under test
/// only exists once ILGPU has compiled it for the device. With no device the fixture
/// <c>Assert.Ignore</c>s and NUnit reports that as <em>skipped</em>, never as passed - an
/// unhostable run is a structural outcome, not a numeric one. Before this file there was no
/// automated test that ran this kernel at all: <c>GpuElementwiseParityTests</c> calls only
/// host-side scalar helpers and <c>SharedMemoryAllocationTests</c> reads IL, so the only real gate
/// was end-to-end <c>modernbert --gpu compare</c>, which needs both weights and a GPU.
/// </para>
/// </remarks>
[TestFixture]
public class GpuAttentionBandTests
{
    const int SeqLen = 8;
    const int NumHeads = 2;
    const int HeadDim = 8;
    const int ModelDim = NumHeads * HeadDim;
    const int Batch = 2;
    const float Scale = 0.5f;
    const float CpuParityTolerance = 1e-5f;

    static IlgpuRuntime? runtime;

    [OneTimeSetUp]
    public void RequireGpu()
    {
        try
        {
            runtime = new IlgpuRuntime();
            GpuBuffers.ValidateAttentionLocalMemory(runtime.Accelerator, HeadDim);
        }
        catch (Exception ex)
        {
            runtime = null;
            Assert.Ignore(
                "no OpenCL GPU is usable here, so the banded attention kernel cannot be compiled or " +
                $"run and this fixture asserts nothing: {ex.Message}");
        }
    }

    [OneTimeTearDown]
    public void ReleaseGpu() => runtime?.Dispose();

    /// <summary>Deterministic fixtures: a fixed xorshift stream, so a failure reproduces exactly.</summary>
    static float[] Random(int n, uint seed)
    {
        var values = new float[n];
        uint rng = seed;
        for (int i = 0; i < n; i++)
        {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            values[i] = (rng / (float)uint.MaxValue - 0.5f) * 0.2f;
        }
        return values;
    }

    /// <summary>A padding mask with the given number of leading valid keys, per batch element.</summary>
    static float[] PaddingMask(int validLength)
    {
        var mask = new float[Batch * SeqLen];
        for (int b = 0; b < Batch; b++)
            for (int j = 0; j < SeqLen; j++)
                mask[b * SeqLen + j] = j < validLength ? 1f : 0f;
        return mask;
    }

    /// <summary>All keys valid - isolates the band from the padding mask.</summary>
    static float[] AllValidMask() => new float[Batch * SeqLen].Select(_ => 1f).ToArray();

    /// <summary>
    /// Runs the real kernel on the GPU and reads the result back. Loads and launches
    /// <c>AttentionKernels.BatchedAttention</c> exactly as the encoder runners do, so this gates
    /// the shipped kernel rather than a host-side reimplementation of it.
    /// </summary>
    static float[] RunOnGpu(float[] q, float[] k, float[] v, float[] mask, int band, out string deviceName)
    {
        var gpu = runtime!;
        deviceName = gpu.DeviceName;

        using var qBuf = gpu.Allocate1D(q.Length);
        using var kBuf = gpu.Allocate1D(k.Length);
        using var vBuf = gpu.Allocate1D(v.Length);
        using var maskBuf = gpu.Allocate1D(mask.Length);
        using var outBuf = gpu.Allocate1D(q.Length);

        qBuf.CopyFromCPU(q);
        kBuf.CopyFromCPU(k);
        vBuf.CopyFromCPU(v);
        maskBuf.CopyFromCPU(mask);

        var kernel = gpu.Accelerator.LoadKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, ArrayView<float>, int, int, int, int, int, float>(
            AttentionKernels.BatchedAttention);

        kernel(gpu.Stream, GpuBuffers.Cfg1D(Batch * NumHeads * SeqLen, GpuBuffers.AttentionGroupSize),
            qBuf.View, kBuf.View, vBuf.View, maskBuf.View, outBuf.View,
            Batch, SeqLen, NumHeads, HeadDim, band, Scale);
        gpu.Synchronize();

        return outBuf.AsContiguous().GetAsArray();
    }

    /// <summary>
    /// The CPU equivalent, through the production kernels: <c>ModernBertMasks.Build</c> fuses the
    /// band and the padding mask into the additive <c>[L, L]</c> matrix <c>MultiHeadAttention</c>
    /// consumes, which is what the GPU kernel applies implicitly through <c>Keep</c>. One batch
    /// element at a time - <c>MultiHeadAttention</c> is unbatched, and the per-batch behaviour is
    /// what the mask vector encodes.
    /// </summary>
    static float[] RunOnCpu(float[] q, float[] k, float[] v, int band, int validLength)
    {
        var mask = ModernBertMasks.Build<float>(SeqLen, band, validLength);

        var output = new float[Batch * SeqLen * ModelDim];
        for (int b = 0; b < Batch; b++)
        {
            int offset = b * SeqLen * ModelDim;
            ReverseGradTensor<float> qb = Slice(q, offset);
            ReverseGradTensor<float> kb = Slice(k, offset);
            ReverseGradTensor<float> vb = Slice(v, offset);

            ReverseGradTensor<float> result =
                ReverseGradOperations.MultiHeadAttention(qb, kb, vb, NumHeads, Scale, mask);
            result.AsSpan().CopyTo(output.AsSpan(offset));
        }
        return output;
    }

    static ReverseGradTensor<float> Slice(float[] source, int offset)
        => ReverseGradTensor<float>.FromMatrix(
            source.Skip(offset).Take(SeqLen * ModelDim).ToArray(), SeqLen, ModelDim, requiresGrad: false);

    static void AssertBitIdentical(float[] expected, float[] actual, string label)
    {
        Assert.That(actual.Length, Is.EqualTo(expected.Length), $"{label}: length changed");
        for (int i = 0; i < expected.Length; i++)
        {
            int want = BitConverter.SingleToInt32Bits(expected[i]);
            int got = BitConverter.SingleToInt32Bits(actual[i]);
            if (want != got)
            {
                Assert.Fail(
                    $"{label}: element {i} differs. expected {expected[i]:R} (0x{want:X8}), " +
                    $"got {actual[i]:R} (0x{got:X8}).");
            }
        }
    }

    static void AssertAgreesWithCpu(float[] q, float[] k, float[] v, int band, int validLength)
    {
        float[] gpu = RunOnGpu(q, k, v, PaddingMask(validLength), band, out _);
        float[] cpu = RunOnCpu(q, k, v, band, validLength);

        float worst = 0f;
        int worstIndex = 0;
        for (int i = 0; i < cpu.Length; i++)
        {
            float diff = Math.Abs(gpu[i] - cpu[i]);
            if (diff > worst) { worst = diff; worstIndex = i; }
        }

        Assert.That(worst, Is.LessThan(CpuParityTolerance),
            $"banded GPU attention (band={band}, validLength={validLength}) deviates from the CPU " +
            $"MultiHeadAttention by {worst:E3} at index {worstIndex} (gpu {gpu[worstIndex]:R} vs cpu " +
            $"{cpu[worstIndex]:R}), above the {CpuParityTolerance:E3} bound. The CPU dot product may use " +
            "FMA where RowScore does not, which this bound accommodates.");
    }

    /// <summary>
    /// The global-attention path, which is what DistilBERT, MiniLM, Laya and the
    /// <c>--gemm-legs</c> baseline all run. <c>band = -1</c> must leave <c>[0, seqLen-1]</c>
    /// untouched, and a band wide enough to span the sequence is the same range reached by
    /// arithmetic instead of by the sentinel - so any difference is a defect in the bounds
    /// computation itself, and this gate is bit-exact by construction: both runs execute the same
    /// loops over the same values in the same order.
    /// </summary>
    [Test]
    public void BatchedAttention_GlobalBand_MatchesTheFullRangeBitForBit()
    {
        var q = Random(Batch * SeqLen * ModelDim, 0x9E3779B9);
        var k = Random(Batch * SeqLen * ModelDim, 0x85EBCA6B);
        var v = Random(Batch * SeqLen * ModelDim, 0xC2B2AE35);
        var mask = AllValidMask();

        float[] global = RunOnGpu(q, k, v, mask, GpuBuffers.GlobalAttentionBand, out string device);
        float[] wideBand = RunOnGpu(q, k, v, mask, SeqLen, out _);

        Assert.That(device, Is.Not.Empty, "no OpenCL device name was reported");
        AssertBitIdentical(global, wideBand, $"band=-1 vs a band spanning the sequence, on {device}");
    }

    /// <summary>
    /// The closed-form gate for the band arithmetic. At <c>band = 0</c> a query's only visible key
    /// is itself, so <c>max</c> is that one score, <c>sum</c> is <c>exp(0) = 1</c>, <c>p</c> is
    /// exactly 1, and every output lane is the corresponding V element - for every batch element,
    /// query position and head. A bound that clips too far in, one that includes a neighbour, or a
    /// <c>Keep</c> that stops testing the band all change the result, and none of them can hide
    /// behind a tolerance here.
    /// </summary>
    [Test]
    public void BatchedAttention_BandZero_EqualsTheValueRowBitForBit()
    {
        var q = Random(Batch * SeqLen * ModelDim, 0x27D4EB2F);
        var k = Random(Batch * SeqLen * ModelDim, 0x165667B1);
        var v = Random(Batch * SeqLen * ModelDim, 0xD3A2646C);
        var mask = AllValidMask();

        float[] output = RunOnGpu(q, k, v, mask, 0, out _);

        for (int b = 0; b < Batch; b++)
        {
            for (int qPos = 0; qPos < SeqLen; qPos++)
            {
                for (int d = 0; d < ModelDim; d++)
                {
                    // The kernel's accumulator starts at +0, so the expected value is 0f + v:
                    // identical to v for every input except a -0 element, which is the one case
                    // where the two differ and where +0 is what the kernel must produce.
                    float expected = 0f + v[(b * SeqLen + qPos) * ModelDim + d];
                    float actual = output[(b * SeqLen + qPos) * ModelDim + d];
                    AssertBitIdentical([expected], [actual],
                        $"band=0, row (b={b}, qPos={qPos}), lane {d} (must be V[b, qPos, {d}])");
                }
            }
        }
    }

    /// <summary>
    /// The fully-masked-row contract, which <b>narrows</b> the band makes more reachable rather
    /// than less. A query at or past <c>validLength + band</c> has every key either outside its
    /// band or masked, so the max pass leaves <c>max</c> at <c>-inf</c> and the kernel must write
    /// zeros - never NaN. This mirrors the CPU safe-softmax clamp.
    /// </summary>
    [Test]
    public void BatchedAttention_RowOutsideTheBand_WritesZerosAndNotNaN()
    {
        const int band = 1;
        const int validLength = 3;

        var q = Random(Batch * SeqLen * ModelDim, 0x5BD1E995);
        var k = Random(Batch * SeqLen * ModelDim, 0x1F83D9AB);
        var v = Random(Batch * SeqLen * ModelDim, 0x9BE056A3);
        float[] output = RunOnGpu(q, k, v, PaddingMask(validLength), band, out _);

        int emptyRows = 0;
        int liveRows = 0;
        for (int b = 0; b < Batch; b++)
        {
            for (int qPos = 0; qPos < SeqLen; qPos++)
            {
                int jFirst = Math.Max(0, qPos - band);
                int jLast = Math.Min(SeqLen - 1, qPos + band);
                int visible = Enumerable.Range(jFirst, jLast - jFirst + 1).Count(j => j < validLength);
                int baseIndex = (b * SeqLen + qPos) * ModelDim;

                if (visible > 0)
                {
                    // A row that does have a visible key must be a real result, not the guard's
                    // zeros: if the narrowed bounds under-reach, this is where it shows.
                    liveRows++;
                    Assert.That(float.IsFinite(output[baseIndex]), Is.True,
                        $"row (b={b}, qPos={qPos}) is non-finite although {visible} of its band keys " +
                        "are valid, so the narrowed range under-reaches.");
                    continue;
                }

                emptyRows++;
                for (int d = 0; d < ModelDim; d++)
                {
                    float value = output[baseIndex + d];
                    Assert.That(float.IsNaN(value), Is.False,
                        $"row (b={b}, qPos={qPos}) lane {d} is NaN; a fully-masked row must write " +
                        "zeros. Narrowing the band makes this row type more common, not less.");
                    Assert.That(value, Is.EqualTo(0f),
                        $"row (b={b}, qPos={qPos}) lane {d} must be exactly 0f, got {value:R}.");
                }
            }
        }

        Assert.That(emptyRows, Is.GreaterThan(0),
            $"no fully-masked row was reachable at validLength={validLength}, band={band}, so this " +
            "test asserted nothing; the guard it exists to check never ran.");
        Assert.That(liveRows, Is.GreaterThan(0),
            $"no live row was reachable at validLength={validLength}, band={band}, so this test only " +
            "exercised the zero path.");
    }

    /// <summary>
    /// A real band against the production CPU attention and the production dense mask: the
    /// semantic gate that the band means what <c>ModernBertMasks.Build</c> says it means.
    /// </summary>
    [Test]
    public void BatchedAttention_BandedRun_AgreesWithCpuMultiHeadAttention()
    {
        var q = Random(Batch * SeqLen * ModelDim, 0xC0AC29B7);
        var k = Random(Batch * SeqLen * ModelDim, 0x3F84D5B5);
        var v = Random(Batch * SeqLen * ModelDim, 0xB5470917);

        AssertAgreesWithCpu(q, k, v, band: 2, validLength: 6);
    }

    /// <summary>
    /// The same gate at <c>band = 1</c>, where <c>Max(0, qPos - band)</c> clips at the start of the
    /// sequence and <c>Min(seqLen - 1, qPos + band)</c> clips at the end. The CPU mask builder
    /// applies the same pair of clamps, so agreement here means the two clipping conventions
    /// match. With SeqLen 8, a band of 1 is also narrower than <c>headDim</c> 8, so the window is
    /// smaller than the score dot product it replaces.
    /// </summary>
    [Test]
    public void BatchedAttention_BandClippedAtBothSequenceEnds_AgreesWithCpuMultiHeadAttention()
    {
        var q = Random(Batch * SeqLen * ModelDim, 0x1B873593);
        var k = Random(Batch * SeqLen * ModelDim, 0xCC9E2D51);
        var v = Random(Batch * SeqLen * ModelDim, 0xE6546B64);

        AssertAgreesWithCpu(q, k, v, band: 1, validLength: 8);
    }
}
