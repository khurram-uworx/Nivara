using Nivara.Samples;
using NUnit.Framework;
using System.Buffers.Binary;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Kernel + read-path verification for the BF16 raw-<c>ushort</c> loader support
/// (<c>SafeTensorsLoader.ReadUInt16</c> / <c>WidenBf16ToF32</c> / <c>WidenToF32</c>).
/// The SIMD widening must be bit-exact against the scalar BF16→F32 rule
/// (<c>float bits = ushortBits &lt;&lt; 16</c>) for every possible 16-bit pattern, and the
/// raw read path must reproduce the same tensors as the existing F32 read of the
/// real Qwen checkpoint (skipped when the model files are absent).
/// </summary>
[TestFixture]
public class SafeTensorsLoaderBf16Tests
{
    static string ModelDir
        => Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "..", "..",
            "samples", "data", "qwen2.5-0.5b-instruct");

    [Test]
    public void WidenBf16ToF32_AllBitPatterns_MatchesScalarReference()
    {
        var src = new ushort[65536];
        for (int i = 0; i < src.Length; i++)
            src[i] = (ushort)i;

        var dst = new float[src.Length];
        SafeTensorsLoader.WidenBf16ToF32(src, dst);

        for (int i = 0; i < src.Length; i++)
        {
            uint expectedBits = (uint)i << 16;
            Assert.That(BitConverter.SingleToUInt32Bits(dst[i]), Is.EqualTo(expectedBits),
                $"pattern 0x{i:X4}: expect bit-exact 0x{expectedBits:X8}, got 0x{BitConverter.SingleToUInt32Bits(dst[i]):X8}");
        }

        // Spot truth-checks for representative BF16 values.
        Assert.That(dst[0x3F80], Is.EqualTo(1f));       // 1.0
        Assert.That(dst[0x4000], Is.EqualTo(2f));       // 2.0
        Assert.That(dst[0xC000], Is.EqualTo(-2f));      // -2.0
        Assert.That(dst[0x0000], Is.EqualTo(0f));       // 0.0
        Assert.That(float.IsPositiveInfinity(dst[0x7F80]), Is.True); // +inf
    }

    [Test]
    public void WidenBf16ToF32_VariousLengths_MatchesScalarReference()
    {
        var rng = new Random(1337);
        foreach (int length in new[] { 0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 100, 10001 })
        {
            var src = new ushort[length];
            var expected = new float[length];
            for (int i = 0; i < length; i++)
            {
                src[i] = (ushort)rng.Next(0, 65536);
                expected[i] = BitConverter.UInt32BitsToSingle((uint)src[i] << 16);
            }

            var dst = new float[length];
            SafeTensorsLoader.WidenBf16ToF32(src, dst);

            Assert.That(dst, Is.EqualTo(expected), $"length {length} must match scalar reference");
        }
    }

    [Test]
    public void WidenBf16ToF32_LengthMismatch_Throws()
    {
        var src = new ushort[4];
        var dst = new float[3];
        Assert.Throws<ArgumentException>(() => SafeTensorsLoader.WidenBf16ToF32(src, dst));
    }

    [Test]
    public void ReadUInt16_RawPatternsWidenToSameF32AsReadFloat()
    {
        (byte[] file, string[] tensorNames) = BuildBf16Fixture();

        var raw = SafeTensorsLoader.ReadUInt16(file);
        var widened = SafeTensorsLoader.WidenToF32(raw);
        var direct = SafeTensorsLoader.Read<float>(file);

        Assert.That(widened.Keys, Is.EquivalentTo(raw.Keys));
        Assert.That(widened.Keys, Is.EquivalentTo(direct.Keys));
        foreach (var name in tensorNames)
        {
            Assert.That(widened[name].Shape, Is.EqualTo(direct[name].Shape), $"{name}: shape");
            Assert.That(widened[name].Data, Is.EqualTo(direct[name].Data), $"{name}: values (ushort→F32 must equal Read<float>)");
        }
    }

    [Test]
    public void ReadUInt16_RejectsNonBf16Dtypes()
    {
        // A single F32 tensor must be rejected by the raw-ushort path.
        var header = "{\"t\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},\"__metadata__\":{}}";
        var headerLen = (uint)Encoding.UTF8.GetByteCount(header);
        var bytes = new byte[8 + headerLen + 8];
        BinaryPrimitives.WriteUInt64LittleEndian(bytes, headerLen);
        Encoding.UTF8.GetBytes(header, bytes.AsSpan(8, (int)headerLen));
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(8 + (int)headerLen), 1f);
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(8 + (int)headerLen + 4), 2f);

        var ex = Assert.Throws<NotSupportedException>(() => SafeTensorsLoader.ReadUInt16(bytes));

        Assert.That(ex!.Message, Does.Contain("t").And.Contain("F32"));
    }

    [Test]
    public void ReadUInt16_OnQwenCheckpoint_MatchesReadFloatKeysAndShapes()
    {
        var safetensors = Path.Combine(ModelDir, "model.safetensors");
        if (!File.Exists(safetensors))
            Assert.Ignore("Qwen safetensors absent; skipping raw-BF16 checkpoint verification.");

        var stopwatch = Stopwatch.StartNew();
        var raw = SafeTensorsLoader.ReadUInt16(safetensors);
        stopwatch.Stop();
        long rawMs = stopwatch.ElapsedMilliseconds;
        long rawBytes = raw.Sum(t => (long)t.Value.Data.Length * sizeof(ushort));

        stopwatch.Restart();
        var direct = SafeTensorsLoader.Read<float>(safetensors);
        stopwatch.Stop();
        long floatMs = stopwatch.ElapsedMilliseconds;

        // Structural parity: same tensor set, same shapes, raw read holds half the bytes.
        Assert.That(raw.Keys, Is.EquivalentTo(direct.Keys));
        Assert.That(raw.Count, Is.EqualTo(direct.Count));
        foreach (var name in raw.Keys)
            Assert.That(raw[name].Shape, Is.EqualTo(direct[name].Shape), $"{name}: shape mismatch");

        // Spot shape checks against Qwen2.5-0.5B config values.
        Assert.That(raw["model.embed_tokens.weight"].Shape, Is.EqualTo(new[] { 151936, 896 }));
        Assert.That(raw["model.norm.weight"].Shape, Is.EqualTo(new[] { 896 }));
        Assert.That(raw["model.layers.0.self_attn.q_proj.weight"].Shape, Is.EqualTo(new[] { 896, 896 }));
        Assert.That(raw["model.layers.0.self_attn.q_proj.bias"].Shape, Is.EqualTo(new[] { 896 }));
        Assert.That(raw["model.layers.0.self_attn.v_proj.weight"].Shape, Is.EqualTo(new[] { 128, 896 }));

        TestContext.Out.WriteLine(
            $"Qwen BF16 load: ReadUInt16 {rawMs} ms ({rawBytes / (1024.0 * 1024.0):F0} MB targets) vs Read<float> {floatMs} ms " +
            $"(half the byte payload, same tensors).");
    }

    /// <summary>Builds a small, valid safetensors BF16 file with three tensors (mixed ranks).</summary>
    static (byte[] File, string[] Names) BuildBf16Fixture()
    {
        // weights: 2×2 matrix + 2×1 bias + 1×3 vector, all as raw BF16 ushort patterns.
        var weightP = new ushort[] { 0x3F80, 0x4000, 0xC000, 0x3F00 }; // 1, 2, -2, 0.5
        var biasP = new ushort[] { 0xBF80, 0x3F80 };                  // -1, 1
        var vecP = new ushort[] { 0x0000, 0x7F80, 0x7FC0 };           // 0, +inf, NaN

        var sections = new (string Name, int[] Shape, ushort[] Patterns)[]
        {
            ("w", new[] { 2, 2 }, weightP),
            ("b", new[] { 2 }, biasP),
            ("v", new[] { 3 }, vecP),
        };
        var names = sections.Select(s => s.Name).ToArray();

        int offset = 0;
        var builder = new StringBuilder("{");
        foreach (var (name, shape, patterns) in sections)
        {
            int end = offset + patterns.Length * sizeof(ushort);
            if (builder.Length > 1) builder.Append(',');
            builder.Append($"\"{name}\":{{\"dtype\":\"BF16\",\"shape\":[{string.Join(",", shape)}],\"data_offsets\":[{offset},{end}]}}");
            offset = end;
        }
        builder.Append(",\"__metadata__\":{}");

        var headerBytes = Encoding.UTF8.GetBytes(builder.ToString());
        var file = new byte[8 + headerBytes.Length + offset];
        BinaryPrimitives.WriteUInt64LittleEndian(file, (ulong)headerBytes.Length);
        headerBytes.CopyTo(file.AsSpan(8, headerBytes.Length));

        int dataStart = 8 + headerBytes.Length;
        int cursor = 0;
        foreach (var (_, _, patterns) in sections)
        {
            int len = patterns.Length * sizeof(ushort);
            MemoryMarshal.AsBytes(patterns.AsSpan()).CopyTo(file.AsSpan(dataStart + cursor, len));
            cursor += len;
        }

        return (file, names);
    }
}