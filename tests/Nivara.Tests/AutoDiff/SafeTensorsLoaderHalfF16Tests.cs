using Nivara.AI;
using NUnit.Framework;
using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// SafeTensors read-path coverage for F16/Half and BF16, including the raw
/// <c>Read&lt;BFloat16&gt;()</c> reinterpret fast path (issue #1021).
/// </summary>
[TestFixture]
public class SafeTensorsLoaderHalfF16Tests
{
    static byte[] BuildFixture(string dtype, int[] shape, byte[] data)
    {
        string shapeStr = string.Join(",", shape);
        string header = $"{{\"t\":{{\"dtype\":\"{dtype}\",\"shape\":[{shapeStr}],\"data_offsets\":[0,{data.Length}]}}}}";
        var headerBytes = Encoding.UTF8.GetBytes(header);
        var file = new byte[8 + headerBytes.Length + data.Length];
        BinaryPrimitives.WriteUInt64LittleEndian(file, (ulong)headerBytes.Length);
        headerBytes.CopyTo(file.AsSpan(8));
        data.CopyTo(file.AsSpan(8 + headerBytes.Length));
        return file;
    }

    [Test]
    public void ReadHalf_OnF16Fixture_EqualsExpectedHalfValues()
    {
        var expected = new Half[] { (Half)1.5f, (Half)2.0f, (Half)(-1.25f) };
        var data = new byte[expected.Length * 2];
        MemoryMarshal.AsBytes(expected.AsSpan()).CopyTo(data);

        var file = BuildFixture("F16", new[] { expected.Length }, data);
        var read = SafeTensorsLoader.Read<Half>(file);

        Assert.That(read["t"].Data, Is.EqualTo(expected));
        Assert.That(read["t"].Shape, Is.EqualTo(new[] { expected.Length }));
    }

    [Test]
    public void ReadFloat_OnF16Fixture_WidensViaHalf()
    {
        var halfValues = new Half[] { (Half)1.5f, (Half)2.0f };
        var data = new byte[halfValues.Length * 2];
        MemoryMarshal.AsBytes(halfValues.AsSpan()).CopyTo(data);

        var file = BuildFixture("F16", new[] { halfValues.Length }, data);
        var read = SafeTensorsLoader.Read<float>(file);

        Assert.That(read["t"].Data, Is.EqualTo(new[] { 1.5f, 2.0f }));
    }

    [Test]
    public void ReadBFloat16_OnBf16Fixture_RawReinterpretMatchesPatterns()
    {
        // Raw ushort patterns already in BFloat16 memory layout; Read<BFloat16>() must
        // return them unchanged (no BF16 -> F32 -> BF16 round trip).
        var patterns = new ushort[] { 0x3F80, 0x4000, 0xC000, 0x3F00 };
        var data = new byte[patterns.Length * sizeof(ushort)];
        MemoryMarshal.AsBytes(patterns.AsSpan()).CopyTo(data);

        var file = BuildFixture("BF16", new[] { patterns.Length }, data);
        var read = SafeTensorsLoader.Read<BFloat16>(file);

        var resultPatterns = MemoryMarshal.Cast<BFloat16, ushort>(read["t"].Data.AsSpan()).ToArray();
        Assert.That(resultPatterns, Is.EqualTo(patterns), "Read<BFloat16> must raw-reinterpret BF16 bytes");
        Assert.That(read["t"].Shape, Is.EqualTo(new[] { patterns.Length }));
    }

    [Test]
    public void ReadFloat_OnBf16Fixture_MatchesWidenBf16ToF32()
    {
        var patterns = new ushort[] { 0x3F80, 0x4000, 0xC000, 0x3F00 };
        var data = new byte[patterns.Length * sizeof(ushort)];
        MemoryMarshal.AsBytes(patterns.AsSpan()).CopyTo(data);

        var file = BuildFixture("BF16", new[] { patterns.Length }, data);
        var read = SafeTensorsLoader.Read<float>(file);

        var expected = new float[patterns.Length];
        SafeTensorsLoader.WidenBf16ToF32(patterns, expected);
        Assert.That(read["t"].Data, Is.EqualTo(expected));
    }

    [Test]
    public void ReadHalf_NarrowsF32Fixture_ViaCreateChecked()
    {
        var floats = new float[] { 1.5f, 2.5f, 3.0f };
        var data = new byte[floats.Length * sizeof(float)];
        MemoryMarshal.AsBytes(floats.AsSpan()).CopyTo(data);

        var file = BuildFixture("F32", new[] { floats.Length }, data);
        var read = SafeTensorsLoader.Read<Half>(file);

        var expected = new Half[floats.Length];
        for (int i = 0; i < floats.Length; i++) expected[i] = Half.CreateChecked(floats[i]);
        Assert.That(read["t"].Data, Is.EqualTo(expected));
    }

    [Test]
    public void ReadBFloat16_NarrowsF32Fixture_ViaCreateChecked()
    {
        var floats = new float[] { 1.5f, 2.5f, 3.0f };
        var data = new byte[floats.Length * sizeof(float)];
        MemoryMarshal.AsBytes(floats.AsSpan()).CopyTo(data);

        var file = BuildFixture("F32", new[] { floats.Length }, data);
        var read = SafeTensorsLoader.Read<BFloat16>(file);

        var expected = new BFloat16[floats.Length];
        for (int i = 0; i < floats.Length; i++) expected[i] = BFloat16.CreateChecked(floats[i]);
        Assert.That(read["t"].Data, Is.EqualTo(expected));
    }

    [Test]
    public void ReadHalf_UnsupportedDtype_ThrowsNotSupportedException()
    {
        var data = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var file = BuildFixture("U8", new[] { 1 }, data);
        Assert.Throws<NotSupportedException>(() => SafeTensorsLoader.Read<Half>(file));
    }

    [Test]
    public void ReadHalf_TruncatedFile_Throws()
    {
        var expected = new Half[] { (Half)1.5f };
        var data = new byte[expected.Length * 2];
        MemoryMarshal.AsBytes(expected.AsSpan()).CopyTo(data);
        var file = BuildFixture("F16", new[] { expected.Length }, data);
        var truncated = file.AsSpan(0, file.Length - 1).ToArray();
        Assert.Catch<Exception>(() => SafeTensorsLoader.Read<Half>(truncated));
    }
}
