using Nivara.AI;
using NUnit.Framework;
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Coverage for the checkpoint dtype map (<c>ReadDtypes</c>) and the zero-hop native reads it
/// makes possible: an F32 checkpoint read as <c>float</c> and an F16 checkpoint read as
/// <c>Half</c> must keep the on-disk bits, not round-trip through <c>CreateChecked</c>.
/// </summary>
[TestFixture]
public class SafeTensorsLoaderDtypeTests
{
    static byte[] BuildFixture(params (string Name, string Dtype, int[] Shape, byte[] Data)[] sections)
    {
        int offset = 0;
        var builder = new StringBuilder("{");
        foreach (var (name, dtype, shape, data) in sections)
        {
            int end = offset + data.Length;
            if (builder.Length > 1) builder.Append(',');
            builder.Append($"\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{string.Join(",", shape)}],\"data_offsets\":[{offset},{end}]}}");
            offset = end;
        }
        if (builder.Length > 1) builder.Append(',');
        builder.Append("\"__metadata__\":{\"format\":\"pt\"}");
        builder.Append('}');

        var headerBytes = Encoding.UTF8.GetBytes(builder.ToString());
        var file = new byte[8 + headerBytes.Length + offset];
        BinaryPrimitives.WriteUInt64LittleEndian(file, (ulong)headerBytes.Length);
        headerBytes.CopyTo(file.AsSpan(8));

        int cursor = 8 + headerBytes.Length;
        foreach (var section in sections)
        {
            section.Data.CopyTo(file.AsSpan(cursor));
            cursor += section.Data.Length;
        }
        return file;
    }

    static (string Name, string Dtype, int[] Shape, byte[] Data)[] MixedSections()
    {
        var floats = new[] { 1.5f, 2.5f };
        var halves = new Half[] { (Half)1.5f, (Half)(-2.0f) };
        var bf16Patterns = new ushort[] { 0x3F80, 0x4000 };
        var ints = new[] { 7, -9 };

        return
        [
            ("w_f32", "F32", [2], MemoryMarshal.AsBytes(floats.AsSpan()).ToArray()),
            ("w_f16", "F16", [2], MemoryMarshal.AsBytes(halves.AsSpan()).ToArray()),
            ("w_bf16", "BF16", [2], MemoryMarshal.AsBytes(bf16Patterns.AsSpan()).ToArray()),
            ("w_i32", "I32", [2], MemoryMarshal.AsBytes(ints.AsSpan()).ToArray()),
        ];
    }

    [Test]
    public void ReadDtypes_OnMixedFixture_ReportsEachTensorsOnDiskDtype()
    {
        var file = BuildFixture(MixedSections());

        var dtypes = SafeTensorsLoader.ReadDtypes(file);

        Assert.That(dtypes, Is.EqualTo(new Dictionary<string, string>
        {
            ["w_f32"] = "F32",
            ["w_f16"] = "F16",
            ["w_bf16"] = "BF16",
            ["w_i32"] = "I32",
        }));
    }

    [Test]
    public void ReadDtypes_OnMixedFixture_ExcludesFileLevelMetadata()
    {
        var file = BuildFixture(MixedSections());

        var dtypes = SafeTensorsLoader.ReadDtypes(file);

        Assert.That(dtypes.Keys, Does.Not.Contain("__metadata__"));
        Assert.That(dtypes.Count, Is.EqualTo(MixedSections().Length));
    }

    [Test]
    public void ReadDtypes_OnFile_MatchesByteArrayPath()
    {
        var file = BuildFixture(MixedSections());
        string tempPath = Path.Combine(Path.GetTempPath(), $"nivara-dtypes-{Guid.NewGuid():N}.safetensors");
        try
        {
            File.WriteAllBytes(tempPath, file);

            var fromFile = SafeTensorsLoader.ReadDtypes(tempPath);
            var fromBytes = SafeTensorsLoader.ReadDtypes(file);

            Assert.That(fromFile, Is.EqualTo(fromBytes));
        }
        finally
        {
            File.Delete(tempPath);
        }
    }

    [Test]
    public void ReadFloat_OnF32Fixture_PreservesBitsExactly()
    {
        // Includes the values a narrowing or naive conversion would disturb: signed zero,
        // the smallest subnormal, a NaN payload and the largest finite value.
        var values = new[] { 1.5f, -0.0f, float.Epsilon, float.NaN, float.MaxValue };
        var data = MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
        var file = BuildFixture(("w", "F32", [values.Length], data));

        var read = SafeTensorsLoader.Read<float>(file);

        Assert.That(MemoryMarshal.AsBytes(read["w"].Data.AsSpan()).ToArray(), Is.EqualTo(data));
    }

    [Test]
    public void ReadHalf_OnF16Fixture_PreservesBitsExactly()
    {
        // Raw F16 patterns for +0, -0, +inf, NaN and 1.0; a value round-trip would hide a
        // sign or payload change, a bit comparison cannot.
        var patterns = new ushort[] { 0x0000, 0x8000, 0x7C00, 0x7E00, 0x3C00 };
        var data = MemoryMarshal.AsBytes(patterns.AsSpan()).ToArray();
        var file = BuildFixture(("w", "F16", [patterns.Length], data));

        var read = SafeTensorsLoader.Read<Half>(file);

        Assert.That(MemoryMarshal.AsBytes(read["w"].Data.AsSpan()).ToArray(), Is.EqualTo(data));
    }
}
