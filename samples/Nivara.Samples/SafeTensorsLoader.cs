using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace Nivara.Samples;

public static class SafeTensorsLoader
{
    public static Dictionary<string, (float[] Data, int[] Shape)> Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
            throw new FileNotFoundException($"SafeTensors file not found: {path}", path);

        return Read(File.ReadAllBytes(path));
    }

    public static Dictionary<string, (float[] Data, int[] Shape)> Read(byte[] bytes)
    {
        if (bytes.Length < 8)
            throw new InvalidDataException("SafeTensors file is too small to contain a header.");

        ulong headerSize = BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(0, 8));

        if (8 + headerSize > (ulong)bytes.Length)
            throw new InvalidDataException(
                $"Header size ({headerSize}) exceeds file size ({bytes.Length}).");

        int dataOffset = 8 + (int)headerSize;
        var dataBuffer = bytes.AsSpan(dataOffset);

        var headerJson = System.Text.Encoding.UTF8.GetString(bytes, 8, (int)headerSize);
        using var doc = JsonDocument.Parse(headerJson);

        var root = doc.RootElement;
        var result = new Dictionary<string, (float[] Data, int[] Shape)>(StringComparer.Ordinal);

        foreach (var property in root.EnumerateObject())
        {
            if (property.Name == "__metadata__")
                continue;

            var tensor = property.Value;
            string dtype = tensor.GetProperty("dtype").GetString()!;
            string name = property.Name;
            int elementSize = DtypeByteSize(dtype);

            var shapeArray = tensor.GetProperty("shape");
            var offsets = tensor.GetProperty("data_offsets");
            int begin = offsets[0].GetInt32();
            int end = offsets[1].GetInt32();

            int[] shape = new int[shapeArray.GetArrayLength()];
            for (int i = 0; i < shape.Length; i++)
                shape[i] = shapeArray[i].GetInt32();

            int byteLength = end - begin;
            int elementCount = 1;
            foreach (var d in shape)
                elementCount *= d;

            int expectedBytes = elementCount * elementSize;
            if (byteLength != expectedBytes)
                throw new InvalidDataException(
                    $"Tensor '{name}': expected {expectedBytes} bytes ({elementCount} × {elementSize}), got {byteLength} bytes.");

            ReadOnlySpan<byte> tensorBytes = dataBuffer.Slice(begin, byteLength);
            float[] data = DtypeToFloatArray(tensorBytes, dtype, name);

            result[name] = (data, shape);
        }

        return result;
    }

    static int DtypeByteSize(string dtype) => dtype switch
    {
        "F32" or "I32" => 4,
        "F16" or "BF16" => 2,
        "I64" => 8,
        _ => throw new NotSupportedException($"Unsupported dtype '{dtype}'.")
    };

    static float[] DtypeToFloatArray(ReadOnlySpan<byte> tensorBytes, string dtype, string name) => dtype switch
    {
        "F32" => MemoryMarshal.Cast<byte, float>(tensorBytes).ToArray(),
        "I32" => ConvertI32(tensorBytes),
        "I64" => ConvertI64(tensorBytes),
        "F16" => ConvertF16(tensorBytes),
        "BF16" => ConvertBF16(tensorBytes),
        _ => throw new NotSupportedException($"Tensor '{name}' has unsupported dtype '{dtype}'. " +
            "Supported dtypes: F32, I32, I64, F16, BF16.")
    };

    static float[] ConvertI32(ReadOnlySpan<byte> bytes)
    {
        var src = MemoryMarshal.Cast<byte, int>(bytes);
        var result = new float[src.Length];
        for (int i = 0; i < src.Length; i++)
            result[i] = src[i];
        return result;
    }

    static float[] ConvertI64(ReadOnlySpan<byte> bytes)
    {
        var src = MemoryMarshal.Cast<byte, long>(bytes);
        var result = new float[src.Length];
        for (int i = 0; i < src.Length; i++)
            result[i] = src[i];
        return result;
    }

    static float[] ConvertF16(ReadOnlySpan<byte> bytes)
    {
        var src = MemoryMarshal.Cast<byte, Half>(bytes);
        var result = new float[src.Length];
        for (int i = 0; i < src.Length; i++)
            result[i] = (float)src[i];
        return result;
    }

    static float[] ConvertBF16(ReadOnlySpan<byte> bytes)
    {
        var src = MemoryMarshal.Cast<byte, ushort>(bytes);
        var result = new float[src.Length];
        for (int i = 0; i < src.Length; i++)
        {
            uint bits = (uint)src[i] << 16;
            result[i] = Unsafe.As<uint, float>(ref bits);
        }
        return result;
    }
}
