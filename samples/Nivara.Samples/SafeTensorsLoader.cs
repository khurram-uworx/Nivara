using System.Buffers.Binary;
using System.Numerics;
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
        => Read<float>(bytes).ToDictionary(kvp => kvp.Key, kvp => (kvp.Value.Data, kvp.Value.Shape));

    public static Dictionary<string, (T[] Data, int[] Shape)> Read<T>(string path)
        where T : struct, IFloatingPointIeee754<T>
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
            throw new FileNotFoundException($"SafeTensors file not found: {path}", path);

        return Read<T>(File.ReadAllBytes(path));
    }

    public static Dictionary<string, (T[] Data, int[] Shape)> Read<T>(byte[] bytes)
        where T : struct, IFloatingPointIeee754<T>
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
        var result = new Dictionary<string, (T[] Data, int[] Shape)>(StringComparer.Ordinal);

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
            T[] data = DtypeToArray<T>(tensorBytes, dtype, name);

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

    static T[] DtypeToArray<T>(ReadOnlySpan<byte> tensorBytes, string dtype, string name)
        where T : struct, IFloatingPointIeee754<T> => dtype switch
        {
            "F32" => ConvertF32<T>(tensorBytes),
            "I32" => ConvertI32<T>(tensorBytes),
            "I64" => ConvertI64<T>(tensorBytes),
            "F16" => ConvertF16<T>(tensorBytes),
            "BF16" => ConvertBF16<T>(tensorBytes),
            _ => throw new NotSupportedException($"Tensor '{name}' has unsupported dtype '{dtype}'. " +
                "Supported dtypes: F32, I32, I64, F16, BF16.")
        };

    static T[] ConvertF32<T>(ReadOnlySpan<byte> bytes)
        where T : struct, IFloatingPointIeee754<T>
    {
        var src = MemoryMarshal.Cast<byte, float>(bytes);
        var result = new T[src.Length];
        for (int i = 0; i < src.Length; i++)
            result[i] = T.CreateChecked(src[i]);
        return result;
    }

    static T[] ConvertI32<T>(ReadOnlySpan<byte> bytes)
        where T : struct, IFloatingPointIeee754<T>
    {
        var src = MemoryMarshal.Cast<byte, int>(bytes);
        var result = new T[src.Length];
        for (int i = 0; i < src.Length; i++)
            result[i] = T.CreateChecked(src[i]);
        return result;
    }

    static T[] ConvertI64<T>(ReadOnlySpan<byte> bytes)
        where T : struct, IFloatingPointIeee754<T>
    {
        var src = MemoryMarshal.Cast<byte, long>(bytes);
        var result = new T[src.Length];
        for (int i = 0; i < src.Length; i++)
            result[i] = T.CreateChecked(src[i]);
        return result;
    }

    static T[] ConvertF16<T>(ReadOnlySpan<byte> bytes)
        where T : struct, IFloatingPointIeee754<T>
    {
        var src = MemoryMarshal.Cast<byte, Half>(bytes);
        var result = new T[src.Length];
        for (int i = 0; i < src.Length; i++)
            result[i] = T.CreateChecked(src[i]);
        return result;
    }

    static T[] ConvertBF16<T>(ReadOnlySpan<byte> bytes)
        where T : struct, IFloatingPointIeee754<T>
    {
        var src = MemoryMarshal.Cast<byte, ushort>(bytes);
        var result = new T[src.Length];
        for (int i = 0; i < src.Length; i++)
        {
            uint bits = (uint)src[i] << 16;
            float f = Unsafe.As<uint, float>(ref bits);
            result[i] = T.CreateChecked(f);
        }
        return result;
    }
}
