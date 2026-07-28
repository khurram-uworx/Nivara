using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace NivaraInference;

internal static class SafeTensorsLoader
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

            if (dtype != "F32")
                ValidateDtype(dtype, name);

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

            int expectedBytes = elementCount * Unsafe.SizeOf<float>();
            if (byteLength != expectedBytes)
                throw new InvalidDataException(
                    $"Tensor '{name}': expected {expectedBytes} bytes ({elementCount} × 4), got {byteLength} bytes.");

            ReadOnlySpan<byte> tensorBytes = dataBuffer.Slice(begin, byteLength);
            float[] data = MemoryMarshal.Cast<byte, float>(tensorBytes).ToArray();

            result[name] = (data, shape);
        }

        return result;
    }

    static void ValidateDtype(string dtype, string tensorName)
    {
        if (dtype == "F32")
            return;

        string message = dtype switch
        {
            "BF16" => $"Tensor '{tensorName}' has dtype 'BF16'. Nivara currently supports F32 only. " +
                      "BF16 support is coming with .NET 11.",
            "F16" => $"Tensor '{tensorName}' has dtype 'F16'. Nivara currently supports F32 only.",
            _ => $"Tensor '{tensorName}' has unsupported dtype '{dtype}'. Only F32 is supported."
        };

        throw new NotSupportedException(message);
    }
}
