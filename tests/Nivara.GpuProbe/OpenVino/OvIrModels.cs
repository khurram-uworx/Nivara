using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using Nivara.GpuProbe.Kernels;

namespace Nivara.GpuProbe.OpenVino;

/// <summary>An IR v11 onnx.xml-style model: the graph text plus the raw weights blob.</summary>
internal sealed record OvIrModel(string Name, string Xml, byte[]? Weights);

/// <summary>
/// Byte-true IR v11 (Model Optimizer XML) for the three SmolLM-shaped kernels,
/// mirroring the graphs validated against the GPU plugin in Python
/// (<c>ov-fresh/full.py</c>): dot16, silu (Sigmoid+Multiply then F32 convert),
/// and gemv (MatMul x·Wᵀ). Each kernel is emitted twice: a BF16-declared model
/// (raw BF16 wire weights) and an F32-declared model (BF16 fixtures widened to
/// F32 through the production Nivara widen — the same value the CPU gold uses,
/// see <see cref="CpuLeg.Widen"/>). Only the BF16 paths carry the Convert→F32
/// output layer; the F32 paths emit the reduction result directly, matching the
/// validated gemv_f32 IR.
/// </summary>
internal static class OvIrModels
{
    public static OvIrModel Dot16(KernelFixtures fixtures, bool bf16)
    {
        int k = KernelFixtures.Dot16Length;
        byte[] weights = Weights(bf16, fixtures.Dot16B);
        string xml = Build(
            "dot16",
            Param(0, "x", bf16, [1, k])
            + Const(1, "w", bf16, new long[2] { k, 1 }, 0, k * (bf16 ? 2 : 4))
            + MatMul(2, "mm", bf16, [1, k], new long[] { k, 1 }, new long[] { 1, 1 }),
            bf16 ? ConvertToF32(3, "tof32", [1, 1]) + Result(4, "result", [1, 1])
                 : Result(3, "result", [1, 1]),
            bf16 ? Edge(0, 0, 2, 0) + Edge(1, 0, 2, 1) + Edge(2, 0, 3, 0) + Edge(3, 0, 4, 0)
                 : Edge(0, 0, 2, 0) + Edge(1, 0, 2, 1) + Edge(2, 0, 3, 0));
        return new OvIrModel("dot16", xml, weights);
    }

    public static OvIrModel Silu(KernelFixtures fixtures, bool bf16)
    {
        int h = KernelFixtures.HiddenSize;
        string xml = Build(
            "silu",
            Param(0, "x", bf16, [h])
            + Sigmoid(1, "sig", bf16, [h])
            + Multiply(2, "mul", bf16, [h]),
            bf16 ? ConvertToF32(3, "tof32", [h]) + Result(4, "result", [h])
                 : Result(3, "result", [h]),
            bf16 ? Edge(0, 0, 1, 0) + Edge(0, 0, 2, 1) + Edge(1, 0, 2, 0) + Edge(2, 0, 3, 0) + Edge(3, 0, 4, 0)
                 : Edge(0, 0, 1, 0) + Edge(0, 0, 2, 1) + Edge(1, 0, 2, 0) + Edge(2, 0, 3, 0));
        return new OvIrModel("silu", xml, null);
    }

    public static OvIrModel Gemv(KernelFixtures fixtures, bool bf16)
    {
        int h = KernelFixtures.HiddenSize;
        int i = KernelFixtures.IntermediateSize;
        byte[] weights = Weights(bf16, fixtures.GemvW);
        string xml = Build(
            "gemv",
            Param(0, "x", bf16, new long[] { 1, h })
            + Const(1, "w", bf16, new long[] { i, h }, 0, i * h * (bf16 ? 2 : 4))
            + MatMul(2, "mm", bf16, new long[] { 1, h }, new long[] { i, h }, new long[] { 1, i }, transposeB: true),
            bf16 ? ConvertToF32(3, "tof32", [1, i]) + Result(4, "result", new long[] { 1, i })
                 : Result(3, "result", new long[] { 1, i }),
            bf16 ? Edge(0, 0, 2, 0) + Edge(1, 0, 2, 1) + Edge(2, 0, 3, 0) + Edge(3, 0, 4, 0)
                 : Edge(0, 0, 2, 0) + Edge(1, 0, 2, 1) + Edge(2, 0, 3, 0));
        return new OvIrModel("gemv", xml, weights);
    }

    /// <summary>
    /// Model weights: raw BF16 byte patterns when the model is BF16-declared
    /// (the GPU wire format is the type itself); otherwise the exact F32 widen via
    /// the production Nivara path (<see cref="CpuLeg.Widen"/>).
    /// </summary>
    private static byte[] Weights(bool bf16, ReadOnlySpan<BFloat16> values)
    {
        if (bf16)
            return MemoryMarshal.AsBytes(values).ToArray();
        return MemoryMarshal.AsBytes(CpuLeg.Widen(values).AsSpan()).ToArray();
    }

    private static string Prec(bool bf16) => bf16 ? "BF16" : "FP32";
    private static string Elem(bool bf16) => bf16 ? "bf16" : "f32";

    private static string Dims(ReadOnlySpan<long> dims)
        => string.Join(',', dims.ToArray());

    private static string Port(bool bf16, int id, ReadOnlySpan<long> dims)
    {
        var sb = new StringBuilder();
        sb.Append($"        <port id=\"{id}\" precision=\"{Prec(bf16)}\">\n");
        foreach (long d in dims)
            sb.Append($"            <dim>{d}</dim>\n");
        sb.Append("        </port>");
        return sb.ToString();
    }

    private static string Param(int id, string name, bool bf16, ReadOnlySpan<long> dims)
        => $"    <layer id=\"{id}\" name=\"{name}\" type=\"Parameter\" version=\"opset1\">\n"
           + $"        <data element_type=\"{Elem(bf16)}\" shape=\"{Dims(dims)}\"/>\n"
           + $"        <output>\n{Port(bf16, 0, dims)}\n        </output>\n    </layer>\n";

    private static string Const(int id, string name, bool bf16, ReadOnlySpan<long> dims, int offset, long size)
        => $"    <layer id=\"{id}\" name=\"{name}\" type=\"Const\" version=\"opset1\">\n"
           + $"        <data element_type=\"{Elem(bf16)}\" shape=\"{Dims(dims)}\" offset=\"{offset}\" size=\"{size}\"/>\n"
           + $"        <output>\n{Port(bf16, 0, dims)}\n        </output>\n    </layer>\n";

    private static string MatMul(int id, string name, bool bf16, ReadOnlySpan<long> a, ReadOnlySpan<long> b, ReadOnlySpan<long> o, bool transposeB = false)
        => $"    <layer id=\"{id}\" name=\"{name}\" type=\"MatMul\" version=\"opset1\">\n"
           + $"        <data transpose_a=\"false\" transpose_b=\"{(transposeB ? "true" : "false")}\"/>\n"
           + $"        <input>\n{Port(bf16, 0, a)}\n{Port(bf16, 1, b)}\n        </input>\n"
           + $"        <output>\n{Port(bf16, 0, o)}\n        </output>\n    </layer>\n";

    private static string Sigmoid(int id, string name, bool bf16, ReadOnlySpan<long> a)
        => $"    <layer id=\"{id}\" name=\"{name}\" type=\"Sigmoid\" version=\"opset1\">\n"
           + $"        <input>\n{Port(bf16, 0, a)}\n        </input>\n"
           + $"        <output>\n{Port(bf16, 0, a)}\n        </output>\n    </layer>\n";

    private static string Multiply(int id, string name, bool bf16, ReadOnlySpan<long> a)
        => $"    <layer id=\"{id}\" name=\"{name}\" type=\"Multiply\" version=\"opset1\">\n"
           + $"        <data auto_broadcast=\"numpy\"/>\n"
           + $"        <input>\n{Port(bf16, 0, a)}\n{Port(bf16, 1, a)}\n        </input>\n"
           + $"        <output>\n{Port(bf16, 0, a)}\n        </output>\n    </layer>\n";

    private static string ConvertToF32(int id, string name, ReadOnlySpan<long> dims)
        => $"    <layer id=\"{id}\" name=\"{name}\" type=\"Convert\" version=\"opset1\">\n"
           + $"        <data destination_type=\"f32\"/>\n"
           + $"        <input>\n{Port(bf16: true, 0, dims)}\n        </input>\n"
           + $"        <output>\n{Port(bf16: false, 0, dims)}\n        </output>\n    </layer>\n";

    private static string Result(int id, string name, ReadOnlySpan<long> dims)
        => $"    <layer id=\"{id}\" name=\"{name}\" type=\"Result\" version=\"opset1\">\n"
           + $"        <input>\n{Port(bf16: false, 0, dims)}\n        </input>\n    </layer>\n";

    private static string Edge(int from, int fromPort, int to, int toPort)
        => $"    <edge from-layer=\"{from}\" from-port=\"{fromPort}\" to-layer=\"{to}\" to-port=\"{toPort}\"/>\n";

    private static string Build(string name, string head, string tail, string edges)
        => $"<?xml version=\"1.0\" ?>\n<net name=\"{name}\" version=\"11\">\n"
           + $"    <layers>\n{head}{tail}    </layers>\n"
           + $"    <edges>\n{edges}    </edges>\n</net>\n";
}