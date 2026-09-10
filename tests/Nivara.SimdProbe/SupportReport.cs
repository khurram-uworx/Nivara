using System.Numerics;
using System.Reflection;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;

namespace Nivara.SimdProbe;

/// <summary>
/// Prints the hardware/ISA vector support surface for BFloat16 / Half, so we can
/// tell at a glance whether the BCL vectorized paths (Vector{T} / Vector128{T} /
/// TensorPrimitives) can use these narrow types on the installed runtime.
/// </summary>
internal static class SupportReport
{
    public static int Run()
    {
        Console.WriteLine("=== Vector<T> / ISA support for BFloat16 and Half ===");

        Console.WriteLine();
        Console.WriteLine("Variable-width Vector<T> (TensorPrimitives dispatch):");
        Print("Vector<BFloat16>.IsSupported", Vector<BFloat16>.IsSupported);
        Print("Vector<Half>.IsSupported", Vector<Half>.IsSupported);
        Print("Vector<float>.IsSupported (reference)", Vector<float>.IsSupported);

        Console.WriteLine();
        Console.WriteLine("Fixed-width hardware vectors:");
        Print("Vector128<BFloat16>.IsSupported", Vector128<BFloat16>.IsSupported);
        Print("Vector128<Half>.IsSupported", Vector128<Half>.IsSupported);
        Print("Vector256<BFloat16>.IsSupported", Vector256<BFloat16>.IsSupported);
        Print("Vector256<Half>.IsSupported", Vector256<Half>.IsSupported);
        Print("Vector512<BFloat16>.IsSupported", Vector512<BFloat16>.IsSupported);
        Print("Vector512<Half>.IsSupported", Vector512<Half>.IsSupported);

        Console.WriteLine();
        Console.WriteLine("x86 intrinsics relevant to FP16 (presence + support):");
        PrintIntrinsic("F16C", GetX86Intrinsic("F16C"));
        PrintIntrinsic("Avx10v1", GetX86Intrinsic("Avx10v1"));
        PrintIntrinsic("Avx10v1.VL", GetNestedX86Intrinsic("Avx10v1", "VL"));
        PrintIntrinsic("Avx512F", GetX86Intrinsic("Avx512F"));
        PrintIntrinsic("Avx512F.VL", GetNestedX86Intrinsic("Avx512F", "VL"));

        Console.WriteLine();
        Console.WriteLine("Arm64 intrinsics (FP16 arithmetic):");
        Print("AdvSimd.IsSupported", AdvSimd.IsSupported);

        Console.WriteLine();
        Console.WriteLine("Notes: no F16C batch intrinsics (ConvertToVector128Single from 16-bit)");
        Console.WriteLine("exist in the managed Surface Area on .NET 11 RC1. The JIT emits");
        Console.WriteLine("vcvtph2ps/vcvtps2ph only for scalar Half<->float conversions, so");
        Console.WriteLine("TensorPrimitives still runs BFloat16/Half through scalar fallback loops.");
        return 0;
    }

    private static Type? GetX86Intrinsic(string name)
    {
        foreach (var type in typeof(X86Base).Assembly.GetTypes())
            if (type.Namespace == "System.Runtime.Intrinsics.X86" && type.Name == name) return type;
        return null;
    }

    private static Type? GetNestedX86Intrinsic(string parent, string nested)
    {
        var parentType = GetX86Intrinsic(parent);
        if (parentType is null) return null;
        return parentType.GetNestedType(nested, BindingFlags.Public | BindingFlags.Static);
    }

    private static void PrintIntrinsic(string label, Type? intrinsicType)
    {
        if (intrinsicType is null)
        {
            Console.WriteLine($"  {label,-40} ABSENT from runtime surface");
            return;
        }

        bool supported = ReadIsSupported(intrinsicType);
        Console.WriteLine($"  {label,-40} {(supported ? "Supported" : "NOT supported")}");
    }

    private static bool ReadIsSupported(Type intrinsicType)
    {
        try
        {
            var prop = intrinsicType.GetProperty("IsSupported", BindingFlags.Public | BindingFlags.Static);
            return prop is not null && (bool)(prop.GetValue(null) ?? false);
        }
        catch
        {
            return false;
        }
    }

    private static void Print(string label, bool supported)
    {
        Console.WriteLine($"  {label,-40} {(supported ? "Supported" : "NOT supported")}");
    }
}