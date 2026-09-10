using System.Runtime.Intrinsics.X86;

namespace Nivara.SimdProbe;

/// <summary>
/// Dumps the raw CPUID feature surface (leaf 0/1/7, extended leaves) so we can
/// tell whether an "unsupported" intrinsic in .NET is a hardware/firmware absence
/// or an OS/hypervisor masking. X86Base.CpuId executes the real CPUID instruction
/// (as filtered by the OS + Hyper-V hypervisor when VBS is on).
/// </summary>
internal static class CpuIdProbe
{
    public static int Run()
    {
        Console.WriteLine("=== Raw CPUID ===");
        if (!X86Base.IsSupported)
        {
            Console.WriteLine("X86Base.CpuId is unavailable on this platform.");
            return 1;
        }

        Leaf(0, 0, out uint maxBasic, out uint ebx, out uint ecx, out uint edx);
        string vendor = Vendor(ebx, edx, ecx);
        Console.WriteLine($"Vendor: {vendor}   Max basic leaf: 0x{maxBasic:X}");

        Leaf(0x80000000, 0, out uint maxExt, out _, out _, out _);
        var brand = maxExt >= 0x80000004 ? BrandString() : null;
        if (!string.IsNullOrWhiteSpace(brand)) Console.WriteLine($"Brand:  {brand}");

        Leaf(1, 0, out _, out _, out uint ecx1, out uint edx1);
        Console.WriteLine();
        Console.WriteLine($"Legacy (leaf 1): ECX=0x{ecx1:X8}  EDX=0x{edx1:X8}");
        Print("SSE", edx1, 25);
        Print("SSE2", edx1, 26);
        Print("SSE3", ecx1, 0);
        Print("SSSE3", ecx1, 9);
        Print("SSE4.1", ecx1, 19);
        Print("SSE4.2", ecx1, 20);
        Print("FMA", ecx1, 12);
        Print("AVX", ecx1, 28);
        Print("OSXSAVE", ecx1, 27);

        Leaf(7, 0, out uint eax7, out uint ebx7, out uint ecx7, out uint edx7);
        Console.WriteLine();
        Console.WriteLine($"Leaf 7 subleaf 0 (extended features): EAX=0x{eax7:X8}  EBX=0x{ebx7:X8}  ECX=0x{ecx7:X8}  EDX=0x{edx7:X8}");
        Print("AVX2", ebx7, 5);
        Print("AVX512F", ebx7, 16);
        Print("AVX512DQ", ebx7, 17);
        Print("AVX512IFMA", ebx7, 21);
        Print("AVX512CD", ebx7, 28);
        Print("AVX512BW", ebx7, 30);
        Print("AVX512VL", ebx7, 31);
        Print("AVX512_VBMI", ecx7, 1);
        Print("AVX512_VBMI2", ecx7, 6);
        Print("GFNI", ecx7, 8);
        Print("VAES", ecx7, 9);
        Print("VPCLMULQDQ", ecx7, 10);
        Print("AVX512_VNNI", ecx7, 11);
        Print("AVX512_BITALG", ecx7, 12);
        Print("AVX512_VPOPCNTDQ", ecx7, 14);
        Print("AVX512_4VNNIW", edx7, 2);
        Print("AVX512_4FMAPS", edx7, 3);
        Print("AVX_VNNI", eax7, 4);
        Print("AVX512_BF16", eax7, 5);
        Print("AVX512_FP16", eax7, 23);
        Print("AMX_BF16", eax7, 22);
        Print("AMX_INT8", eax7, 25);
        Print("AVX_IFMA", eax7, 21);
        Print("Hybrid (Intel)", edx7, 15);

        Leaf(7, 1, out uint eax71, out uint ebx71, out uint ecx71, out uint edx71);
        Console.WriteLine();
        Console.WriteLine($"Leaf 7 subleaf 1 (AVX-VNNI / AVX-IFMA / AVX10 pointer): EAX=0x{eax71:X8}  EBX=0x{ebx71:X8}  ECX=0x{ecx71:X8}  EDX=0x{edx71:X8}");
        Print("AVX_VNNI (EAX.4)", eax71, 4);
        Print("AVX_IFMA (EAX.23)", eax71, 23);
        Print("AVX_VNNI_INT8 (EDX.4)", edx71, 4);
        Print("AVX_VNNI_INT16 (EDX.10)", edx71, 10);
        Print("AVX10 pointer (EDX.19, .NET gate)", edx71, 19);

        Console.WriteLine();
        Console.WriteLine($"CPUID leaf 0x24 (dedicated AVX10 leaf): max basic leaf is 0x{maxBasic:X}, so leaf 0x24 is {(maxBasic >= 0x24 ? "queryable" : "NOT exposed by this CPU")}");
        bool avx10LeafPresent = maxBasic >= 0x24 && Bit(edx71, 19);
        if (maxBasic >= 0x24)
        {
            Leaf(0x24, 0, out uint eax24, out uint ebx24, out uint ecx24, out uint edx24);
            uint avx10Version = ebx24 & 0xFF;
            bool v128 = Bit(ebx24, 16), v256 = Bit(ebx24, 17), v512 = Bit(ebx24, 18);
            Console.WriteLine($"  Leaf 0x24: EAX=0x{eax24:X8}  EBX=0x{ebx24:X8}(ver={avx10Version})  ECX=0x{ecx24:X8}  EDX=0x{edx24:X8}");
            Console.WriteLine($"  V128={v128}  V256={v256}  V512={v512}  AVX10 version={avx10Version}");
        }

        if (Bit(edx7, 15))
        {
            Leaf(0x1A, 0, out uint eax1a, out _, out _, out _);
            // Documented core-type codes: 0x20 = Intel Atom/E-core, 0x40 = Intel Core/P-core.
            // Arrow Lake ships Lion Cove P-cores + Skymont/Crestmont E-cores (hybrid confirmed).
            Console.WriteLine($"  Hybrid (leaf 0x1A) EAX=0x{eax1a:X8} (core type bits: reinterpretation varies by revision; raw value shown)");
        }

        Console.WriteLine();
        Console.WriteLine("Verdict (matching .NET 11 detection semantics):");
        bool anyAvx512 = Bit(ebx7, 16) || Bit(ebx7, 17) || Bit(ebx7, 21) || Bit(ebx7, 28)
                      || Bit(ebx7, 30) || Bit(ebx7, 31) || Bit(ecx7, 1) || Bit(ecx7, 6)
                      || Bit(ecx7, 11) || Bit(ecx7, 12) || Bit(ecx7, 14) || Bit(edx7, 2)
                      || Bit(edx7, 3) || Bit(eax7, 5) || Bit(eax7, 23);
        bool avx10 = avx10LeafPresent; // leaf 0x24 present + leaf 7.1 EDX.19 gate
        bool anyAmx = Bit(eax7, 22) || Bit(eax7, 25);
        Console.WriteLine($"  AVX-512 family bits:  {(anyAvx512 ? "PRESENT" : "absent in hardware/CPUID")}");
        Console.WriteLine($"  AVX10 (leaf 0x24):    {(avx10 ? "PRESENT" : "absent in hardware/CPUID")}");
        Console.WriteLine($"  AMX:                  {(anyAmx ? "PRESENT" : "absent in hardware/CPUID")}");
        return 0;
    }

    private static void Leaf(uint leaf, uint subleaf, out uint eax, out uint ebx, out uint ecx, out uint edx)
    {
        var (a, b, c, d) = X86Base.CpuId((int)leaf, (int)subleaf);
        eax = unchecked((uint)a);
        ebx = unchecked((uint)b);
        ecx = unchecked((uint)c);
        edx = unchecked((uint)d);
    }

    private static string Vendor(uint ebx, uint edx, uint ecx)
    {
        Span<char> chars = stackalloc char[12];
        int pos = 0;
        foreach (uint value in new[] { ebx, edx, ecx })
        {
            for (int i = 0; i < 4; i++) chars[pos++] = (char)((value >> (i * 8)) & 0xFF);
        }

        return new string(chars);
    }

    private static string BrandString()
    {
        Span<char> brand = stackalloc char[48];
        int pos = 0;
        for (uint leaf = 0x80000002; leaf <= 0x80000004; leaf++)
        {
            Leaf(leaf, 0, out uint eax, out uint ebx, out uint ecx, out uint edx);
            foreach (uint value in new[] { eax, ebx, ecx, edx })
            {
                for (int i = 0; i < 4; i++)
                {
                    byte b = (byte)((value >> (i * 8)) & 0xFF);
                    if (b != 0) brand[pos++] = (char)b;
                }
            }
        }

        return new string(brand[..pos]).Trim();
    }

    private static bool Bit(uint reg, int bit) => (reg & (1u << bit)) != 0;

    private static void Print(string name, uint reg, int bit)
        => Console.WriteLine($"  {name,-22} {(Bit(reg, bit) ? "Supported" : "not present")}");
}