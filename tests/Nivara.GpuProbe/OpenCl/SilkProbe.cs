using Silk.NET.OpenCL;
using System.Text;

namespace Nivara.GpuProbe.OpenCl;

internal static class SilkProbe
{
    private static int _failures;
    private const ulong CL_MEM_READ_WRITE = (ulong)(1 << 1);

    public static int Run()
    {
        Console.WriteLine("--- Silk.NET OpenCL (Intel/AMD iGPU focus) ---");
        try
        {
            var cl = CL.GetApi();
            uint platformCount = 0;
            unsafe
            {
                int err = cl.GetPlatformIDs(0, null, &platformCount);
                if (err != 0 || platformCount == 0)
                {
                    Console.WriteLine("  [FAIL] No OpenCL platforms");
                    _failures++;
                    return _failures;
                }
            }
            Console.WriteLine($"  Found {platformCount} platform(s)");
            unsafe
            {
                nint* platforms = stackalloc nint[(int)platformCount];
                int err = cl.GetPlatformIDs(platformCount, platforms, null);
                if (err != 0) { _failures++; return _failures; }
                for (int pi = 0; pi < platformCount; pi++)
                {
                    nint p = platforms[pi];
                    nuint nameSize = 0;
                    cl.GetPlatformInfo(p, PlatformInfo.Name, 0, null, &nameSize);
                    byte[] nameBuf = new byte[nameSize];
                    fixed (byte* nb = nameBuf) cl.GetPlatformInfo(p, PlatformInfo.Name, nameSize, nb, null);
                    string pname = Encoding.UTF8.GetString(nameBuf, 0, (int)(nameSize > 0 ? nameSize - 1 : 0));
                    Console.WriteLine($"  [{pi}] {pname}");
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  [FAIL] {ex}");
            _failures++;
        }
        return _failures;
    }
}
