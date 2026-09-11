namespace Nivara.GpuProbe.LevelZero;

/// <summary>
/// Debug helper: dumps the hand-authored SPIR-V modules to .spv files so they can
/// be validated with an external SPIR-V validator (spirv-val) without running the
/// GPU driver. Not part of the radiation-surface check itself.
/// </summary>
internal static class SpvDump
{
    public static int Run()
    {
        try
        {
            string dir = Path.Combine(Path.GetTempPath(), "opencode", "spv");
            Directory.CreateDirectory(dir);
            Write(Path.Combine(dir, "add_parallel.spv"), SpvKernels.AddParallel(SpvKernels.Version10));
            Write(Path.Combine(dir, "add_loop.spv"), SpvKernels.AddLoop(SpvKernels.Version10));
            Console.WriteLine($"Wrote SPIR-V modules to {dir}");
            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[FAIL] {ex.Message}");
            return 1;
        }
    }

    private static void Write(string path, uint[] words)
    {
        byte[] bytes = new byte[words.Length * 4];
        for (int i = 0; i < words.Length; i++)
        {
            bytes[i * 4] = (byte)(words[i] & 0xFF);
            bytes[i * 4 + 1] = (byte)((words[i] >> 8) & 0xFF);
            bytes[i * 4 + 2] = (byte)((words[i] >> 16) & 0xFF);
            bytes[i * 4 + 3] = (byte)((words[i] >> 24) & 0xFF);
        }

        File.WriteAllBytes(path, bytes);
    }
}