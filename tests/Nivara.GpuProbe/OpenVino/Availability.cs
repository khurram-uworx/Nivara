namespace Nivara.GpuProbe.OpenVino;

/// <summary>
/// `ov` mode diagnostics: runtime version and the GPU device the leg will compile to.
/// Prints the load failure + install hint when the runtime is missing.
/// </summary>
internal static class Availability
{
    public static int Run()
    {
        Console.WriteLine();
        Console.WriteLine("--- OpenVINO availability ---");
        using OpenVinoNative? ov = OpenVinoRunner.TryLoad();
        if (ov is null)
            return 1;
        Console.WriteLine($"  runtime: {ov.OpenVinoVersion()}");
        Console.WriteLine($"  GPU:     {ov.GetCoreProperty("GPU", "FULL_DEVICE_NAME")}");
        return 0;
    }
}