using System.Runtime.InteropServices;

namespace Nivara.GpuProbe.OpenVino;

/// <summary>
/// Locates and loads the OpenVINO runtime (<c>openvino_c.dll</c>) the same way the
/// L0 leg loads <c>ze_loader.dll</c>: no packages, no NuGet. Search order:
/// <c>NIVARA_OPENVINO_DIR</c> env override &gt; the default pip site-packages libs
/// folder &gt; a live <c>python -c</c> probe of the installed <c>openvino</c> package.
/// The DLL is loaded with the altered-search-path flag so its siblings in the same
/// libs folder (<c>openvino.dll</c>, <c>tbb12.dll</c>, the <c>intel_*_plugin.dll</c>
/// GPU/CPU plugins) resolve at load.
/// </summary>
internal static class OpenVinoRunner
{
    public const string LibraryName = "openvino_c.dll";

    /// <summary>Resolves the folder containing <see cref="LibraryName"/>, or null.</summary>
    public static string? TryResolveLibraryDir()
    {
        string? env = Environment.GetEnvironmentVariable("NIVARA_OPENVINO_DIR");
        if (!string.IsNullOrEmpty(env) && File.Exists(Path.Combine(env, LibraryName)))
            return env;

        string defaultLibraries = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            "Python", "Python312", "site-packages", "openvino", "libs");
        if (File.Exists(Path.Combine(defaultLibraries, LibraryName)))
            return defaultLibraries;

        return ProbeViaPython();
    }

    /// <summary>Queries the installed Python openvino package for its libs folder.</summary>
    private static string? ProbeViaPython()
    {
        try
        {
            var psi = new System.Diagnostics.ProcessStartInfo("python")
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            psi.ArgumentList.Add("-c");
            psi.ArgumentList.Add("import openvino, pathlib; print(pathlib.Path(openvino.__file__).parent / 'libs')");
            using var process = System.Diagnostics.Process.Start(psi);
            if (process is null)
                return null;
            string stdout = process.StandardOutput.ReadToEnd();
            process.WaitForExit(5000);
            if (process.ExitCode != 0)
                return null;
            string line = stdout.Trim().Split('\n')[0].Trim();
            return line.Length > 0 && File.Exists(Path.Combine(line, LibraryName)) ? line : null;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>Loads the OpenVINO runtime, or returns null with a diagnostic printed.</summary>
    public static OpenVinoNative? TryLoad()
    {
        string? dir = TryResolveLibraryDir();
        if (dir is null)
        {
            Console.WriteLine("[openvino] openvino_c.dll not found. Install the runtime with:");
            Console.WriteLine("           python -m pip install openvino==2026.2.1");
            Console.WriteLine("           (or set NIVARA_OPENVINO_DIR to the folder containing openvino_c.dll)");
            return null;
        }
        try
        {
            return new OpenVinoNative(dir);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[openvino] failed to load {LibraryName} from {dir}: {ex.Message}");
            return null;
        }
    }

    [DllImport("kernel32", CharSet = CharSet.Ansi, SetLastError = true)]
    private static extern IntPtr LoadLibraryEx(string lpFileName, IntPtr hFile, uint dwFlags);

    [DllImport("kernel32", CharSet = CharSet.Ansi, ExactSpelling = true, SetLastError = true)]
    private static extern IntPtr GetProcAddress(IntPtr hModule, string procName);

    [DllImport("kernel32", SetLastError = true)]
    private static extern bool FreeLibrary(IntPtr hModule);

    internal static IntPtr Load(string libraryDir)
    {
        const uint LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008;
        string dll = Path.Combine(libraryDir, LibraryName);
        IntPtr handle = LoadLibraryEx(dll, IntPtr.Zero, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (handle == IntPtr.Zero)
            throw new InvalidOperationException(
                $"LoadLibraryEx({dll}) failed (Win32 error 0x{Marshal.GetLastWin32Error():X8})");
        return handle;
    }

    internal static T GetProc<T>(IntPtr handle, string name) where T : Delegate
    {
        IntPtr fn = GetProcAddress(handle, name);
        if (fn == IntPtr.Zero)
            throw new EntryPointNotFoundException($"{name} is not exported by {LibraryName}");
        return Marshal.GetDelegateForFunctionPointer<T>(fn);
    }
}