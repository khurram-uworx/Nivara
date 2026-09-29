using ILGPU.Runtime.OpenCL;
using System.Runtime.InteropServices;
using System.Text;

namespace Nivara.PerformanceTests;

/// <summary>
/// Reads <c>CL_DRIVER_VERSION</c> for an ILGPU <see cref="CLDevice"/> straight from the OpenCL
/// ICD, because ILGPU 1.5.3 does not surface it: <see cref="CLDevice.DeviceVersion"/> is a
/// <c>CLDeviceVersion</c> (Major/Minor) holding the OpenCL version the device *supports*, not the
/// driver that compiled the kernels. That distinction is the whole point of asking — the
/// fingerprint baseline is keyed on the toolchain so that a driver bump, which may legitimately
/// change f32 rounding, is one stated precondition rather than 171 numeric failures that read
/// like a kernel regression.
/// </summary>
/// <remarks>
/// Windows-only by construction: this harness already requires the in-box <c>OpenCL.dll</c> plus a
/// GPU device, and it returns null rather than throwing when the ICD entry point is not reachable
/// so that a run on a differently-provisioned machine degrades to a visibly weaker key instead of
/// failing before it measures anything.
/// </remarks>
internal static class ClDriverVersion
{
    const uint CL_SUCCESS = 0;
    const uint CL_DEVICE_DRIVER_VERSION = 0x102D;

    /// <summary>Guards against a pathological size query; real driver strings are tens of bytes.</summary>
    const nuint MaxPlausibleLength = 4096;

    [DllImport("OpenCL.dll", CallingConvention = CallingConvention.StdCall)]
    static extern int clGetDeviceInfo(
        IntPtr device,
        uint paramName,
        nuint paramValueSize,
        [Out] byte[]? paramValue,
        out nuint paramValueSizeRet);

    /// <summary>
    /// The device's <c>CL_DRIVER_VERSION</c> string, or null if it cannot be read. Null is a
    /// weaker key, not a pass: the caller is expected to say so in the recorded toolchain rather
    /// than silently fall back to something that looks equally identifying.
    /// </summary>
    public static string? TryGet(CLDevice device)
    {
        try
        {
            IntPtr id = device.DeviceId;
            if (id == IntPtr.Zero)
                return null;

            if (clGetDeviceInfo(id, CL_DEVICE_DRIVER_VERSION, 0, paramValue: null, out nuint required) != CL_SUCCESS
                || required == 0 || required > MaxPlausibleLength)
                return null;

            var buffer = new byte[(int)required];
            if (clGetDeviceInfo(id, CL_DEVICE_DRIVER_VERSION, required, buffer, out _) != CL_SUCCESS)
                return null;

            int end = Array.IndexOf(buffer, (byte)0);
            return end < 0 ? null : Encoding.UTF8.GetString(buffer, 0, end);
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException)
        {
            return null;
        }
    }
}
