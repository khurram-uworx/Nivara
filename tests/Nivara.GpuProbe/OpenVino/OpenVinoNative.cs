using System.Runtime.InteropServices;

namespace Nivara.GpuProbe.OpenVino;

/// <summary>Element types from openvino/c/ov_common.h ov_element_type_e (2026.2.1).</summary>
internal enum OvElementType
{
    Dynamic = 0,
    Boolean = 1,
    Bf16 = 2,
    F16 = 3,
    F32 = 4,
    F64 = 5,
    I4 = 6,
    I8 = 7,
    I16 = 8,
    I32 = 9,
    I64 = 10,
    U1 = 11,
    U2 = 12,
    U3 = 13,
    U4 = 14,
    U6 = 15,
    U8 = 16,
    U16 = 17,
    U32 = 18,
    U64 = 19,
    Nf4 = 20,
    F8E4M3 = 21,
    F8E5M3 = 22,
    String = 23,
    F4E2M1 = 24,
    F8E8M0 = 25
}

/// <summary>ov_shape_t from openvino/c/ov_shape.h: { int64_t rank; int64_t* dims; }.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct OvShape
{
    public long Rank;
    public IntPtr Dims;
}

/// <summary>ov_version_t from openvino/c/ov_core.h: { const char* buildNumber; const char* description; }.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct OvVersion
{
    public IntPtr BuildNumber;
    public IntPtr Description;
}

/// <summary>
/// Pure P/Invoke surface over <c>openvino_c.dll</c>. All function pointers are
/// resolved by name (mirroring <c>L0Native</c> in the LevelZero leg); every
/// calling convention is <see cref="CallingConvention.Cdecl"/> and every string
/// is ANSI <c>char*</c> per the C headers. Signatures, the struct layouts, and
/// the <c>ov_element_type_e</c>/<c>ov_status_e</c> values were read from the
/// 2026.2.1 headers, not from memory.
/// </summary>
internal sealed class OpenVinoNative : IDisposable
{
    private readonly IntPtr handle;
    private readonly IntPtr core;

    public IntPtr Core => core;

    public OpenVinoNative(string libraryDir)
    {
        handle = OpenVinoRunner.Load(libraryDir);
        try
        {
            GetOpenVinoVersion = OpenVinoRunner.GetProc<OvGetOpenVinoVersion>(handle, "ov_get_openvino_version");
            VersionFree = OpenVinoRunner.GetProc<OvVersionFree>(handle, "ov_version_free");
            CoreCreate = OpenVinoRunner.GetProc<OvCoreCreate>(handle, "ov_core_create");
            CoreFree = OpenVinoRunner.GetProc<OvCoreFree>(handle, "ov_core_free");
            CoreReadModel = OpenVinoRunner.GetProc<OvCoreReadModel>(handle, "ov_core_read_model");
            CoreCompileModel = OpenVinoRunner.GetProc<OvCoreCompileModel>(handle, "ov_core_compile_model");
            CoreSetProperty = OpenVinoRunner.GetProc<OvCoreSetProperty>(handle, "ov_core_set_property");
            CoreGetProperty = OpenVinoRunner.GetProc<OvCoreGetProperty>(handle, "ov_core_get_property");
            CompiledModelFreeFn = OpenVinoRunner.GetProc<OvCompiledModelFree>(handle, "ov_compiled_model_free");
            CompiledModelGetProperty = OpenVinoRunner.GetProc<OvCompiledModelGetProperty>(handle, "ov_compiled_model_get_property");
            CompiledModelCreateInferRequest = OpenVinoRunner.GetProc<OvCompiledModelCreateInferRequest>(handle, "ov_compiled_model_create_infer_request");
            InferRequestFreeFn = OpenVinoRunner.GetProc<OvInferRequestFree>(handle, "ov_infer_request_free");
            InferRequestSetInputTensor = OpenVinoRunner.GetProc<OvInferRequestSetInputTensor>(handle, "ov_infer_request_set_input_tensor_by_index");
            InferRequestGetOutputTensor = OpenVinoRunner.GetProc<OvInferRequestGetOutputTensor>(handle, "ov_infer_request_get_output_tensor_by_index");
            InferRequestGetInputTensor = OpenVinoRunner.GetProc<OvInferRequestGetInputTensor>(handle, "ov_infer_request_get_input_tensor_by_index");
            InferFn = OpenVinoRunner.GetProc<OvInfer>(handle, "ov_infer_request_infer");
            TensorCreate = OpenVinoRunner.GetProc<OvTensorCreate>(handle, "ov_tensor_create");
            TensorDataFn = OpenVinoRunner.GetProc<OvTensorData>(handle, "ov_tensor_data");
            TensorGetShape = OpenVinoRunner.GetProc<OvTensorGetShape>(handle, "ov_tensor_get_shape");
            ShapeFree = OpenVinoRunner.GetProc<OvShapeFree>(handle, "ov_shape_free");
            TensorGetElementType = OpenVinoRunner.GetProc<OvTensorGetElementType>(handle, "ov_tensor_get_element_type");
            TensorFreeFn = OpenVinoRunner.GetProc<OvTensorFree>(handle, "ov_tensor_free");
            ModelFreeFn = OpenVinoRunner.GetProc<OvModelFree>(handle, "ov_model_free");
            ModelInputsSize = OpenVinoRunner.GetProc<OvModelInputsSize>(handle, "ov_model_inputs_size");
            ModelOutputsSize = OpenVinoRunner.GetProc<OvModelOutputsSize>(handle, "ov_model_outputs_size");
            Free = OpenVinoRunner.GetProc<OvFree>(handle, "ov_free");
            GetErrorInfo = OpenVinoRunner.GetProc<OvGetErrorInfo>(handle, "ov_get_error_info");
            lastErrMsg = OpenVinoRunner.GetProc<OvLastErrMsg>(handle, "ov_get_last_err_msg");

            int status = CoreCreate(out core);
            Check(status, "ov_core_create");
        }
        catch
        {
            FreeLibrary(handle);
            throw;
        }
    }

    public void Dispose()
    {
        if (!core.Equals(IntPtr.Zero))
            CoreFree(core);
        FreeLibrary(handle);
        GC.SuppressFinalize(this);
    }

    public void Check(int status, string operation)
    {
        if (status == 0)
            return;
        IntPtr msg = GetErrorInfo(status);
        string text = msg == IntPtr.Zero ? string.Empty : (Marshal.PtrToStringAnsi(msg) ?? string.Empty);
        string last = LastErrMsg();
        string detail = last.Trim().Length > 0 ? $" — {last.Trim()}" : string.Empty;
        throw new InvalidOperationException(
            $"{operation} failed: ov_status_e={status} ({(text.Trim().Length > 0 ? text.Trim() : "no message")}){detail}");
    }

    private string LastErrMsg()
    {
        try
        {
            if (lastErrMsg is null)
                return string.Empty;
            IntPtr msg = lastErrMsg();
            return msg == IntPtr.Zero ? string.Empty : (Marshal.PtrToStringAnsi(msg) ?? string.Empty);
        }
        catch
        {
            return string.Empty;
        }
    }

    public string OpenVinoVersion()
    {
        Check(GetOpenVinoVersion(out OvVersion version), "ov_get_openvino_version");
        try
        {
            string build = Marshal.PtrToStringAnsi(version.BuildNumber) ?? string.Empty;
            string desc = Marshal.PtrToStringAnsi(version.Description) ?? string.Empty;
            return $"{desc} {build}";
        }
        finally
        {
            VersionFree(ref version);
        }
    }

    public IntPtr ReadModel(string xmlPath, string binPath)
    {
        Check(CoreReadModel(core, xmlPath, binPath, out IntPtr model), "ov_core_read_model");
        return model;
    }

    public (int Inputs, int Outputs) ModelIoCounts(IntPtr model)
    {
        Check(ModelInputsSize(model, out UIntPtr inCount), "ov_model_inputs_size");
        Check(ModelOutputsSize(model, out UIntPtr outCount), "ov_model_outputs_size");
        return (checked((int)inCount.ToUInt64()), checked((int)outCount.ToUInt64()));
    }

    public IntPtr CompileModel(string device, IntPtr model)
    {
        Check(CoreCompileModel(core, model, device, UIntPtr.Zero, out IntPtr compiled), "ov_core_compile_model");
        return compiled;
    }

    public void SetProperty(string device, string key, string value)
    {
        IntPtr dev = Marshal.StringToCoTaskMemAnsi(device);
        IntPtr k = Marshal.StringToCoTaskMemAnsi(key);
        IntPtr v = Marshal.StringToCoTaskMemAnsi(value);
        try
        {
            int status = CoreSetProperty(core, dev, k, v);
            Check(status, $"ov_core_set_property({device}, {key}={value})");
        }
        finally
        {
            Marshal.FreeCoTaskMem(dev);
            Marshal.FreeCoTaskMem(k);
            Marshal.FreeCoTaskMem(v);
        }
    }

    public string GetCoreProperty(string device, string key)
    {
        IntPtr dev = Marshal.StringToCoTaskMemAnsi(device);
        IntPtr k = Marshal.StringToCoTaskMemAnsi(key);
        try
        {
            Check(CoreGetProperty(core, dev, k, out IntPtr value), $"ov_core_get_property({device}, {key})");
            return ReadAndFree(value);
        }
        finally
        {
            Marshal.FreeCoTaskMem(dev);
            Marshal.FreeCoTaskMem(k);
        }
    }

    public string GetCompiledModelProperty(IntPtr compiledModel, string key)
    {
        IntPtr k = Marshal.StringToCoTaskMemAnsi(key);
        try
        {
            Check(CompiledModelGetProperty(compiledModel, k, out IntPtr value), $"ov_compiled_model_get_property({key})");
            return ReadAndFree(value);
        }
        finally
        {
            Marshal.FreeCoTaskMem(k);
        }
    }

    public IntPtr CreateInferRequest(IntPtr compiledModel)
    {
        Check(CompiledModelCreateInferRequest(compiledModel, out IntPtr request), "ov_compiled_model_create_infer_request");
        return request;
    }

    public void SetInputTensor(IntPtr request, IntPtr tensor, ulong index = 0)
    {
        Check(InferRequestSetInputTensor(request, new UIntPtr(index), tensor), "ov_infer_request_set_input_tensor_by_index");
    }

    public IntPtr GetOutputTensor(IntPtr request, ulong index = 0)
    {
        Check(InferRequestGetOutputTensor(request, new UIntPtr(index), out IntPtr tensor), "ov_infer_request_get_output_tensor_by_index");
        return tensor;
    }

    public IntPtr GetInputTensor(IntPtr request, ulong index = 0)
    {
        Check(InferRequestGetInputTensor(request, new UIntPtr(index), out IntPtr tensor), "ov_infer_request_get_input_tensor_by_index");
        return tensor;
    }

    public void Infer(IntPtr request)
    {
        Check(InferFn(request), "ov_infer_request_infer");
    }

    public IntPtr CreateTensor(OvElementType type, ReadOnlySpan<long> dims)
    {
        IntPtr dimsPtr = IntPtr.Zero;
        if (dims.Length > 0)
        {
            dimsPtr = Marshal.AllocHGlobal(dims.Length * sizeof(long));
            Marshal.Copy(dims.ToArray(), 0, dimsPtr, dims.Length);
        }
        var shape = new OvShape { Rank = dims.Length, Dims = dimsPtr };
        try
        {
            Check(TensorCreate((int)type, shape, out IntPtr tensor), $"ov_tensor_create({type}, [{string.Join(',', dims.ToArray())}])");
            return tensor;
        }
        finally
        {
            if (dimsPtr != IntPtr.Zero)
                Marshal.FreeHGlobal(dimsPtr);
        }
    }

    public void FillTensor(IntPtr tensor, ReadOnlySpan<byte> data)
    {
        Check(TensorDataFn(tensor, out IntPtr dataPtr), "ov_tensor_data");
        if (dataPtr == IntPtr.Zero)
            throw new InvalidOperationException("ov_tensor_data returned null");
        Marshal.Copy(data.ToArray(), 0, dataPtr, data.Length);
    }

    public IntPtr TensorDataPtr(IntPtr tensor)
    {
        Check(TensorDataFn(tensor, out IntPtr dataPtr), "ov_tensor_data");
        if (dataPtr == IntPtr.Zero)
            throw new InvalidOperationException("ov_tensor_data returned null");
        return dataPtr;
    }

    public void TensorFree(IntPtr tensor)
    {
        if (tensor != IntPtr.Zero)
            TensorFreeFn(tensor);
    }

    public void InferRequestFree(IntPtr request)
    {
        if (request != IntPtr.Zero)
            InferRequestFreeFn(request);
    }

    public void CompiledModelFree(IntPtr compiledModel)
    {
        if (compiledModel != IntPtr.Zero)
            CompiledModelFreeFn(compiledModel);
    }

    public void ModelFree(IntPtr model)
    {
        if (model != IntPtr.Zero)
            ModelFreeFn(model);
    }

    public long[] GetShape(IntPtr tensor)
    {
        Check(TensorGetShape(tensor, out OvShape shape), "ov_tensor_get_shape");
        int rank = checked((int)shape.Rank);
        long[] result = new long[rank];
        try
        {
            if (rank > 0)
                Marshal.Copy(shape.Dims, result, 0, rank);
        }
        finally
        {
            ShapeFree(ref shape);
        }
        return result;
    }

    public OvElementType GetElementType(IntPtr tensor)
    {
        Check(TensorGetElementType(tensor, out int type), "ov_tensor_get_element_type");
        return (OvElementType)type;
    }

    public int ElementCount(IntPtr tensor)
    {
        long[] shape = GetShape(tensor);
        long count = 1;
        foreach (long dim in shape)
            count *= dim;
        return checked((int)count);
    }

    private string ReadAndFree(IntPtr value)
    {
        if (value == IntPtr.Zero)
            return string.Empty;
        string result = Marshal.PtrToStringAnsi(value) ?? string.Empty;
        Free(value);
        return result;
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int OvGetOpenVinoVersion(out OvVersion version);
    private OvGetOpenVinoVersion GetOpenVinoVersion = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void OvVersionFree(ref OvVersion version);
    private OvVersionFree VersionFree = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int OvCoreCreate(out IntPtr core);
    private OvCoreCreate CoreCreate = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void OvCoreFree(IntPtr core);
    private OvCoreFree CoreFree = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int OvCoreReadModel(
        IntPtr core,
        [MarshalAs(UnmanagedType.LPStr)] string modelPath,
        [MarshalAs(UnmanagedType.LPStr)] string weightsPath,
        out IntPtr model);
    private OvCoreReadModel CoreReadModel = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int OvCoreCompileModel(
        IntPtr core,
        IntPtr model,
        [MarshalAs(UnmanagedType.LPStr)] string device,
        UIntPtr propertyArgsSize,
        out IntPtr compiledModel);
    private OvCoreCompileModel CoreCompileModel = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int OvCoreSetProperty(IntPtr core, IntPtr device, IntPtr key, IntPtr value);
    private OvCoreSetProperty CoreSetProperty = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int OvCoreGetProperty(IntPtr core, IntPtr device, IntPtr key, out IntPtr value);
    private OvCoreGetProperty CoreGetProperty = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void OvModelFree(IntPtr model);
    private OvModelFree ModelFreeFn = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int OvModelInputsSize(IntPtr model, out UIntPtr size);
    private OvModelInputsSize ModelInputsSize = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int OvModelOutputsSize(IntPtr model, out UIntPtr size);
    private OvModelOutputsSize ModelOutputsSize = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void OvCompiledModelFree(IntPtr compiledModel);
    private OvCompiledModelFree CompiledModelFreeFn = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int OvCompiledModelGetProperty(IntPtr compiledModel, IntPtr key, out IntPtr value);
    private OvCompiledModelGetProperty CompiledModelGetProperty = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int OvCompiledModelCreateInferRequest(IntPtr compiledModel, out IntPtr request);
    private OvCompiledModelCreateInferRequest CompiledModelCreateInferRequest = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void OvInferRequestFree(IntPtr request);
    private OvInferRequestFree InferRequestFreeFn = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int OvInferRequestSetInputTensor(IntPtr request, UIntPtr index, IntPtr tensor);
    private OvInferRequestSetInputTensor InferRequestSetInputTensor = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int OvInferRequestGetOutputTensor(IntPtr request, UIntPtr index, out IntPtr tensor);
    private OvInferRequestGetOutputTensor InferRequestGetOutputTensor = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int OvInferRequestGetInputTensor(IntPtr request, UIntPtr index, out IntPtr tensor);
    private OvInferRequestGetInputTensor InferRequestGetInputTensor = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int OvInfer(IntPtr request);
    private OvInfer InferFn = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int OvTensorCreate(int type, OvShape shape, out IntPtr tensor);
    private OvTensorCreate TensorCreate = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int OvTensorData(IntPtr tensor, out IntPtr data);
    private OvTensorData TensorDataFn = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int OvTensorGetShape(IntPtr tensor, out OvShape shape);
    private OvTensorGetShape TensorGetShape = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int OvTensorGetElementType(IntPtr tensor, out int type);
    private OvTensorGetElementType TensorGetElementType = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void OvShapeFree(ref OvShape shape);
    private OvShapeFree ShapeFree = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void OvTensorFree(IntPtr tensor);
    private OvTensorFree TensorFreeFn = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void OvFree(IntPtr content);
    private OvFree Free = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate IntPtr OvGetErrorInfo(int status);
    private OvGetErrorInfo GetErrorInfo = null!;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate IntPtr OvLastErrMsg();
    private OvLastErrMsg? lastErrMsg = null;

    [DllImport("kernel32", SetLastError = true)]
    private static extern bool FreeLibrary(IntPtr hModule);
}