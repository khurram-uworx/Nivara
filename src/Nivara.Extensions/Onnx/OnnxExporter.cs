using Google.Protobuf;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.Diagnostics;
using Onnx;
using System.Numerics;

namespace Nivara.Onnx;

public sealed class OnnxExportOptions
{
    public int[] InputShape { get; set; } = Array.Empty<int>();
    public int OpsetVersion { get; set; } = 20;
    public string ProducerName { get; set; } = "Nivara";
    public string ProducerVersion { get; set; } = "2.0.0";
    public string GraphName { get; set; } = "main_graph";
    /// <summary>
    /// When true, attempt to preserve the module's native element type in the exported
    /// graph instead of the F32 default. Default false (F32-friendly widening).
    /// </summary>
    public bool NativeDtype
    {
        get => nativeDtype;
        set => nativeDtype = value;
    }
    public bool UseNativeDtype { get => nativeDtype; set => nativeDtype = value; }
    public bool ExportNativeDtype { get => nativeDtype; set => nativeDtype = value; }
    bool nativeDtype;
}

public static class OnnxExporter
{
    public static void Export<T>(Module<T> module, Stream stream, OnnxExportOptions? options = null)
        where T : struct, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(module);
        ArgumentNullException.ThrowIfNull(stream);
        options ??= new OnnxExportOptions();

        var model = new ModelProto();
        model.IrVersion = 9;
        model.ProducerName = options.ProducerName;
        model.ProducerVersion = options.ProducerVersion;
        model.Domain = "";
        model.ModelVersion = 1;
        model.DocString = "Exported from Nivara AutoDiff";

        var opset = new OperatorSetIdProto { Domain = "", Version = options.OpsetVersion };
        model.OpsetImport.Add(opset);

        var graph = new GraphProto { Name = options.GraphName };
        model.Graph = graph;

        var inputShape = options.InputShape.Length > 0
            ? options.InputShape
            : new int[Math.Max(1, module is TransformerBlock<T> ? 2 : 2)];
        if (inputShape.Length == 0) inputShape = new[] { 1, 4 };

        var inputName = "input";
        var inputInfo = new ValueInfoProto { Name = inputName };
        inputInfo.Type = new TypeProto
        {
            TensorType = new TypeProto.Types.Tensor { ElemType = (int)ToOnnxDataType<T>(), Shape = MakeShape(inputShape) }
        };
        graph.Input.Add(inputInfo);

        var ctx = new Context<T>(graph, inputShape);
        ctx.SetShape(inputName, inputShape.ToArray());

        var exportEnabled = DiagnosticsTracker.IsEnabled || OnnxExportDiagnostics.Enabled;
        var measurement = exportEnabled ? DiagnosticsTracker.StartMeasurement() : default;
        Exception? exportFailure = null;
        try
        {
            var outputName = module.ExportToOnnx(ctx, inputName);
            var outputShape = ctx.GetShape(outputName);
            var outputInfo = new ValueInfoProto { Name = outputName };
            outputInfo.Type = new TypeProto
            {
                TensorType = new TypeProto.Types.Tensor { ElemType = (int)ToOnnxDataType<T>(), Shape = MakeShape(outputShape) }
            };
            graph.Output.Add(outputInfo);

            stream.Write(model.ToByteArray());
        }
        catch (Exception ex)
        {
            exportFailure = ex;
            throw;
        }
        finally
        {
            var notes = exportFailure is null
                ? "OnnxExport=Module;Emitters=recorded"
                : $"OnnxExport=Failed;Emitter=Unsupported;Reason={exportFailure.Message}";
            measurement.Record("OnnxExport", KernelType.Scalar, 0, typeof(T), false, notes);
        }
    }

    static TensorProto.Types.DataType ToOnnxDataType<T>() where T : struct, IFloatingPointIeee754<T> =>
        typeof(T) == typeof(float) ? TensorProto.Types.DataType.Float :
        typeof(T) == typeof(double) ? TensorProto.Types.DataType.Double :
        typeof(T) == typeof(Half) ? TensorProto.Types.DataType.Float16 :
        typeof(T) == typeof(BFloat16) ? TensorProto.Types.DataType.Bfloat16 :
        TensorProto.Types.DataType.Float;

    internal static TensorShapeProto MakeShape(int[] shape)
    {
        var ts = new TensorShapeProto();
        foreach (var d in shape)
        {
            var dim = new TensorShapeProto.Types.Dimension();
            if (d <= 0)
                dim.DimParam = "dim";
            else
                dim.DimValue = d;
            ts.Dim.Add(dim);
        }
        return ts;
    }

    sealed class Context<T> : IOnnxExportContext<T> where T : struct, IFloatingPointIeee754<T>
    {
        readonly GraphProto graph;
        readonly Dictionary<string, int[]> shapes = new();
        int counter;
        public Context(GraphProto graph, int[] inputShape) { this.graph = graph; }

        public void SetShape(string name, int[] shape) => shapes[name] = shape.ToArray();
        public int[] GetShape(string name) => shapes.TryGetValue(name, out var s) ? s : Array.Empty<int>();

        string New(string prefix) => $"{prefix}_{counter++}";

        string AddNode(string opType, string[] inputs, string[] outputs, IEnumerable<AttributeProto>? attrs = null)
        {
            var node = new NodeProto { OpType = opType };
            foreach (var i in inputs) node.Input.Add(i);
            foreach (var o in outputs) node.Output.Add(o);
            if (attrs != null)
                foreach (var a in attrs) node.Attribute.Add(a);
            graph.Node.Add(node);
            return outputs[0];
        }

        string AddInitializer(string name, T[] data, int[] shape)
        {
            var tp = new TensorProto { Name = name, DataType = (int)ToOnnxDataType<T>() };
            foreach (var d in shape) tp.Dims.Add(d);
            var bytes = new byte[data.Length * SizeOf()];
            Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
            tp.RawData = ByteString.CopyFrom(bytes);
            graph.Initializer.Add(tp);
            return name;
        }

        int SizeOf() =>
            typeof(T) == typeof(double) ? 8 :
            typeof(T) == typeof(Half) ? 2 :
            typeof(T) == typeof(BFloat16) ? 2 : 4;

        public string Linear(string inputName, Linear<T> module)
        {
            var __m = (DiagnosticsTracker.IsEnabled || OnnxExportDiagnostics.Enabled) ? DiagnosticsTracker.StartMeasurement() : default;
            var wName = New("weight");
            var wShape = module.Weight?.Tensor.Shape ?? new[] { module.OutFeatures, module.InFeatures };
            var wData = new T[module.Weight!.Tensor.Length];
            module.Weight!.Tensor.Data.CopyTo(wData, T.Zero);
            AddInitializer(wName, wData, wShape);

            var inputs = new List<string> { inputName, wName };
            string? bName = null;
            if (module.Bias != null)
            {
                bName = New("bias");
                var bShape = module.Bias!.Tensor.Shape;
                var bData = new T[module.Bias!.Tensor.Length];
                module.Bias!.Tensor.Data.CopyTo(bData, T.Zero);
                AddInitializer(bName, bData, bShape);
                inputs.Add(bName);
            }

            var outName = New("linear_out");
            var attrs = new List<AttributeProto>
            {
                new AttributeProto { Name = "alpha", Type = AttributeProto.Types.AttributeType.Float, F = 1f },
                new AttributeProto { Name = "beta", Type = AttributeProto.Types.AttributeType.Float, F = 1f },
                new AttributeProto { Name = "transA", Type = AttributeProto.Types.AttributeType.Int, I = 0 },
                new AttributeProto { Name = "transB", Type = AttributeProto.Types.AttributeType.Int, I = 1 },
            };
            AddNode("Gemm", inputs.ToArray(), new[] { outName }, attrs);

            var inShape = GetShape(inputName);
            var outShape = inShape.Length >= 2
                ? new[] { inShape[0], module.OutFeatures }
                : new[] { module.OutFeatures };
            SetShape(outName, outShape);
            RecordOnnxExportDiagnostic(__m, "Linear", "Gemm", 0);
            return outName;
        }

        public string Linear<TWeight>(string inputName, Linear<T, TWeight> module)
            where TWeight : struct, IFloatingPointIeee754<TWeight>
        {
            var __m = (DiagnosticsTracker.IsEnabled || OnnxExportDiagnostics.Enabled) ? DiagnosticsTracker.StartMeasurement() : default;
            var wName = New("weight");
            var wShape = module.Weight?.Tensor.Shape ?? new[] { module.OutFeatures, module.InFeatures };
            var wStorage = new TWeight[module.Weight!.Tensor.Length];
            module.Weight!.Tensor.Data.CopyTo(wStorage, TWeight.Zero);
            var wData = new T[wStorage.Length];
            for (int i = 0; i < wData.Length; i++)
                wData[i] = T.CreateChecked(wStorage[i]);
            AddInitializer(wName, wData, wShape);

            var inputs = new List<string> { inputName, wName };
            string? bName = null;
            if (module.Bias != null)
            {
                bName = New("bias");
                var bShape = module.Bias!.Tensor.Shape;
                var bData = new T[module.Bias!.Tensor.Length];
                module.Bias!.Tensor.Data.CopyTo(bData, T.Zero);
                AddInitializer(bName, bData, bShape);
                inputs.Add(bName);
            }

            var outName = New("linear_out");
            var attrs = new List<AttributeProto>
            {
                new AttributeProto { Name = "alpha", Type = AttributeProto.Types.AttributeType.Float, F = 1f },
                new AttributeProto { Name = "beta", Type = AttributeProto.Types.AttributeType.Float, F = 1f },
                new AttributeProto { Name = "transA", Type = AttributeProto.Types.AttributeType.Int, I = 0 },
                new AttributeProto { Name = "transB", Type = AttributeProto.Types.AttributeType.Int, I = 1 },
            };
            AddNode("Gemm", inputs.ToArray(), new[] { outName }, attrs);

            var inShape = GetShape(inputName);
            var outShape = inShape.Length >= 2
                ? new[] { inShape[0], module.OutFeatures }
                : new[] { module.OutFeatures };
            SetShape(outName, outShape);
            RecordOnnxExportDiagnostic(__m, "Linear", "Gemm", 0);
            return outName;
        }

        public string Conv1d(string inputName, Conv1d<T> module)
        {
            var __m = (DiagnosticsTracker.IsEnabled || OnnxExportDiagnostics.Enabled) ? DiagnosticsTracker.StartMeasurement() : default;
            var wName = New("conv1d_weight");
            var wShape = new[] { module.OutChannels, module.InChannels, module.KernelSize };
            var wData = new T[module.Weight!.Tensor.Length];
            module.Weight!.Tensor.Data.CopyTo(wData, T.Zero);
            AddInitializer(wName, wData, wShape);

            var inputs = new List<string> { inputName, wName };
            if (module.Bias != null)
            {
                var bName = New("conv1d_bias");
                var bData = new T[module.Bias!.Tensor.Length];
                module.Bias!.Tensor.Data.CopyTo(bData, T.Zero);
                AddInitializer(bName, bData, module.Bias!.Tensor.Shape);
                inputs.Add(bName);
            }
            var outName = New("conv1d_out");
            var attrs = new List<AttributeProto>
            {
                new AttributeProto { Name="kernel_shape", Type=AttributeProto.Types.AttributeType.Ints, Ints = { module.KernelSize } },
                new AttributeProto { Name="strides", Type=AttributeProto.Types.AttributeType.Ints, Ints = { module.Stride } },
                new AttributeProto { Name="pads", Type=AttributeProto.Types.AttributeType.Ints, Ints = { module.Padding, module.Padding } },
                new AttributeProto { Name="group", Type=AttributeProto.Types.AttributeType.Int, I = 1 },
            };
            AddNode("Conv", inputs.ToArray(), new[] { outName }, attrs);
            var inShape1d = GetShape(inputName);
            if (inShape1d.Length >= 3 && inShape1d[0] > 0 && inShape1d[2] > 0)
            {
                int n = inShape1d[0];
                int l = inShape1d[2];
                int oL = (l + 2 * module.Padding - module.KernelSize) / module.Stride + 1;
                if (oL <= 0) oL = l;
                SetShape(outName, new[] { n, module.OutChannels, oL });
            }
            RecordOnnxExportDiagnostic(__m, "Conv1d", "Conv", 0);
            return outName;
        }

        public string Conv2d(string inputName, Conv2d<T> module)
        {
            var __m = (DiagnosticsTracker.IsEnabled || OnnxExportDiagnostics.Enabled) ? DiagnosticsTracker.StartMeasurement() : default;
            ArgumentNullException.ThrowIfNull(module);
            if (module.Groups <= 0
                || module.InChannels % module.Groups != 0
                || module.OutChannels % module.Groups != 0)
            {
                RecordOnnxExportDiagnostic(__m, "Conv2d", "Unsupported", 0, true, "Conv2d export requires groups > 0 with channels divisible by groups.");
                throw new NotSupportedException("Conv2d export requires groups > 0 with channels divisible by groups.");
            }
            if (module.KernelSize <= 0 || module.Stride <= 0)
            {
                RecordOnnxExportDiagnostic(__m, "Conv2d", "Unsupported", 0, true, "Conv2d export requires positive kernel size and stride.");
                throw new NotSupportedException("Conv2d export requires positive kernel size and stride.");
            }
            if (module.PaddingTop < 0 || module.PaddingBottom < 0
                || module.PaddingLeft < 0 || module.PaddingRight < 0)
            {
                RecordOnnxExportDiagnostic(__m, "Conv2d", "Unsupported", 0, true, "Conv2d export requires non-negative padding.");
                throw new NotSupportedException("Conv2d export requires non-negative padding.");
            }
            var wName = New("conv2d_weight");
            var wShape = new[] { module.OutChannels, module.InChannels / module.Groups, module.KernelSize, module.KernelSize };
            var wData = new T[module.Weight!.Tensor.Length];
            module.Weight!.Tensor.Data.CopyTo(wData, T.Zero);
            AddInitializer(wName, wData, wShape);

            var inputs = new List<string> { inputName, wName };
            if (module.Bias != null)
            {
                var bName = New("conv2d_bias");
                var bData = new T[module.Bias!.Tensor.Length];
                module.Bias!.Tensor.Data.CopyTo(bData, T.Zero);
                AddInitializer(bName, bData, module.Bias!.Tensor.Shape);
                inputs.Add(bName);
            }
            var outName = New("conv2d_out");
            var k = module.KernelSize;
            var attrs = new List<AttributeProto>
            {
                new AttributeProto { Name="kernel_shape", Type=AttributeProto.Types.AttributeType.Ints, Ints = { k, k } },
                new AttributeProto { Name="strides", Type=AttributeProto.Types.AttributeType.Ints, Ints = { module.Stride, module.Stride } },
                new AttributeProto { Name="pads", Type=AttributeProto.Types.AttributeType.Ints, Ints = { module.PaddingTop, module.PaddingLeft, module.PaddingBottom, module.PaddingRight } },
                new AttributeProto { Name="group", Type=AttributeProto.Types.AttributeType.Int, I = module.Groups },
            };
            AddNode("Conv", inputs.ToArray(), new[] { outName }, attrs);
            var inShape2d = GetShape(inputName);
            if (inShape2d.Length >= 4 && inShape2d[0] > 0 && inShape2d[2] > 0 && inShape2d[3] > 0)
            {
                int n = inShape2d[0];
                int h = inShape2d[2];
                int w = inShape2d[3];
                int oH = (h + module.PaddingTop + module.PaddingBottom - module.KernelSize) / module.Stride + 1;
                int oW = (w + module.PaddingLeft + module.PaddingRight - module.KernelSize) / module.Stride + 1;
                if (oH <= 0 || oW <= 0)
                {
                    RecordOnnxExportDiagnostic(__m, "Conv2d", "Unsupported", 0, true, "Conv2d export produced non-positive output dimensions ({oH}x{oW}).");
                    throw new NotSupportedException($"Conv2d export produced non-positive output dimensions ({oH}x{oW}).");
                }
                SetShape(outName, new[] { n, module.OutChannels, oH, oW });
            }
            RecordOnnxExportDiagnostic(__m, "Conv2d", "Conv", 0);
            return outName;
        }

        public string Embedding(string inputName, Embedding<T> module)
        {
            var __m = (DiagnosticsTracker.IsEnabled || OnnxExportDiagnostics.Enabled) ? DiagnosticsTracker.StartMeasurement() : default;
            // Cast indices to int64 when needed
            var castName = inputName;
            if (typeof(T) != typeof(long) && typeof(T) != typeof(int))
            {
                castName = New("indices_int64");
                AddNode("Cast", new[] { inputName }, new[] { castName },
                    new[] { new AttributeProto { Name = "to", Type = AttributeProto.Types.AttributeType.Int, I = (int)TensorProto.Types.DataType.Int64 } });
            }
            var wName = New("embedding_weight");
            var wData = new T[module.Weight!.Tensor.Length];
            module.Weight!.Tensor.Data.CopyTo(wData, T.Zero);
            AddInitializer(wName, wData, module.Weight!.Tensor.Shape);
            var outName = New("embedding_out");
            AddNode("Gather", new[] { wName, castName }, new[] { outName },
                new[] { new AttributeProto { Name = "axis", Type = AttributeProto.Types.AttributeType.Int, I = 0 } });
            var inShapeEmb = GetShape(inputName);
            int[] outShapeEmb;
            if (inShapeEmb.Length == 0)
                outShapeEmb = new[] { module.EmbeddingDim };
            else
            {
                outShapeEmb = new int[inShapeEmb.Length + 1];
                Array.Copy(inShapeEmb, outShapeEmb, inShapeEmb.Length);
                outShapeEmb[inShapeEmb.Length] = module.EmbeddingDim;
            }
            SetShape(outName, outShapeEmb);
            RecordOnnxExportDiagnostic(__m, "Embedding", "Gather", 0);
            return outName;
        }

        public string Embedding<TWeight>(string inputName, Embedding<T, TWeight> module)
            where TWeight : struct, IFloatingPointIeee754<TWeight>
        {
            var __m = (DiagnosticsTracker.IsEnabled || OnnxExportDiagnostics.Enabled) ? DiagnosticsTracker.StartMeasurement() : default;
            var castName = inputName;
            if (typeof(T) != typeof(long) && typeof(T) != typeof(int))
            {
                castName = New("indices_int64");
                AddNode("Cast", new[] { inputName }, new[] { castName },
                    new[] { new AttributeProto { Name = "to", Type = AttributeProto.Types.AttributeType.Int, I = (int)TensorProto.Types.DataType.Int64 } });
            }
            var wName = New("embedding_weight");
            var wStorage = new TWeight[module.Weight!.Tensor.Length];
            module.Weight!.Tensor.Data.CopyTo(wStorage, TWeight.Zero);
            var wData = new T[wStorage.Length];
            for (int i = 0; i < wData.Length; i++)
                wData[i] = T.CreateChecked(wStorage[i]);
            AddInitializer(wName, wData, module.Weight!.Tensor.Shape);
            var outName = New("embedding_out");
            AddNode("Gather", new[] { wName, castName }, new[] { outName },
                new[] { new AttributeProto { Name = "axis", Type = AttributeProto.Types.AttributeType.Int, I = 0 } });
            var inShapeEmb = GetShape(inputName);
            int[] outShapeEmb;
            if (inShapeEmb.Length == 0)
                outShapeEmb = new[] { module.EmbeddingDim };
            else
            {
                outShapeEmb = new int[inShapeEmb.Length + 1];
                Array.Copy(inShapeEmb, outShapeEmb, inShapeEmb.Length);
                outShapeEmb[inShapeEmb.Length] = module.EmbeddingDim;
            }
            SetShape(outName, outShapeEmb);
            RecordOnnxExportDiagnostic(__m, "Embedding", "Gather", 0);
            return outName;
        }

        public string BatchNorm1d(string inputName, BatchNorm1d<T> module)
        {
            var __m = (DiagnosticsTracker.IsEnabled || OnnxExportDiagnostics.Enabled) ? DiagnosticsTracker.StartMeasurement() : default;
            if (!module.TrackRunningStats)
            {
                RecordOnnxExportDiagnostic(__m, "BatchNorm1d", "Unsupported", 0, true, "BatchNorm1d export requires trackRunningStats.");
                throw new NotSupportedException("BatchNorm1d export requires trackRunningStats.");
            }
            var scaleName = New("bn_scale");
            var biasName = New("bn_bias");
            var meanName = New("bn_mean");
            var varName = New("bn_var");
            T[] scaleData;
            T[] biasData;
            int[] scaleShape;
            if (module.Weight != null)
            {
                scaleData = new T[module.Weight!.Tensor.Length];
                module.Weight!.Tensor.Data.CopyTo(scaleData, T.Zero);
                scaleShape = module.Weight!.Tensor.Shape;
            }
            else
            {
                scaleData = new T[module.NumFeatures];
                for (int i = 0; i < scaleData.Length; i++) scaleData[i] = T.One;
                scaleShape = new[] { module.NumFeatures };
            }
            if (module.Bias != null)
            {
                biasData = new T[module.Bias!.Tensor.Length];
                module.Bias!.Tensor.Data.CopyTo(biasData, T.Zero);
            }
            else
            {
                biasData = new T[module.NumFeatures];
            }
            AddInitializer(scaleName, scaleData, scaleShape);
            AddInitializer(biasName, biasData, new[] { module.NumFeatures });
            AddInitializer(meanName, ToArray(module.RunningMean), new[] { module.NumFeatures });
            AddInitializer(varName, ToArray(module.RunningVar), new[] { module.NumFeatures });

            var outName = New("bn_out");
            AddNode("BatchNormalization", new[] { inputName, scaleName, biasName, meanName, varName }, new[] { outName },
                new[] { new AttributeProto { Name = "epsilon", Type = AttributeProto.Types.AttributeType.Float, F = float.CreateChecked(module.Eps) } });
            SetShape(outName, GetShape(inputName));
            RecordOnnxExportDiagnostic(__m, "BatchNorm1d", "BatchNormalization", 0);
            return outName;
        }

        public string BatchNorm2d(string inputName, BatchNorm2d<T> module, int? n = null, int? h = null, int? w = null)
        {
            var __m = (DiagnosticsTracker.IsEnabled || OnnxExportDiagnostics.Enabled) ? DiagnosticsTracker.StartMeasurement() : default;
            if (!module.TrackRunningStats)
            {
                RecordOnnxExportDiagnostic(__m, "BatchNorm2d", "Unsupported", 0, true, "BatchNorm2d export requires trackRunningStats.");
                throw new NotSupportedException("BatchNorm2d export requires trackRunningStats.");
            }
            string scaleName = New("bn_scale");
            string biasName = New("bn_bias");
            string meanName = New("bn_mean");
            string varName = New("bn_var");

            T[] scaleData;
            T[] biasData;
            int[] scaleShape = new[] { module.NumFeatures };
            if (module.Weight != null)
            {
                scaleData = new T[module.Weight!.Tensor.Length];
                module.Weight!.Tensor.Data.CopyTo(scaleData, T.Zero);
                scaleShape = module.Weight!.Tensor.Shape;
            }
            else
            {
                scaleData = new T[module.NumFeatures];
                for (int i = 0; i < scaleData.Length; i++) scaleData[i] = T.One;
            }
            if (module.Bias != null)
            {
                biasData = new T[module.Bias!.Tensor.Length];
                module.Bias!.Tensor.Data.CopyTo(biasData, T.Zero);
            }
            else
            {
                biasData = new T[module.NumFeatures];
            }

            AddInitializer(scaleName, scaleData, scaleShape);
            AddInitializer(biasName, biasData, new[] { module.NumFeatures });
            AddInitializer(meanName, ToArray(module.RunningMean), new[] { module.NumFeatures });
            AddInitializer(varName, ToArray(module.RunningVar), new[] { module.NumFeatures });

            var outName = New("bn_out");
            AddNode("BatchNormalization", new[] { inputName, scaleName, biasName, meanName, varName }, new[] { outName },
                new[] { new AttributeProto { Name = "epsilon", Type = AttributeProto.Types.AttributeType.Float, F = float.CreateChecked(module.Eps) } });
            SetShape(outName, GetShape(inputName));
            RecordOnnxExportDiagnostic(__m, "BatchNorm2d", "BatchNormalization", 0);
            return outName;
        }

        public string LayerNorm(string inputName, LayerNorm<T> module)
        {
            var __m = (DiagnosticsTracker.IsEnabled || OnnxExportDiagnostics.Enabled) ? DiagnosticsTracker.StartMeasurement() : default;
            string scaleName;
            string biasName;
            if (module.Affine)
            {
                scaleName = New("ln_scale");
                biasName = New("ln_bias");
                var scaleData = new T[module.Weight!.Tensor.Length]; module.Weight!.Tensor.Data.CopyTo(scaleData, T.Zero);
                AddInitializer(scaleName, scaleData, module.Weight!.Tensor.Shape);
                if (module.Bias != null)
                {
                    var biasData = new T[module.Bias!.Tensor.Length]; module.Bias!.Tensor.Data.CopyTo(biasData, T.Zero);
                    AddInitializer(biasName, biasData, module.Bias!.Tensor.Shape);
                }
                else
                {
                    var biasData = new T[module.NormalizedShape];
                    AddInitializer(biasName, biasData, new[] { module.NormalizedShape });
                }
            }
            else
            {
                scaleName = New("ln_scale");
                biasName = New("ln_bias");
                var scaleData = new T[module.NormalizedShape];
                var biasData = new T[module.NormalizedShape];
                for (int i = 0; i < module.NormalizedShape; i++) { scaleData[i] = T.One; biasData[i] = T.Zero; }
                AddInitializer(scaleName, scaleData, new[] { module.NormalizedShape });
                AddInitializer(biasName, biasData, new[] { module.NormalizedShape });
            }

            var outName = New("ln_out");
            AddNode("LayerNormalization", new[] { inputName, scaleName, biasName }, new[] { outName },
                new[] {
                    new AttributeProto { Name = "axis", Type = AttributeProto.Types.AttributeType.Int, I = -1 },
                    new AttributeProto { Name = "epsilon", Type = AttributeProto.Types.AttributeType.Float, F = float.CreateChecked(module.Eps) }
                });
            SetShape(outName, GetShape(inputName));
            RecordOnnxExportDiagnostic(__m, "LayerNorm", "LayerNormalization", 0);
            return outName;
        }

        public string RMSNorm(string inputName, RMSNorm<T> module)
        {
            var __m = (DiagnosticsTracker.IsEnabled || OnnxExportDiagnostics.Enabled) ? DiagnosticsTracker.StartMeasurement() : default;
            // sq
            var sq = New("rms_sq");
            AddNode("Mul", new[] { inputName, inputName }, new[] { sq });
            // reduce mean over last axis
            var axesName = New("rms_axes");
            var axesTp = new TensorProto { Name = axesName, DataType = (int)TensorProto.Types.DataType.Int64 };
            axesTp.Dims.Add(1);
            var axesBytes = new byte[8];
            Buffer.BlockCopy(new long[] { -1 }, 0, axesBytes, 0, 8);
            axesTp.RawData = ByteString.CopyFrom(axesBytes);
            graph.Initializer.Add(axesTp);
            SetShape(axesName, new[] { 1 });
            var mean = New("rms_mean");
            AddNode("ReduceMean", new[] { sq, axesName }, new[] { mean },
                new[] { new AttributeProto { Name = "keepdims", Type = AttributeProto.Types.AttributeType.Int, I = 1 } });
            var epsName = New("rms_eps");
            AddInitializer(epsName, new[] { T.CreateChecked(module.Eps) }, new[] { 1 });
            var meanEps = New("rms_mean_eps");
            AddNode("Add", new[] { mean, epsName }, new[] { meanEps });
            var rms = New("rms_sqrt");
            AddNode("Sqrt", new[] { meanEps }, new[] { rms });
            var div = New("rms_div");
            AddNode("Div", new[] { inputName, rms }, new[] { div });
            var scaleName = New("rms_scale");
            var scaleData = new T[module.Weight!.Tensor.Length];
            module.Weight!.Tensor.Data.CopyTo(scaleData, T.Zero);
            AddInitializer(scaleName, scaleData, module.Weight!.Tensor.Shape);
            var outName = New("rms_out");
            AddNode("Mul", new[] { div, scaleName }, new[] { outName });
            SetShape(outName, GetShape(inputName));
            RecordOnnxExportDiagnostic(__m, "RMSNorm", "RmsNormPattern", 0);
            return outName;
        }

        public string TransformerBlock(string inputName, TransformerBlock<T> module, int sequenceLength)
        {
            var __m = (DiagnosticsTracker.IsEnabled || OnnxExportDiagnostics.Enabled) ? DiagnosticsTracker.StartMeasurement() : default;
            var xResidual = inputName;
            var inShape = GetShape(inputName);
            int L = inShape.Length > 0 && inShape[0] > 0 ? inShape[0] : sequenceLength;
            int D = inShape.Length > 1 && inShape[1] > 0 ? inShape[1] : module.NEmbd;
            int H = module.NHead;
            int Hd = module.HeadDim;

            T eps = T.CreateChecked(1e-5);

            string NormUnparam(string x)
            {
                if (module.NormType == NormType.RMSNorm)
                {
                    var sq = New("rms_sq"); AddNode("Mul", new[] { x, x }, new[] { sq }); SetShape(sq, GetShape(x));
                    var axesName = New("rms_axes");
                    var axesTp = new TensorProto { Name = axesName, DataType = (int)TensorProto.Types.DataType.Int64 };
                    axesTp.Dims.Add(1);
                    var axesBytes = new byte[8];
                    Buffer.BlockCopy(new long[] { -1 }, 0, axesBytes, 0, 8);
                    axesTp.RawData = ByteString.CopyFrom(axesBytes);
                    graph.Initializer.Add(axesTp);
                    SetShape(axesName, new[] { 1 });
                    var mean = New("rms_mean");
                    AddNode("ReduceMean", new[] { sq, axesName }, new[] { mean },
                        new[] { new AttributeProto { Name = "keepdims", Type = AttributeProto.Types.AttributeType.Int, I = 1 } });
                    SetShape(mean, new[] { GetShape(x)[0], 1 });
                    var epsName = New("rms_eps"); AddInitializer(epsName, new[] { eps }, new[] { 1 });
                    var meanEps = New("rms_mean_eps"); AddNode("Add", new[] { mean, epsName }, new[] { meanEps }); SetShape(meanEps, GetShape(mean));
                    var rms = New("rms_sqrt"); AddNode("Sqrt", new[] { meanEps }, new[] { rms }); SetShape(rms, GetShape(mean));
                    var div = New("rms_div"); AddNode("Div", new[] { x, rms }, new[] { div }); SetShape(div, GetShape(x));
                    RecordOnnxExportDiagnostic(__m, "TransformerBlock", "TransformerBlock", 0);
                    return div;
                }
                else
                {
                    var scaleName = New("ln_scale");
                    var biasName = New("ln_bias");
                    var scaleData = new T[D]; var biasData = new T[D];
                    for (int i = 0; i < D; i++) { scaleData[i] = T.One; biasData[i] = T.Zero; }
                    AddInitializer(scaleName, scaleData, new[] { D });
                    AddInitializer(biasName, biasData, new[] { D });
                    var outName = New("ln_out");
                    AddNode("LayerNormalization", new[] { x, scaleName, biasName }, new[] { outName },
                        new[] { new AttributeProto { Name = "axis", Type = AttributeProto.Types.AttributeType.Int, I = -1 },
                                new AttributeProto { Name = "epsilon", Type = AttributeProto.Types.AttributeType.Float, F = 1e-5f } });
                    SetShape(outName, GetShape(x));
                    RecordOnnxExportDiagnostic(__m, "TransformerBlock", "TransformerBlock", 0);
                    return outName;
                }
            }

            string AddMaskInt(int Llen)
            {
                var name = New("causal_mask");
                var data = new T[Llen * Llen];
                for (int i = 0; i < Llen; i++)
                    for (int j = 0; j < Llen; j++)
                        if (j > i) data[i * Llen + j] = T.CreateChecked(double.NegativeInfinity);
                var bytes = new byte[data.Length * SizeOf()];
                Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
                var tp = new TensorProto { Name = name, DataType = (int)ToOnnxDataType<T>() };
                tp.Dims.Add(Llen); tp.Dims.Add(Llen);
                tp.RawData = ByteString.CopyFrom(bytes);
                graph.Initializer.Add(tp);
                SetShape(name, new[] { Llen, Llen });
                RecordOnnxExportDiagnostic(__m, "TransformerBlock", "TransformerBlock", 0);
                return name;
            }

            string ReshapeCall(string x, int[] newShape)
            {
                var initName = New("shape_const");
                var tp = new TensorProto { Name = initName, DataType = (int)TensorProto.Types.DataType.Int64 };
                tp.Dims.Add(newShape.Length);
                var bytes = new byte[newShape.Length * 8];
                var longs = newShape.Select(v => (long)v).ToArray();
                Buffer.BlockCopy(longs, 0, bytes, 0, bytes.Length);
                tp.RawData = ByteString.CopyFrom(bytes);
                graph.Initializer.Add(tp);
                SetShape(initName, new[] { newShape.Length });
                var outName = New("reshape");
                AddNode("Reshape", new[] { x, initName }, new[] { outName });
                SetShape(outName, newShape);
                RecordOnnxExportDiagnostic(__m, "TransformerBlock", "TransformerBlock", 0);
                return outName;
            }

            string TransposeCall(string x, int[] perm)
            {
                var attr = new AttributeProto { Name = "perm", Type = AttributeProto.Types.AttributeType.Ints };
                attr.Ints.AddRange(perm.Select(v => (long)v));
                var outName = New("transpose");
                AddNode("Transpose", new[] { x }, new[] { outName }, new[] { attr });
                RecordOnnxExportDiagnostic(__m, "TransformerBlock", "TransformerBlock", 0);
                return outName;
            }

            var xN1 = NormUnparam(inputName);
            var Q = Linear(xN1, module.QProj);
            var K = Linear(xN1, module.KProj);
            var V = Linear(xN1, module.VProj);

            var Qr = ReshapeCall(Q, new[] { L, H, Hd });
            var Kr = ReshapeCall(K, new[] { L, H, Hd });
            var Vr = ReshapeCall(V, new[] { L, H, Hd });
            var Qh = TransposeCall(Qr, new[] { 1, 0, 2 });
            var Kh = TransposeCall(Kr, new[] { 1, 0, 2 });
            var Vh = TransposeCall(Vr, new[] { 1, 0, 2 });
            var KhT = TransposeCall(Kh, new[] { 0, 2, 1 });

            var scores = New("scores");
            AddNode("MatMul", new[] { Qh, KhT }, new[] { scores });
            SetShape(scores, new[] { H, L, L });

            var scaleInit = New("attn_scale");
            AddInitializer(scaleInit, new[] { T.CreateChecked(1.0 / Math.Sqrt(Hd)) }, new[] { 1 });
            var scoresScaled = New("scores_scaled");
            AddNode("Mul", new[] { scores, scaleInit }, new[] { scoresScaled });
            SetShape(scoresScaled, new[] { H, L, L });

            var maskName = AddMaskInt(L);
            var scoresMasked = New("scores_masked");
            AddNode("Add", new[] { scoresScaled, maskName }, new[] { scoresMasked });
            SetShape(scoresMasked, new[] { H, L, L });

            var weights = New("attn_weights");
            AddNode("Softmax", new[] { scoresMasked }, new[] { weights },
                new[] { new AttributeProto { Name = "axis", Type = AttributeProto.Types.AttributeType.Int, I = -1 } });
            SetShape(weights, new[] { H, L, L });

            var attnOut = New("attn_matmul");
            AddNode("MatMul", new[] { weights, Vh }, new[] { attnOut });
            SetShape(attnOut, new[] { H, L, Hd });

            var attnT = TransposeCall(attnOut, new[] { 1, 0, 2 });
            var attnR = ReshapeCall(attnT, new[] { L, D });
            var proj = Linear(attnR, module.OProj);
            var resid1 = Add(proj, inputName);

            var xN2 = NormUnparam(resid1);
            var mlp1 = Linear(xN2, module.MlpFc1);
            var geluOut = Gelu(mlp1);
            var mlp2 = Linear(geluOut, module.MlpFc2);
            var resid2 = Add(mlp2, resid1);

            RecordOnnxExportDiagnostic(__m, "TransformerBlock", "TransformerBlock", 0);
            return resid2;
        }

        public string MultiheadAttention(string inputName, MultiheadAttention<T> module)
        {
            var __m = (DiagnosticsTracker.IsEnabled || OnnxExportDiagnostics.Enabled) ? DiagnosticsTracker.StartMeasurement() : default;
            {
                RecordOnnxExportDiagnostic(__m, "MultiheadAttention", "Unsupported", 0, true, "MultiheadAttention export not yet enabled");
                throw new NotSupportedException("MultiheadAttention export not yet enabled");
            }
        }

        public string Sequential(string inputName, Sequential<T> module)
        {
            var __m = (DiagnosticsTracker.IsEnabled || OnnxExportDiagnostics.Enabled) ? DiagnosticsTracker.StartMeasurement() : default;
            var outputName = ((IOnnxExportContext<T>)this).Sequential(inputName, module);
            RecordOnnxExportDiagnostic(__m, "Sequential", "Sequential", 0);
            return outputName;
        }

        public string Relu(string inputName)
        {
            var __m = (DiagnosticsTracker.IsEnabled || OnnxExportDiagnostics.Enabled) ? DiagnosticsTracker.StartMeasurement() : default;
            var outName = New("relu_out");
            AddNode("Relu", new[] { inputName }, new[] { outName });
            SetShape(outName, GetShape(inputName));
            RecordOnnxExportDiagnostic(__m, "Relu", "Relu", 0);
            return outName;
        }

        public string Gelu(string inputName)
        {
            var __m = (DiagnosticsTracker.IsEnabled || OnnxExportDiagnostics.Enabled) ? DiagnosticsTracker.StartMeasurement() : default;
            var outName = New("gelu_out");
            AddNode("Gelu", new[] { inputName }, new[] { outName });
            SetShape(outName, GetShape(inputName));
            RecordOnnxExportDiagnostic(__m, "Gelu", "Gelu", 0);
            return outName;
        }

        public string Sigmoid(string inputName)
        {
            var __m = (DiagnosticsTracker.IsEnabled || OnnxExportDiagnostics.Enabled) ? DiagnosticsTracker.StartMeasurement() : default;
            var outName = New("sigmoid_out");
            AddNode("Sigmoid", new[] { inputName }, new[] { outName });
            SetShape(outName, GetShape(inputName));
            RecordOnnxExportDiagnostic(__m, "Sigmoid", "Sigmoid", 0);
            return outName;
        }

        public string Tanh(string inputName)
        {
            var __m = (DiagnosticsTracker.IsEnabled || OnnxExportDiagnostics.Enabled) ? DiagnosticsTracker.StartMeasurement() : default;
            var outName = New("tanh_out");
            AddNode("Tanh", new[] { inputName }, new[] { outName });
            SetShape(outName, GetShape(inputName));
            RecordOnnxExportDiagnostic(__m, "Tanh", "Tanh", 0);
            return outName;
        }

        public string TransposeAxes(string inputName, int[] perm)
        {
            var __m = (DiagnosticsTracker.IsEnabled || OnnxExportDiagnostics.Enabled) ? DiagnosticsTracker.StartMeasurement() : default;
            var outName = New("transpose_out");
            var attr = new AttributeProto { Name = "perm", Type = AttributeProto.Types.AttributeType.Ints };
            attr.Ints.AddRange(perm.Select(v => (long)v));
            AddNode("Transpose", new[] { inputName }, new[] { outName }, new[] { attr });
            var inShape = GetShape(inputName);
            int[] outShape;
            if (inShape.Length > 0 && inShape.Length == perm.Length)
                outShape = perm.Select(i => inShape[i]).ToArray();
            else
                outShape = inShape.ToArray();
            SetShape(outName, outShape);
            RecordOnnxExportDiagnostic(__m, "TransposeAxes", "Transpose", 0);
            return outName;
        }

        public string Reshape(string inputName, int[] newShape)
        {
            var __m = (DiagnosticsTracker.IsEnabled || OnnxExportDiagnostics.Enabled) ? DiagnosticsTracker.StartMeasurement() : default;
            var initName = New("shape_const");
            var tp = new TensorProto { Name = initName, DataType = (int)TensorProto.Types.DataType.Int64 };
            tp.Dims.Add(newShape.Length);
            var bytes = new byte[newShape.Length * 8];
            var longs = newShape.Select(v => (long)v).ToArray();
            Buffer.BlockCopy(longs, 0, bytes, 0, bytes.Length);
            tp.RawData = ByteString.CopyFrom(bytes);
            graph.Initializer.Add(tp);
            SetShape(initName, new[] { newShape.Length });
            var outName = New("reshape_out");
            AddNode("Reshape", new[] { inputName, initName }, new[] { outName });
            SetShape(outName, newShape);
            RecordOnnxExportDiagnostic(__m, "Reshape", "Reshape", 0);
            return outName;
        }

        public string Concat(string[] inputNames, int axis)
        {
            var __m = (DiagnosticsTracker.IsEnabled || OnnxExportDiagnostics.Enabled) ? DiagnosticsTracker.StartMeasurement() : default;
            var outName = New("concat_out");
            var attr = new AttributeProto { Name = "axis", Type = AttributeProto.Types.AttributeType.Int, I = axis };
            AddNode("Concat", inputNames, new[] { outName }, new[] { attr });
            var inShape = GetShape(inputNames[0]);
            var outShape = inShape.ToArray();
            if (inShape.Length > 0 && axis >= 0 && axis < inShape.Length)
            {
                int total = 0;
                bool known = true;
                foreach (var n in inputNames)
                {
                    var s = GetShape(n);
                    if (s.Length <= axis || s[axis] <= 0) { known = false; break; }
                    total += s[axis];
                }
                if (known) outShape[axis] = total;
            }
            SetShape(outName, outShape);
            RecordOnnxExportDiagnostic(__m, "Concat", "Concat", 0);
            return outName;
        }

        public string ReduceMean(string inputName, int[] axes, bool keepDims = false)
        {
            var __m = (DiagnosticsTracker.IsEnabled || OnnxExportDiagnostics.Enabled) ? DiagnosticsTracker.StartMeasurement() : default;
            var axesName = New("reduce_axes");
            var tp = new TensorProto { Name = axesName, DataType = (int)TensorProto.Types.DataType.Int64 };
            tp.Dims.Add(axes.Length);
            var axesLong = axes.Select(v => (long)v).ToArray();
            var bytes = new byte[axes.Length * 8];
            Buffer.BlockCopy(axesLong, 0, bytes, 0, bytes.Length);
            tp.RawData = ByteString.CopyFrom(bytes);
            graph.Initializer.Add(tp);
            SetShape(axesName, new[] { axes.Length });
            var outName = New("reduce_mean_out");
            var attr = new AttributeProto { Name = "keepdims", Type = AttributeProto.Types.AttributeType.Int, I = keepDims ? 1 : 0 };
            AddNode("ReduceMean", new[] { inputName, axesName }, new[] { outName }, new[] { attr });
            var inShape = GetShape(inputName);
            var axisSet = new HashSet<int>(axes.Select(a => a < 0 ? inShape.Length + a : a));
            var outShape = new List<int>();
            for (int i = 0; i < inShape.Length; i++)
            {
                if (axisSet.Contains(i))
                {
                    if (keepDims) outShape.Add(1);
                }
                else
                {
                    outShape.Add(inShape[i]);
                }
            }
            if (outShape.Count == 0) outShape.Add(1);
            SetShape(outName, outShape.ToArray());
            RecordOnnxExportDiagnostic(__m, "ReduceMean", "ReduceMean", 0);
            return outName;
        }

        public string Add(string left, string right)
        {
            var __m = (DiagnosticsTracker.IsEnabled || OnnxExportDiagnostics.Enabled) ? DiagnosticsTracker.StartMeasurement() : default;
            var outName = New("add_out");
            AddNode("Add", new[] { left, right }, new[] { outName });
            SetShape(outName, GetShape(left));
            RecordOnnxExportDiagnostic(__m, "Add", "Add", 0);
            return outName;
        }

        // IOnnxExportContext implementation for Sequential (chain)
        string IOnnxExportContext<T>.Sequential(string inputName, Sequential<T> module)
        {
            var current = inputName;
            foreach (var m in module.Modules)
                current = m.ExportToOnnx(this, current);
            return current;
        }

        string IOnnxExportContext<T>.Conv1d(string inputName, Conv1d<T> module) => Conv1d(inputName, module);
        string IOnnxExportContext<T>.Conv2d(string inputName, Conv2d<T> module) => Conv2d(inputName, module);
        string IOnnxExportContext<T>.Embedding(string inputName, Embedding<T> module) => Embedding(inputName, module);
        string IOnnxExportContext<T>.Embedding<TWeight>(string inputName, Embedding<T, TWeight> module) => Embedding(inputName, module);
        string IOnnxExportContext<T>.BatchNorm1d(string inputName, BatchNorm1d<T> module) => BatchNorm1d(inputName, module);
        string IOnnxExportContext<T>.BatchNorm2d(string inputName, BatchNorm2d<T> module, int? n, int? h, int? w) => BatchNorm2d(inputName, module, n, h, w);
        string IOnnxExportContext<T>.LayerNorm(string inputName, LayerNorm<T> module) => LayerNorm(inputName, module);
        string IOnnxExportContext<T>.RMSNorm(string inputName, RMSNorm<T> module) => RMSNorm(inputName, module);
        string IOnnxExportContext<T>.TransformerBlock(string inputName, TransformerBlock<T> module, int sequenceLength) => TransformerBlock(inputName, module, sequenceLength);
        string IOnnxExportContext<T>.MultiheadAttention(string inputName, MultiheadAttention<T> module) => MultiheadAttention(inputName, module);
        string IOnnxExportContext<T>.Linear(string inputName, Linear<T> module) => Linear(inputName, module);
        string IOnnxExportContext<T>.Linear<TWeight>(string inputName, Linear<T, TWeight> module) => Linear(inputName, module);
        string IOnnxExportContext<T>.Relu(string inputName) => Relu(inputName);
        string IOnnxExportContext<T>.Gelu(string inputName) => Gelu(inputName);
        string IOnnxExportContext<T>.Sigmoid(string inputName) => Sigmoid(inputName);
        string IOnnxExportContext<T>.Tanh(string inputName) => Tanh(inputName);
        string IOnnxExportContext<T>.Add(string left, string right) => Add(left, right);
        string IOnnxExportContext<T>.TransposeAxes(string inputName, int[] perm) => TransposeAxes(inputName, perm);
        string IOnnxExportContext<T>.Reshape(string inputName, int[] newShape) => Reshape(inputName, newShape);
        string IOnnxExportContext<T>.Concat(string[] inputNames, int axis) => Concat(inputNames, axis);
        string IOnnxExportContext<T>.ReduceMean(string inputName, int[] axes, bool keepDims) => ReduceMean(inputName, axes, keepDims);

        static T[] ToArray(ReverseGradTensor<T>? tensor)
        {
            if (tensor == null) return Array.Empty<T>();
            var data = new T[tensor.Length];
            tensor.Data.CopyTo(data, T.Zero);
            return data;
        }
        void RecordOnnxExportDiagnostic(OperationMeasurement measurement, string methodName, string emitter, int inputLength, bool unsupported = false, string? reason = null)
        {
            var notes = unsupported
                ? $"OnnxExport={methodName};Emitter=Unsupported;Reason={reason}"
                : $"OnnxExport={methodName};Emitter={emitter}";
            var aliases = new[]
            {
                "OnnxExport" + methodName,
                "Onnx" + methodName,
            };
            if (unsupported)
            {
                aliases = new[]
                {
                    "OnnxExport" + methodName,
                    "Onnx" + methodName,
                    "OnnxExportUnsupported",
                    "OnnxUnsupported",
                };
            }
            foreach (var opType in aliases)
                measurement.Record(opType, KernelType.Scalar, inputLength, typeof(T), false, notes);
        }


    }

}
