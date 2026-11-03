using System.Numerics;
using Google.Protobuf;
using Onnx;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;

namespace Nivara.Onnx;

public sealed class OnnxExportOptions
{
    public int[] InputShape { get; set; } = Array.Empty<int>();
    public int OpsetVersion { get; set; } = 20;
    public string ProducerName { get; set; } = "Nivara";
    public string ProducerVersion { get; set; } = "2.0.0";
    public string GraphName { get; set; } = "main_graph";
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
            return outName;
        }

        public string Conv1d(string inputName, Conv1d<T> module)
        {
            var wName = New("conv1d_weight");
            var wShape = module.Weight!.Tensor.Shape;
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
            return outName;
        }

        public string Conv2d(string inputName, Conv2d<T> module)
        {
            var wName = New("conv2d_weight");
            var wShape = module.Weight!.Tensor.Shape;
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
            return outName;
        }

        public string Embedding(string inputName, Embedding<T> module)
        {
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
            return outName;
        }

        public string BatchNorm1d(string inputName, BatchNorm1d<T> module)
        {
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
            return outName;
        }

        public string BatchNorm2d(string inputName, BatchNorm2d<T> module, int? n = null, int? h = null, int? w = null)
        {
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
            return outName;
        }

        public string LayerNorm(string inputName, LayerNorm<T> module)
        {
            string scaleName;
            string biasName;
            if (module.Affine)
            {
                scaleName = New("ln_scale");
                biasName = New("ln_bias");
                var scaleData = new T[module.Weight!.Tensor.Length]; module.Weight!.Tensor.Data.CopyTo(scaleData, T.Zero);
                var biasData = new T[module.Bias!.Tensor.Length]; module.Bias!.Tensor.Data.CopyTo(biasData, T.Zero);
                AddInitializer(scaleName, scaleData, module.Weight!.Tensor.Shape);
                AddInitializer(biasName, biasData, module.Bias!.Tensor.Shape);
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
            return outName;
        }

        public string RMSNorm(string inputName, RMSNorm<T> module)
        {
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
            return outName;
        }

        public string TransformerBlock(string inputName, TransformerBlock<T> module, int sequenceLength)
        {
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
                return outName;
            }

            string TransposeCall(string x, int[] perm)
            {
                var attr = new AttributeProto { Name = "perm", Type = AttributeProto.Types.AttributeType.Ints };
                attr.Ints.AddRange(perm.Select(v => (long)v));
                var outName = New("transpose");
                AddNode("Transpose", new[] { x }, new[] { outName }, new[] { attr });
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

            return resid2;
        }

        public string MultiheadAttention(string inputName, MultiheadAttention<T> module)
        {
            throw new NotSupportedException("MultiheadAttention export not yet enabled");
        }

        public string Sequential(string inputName, Sequential<T> module) => ((IOnnxExportContext<T>)this).Sequential(inputName, module);

        public string Relu(string inputName)
        {
            var outName = New("relu_out");
            AddNode("Relu", new[] { inputName }, new[] { outName });
            SetShape(outName, GetShape(inputName));
            return outName;
        }

        public string Gelu(string inputName)
        {
            var outName = New("gelu_out");
            AddNode("Gelu", new[] { inputName }, new[] { outName });
            SetShape(outName, GetShape(inputName));
            return outName;
        }

        public string Sigmoid(string inputName)
        {
            var outName = New("sigmoid_out");
            AddNode("Sigmoid", new[] { inputName }, new[] { outName });
            SetShape(outName, GetShape(inputName));
            return outName;
        }

        public string Tanh(string inputName)
        {
            var outName = New("tanh_out");
            AddNode("Tanh", new[] { inputName }, new[] { outName });
            SetShape(outName, GetShape(inputName));
            return outName;
        }

        public string Add(string left, string right)
        {
            var outName = New("add_out");
            AddNode("Add", new[] { left, right }, new[] { outName });
            SetShape(outName, GetShape(left));
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
        string IOnnxExportContext<T>.BatchNorm1d(string inputName, BatchNorm1d<T> module) => BatchNorm1d(inputName, module);
        string IOnnxExportContext<T>.BatchNorm2d(string inputName, BatchNorm2d<T> module, int? n, int? h, int? w) => BatchNorm2d(inputName, module, n, h, w);
        string IOnnxExportContext<T>.LayerNorm(string inputName, LayerNorm<T> module) => LayerNorm(inputName, module);
        string IOnnxExportContext<T>.RMSNorm(string inputName, RMSNorm<T> module) => RMSNorm(inputName, module);
        string IOnnxExportContext<T>.TransformerBlock(string inputName, TransformerBlock<T> module, int sequenceLength) => TransformerBlock(inputName, module, sequenceLength);
        string IOnnxExportContext<T>.MultiheadAttention(string inputName, MultiheadAttention<T> module) => MultiheadAttention(inputName, module);
        string IOnnxExportContext<T>.Linear(string inputName, Linear<T> module) => Linear(inputName, module);
        string IOnnxExportContext<T>.Relu(string inputName) => Relu(inputName);
        string IOnnxExportContext<T>.Gelu(string inputName) => Gelu(inputName);
        string IOnnxExportContext<T>.Sigmoid(string inputName) => Sigmoid(inputName);
        string IOnnxExportContext<T>.Tanh(string inputName) => Tanh(inputName);
        string IOnnxExportContext<T>.Add(string left, string right) => Add(left, right);

        static T[] ToArray(ReverseGradTensor<T>? tensor)
        {
            if (tensor == null) return Array.Empty<T>();
            var data = new T[tensor.Length];
            tensor.Data.CopyTo(data, T.Zero);
            return data;
        }
    }

}