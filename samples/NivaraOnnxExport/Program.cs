using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Nivara;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using Nivara.Onnx;

const int epochs = 20;

var model = new MlpModel();
model.Train();

var features = MlpModel.Features;
const int outputs = MlpModel.Outputs;
const int rows = 8;

// Deterministic synthetic regression targets.
var inputData = new float[rows * features];
var targetData = new float[rows * outputs];
for (int r = 0; r < rows; r++)
{
    float sum = 0f;
    for (int c = 0; c < features; c++)
    {
        inputData[r * features + c] = (r + 1) * 0.1f + c * 0.05f;
        sum += inputData[r * features + c];
    }
    for (int o = 0; o < outputs; o++)
        targetData[r * outputs + o] = sum / outputs;
}

var optimizer = new SGD<float>(0.01f);
optimizer.AddParameterGroup(model.GetParameters().Values);

for (int epoch = 0; epoch < epochs; epoch++)
{
    using var scope = GradientUtils.Grad();
    optimizer.ZeroGrad();

    var input = new ReverseGradTensor<float>(
        NivaraColumn<float>.Create(inputData), false);
    input.Reshape(rows, features);

    var prediction = model.Forward(input);
    var target = new ReverseGradTensor<float>(
        NivaraColumn<float>.Create(targetData), false);
    target.Reshape(rows, outputs);

    var diff = prediction - target;
    var loss = ReverseGradOperations.Mean(diff * diff);
    loss.Backward();
    optimizer.Step();

    if (epoch == 0 || epoch == epochs - 1)
        Console.WriteLine($"epoch {epoch + 1,2}/{epochs}: mse = {loss[0]:F6}");
}

model.Eval();

// Nivara reference output.
float[] expected;
{
    using var scope = GradientUtils.Grad();
    var input = new ReverseGradTensor<float>(
        NivaraColumn<float>.Create(inputData), false);
    input.Reshape(rows, features);
    var pred = model.Forward(input);
    expected = new float[pred.Length];
    pred.Data.CopyTo(expected, 0f);
}

// Export.
var ms = new System.IO.MemoryStream();
OnnxExporter.Export(model, ms, new OnnxExportOptions
{
    InputShape = new[] { rows, features },
});
var bytes = ms.ToArray();
var onnxPath = Path.Combine(Directory.GetCurrentDirectory(), "nivara-mlp.onnx");
File.WriteAllBytes(onnxPath, bytes);
Console.WriteLine($"exported: {onnxPath} ({bytes.Length} bytes)");

// ONNX Runtime verify.
using var session = new InferenceSession(onnxPath);
var ortInput = new DenseTensor<float>(inputData, new[] { rows, features });
using var results = session.Run(new[]
{
    NamedOnnxValue.CreateFromTensor("input", ortInput)
});
var ortOutput = results.First().AsTensor<float>().ToArray();

float maxAbs = 0f;
for (int i = 0; i < expected.Length; i++)
    maxAbs = MathF.Max(maxAbs, MathF.Abs(ortOutput[i] - expected[i]));

Console.WriteLine($"ONNX Runtime output shape: [{rows}, {MlpModel.Outputs}]");
Console.WriteLine($"max abs diff vs Nivara: {maxAbs:E3}");
if (maxAbs > 1e-4f)
{
    Console.WriteLine("FAIL: parity exceeded tolerance");
    Environment.ExitCode = 1;
}
else
{
    Console.WriteLine("OK: Nivara / ONNX Runtime parity within 1e-4");
}

public sealed class MlpModel : Module<float>
{
    public const int Features = 4;
    public const int Hidden = 8;
    public const int Outputs = 3;

    readonly Linear<float> fc1;
    readonly Linear<float> fc2;

    public MlpModel() : base()
    {
        fc1 = new Linear<float>(Features, Hidden);
        fc2 = new Linear<float>(Hidden, Outputs);
        RegisterModules(fc1, fc2);
    }

    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input)
    {
        var h = fc1.Forward(input);
        h = ReverseGradOperations.Relu(h);
        return fc2.Forward(h);
    }

    public override string ExportToOnnx(IOnnxExportContext<float> context, string inputName)
    {
        var h = context.Linear(inputName, fc1);
        h = context.Relu(h);
        return context.Linear(h, fc2);
    }
}
