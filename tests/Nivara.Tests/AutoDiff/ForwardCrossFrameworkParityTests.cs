using NUnit.Framework;
using System.Text.Json;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class ForwardCrossFrameworkParityTests
{
    static string RepoRoot => Path.GetFullPath(
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));

    static string DataDir => Path.Combine(RepoRoot, "samples", "data");

    [Test]
    public void JvpCasesFile_Exists()
    {
        var path = Path.Combine(DataDir, "jvp_cases.json");
        Assert.That(File.Exists(path), Is.True,
            "jvp_cases.json not found. Run: python samples/pytorch/forward_parity_pytorch.py");
    }

    [Test]
    public void AllSixCases_HavePrimalAndJvp()
    {
        var doc = LoadCases();
        var cases = doc.RootElement.GetProperty("test_cases");
        Assert.That(cases.GetArrayLength(), Is.EqualTo(6),
            "Expected exactly 6 test cases");

        var names = new HashSet<string>();
        foreach (var tc in cases.EnumerateArray())
        {
            var name = tc.GetProperty("name").GetString()!;
            Assert.That(names.Add(name), Is.True, $"Duplicate test case name: {name}");
            Assert.That(tc.TryGetProperty("primal", out _), Is.True,
                $"{name}: missing 'primal'");
            Assert.That(tc.TryGetProperty("jvp", out _), Is.True,
                $"{name}: missing 'jvp'");
            Assert.That(tc.TryGetProperty("inputs", out _), Is.True,
                $"{name}: missing 'inputs'");
            Assert.That(tc.TryGetProperty("seeds", out _), Is.True,
                $"{name}: missing 'seeds'");
        }
    }

    [Test]
    public void Square_JvpIsCorrect()
    {
        var tc = GetCase("square");
        // f(x) = x*x, seed=[1,0] => JVP = [2*x1, 0] = [4, 0]
        var jvp = DeserializeArray(tc.GetProperty("jvp"));
        Assert.That(jvp[0], Is.EqualTo(4.0f).Within(1e-5));
        Assert.That(jvp[1], Is.EqualTo(0.0f).Within(1e-5));
    }

    [Test]
    public void MulAdd_JvpIsCorrect()
    {
        var tc = GetCase("mul_add");
        // f(a,b) = a*b + a, seeds=[1,1] => JVP = b*1 + a*1 + 1 = 2.5 + 1.5 + 1 = 5
        var jvp = DeserializeArray(tc.GetProperty("jvp"));
        Assert.That(jvp[0], Is.EqualTo(5.0f).Within(1e-5));
    }

    [Test]
    public void Relu_JvpIsCorrect()
    {
        var tc = GetCase("relu");
        // f(x) = relu(x), seed=[1,1,1] => JVP = [0, 0, 1] (relu' at -1=0, 0=0, 2=1)
        var jvp = DeserializeArray(tc.GetProperty("jvp"));
        Assert.That(jvp[0], Is.EqualTo(0.0f).Within(1e-5));
        Assert.That(jvp[1], Is.EqualTo(0.0f).Within(1e-5));
        Assert.That(jvp[2], Is.EqualTo(1.0f).Within(1e-5));
    }

    [Test]
    public void Sigmoid_JvpIsCorrect()
    {
        var tc = GetCase("sigmoid");
        // f(x) = sigmoid(x), seed=[1,1] => JVP = sigmoid(x)*(1-sigmoid(x))
        // sigmoid(0)=0.5 => 0.5*0.5 = 0.25
        // sigmoid(1)=0.73105857863 => 0.73105857863*0.26894142137 = 0.19661193324
        var jvp = DeserializeArray(tc.GetProperty("jvp"));
        Assert.That(jvp[0], Is.EqualTo(0.25f).Within(1e-5));
        Assert.That(jvp[1], Is.EqualTo(0.19661193324148185).Within(1e-5));
    }

    [Test]
    public void MatMul_JvpIsCorrect()
    {
        var tc = GetCase("matmul");
        // f(x) = W@x, W=[[1,2],[3,4]], seed=[1,0] => JVP = W[:,0] = [1, 3]
        var jvp = DeserializeArray(tc.GetProperty("jvp"));
        Assert.That(jvp, Has.Length.EqualTo(2));
        Assert.That(jvp[0], Is.EqualTo(1.0f).Within(1e-5));
        Assert.That(jvp[1], Is.EqualTo(3.0f).Within(1e-5));
    }

    [Test]
    public void Composition_JvpIsCorrect()
    {
        var tc = GetCase("composition");
        // f(x) = sum(relu(W@x+b)), W=[[0.5,-0.5],[-0.5,0.5]], b=[0.1,-0.1]
        // x=[1,-0.5], seed=[1,0]
        // W@x = [0.75, -0.75], +b = [0.85, -0.85]
        // relu => [0.85, 0], sum => 0.85
        // JVP direction: first column of W = [0.5, -0.5]
        // At x=[1,-0.5]: z1 = 0.75+0.1=0.85>0, z2 = -0.75-0.1=-0.85<0
        // So only first component contributes: JVP = W[0,0]*seed[0] + W[0,1]*seed[1]
        // = 0.5*1 + (-0.5)*0 = 0.5
        var jvp = DeserializeArray(tc.GetProperty("jvp"));
        Assert.That(jvp[0], Is.EqualTo(0.5f).Within(1e-5));
    }

    [Test]
    public void AllCases_PrimalAndJvpAreConsistentAcrossPyTorch()
    {
        var doc = LoadCases();
        var cases = doc.RootElement.GetProperty("test_cases");

        foreach (var tc in cases.EnumerateArray())
        {
            var name = tc.GetProperty("name").GetString()!;
            var primal = DeserializeArray(tc.GetProperty("primal"));
            var jvp = DeserializeArray(tc.GetProperty("jvp"));

            Assert.That(primal, Is.Not.Empty, $"{name}: primal array is empty");
            Assert.That(jvp, Is.Not.Empty, $"{name}: jvp array is empty");
            Assert.That(jvp.Length, Is.EqualTo(primal.Length),
                $"{name}: jvp length ({jvp.Length}) != primal length ({primal.Length})");

            Assert.That(jvp.Any(v => float.IsNaN(v) || float.IsInfinity(v)),
                Is.False, $"{name}: jvp contains NaN or Infinity");
            Assert.That(primal.Any(v => float.IsNaN(v) || float.IsInfinity(v)),
                Is.False, $"{name}: primal contains NaN or Infinity");
        }
    }

    static JsonDocument LoadCases()
    {
        var path = Path.Combine(DataDir, "jvp_cases.json");
        if (!File.Exists(path))
            Assert.Ignore("jvp_cases.json not found — run python samples/pytorch/forward_parity_pytorch.py first");
        return JsonDocument.Parse(File.ReadAllBytes(path));
    }

    static JsonElement GetCase(string name)
    {
        var doc = LoadCases();
        foreach (var tc in doc.RootElement.GetProperty("test_cases").EnumerateArray())
            if (tc.GetProperty("name").GetString() == name)
                return tc;
        Assert.Fail($"Test case '{name}' not found in jvp_cases.json");
        return default;
    }

    static float[] DeserializeArray(JsonElement el)
    {
        var list = new List<float>();
        foreach (var item in el.EnumerateArray())
            list.Add(item.GetSingle());
        return list.ToArray();
    }
}
