using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using System.Text.Json;

namespace Nivara.SampleApp;

public static class ForwardParityExample
{
    static string DataDir => Path.GetFullPath(
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "examples", "data"));

    public static void Run()
    {
        Console.WriteLine("=== Forward-Mode JVP Parity: Nivara vs PyTorch ===");
        Console.WriteLine();

        var jvpPath = Path.Combine(DataDir, "jvp_cases.json");
        if (!File.Exists(jvpPath))
        {
            Console.WriteLine("No jvp_cases.json found. Run the PyTorch script first:");
            Console.WriteLine("  python examples/pytorch/forward_parity_pytorch.py");
            return;
        }

        var json = File.ReadAllText(jvpPath);
        using var doc = JsonDocument.Parse(json);
        var cases = doc.RootElement.GetProperty("test_cases");

        int passed = 0, failed = 0;
        double maxError = 0;

        foreach (var tc in cases.EnumerateArray())
        {
            var name = tc.GetProperty("name").GetString()!;
            var description = tc.GetProperty("description").GetString()!;
            var expectedPrimal = DeserializeArray(tc.GetProperty("primal"));
            var expectedJvp = DeserializeArray(tc.GetProperty("jvp"));

            Console.WriteLine($"Test: {name}");
            Console.WriteLine($"  {description}");

            try
            {
                var (primal, tangent) = ComputeJvp(name, tc);

                bool primalOk = ApproxEqual(primal, expectedPrimal, 1e-5f);
                bool jvpOk = ApproxEqual(tangent, expectedJvp, 1e-5f);

                if (!primalOk)
                {
                    var err = MaxDiff(primal, expectedPrimal);
                    maxError = Math.Max(maxError, err);
                    Console.WriteLine($"  ✗ PRIMAL MISMATCH  max_err={err:G6}");
                    Console.WriteLine($"    got:     [{string.Join(", ", primal)}]");
                    Console.WriteLine($"    expect:  [{string.Join(", ", expectedPrimal)}]");
                }

                if (!jvpOk)
                {
                    var err = MaxDiff(tangent, expectedJvp);
                    maxError = Math.Max(maxError, err);
                    Console.WriteLine($"  ✗ JVP MISMATCH     max_err={err:G6}");
                    Console.WriteLine($"    got:     [{string.Join(", ", tangent)}]");
                    Console.WriteLine($"    expect:  [{string.Join(", ", expectedJvp)}]");
                }

                if (primalOk && jvpOk)
                {
                    passed++;
                    Console.WriteLine($"  ✓ primal={FormatArray(primal)}, jvp={FormatArray(tangent)}");
                }
                else
                {
                    failed++;
                }
            }
            catch (Exception ex)
            {
                failed++;
                Console.WriteLine($"  ✗ ERROR: {ex.Message}");
            }

            Console.WriteLine();
        }

        Console.WriteLine($"Results: {passed} passed, {failed} failed");
        if (failed == 0)
            Console.WriteLine("✓ All JVP values match PyTorch reference.");
        else
            Console.WriteLine($"✗ {failed} test(s) failed. Max error: {maxError:G6}");
        Console.WriteLine();
    }

    static (float[] primal, float[] tangent) ComputeJvp(string name, JsonElement tc)
    {
        var inputs = tc.GetProperty("inputs");
        var seeds = tc.GetProperty("seeds");

        return name switch
        {
            "square" => ComputeSquare(inputs, seeds),
            "mul_add" => ComputeMulAdd(inputs, seeds),
            "relu" => ComputeRelu(inputs, seeds),
            "sigmoid" => ComputeSigmoid(inputs, seeds),
            "matmul" => ComputeMatMul(inputs, seeds),
            "composition" => ComputeComposition(inputs, seeds),
            _ => throw new ArgumentException($"Unknown test case: {name}"),
        };
    }

    static (float[], float[]) ComputeSquare(JsonElement inputs, JsonElement seeds)
    {
        var x = ForwardGradTensor<float>.FromArray(
            DeserializeArray(inputs.GetProperty("x")),
            DeserializeArray(seeds.GetProperty("x")));
        var result = x * x;
        return Extract(result);
    }

    static (float[], float[]) ComputeMulAdd(JsonElement inputs, JsonElement seeds)
    {
        var a = ForwardGradTensor<float>.FromArray(
            DeserializeArray(inputs.GetProperty("a")),
            DeserializeArray(seeds.GetProperty("a")));
        var b = ForwardGradTensor<float>.FromArray(
            DeserializeArray(inputs.GetProperty("b")),
            DeserializeArray(seeds.GetProperty("b")));
        var result = a * b + a;
        return Extract(result);
    }

    static (float[], float[]) ComputeRelu(JsonElement inputs, JsonElement seeds)
    {
        var x = ForwardGradTensor<float>.FromArray(
            DeserializeArray(inputs.GetProperty("x")),
            DeserializeArray(seeds.GetProperty("x")));
        var result = ForwardGradOperations.Relu(x);
        return Extract(result);
    }

    static (float[], float[]) ComputeSigmoid(JsonElement inputs, JsonElement seeds)
    {
        var x = ForwardGradTensor<float>.FromArray(
            DeserializeArray(inputs.GetProperty("x")),
            DeserializeArray(seeds.GetProperty("x")));
        var result = ForwardGradOperations.Sigmoid(x);
        return Extract(result);
    }

    static (float[], float[]) ComputeMatMul(JsonElement inputs, JsonElement seeds)
    {
        var x = ForwardGradTensor<float>.FromArray(
            DeserializeArray(inputs.GetProperty("x")),
            DeserializeArray(seeds.GetProperty("x")));
        var W = ForwardGradTensor<float>.FromMatrix(
            new float[] { 1, 2, 3, 4 }, rows: 2, cols: 2);
        var result = ForwardGradOperations.MatMul(W, x);
        return Extract(result);
    }

    static (float[], float[]) ComputeComposition(JsonElement inputs, JsonElement seeds)
    {
        var x = ForwardGradTensor<float>.FromArray(
            DeserializeArray(inputs.GetProperty("x")),
            DeserializeArray(seeds.GetProperty("x")));
        var W = ForwardGradTensor<float>.FromMatrix(
            new float[] { 0.5f, -0.5f, -0.5f, 0.5f }, rows: 2, cols: 2);
        var b = ForwardGradTensor<float>.FromArray(
            new float[] { 0.1f, -0.1f });
        var h = ForwardGradOperations.Relu(ForwardGradOperations.Add(ForwardGradOperations.MatMul(W, x), b));
        var result = ForwardGradOperations.Sum(h);
        return Extract(result);
    }

    static (float[] primal, float[] tangent) Extract(ForwardGradTensor<float> t)
    {
        var data = new float[t.Length];
        t.Data.CopyTo(data, 0.0f);
        float[]? tan = null;
        if (t.Tangent != null)
        {
            tan = new float[t.Tangent.Length];
            t.Tangent.CopyTo(tan, 0.0f);
        }
        return (data, tan ?? []);
    }

    static float[] DeserializeArray(JsonElement el)
    {
        var list = new List<float>();
        foreach (var item in el.EnumerateArray())
            list.Add(item.GetSingle());
        return list.ToArray();
    }

    static bool ApproxEqual(float[] a, float[] b, float tol)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (Math.Abs(a[i] - b[i]) > tol) return false;
        return true;
    }

    static float MaxDiff(float[] a, float[] b)
    {
        float max = 0;
        for (int i = 0; i < Math.Min(a.Length, b.Length); i++)
            max = Math.Max(max, Math.Abs(a[i] - b[i]));
        return max;
    }

    static string FormatArray(float[] arr) =>
        $"[{string.Join(", ", arr.Select(x => $"{x:G6}"))}]";
}
