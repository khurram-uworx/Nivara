using Nivara;
using Nivara.Extensions.AutoDiff;
using Nivara.Extensions.AutoDiff.Operations;

namespace Nivara.SampleApp;

/// <summary>
/// Demonstrates advanced automatic differentiation operations including
/// reduction operations (Sum, Mean) and activation functions (ReLU, Sigmoid, Tanh).
/// </summary>
public static class AutoDiffExample
{
    public static void Run()
    {
        Console.WriteLine("=== Advanced AutoDiff Operations Example ===");
        Console.WriteLine();

        RunReductionOperationsDemo();
        Console.WriteLine();

        RunActivationFunctionsDemo();
        Console.WriteLine();

        RunNeuralNetworkDemo();
        Console.WriteLine();

        Console.WriteLine("=== Example Complete ===");
    }

    private static void RunReductionOperationsDemo()
    {
        Console.WriteLine("1. Reduction Operations (Sum and Mean)");
        Console.WriteLine("--------------------------------------");

        var data = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        var x = new GradTensor<float>(data, requiresGrad: true);

        // Sum operation
        var sum = GradOperations.Sum(x);
        Console.WriteLine($"Input: [1, 2, 3, 4]");
        Console.WriteLine($"Sum: {sum[0]}");

        sum.Backward();
        Console.WriteLine($"Gradient (d_sum/d_x): [{x.Grad![0]}, {x.Grad[1]}, {x.Grad[2]}, {x.Grad[3]}]");
        Console.WriteLine("(All gradients are 1 because d/dx(sum) = 1 for each element)");
        Console.WriteLine();

        // Reset gradients
        x.ZeroGrad();

        // Mean operation
        var mean = GradOperations.Mean(x);
        Console.WriteLine($"Mean: {mean[0]}");

        mean.Backward();
        Console.WriteLine($"Gradient (d_mean/d_x): [{x.Grad![0]}, {x.Grad[1]}, {x.Grad[2]}, {x.Grad[3]}]");
        Console.WriteLine("(All gradients are 0.25 because d/dx(mean) = 1/n = 1/4)");
    }

    private static void RunActivationFunctionsDemo()
    {
        Console.WriteLine("2. Activation Functions (ReLU, Sigmoid, Tanh)");
        Console.WriteLine("---------------------------------------------");

        var activationData = NivaraColumn<float>.Create(new float[] { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f });
        var a = new GradTensor<float>(activationData, requiresGrad: true);

        Console.WriteLine($"Input: [-2, -1, 0, 1, 2]");
        Console.WriteLine();

        // ReLU
        var relu = GradOperations.Relu(a);
        Console.WriteLine($"ReLU: [{relu[0]}, {relu[1]}, {relu[2]}, {relu[3]}, {relu[4]}]");
        Console.WriteLine("(ReLU(x) = max(0, x))");
        Console.WriteLine();

        // Sigmoid
        var sigmoid = GradOperations.Sigmoid(a);
        Console.WriteLine($"Sigmoid: [{sigmoid[0]:F4}, {sigmoid[1]:F4}, {sigmoid[2]:F4}, {sigmoid[3]:F4}, {sigmoid[4]:F4}]");
        Console.WriteLine("(Sigmoid(x) = 1 / (1 + exp(-x)))");
        Console.WriteLine();

        // Tanh
        var tanh = GradOperations.Tanh(a);
        Console.WriteLine($"Tanh: [{tanh[0]:F4}, {tanh[1]:F4}, {tanh[2]:F4}, {tanh[3]:F4}, {tanh[4]:F4}]");
        Console.WriteLine("(Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)))");
    }

    private static void RunNeuralNetworkDemo()
    {
        Console.WriteLine("3. Neural Network-like Computation");
        Console.WriteLine("----------------------------------");

        var inputData = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f });
        var weightData = NivaraColumn<float>.Create(new float[] { 0.5f, 0.5f, 0.5f });
        var biasData = NivaraColumn<float>.Create(new float[] { -1.0f, 0.0f, 1.0f });

        var input = new GradTensor<float>(inputData, requiresGrad: false);
        var weight = new GradTensor<float>(weightData, requiresGrad: true);
        var bias = new GradTensor<float>(biasData, requiresGrad: true);

        Console.WriteLine("Computing: y = mean(relu(x * w + b))");
        Console.WriteLine($"Input (x): [1, 2, 3]");
        Console.WriteLine($"Weight (w): [0.5, 0.5, 0.5]");
        Console.WriteLine($"Bias (b): [-1, 0, 1]");
        Console.WriteLine();

        // Forward pass
        var mul = GradOperations.Multiply(input, weight);
        Console.WriteLine($"Step 1: x * w = [{mul[0]}, {mul[1]}, {mul[2]}]");

        var add = GradOperations.Add(mul, bias);
        Console.WriteLine($"Step 2: x * w + b = [{add[0]}, {add[1]}, {add[2]}]");

        var reluResult = GradOperations.Relu(add);
        Console.WriteLine($"Step 3: relu(x * w + b) = [{reluResult[0]}, {reluResult[1]}, {reluResult[2]}]");

        var output = GradOperations.Mean(reluResult);
        Console.WriteLine($"Step 4: mean(relu(x * w + b)) = {output[0]:F4}");
        Console.WriteLine();

        // Backward pass
        output.Backward();

        Console.WriteLine($"Gradients after backward pass:");
        Console.WriteLine($"  d_output/d_weight: [{weight.Grad![0]:F4}, {weight.Grad[1]:F4}, {weight.Grad[2]:F4}]");
        Console.WriteLine($"  d_output/d_bias: [{bias.Grad![0]:F4}, {bias.Grad[1]:F4}, {bias.Grad[2]:F4}]");
        Console.WriteLine();
        Console.WriteLine("These gradients show how to adjust weights and biases to increase the output.");
        Console.WriteLine("In a real neural network, you would use these gradients to update parameters:");
        Console.WriteLine("  new_weight = old_weight - learning_rate * gradient");
    }
}
