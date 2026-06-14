using Nivara.AutoDiff;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;

namespace Nivara.SampleApp;

/// <summary>
/// Demonstrates advanced automatic differentiation operations including
/// reduction operations (Sum, Mean), activation functions (ReLU, Sigmoid, Tanh),
/// and gradient utilities for training workflows.
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

        RunGradientUtilitiesDemo();
        Console.WriteLine();

        RunForwardModeDemo();
        Console.WriteLine();

        Console.WriteLine("=== Example Complete ===");
    }

    private static void RunReductionOperationsDemo()
    {
        Console.WriteLine("1. Reduction Operations (Sum and Mean)");
        Console.WriteLine("--------------------------------------");

        var data = NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        var x = new ReverseGradTensor<float>(data, requiresGrad: true);

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
        var a = new ReverseGradTensor<float>(activationData, requiresGrad: true);

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

        var input = new ReverseGradTensor<float>(inputData, requiresGrad: false);
        var weight = new ReverseGradTensor<float>(weightData, requiresGrad: true);
        var bias = new ReverseGradTensor<float>(biasData, requiresGrad: true);

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

    private static void RunGradientUtilitiesDemo()
    {
        Console.WriteLine("4. Gradient Utilities for Training Workflows");
        Console.WriteLine("--------------------------------------------");

        // Create parameters for a simple model
        var weights = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 0.5f, 0.5f, 0.5f }),
            requiresGrad: true);
        var bias = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 0.1f }),
            requiresGrad: true);

        Console.WriteLine("Simulating a training loop with gradient utilities:");
        Console.WriteLine();

        // Iteration 1
        Console.WriteLine("Iteration 1:");
        var input1 = GradientUtils.Constant(new float[] { 1.0f, 2.0f, 3.0f });
        var output1 = GradOperations.Sum(GradOperations.Multiply(input1, weights));
        output1 = GradOperations.Add(output1, bias);

        output1.Backward();
        Console.WriteLine($"  Gradient norm: {GradientUtils.GetGradientNorm(weights):F4}");
        Console.WriteLine($"  Has gradient: {GradientUtils.HasGradient(weights)}");

        // Clear gradients for next iteration
        GradientUtils.ZeroGrad(new[] { weights, bias });
        Console.WriteLine($"  After ZeroGrad: {GradientUtils.HasGradient(weights)}");
        Console.WriteLine();

        // Iteration 2 - demonstrate gradient clipping
        Console.WriteLine("Iteration 2 (with gradient clipping):");
        var input2 = GradientUtils.Constant(new float[] { 10.0f, 20.0f, 30.0f });
        var output2 = GradOperations.Sum(GradOperations.Multiply(input2, weights));
        output2 = GradOperations.Add(output2, bias);

        output2.Backward();
        Console.WriteLine($"  Gradient norm before clipping: {GradientUtils.GetGradientNorm(weights):F4}");

        // Clip gradients to prevent exploding gradients
        GradientUtils.ClipGradNorm(weights, 5.0);
        Console.WriteLine($"  Gradient norm after clipping: {GradientUtils.GetGradientNorm(weights):F4}");
        Console.WriteLine();

        // Demonstrate constant tensor creation
        Console.WriteLine("Creating constant tensors (no gradient tracking):");
        var zeros = GradientUtils.Zeros<float>(3);
        var ones = GradientUtils.Ones<float>(3);
        var full = GradientUtils.Full(3, 7.5f);

        Console.WriteLine($"  Zeros: [{zeros[0]}, {zeros[1]}, {zeros[2]}] (requiresGrad: {zeros.RequiresGrad})");
        Console.WriteLine($"  Ones: [{ones[0]}, {ones[1]}, {ones[2]}] (requiresGrad: {ones.RequiresGrad})");
        Console.WriteLine($"  Full(7.5): [{full[0]}, {full[1]}, {full[2]}] (requiresGrad: {full.RequiresGrad})");
        Console.WriteLine();

        // Demonstrate graph inspection
        Console.WriteLine("Computation graph inspection:");
        var a = new ReverseGradTensor<float>(NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f }), requiresGrad: true);
        var b = new ReverseGradTensor<float>(NivaraColumn<float>.Create(new float[] { 3.0f, 4.0f }), requiresGrad: true);
        var result = GradOperations.Add(a, b);
        result = GradOperations.Multiply(result, a);
        result = GradOperations.Sum(result);

        var graphInfo = GradientUtils.GetGraphInfo(result);
        Console.WriteLine($"  Total nodes: {graphInfo["TotalNodes"]}");
        Console.WriteLine($"  Is leaf: {graphInfo["IsLeaf"]}");
        Console.WriteLine($"  Can backward: {GradientUtils.CanBackward(result)}");
        Console.WriteLine();

        Console.WriteLine("Graph summary:");
        Console.WriteLine(GradientUtils.PrintGraphSummary(result));
    }

    private static void RunForwardModeDemo()
    {
        Console.WriteLine("5. Forward-Mode AutoDiff (Tangent Propagation)");
        Console.WriteLine("----------------------------------------------");

        // ForwardGradTensor computes a directional derivative (tangent)
        // alongside the primal value — no computation graph, no backward pass.
        // Seed a tangent on the input, then every operation propagates it.

        var x = ForwardGradTensor<float>.FromArray(
            new float[] { 1.0f, 2.0f, 3.0f },
            new float[] { 1.0f, 1.0f, 1.0f });

        Console.WriteLine($"Primal: [1, 2, 3]");
        Console.WriteLine($"Tangent (seed): [1, 1, 1]");
        Console.WriteLine("(Tangent = directional derivative along all-ones direction)");
        Console.WriteLine();

        // Forward propagation through operations
        var squared = ForwardGradOperations.Multiply(x, x);
        Console.WriteLine($"x * x: primal=[{squared[0]}, {squared[1]}, {squared[2]}], " +
                          $"tangent=[{squared.Tangent![0]}, {squared.Tangent[1]}, {squared.Tangent[2]}]");
        Console.WriteLine("  d(x^2)/dx = 2x => 2x * tangent = [2, 4, 6] ✓");
        Console.WriteLine();

        var summed = ForwardGradOperations.Sum(squared);
        Console.WriteLine($"sum(x^2): primal={summed[0]}, tangent={summed.Tangent![0]}");
        Console.WriteLine("  JVP = sum(2x * tangent) = 2 + 4 + 6 = 12 ✓");
        Console.WriteLine();

        // Compare with backward mode on the same computation
        var bx = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(new float[] { 1.0f, 2.0f, 3.0f }),
            requiresGrad: true);
        var bSquared = GradOperations.Multiply(bx, bx);
        var bSummed = GradOperations.Sum(bSquared);
        bSummed.Backward();
        Console.WriteLine("Backward mode (same computation, sum(x^2)):");
        Console.WriteLine($"  grad = [{bx.Grad![0]}, {bx.Grad[1]}, {bx.Grad[2]}]");
        Console.WriteLine("  (Forward tangent at Sum = backward gradient magnitude = 12)");
        Console.WriteLine();

        // Works with activations too
        var act = ForwardGradTensor<float>.FromArray(
            new float[] { -2.0f, 0.0f, 2.0f },
            new float[] { 1.0f, 1.0f, 1.0f });
        var relu = ForwardGradOperations.Relu(act);
        Console.WriteLine("Forward-mode through ReLU:");
        Console.WriteLine($"  primal=[{relu[0]}, {relu[1]}, {relu[2]}]");
        Console.WriteLine($"  tangent=[{relu.Tangent![0]}, {relu.Tangent[1]}, {relu.Tangent[2]}]");
        Console.WriteLine("  (ReLU' = 0 for x <= 0, 1 for x > 0)");
    }
}
