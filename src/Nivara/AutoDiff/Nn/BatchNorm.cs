using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class BatchNorm1d<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int numFeatures;
    readonly T eps;
    readonly T momentum;
    readonly bool affine;
    readonly bool trackRunningStats;

    readonly Parameter<T>? weight;
    readonly Parameter<T>? bias;

    ReverseGradTensor<T>? runningMean;
    ReverseGradTensor<T>? runningVar;
    ReverseGradTensor<T>? numBatchesTracked;

    public BatchNorm1d(
        int numFeatures,
        float eps = 1e-5f,
        float momentum = 0.1f,
        bool affine = true,
        bool trackRunningStats = true)
    {
        if (numFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(numFeatures));
        if (eps <= 0) throw new ArgumentOutOfRangeException(nameof(eps));
        if (momentum <= 0 || momentum >= 1) throw new ArgumentOutOfRangeException(nameof(momentum));

        this.numFeatures = numFeatures;
        this.eps = T.CreateChecked(eps);
        this.momentum = T.CreateChecked(momentum);
        this.affine = affine;
        this.trackRunningStats = trackRunningStats;

        if (affine)
        {
            var weightData = new T[numFeatures];
            var biasData = new T[numFeatures];
            for (int i = 0; i < numFeatures; i++)
            {
                weightData[i] = T.One;
                biasData[i] = T.Zero;
            }
            weight = new Parameter<T>("Weight", ReverseGradTensor<T>.FromArray(weightData, requiresGrad: true));
            bias = new Parameter<T>("Bias", ReverseGradTensor<T>.FromArray(biasData, requiresGrad: true));
            RegisterParameters(weight, bias);
        }

        if (trackRunningStats)
        {
            var runningMeanData = new T[numFeatures];
            var runningVarData = new T[numFeatures];
            for (int i = 0; i < numFeatures; i++)
            {
                runningMeanData[i] = T.Zero;
                runningVarData[i] = T.One;
            }
            runningMean = ReverseGradTensor<T>.FromArray(runningMeanData, requiresGrad: false);
            runningVar = ReverseGradTensor<T>.FromArray(runningVarData, requiresGrad: false);
            numBatchesTracked = ReverseGradTensor<T>.FromArray(new T[] { T.Zero }, requiresGrad: false);
        }
    }

    public ReverseGradTensor<T> RunningMean => runningMean
        ?? throw new InvalidOperationException("RunningMean is unavailable because this BatchNorm1d was created with trackRunningStats: false.");
    public ReverseGradTensor<T> RunningVar => runningVar
        ?? throw new InvalidOperationException("RunningVar is unavailable because this BatchNorm1d was created with trackRunningStats: false.");
    public ReverseGradTensor<T> NumBatchesTracked => numBatchesTracked
        ?? throw new InvalidOperationException("NumBatchesTracked is unavailable because this BatchNorm1d was created with trackRunningStats: false.");
    public bool TrackRunningStats => trackRunningStats;
    public Parameter<T>? Weight => weight;
    public Parameter<T>? Bias => bias;

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 2 && input.Rank != 3)
            throw new ArgumentException($"BatchNorm1d expects 2D [N, C] or 3D [B, C, L] input, got {input.Rank}D");

        int n = input.Shape[0];
        int c = input.Shape[1];
        int planeSize = input.Rank == 3 ? input.Shape[2] : 1;
        if (c != numFeatures) throw new ArgumentException($"Expected {numFeatures} channels, got {c}");

        var gamma = affine && weight != null
            ? GetParamSpan(weight.Tensor)
            : ReadOnlySpan<T>.Empty;
        var beta = affine && bias != null
            ? GetParamSpan(bias.Tensor)
            : ReadOnlySpan<T>.Empty;

        bool useRunningStats = !IsTraining;

        if (useRunningStats && runningMean != null && runningVar != null)
        {
            var rmSpan = GetParamSpan(runningMean);
            var rvSpan = GetParamSpan(runningVar);
            var evalResult = BatchNormKernel<T>.ForwardEval(
                GetInputSpan(input), n, c, planeSize,
                gamma, beta, rmSpan, rvSpan, eps, affine);

            var evalTensor = new ReverseGradTensor<T>(
                NivaraColumn<T>.Create(evalResult.Output),
                input.RequiresGrad, input.Shape);

            if (input.RequiresGrad)
            {
                var savedXHat = evalResult.XHat;
                var savedInvStd = evalResult.InvStd;
                var savedGamma = gamma.Length > 0 ? gamma.ToArray() : [];
                int savedN = n, savedC = c, savedPlaneSize = planeSize;

                var gradFn = new OpNode<T>("BatchNorm1dEval", [input], (typedGradOutput) =>
                {
                    var gradOutData = new T[typedGradOutput.Length];
                    typedGradOutput.CopyTo(gradOutData, default(T)!);

                    var gradInputData = BatchNormKernel<T>.BackwardInput(
                        gradOutData, savedXHat, savedGamma, savedInvStd,
                        savedN, savedC, savedPlaneSize, savedGamma.Length > 0);

                    ReverseGradOperations.AccumulateGradient(input, NivaraColumn<T>.Create(gradInputData));
                });

                ComputationGraph.AddNode(evalTensor, gradFn);
            }

            return evalTensor;
        }

        var inputData = GetInputSpan(input);
        var result = BatchNormKernel<T>.Forward(inputData, n, c, planeSize, gamma, beta, eps, affine);

        if (trackRunningStats)
            UpdateRunningStatsDirect(result.Mean, result.InvStd, n * planeSize);

        var resultTensor = new ReverseGradTensor<T>(
            NivaraColumn<T>.Create(result.Output),
            input.RequiresGrad, input.Shape);

        if (input.RequiresGrad)
        {
            var savedXHat = result.XHat;
            var savedInvStd = result.InvStd;
            var savedGamma = gamma.Length > 0 ? gamma.ToArray() : [];
            bool useAffine = affine;
            int savedN = n, savedC = c, savedPlaneSize = planeSize;

            var gradFn = new OpNode<T>("BatchNorm1dTrain", [input], (typedGradOutput) =>
            {
                var gradOutData = new T[typedGradOutput.Length];
                typedGradOutput.CopyTo(gradOutData, default(T)!);

                var gradInputData = BatchNormKernel<T>.BackwardInput(
                    gradOutData, savedXHat, savedGamma, savedInvStd,
                    savedN, savedC, savedPlaneSize, useAffine);

                ReverseGradOperations.AccumulateGradient(input, NivaraColumn<T>.Create(gradInputData));

                if (useAffine)
                {
                    var gradGammaData = BatchNormKernel<T>.BackwardWeight(
                        gradOutData, savedXHat, savedN, savedC, savedPlaneSize);
                    var gradBetaData = BatchNormKernel<T>.BackwardBias(
                        gradOutData, savedN, savedC, savedPlaneSize);

                    if (weight != null)
                        ReverseGradOperations.AccumulateGradient(weight.Tensor, NivaraColumn<T>.Create(gradGammaData));
                    if (bias != null)
                        ReverseGradOperations.AccumulateGradient(bias.Tensor, NivaraColumn<T>.Create(gradBetaData));
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    void UpdateRunningStatsDirect(T[] batchMean, T[] batchInvStd, int channelTotal)
    {
        var result = ModuleHelpers<T>.UpdateRunningStats(
            runningMean, runningVar, numBatchesTracked,
            batchMean, batchInvStd, numFeatures, momentum, eps);
        runningMean = result.runningMean;
        runningVar = result.runningVar;
        numBatchesTracked = result.numBatchesTracked;
    }

    static ReadOnlySpan<T> GetInputSpan(ReverseGradTensor<T> tensor)
        => ModuleHelpers<T>.GetSpan(tensor);

    static ReadOnlySpan<T> GetParamSpan(ReverseGradTensor<T> tensor)
        => ModuleHelpers<T>.GetSpan(tensor);

    public override Dictionary<string, ReverseGradTensor<T>> StateDict()
    {
        var state = base.StateDict();
        if (runningMean != null) state["running_mean"] = runningMean;
        if (runningVar != null) state["running_var"] = runningVar;
        if (numBatchesTracked != null) state["num_batches_tracked"] = numBatchesTracked;
        return state;
    }

    public override void LoadStateDict(IReadOnlyDictionary<string, ReverseGradTensor<T>> stateDict, bool strict = false)
    {
        var paramKeys = new HashSet<string>(stateDict.Keys.Where(k => k is "Weight" or "Bias"));
        var paramDict = paramKeys.Count > 0
            ? paramKeys.ToDictionary(k => k, k => stateDict[k])
            : new Dictionary<string, ReverseGradTensor<T>>();
        base.LoadStateDict(paramDict, strict);
        if (stateDict.TryGetValue("running_mean", out var rm)) runningMean = rm;
        if (stateDict.TryGetValue("running_var", out var rv)) runningVar = rv;
        if (stateDict.TryGetValue("num_batches_tracked", out var nbt)) numBatchesTracked = nbt;
    }
}

public sealed class BatchNorm2d<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly int numFeatures;
    readonly T eps;
    readonly T momentum;
    readonly bool affine;
    readonly bool trackRunningStats;

    readonly Parameter<T>? weight;
    readonly Parameter<T>? bias;

    ReverseGradTensor<T>? runningMean;
    ReverseGradTensor<T>? runningVar;
    ReverseGradTensor<T>? numBatchesTracked;

    public BatchNorm2d(
        int numFeatures,
        float eps = 1e-5f,
        float momentum = 0.1f,
        bool affine = true,
        bool trackRunningStats = true)
    {
        if (numFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(numFeatures));
        if (eps <= 0) throw new ArgumentOutOfRangeException(nameof(eps));
        if (momentum <= 0 || momentum >= 1) throw new ArgumentOutOfRangeException(nameof(momentum));

        this.numFeatures = numFeatures;
        this.eps = T.CreateChecked(eps);
        this.momentum = T.CreateChecked(momentum);
        this.affine = affine;
        this.trackRunningStats = trackRunningStats;

        if (affine)
        {
            var weightData = new T[numFeatures];
            var biasData = new T[numFeatures];
            for (int i = 0; i < numFeatures; i++)
            {
                weightData[i] = T.One;
                biasData[i] = T.Zero;
            }
            weight = new Parameter<T>("Weight", ReverseGradTensor<T>.FromArray(weightData, requiresGrad: true));
            bias = new Parameter<T>("Bias", ReverseGradTensor<T>.FromArray(biasData, requiresGrad: true));
            RegisterParameters(weight, bias);
        }

        if (trackRunningStats)
        {
            var runningMeanData = new T[numFeatures];
            var runningVarData = new T[numFeatures];
            for (int i = 0; i < numFeatures; i++)
            {
                runningMeanData[i] = T.Zero;
                runningVarData[i] = T.One;
            }
            runningMean = ReverseGradTensor<T>.FromArray(runningMeanData, requiresGrad: false);
            runningVar = ReverseGradTensor<T>.FromArray(runningVarData, requiresGrad: false);
            numBatchesTracked = ReverseGradTensor<T>.FromArray(new T[] { T.Zero }, requiresGrad: false);
        }
    }

    public ReverseGradTensor<T> RunningMean => runningMean
        ?? throw new InvalidOperationException("RunningMean is unavailable because this BatchNorm2d was created with trackRunningStats: false.");
    public ReverseGradTensor<T> RunningVar => runningVar
        ?? throw new InvalidOperationException("RunningVar is unavailable because this BatchNorm2d was created with trackRunningStats: false.");
    public ReverseGradTensor<T> NumBatchesTracked => numBatchesTracked
        ?? throw new InvalidOperationException("NumBatchesTracked is unavailable because this BatchNorm2d was created with trackRunningStats: false.");
    public bool TrackRunningStats => trackRunningStats;
    public Parameter<T>? Weight => weight;
    public Parameter<T>? Bias => bias;

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 4) throw new ArgumentException($"BatchNorm2d expects 4D input [N, C, H, W], got {input.Rank}D");

        int n = input.Shape[0];
        int c = input.Shape[1];
        if (c != numFeatures) throw new ArgumentException($"Expected {numFeatures} channels, got {c}");
        int h = input.Shape[2];
        int w = input.Shape[3];
        int hw = h * w;

        var gamma = affine && weight != null
            ? GetParamSpan(weight.Tensor)
            : ReadOnlySpan<T>.Empty;
        var beta = affine && bias != null
            ? GetParamSpan(bias.Tensor)
            : ReadOnlySpan<T>.Empty;

        bool useRunningStats = !IsTraining;

        if (useRunningStats && runningMean != null && runningVar != null)
        {
            var rmSpan = GetParamSpan(runningMean);
            var rvSpan = GetParamSpan(runningVar);
            var evalResult = BatchNormKernel<T>.ForwardEval(
                GetInputSpan(input), n, c, hw,
                gamma, beta, rmSpan, rvSpan, eps, affine);

            var evalTensor = new ReverseGradTensor<T>(
                NivaraColumn<T>.Create(evalResult.Output),
                input.RequiresGrad, input.Shape);

            if (input.RequiresGrad)
            {
                var savedXHat = evalResult.XHat;
                var savedInvStd = evalResult.InvStd;
                var savedGamma = gamma.Length > 0 ? gamma.ToArray() : [];
                bool useAffine = affine;
                int savedN = n, savedC = c;

                var gradFn = new OpNode<T>("BatchNorm2dEval", [input], (typedGradOutput) =>
                {
                    var gradOutData = new T[typedGradOutput.Length];
                    typedGradOutput.CopyTo(gradOutData, default(T)!);

                    var gradInputData = BatchNormKernel<T>.BackwardInput(
                        gradOutData, savedXHat, savedGamma, savedInvStd,
                        savedN, savedC, hw, useAffine);

                    ReverseGradOperations.AccumulateGradient(input, NivaraColumn<T>.Create(gradInputData));
                });

                ComputationGraph.AddNode(evalTensor, gradFn);
            }

            return evalTensor;
        }

        var inputData = GetInputSpan(input);

        var result = BatchNormKernel<T>.Forward(inputData, n, c, hw, gamma, beta, eps, affine);

        if (IsTraining && trackRunningStats)
            UpdateRunningStatsDirect(result.Mean, result.InvStd, n * hw);

        var resultTensor = new ReverseGradTensor<T>(
            NivaraColumn<T>.Create(result.Output),
            input.RequiresGrad, input.Shape);

        if (input.RequiresGrad)
        {
            var savedXHat = result.XHat;
            var savedInvStd = result.InvStd;
            var savedGamma = gamma.Length > 0 ? gamma.ToArray() : [];
            bool useAffine = affine;
            int savedN = n, savedC = c;

            var gradFn = new OpNode<T>("BatchNorm2d", [input], (typedGradOutput) =>
            {
                var gradOutData = new T[typedGradOutput.Length];
                typedGradOutput.CopyTo(gradOutData, default(T)!);

                var gradInputData = BatchNormKernel<T>.BackwardInput(
                    gradOutData, savedXHat, savedGamma, savedInvStd,
                    savedN, savedC, hw, useAffine);

                ReverseGradOperations.AccumulateGradient(input, NivaraColumn<T>.Create(gradInputData));

                if (useAffine)
                {
                    var gradGammaData = BatchNormKernel<T>.BackwardWeight(
                        gradOutData, savedXHat, savedN, savedC, hw);
                    var gradBetaData = BatchNormKernel<T>.BackwardBias(
                        gradOutData, savedN, savedC, hw);

                    if (weight != null)
                        ReverseGradOperations.AccumulateGradient(weight.Tensor, NivaraColumn<T>.Create(gradGammaData));
                    if (bias != null)
                        ReverseGradOperations.AccumulateGradient(bias.Tensor, NivaraColumn<T>.Create(gradBetaData));
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    void UpdateRunningStatsDirect(T[] batchMean, T[] batchInvStd, int channelTotal)
    {
        var result = ModuleHelpers<T>.UpdateRunningStats(
            runningMean, runningVar, numBatchesTracked,
            batchMean, batchInvStd, numFeatures, momentum, eps);
        runningMean = result.runningMean;
        runningVar = result.runningVar;
        numBatchesTracked = result.numBatchesTracked;
    }

    static ReadOnlySpan<T> GetInputSpan(ReverseGradTensor<T> tensor)
        => ModuleHelpers<T>.GetSpan(tensor);

    static ReadOnlySpan<T> GetParamSpan(ReverseGradTensor<T> tensor)
        => ModuleHelpers<T>.GetSpan(tensor);

    public override Dictionary<string, ReverseGradTensor<T>> StateDict()
    {
        var state = base.StateDict();
        if (runningMean != null) state["running_mean"] = runningMean;
        if (runningVar != null) state["running_var"] = runningVar;
        if (numBatchesTracked != null) state["num_batches_tracked"] = numBatchesTracked;
        return state;
    }

    public override void LoadStateDict(IReadOnlyDictionary<string, ReverseGradTensor<T>> stateDict, bool strict = false)
    {
        var paramKeys = new HashSet<string>(stateDict.Keys.Where(k => k is "Weight" or "Bias"));
        var paramDict = paramKeys.Count > 0
            ? paramKeys.ToDictionary(k => k, k => stateDict[k])
            : new Dictionary<string, ReverseGradTensor<T>>();
        base.LoadStateDict(paramDict, strict);
        if (stateDict.TryGetValue("running_mean", out var rm)) runningMean = rm;
        if (stateDict.TryGetValue("running_var", out var rv)) runningVar = rv;
        if (stateDict.TryGetValue("num_batches_tracked", out var nbt)) numBatchesTracked = nbt;
    }
}