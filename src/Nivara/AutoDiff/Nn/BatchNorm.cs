using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class BatchNorm1d<T> : Module<T> where T : struct, INumber<T>
{
    readonly int _numFeatures;
    readonly T _eps;
    readonly T _momentum;
    readonly bool _affine;
    readonly bool _trackRunningStats;

    readonly Parameter<T>? _weight;
    readonly Parameter<T>? _bias;

    ReverseGradTensor<T>? _runningMean;
    ReverseGradTensor<T>? _runningVar;
    ReverseGradTensor<T>? _numBatchesTracked;

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

        _numFeatures = numFeatures;
        _eps = T.CreateChecked(eps);
        _momentum = T.CreateChecked(momentum);
        _affine = affine;
        _trackRunningStats = trackRunningStats;

        if (affine)
        {
            var weightData = new T[numFeatures];
            var biasData = new T[numFeatures];
            for (int i = 0; i < numFeatures; i++)
            {
                weightData[i] = T.One;
                biasData[i] = T.Zero;
            }
            _weight = new Parameter<T>("Weight", ReverseGradTensor<T>.FromArray(weightData, requiresGrad: true));
            _bias = new Parameter<T>("Bias", ReverseGradTensor<T>.FromArray(biasData, requiresGrad: true));
            RegisterParameters(_weight, _bias);
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
            _runningMean = ReverseGradTensor<T>.FromArray(runningMeanData, requiresGrad: false);
            _runningVar = ReverseGradTensor<T>.FromArray(runningVarData, requiresGrad: false);
            _numBatchesTracked = ReverseGradTensor<T>.FromArray(new T[] { T.Zero }, requiresGrad: false);
        }
    }

    public ReverseGradTensor<T> RunningMean => _runningMean!;
    public ReverseGradTensor<T> RunningVar => _runningVar!;
    public ReverseGradTensor<T> NumBatchesTracked => _numBatchesTracked!;
    public Parameter<T>? Weight => _weight;
    public Parameter<T>? Bias => _bias;

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 2 && input.Rank != 3)
            throw new ArgumentException($"BatchNorm1d expects 2D [N, C] or 3D [B, C, L] input, got {input.Rank}D");

        int n = input.Shape[0];
        int c = input.Shape[1];
        int planeSize = input.Rank == 3 ? input.Shape[2] : 1;
        if (c != _numFeatures) throw new ArgumentException($"Expected {_numFeatures} channels, got {c}");

        var gamma = _affine && _weight != null
            ? GetParamSpan(_weight.Tensor)
            : ReadOnlySpan<T>.Empty;
        var beta = _affine && _bias != null
            ? GetParamSpan(_bias.Tensor)
            : ReadOnlySpan<T>.Empty;

        bool useRunningStats = !IsTraining;

        if (useRunningStats && _runningMean != null && _runningVar != null)
        {
            var rmSpan = GetParamSpan(_runningMean);
            var rvSpan = GetParamSpan(_runningVar);
            var evalResult = BatchNormKernel<T>.ForwardEval(
                GetInputSpan(input), n, c, planeSize,
                gamma, beta, rmSpan, rvSpan, _eps, _affine);

            var evalTensor = new ReverseGradTensor<T>(
                NivaraColumn<T>.Create(evalResult.Output),
                input.RequiresGrad, input.Shape);

            if (input.RequiresGrad)
            {
                var savedXHat = evalResult.XHat;
                var savedInvStd = evalResult.InvStd;
                var savedGamma = gamma.Length > 0 ? gamma.ToArray() : [];
                int savedN = n, savedC = c, savedPlaneSize = planeSize;

                var gradFn = new OpNode<T>("BatchNorm1dEval", new object[] { input }, (typedGradOutput) =>
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
        var result = BatchNormKernel<T>.Forward(inputData, n, c, planeSize, gamma, beta, _eps, _affine);

        if (_trackRunningStats)
            UpdateRunningStatsDirect(result.Mean, result.InvStd, n * planeSize);

        var resultTensor = new ReverseGradTensor<T>(
            NivaraColumn<T>.Create(result.Output),
            input.RequiresGrad, input.Shape);

        if (input.RequiresGrad)
        {
            var savedXHat = result.XHat;
            var savedInvStd = result.InvStd;
            var savedGamma = gamma.Length > 0 ? gamma.ToArray() : [];
            bool affine = _affine;
            int savedN = n, savedC = c, savedPlaneSize = planeSize;

            var gradFn = new OpNode<T>("BatchNorm1dTrain", new object[] { input }, (typedGradOutput) =>
            {
                var gradOutData = new T[typedGradOutput.Length];
                typedGradOutput.CopyTo(gradOutData, default(T)!);

                var gradInputData = BatchNormKernel<T>.BackwardInput(
                    gradOutData, savedXHat, savedGamma, savedInvStd,
                    savedN, savedC, savedPlaneSize, affine);

                ReverseGradOperations.AccumulateGradient(input, NivaraColumn<T>.Create(gradInputData));

                if (affine)
                {
                    var gradGammaData = BatchNormKernel<T>.BackwardWeight(
                        gradOutData, savedXHat, savedN, savedC, savedPlaneSize);
                    var gradBetaData = BatchNormKernel<T>.BackwardBias(
                        gradOutData, savedN, savedC, savedPlaneSize);

                    if (_weight != null)
                        ReverseGradOperations.AccumulateGradient(_weight.Tensor, NivaraColumn<T>.Create(gradGammaData));
                    if (_bias != null)
                        ReverseGradOperations.AccumulateGradient(_bias.Tensor, NivaraColumn<T>.Create(gradBetaData));
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    void UpdateRunningStatsDirect(T[] batchMean, T[] batchInvStd, int channelTotal)
    {
        var result = ModuleHelpers<T>.UpdateRunningStats(
            _runningMean, _runningVar, _numBatchesTracked,
            batchMean, batchInvStd, _numFeatures, _momentum, _eps);
        _runningMean = result.runningMean;
        _runningVar = result.runningVar;
        _numBatchesTracked = result.numBatchesTracked;
    }

    static ReadOnlySpan<T> GetInputSpan(ReverseGradTensor<T> tensor)
        => ModuleHelpers<T>.GetSpan(tensor);

    static ReadOnlySpan<T> GetParamSpan(ReverseGradTensor<T> tensor)
        => ModuleHelpers<T>.GetSpan(tensor);

    public override Dictionary<string, ReverseGradTensor<T>> StateDict()
    {
        var state = base.StateDict();
        if (_runningMean != null) state["running_mean"] = _runningMean;
        if (_runningVar != null) state["running_var"] = _runningVar;
        if (_numBatchesTracked != null) state["num_batches_tracked"] = _numBatchesTracked;
        return state;
    }

    public override void LoadStateDict(IReadOnlyDictionary<string, ReverseGradTensor<T>> stateDict, bool strict = false)
    {
        var paramKeys = new HashSet<string>(stateDict.Keys.Where(k => k is "Weight" or "Bias"));
        var paramDict = paramKeys.Count > 0
            ? paramKeys.ToDictionary(k => k, k => stateDict[k])
            : new Dictionary<string, ReverseGradTensor<T>>();
        base.LoadStateDict(paramDict, strict);
        if (stateDict.TryGetValue("running_mean", out var rm)) _runningMean = rm;
        if (stateDict.TryGetValue("running_var", out var rv)) _runningVar = rv;
        if (stateDict.TryGetValue("num_batches_tracked", out var nbt)) _numBatchesTracked = nbt;
    }
}

public sealed class BatchNorm2d<T> : Module<T> where T : struct, INumber<T>
{
    readonly int _numFeatures;
    readonly T _eps;
    readonly T _momentum;
    readonly bool _affine;
    readonly bool _trackRunningStats;

    readonly Parameter<T>? _weight;
    readonly Parameter<T>? _bias;

    ReverseGradTensor<T>? _runningMean;
    ReverseGradTensor<T>? _runningVar;
    ReverseGradTensor<T>? _numBatchesTracked;

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

        _numFeatures = numFeatures;
        _eps = T.CreateChecked(eps);
        _momentum = T.CreateChecked(momentum);
        _affine = affine;
        _trackRunningStats = trackRunningStats;

        if (affine)
        {
            var weightData = new T[numFeatures];
            var biasData = new T[numFeatures];
            for (int i = 0; i < numFeatures; i++)
            {
                weightData[i] = T.One;
                biasData[i] = T.Zero;
            }
            _weight = new Parameter<T>("Weight", ReverseGradTensor<T>.FromArray(weightData, requiresGrad: true));
            _bias = new Parameter<T>("Bias", ReverseGradTensor<T>.FromArray(biasData, requiresGrad: true));
            RegisterParameters(_weight, _bias);
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
            _runningMean = ReverseGradTensor<T>.FromArray(runningMeanData, requiresGrad: false);
            _runningVar = ReverseGradTensor<T>.FromArray(runningVarData, requiresGrad: false);
            _numBatchesTracked = ReverseGradTensor<T>.FromArray(new T[] { T.Zero }, requiresGrad: false);
        }
    }

    public ReverseGradTensor<T> RunningMean => _runningMean!;
    public ReverseGradTensor<T> RunningVar => _runningVar!;
    public ReverseGradTensor<T> NumBatchesTracked => _numBatchesTracked!;
    public Parameter<T>? Weight => _weight;
    public Parameter<T>? Bias => _bias;

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 4) throw new ArgumentException($"BatchNorm2d expects 4D input [N, C, H, W], got {input.Rank}D");

        int n = input.Shape[0];
        int c = input.Shape[1];
        if (c != _numFeatures) throw new ArgumentException($"Expected {_numFeatures} channels, got {c}");
        int h = input.Shape[2];
        int w = input.Shape[3];
        int hw = h * w;

        var gamma = _affine && _weight != null
            ? GetParamSpan(_weight.Tensor)
            : ReadOnlySpan<T>.Empty;
        var beta = _affine && _bias != null
            ? GetParamSpan(_bias.Tensor)
            : ReadOnlySpan<T>.Empty;

        bool useRunningStats = !IsTraining;

        if (useRunningStats && _runningMean != null && _runningVar != null)
        {
            var rmSpan = GetParamSpan(_runningMean);
            var rvSpan = GetParamSpan(_runningVar);
            var evalResult = BatchNormKernel<T>.ForwardEval(
                GetInputSpan(input), n, c, hw,
                gamma, beta, rmSpan, rvSpan, _eps, _affine);

            var evalTensor = new ReverseGradTensor<T>(
                NivaraColumn<T>.Create(evalResult.Output),
                input.RequiresGrad, input.Shape);

            if (input.RequiresGrad)
            {
                var savedXHat = evalResult.XHat;
                var savedInvStd = evalResult.InvStd;
                var savedGamma = gamma.Length > 0 ? gamma.ToArray() : [];
                bool affine = _affine;
                int savedN = n, savedC = c;

                var gradFn = new OpNode<T>("BatchNorm2dEval", new object[] { input }, (typedGradOutput) =>
                {
                    var gradOutData = new T[typedGradOutput.Length];
                    typedGradOutput.CopyTo(gradOutData, default(T)!);

                    var gradInputData = BatchNormKernel<T>.BackwardInput(
                        gradOutData, savedXHat, savedGamma, savedInvStd,
                        savedN, savedC, hw, affine);

                    ReverseGradOperations.AccumulateGradient(input, NivaraColumn<T>.Create(gradInputData));
                });

                ComputationGraph.AddNode(evalTensor, gradFn);
            }

            return evalTensor;
        }

        var inputData = GetInputSpan(input);

        var result = BatchNormKernel<T>.Forward(inputData, n, c, hw, gamma, beta, _eps, _affine);

        if (IsTraining && _trackRunningStats)
            UpdateRunningStatsDirect(result.Mean, result.InvStd, n * hw);

        var resultTensor = new ReverseGradTensor<T>(
            NivaraColumn<T>.Create(result.Output),
            input.RequiresGrad, input.Shape);

        if (input.RequiresGrad)
        {
            var savedXHat = result.XHat;
            var savedInvStd = result.InvStd;
            var savedGamma = gamma.Length > 0 ? gamma.ToArray() : [];
            bool affine = _affine;
            int savedN = n, savedC = c;

            var gradFn = new OpNode<T>("BatchNorm2d", new object[] { input }, (typedGradOutput) =>
            {
                var gradOutData = new T[typedGradOutput.Length];
                typedGradOutput.CopyTo(gradOutData, default(T)!);

                var gradInputData = BatchNormKernel<T>.BackwardInput(
                    gradOutData, savedXHat, savedGamma, savedInvStd,
                    savedN, savedC, hw, affine);

                ReverseGradOperations.AccumulateGradient(input, NivaraColumn<T>.Create(gradInputData));

                if (affine)
                {
                    var gradGammaData = BatchNormKernel<T>.BackwardWeight(
                        gradOutData, savedXHat, savedN, savedC, hw);
                    var gradBetaData = BatchNormKernel<T>.BackwardBias(
                        gradOutData, savedN, savedC, hw);

                    if (_weight != null)
                        ReverseGradOperations.AccumulateGradient(_weight.Tensor, NivaraColumn<T>.Create(gradGammaData));
                    if (_bias != null)
                        ReverseGradOperations.AccumulateGradient(_bias.Tensor, NivaraColumn<T>.Create(gradBetaData));
                }
            });

            ComputationGraph.AddNode(resultTensor, gradFn);
        }

        return resultTensor;
    }

    void UpdateRunningStatsDirect(T[] batchMean, T[] batchInvStd, int channelTotal)
    {
        var result = ModuleHelpers<T>.UpdateRunningStats(
            _runningMean, _runningVar, _numBatchesTracked,
            batchMean, batchInvStd, _numFeatures, _momentum, _eps);
        _runningMean = result.runningMean;
        _runningVar = result.runningVar;
        _numBatchesTracked = result.numBatchesTracked;
    }

    static ReadOnlySpan<T> GetInputSpan(ReverseGradTensor<T> tensor)
        => ModuleHelpers<T>.GetSpan(tensor);

    static ReadOnlySpan<T> GetParamSpan(ReverseGradTensor<T> tensor)
        => ModuleHelpers<T>.GetSpan(tensor);

    public override Dictionary<string, ReverseGradTensor<T>> StateDict()
    {
        var state = base.StateDict();
        if (_runningMean != null) state["running_mean"] = _runningMean;
        if (_runningVar != null) state["running_var"] = _runningVar;
        if (_numBatchesTracked != null) state["num_batches_tracked"] = _numBatchesTracked;
        return state;
    }

    public override void LoadStateDict(IReadOnlyDictionary<string, ReverseGradTensor<T>> stateDict, bool strict = false)
    {
        var paramKeys = new HashSet<string>(stateDict.Keys.Where(k => k is "Weight" or "Bias"));
        var paramDict = paramKeys.Count > 0
            ? paramKeys.ToDictionary(k => k, k => stateDict[k])
            : new Dictionary<string, ReverseGradTensor<T>>();
        base.LoadStateDict(paramDict, strict);
        if (stateDict.TryGetValue("running_mean", out var rm)) _runningMean = rm;
        if (stateDict.TryGetValue("running_var", out var rv)) _runningVar = rv;
        if (stateDict.TryGetValue("num_batches_tracked", out var nbt)) _numBatchesTracked = nbt;
    }
}