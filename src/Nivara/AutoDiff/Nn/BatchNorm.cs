using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
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
        if (input.Rank != 2) throw new ArgumentException($"BatchNorm1d expects 2D input [N, C], got {input.Rank}D");

        int n = input.Shape[0];
        int c = input.Shape[1];
        if (c != _numFeatures) throw new ArgumentException($"Expected {_numFeatures} channels, got {c}");

        if (IsTraining)
            return ForwardTrain(input, n, c);
        else
            return ForwardEval(input, n, c);
    }

    ReverseGradTensor<T> ForwardTrain(ReverseGradTensor<T> input, int n, int c)
    {
        var mean = ComputeMean(input, n, c);
        var var = ComputeVariance(input, mean, n, c);

        if (_trackRunningStats && _runningMean != null && _runningVar != null && _numBatchesTracked != null)
            UpdateRunningStats(mean, var);

        var normalized = Normalize(input, mean, var, n, c);
        return ApplyAffine(normalized, n, c);
    }

    ReverseGradTensor<T> ForwardEval(ReverseGradTensor<T> input, int n, int c)
    {
        var mean = _runningMean!;
        var var = _runningVar!;
        var normalized = Normalize(input, mean, var, n, c);
        return ApplyAffine(normalized, n, c);
    }

    ReverseGradTensor<T> ComputeMean(ReverseGradTensor<T> input, int n, int c)
    {
        var inputData = new T[n * c];
        input.Data.CopyTo(inputData, T.Zero);

        var meanData = new T[c];
        for (int j = 0; j < c; j++)
        {
            T sum = T.Zero;
            for (int i = 0; i < n; i++)
                sum += inputData[i * c + j];
            meanData[j] = sum / T.CreateChecked(n);
        }

        var meanTensor = ReverseGradTensor<T>.FromArray(meanData, requiresGrad: false);
        meanTensor.Reshape(1, c);
        return meanTensor;
    }

    ReverseGradTensor<T> ComputeVariance(ReverseGradTensor<T> input, ReverseGradTensor<T> mean, int n, int c)
    {
        var inputData = new T[n * c];
        input.Data.CopyTo(inputData, T.Zero);
        var meanData = new T[c];
        mean.Data.CopyTo(meanData, T.Zero);

        var varData = new T[c];
        for (int j = 0; j < c; j++)
        {
            T sum = T.Zero;
            for (int i = 0; i < n; i++)
            {
                T diff = inputData[i * c + j] - meanData[j];
                sum += diff * diff;
            }
            varData[j] = sum / T.CreateChecked(n);
        }

        var varTensor = ReverseGradTensor<T>.FromArray(varData, requiresGrad: false);
        varTensor.Reshape(1, c);
        return varTensor;
    }

    ReverseGradTensor<T> Normalize(ReverseGradTensor<T> input, ReverseGradTensor<T> mean, ReverseGradTensor<T> var, int n, int c)
    {
        var meanExpanded = ExpandParam(mean, n, c);
        var varExpanded = ExpandParam(var, n, c);

        var diff = ReverseGradOperations.Subtract(input, meanExpanded);
        var epsTensor = GradientUtils.Constant(CreateEpsArray(n * c));
        var denom = ReverseGradOperations.Add(varExpanded, epsTensor);
        var std = ReverseGradOperations.Pow(denom, 0.5);
        var normalized = ReverseGradOperations.Divide(diff, std);
        return normalized;
    }

    ReverseGradTensor<T> ApplyAffine(ReverseGradTensor<T> normalized, int n, int c)
    {
        if (!_affine || _weight == null || _bias == null)
            return normalized;

        var scaled = ReverseGradOperations.BroadcastMultiply(normalized, _weight.Tensor);
        var result = ReverseGradOperations.BroadcastAdd(scaled, _bias.Tensor);
        return result;
    }

    ReverseGradTensor<T> ExpandParam(ReverseGradTensor<T> param, int n, int c)
    {
        var expanded = new T[n * c];
        var paramData = new T[c];
        param.Data.CopyTo(paramData, T.Zero);

        for (int i = 0; i < n; i++)
            for (int j = 0; j < c; j++)
                expanded[i * c + j] = paramData[j];

        var expandedTensor = ReverseGradTensor<T>.FromArray(expanded, requiresGrad: false);
        expandedTensor.Reshape(n, c);
        return expandedTensor;
    }

    void UpdateRunningStats(ReverseGradTensor<T> batchMean, ReverseGradTensor<T> batchVar)
    {
        if (_runningMean == null || _runningVar == null || _numBatchesTracked == null)
            return;

        var momentum = _momentum;
        var oneMinusMomentum = T.One - momentum;

        var omArr = new T[_numFeatures];
        var mArr = new T[_numFeatures];
        for (int i = 0; i < _numFeatures; i++)
        {
            omArr[i] = oneMinusMomentum;
            mArr[i] = momentum;
        }

        var newRunningMean = ReverseGradOperations.Add(
            ReverseGradOperations.Multiply(_runningMean, GradientUtils.Constant(omArr)),
            ReverseGradOperations.Multiply(batchMean, GradientUtils.Constant(mArr)));

        var newRunningVar = ReverseGradOperations.Add(
            ReverseGradOperations.Multiply(_runningVar, GradientUtils.Constant(omArr)),
            ReverseGradOperations.Multiply(batchVar, GradientUtils.Constant(mArr)));

        var newCount = ReverseGradOperations.Add(
            _numBatchesTracked,
            GradientUtils.Constant(new T[] { T.One }));

        _runningMean = newRunningMean;
        _runningVar = newRunningVar;
        _numBatchesTracked = newCount;
    }

    T[] CreateEpsArray(int len)
    {
        var arr = new T[len];
        for (int i = 0; i < len; i++) arr[i] = _eps;
        return arr;
    }

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

        if (IsTraining)
            return ForwardTrain(input, n, c, h, w);
        else
            return ForwardEval(input, n, c, h, w);
    }

    ReverseGradTensor<T> ForwardTrain(ReverseGradTensor<T> input, int n, int c, int h, int w)
    {
        var mean = ComputeMean(input, n, c, h, w);
        var var = ComputeVariance(input, mean, n, c, h, w);

        if (_trackRunningStats && _runningMean != null && _runningVar != null && _numBatchesTracked != null)
            UpdateRunningStats(mean, var);

        var normalized = Normalize(input, mean, var, n, c, h, w);
        return ApplyAffine(normalized, n, c, h, w);
    }

    ReverseGradTensor<T> ForwardEval(ReverseGradTensor<T> input, int n, int c, int h, int w)
    {
        var mean = _runningMean!;
        var var = _runningVar!;
        var normalized = Normalize(input, mean, var, n, c, h, w);
        return ApplyAffine(normalized, n, c, h, w);
    }

    ReverseGradTensor<T> ComputeMean(ReverseGradTensor<T> input, int n, int c, int h, int w)
    {
        var inputData = new T[n * c * h * w];
        input.Data.CopyTo(inputData, T.Zero);

        var meanData = new T[c];
        int hw = h * w;
        for (int j = 0; j < c; j++)
        {
            T sum = T.Zero;
            for (int i = 0; i < n; i++)
            {
                for (int k = 0; k < h; k++)
                {
                    for (int l = 0; l < w; l++)
                    {
                        int idx = ((i * c + j) * h + k) * w + l;
                        sum += inputData[idx];
                    }
                }
            }
            meanData[j] = sum / T.CreateChecked(n * hw);
        }

        var meanTensor = ReverseGradTensor<T>.FromArray(meanData, requiresGrad: false);
        meanTensor.Reshape(1, c, 1, 1);
        return meanTensor;
    }

    ReverseGradTensor<T> ComputeVariance(ReverseGradTensor<T> input, ReverseGradTensor<T> mean, int n, int c, int h, int w)
    {
        var inputData = new T[n * c * h * w];
        input.Data.CopyTo(inputData, T.Zero);
        var meanData = new T[c];
        mean.Data.CopyTo(meanData, T.Zero);

        var varData = new T[c];
        int hw = h * w;
        for (int j = 0; j < c; j++)
        {
            T sum = T.Zero;
            for (int i = 0; i < n; i++)
            {
                for (int k = 0; k < h; k++)
                {
                    for (int l = 0; l < w; l++)
                    {
                        int idx = ((i * c + j) * h + k) * w + l;
                        T diff = inputData[idx] - meanData[j];
                        sum += diff * diff;
                    }
                }
            }
            varData[j] = sum / T.CreateChecked(n * hw);
        }

        var varTensor = ReverseGradTensor<T>.FromArray(varData, requiresGrad: false);
        varTensor.Reshape(1, c, 1, 1);
        return varTensor;
    }

    ReverseGradTensor<T> Normalize(ReverseGradTensor<T> input, ReverseGradTensor<T> mean, ReverseGradTensor<T> var, int n, int c, int h, int w)
    {
        var meanExpanded = ExpandParam(mean, n, c, h, w);
        var varExpanded = ExpandParam(var, n, c, h, w);

        var diff = ReverseGradOperations.Subtract(input, meanExpanded);
        var epsTensor = GradientUtils.Constant(CreateEpsArray(n * c * h * w));
        var denom = ReverseGradOperations.Add(varExpanded, epsTensor);
        var std = ReverseGradOperations.Pow(denom, 0.5);
        var normalized = ReverseGradOperations.Divide(diff, std);
        return normalized;
    }

    ReverseGradTensor<T> ApplyAffine(ReverseGradTensor<T> normalized, int n, int c, int h, int w)
    {
        if (!_affine || _weight == null || _bias == null)
            return normalized;

        var scaled = ReverseGradOperations.BroadcastMultiply(normalized, _weight.Tensor);
        var result = ReverseGradOperations.BroadcastAdd(scaled, _bias.Tensor);
        return result;
    }

    ReverseGradTensor<T> ExpandParam(ReverseGradTensor<T> param, int n, int c, int h, int w)
    {
        var expanded = new T[n * c * h * w];
        var paramData = new T[c];
        param.Data.CopyTo(paramData, T.Zero);

        for (int i = 0; i < n; i++)
            for (int j = 0; j < c; j++)
                for (int k = 0; k < h; k++)
                    for (int l = 0; l < w; l++)
                        expanded[((i * c + j) * h + k) * w + l] = paramData[j];

        var expandedTensor = ReverseGradTensor<T>.FromArray(expanded, requiresGrad: false);
        expandedTensor.Reshape(n, c, h, w);
        return expandedTensor;
    }

    void UpdateRunningStats(ReverseGradTensor<T> batchMean, ReverseGradTensor<T> batchVar)
    {
        if (_runningMean == null || _runningVar == null || _numBatchesTracked == null)
            return;

        var momentum = _momentum;
        var oneMinusMomentum = T.One - momentum;

        var meanFlat = new T[_numFeatures];
        var varFlat = new T[_numFeatures];
        batchMean.Data.CopyTo(meanFlat, T.Zero);
        batchVar.Data.CopyTo(varFlat, T.Zero);
        var meanFlatTensor = ReverseGradTensor<T>.FromArray(meanFlat, requiresGrad: false);
        var varFlatTensor = ReverseGradTensor<T>.FromArray(varFlat, requiresGrad: false);

        var omArr = new T[_numFeatures];
        var mArr = new T[_numFeatures];
        for (int i = 0; i < _numFeatures; i++)
        {
            omArr[i] = oneMinusMomentum;
            mArr[i] = momentum;
        }

        var newRunningMean = ReverseGradOperations.Add(
            ReverseGradOperations.Multiply(_runningMean, GradientUtils.Constant(omArr)),
            ReverseGradOperations.Multiply(meanFlatTensor, GradientUtils.Constant(mArr)));

        var newRunningVar = ReverseGradOperations.Add(
            ReverseGradOperations.Multiply(_runningVar, GradientUtils.Constant(omArr)),
            ReverseGradOperations.Multiply(varFlatTensor, GradientUtils.Constant(mArr)));

        var newCount = ReverseGradOperations.Add(
            _numBatchesTracked,
            GradientUtils.Constant(new T[] { T.One }));

        _runningMean = newRunningMean;
        _runningVar = newRunningVar;
        _numBatchesTracked = newCount;
    }

    T[] CreateEpsArray(int len)
    {
        var arr = new T[len];
        for (int i = 0; i < len; i++) arr[i] = _eps;
        return arr;
    }

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