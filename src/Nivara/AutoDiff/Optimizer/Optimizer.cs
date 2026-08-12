using Nivara.AutoDiff.Nn;
using System.Numerics;

namespace Nivara.AutoDiff.Optimizer;

public abstract class Optimizer<T> : IDisposable where T : struct, IFloatingPointIeee754<T>
{
    protected readonly List<ParameterGroup> ParameterGroups = [];
    bool disposed;
    T learningRate;

    protected Optimizer(T learningRate)
    {
        ValidateLearningRate(learningRate);
        this.learningRate = learningRate;
    }

    /// <summary>
    /// Gets or sets the default learning rate. Setting it forwards to every
    /// parameter group that was created without an explicit learning-rate
    /// override; groups created with an explicit override (or later managed via
    /// <see cref="SetGroupLearningRate"/>) are left untouched.
    /// </summary>
    public T LearningRate
    {
        get => learningRate;
        set
        {
            ValidateLearningRate(value);
            learningRate = value;
            foreach (var group in ParameterGroups)
            {
                if (group.UsesDefaultLearningRate)
                    group.LearningRate = value;
            }
        }
    }

    public class ParameterGroup
    {
        public IReadOnlyList<Parameter<T>> Parameters { get; }
        public T LearningRate { get; internal set; }
        public T WeightDecay { get; internal set; }

        internal bool UsesDefaultLearningRate { get; set; }

        public ParameterGroup(IReadOnlyList<Parameter<T>> parameters, T learningRate, T weightDecay)
            : this(parameters, learningRate, weightDecay, usesDefaultLearningRate: false)
        {
        }

        internal ParameterGroup(IReadOnlyList<Parameter<T>> parameters, T learningRate, T weightDecay, bool usesDefaultLearningRate)
        {
            Parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            LearningRate = learningRate;
            WeightDecay = weightDecay;
            UsesDefaultLearningRate = usesDefaultLearningRate;
        }
    }

    public void AddParameterGroup(
        IEnumerable<Parameter<T>> parameters)
    {
        AddParameterGroup(parameters, LearningRate, default, usesDefaultLearningRate: true);
    }

    public void AddParameterGroup(
        IEnumerable<Parameter<T>> parameters,
        T learningRate,
        T weightDecay = default)
    {
        AddParameterGroup(parameters, learningRate, weightDecay, usesDefaultLearningRate: false);
    }

    void AddParameterGroup(
        IEnumerable<Parameter<T>> parameters,
        T learningRate,
        T weightDecay,
        bool usesDefaultLearningRate)
    {
        if (parameters == null) throw new ArgumentNullException(nameof(parameters));
        ValidateLearningRate(learningRate);
        var list = parameters.Where(p => p != null).ToList();
        if (list.Count > 0)
            ParameterGroups.Add(new ParameterGroup(list.AsReadOnly(), learningRate, weightDecay, usesDefaultLearningRate));
    }

    public void AddParameterGroup(
        Parameter<T> parameter)
    {
        AddParameterGroup(parameter, LearningRate, default, usesDefaultLearningRate: true);
    }

    public void AddParameterGroup(
        Parameter<T> parameter,
        T learningRate,
        T weightDecay = default)
    {
        AddParameterGroup(parameter, learningRate, weightDecay, usesDefaultLearningRate: false);
    }

    void AddParameterGroup(
        Parameter<T> parameter,
        T learningRate,
        T weightDecay,
        bool usesDefaultLearningRate)
    {
        if (parameter == null) throw new ArgumentNullException(nameof(parameter));
        ValidateLearningRate(learningRate);
        ParameterGroups.Add(new ParameterGroup(
            new List<Parameter<T>> { parameter }.AsReadOnly(),
            learningRate,
            weightDecay,
            usesDefaultLearningRate));
    }

    protected static void ValidateLearningRate(T learningRate)
    {
        if (learningRate <= T.Zero)
            throw new ArgumentException("Learning rate must be positive", nameof(learningRate));
    }

    public void SetGroupLearningRate(int groupIndex, T learningRate)
    {
        if (groupIndex < 0 || groupIndex >= ParameterGroups.Count)
            throw new ArgumentOutOfRangeException(nameof(groupIndex));
        ValidateLearningRate(learningRate);
        var group = ParameterGroups[groupIndex];
        group.LearningRate = learningRate;
        group.UsesDefaultLearningRate = false;
    }

    public void SetGroupWeightDecay(int groupIndex, T weightDecay)
    {
        if (groupIndex < 0 || groupIndex >= ParameterGroups.Count)
            throw new ArgumentOutOfRangeException(nameof(groupIndex));
        ParameterGroups[groupIndex].WeightDecay = weightDecay;
    }

    /// <summary>
    /// Applies updates to all registered parameters, writing in place: each
    /// parameter's tensor is reused (never replaced) and its version is bumped
    /// via <c>Touch()</c>. This method leaves each parameter's <c>Grad</c> slot
    /// intact — accumulation happens during <c>Backward()</c>, which adds into
    /// the existing slot. A <c>Step()</c> without a subsequent <c>ZeroGrad()</c>
    /// therefore accumulates stale gradients across steps (PyTorch semantics).
    /// The built-in training loops call <c>ZeroGrad()</c> each iteration.
    /// </summary>
    public abstract void Step();

    public abstract Dictionary<string, T[]> StateDict();

    public abstract void LoadStateDict(Dictionary<string, T[]> state);

    public virtual void ZeroGrad()
    {
        foreach (var group in ParameterGroups)
        {
            foreach (var param in group.Parameters)
            {
                param.Tensor.ZeroGrad();
            }
        }
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (disposed) return;
        if (disposing)
        {
            DisposeManaged();
        }
        disposed = true;
    }

    protected virtual void DisposeManaged()
    {
    }
}
