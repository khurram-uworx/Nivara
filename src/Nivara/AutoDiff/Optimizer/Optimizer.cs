using Nivara.AutoDiff.Nn;
using System.Numerics;

namespace Nivara.AutoDiff.Optimizer;

/// <summary>
/// Abstract base for gradient-descent optimizers. Manages one or more parameter groups with
/// per-group learning rate and weight decay, applies updates in place on <see cref="Step"/>,
/// and supports state-dict save/load and disposal of pooled buffers.
/// </summary>
public abstract class Optimizer<T> : IDisposable where T : struct, IFloatingPointIeee754<T>
{
    protected readonly List<ParameterGroup> ParameterGroups = [];
    bool disposed;
    T learningRate;

    /// <summary>
    /// Creates an optimizer with a default learning rate.
    /// </summary>
    /// <param name="learningRate">The default learning rate (must be positive)</param>
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

    /// <summary>A set of parameters sharing a common learning rate and weight decay.</summary>
    public class ParameterGroup
    {
        /// <summary>The parameters in this group.</summary>
        public IReadOnlyList<Parameter<T>> Parameters { get; }

        /// <summary>The learning rate applied to this group.</summary>
        public T LearningRate { get; internal set; }

        /// <summary>The L2 weight decay applied to this group.</summary>
        public T WeightDecay { get; internal set; }

        internal bool UsesDefaultLearningRate { get; set; }

        /// <summary>
        /// Creates a parameter group. Primarily for internal use; prefer
        /// <see cref="AddParameterGroup(IEnumerable{Parameter{T}})"/>.
        /// </summary>
        /// <param name="parameters">The parameters to track</param>
        /// <param name="learningRate">The group learning rate</param>
        /// <param name="weightDecay">The L2 weight decay</param>
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

    /// <summary>
    /// Registers parameters with the optimizer's current default learning rate and no weight decay.
    /// </summary>
    /// <param name="parameters">The parameters to register (null entries are skipped)</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="parameters"/> is null</exception>
    public void AddParameterGroup(
        IEnumerable<Parameter<T>> parameters)
    {
        AddParameterGroup(parameters, LearningRate, default, usesDefaultLearningRate: true);
    }

    /// <summary>
    /// Registers parameters with an explicit learning rate and weight decay.
    /// </summary>
    /// <param name="parameters">The parameters to register (null entries are skipped)</param>
    /// <param name="learningRate">The group learning rate (must be positive)</param>
    /// <param name="weightDecay">The L2 weight decay</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="parameters"/> is null</exception>
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

    /// <summary>
    /// Registers a single parameter with the optimizer's current default learning rate and
    /// no weight decay.
    /// </summary>
    /// <param name="parameter">The parameter to register</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="parameter"/> is null</exception>
    public void AddParameterGroup(
        Parameter<T> parameter)
    {
        AddParameterGroup(parameter, LearningRate, default, usesDefaultLearningRate: true);
    }

    /// <summary>
    /// Registers a single parameter with an explicit learning rate and weight decay.
    /// </summary>
    /// <param name="parameter">The parameter to register</param>
    /// <param name="learningRate">The group learning rate (must be positive)</param>
    /// <param name="weightDecay">The L2 weight decay</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="parameter"/> is null</exception>
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

    /// <summary>
    /// Validates that a learning rate is positive.
    /// </summary>
    /// <param name="learningRate">The learning rate to validate</param>
    /// <exception cref="ArgumentException">Thrown when the learning rate is not positive</exception>
    protected static void ValidateLearningRate(T learningRate)
    {
        if (learningRate <= T.Zero)
            throw new ArgumentException("Learning rate must be positive", nameof(learningRate));
    }

    /// <summary>
    /// Overrides the learning rate of an existing parameter group.
    /// </summary>
    /// <param name="groupIndex">The zero-based group index</param>
    /// <param name="learningRate">The new learning rate (must be positive)</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="groupIndex"/> is out of range</exception>
    public void SetGroupLearningRate(int groupIndex, T learningRate)
    {
        if (groupIndex < 0 || groupIndex >= ParameterGroups.Count)
            throw new ArgumentOutOfRangeException(nameof(groupIndex));
        ValidateLearningRate(learningRate);
        var group = ParameterGroups[groupIndex];
        group.LearningRate = learningRate;
        group.UsesDefaultLearningRate = false;
    }

    /// <summary>
    /// Overrides the weight decay of an existing parameter group.
    /// </summary>
    /// <param name="groupIndex">The zero-based group index</param>
    /// <param name="weightDecay">The new L2 weight decay</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="groupIndex"/> is out of range</exception>
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
    /// <summary>Applies one optimization step, updating all registered parameters in place.</summary>
    public abstract void Step();

    /// <summary>
    /// Saves the optimizer state (momentum/adaptive buffers and step counters) keyed by name.
    /// </summary>
    /// <returns>A dictionary that can be passed to <see cref="LoadStateDict"/> to resume training</returns>
    public abstract Dictionary<string, T[]> StateDict();

    /// <summary>
    /// Restores optimizer state saved by <see cref="StateDict"/>.
    /// </summary>
    /// <param name="state">The state dictionary produced by a prior <see cref="StateDict"/> call</param>
    public abstract void LoadStateDict(Dictionary<string, T[]> state);

    /// <summary>
    /// Zeroes the accumulated gradients of every registered parameter.
    /// </summary>
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

    /// <summary>
    /// Releases pooled state buffers.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases pooled state buffers.
    /// </summary>
    /// <param name="disposing">Whether managed resources should be released</param>
    protected virtual void Dispose(bool disposing)
    {
        if (disposed) return;
        if (disposing)
        {
            DisposeManaged();
        }
        disposed = true;
    }

    /// <summary>
    /// Releases managed resources held by the optimizer; called from
    /// <see cref="Dispose(bool)"/>. Override to return pooled buffers.
    /// </summary>
    protected virtual void DisposeManaged()
    {
    }
}
