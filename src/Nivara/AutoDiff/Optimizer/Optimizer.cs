using Nivara.AutoDiff.Nn;
using System.Numerics;

namespace Nivara.AutoDiff.Optimizer;

public abstract class Optimizer<T> : IDisposable where T : struct, INumber<T>
{
    protected readonly List<ParameterGroup> ParameterGroups = [];
    bool disposed;

    protected Optimizer(T learningRate)
    {
        ValidateLearningRate(learningRate);
        LearningRate = learningRate;
    }

    public T LearningRate { get; }

    public class ParameterGroup
    {
        public IReadOnlyList<Parameter<T>> Parameters { get; }
        public T LearningRate { get; set; }
        public T WeightDecay { get; set; }

        public ParameterGroup(IReadOnlyList<Parameter<T>> parameters, T learningRate, T weightDecay)
        {
            Parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            LearningRate = learningRate;
            WeightDecay = weightDecay;
        }
    }

    public void AddParameterGroup(
        IEnumerable<Parameter<T>> parameters)
    {
        AddParameterGroup(parameters, LearningRate);
    }

    public void AddParameterGroup(
        IEnumerable<Parameter<T>> parameters,
        T learningRate,
        T weightDecay = default)
    {
        if (parameters == null) throw new ArgumentNullException(nameof(parameters));
        ValidateLearningRate(learningRate);
        var list = parameters.Where(p => p != null).ToList();
        if (list.Count > 0)
            ParameterGroups.Add(new ParameterGroup(list.AsReadOnly(), learningRate, weightDecay));
    }

    public void AddParameterGroup(
        Parameter<T> parameter)
    {
        AddParameterGroup(parameter, LearningRate);
    }

    public void AddParameterGroup(
        Parameter<T> parameter,
        T learningRate,
        T weightDecay = default)
    {
        if (parameter == null) throw new ArgumentNullException(nameof(parameter));
        ValidateLearningRate(learningRate);
        ParameterGroups.Add(new ParameterGroup(
            new List<Parameter<T>> { parameter }.AsReadOnly(),
            learningRate,
            weightDecay));
    }

    protected static void ValidateLearningRate(T learningRate)
    {
        if (learningRate <= T.Zero)
            throw new ArgumentException("Learning rate must be positive", nameof(learningRate));
    }

    public abstract void Step();

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
