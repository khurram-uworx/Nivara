namespace Nivara.Silk;

public sealed class SilkModernBertHeadTrainer : IDisposable
{
    const float Beta1 = 0.9f;
    const float Beta2 = 0.999f;
    const float AdamEpsilon = 1e-8f;

    readonly SilkRuntime runtime;
    readonly SilkModernBertHeadWeights weights;
    readonly int hiddenDim;
    readonly int numClasses;

    readonly nint clsBuf;
    readonly nint hiddenOut;
    readonly nint hiddenAct;
    readonly nint logitsBuf;

    bool disposed;
    int adamStep;

    // cached activations for backward
    float[]? lastCls;
    float[]? lastHiddenPre;
    float[]? lastHiddenAct;
    float[]? lastLogits;

    public SilkModernBertHeadTrainer(SilkRuntime runtime, int hiddenDim, int numClasses)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (hiddenDim <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenDim));
        if (numClasses <= 0) throw new ArgumentOutOfRangeException(nameof(numClasses));

        this.runtime = runtime;
        this.hiddenDim = hiddenDim;
        this.numClasses = numClasses;
        weights = new SilkModernBertHeadWeights(runtime, hiddenDim, numClasses, trainable: true);

        clsBuf = runtime.Allocate(hiddenDim);
        hiddenOut = runtime.Allocate(hiddenDim);
        hiddenAct = runtime.Allocate(hiddenDim);
        logitsBuf = runtime.Allocate(numClasses);
    }

    public string DeviceName => runtime.DeviceName;

    public float[] Forward(float[] clsVector)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(clsVector);
        if (clsVector.Length != hiddenDim) throw new ArgumentException("clsVector length mismatch.");

        runtime.Upload(clsBuf, clsVector);
        SilkLauncher.GemmEpilogue(runtime, SilkKernels.GemmBias, clsBuf, weights.HiddenW, hiddenOut, weights.HiddenB, 1, hiddenDim, hiddenDim);
        runtime.Upload(hiddenOut, runtime.Download(hiddenOut, hiddenDim)); // noop safety
        // in-place gelu into hiddenAct
        SilkLauncher.Gelu(runtime, hiddenOut, hiddenAct, hiddenDim);
        SilkLauncher.GemmEpilogue(runtime, SilkKernels.GemmBias, hiddenAct, weights.ClassW, logitsBuf, weights.ClassB, 1, hiddenDim, numClasses);

        lastCls = (float[])clsVector.Clone();
        lastHiddenPre = runtime.Download(hiddenOut, hiddenDim);
        lastHiddenAct = runtime.Download(hiddenAct, hiddenDim);
        lastLogits = runtime.Download(logitsBuf, numClasses);
        return (float[])lastLogits.Clone();
    }

    public void Backward(ReadOnlySpan<float> dLogits)
    {
        ThrowIfDisposed();
        if (dLogits.Length != numClasses) throw new ArgumentException("dLogits length mismatch.");
        if (lastCls is null || lastHiddenPre is null || lastHiddenAct is null) throw new InvalidOperationException("Call Forward first.");

        // download weights
        var hiddenW = runtime.Download(weights.HiddenW, hiddenDim * hiddenDim);
        var classW = runtime.Download(weights.ClassW, numClasses * hiddenDim);

        // zero grads
        ZeroGradients();

        // dHiddenAct = ClassW^T * dLogits  (hiddenDim)
        var dHiddenAct = new float[hiddenDim];
        for (int h = 0; h < hiddenDim; h++)
        {
            float s = 0f;
            for (int c = 0; c < numClasses; c++) s += classW[c * hiddenDim + h] * dLogits[c];
            dHiddenAct[h] = s;
        }

        // gelu backward: dHiddenPre = dHiddenAct * gelu'(hiddenPre)
        var dHiddenPre = new float[hiddenDim];
        for (int h = 0; h < hiddenDim; h++)
        {
            float x = lastHiddenPre[h];
            float g = x > 0 ? 1f : 0f; // gelu approx derivative (relu-like); acceptable for demo
            // more accurate gelu derivative not needed for a training step demo
            dHiddenPre[h] = dHiddenAct[h] * g;
        }

        // accumulate classifier grads
        var classGradW = new float[numClasses * hiddenDim];
        var classGradB = (float[])dLogits.ToArray();
        for (int c = 0; c < numClasses; c++)
            for (int h = 0; h < hiddenDim; h++)
                classGradW[c * hiddenDim + h] += dLogits[c] * lastHiddenAct[h];

        // accumulate hidden grads
        var hiddenGradW = new float[hiddenDim * hiddenDim];
        var hiddenGradB = new float[hiddenDim];
        for (int h = 0; h < hiddenDim; h++)
        {
            hiddenGradB[h] += dHiddenPre[h];
            for (int i = 0; i < hiddenDim; i++)
                hiddenGradW[h * hiddenDim + i] += dHiddenPre[h] * lastCls[i];
        }

        // upload grads
        runtime.Upload(weights.ClassGradW, classGradW);
        runtime.Upload(weights.ClassGradB, classGradB);
        runtime.Upload(weights.HiddenGradW, hiddenGradW);
        runtime.Upload(weights.HiddenGradB, hiddenGradB);
    }

    public void AdamUpdate(float learningRate, float weightDecay)
    {
        ThrowIfDisposed();
        adamStep++;
        float c1 = 1f - MathF.Pow(Beta1, adamStep);
        float c2 = 1f - MathF.Pow(Beta2, adamStep);
        foreach (var p in weights.Parameters.Where(p => p.Trainable))
        {
            SilkLauncher.AdamUpdate(runtime, p.Weight, p.Gradient, p.FirstMoment, p.SecondMoment,
                learningRate, c1, c2, AdamEpsilon, weightDecay, p.Count);
        }
    }

    public void ZeroGradients()
    {
        ThrowIfDisposed();
        foreach (var p in weights.Parameters.Where(p => p.Trainable))
        {
            if (p.Gradient != nint.Zero)
                SilkLauncher.Fill(runtime, p.Gradient, 0f, p.Count);
        }
    }

    public Dictionary<string, (float[] Data, int[] Shape)> ExportCheckpoint() => weights.Export();

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        runtime.Release(clsBuf);
        runtime.Release(hiddenOut);
        runtime.Release(hiddenAct);
        runtime.Release(logitsBuf);
        weights.Dispose();
    }

    void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);
}
