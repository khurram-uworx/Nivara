namespace Nivara.Silk;

/// <summary>F32 weights for the ModernBERT classification head on the Silk (OpenCL) device.</summary>
public sealed class SilkModernBertHeadWeights : IDisposable
{
    public readonly int HiddenDim;
    public readonly int NumClasses;

    public readonly nint HiddenW;
    public readonly nint HiddenB;
    public readonly nint HiddenGradW;
    public readonly nint HiddenGradB;
    public readonly nint HiddenM1W;
    public readonly nint HiddenM1B;
    public readonly nint HiddenM2W;
    public readonly nint HiddenM2B;

    public readonly nint ClassW;
    public readonly nint ClassB;
    public readonly nint ClassGradW;
    public readonly nint ClassGradB;
    public readonly nint ClassM1W;
    public readonly nint ClassM1B;
    public readonly nint ClassM2W;
    public readonly nint ClassM2B;

    readonly SilkRuntime runtime;
    readonly List<nint> owned = [];
    bool disposed;

    public sealed record Param(nint Weight, nint Gradient, nint FirstMoment, nint SecondMoment, int Count, bool Trainable, bool NarrowMatrix);

    public List<Param> Parameters { get; } = [];

    public SilkModernBertHeadWeights(SilkRuntime runtime, int hiddenDim, int numClasses, bool trainable = true)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (hiddenDim <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenDim));
        if (numClasses <= 0) throw new ArgumentOutOfRangeException(nameof(numClasses));

        this.runtime = runtime;
        HiddenDim = hiddenDim;
        NumClasses = numClasses;

        int h2 = hiddenDim * hiddenDim;
        int hc = numClasses * hiddenDim;

        HiddenW = Alloc(h2); HiddenB = Alloc(hiddenDim);
        HiddenGradW = Alloc(h2); HiddenGradB = Alloc(hiddenDim);
        HiddenM1W = Alloc(h2); HiddenM1B = Alloc(hiddenDim);
        HiddenM2W = Alloc(h2); HiddenM2B = Alloc(hiddenDim);

        ClassW = Alloc(hc); ClassB = Alloc(numClasses);
        ClassGradW = Alloc(hc); ClassGradB = Alloc(numClasses);
        ClassM1W = Alloc(hc); ClassM1B = Alloc(numClasses);
        ClassM2W = Alloc(hc); ClassM2B = Alloc(numClasses);

        var rng = new Random(7);
        Init(HiddenW, h2, rng);
        Init(HiddenB, hiddenDim, rng);
        Init(ClassW, hc, rng);
        Init(ClassB, numClasses, rng);

        if (trainable)
        {
            Parameters.Add(new Param(HiddenW, HiddenGradW, HiddenM1W, HiddenM2W, h2, true, false));
            Parameters.Add(new Param(HiddenB, HiddenGradB, HiddenM1B, HiddenM2B, hiddenDim, true, false));
            Parameters.Add(new Param(ClassW, ClassGradW, ClassM1W, ClassM2W, hc, true, false));
            Parameters.Add(new Param(ClassB, ClassGradB, ClassM1B, ClassM2B, numClasses, true, false));
        }
    }

    void Init(nint buf, int count, Random rng)
    {
        var data = new float[count];
        for (int i = 0; i < count; i++) data[i] = (float)(rng.NextDouble() * 2 - 1) * 0.01f;
        runtime.Upload(buf, data);
    }

    nint Alloc(int count)
    {
        nint b = runtime.Allocate(count);
        owned.Add(b);
        return b;
    }

    public Dictionary<string, (float[] Data, int[] Shape)> Export()
    {
        var dict = new Dictionary<string, (float[] Data, int[] Shape)>();
        dict["head.hidden.weight"] = (runtime.Download(HiddenW, HiddenDim * HiddenDim), new[] { HiddenDim, HiddenDim });
        dict["head.hidden.bias"] = (runtime.Download(HiddenB, HiddenDim), new[] { HiddenDim });
        dict["head.classifier.weight"] = (runtime.Download(ClassW, NumClasses * HiddenDim), new[] { NumClasses, HiddenDim });
        dict["head.classifier.bias"] = (runtime.Download(ClassB, NumClasses), new[] { NumClasses });
        return dict;
    }

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        foreach (nint b in owned) runtime.Release(b);
        owned.Clear();
    }
}
