using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using Nivara;

namespace NivaraTimeSeries;

public enum AnomalyType { None, Spike, LevelShift, TrendChange }

public sealed class MetricsGenerator
{
    readonly float[] _data;
    readonly bool[] _isAnomaly;
    readonly AnomalyType[] _anomalyTypes;
    readonly int _seed;

    public int Count { get; }
    public int NumChannels => 4;
    public int WindowSize { get; }
    public int ElementsPerWindow => NumChannels * WindowSize;

    public MetricsGenerator(int numSamples, int windowSize, int seed = 42, float anomalyRatio = 0.15f)
    {
        Count = numSamples;
        WindowSize = windowSize;
        _seed = seed;
        _data = new float[numSamples * ElementsPerWindow];
        _isAnomaly = new bool[numSamples];
        _anomalyTypes = new AnomalyType[numSamples];

        var rng = new Random(seed);
        int anomalyCount = (int)(numSamples * anomalyRatio);

        var anomalyIndices = new HashSet<int>();
        while (anomalyIndices.Count < anomalyCount)
            anomalyIndices.Add(rng.Next(numSamples));

        for (int i = 0; i < numSamples; i++)
        {
            bool isAnom = anomalyIndices.Contains(i);
            _isAnomaly[i] = isAnom;
            _anomalyTypes[i] = isAnom ? (AnomalyType)(rng.Next(3) + 1) : AnomalyType.None;

            GenerateWindow(i, rng, isAnom, _anomalyTypes[i]);
        }
    }

    void GenerateWindow(int index, Random rng, bool injectAnomaly, AnomalyType anomalyType)
    {
        int baseT = index * WindowSize;
        int offset = index * ElementsPerWindow;

        var cpu = ArrayPool<float>.Shared.Rent(WindowSize);
        var mem = ArrayPool<float>.Shared.Rent(WindowSize);
        var disk = ArrayPool<float>.Shared.Rent(WindowSize);
        var net = ArrayPool<float>.Shared.Rent(WindowSize);
        try
        {
            for (int t = 0; t < WindowSize; t++)
            {
                int globalT = baseT + t;
                float noise3 = (float)(rng.NextDouble() * 6 - 3);
                float noise5 = (float)(rng.NextDouble() * 10 - 5);
                float noise2 = (float)(rng.NextDouble() * 4 - 2);

                cpu[t] = 50 + 20 * MathF.Sin(2 * MathF.PI * globalT / 96) + noise3;
                mem[t] = 40 + 0.05f * globalT - 5 * MathF.Floor(globalT / 20f) + noise3;
                disk[t] = 5 + 15 * MathF.Max(0, MathF.Sin(2 * MathF.PI * globalT / 24)) + noise2;
                net[t] = 30 + 15 * MathF.Sin(2 * MathF.PI * globalT / 96) + noise5;
            }

            if (injectAnomaly)
            {
                int anomalyStart = rng.Next(Math.Max(1, WindowSize / 4), Math.Max(2, WindowSize / 2));
                int anomalyLen = Math.Min(10, WindowSize - anomalyStart);

                switch (anomalyType)
                {
                    case AnomalyType.Spike:
                        for (int t = anomalyStart; t < anomalyStart + anomalyLen && t < WindowSize; t++)
                            cpu[t] = 95 + (float)(rng.NextDouble() * 4 - 2);
                        break;
                    case AnomalyType.LevelShift:
                        for (int t = anomalyStart; t < WindowSize; t++)
                            mem[t] += 25;
                        break;
                    case AnomalyType.TrendChange:
                        for (int t = anomalyStart; t < WindowSize; t++)
                            disk[t] *= 1.5f;
                        break;
                }
            }

            NormalizeChannel(cpu, WindowSize);
            NormalizeChannel(mem, WindowSize);
            NormalizeChannel(disk, WindowSize);
            NormalizeChannel(net, WindowSize);

            CopyChannel(cpu, _data, offset, WindowSize);
            CopyChannel(mem, _data, offset + WindowSize, WindowSize);
            CopyChannel(disk, _data, offset + 2 * WindowSize, WindowSize);
            CopyChannel(net, _data, offset + 3 * WindowSize, WindowSize);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(cpu);
            ArrayPool<float>.Shared.Return(mem);
            ArrayPool<float>.Shared.Return(disk);
            ArrayPool<float>.Shared.Return(net);
        }
    }

    static void NormalizeChannel(float[] channel, int length)
    {
        var span = channel.AsSpan(0, length);
        float min = TensorPrimitives.Min(span);
        float max = TensorPrimitives.Max(span);
        float range = max - min;
        if (range < 1e-6f)
        {
            span.Fill(0.5f);
            return;
        }
        TensorPrimitives.Add(span, -min, span);
        TensorPrimitives.Divide(span, range, span);
    }

    static void CopyChannel(float[] source, float[] dest, int destOffset, int length)
    {
        source.AsSpan(0, length).CopyTo(dest.AsSpan(destOffset, length));
    }

    public ReadOnlySpan<float> GetWindow(int index) =>
        _data.AsSpan(index * ElementsPerWindow, ElementsPerWindow);

    public bool IsAnomaly(int index) => _isAnomaly[index];

    public AnomalyType GetAnomalyType(int index) => _anomalyTypes[index];

    public float[] GetWindowArray(int index)
    {
        var window = new float[ElementsPerWindow];
        GetWindow(index).CopyTo(window);
        return window;
    }

    public NivaraFrame Frame
    {
        get
        {
            var column = NivaraColumn<float>.Create(_data);
            return NivaraFrame.Create(("metrics", column));
        }
    }

    public void Save(string path)
    {
        using var writer = new StreamWriter(path);
        writer.WriteLine("index,is_anomaly,anomaly_type,cpu_0..net_W");
        for (int i = 0; i < Count; i++)
        {
            var window = GetWindow(i);
            var values = new string[window.Length];
            for (int j = 0; j < window.Length; j++)
                values[j] = window[j].ToString("F4");
            writer.WriteLine($"{i},{_isAnomaly[i]},{_anomalyTypes[i]},{string.Join(",", values)}");
        }
    }

    public static MetricsGenerator Load(string path)
    {
        var lines = File.ReadAllLines(path);
        int count = lines.Length - 1;
        if (count <= 0) throw new InvalidOperationException("Empty data file");

        var firstValues = lines[1].Split(',');
        int totalElements = firstValues.Length - 3;
        int windowSize = totalElements / 4;

        var generator = new MetricsGenerator(count, windowSize, seed: 0, anomalyRatio: 0f);

        for (int i = 0; i < count; i++)
        {
            var parts = lines[i + 1].Split(',');
            generator._isAnomaly[i] = bool.Parse(parts[1]);
            generator._anomalyTypes[i] = Enum.Parse<AnomalyType>(parts[2]);
            for (int j = 0; j < totalElements; j++)
                generator._data[i * totalElements + j] = float.Parse(parts[j + 3]);
        }

        return generator;
    }
}
