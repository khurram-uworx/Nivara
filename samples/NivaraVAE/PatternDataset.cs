using Nivara;

namespace NivaraVAE;

public sealed class PatternDataset
{
    readonly float[] _data;

    public NivaraFrame Frame { get; }
    public int Count { get; }
    public int GridSize { get; }
    public int NumPixels { get; }

    public PatternDataset(int numPatterns, int gridSize, int seed = 42)
    {
        GridSize = gridSize;
        NumPixels = gridSize * gridSize;
        Count = numPatterns;
        _data = new float[numPatterns * NumPixels];

        var rng = new Random(seed);
        for (int i = 0; i < numPatterns; i++)
        {
            var pattern = new float[NumPixels];
            var type = (PatternType)(rng.Next(6));

            switch (type)
            {
                case PatternType.Circle:
                    GenerateCircle(pattern, gridSize, rng);
                    break;
                case PatternType.Stripes:
                    GenerateStripes(pattern, gridSize, rng);
                    break;
                case PatternType.Blob:
                    GenerateBlob(pattern, gridSize, rng);
                    break;
                case PatternType.Checkerboard:
                    GenerateCheckerboard(pattern, gridSize, rng);
                    break;
                case PatternType.Corner:
                    GenerateCorner(pattern, gridSize, rng);
                    break;
                case PatternType.Cross:
                    GenerateCross(pattern, gridSize, rng);
                    break;
            }

            Array.Copy(pattern, 0, _data, i * NumPixels, NumPixels);
        }

        Frame = NivaraFrame.Create(("pixels", NivaraColumn<float>.Create(_data)));
    }

    PatternDataset(float[] data, int count, int gridSize)
    {
        _data = data;
        Count = count;
        GridSize = gridSize;
        NumPixels = gridSize * gridSize;
        Frame = NivaraFrame.Create(("pixels", NivaraColumn<float>.Create(_data)));
    }

    public void Save(string path)
    {
        using var writer = new StreamWriter(path);
        writer.WriteLine($"{GridSize} {Count}");
        for (int i = 0; i < Count; i++)
        {
            var start = i * NumPixels;
            for (int j = 0; j < NumPixels; j++)
            {
                if (j > 0) writer.Write(' ');
                writer.Write(_data[start + j].ToString("F4"));
            }
            writer.WriteLine();
        }
    }

    public static PatternDataset Load(string path)
    {
        using var reader = new StreamReader(path);
        var header = reader.ReadLine()!.Split(' ');
        int gridSize = int.Parse(header[0]);
        int count = int.Parse(header[1]);
        int numPixels = gridSize * gridSize;
        var data = new float[count * numPixels];

        for (int i = 0; i < count; i++)
        {
            var line = reader.ReadLine()!.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            for (int j = 0; j < numPixels && j < line.Length; j++)
                data[i * numPixels + j] = float.Parse(line[j]);
        }

        return new PatternDataset(data, count, gridSize);
    }

    public float[] GetPattern(int index)
    {
        var pattern = new float[NumPixels];
        Array.Copy(_data, index * NumPixels, pattern, 0, NumPixels);
        return pattern;
    }

    public static string RenderPattern(ReadOnlySpan<float> pixels, int gridSize)
    {
        var sb = new System.Text.StringBuilder();
        for (int y = 0; y < gridSize; y++)
        {
            for (int x = 0; x < gridSize; x++)
            {
                sb.Append(pixels[y * gridSize + x] > 0.5f ? "█" : "·");
            }
            if (y < gridSize - 1) sb.AppendLine();
        }
        return sb.ToString();
    }

    static void GenerateCircle(float[] pattern, int gridSize, Random rng)
    {
        var cx = rng.Next(gridSize);
        var cy = rng.Next(gridSize);
        var radius = rng.Next(gridSize / 4, gridSize / 2 + 1);

        for (int y = 0; y < gridSize; y++)
            for (int x = 0; x < gridSize; x++)
            {
                double dist = Math.Sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
                pattern[y * gridSize + x] = dist <= radius ? 1.0f : 0.0f;
            }
    }

    static void GenerateStripes(float[] pattern, int gridSize, Random rng)
    {
        bool horizontal = rng.Next(2) == 0;
        int thickness = rng.Next(1, gridSize / 2 + 1);
        int phase = rng.Next(gridSize);

        for (int y = 0; y < gridSize; y++)
            for (int x = 0; x < gridSize; x++)
            {
                int coord = horizontal ? y : x;
                pattern[y * gridSize + x] = ((coord + phase) % (2 * thickness)) < thickness ? 1.0f : 0.0f;
            }
    }

    static void GenerateBlob(float[] pattern, int gridSize, Random rng)
    {
        double cx = rng.NextDouble() * gridSize;
        double cy = rng.NextDouble() * gridSize;
        double sigma = rng.NextDouble() * gridSize / 4 + gridSize / 8;

        for (int y = 0; y < gridSize; y++)
            for (int x = 0; x < gridSize; x++)
            {
                double val = Math.Exp(-((x - cx) * (x - cx) + (y - cy) * (y - cy)) / (2 * sigma * sigma));
                pattern[y * gridSize + x] = val > 0.5 ? 1.0f : 0.0f;
            }
    }

    static void GenerateCheckerboard(float[] pattern, int gridSize, Random rng)
    {
        int cellSize = rng.Next(1, gridSize / 2 + 1);

        for (int y = 0; y < gridSize; y++)
            for (int x = 0; x < gridSize; x++)
                pattern[y * gridSize + x] = ((x / cellSize) + (y / cellSize)) % 2 == 0 ? 1.0f : 0.0f;
    }

    static void GenerateCorner(float[] pattern, int gridSize, Random rng)
    {
        bool left = rng.Next(2) == 0;
        bool top = rng.Next(2) == 0;
        int half = gridSize / 2;

        for (int y = 0; y < gridSize; y++)
            for (int x = 0; x < gridSize; x++)
            {
                bool inX = left ? x < half : x >= half;
                bool inY = top ? y < half : y >= half;
                pattern[y * gridSize + x] = (inX && inY) ? 1.0f : 0.0f;
            }
    }

    static void GenerateCross(float[] pattern, int gridSize, Random rng)
    {
        int center = gridSize / 2;
        int armWidth = rng.Next(1, gridSize / 2 + 1);
        int halfArm = armWidth / 2;

        for (int y = 0; y < gridSize; y++)
            for (int x = 0; x < gridSize; x++)
            {
                bool horiz = Math.Abs(y - center) <= halfArm;
                bool vert = Math.Abs(x - center) <= halfArm;
                pattern[y * gridSize + x] = (horiz || vert) ? 1.0f : 0.0f;
            }
    }

    enum PatternType
    {
        Circle,
        Stripes,
        Blob,
        Checkerboard,
        Corner,
        Cross
    }
}
