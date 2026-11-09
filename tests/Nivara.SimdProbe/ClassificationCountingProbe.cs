using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Nivara.SimdProbe;

internal static class ClassificationCountingProbe
{
    static readonly int[] RowCounts = [10_000, 100_000, 1_000_000];
    static readonly int[] ClassCounts = [2, 10, 100, 1_000];
    const int Rounds = 12;
    const int Warmups = 4;

    public static int Run()
    {
        Console.WriteLine("=== Classification counting probe (issue #1067) ===");
        Console.WriteLine($"Runtime: {Environment.Version}  Platform: {RuntimeInformation.OSArchitecture}");
        if (IsDebugBuild())
        {
            Console.WriteLine();
            Console.WriteLine("  *** DEBUG BUILD - Re-run with: dotnet run -c Release --project tests/Nivara.SimdProbe -- metrics");
        }
        Console.WriteLine();

        Console.WriteLine("H2 -- dense histogram, int labels (dense 0..k-1)");
        Console.WriteLine($"{"n",10} {"k",6} {"base ms",10} {"dense ms",10} {"ratio",8}  {"verdict",-30}");
        foreach (int n in RowCounts)
            foreach (int k in ClassCounts)
            {
                var (pred, actual) = MakeInt(n, k);
                var (baseMs, candMs, baseWins, candWins) = Compare(
                    () => BaselineHashInt(pred, actual, k),
                    () => DenseInt(pred, actual, k));
                PrintRow(n, k, baseMs, candMs, baseWins, candWins);
            }

        Console.WriteLine();
        Console.WriteLine("H1 -- marginal-only counting (no full k*k matrix)");
        Console.WriteLine($"{"type",8} {"n",10} {"k",6} {"base ms",10} {"margin ms",10} {"ratio",8}  {"verdict",-30}");
        foreach (string type in new[] { "int", "string" })
            foreach (int n in RowCounts)
                foreach (int k in ClassCounts)
                {
                    Action baseline;
                    Action candidate;
                    if (type == "int")
                    {
                        var (pred, actual) = MakeInt(n, k);
                        baseline = () => BaselineHashInt(pred, actual, k);
                        candidate = () => MarginalInt(pred, actual, k);
                    }
                    else
                    {
                        var (pred, actual) = MakeString(n, k);
                        baseline = () => BaselineHashString(pred, actual, k);
                        candidate = () => MarginalString(pred, actual, k);
                    }
                    var (bMs, cMs, bW, cW) = Compare(baseline, candidate);
                    Console.Write($"{type,8} ");
                    PrintRow(n, k, bMs, cMs, bW, cW, false);
                    Console.WriteLine();
                }
        return 0;
    }

    static (int[] Pred, int[] Actual) MakeInt(int n, int k)
    {
        var pred = new int[n];
        var actual = new int[n];
        for (int i = 0; i < n; i++)
        {
            pred[i] = i % k;
            actual[i] = (i * 31 + 7) % k;
        }
        return (pred, actual);
    }

    static (string[] Pred, string[] Actual) MakeString(int n, int k)
    {
        var labels = new string[k];
        for (int c = 0; c < k; c++) labels[c] = "c" + c;
        var pred = new string[n];
        var actual = new string[n];
        for (int i = 0; i < n; i++)
        {
            pred[i] = labels[i % k];
            actual[i] = labels[(i * 31 + 7) % k];
        }
        return (pred, actual);
    }

    static void BaselineHashInt(int[] predicted, int[] actual, int k)
    {
        int n = predicted.Length;
        var seen = new HashSet<int>();
        var labels = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (seen.Add(predicted[i])) labels.Add(predicted[i]);
            if (seen.Add(actual[i])) labels.Add(actual[i]);
        }
        if (seen.Count == 0) return;
        labels.Sort();
        int classes = labels.Count;
        var index = new Dictionary<int, int>(classes);
        for (int i = 0; i < classes; i++) index[labels[i]] = i;
        var matrix = new long[(long)classes * classes];
        int validPairs = 0;
        for (int i = 0; i < n; i++)
        {
            matrix[(long)index[predicted[i]] * classes + index[actual[i]]]++;
            validPairs++;
        }
        _ = validPairs;
        _ = matrix;
    }

    static void BaselineHashString(string[] predicted, string[] actual, int k)
    {
        int n = predicted.Length;
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var labels = new List<string>();
        for (int i = 0; i < n; i++)
        {
            if (seen.Add(predicted[i])) labels.Add(predicted[i]);
            if (seen.Add(actual[i])) labels.Add(actual[i]);
        }
        if (seen.Count == 0) return;
        labels.Sort(StringComparer.Ordinal);
        int classes = labels.Count;
        var index = new Dictionary<string, int>(classes, StringComparer.Ordinal);
        for (int i = 0; i < classes; i++) index[labels[i]] = i;
        var matrix = new long[(long)classes * classes];
        int validPairs = 0;
        for (int i = 0; i < n; i++)
        {
            matrix[(long)index[predicted[i]] * classes + index[actual[i]]]++;
            validPairs++;
        }
        _ = validPairs;
        _ = matrix;
    }

    static void DenseInt(int[] predicted, int[] actual, int k)
    {
        int n = predicted.Length;
        var matrix = new long[(long)k * k];
        for (int i = 0; i < n; i++)
            matrix[(long)predicted[i] * k + actual[i]]++;
        _ = matrix;
    }

    static void MarginalInt(int[] predicted, int[] actual, int k)
    {
        int n = predicted.Length;
        var seen = new HashSet<int>();
        var labels = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (seen.Add(predicted[i])) labels.Add(predicted[i]);
            if (seen.Add(actual[i])) labels.Add(actual[i]);
        }
        if (seen.Count == 0) return;
        labels.Sort();
        int classes = labels.Count;
        var index = new Dictionary<int, int>(classes);
        for (int i = 0; i < classes; i++) index[labels[i]] = i;
        var predMarg = new long[classes];
        var actMarg = new long[classes];
        var tp = new long[classes];
        for (int i = 0; i < n; i++)
        {
            int pi = index[predicted[i]];
            int ai = index[actual[i]];
            predMarg[pi]++;
            actMarg[ai]++;
            if (predicted[i] == actual[i]) tp[pi]++;
        }
        _ = predMarg;
        _ = actMarg;
        _ = tp;
    }

    static void MarginalString(string[] predicted, string[] actual, int k)
    {
        int n = predicted.Length;
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var labels = new List<string>();
        for (int i = 0; i < n; i++)
        {
            if (seen.Add(predicted[i])) labels.Add(predicted[i]);
            if (seen.Add(actual[i])) labels.Add(actual[i]);
        }
        if (seen.Count == 0) return;
        labels.Sort(StringComparer.Ordinal);
        int classes = labels.Count;
        var index = new Dictionary<string, int>(classes, StringComparer.Ordinal);
        for (int i = 0; i < classes; i++) index[labels[i]] = i;
        var predMarg = new long[classes];
        var actMarg = new long[classes];
        var tp = new long[classes];
        for (int i = 0; i < n; i++)
        {
            int pi = index[predicted[i]];
            int ai = index[actual[i]];
            predMarg[pi]++;
            actMarg[ai]++;
            if (string.Equals(predicted[i], actual[i], StringComparison.Ordinal)) tp[pi]++;
        }
        _ = predMarg;
        _ = actMarg;
        _ = tp;
    }

    static (double BaseMs, double CandMs, int BaseWins, int CandWins) Compare(Action baseline, Action candidate)
    {
        for (int w = 0; w < Warmups; w++)
        {
            baseline();
            candidate();
        }
        var baseTicks = new long[Rounds];
        var candTicks = new long[Rounds];
        int baseWins = 0, candWins = 0;
        for (int r = 0; r < Rounds; r++)
        {
            long b, c;
            if (r % 2 == 0)
            {
                b = Time(baseline);
                c = Time(candidate);
            }
            else
            {
                c = Time(candidate);
                b = Time(baseline);
            }
            baseTicks[r] = b;
            candTicks[r] = c;
            if (b < c) baseWins++;
            else if (c < b) candWins++;
        }
        double toMs = 1000.0 / Stopwatch.Frequency;
        return (Median(baseTicks) * toMs, Median(candTicks) * toMs, baseWins, candWins);
    }

    static long Time(Action action)
    {
        var sw = Stopwatch.StartNew();
        action();
        sw.Stop();
        return sw.ElapsedTicks;
    }

    static void PrintRow(int n, int k, double baseMs, double candMs, int baseWins, int candWins, bool newline = true)
    {
        double ratio = baseMs > 0 ? candMs / baseMs : double.NaN;
        string verdict = ratio < 0.95 ? "CANDIDATE WINS"
            : ratio > 1.05 ? "baseline wins"
            : "noise / comparable";
        string line = $"{n,10} {k,6} {baseMs,10:F3} {candMs,10:F3} {ratio,8:F3}  {verdict,-30} (wins b/c {baseWins}/{candWins})";
        if (newline) Console.WriteLine(line); else Console.Write(line);
    }

    static double Median(long[] values)
    {
        var sorted = (long[])values.Clone();
        Array.Sort(sorted);
        int n = sorted.Length;
        if (n % 2 == 1) return sorted[n / 2];
        return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    static bool IsDebugBuild()
    {
#if DEBUG
        return true;
#else
        return false;
#endif
    }
}
