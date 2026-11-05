using System;
using System.Collections.Generic;
using System.Linq;

namespace NivaraChat.Helpers;

/// <summary>
/// Pure, dependency-free evaluation metrics for retrieval. Kept separate from
/// <see cref="RagEvalMode"/> so they can be unit-tested without MiniLM/Ollama.
/// </summary>
public static class RagEvalMetrics
{
    public static double PrecisionAtK(IReadOnlyList<string> retrieved, IReadOnlySet<string> relevant, int k)
    {
        if (k <= 0 || retrieved.Count == 0 || relevant.Count == 0)
            return 0d;

        var n = Math.Min(k, retrieved.Count);
        var hits = 0;
        for (var i = 0; i < n; i++)
            if (relevant.Contains(retrieved[i]))
                hits++;

        return (double)hits / k;
    }

    public static double RecallAtK(IReadOnlyList<string> retrieved, IReadOnlySet<string> relevant, int k)
    {
        if (k <= 0 || retrieved.Count == 0 || relevant.Count == 0)
            return 0d;

        var n = Math.Min(k, retrieved.Count);
        var hits = 0;
        for (var i = 0; i < n; i++)
            if (relevant.Contains(retrieved[i]))
                hits++;

        return (double)hits / relevant.Count;
    }

    public static double Mrr(IReadOnlyList<string> retrieved, IReadOnlySet<string> relevant)
    {
        if (retrieved.Count == 0 || relevant.Count == 0)
            return 0d;

        for (var i = 0; i < retrieved.Count; i++)
        {
            if (relevant.Contains(retrieved[i]))
                return 1d / (i + 1);
        }

        return 0d;
    }

    public static double NdcgAtK(IReadOnlyList<string> retrieved, IReadOnlyDictionary<string, int> grades, int k)
    {
        if (k <= 0 || retrieved.Count == 0 || grades.Count == 0)
            return 0d;

        var n = Math.Min(k, retrieved.Count);
        var dcg = 0d;
        for (var i = 0; i < n; i++)
        {
            if (!grades.TryGetValue(retrieved[i], out var grade) || grade <= 0)
                continue;

            var gain = Math.Pow(2, grade) - 1;
            var discount = Math.Log(i + 2, 2); // log2(i+2)
            dcg += gain / discount;
        }

        // Ideal DCG: sort all known grades descending.
        var ideal = grades.Values.Where(g => g > 0).OrderByDescending(g => g).Take(k).ToList();
        var idcg = 0d;
        for (var i = 0; i < ideal.Count; i++)
        {
            var gain = Math.Pow(2, ideal[i]) - 1;
            var discount = Math.Log(i + 2, 2);
            idcg += gain / discount;
        }

        return idcg <= 0 ? 0d : dcg / idcg;
    }

    public static (int Wins, int Losses, int Ties) WinLossTie(
        IReadOnlyList<double> left, IReadOnlyList<double> right)
    {
        var wins = 0;
        var losses = 0;
        var ties = 0;
        var n = Math.Min(left.Count, right.Count);
        for (var i = 0; i < n; i++)
        {
            if (Math.Abs(left[i] - right[i]) < 1e-12)
                ties++;
            else if (left[i] > right[i])
                wins++;
            else
                losses++;
        }

        return (wins, losses, ties);
    }

    /// <summary>Two-sided exact binomial sign-test p-value over non-tie pairs.</summary>
    public static double SignTestPValue(int wins, int losses)
    {
        var n = wins + losses;
        if (n <= 0)
            return 1d;

        var k = Math.Min(wins, losses);
        var sum = 0d;
        var pow = Math.Pow(0.5, n);
        for (var i = 0; i <= k; i++)
            sum += Binomial(n, i);

        var p = 2d * sum * pow;
        return Math.Min(1d, p);
    }

    public static double CohensD(IReadOnlyList<double> left, IReadOnlyList<double> right)
    {
        var n = Math.Min(left.Count, right.Count);
        if (n < 2)
            return 0d;

        var diffs = new double[n];
        for (var i = 0; i < n; i++)
            diffs[i] = left[i] - right[i];

        var mean = diffs.Average();
        var variance = diffs.Sum(d => (d - mean) * (d - mean)) / (n - 1);
        var sd = Math.Sqrt(variance);
        return sd <= 0 ? 0d : mean / sd;
    }

    public static double Mean(IReadOnlyList<double> values) =>
        values.Count == 0 ? 0d : values.Average();

    public static double StdDev(IReadOnlyList<double> values)
    {
        if (values.Count < 2)
            return 0d;

        var mean = values.Average();
        var variance = values.Sum(v => (v - mean) * (v - mean)) / (values.Count - 1);
        return Math.Sqrt(variance);
    }

    static double Binomial(int n, int k)
    {
        if (k < 0 || k > n)
            return 0d;

        // log-space for stability
        double result = 1d;
        for (var i = 1; i <= k; i++)
            result = result * (n - k + i) / i;

        return result;
    }
}
