namespace Nivara.Helpers;

using System.Numerics;

static class RandomGeneration
{
    public static T[] GenerateStandardNormal<T>(int n, int? seed = null) where T : struct, INumber<T>
    {
        var rng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
        var result = new T[n];
        for (int i = 0; i < n; i++)
        {
            double u1, u2;
            u1 = rng.NextDouble();
            u2 = rng.NextDouble();
            var z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            result[i] = T.CreateChecked(z);
        }
        return result;
    }
}
