namespace Nivara.Helpers;

using System.Numerics;

static class RandomGeneration
{
    /// <summary>
    /// Returns a new seeded <see cref="Random"/>, or <see cref="Random.Shared"/> when
    /// <paramref name="seed"/> is null.
    /// </summary>
    /// <param name="seed">Optional RNG seed; null selects the process-global shared RNG</param>
    public static Random Create(int? seed) => seed.HasValue ? new Random(seed.Value) : Random.Shared;

    public static T[] GenerateStandardNormal<T>(int n, int? seed = null) where T : struct, INumber<T>
    {
        var rng = Create(seed);
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
