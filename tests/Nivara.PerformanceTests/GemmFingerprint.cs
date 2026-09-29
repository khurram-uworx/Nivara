using System.Text.Encodings.Web;
using System.Text.Json;

namespace Nivara.PerformanceTests;

/// <summary>
/// A bit-exact record of what every (kernel, shape) cell of the <c>--gemm</c> gate produced: an
/// FNV-1a 64 hash over the f32 bit patterns of the read-back, plus the element count, keyed
/// <c>variant|shape</c>. The gate compares every cell against the committed baseline rather than
/// against a tolerance, because the contract these kernels promise is exactness — the same f32
/// values added to the same accumulator in the same ascending-K order — and a tolerance converts
/// a structural defect into a judgement call (docs/ACCELERATION.md lesson 19,
/// GUIDELINES.md "Assert Exactness When Exactness Is Available").
/// </summary>
/// <remarks>
/// The baseline is keyed by toolchain as well as by cell, because the same kernel source can
/// legitimately produce different bits on a different driver: a change in how the OpenCL compiler
/// contracts <c>a * b + acc</c> into an FMA alters the rounding without any change to the kernel.
/// A toolchain difference is therefore reported as a stated precondition with a deliberate
/// re-record, not as 171 numeric failures that read like a kernel regression.
///
/// Hash equality is used as bit-identity up to a 64-bit collision. The gate is not a security
/// boundary; a collision would need a change in kernel output to coincide with an existing hash,
/// and the element count is compared alongside so a shape change cannot hide behind one.
/// </remarks>
internal static class GemmFingerprint
{
    public const string FileName = "gemm-f32-baseline.json";

    /// <summary>Regenerates the baseline. Never implied by a normal gate run.</summary>
    public const string WriteFlag = "--write-gemm-baseline";

    static readonly JsonSerializerOptions s_jsonOptions = new(JsonSerializerDefaults.Web)
    {
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        WriteIndented = true,
    };

    /// <summary>One cell's bit-exact result: how many floats, and their combined hash.</summary>
    public sealed record Cell(int Count, string Hash);

    /// <summary>The committed baseline: the toolchain it was recorded on, and every cell.</summary>
    /// <param name="Device">Accelerator name the cells were recorded on.</param>
    /// <param name="Toolchain">What produced the f32 bits: OpenCL version plus the driver's
    /// <c>CL_DRIVER_VERSION</c>, as composed by the caller. This is the only field that changes
    /// when the toolchain does without the kernel changing.</param>
    /// <param name="Cells">Cell key to cell result, sorted so the file diffs cleanly.</param>
    public sealed record Baseline(string Device, string Toolchain, SortedDictionary<string, Cell> Cells);

    /// <summary>
    /// What comparing this run's cells against the baseline found. Every failure mode is a
    /// separate count so a structural gap — no baseline, a toolchain that was not the one the
    /// baseline was recorded on — can never be reported as a numeric pass, and a numeric
    /// mismatch can never be reported as a skip.
    /// </summary>
    /// <param name="Compared">Cells measured this run that had a baseline row to compare against.</param>
    /// <param name="Matched">Those cells whose bit patterns were identical.</param>
    /// <param name="Mismatched">Cells whose bits or element count differed, with both readings.</param>
    /// <param name="Missing">Cells measured this run with no baseline row at all.</param>
    /// <param name="NotExercised">Baseline rows this run did not measure (e.g. a #440 geometry
    /// the device cannot host), named so a shrunken run cannot read like a full one.</param>
    /// <param name="ToolchainDifference">Non-null when the live device or driver differs from the
    /// one the baseline was recorded on; the comparisons are then not meaningful.</param>
    /// <param name="BaselineAbsent">The baseline file was not found where it is looked for.</param>
    public sealed record Verdict(
        int Compared,
        int Matched,
        IReadOnlyList<string> Mismatched,
        IReadOnlyList<string> Missing,
        IReadOnlyList<string> NotExercised,
        string? ToolchainDifference,
        bool BaselineAbsent)
    {
        /// <summary>Distinct failure reasons this verdict contributes to the gate's exit code.</summary>
        public int Failures =>
            Mismatched.Count + Missing.Count + (BaselineAbsent || ToolchainDifference is not null ? 1 : 0);
    }

    public static string Key(string variant, string shape) => $"{variant}|{shape}";

    public static Cell Of(float[] values) => new(values.Length, Hash(values));

    static string Hash(float[] values)
    {
        const ulong basis = 14695981039346656037;
        const ulong prime = 1099511628211;

        ulong hash = basis;
        foreach (float value in values)
        {
            // Bit patterns, not values: -0.0f and 0.0f compare equal but are different results,
            // and a NaN payload is part of the output exactly as much as a sign is.
            int bits = BitConverter.SingleToInt32Bits(value);
            for (int b = 0; b < sizeof(int); b++)
            {
                hash ^= (byte)(bits >> (b * 8));
                hash *= prime;
            }
        }

        return $"0x{hash:x16}";
    }

    /// <summary>
    /// Finds the committed baseline: the first of <see cref="SearchedPaths"/> that exists. Both
    /// are real layouts this harness runs from — the repo root (where the README's
    /// <c>dotnet run --project tests/Nivara.PerformanceTests</c> puts the working directory) and
    /// the project directory (reached from the build output).
    /// </summary>
    public static string? Locate()
    {
        foreach (string candidate in SearchedPaths)
            if (File.Exists(candidate))
                return candidate;
        return null;
    }

    /// <summary>Where the baseline is looked for, in order, and for the failure message when it is
    /// not there. <see cref="Locate"/> searches exactly this list, so the two cannot disagree.</summary>
    public static string[] SearchedPaths =>
    [
        Path.Combine(Environment.CurrentDirectory, "tests", "Nivara.PerformanceTests", FileName),
        Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", FileName)),
    ];

    public static void Write(string path, string device, string toolchain, SortedDictionary<string, Cell> cells)
        => File.WriteAllText(path, JsonSerializer.Serialize(new Baseline(device, toolchain, cells), s_jsonOptions));

    public static Baseline Read(string path)
        => JsonSerializer.Deserialize<Baseline>(File.ReadAllText(path), s_jsonOptions)
           ?? throw new InvalidDataException($"empty fingerprint baseline: {path}");

    /// <summary>
    /// Compares this run's cells against the committed baseline. A null
    /// <paramref name="baseline"/> is a structural gap, not a pass: there is nothing to be
    /// exact against, so the verdict carries <see cref="Verdict.BaselineAbsent"/> and the gate
    /// fails on it rather than reporting an unverified run as a clean one.
    /// </summary>
    public static Verdict Compare(
        Baseline? baseline,
        SortedDictionary<string, Cell> measured,
        string device,
        string toolchain)
    {
        if (baseline is null)
            return new Verdict(0, 0, [], [], [], null, BaselineAbsent: true);

        var mismatched = new List<string>();
        var missing = new List<string>();
        int matched = 0;

        foreach (var (key, cell) in measured)
        {
            if (!baseline.Cells.TryGetValue(key, out var expected))
            {
                missing.Add(key);
                continue;
            }

            if (expected.Count == cell.Count && expected.Hash == cell.Hash)
            {
                matched++;
                continue;
            }

            mismatched.Add($"{key} (baseline {expected.Count} {expected.Hash}, now {cell.Count} {cell.Hash})");
        }

        var notExercised = baseline.Cells.Keys.Where(key => !measured.ContainsKey(key)).ToList();

        string? toolchainDifference = (baseline.Device == device && baseline.Toolchain == toolchain)
            ? null
            : $"baseline recorded on {baseline.Device} / {baseline.Toolchain}, running on {device} / {toolchain}";

        return new Verdict(matched + mismatched.Count, matched, mismatched, missing, notExercised, toolchainDifference, BaselineAbsent: false);
    }
}
