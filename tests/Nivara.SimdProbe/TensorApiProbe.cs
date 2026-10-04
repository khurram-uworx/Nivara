using System.Numerics.Tensors;
using System.Reflection;

namespace Nivara.SimdProbe;

/// <summary>
/// Probe: what is the correct way to extract a single row from a rank-2 <c>Tensor&lt;T&gt;</c>,
/// and which superficially-plausible forms are traps?
///
/// <para><b>Question.</b> <c>EXAMPLES.md:222</c> and <c>EXAMPLES.md:444</c> (issue #528) scored a
/// frame's embeddings row-by-row via <c>tensor.AsSpan().Slice(i * dims, dims)</c>, which does not
/// compile. The issue's proposed fix — "the property is <c>.Span</c>" — is also wrong: <c>Tensor&lt;T&gt;</c>
/// has no <c>Span</c> property at all. An agent fixing this has to answer two questions the docs
/// do not answer, and both have a failure mode that is invisible to the doc-snippet gate:
/// which member to call, and how many indices to pass it.</para>
///
/// <para><b>Why this is a probe and not just a doc edit.</b> The snippet gate compiles; it does not
/// run. So a form that compiles and then throws on first use passes the gate. The worst candidate
/// here, <c>GetSpan([i], dims)</c>, compiles cleanly and throws
/// <see cref="ArgumentOutOfRangeException"/> because <c>GetSpan</c> wants one index per dimension.
/// A gate that only asks "does it compile" would certify exactly the wrong answer. This probe
/// therefore evaluates every candidate on both axes and reports them separately.</para>
///
/// <para><b>Why the non-compiling forms are checked by reflection.</b> A form that fails to compile
/// cannot live in a file that must itself compile, so "compiles" cannot be observed directly for
/// those. The probe reflects over the real assembly instead — asserting <c>Tensor&lt;T&gt;</c> has
/// no <c>Span</c>/<c>Memory</c>/<c>AsSpan</c> member is precisely the check that would have made
/// the .Span substitution a hard failure at the point it was written, rather than CS1061 later.</para>
///
/// <para><b>Also asserts.</b> That the recommended form reproduces the flat row-major slice exactly
/// (not within a tolerance), and that the scored ranking is the one <c>EXAMPLES.md</c> claims. Those
/// are the doc's actual load-bearing claims, and a wrong row offset would still compile.</para>
///
/// <para>Self-contained: references only <c>System.Numerics.Tensors</c>, not Nivara. The subject is
/// the BCL contract the docs are written against, so coupling it to the library under test would
/// make it fail for the wrong reason.</para>
/// </summary>
internal static class TensorApiProbe
{
    /// <summary>The EXAMPLES.md Act 4 dataset: 3 documents x 4 embedding dimensions, row-major.</summary>
    static readonly float[] RowMajor =
    [
        0.9f, 0.2f, 0.5f, 0.4f,   // doc-101
        0.1f, 0.9f, 0.2f, 0.7f,   // doc-102
        0.7f, 0.1f, 0.8f, 0.2f,   // doc-103
    ];

    static readonly string[] Labels = ["doc-101", "doc-102", "doc-103"];
    static readonly float[] Query = [0.8f, 0.1f, 0.6f, 0.3f];

    const int Rows = 3;
    const int Dims = 4;

    public static int Run()
    {
        Console.WriteLine("=== Tensor<T> row-span extraction (issue #528) ===");
        Console.WriteLine($"Runtime: {Environment.Version}   Tensors: {typeof(Tensor).Assembly.GetName().Version}");
        Console.WriteLine();

        int failures = 0;

        failures += ReportApiSurface();
        failures += ReportOverloads();
        failures += CheckCandidates();
        failures += CheckScoring();

        Console.WriteLine();
        Console.WriteLine(failures == 0
            ? "All tensor-api checks PASSED."
            : $"{failures} tensor-api check(s) FAILED.");

        Console.WriteLine();
        Console.WriteLine(Notes());

        return failures;
    }

    /// <summary>
    /// The compile-time contract, checked reflectively so the forms that do not compile can still be
    /// named. This is the guard against the docs regressing to a member that was never there.
    /// </summary>
    static int ReportApiSurface()
    {
        Console.WriteLine("--- Tensor<float> public surface ---");
        var type = typeof(Tensor<float>);

        foreach (var property in type.GetProperties(BindingFlags.Public | BindingFlags.Instance)
                     .OrderBy(p => p.Name, StringComparer.Ordinal))
            Console.WriteLine($"  property  {TypeName(property.PropertyType)} {property.Name}");

        Console.WriteLine();

        foreach (var method in type.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
                     .Where(m => !m.IsSpecialName)
                     .OrderBy(m => m.Name, StringComparer.Ordinal))
            Console.WriteLine($"  method    {Signature(method)}");

        Console.WriteLine();

        Console.WriteLine("--- members the docs must not name ---");

        // EXAMPLES.md:222 and :444 used AsSpan(); issue #528 then prescribed .Span. Neither exists.
        // If a future Tensors release adds one of these, this flips to FAIL and the docs get a
        // second opinion rather than a silent second bug.
        int failures = 0;
        failures += Absent(type, "property", "Span");
        failures += Absent(type, "property", "Memory");
        failures += Absent(type, "method", "AsSpan");
        failures += Absent(type, "method", "AsMemory");

        Console.WriteLine();
        return failures;
    }

    static int Absent(Type type, string kind, string name)
    {
        bool found = kind == "property"
            ? type.GetProperty(name, BindingFlags.Public | BindingFlags.Instance) is not null
            : type.GetMethod(name, BindingFlags.Public | BindingFlags.Instance) is not null;

        Console.WriteLine($"  {(found ? "FAIL" : "ok  ")}  Tensor<float> has no public {kind} '{name}'");
        return found ? 1 : 0;
    }

    /// <summary>
    /// <c>GetSpan</c> is overloaded on <c>ReadOnlySpan&lt;nint&gt;</c> and <c>ReadOnlySpan&lt;NIndex&gt;</c>,
    /// and <c>int</c> converts implicitly to both. Print the real signatures so a reader picks the
    /// right one instead of discovering the ambiguity from CS0121.
    /// </summary>
    static int ReportOverloads()
    {
        Console.WriteLine("--- GetSpan / TryGetSpan overloads ---");

        var methods = typeof(Tensor<float>)
            .GetMethods(BindingFlags.Public | BindingFlags.Instance)
            .Where(m => m.Name is "GetSpan" or "TryGetSpan")
            .OrderBy(m => m.Name, StringComparer.Ordinal)
            .ThenBy(m => Signature(m), StringComparer.Ordinal);

        foreach (var method in methods)
            Console.WriteLine($"  {Signature(method)}");

        Console.WriteLine();
        return 0;
    }

    /// <summary>
    /// Every candidate form, on both axes: what the compiler accepts, and whether it actually returns
    /// the row. Only the forms that compile can be present in this file.
    /// </summary>
    static int CheckCandidates()
    {
        Console.WriteLine("--- candidate row-extraction forms ---");
        Console.WriteLine("  (a form that does not compile cannot live in this file, so its compile");
        Console.WriteLine("   status is asserted reflectively above and not reproduced here)");
        Console.WriteLine();

        var tensor = Tensor.Create(RowMajor, [Rows, Dims]);
        // An int, deliberately: GetSpan's second parameter is `int` length, while its first is a
        // span of indices. The two have different element types and mixing them up is a CS1503.
        int dims = Dims;
        int failures = 0;

        var candidates = new (string Name, string Source, RowForm Form)[]
        {
            ("GetSpan([i, 0], dims)", "docVectors.GetSpan([i, 0], dims)", i => tensor.GetSpan([i, 0], dims)),
            ("GetSpan(new nint[]{i,0}, dims)", "docVectors.GetSpan(new nint[] { i, 0 }, dims)",
                i => tensor.GetSpan(new nint[] { i, 0 }, dims)),
            ("GetSpan([i], dims)  <-- TRAP", "docVectors.GetSpan([i], dims)", i => tensor.GetSpan([i], dims)),
        };

        foreach (var (name, source, form) in candidates)
        {
            Console.WriteLine($"  {name}");
            Console.WriteLine($"    compiles: yes   (documented form: {source})");

            bool ok = true;
            try
            {
                for (int i = 0; i < Rows; i++)
                    ok &= Matches(form(i), RowMajor, i, Dims);

                if (ok)
                    Console.WriteLine("    runs:     yes — all rows equal the flat row-major slice, exactly");
                else
                {
                    Console.WriteLine("    runs:     NO — row contents do not match the flat slice");
                    failures++;
                }
            }
            catch (Exception ex)
            {
                // The trap. It compiles, so the doc-snippet gate certifies it, and it fails on the
                // very first row instead. Reported as a trap, not a probe failure — the probe is
                // behaving correctly by observing it.
                Console.WriteLine($"    runs:     NO — throws {ex.GetType().Name} on row 0");
                Console.WriteLine("              TRAP: compiles, throws at run time. The gate cannot catch this.");
            }

            Console.WriteLine();
        }

        return failures;
    }

    /// <summary>
    /// The two EXAMPLES.md snippets end to end, on the recommended form. Asserts the ranking the
    /// document claims, which is discrete and so has no tolerance to argue about.
    /// </summary>
    static int CheckScoring()
    {
        Console.WriteLine("--- EXAMPLES.md scoring, using GetSpan([i, 0], dims) ---");

        var docVectors = Tensor.Create(RowMajor, [Rows, Dims]);
        var scores = new float[Rows];
        int dims = Dims;

        for (int i = 0; i < Rows; i++)
            scores[i] = TensorPrimitives.CosineSimilarity(docVectors.GetSpan([i, 0], dims), Query);

        Console.Write("  python   v.q / (||v|| ||q||):");
        foreach (var score in ReferenceScores())
            Console.Write($"  {score:F6}");
        Console.WriteLine();

        Console.Write("  GetSpan  TensorPrimitives.CosineSimilarity:");
        foreach (var score in scores)
            Console.Write($"  {score:F6}");
        Console.WriteLine();

        var actual = scores.Select((score, i) => (Label: Labels[i], Score: score))
            .OrderByDescending(x => x.Score).Take(2).Select(x => x.Label).ToArray();

        bool rankingOk = actual.SequenceEqual(["doc-101", "doc-103"]);
        Console.WriteLine($"  ranking  {string.Join(", ", actual)}   EXAMPLES.md claims: doc-101, doc-103   " +
                          $"{(rankingOk ? "PASS" : "FAIL")}");

        // Cosine similarity in the BCL against numpy's reference is a last-bit question, and a
        // tolerance band would hide a genuine row-offset bug behind a judgement call. The ranking
        // is the doc's actual claim and is discrete, so that is what is asserted. Agreement with the
        // numpy reference is printed above for a human to read.
        return rankingOk ? 0 : 1;
    }

    /// <summary>
    /// numpy <c>v @ q / (norm(v) * norm(q))</c> on the Act 4 data, for side-by-side comparison.
    /// </summary>
    static float[] ReferenceScores()
    {
        float queryNorm = Norm(Query);
        var result = new float[Rows];

        for (int i = 0; i < Rows; i++)
        {
            var row = RowMajor.AsSpan(i * Dims, Dims);
            float dot = 0f;
            for (int j = 0; j < Dims; j++) dot += row[j] * Query[j];
            result[i] = dot / (Norm(row) * queryNorm);
        }

        return result;
    }

    static float Norm(ReadOnlySpan<float> v)
    {
        float sum = 0f;
        foreach (var value in v) sum += value * value;
        return MathF.Sqrt(sum);
    }

    /// <summary>Exact comparison. A row offset or stride error is structural, not a precision question.</summary>
    static bool Matches(ReadOnlySpan<float> actual, float[] flat, int row, int dims) =>
        actual.SequenceEqual(flat.AsSpan(row * dims, dims));

    static string Signature(MethodInfo method) =>
        $"{TypeName(method.ReturnType)} {method.Name}({string.Join(", ", method.GetParameters().Select(Describe))})";

    static string Describe(ParameterInfo parameter) =>
        // TypeName already renders the byref '&', so this must not add another.
        TypeName(parameter.ParameterType) + " " + parameter.Name;

    /// <summary>
    /// Full type names with generic arguments expanded. Plain <c>Type.Name</c> renders both
    /// <c>ReadOnlySpan&lt;nint&gt;</c> and <c>ReadOnlySpan&lt;NIndex&gt;</c> as <c>ReadOnlySpan`1</c>,
    /// which makes the two <c>GetSpan</c> overloads indistinguishable — and telling them apart is the
    /// entire reason for printing them. <c>nint</c> is <see cref="IntPtr"/> at run time, so it has to
    /// be mapped back by hand.
    /// </summary>
    static string TypeName(Type type)
    {
        if (type == typeof(nint)) return "nint";

        // An `out Span<T>` parameter reflects as Span<T>&, and ByRef is not itself generic — so the
        // element type has to come off first or it prints as Span`1&.
        bool byRef = type.IsByRef;
        if (byRef) type = type.GetElementType()!;

        // IsGenericType is true for a type nested inside a generic type even when the name carries
        // no backtick — Tensor<float>.Enumerator is the case here — so the backtick must be probed,
        // not assumed.
        string name = type.IsGenericType && type.Name.IndexOf('`') is var tick && tick > 0
            ? $"{type.Name[..tick]}<{string.Join(", ", type.GetGenericArguments().Select(TypeName))}>"
            : type.Name;

        return byRef ? name + "&" : name;
    }

    /// <summary>
    /// The findings, kept in the probe's own output so a session that runs the mode and moves on
    /// still carries them. #524-#532 are eight more "the doc names the wrong API" fixes.
    /// </summary>
    static string Notes() =>
        "  Findings (the whole point of this mode):\n" +
        "    1. Tensor<T> has NO Span property, no Memory, no AsSpan(). Issue #528's\n" +
        "       prescription (\"the property is .Span\") is CS1061, not a fix. The CS1929 the\n" +
        "       gate reports is the compiler having resolved AsSpan to MemoryExtensions.AsSpan(string?).\n" +
        "    2. GetSpan wants one index PER DIMENSION. GetSpan([i], dims) compiles and throws\n" +
        "       ArgumentOutOfRangeException on the first row. A compile-only gate certifies it.\n" +
        "       For a [rows, cols] tensor: GetSpan([i, 0], cols).\n" +
        "    3. GetSpan is overloaded on ReadOnlySpan<nint> and ReadOnlySpan<NIndex>, and int\n" +
        "       converts to both. A collection expression ([i, 0]) resolves; [] and Slice([i], ..)\n" +
        "       do not (CS0121 / CS9174).\n" +
        "    4. GetSpan's LENGTH parameter is `int`, not `nint`, while Tensor.Lengths is ReadOnlySpan<nint>.\n" +
        "       `nint dims = tensor.Lengths[1]` is the natural spelling and is CS1503. Cast to int.\n" +
        "    5. A contiguous Nivara column already IS its Tensor<T> (docs/TENSORS.md); the row\n" +
        "       loop is only needed because ToTensor produces a rank-2 tensor.\n" +
        "    6. Re-run: dotnet run -c Release --project tests/Nivara.SimdProbe -- tensor-api";
}

/// <summary>A row extractor. A lambda cannot return <c>Span&lt;T&gt;</c> directly — <c>Span&lt;T&gt;</c> is a ref struct.</summary>
delegate ReadOnlySpan<float> RowForm(int i);
