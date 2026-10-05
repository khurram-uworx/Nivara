using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using NUnit.Framework;

namespace Nivara.Tests;

/// <summary>
/// Fails when a test measures wall-clock or allocations without <c>[Category("Performance")]</c>.
/// </summary>
/// <remarks>
/// <para>
/// <c>Category!=Performance</c> is opt-in. A new timing test that forgets the category does not
/// fail — it runs on every shared-runner push. This gate is what stops that decay. It is
/// construct-driven, not duration-driven: it cannot see a slow test that measures nothing.
/// <c>EveryGatedBlock_CompilesWithoutErrors</c> is that case. The 2-second rule in
/// <c>AGENTS.md</c> and the trx artefact are what cover it.
/// </para>
/// <para>
/// <b>Coverage.</b> <c>tests/Nivara.Tests</c> source only, excluding <c>bin</c> and <c>obj</c>.
/// A test is in scope when its body, or a method it calls in the same type, names
/// <c>Stopwatch</c>, <c>GetAllocatedBytesForCurrentThread</c>, <c>ElapsedMilliseconds</c>,
/// <c>ElapsedTicks</c>, or <c>GetTotalMilliseconds</c>. The effective category is the method's
/// own, plus the declaring type's, plus assembly-level <c>[assembly: Category]</c>, because
/// NUnit inherits fixture and assembly categories. Not covered: categories inherited from a
/// base fixture, calls into a helper in another type, and <c>partial</c> types split so the
/// category and the measurement live in different files. None of those shapes exist in this
/// project today; a negative control drives the shapes that do. The optimized-build guard
/// is not scanned: a categorised test can still forget <c>TimingGuards.RequireOptimizedBuildForTiming()</c>
/// and report a false Debug verdict. That is a Debug-only miss, not a shared-runner cost.
/// </para>
/// </remarks>
[TestFixture]
public class SlowTestCategorisationTests
{
    static readonly HashSet<string> MeasurementNames = new(StringComparer.Ordinal)
    {
        "Stopwatch",
        "GetAllocatedBytesForCurrentThread",
        "ElapsedMilliseconds",
        "ElapsedTicks",
        "GetTotalMilliseconds",
    };

    internal readonly record struct Violation(string Location, string Method);

    /// <summary>
    /// Kept separate from the repository assertion so the negative controls can drive it with
    /// a violation that genuinely exists. A gate that only ever scans the real tree can pass
    /// because the parser silently found nothing.
    /// </summary>
    internal static IReadOnlyList<Violation> FindViolations(IEnumerable<(string Path, string Source)> files)
    {
        ArgumentNullException.ThrowIfNull(files);

        var violations = new List<Violation>();
        foreach (var (path, source) in files)
            violations.AddRange(FindViolations(path, source));

        return violations
            .OrderBy(v => v.Location, StringComparer.Ordinal)
            .ThenBy(v => v.Method, StringComparer.Ordinal)
            .ToArray();
    }

    internal static IReadOnlyList<Violation> FindViolations(string path, string source)
    {
        var root = CSharpSyntaxTree.ParseText(source).GetCompilationUnitRoot();
        var assemblyCategories = root.AttributeLists
            .Where(list => list.Target?.Identifier.IsKind(SyntaxKind.AssemblyKeyword) == true)
            .SelectMany(ReadCategories)
            .ToHashSet(StringComparer.Ordinal);

        var violations = new List<Violation>();
        foreach (var type in root.DescendantNodes().OfType<TypeDeclarationSyntax>())
            violations.AddRange(FindViolations(path, type, assemblyCategories));

        return violations;
    }

    [Test]
    public void TheRepository_HasNoUncategorisedMeasurement()
    {
        var files = TestSources().Select(path => (path, File.ReadAllText(path))).ToArray();
        var violations = FindViolations(files);

        Assert.That(files, Is.Not.Empty, "the scan found no test sources, so a green run means nothing");
        Assert.That(violations, Is.Empty,
            "these tests measure wall-clock or allocations without [Category(\"Performance\")]. "
            + "NUnit inherits the category from the fixture and the assembly, so either form counts. "
            + Environment.NewLine
            + string.Join(Environment.NewLine, violations.Select(v => $"{v.Location} {v.Method}")));
    }

    [Test]
    public void NegativeControl_AMeasuringTestWithoutTheCategory_IsReported()
    {
        const string source = """
            using NUnit.Framework;
            using System.Diagnostics;
            class Probe {
                [Test]
                public void Slow() { var sw = Stopwatch.StartNew(); }
            }
            """;

        var violations = FindViolations("probe.cs", source);

        Assert.That(violations, Has.Count.EqualTo(1));
        Assert.That(violations[0].Method, Is.EqualTo("Slow"));
    }

    [Test]
    public void NegativeControl_MethodCategory_SuppressesTheReport()
    {
        const string source = """
            using NUnit.Framework;
            using System.Diagnostics;
            class Probe {
                [Test]
                [Category("Performance")]
                public void Slow() { var sw = Stopwatch.StartNew(); }
            }
            """;

        Assert.That(FindViolations("probe.cs", source), Is.Empty);
    }

    [Test]
    public void NegativeControl_FixtureCategory_IsInherited()
    {
        const string source = """
            using NUnit.Framework;
            class Probe {
                [Category("Performance")]
                class Inner {
                    [Test]
                    public void Slow() { _ = GC.GetAllocatedBytesForCurrentThread(); }
                }
            }
            """;

        // The category is on the type that declares the test, which is what NUnit inherits.
        var declaring = """
            using NUnit.Framework;
            [Category("Performance")]
            class Probe {
                [Test]
                public void Slow() { _ = GC.GetAllocatedBytesForCurrentThread(); }
            }
            """;

        Assert.That(FindViolations("probe.cs", declaring), Is.Empty,
            "a fixture-level category excludes every test in the fixture; a method-only scan would false-positive it");
        Assert.That(FindViolations("probe.cs", source).Select(v => v.Method), Is.Empty);
    }

    [Test]
    public void NegativeControl_AssemblyCategory_IsInherited()
    {
        const string source = """
            using NUnit.Framework;
            [assembly: Category("Performance")]
            class Probe {
                [Test]
                public void Slow() { var ms = 1; _ = ms; }
                [Test]
                public void Also() { var sw = new System.Diagnostics.Stopwatch(); }
            }
            """;

        Assert.That(FindViolations("probe.cs", source), Is.Empty);
    }

    [Test]
    public void NegativeControl_SameTypeHelper_TaintsTheCallingTestOnly()
    {
        const string source = """
            using NUnit.Framework;
            class Probe {
                [Test]
                public void Measures() { Touch(); }
                [Test]
                public void DoesNot() { }
                static void Touch() { _ = GC.GetAllocatedBytesForCurrentThread(); }
            }
            """;

        var violations = FindViolations("probe.cs", source);

        Assert.That(violations.Select(v => v.Method), Is.EqualTo(new[] { "Measures" }));
    }

    [Test]
    public void NegativeControl_ACommentIsNotAMeasurement()
    {
        const string source = """
            using NUnit.Framework;
            class Probe {
                [Test]
                public void MentionsIt()
                {
                    // Stopwatch.ElapsedMilliseconds would be a measurement. This is not.
                }
            }
            """;

        Assert.That(FindViolations("probe.cs", source), Is.Empty,
            "a comment naming the construct must not force the category, or the gate can be silenced by deleting the call and keeping the word");
    }

    static IEnumerable<Violation> FindViolations(
        string path, TypeDeclarationSyntax type, HashSet<string> assemblyCategories)
    {
        var methods = type.Members.OfType<MethodDeclarationSyntax>().ToArray();
        if (methods.Length == 0)
            yield break;

        var typeCategories = type.AttributeLists.SelectMany(ReadCategories).ToHashSet(StringComparer.Ordinal);
        var byName = methods.GroupBy(m => m.Identifier.ValueText, StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => g.ToArray(), StringComparer.Ordinal);

        var tainted = methods.Where(ContainsMeasurement).ToHashSet();
        bool grew;
        do
        {
            grew = false;
            foreach (var method in methods)
            {
                if (tainted.Contains(method))
                    continue;
                if (Calls(method, byName).Any(tainted.Contains))
                {
                    tainted.Add(method);
                    grew = true;
                }
            }
        }
        while (grew);

        foreach (var method in methods.Where(IsTest))
        {
            if (!tainted.Contains(method))
                continue;

            var categories = ReadCategories(method.AttributeLists)
                .Concat(typeCategories)
                .Concat(assemblyCategories);
            if (categories.Contains("Performance", StringComparer.Ordinal))
                continue;

            var line = method.Identifier.GetLocation().GetLineSpan().StartLinePosition.Line + 1;
            yield return new Violation($"{path}:{line}", method.Identifier.ValueText);
        }
    }

    static bool ContainsMeasurement(MethodDeclarationSyntax method) =>
        method.DescendantNodes().OfType<IdentifierNameSyntax>()
            .Any(name => MeasurementNames.Contains(name.Identifier.ValueText));

    static IEnumerable<MethodDeclarationSyntax> Calls(
        MethodDeclarationSyntax method, Dictionary<string, MethodDeclarationSyntax[]> byName)
    {
        foreach (var invocation in method.DescendantNodes().OfType<InvocationExpressionSyntax>())
        {
            var name = invocation.Expression switch
            {
                IdentifierNameSyntax id => id.Identifier.ValueText,
                MemberAccessExpressionSyntax access => access.Name.Identifier.ValueText,
                _ => null,
            };
            if (name is not null && byName.TryGetValue(name, out var targets))
            {
                foreach (var target in targets)
                    yield return target;
            }
        }
    }

    static bool IsTest(MethodDeclarationSyntax method) =>
        method.AttributeLists.SelectMany(list => list.Attributes).Any(IsTestAttribute);

    static bool IsTestAttribute(AttributeSyntax attribute)
    {
        var name = AttributeName(attribute);
        return name is "Test" or "TestAttribute" or "TestCase" or "TestCaseAttribute";
    }

    static IEnumerable<string> ReadCategories(IEnumerable<AttributeListSyntax> lists) =>
        lists.SelectMany(ReadCategories);

    static IEnumerable<string> ReadCategories(AttributeListSyntax list)
    {
        foreach (var attribute in list.Attributes)
        {
            var name = AttributeName(attribute);
            if (name is not "Category" and not "CategoryAttribute")
                continue;
            if (attribute.ArgumentList?.Arguments.Count == 1
                && attribute.ArgumentList.Arguments[0].Expression is LiteralExpressionSyntax literal
                && literal.IsKind(SyntaxKind.StringLiteralExpression)
                && literal.Token.ValueText is string value)
                yield return value;
        }
    }

    static string? AttributeName(AttributeSyntax attribute) =>
        attribute.Name switch
        {
            IdentifierNameSyntax id => id.Identifier.ValueText,
            QualifiedNameSyntax qualified => qualified.Right.Identifier.ValueText,
            AliasQualifiedNameSyntax alias => alias.Name.Identifier.ValueText,
            _ => null,
        };

    static IEnumerable<string> TestSources()
    {
        var root = RepoRoot();
        var tests = Path.Combine(root, "tests", "Nivara.Tests");
        return Directory.EnumerateFiles(tests, "*.cs", SearchOption.AllDirectories)
            .Where(path => !path.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
                .Any(part => part is "bin" or "obj"));
    }

    static string RepoRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "Nivara.slnx")))
                return dir.FullName;
        }

        throw new InvalidOperationException(
            $"Nivara.slnx not found above {AppContext.BaseDirectory}; the slow-test gate cannot locate the repository root");
    }
}
