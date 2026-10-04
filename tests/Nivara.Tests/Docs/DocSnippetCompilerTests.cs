using NUnit.Framework;

namespace Nivara.Tests.Docs;

[TestFixture]
public class DocSnippetCompilerTests
{
    const string FixturePath = "tests/Nivara.Tests/Docs/Fixtures/Broken.md";

    static DocBlock Block(string code, DocWrapMode mode = DocWrapMode.TopLevel, int startLine = 0, string locals = "")
        => new("docs/INLINE.md", startLine, code, null, locals, mode, null, null);

    static DocSnippetErrors Compile(DocBlock block) => new(DocSnippetCompiler.Compile(block));

    [Test]
    public void Compile_RealNivaraSnippet_ReportsNoErrors()
    {
        var errors = Compile(Block(
            """
            NivaraColumn<int> c = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
            var sum = c.Sum();
            var masked = ColumnExpressions.Col("sum") > 2;
            """));

        Assert.That(errors.IsEmpty, Is.True, errors.Describe());
    }

    [Test]
    public void Compile_UnknownName_ReportsErrorNamingTheName()
    {
        var errors = Compile(Block("var x = NoSuchType.Foo();"));

        Assert.That(errors.Ids, Does.Contain("CS0103"));
    }

    [Test]
    public void Compile_UnknownMemberOnRealType_ReportsError()
    {
        var errors = Compile(Block(
            """
            NivaraColumn<int> c = NivaraColumn<int>.Create(new[] { 1, 2, 3 });
            var total = c.NoSuchMethod();
            """));

        Assert.That(errors.Ids, Does.Contain("CS1061"));
    }

    [Test]
    public void Compile_ErrorOnSecondBodyLine_MapsBackToDocumentLine()
    {
        var errors = Compile(Block(
            """
            var a = 1;
            var b = NoSuch();
            """,
            startLine: 40));

        Assert.That(errors.IsEmpty, Is.False);
        Assert.That(errors.Locations, Does.Contain("docs/INLINE.md:42"));
    }

    [Test]
    public void Compile_TopLevelWithAwait_CompilesAsExecutable()
    {
        var errors = Compile(Block(
            """
            await using var scan = Csv.ScanAsQueryFrame("events.csv");
            await foreach (var chunk in scan.AsStream()) { chunk.Dispose(); }
            """,
            locals: "CancellationToken ct = default;"));

        Assert.That(errors.IsEmpty, Is.True, errors.Describe());
    }

    [Test]
    public void Compile_GenericLocalWithTypeParameter_CompilesInsideWrapper()
    {
        var errors = Compile(Block(
            """
            var data = new T[len];
            var nullMask = new bool[len];
            """,
            mode: DocWrapMode.GenericLocal,
            locals: "int len = 4;"));

        Assert.That(errors.IsEmpty, Is.True, errors.Describe());
    }

    [Test]
    public void Compile_TypeParameterAtTopLevel_IsRejected()
    {
        var errors = Compile(Block("var data = new T[4];", locals: "int len = 4;"));

        Assert.That(errors.IsEmpty, Is.False);
        Assert.That(errors.Ids, Does.Contain("CS0246"));
    }

    [Test]
    public void Compile_FileModeWithOnlyTypes_CompilesAsLibrary()
    {
        var errors = Compile(Block(
            """
            public sealed class Person
            {
                public string Name { get; set; } = string.Empty;
            }
            """,
            mode: DocWrapMode.File));

        Assert.That(errors.IsEmpty, Is.True, errors.Describe());
    }

    [Test]
    public void Extract_BrokenFixture_ProducesExactlyOneExcludedFreeBlock()
    {
        var blocks = DocSnippetExtractor.Extract(FixturePath);

        Assert.That(blocks, Has.Count.EqualTo(1));
        Assert.That(blocks[0].Code, Does.Contain("NoSuchMethod"));
        Assert.That(blocks[0].Locals, Does.Contain("NivaraColumn<int> column"));
    }

    [Test]
    public void Compile_BrokenFixture_ReportsTheFixtureLine()
    {
        var block = DocSnippetExtractor.Extract(FixturePath).Single();
        var errors = Compile(block);

        Assert.That(errors.IsEmpty, Is.False, "the negative control must fail or the gate proves nothing");
        Assert.That(errors.Ids, Does.Contain("CS1061"));
        Assert.That(errors.Locations, Has.Some.EqualTo($"{FixturePath}:13"));
    }
}

/// <summary>Small aggregation so a failing assertion reports the compiler's own verdict.</summary>
sealed class DocSnippetErrors(IReadOnlyList<DocSnippetError> errors)
{
    public bool IsEmpty => errors.Count == 0;

    public string[] Ids => [.. errors.Select(e => e.Id)];

    public string[] Locations => [.. errors.Select(e => e.Location)];

    public string Describe() =>
        errors.Count == 0 ? "no errors" : Environment.NewLine + string.Join(Environment.NewLine, errors);
}