using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

namespace Nivara.Tests.Docs;

/// <param name="DocumentPath">Repo-relative path of the document the block came from.</param>
/// <param name="Line">Zero-based line in that document.</param>
/// <param name="Id">Roslyn diagnostic id, e.g. <c>CS0103</c>.</param>
/// <param name="Message">Diagnostic message.</param>
sealed record DocSnippetError(string DocumentPath, int Line, string Id, string Message)
{
    public string Location => $"{DocumentPath}:{Line + 1}";

    public override string ToString() => $"{Location}: {Id} {Message}";
}

static class DocSnippetCompiler
{
    /// <summary>
    /// Base usings prepended to every generated unit. Kept broad on purpose: a snippet should fail
    /// for naming a member that does not exist, not for omitting a using the tutorial assumes.
    /// </summary>
    static readonly string[] BaseUsings =
    [
        "using System;",
        "using System.Collections.Generic;",
        "using System.Linq;",
        "using System.Threading;",
        "using System.Threading.Tasks;",
        "using Nivara;",
        "using Nivara.AutoDiff;",
        "using Nivara.AutoDiff.Nn;",
        "using Nivara.AutoDiff.Nn.Functional;",
        "using Nivara.AutoDiff.Optimizer;",
        "using Nivara.AutoDiff.Utilities;",
        "using Nivara.Expressions;",
        "using Nivara.IO;",
        "using Nivara.Linq;",
        "using Nivara.Query;",
        "using Nivara.Streamix;",
        "using Nivara.Tests.Docs.Preambles;",
        "using static Nivara.Tests.Docs.Preambles.SnippetReport;",
        "using Streamix;",
    ];

    /// <summary>
    /// Deliberately not <c>Nivara.Tests</c>. Documentation must teach the public surface, so the
    /// generated assembly gets no <c>InternalsVisibleTo</c> grant and an internal type in a snippet
    /// fails the gate instead of quietly validating.
    /// </summary>
    const string AssemblyName = "Nivara.DocSnippet";

    static readonly CSharpParseOptions ParseOptions = new(LanguageVersion.Latest);

    /// <summary>Compiles one block and returns its errors, each mapped back to a document line.</summary>
    internal static IReadOnlyList<DocSnippetError> Compile(DocBlock block)
    {
        var generated = new List<string>(BaseUsings);
        var body = block.Code.Split('\n');
        int bodyStart;

        if (block.Mode == DocWrapMode.GenericLocal)
        {
            generated.Add(string.Empty);
            generated.Add("static void __DocSnippet<T>() where T : unmanaged");
            generated.Add("{");
            generated.AddRange(Indented(block.Locals, 1));
            if (block.Locals.Length > 0)
                generated.Add(string.Empty);
            bodyStart = generated.Count;
            generated.AddRange(Indented(string.Join("\n", body), 1));
            generated.Add("}");
        }
        else
        {
            generated.Add(string.Empty);
            generated.AddRange(block.Locals.Split('\n'));
            if (block.Locals.Length > 0)
                generated.Add(string.Empty);
            bodyStart = generated.Count;
            generated.AddRange(body);
        }

        var tree = CSharpSyntaxTree.ParseText(
            string.Join("\n", generated), ParseOptions, path: block.DocumentPath);

        var compilation = CSharpCompilation.Create(
            AssemblyName,
            [tree],
            References(),
            new CSharpCompilationOptions(
                block.Mode == DocWrapMode.TopLevel
                    ? OutputKind.ConsoleApplication
                    : OutputKind.DynamicallyLinkedLibrary));

        return compilation
            .GetDiagnostics()
            .Where(d => d.Severity == DiagnosticSeverity.Error)
            .Select(d => Map(d, block, bodyStart))
            .ToArray();
    }

    /// <summary>
    /// Maps a diagnostic line in the generated unit back to the document. Diagnostics landing in the
    /// generated prologue report the block's own line, since that is the snippet the reader must fix.
    /// </summary>
    static DocSnippetError Map(Diagnostic diagnostic, DocBlock block, int bodyStart)
    {
        var generatedLine = diagnostic.Location.GetLineSpan().StartLinePosition.Line;
        var documentLine = generatedLine >= bodyStart
            ? block.StartLine + (generatedLine - bodyStart) + 1
            : block.StartLine + 1;

        return new DocSnippetError(block.DocumentPath, documentLine, diagnostic.Id, diagnostic.GetMessage());
    }

    /// <summary>
    /// Builds references from the running process's platform assembly list, which is the exact
    /// runtime closure (Nivara, Nivara.Extensions, Streamix, NUnit and the framework) on any OS.
    /// </summary>
    static IReadOnlyList<MetadataReference> References()
    {
        if (AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") is not string tpa || tpa.Length == 0)
            throw new InvalidOperationException(
                "TRUSTED_PLATFORM_ASSEMBLIES is not populated; the doc gate cannot resolve references");

        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var references = new List<MetadataReference>();

        foreach (var path in tpa.Split(Path.PathSeparator))
        {
            if (path.Length == 0 || !File.Exists(path))
                continue;

            if (!seen.Add(Path.GetFileNameWithoutExtension(path)))
                continue;

            references.Add(MetadataReference.CreateFromFile(path));
        }

        if (references.Count == 0)
            throw new InvalidOperationException("No metadata references resolved from TRUSTED_PLATFORM_ASSEMBLIES");

        return references;
    }

    static string[] Indented(string text, int levels)
    {
        if (text.Length == 0)
            return [];

        var pad = new string(' ', levels * 4);
        return text.Split('\n').Select(line => line.Length == 0 ? string.Empty : pad + line).ToArray();
    }
}