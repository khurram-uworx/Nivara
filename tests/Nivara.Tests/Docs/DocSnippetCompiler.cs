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
    ///
    /// This list must cover every namespace the public surface actually occupies. A narrower list
    /// does not merely add friction, it produces false defect reports: a first cut omitted
    /// <c>Nivara.Tensors</c> and <c>Nivara.Operations</c>, so <c>frame.ToTensor()</c> and
    /// <c>OrderBy(..., NullOrdering.NullsFirst)</c> were both reported as missing members when they
    /// exist and compile. Widening the list is what makes the diagnostics trustworthy; it cannot
    /// mask a defect, because it only affects name resolution, never whether a member is present.
    /// </summary>
    static readonly string[] BaseUsings =
    [
        "using System;",
        "using System.Collections.Generic;",
        "using System.IO;",
        "using System.Linq;",
        "using System.Numerics.Tensors;",
        "using System.Threading;",
        "using System.Threading.Tasks;",
        "using Nivara;",
        "using Nivara.AutoDiff;",
        "using Nivara.AutoDiff.Exceptions;",
        "using Nivara.AutoDiff.Extensions;",
        "using Nivara.AutoDiff.Nn;",
        "using Nivara.AutoDiff.Nn.Functional;",
        "using Nivara.AutoDiff.Nn.Initializers;",
        "using Nivara.AutoDiff.Operations;",
        "using Nivara.AutoDiff.Optimizer;",
        "using Nivara.AutoDiff.Serialization;",
        "using Nivara.AutoDiff.Training;",
        "using Nivara.AutoDiff.Utilities;",
        "using Nivara.Diagnostics;",
        "using Nivara.Exceptions;",
        "using Nivara.Execution;",
        "using Nivara.Expressions;",
        "using Nivara.Helpers;",
        "using Nivara.IO;",
        "using Nivara.Linq;",
        "using Nivara.Operations;",
        "using Nivara.Optimization;",
        "using Nivara.Primitives;",
        "using Nivara.Query;",
        "using Nivara.Storage;",
        "using Nivara.Streamix;",
        "using Nivara.Tensors;",
        "using Nivara.Tests.Docs.Preambles;",
        "using NUnit.Framework;",
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
        var hoisted = HoistLeadingUsings(string.Join("\n", block.Code.Split('\n')));
        var body = hoisted.Region;

        var generated = new List<string>(BaseUsings);
        if (hoisted.Directives.Length > 0)
        {
            generated.AddRange(hoisted.Directives);
            generated.Add(string.Empty);
        }

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
        else if (block.Mode == DocWrapMode.MemberDecl)
        {
            generated.Add(string.Empty);
            generated.Add("public class __DocSnippet");
            generated.Add("{");
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
                // Only TopLevel statements and GenericLocal's wrapper local function are top-level
                // statements, and CS8805 requires those to be in an executable. File and MemberDecl
                // wrap in a type declaration and are genuine libraries.
                block.Mode is DocWrapMode.TopLevel or DocWrapMode.GenericLocal
                    ? OutputKind.ConsoleApplication
                    : OutputKind.DynamicallyLinkedLibrary));

        return compilation
            .GetDiagnostics()
            .Where(d => d.Severity == DiagnosticSeverity.Error)
            .Select(d => Map(d, block, bodyStart))
            .ToArray();
    }

    /// <summary>
    /// Lifts a block's own using directives into the generated prologue. A using directive is only
    /// legal ahead of every other element in a compilation unit (CS1529), so a snippet that spells
    /// out <c>using Nivara.Linq;</c> cannot also be spliced in as top-level statements.
    ///
    /// Directive lines are blanked rather than removed, which keeps the body region's line count
    /// identical to the document's and leaves <see cref="Map"/>'s arithmetic untouched. Removal would
    /// have required threading a hoisted-line offset through it, and a silent off-by-N there would
    /// misdirect every future reader to the wrong line.
    ///
    /// Only genuine directives move. <c>using var csv = …</c> and <c>using (GradientUtils.Grad())</c>
    /// are statements that are legal where they stand, and the parser already separates the two forms
    /// — a line-based <c>^\s*using\s</c> hoist would have lifted them into the prologue and broken
    /// every block that disposes a resource.
    /// </summary>
    static (string[] Directives, string[] Region) HoistLeadingUsings(string bodyText)
    {
        var root = CSharpSyntaxTree.ParseText(bodyText, ParseOptions).GetCompilationUnitRoot();

        // Usings are in source order, so the first one at or after the first member is misplaced and
        // is left alone to surface as CS1529 against the document rather than being silently fixed.
        var firstMemberStart = root.Members.Count > 0 ? root.Members[0].SpanStart : int.MaxValue;

        var region = bodyText.Split('\n');
        var blanked = new bool[region.Length];
        var directives = new List<string>();
        var seen = new HashSet<string>(BaseUsings.Select(u => u.Trim()), StringComparer.Ordinal);

        foreach (var directive in root.Usings)
        {
            if (directive.SpanStart >= firstMemberStart)
                break;

            var text = directive.ToString().Trim();
            if (text.Length > 0 && seen.Add(text))
                directives.Add(text);

            blanked[directive.GetLocation().GetLineSpan().StartLinePosition.Line] = true;
        }

        // A directive already present in BaseUsings is blanked but not re-emitted; a repeat would be
        // CS0105, a warning, so deduplication is hygiene here rather than a correctness requirement.
        for (var i = 0; i < region.Length; i++)
            if (blanked[i])
                region[i] = string.Empty;

        return ([.. directives], region);
    }

    /// <summary>
    /// Maps a diagnostic line in the generated unit back to the document. Diagnostics landing in the
    /// generated prologue report the block's own line, since that is the snippet the reader must fix.
    /// Hoisting preserves the body's line count, so the body term needs no offset.
    /// </summary>
    static DocSnippetError Map(Diagnostic diagnostic, DocBlock block, int bodyStart)
    {
        var generatedLine = diagnostic.Location.GetLineSpan().StartLinePosition.Line;
        var documentLine = generatedLine >= bodyStart
            ? block.StartLine + (generatedLine - bodyStart)
            : block.StartLine;

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