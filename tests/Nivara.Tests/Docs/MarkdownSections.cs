namespace Nivara.Tests.Docs;

/// <param name="Heading">Heading text, without its leading <c>#</c> run.</param>
/// <param name="Level">Number of leading <c>#</c> characters, 1 to 6.</param>
/// <param name="StartLine">One-based line of the heading itself.</param>
/// <param name="EndLine">One-based line number of the section's last line, inclusive.</param>
/// <param name="Body">Lines between the heading and <paramref name="EndLine"/>.</param>
sealed record MarkdownSection(
    string Heading,
    int Level,
    int StartLine,
    int EndLine,
    IReadOnlyList<string> Body)
{
    /// <summary>Sections nested directly inside this one, in document order.</summary>
    public IReadOnlyList<MarkdownSection> Children { get; init; } = [];

    /// <summary>The body as one string, for substring checks.</summary>
    public string Text => string.Join(Environment.NewLine, Body);

    /// <summary>
    /// Lines that are list items, with the marker and trailing whitespace removed. A dash that
    /// is not a list marker is not one — the marker must be followed by a space — so an
    /// ordinary hyphenated phrase in prose stays prose.
    /// </summary>
    public IReadOnlyList<string> Bullets()
    {
        var bullets = new List<string>();

        foreach (var line in Body)
        {
            var trimmed = line.TrimStart();
            if (trimmed.Length > 1 && trimmed[0] == '-' && trimmed[1] == ' ')
                bullets.Add(trimmed[2..].Trim());
        }

        return bullets;
    }
}

/// <summary>
/// Reads the heading structure out of a markdown document, so a gate can talk about a section
/// by name instead of about a pattern.
/// </summary>
/// <remarks>
/// <para>
/// Every rule here is a plain string test a reader can check by eye, in the same style
/// <see cref="DocSnippetExtractor"/> already uses to walk fences. The point being that a gate's
/// own machinery should not be the part nobody understands.
/// </para>
/// <para>
/// A heading is one to six <c>#</c> followed by whitespace. The whitespace is required, so
/// <c>#1 in review</c> is text rather than a heading, and more than six <c>#</c> is not a heading
/// either. A line of only <c>#</c> is also treated as text: CommonMark accepts it as an empty
/// heading, but an unnamed section cannot be required by name, so treating it as a heading would
/// only produce a section no caller can address.
/// </para>
/// </remarks>
static class MarkdownSections
{
    /// <summary>Every heading in the file at this repo-relative path, flattened, in document order.</summary>
    internal static IReadOnlyList<MarkdownSection> Extract(string documentPath) =>
        ExtractLines(File.ReadAllLines(Absolute(documentPath)));

    /// <summary>
    /// The same, over lines the caller supplies. Separate from <see cref="Extract"/> so the
    /// negative controls can drive it with input that genuinely disagrees, instead of only ever
    /// running over a real document — a check that only ever sees valid input can pass because
    /// it silently found nothing.
    /// </summary>
    internal static IReadOnlyList<MarkdownSection> ExtractLines(string[] lines)
    {
        var headings = new List<(int Level, string Heading, int Line)>();
        for (var i = 0; i < lines.Length; i++)
            if (TryParseHeading(lines[i], out var level, out var heading))
                headings.Add((level, heading, i));

        var sections = new List<MarkdownSection>(headings.Count);

        // A section runs until the next heading at its own level or shallower, so a subsection
        // belongs to the enclosing section rather than ending it.
        for (var i = 0; i < headings.Count; i++)
        {
            var (level, heading, start) = headings[i];
            var end = lines.Length;

            for (var j = i + 1; j < headings.Count; j++)
            {
                if (headings[j].Level > level)
                    continue;

                end = headings[j].Line;
                break;
            }

            // `end` is the zero-based index of the next heading of this level, which is also the one-based
            // line number of this section's last line — so it is the inclusive end.
            sections.Add(new MarkdownSection(heading, level, start + 1, end, lines[(start + 1)..end]));
        }

        return [.. AttachChildren(sections)];
    }

    /// <summary>The single section with this heading, or a failure naming what is available.</summary>
    internal static MarkdownSection Require(string documentPath, string heading)
    {
        var sections = Extract(documentPath);
        var matches = sections.Where(s => string.Equals(s.Heading, heading, StringComparison.Ordinal)).ToArray();

        if (matches.Length == 1)
            return matches[0];

        var reason = matches.Length == 0
            ? "no section has that heading"
            : $"{matches.Length} sections have that heading, so the name does not identify one";

        var available = string.Join(" | ", sections.Select(s => s.Heading));
        throw new InvalidOperationException(
            $"{documentPath}: {reason}. Expected '{heading}'. Headings present: {available}");
    }

    /// <summary>
    /// The text inside the first pair of backticks on a line, or null when there is none or only
    /// one backtick.
    /// </summary>
    /// <remarks>
    /// How a code span is located without a pattern: a markdown code span is delimited by a run
    /// of backticks, so the first backtick and the next one bound the content.
    /// </remarks>
    internal static string? FirstCodeSpan(string line)
    {
        var open = line.IndexOf('`');
        if (open < 0)
            return null;

        var close = line.IndexOf('`', open + 1);
        return close < 0 ? null : line[(open + 1)..close];
    }

    /// <summary>
    /// The ordinal a record file is numbered by — the digits before the first dash, so
    /// <c>005-snippet-gate.md</c> is <c>005</c>. Null when the name carries no leading number.
    /// </summary>
    /// <remarks>
    /// Deliberately tolerant about what follows the number: the prose after the dash is the
    /// title, which is free to change when the decision is amended.
    /// </remarks>
    internal static string? OrdinalOf(string fileName)
    {
        var dash = fileName.IndexOf('-');
        if (dash < 1)
            return null;

        var digits = fileName[..dash];
        foreach (var c in digits)
            if (!char.IsAsciiDigit(c))
                return null;

        return digits;
    }

    static bool TryParseHeading(string line, out int level, out string heading)
    {
        level = 0;
        heading = string.Empty;

        var trimmed = line.TrimStart();
        while (level < trimmed.Length && trimmed[level] == '#')
            level++;

        // More than six is not a heading, and a '#' with no space after it is text.
        if (level == 0 || level > 6 || level >= trimmed.Length || !char.IsWhiteSpace(trimmed[level]))
            return false;

        heading = trimmed[level..].Trim();
        return heading.Length > 0;
    }

    /// <summary>
    /// Attaches each section to the nearest enclosing one, so a caller holding a parent can see
    /// its subsections without walking the flattened list.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Headings arrive in document order, so the enclosing section is the last earlier heading of
    /// a shallower level that spans it. Indexes rather than <see cref="MarkdownSection"/>
    /// instances as keys: a record compares by value, so two identical sections would collide.
    /// </para>
    /// <para>
    /// The span check is redundant with the ordering and the level check — it exists so a
    /// malformed or hand-written document cannot attach a section to an unrelated earlier
    /// heading that happens to be shallower. Both halves of the range are tested because
    /// <see cref="MarkdownSection.EndLine"/> is inclusive, so an earlier sibling's end line is
    /// large enough to contain a later heading's start line by accident.
    /// </para>
    /// </remarks>
    static IReadOnlyList<MarkdownSection> AttachChildren(List<MarkdownSection> sections)
    {
        var children = new List<MarkdownSection>[sections.Count];

        for (var i = 0; i < sections.Count; i++)
        {
            for (var parent = i - 1; parent >= 0; parent--)
            {
                var candidate = sections[parent];

                var encloses = candidate.Level < sections[i].Level
                    && candidate.StartLine < sections[i].StartLine
                    && sections[i].StartLine <= candidate.EndLine;

                if (!encloses)
                    continue;

                (children[parent] ??= []).Add(sections[i]);
                break;
            }
        }

        return
        [
            .. sections.Select((s, i) => s with { Children = children[i] ?? [] }),
        ];
    }

    static string Absolute(string documentPath) =>
        Path.Combine(DocSnippetExtractor.RepoRoot, documentPath.Replace('/', Path.DirectorySeparatorChar));
}