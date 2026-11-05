using System.Text;

namespace NivaraChat.Helpers;

/// <summary>
/// A section-scoped chunk carrying the four texts #914 embeds independently. <see cref="Content"/>
/// is the chunk as written; <see cref="Code"/> holds only the fence bodies inside it, so the two
/// views never duplicate one another. <see cref="Title"/> is the document title,
/// <see cref="Section"/> the heading breadcrumb, and <see cref="AclGroup"/> the parent folder name
/// (matching <see cref="RagChunker"/>) or <c>public</c> at the corpus root.
/// </summary>
public sealed record MultiViewChunk(
    string ChunkId,
    string Source,
    string SourcePath,
    string Title,
    string Section,
    string AclGroup,
    string Content,
    string? Code);

/// <summary>
/// Fence-aware markdown chunker for multi-view retrieval. Unlike <see cref="RagChunker"/>, a line
/// inside a fenced block is never read as a heading, and the document title is kept separate from
/// the section breadcrumb.
/// </summary>
public static class MultiViewChunker
{
    static readonly char[] s_breakCharacters = [' ', '\n', '\t'];

    public static List<MultiViewChunk> ChunkMarkdown(
        string content,
        string sourcePath,
        string repoRoot,
        int maxChunkSize = 500)
    {
        ArgumentNullException.ThrowIfNull(content);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxChunkSize);

        var chunks = new List<MultiViewChunk>();
        var lines = content.Replace("\r\n", "\n").Replace('\r', '\n').Split('\n');
        var source = Path.GetFileName(sourcePath);
        var relativePath = ToRelativePath(sourcePath, repoRoot);
        var aclGroup = ResolveAclGroup(relativePath);
        var title = FindDocumentTitle(lines) ?? Path.GetFileNameWithoutExtension(sourcePath);

        // The title occupies the level-1 slot until (or unless) the document supplies an H1, so a
        // file without one still reads as "Title > Section" rather than " > Section".
        var breadcrumb = new List<string> { title };
        var prose = new List<string>();
        var contentBuilder = new StringBuilder();
        var codeBuilder = new StringBuilder();
        var index = 0;

        string CurrentSection()
        {
            var parts = breadcrumb.Where(part => part.Length > 0).ToList();
            return parts.Count == 0 ? title : string.Join(" > ", parts);
        }

        void FlushChunk()
        {
            var text = contentBuilder.ToString().Trim();
            var code = codeBuilder.ToString().Trim();
            contentBuilder.Clear();
            codeBuilder.Clear();
            if (text.Length == 0)
                return;

            chunks.Add(new MultiViewChunk(
                $"{relativePath}#{index++}",
                source,
                relativePath,
                title,
                CurrentSection(),
                aclGroup,
                text,
                code.Length == 0 ? null : code));
        }

        void AppendBlock(string text, string? code, bool splittable)
        {
            if (text.Length == 0)
                return;

            if (splittable && text.Length > maxChunkSize)
            {
                FlushChunk();
                foreach (var piece in SplitProse(text, maxChunkSize))
                {
                    contentBuilder.Append(piece);
                    FlushChunk();
                }

                return;
            }

            if (contentBuilder.Length > 0 && contentBuilder.Length + text.Length + 2 > maxChunkSize)
                FlushChunk();

            if (contentBuilder.Length > 0)
                contentBuilder.Append("\n\n");
            contentBuilder.Append(text);

            if (code is { Length: > 0 })
            {
                if (codeBuilder.Length > 0)
                    codeBuilder.Append("\n\n");
                codeBuilder.Append(code);
            }
        }

        void FlushProse()
        {
            if (prose.Count == 0)
                return;
            var text = string.Join("\n", prose).Trim();
            prose.Clear();
            AppendBlock(text, null, splittable: true);
        }

        for (var i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            var trimmed = line.TrimStart();

            if (TryReadFence(trimmed, out var fenceChar, out var fenceLength))
            {
                FlushProse();
                var fenceLines = new List<string> { line };
                var bodyLines = new List<string>();
                for (i++; i < lines.Length; i++)
                {
                    var current = lines[i];
                    if (IsFenceClose(current.TrimStart(), fenceChar, fenceLength))
                    {
                        fenceLines.Add(current);
                        break;
                    }

                    fenceLines.Add(current);
                    bodyLines.Add(current);
                }

                var fenceText = string.Join("\n", fenceLines).TrimEnd();
                var codeText = string.Join("\n", bodyLines).Trim('\n', ' ', '\t');
                AppendBlock(fenceText, codeText.Length == 0 ? null : codeText, splittable: false);
                continue;
            }

            if (TryReadHeading(trimmed, out var level, out var headingText))
            {
                FlushProse();
                FlushChunk();
                if (level <= 3)
                {
                    while (breadcrumb.Count < level)
                        breadcrumb.Add(string.Empty);
                    breadcrumb[level - 1] = headingText;
                    if (breadcrumb.Count > level)
                        breadcrumb.RemoveRange(level, breadcrumb.Count - level);
                }

                continue;
            }

            if (string.IsNullOrWhiteSpace(line))
            {
                FlushProse();
                continue;
            }

            prose.Add(line);
        }

        FlushProse();
        FlushChunk();
        return chunks;
    }

    static string? FindDocumentTitle(string[] lines)
    {
        for (var i = 0; i < lines.Length; i++)
        {
            var trimmed = lines[i].TrimStart();
            if (TryReadFence(trimmed, out var fenceChar, out var fenceLength))
            {
                for (i++; i < lines.Length; i++)
                {
                    if (IsFenceClose(lines[i].TrimStart(), fenceChar, fenceLength))
                        break;
                }

                continue;
            }

            if (TryReadHeading(trimmed, out var level, out var text) && level == 1)
                return text;
        }

        return null;
    }

    static bool TryReadFence(string trimmed, out char fenceChar, out int fenceLength)
    {
        fenceChar = default;
        fenceLength = 0;
        if (trimmed.Length < 3)
            return false;

        var c = trimmed[0];
        if (c != '`' && c != '~')
            return false;

        var count = 0;
        while (count < trimmed.Length && trimmed[count] == c)
            count++;
        if (count < 3)
            return false;

        fenceChar = c;
        fenceLength = count;
        return true;
    }

    static bool IsFenceClose(string trimmed, char fenceChar, int fenceLength)
    {
        var count = 0;
        while (count < trimmed.Length && trimmed[count] == fenceChar)
            count++;
        if (count < fenceLength)
            return false;

        for (var i = count; i < trimmed.Length; i++)
        {
            if (!char.IsWhiteSpace(trimmed[i]))
                return false;
        }

        return true;
    }

    static bool TryReadHeading(string trimmed, out int level, out string text)
    {
        level = 0;
        text = string.Empty;
        if (trimmed.Length == 0 || trimmed[0] != '#')
            return false;

        var count = 0;
        while (count < trimmed.Length && trimmed[count] == '#')
            count++;
        if (count > 6 || count == trimmed.Length || !char.IsWhiteSpace(trimmed[count]))
            return false;

        var heading = trimmed[count..].Trim();
        if (heading.Length == 0)
            return false;

        level = count;
        text = heading;
        return true;
    }

    static string ToRelativePath(string sourcePath, string repoRoot)
    {
        if (string.IsNullOrWhiteSpace(repoRoot))
            return Path.GetFileName(sourcePath);

        try
        {
            var relative = Path.GetRelativePath(repoRoot, sourcePath).Replace('\\', '/');
            return relative.StartsWith("..", StringComparison.Ordinal) ? Path.GetFileName(sourcePath) : relative;
        }
        catch
        {
            return Path.GetFileName(sourcePath);
        }
    }

    static string ResolveAclGroup(string relativePath)
    {
        try
        {
            var parent = Path.GetFileName(Path.GetDirectoryName(relativePath));
            return string.IsNullOrEmpty(parent) ? "public" : parent.ToLowerInvariant();
        }
        catch
        {
            return "public";
        }
    }

    static IEnumerable<string> SplitProse(string text, int maxChunkSize)
    {
        var start = 0;
        while (text.Length - start > maxChunkSize)
        {
            var cut = text.Substring(start, maxChunkSize).LastIndexOfAny(s_breakCharacters);
            if (cut <= 0)
                cut = maxChunkSize;

            var piece = text.Substring(start, cut).Trim();
            if (piece.Length > 0)
                yield return piece;

            start += cut;
            while (start < text.Length && char.IsWhiteSpace(text[start]))
                start++;
        }

        if (start < text.Length)
        {
            var tail = text[start..].Trim();
            if (tail.Length > 0)
                yield return tail;
        }
    }
}
