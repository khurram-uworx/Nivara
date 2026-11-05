using Microsoft.Extensions.VectorData;

namespace NivaraChat.Helpers;

/// <summary>
/// Metadata-rich chunk used by the Nivara-convention RAG path (#912).
/// </summary>
public sealed record RagChunk(
    string ChunkId,
    string Source,
    string Title,
    string SectionPath,
    string AclGroup,
    string Content);

public static class RagChunker
{
    public static List<RagChunk> ChunkMarkdown(string content, string sourcePath, string repoRoot, int maxChunkSize = 500)
    {
        var chunks = new List<RagChunk>();
        // First, walk lines to pair each paragraph with the most recent heading.
        var lines = content.Split('\n');
        string title = Path.GetFileNameWithoutExtension(sourcePath);
        string section = title;
        var paraLines = new List<string>();
        int idx = 0;

        void FlushParagraph()
        {
            var text = string.Join("\n", paraLines).Trim();
            paraLines.Clear();
            if (string.IsNullOrWhiteSpace(text)) return;
            if (text.Length > 1200)
            {
                int pos = 0;
                while (pos < text.Length)
                {
                    int len = Math.Min(maxChunkSize, text.Length - pos);
                    var piece = text.Substring(pos, len).Trim();
                    if (!string.IsNullOrEmpty(piece))
                        chunks.Add(MakeChunk(sourcePath, repoRoot, title, section, piece, idx++));
                    pos += len;
                }
            }
            else
            {
                chunks.Add(MakeChunk(sourcePath, repoRoot, title, section, text, idx++));
            }
        }

        foreach (var raw in lines)
        {
            var line = raw.TrimEnd('\r');
            if (line.TrimStart().StartsWith("#"))
            {
                FlushParagraph();
                var h = line.TrimStart('#').Trim();
                if (!string.IsNullOrEmpty(h)) { title = h; section = h; }
                continue;
            }
            if (string.IsNullOrWhiteSpace(line))
            {
                FlushParagraph();
            }
            else
            {
                paraLines.Add(line);
            }
        }
        FlushParagraph();

        if (chunks.Count == 0)
            chunks.Add(MakeChunk(sourcePath, repoRoot, title, section, content.Trim(), 0));

        return chunks;
    }

    static RagChunk MakeChunk(string sourcePath, string repoRoot, string title, string section, string content, int idx)
    {
        var source = Path.GetFileName(sourcePath);
        string aclGroup;
        try
        {
            var parent = Path.GetFileName(Path.GetDirectoryName(sourcePath));
            aclGroup = string.IsNullOrEmpty(parent) ? "public" : parent.ToLowerInvariant();
        }
        catch
        {
            aclGroup = "public";
        }
        return new RagChunk($"{source}-{idx}", source, title, section, aclGroup, content);
    }
}

public sealed class DocumentChunk
{
    [VectorStoreKey]
    public string Key { get; set; } = string.Empty;

    [VectorStoreData]
    public string Text { get; set; } = string.Empty;

    [VectorStoreData]
    public string Source { get; set; } = string.Empty;

    [VectorStoreVector(dimensions: 384)]
    public string Embedding => Text;
}

public static class DocumentChunker
{
    public static List<(string Text, int Index)> ChunkText(string text, int maxChunkSize = 500)
    {
        var paragraphs = text.Split(["\n\n"], StringSplitOptions.RemoveEmptyEntries);
        var chunks = new List<(string Text, int Index)>();
        var currentChunk = new System.Text.StringBuilder();
        int index = 0;

        foreach (var paragraph in paragraphs)
        {
            var trimmed = paragraph.Trim();
            if (string.IsNullOrEmpty(trimmed)) continue;

            if (currentChunk.Length + trimmed.Length + 2 > maxChunkSize && currentChunk.Length > 0)
            {
                chunks.Add((currentChunk.ToString().Trim(), index++));
                currentChunk.Clear();
            }

            if (currentChunk.Length > 0)
                currentChunk.Append("\n\n");
            currentChunk.Append(trimmed);
        }

        if (currentChunk.Length > 0)
            chunks.Add((currentChunk.ToString().Trim(), index));

        return chunks;
    }

    public static async Task IndexMarkdownFiles(
        VectorStoreCollection<string, DocumentChunk> collection,
        string[] filePaths)
    {
        foreach (var filePath in filePaths)
        {
            if (!File.Exists(filePath)) continue;

            var content = await File.ReadAllTextAsync(filePath);
            var source = Path.GetFileName(filePath);
            var chunks = ChunkText(content);

            foreach (var (text, idx) in chunks)
            {
                var chunk = new DocumentChunk
                {
                    Key = $"{source}-{idx}",
                    Text = text,
                    Source = source
                };
                await collection.UpsertAsync(chunk);
            }
        }
    }
}
