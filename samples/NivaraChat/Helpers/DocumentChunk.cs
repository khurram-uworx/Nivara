using Microsoft.Extensions.VectorData;

namespace NivaraChat.Helpers;

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
